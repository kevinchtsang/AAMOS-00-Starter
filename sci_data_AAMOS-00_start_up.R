#' Scientific Data
#' AAMOS-00 Study - Predicting Asthma Attacks Using Connected Mobile Devices and Machine Learning
#' 
#' Structure of file
#' - Import data
#' - Data processing
#' - Binary classifier example
#' - Validation


library(tidyverse)
library(lubridate)
library(caret)
library(PRROC)
library(ggpubr)


# Import data ----
daily_df        <- read.csv("anonym_aamos00_dailyquestionnaire.csv")
endquest_df     <- read.csv("anonym_aamos00_endquestionnaire.csv")
environment_df  <- read.csv("anonym_aamos00_environment.csv")
patient_info_df <- read.csv("anonym_aamos00_patient_info.csv")
peakflow_df     <- read.csv("anonym_aamos00_peakflow.csv")
inhaler_df      <- read.csv("anonym_aamos00_smartinhaler.csv")
watch_df1       <- read.csv("anonym_aamos00_smartwatch1.csv")
watch_df2       <- read.csv("anonym_aamos00_smartwatch2.csv")
watch_df3       <- read.csv("anonym_aamos00_smartwatch3.csv")
weekly_df       <- read.csv("anonym_aamos00_weeklyquestionnaire.csv")

watch_activity_lookup <- readxl::read_excel("aamos00_data_dictionary.xlsx", sheet = "miband_activity_lookup")


# Smart peak flow meter data processing ----
# Calculate percentage of maximum PEF
peakflow_day_df <- peakflow_df %>%
  group_by(user_key, date) %>%
  summarise(
    pef = max(pef_max, na.rm = T)
  ) %>%
  mutate(
    pef_norm = pef/max(pef, na.rm = T)
  ) %>%
  ungroup()


# Smart inhaler data processing ----
# Total daily relief inhaler usage
inhaler_day_df <- inhaler_df %>%
  group_by(user_key, date) %>%
  summarise(
    relief_use = n()
  ) %>%
  ungroup()


# Smartwatch data processing ----
# Use activity type lookup
# Convert minute-by-minute data into daily data
# Extract daily steps, mean heart rate, minutes slept
watch_df <- rbind(watch_df1, watch_df2, watch_df3)
watch_day_df <- watch_df %>%
  left_join(
    watch_activity_lookup,
    by = c("activity_type" = "number")
  ) %>%
  group_by(user_key, date) %>%
  summarise(
    steps         = sum(steps, na.rm = T),
    mean_hr       = mean(hr, na.rm = T),
    sleep_minutes = sum(activity == "sleep", na.rm = T)
  ) %>%
  ungroup() %>%
  filter(!is.na(mean_hr) & mean_hr > 1)


# Weekly questionnaire data processing ----
# Convert answers to dates
adverse_events <- weekly_df %>%
  select(user_key, date, weekly_doc, weekly_hospital, weekly_er, weekly_oral) %>%
  filter(
    weekly_doc        !="0" | 
      weekly_hospital !="0" | 
      weekly_er       !="0" |
      weekly_oral     =="3"
  ) %>%
  # one row per answer
  mutate(
    weekly_doc      = str_split(weekly_doc,","),
    weekly_hospital = str_split(weekly_hospital,","),
    weekly_er       = str_split(weekly_er,",")
  ) %>%
  unnest(weekly_doc) %>%
  unnest(weekly_hospital) %>%
  unnest(weekly_er) %>%
  ungroup() %>%
  # column types
  mutate_at(vars("weekly_doc", "weekly_hospital", "weekly_er"), as.integer) %>%
  # update date using value
  pivot_longer(
    cols      = starts_with("weekly_"),
    names_to  = "event_type",
    values_to = "days"
  ) %>%
  mutate(date = ifelse(event_type != "weekly_oral",
                       date + days,
                       date)) %>%
  # clean up
  filter(days != 0) %>%
  select(-days) %>%
  arrange(user_key, date) %>%
  distinct()

adverse_events_day_df <- adverse_events %>%
  mutate(event = TRUE) %>%
  pivot_wider(id_cols     = c("user_key", "date"),
              names_from  = "event_type",
              values_from = "event") %>%
  replace(is.na(.), FALSE)


# Join all data ----
all_data_day_df <- plyr::join_all(
  list(daily_df, environment_df, peakflow_day_df, 
       inhaler_day_df, watch_day_df, adverse_events_day_df),
  by   = c("user_key", "date"),
  type = "full"
)

# if no daily smart inhaler recording, set 0 daily usage
# if no adverse events, set false
all_data_day_df <- all_data_day_df %>%
  mutate(
    relief_use      = ifelse(is.na(relief_use), 0, relief_use),
    weekly_doc      = ifelse(is.na(weekly_doc), FALSE, weekly_doc),
    weekly_oral     = ifelse(is.na(weekly_oral), FALSE, weekly_oral),
    weekly_er       = ifelse(is.na(weekly_er), FALSE, weekly_er),
    weekly_hospital = ifelse(is.na(weekly_hospital), FALSE, weekly_hospital)
  )


# Total count of data ----
nrow(daily_df        %>% distinct(user_key, date)) # 1583 daily questionnaires
nrow(peakflow_day_df %>% distinct(user_key, date)) # 1099 patient-days of smart peak flow meter
nrow(inhaler_day_df  %>% distinct(user_key, date)) # 694  patient-days of smart inhaler
nrow(watch_df        %>% distinct(user_key, date)) # 1567 patient-days of smartwatch
nrow(environment_df  %>% distinct(user_key, date)) # 1657 patient-days of sent locations
nrow(weekly_df       %>% distinct(user_key, date)) # 324  weekly questionnaires

nrow(all_data_day_df %>% distinct(user_key, date)) # 2054 patient-days of data in AAMOS-00 phase 2


# Binary classifier ----
# Train linear (glm) model and random forest (rf) model
# class variable = weekly_doc
# section data into calendar weeks
ml_df <- all_data_day_df %>%
  arrange(user_key, date) %>%
  mutate(
    week_num = floor(date / 7)
  ) %>%
  select(-c(date, daily_triggers)) %>%
  group_by(user_key, week_num) %>%
  summarise(
    daily_night_symp     = mean(daily_night_symp, na.rm = T),
    daily_day_symp       = mean(daily_day_symp, na.rm = T),
    daily_limit_activity = mean(daily_limit_activity, na.rm = T),
    daily_prev_inhaler   = mean(daily_prev_inhaler, na.rm = T),
    daily_relief_inhaler = mean(daily_relief_inhaler, na.rm = T),
    temperature          = mean(temperature, na.rm = T),
    temperature_min      = min(temperature_min, na.rm = T),
    temperature_max      = max(temperature_max, na.rm = T),
    aqi                  = max(aqi, na.rm = T),
    pef_norm             = mean(pef_norm, na.rm = T),
    relief_use           = mean(relief_use, na.rm = T),
    steps                = mean(steps, na.rm = T),
    mean_hr              = mean(mean_hr, na.rm = T),
    sleep_minutes        = mean(sleep_minutes, na.rm = T),
    weekly_doc           = any(weekly_doc, na.rm = T),
    weekly_oral          = any(weekly_oral, na.rm = T),
    weekly_er            = any(weekly_er, na.rm = T),
    weekly_hospital      = any(weekly_hospital, na.rm = T)
  ) %>%
  ungroup() %>%
  select(-c(user_key, week_num)) %>%
  na.omit()

# train test split
set.seed(12345)
split_index <- sample(c(rep(1,8), rep(2,2)),
                      nrow(ml_df),
                      replace = TRUE)
ml_train_df <- ml_df[split_index == 1,]
ml_test_df  <- ml_df[split_index == 2,]

# standardise data
ml_train_df <- ml_train_df %>%
  mutate(across(where(is.numeric), scale))
ml_test_df <- ml_test_df %>%
  mutate(across(where(is.numeric), scale))

# train models using cross validation
train.control <- trainControl(method          = "cv", 
                              number          = 3,
                              classProbs      = TRUE,
                              summaryFunction = twoClassSummary)

# class distribution
table(ml_df$weekly_doc)
# FALSE  TRUE 
#   159    15 

# train glm
model_glm <- train(
  as.factor(make.names(weekly_doc)) ~.,
  data      = ml_train_df, 
  method    = "glm",
  trControl = train.control,
  metric    = "ROC"
  )

# test glm
test_glm_scores <- predict(model_glm, ml_test_df, "prob")[,2]
test_glm_roc    <- roc.curve(test_glm_scores[ml_test_df$weekly_doc], 
                             test_glm_scores[!ml_test_df$weekly_doc], 
                             curve = TRUE)
test_glm_pr     <- pr.curve(test_glm_scores[ml_test_df$weekly_doc], 
                            test_glm_scores[!ml_test_df$weekly_doc], 
                            curve = TRUE)

plot(test_glm_roc) # AUC = 0.88
plot(test_glm_pr)  # AUPRC = 0.23


# train rf
model_rf <- train(
  as.factor(make.names(weekly_doc)) ~.,
  data      = ml_train_df, 
  method    = "rf",
  trControl = train.control,
  metric    = "ROC")


# test rf
test_rf_scores <- predict(model_rf, ml_test_df, "prob")[,2]
test_rf_roc    <- roc.curve(test_rf_scores[ml_test_df$weekly_doc], 
                            test_rf_scores[!ml_test_df$weekly_doc], 
                            curve = TRUE)
test_rf_pr     <- pr.curve(test_rf_scores[ml_test_df$weekly_doc], 
                           test_rf_scores[!ml_test_df$weekly_doc],
                           curve = TRUE)

plot(test_rf_roc) # AUC = 0.93
plot(test_rf_pr)  # AUPRC = 0.55

# save plot
png(filename = "rf_roc.png", width = 2000, height = 1500, res = 300, units = "px")
plot(test_rf_roc)
dev.off()

# Daily answers to questions for asthma control ----
# Consider all combinations of daily questions about asthma control:
#  - daily_day_symp
#  - daily_night_symp
#  - daily_limit_activity
rcp3_plot_day_night <- ggplot(
  ml_df,
  aes(
    x = daily_day_symp,
    y = daily_night_symp
  )) +
  geom_point(alpha = 0.2) +
  geom_abline(
    slope     = 1,
    intercept = 0,
    linetype  = 2
  ) +
  labs(
    x = "Day symptoms (mean)",
    y = "Night symptoms (mean)"
  ) +
  theme_minimal()
rcp3_plot_day_activity <- ggplot(
  ml_df,
  aes(
    x = daily_day_symp,
    y = daily_limit_activity
  )) +
  geom_point(alpha = 0.2) +
  geom_abline(
    slope     = 1,
    intercept = 0,
    linetype  = 2
  ) +
  labs(
    x = "Day symptoms (mean)",
    y = "Activity limitation (mean)"
  ) +
  theme_minimal()
rcp3_plot_night_activity <- ggplot(
  ml_df,
  aes(
    x = daily_limit_activity,
    y = daily_night_symp
  )) +
  geom_point(alpha = 0.2) +
  geom_abline(
    slope     = 1,
    intercept = 0,
    linetype  = 2
  ) +
  labs(
    x = "Activity limitation (mean)",
    y = "Night symptoms (mean)"
  ) +
  theme_minimal()
rcp3_plot <- ggarrange(
  rcp3_plot_day_night, rcp3_plot_night_activity, rcp3_plot_day_activity,
  ncol = 2, nrow = 2)
rcp3_plot

ggsave("rcp3_plot.png",rcp3_plot, width = 2000, height = 2000, dpi=300, units = "px")
