args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2) {
  stop("Usage: Rscript check_cox_ph_assumptions.R <input.csv> <output.csv>")
}

suppressPackageStartupMessages(library(survival))

input_path <- args[[1]]
output_path <- args[[2]]
dat <- read.csv(input_path, stringsAsFactors = FALSE, check.names = FALSE)

landmark_order <- c("Diagnosis", "3-year survivor", "5-year survivor")
outcome_order <- c("Urethral cancer death", "Other-cause death")
results <- list()
index <- 1L

for (landmark_name in landmark_order) {
  for (outcome_name in outcome_order) {
    subset_data <- dat[
      dat$landmark == landmark_name & dat$outcome == outcome_name,
      ,
      drop = FALSE
    ]
    subset_data$stage <- factor(
      subset_data$stage,
      levels = c("Localized", "Regional", "Distant", "Unknown")
    )
    fit <- coxph(
      Surv(follow_up_time, status) ~ age_per_10_years + male + stage,
      data = subset_data,
      ties = "breslow",
      x = TRUE,
      y = TRUE
    )
    test <- cox.zph(fit, transform = "km", terms = TRUE, singledf = FALSE, global = TRUE)
    test_table <- as.data.frame(test$table)
    test_table$term <- rownames(test$table)
    rownames(test_table) <- NULL
    names(test_table)[1:3] <- c("chisq", "df", "p_value")
    test_table$landmark <- landmark_name
    test_table$outcome <- outcome_name
    test_table$n <- nrow(subset_data)
    test_table$events <- sum(subset_data$status)
    results[[index]] <- test_table[
      , c("landmark", "outcome", "n", "events", "term", "chisq", "df", "p_value")
    ]
    index <- index + 1L
  }
}

output <- do.call(rbind, results)
write.csv(output, output_path, row.names = FALSE, fileEncoding = "UTF-8")
