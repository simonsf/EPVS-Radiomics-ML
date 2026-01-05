rm(list = ls())  # Clear workspace

# Load required packages & install automatically if missing
if (!require("ppcor")) install.packages("ppcor", dependencies = TRUE)
if (!require("tidyverse")) install.packages("tidyverse", dependencies = TRUE)
library(ppcor)    
library(tidyverse)

data1 <- read_csv("C:/Windows/Temp/AA/Partial correlation analysis.csv", show_col_types = FALSE)

# Data basic info & variable validation
cat("=== Basic Data Info ===\n")
print(dim(data1))
print(names(data1))
cat("\n=== Critical Variable Type Check ===\n")
print(str(data1[, c("Number_of_EPVS", "Total_volume_of_EPVS", "Intracranial_volume", "Sex", "Age")]))

# Convert Sex to numeric if it is character type
if (is.character(data1$Sex)) {
  data1$Sex <- as.integer(factor(data1$Sex, levels = unique(data1$Sex)))
  cat("\n=== Sex converted to numeric: ", unique(data1$Sex), "\n")
}

# Partial correlation 1: Number_of_EPVS ~ Intracranial_volume (Control: Sex, Age)
cat("\n=== Analysis 1: Number_of_EPVS vs Intracranial_volume (Spearman) ===\n")
pcor_result1 <- pcor.test(
  x = data1$Number_of_EPVS,
  y = data1$Intracranial_volume,
  z = data1[, c("Sex", "Age")],
  method = "spearman"
)
print(pcor_result1)
cat("Partial r: ", round(pcor_result1$estimate, 4), "\n")
cat("P-value: ", round(pcor_result1$p.value, 4), "\n")
cat("Significance: ", ifelse(pcor_result1$p.value < 0.05, "Significant (p<0.05)", "Not significant"), "\n")

# Partial correlation 2: Total_volume_of_EPVS ~ Intracranial_volume (Control: Sex, Age)
cat("\n=== Analysis 2: Total_volume_of_EPVS vs Intracranial_volume (Spearman) ===\n")
pcor_result2 <- pcor.test(
  x = data1$Total_volume_of_EPVS,
  y = data1$Intracranial_volume,
  z = data1[, c("Sex", "Age")],
  method = "spearman"
)
print(pcor_result2)
cat("Partial r: ", round(pcor_result2$estimate, 4), "\n")
cat("P-value: ", round(pcor_result2$p.value, 4), "\n")
cat("Significance: ", ifelse(pcor_result2$p.value < 0.05, "Significant (p<0.05)", "Not significant"), "\n")

# Missing value check for critical variables
cat("\n=== Missing Value Check (Critical Variables) ===\n")
missing_info <- data1[, c("Number_of_EPVS", "Total_volume_of_EPVS", "Intracranial_volume", "Sex", "Age")] %>%
  summarise_all(~sum(is.na(.))) %>%
  pivot_longer(everything(), names_to = "Variable", values_to = "Missing_Count")
print(missing_info)
if (any(missing_info$Missing_Count > 0)) {
  cat("Warning: Missing values detected, please handle first\n")
} else {
  cat("✅ No missing values, results are reliable\n")
}
