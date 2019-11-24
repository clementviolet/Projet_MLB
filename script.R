library(readr)
library(forcats)
library(missMDA)

set.seed(42)

colnames <- scan("colnames.txt", what = character(), sep="\t")
data <- read_csv("CommViolPredUnnormalizedData.txt", na = c("", "NA", "?"), col_names = colnames)

str(data)
summary(data)

meta_data <- data[,c(1:5)]

data_raw <- data[, -c(1:5)]

# Recode cette variable numérique en facteur
# data_raw$LemasGangUnitDeploy <- data_raw$LemasGangUnitDeploy %>%
#   as.factor() %>%
#   fct_recode("No" = "0", "Part time" = "5", "Yes" = "10")


# estim_ncpFAMD(data_raw, ncp.max = 30, verbose = TRUE) # Time consuming, n = 5 tested for 5, 15 and 30 principal components


# data <- data_raw %>%
#   as.data.frame() %>%
#   {imputeFAMD(., ncp = 5)$completeObs} %>%
#   as.data.frame()

data <- data_raw %>%
  as.data.frame() %>%
  {imputePCA(., ncp = 5)$completeObs} %>%
  as.data.frame()


write.csv(data, file = "./complet_data.csv")