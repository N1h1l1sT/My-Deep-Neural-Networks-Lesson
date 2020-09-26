#Ο Κώδικας σε αυτό το script είναι μέρος του μαθήματος "Νευρωνικά Δίκτυα 13: Κώδικας Ridge Παλινδρόμησης"
#https://www.youtube.com/watch?v=XJRId4tImIg
library(dplyr)
library(MASS)
library(ggplot2)

###########
#Functions#
###########
R2 <- function(YTest, Yhat) {
  d1 <- YTest - Yhat
  d2 <- YTest - mean(YTest)
  r2 <- 1 - (d1 %*% d1) / (d2 %*% d2)
  print(paste0("r-squared: ", round(r2, 3)))
}

################################
#Multiple Independent Variables#
################################
r13 <- 0.55
r23 <- 0.85

DataDF <- mvrnorm(
  n = 100,
  mu = c(0, 0, 50),
  Sigma = matrix(c(1, 0.1, r13, 0.1, 1, r23, r13, r23, 1), nrow = 3)
) %>%
  as_tibble()

X <- DataDF %>% mutate(x0 = 1) %>% dplyr::select(4, 1, 2) %>% as.matrix()
Y <- DataDF %>% pull(3)
XTrain <- X[1:(NROW(X)*0.8),]
YTrain <- Y[1:(NROW(X)*0.8)]
XTest <- X[((NROW(X)*0.8)+1):NROW(X),]
YTest <- Y[((NROW(X)*0.8)+1):NROW(X)]

#########################################################
#L2 Linear Regression (Closed Form) Maximum a posteriori#
#########################################################
L2 <- 0.1
LI <- diag(L2, nrow = NCOL(XTrain), ncol = NCOL(XTrain)) #λ regularisation times the Identity Matrix (I)

W <- base::solve(LI + t(XTrain) %*% XTrain, (t(XTrain) %*% YTrain))
Yhat <- (XTest %*% W)[,1]
R2(YTest, Yhat)

Yhat[1]
YTest[1]