#Ο Κώδικας σε αυτό το script είναι μέρος του μαθήματος "Νευρωνικά Δίκτυα 14: Lasso Παλινδρόμηση με Αδύνατη Closed Form Λύση"
#https://www.youtube.com/watch?v=L6cqXVghUFY
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

########################################################
#Linear Regression w/ Gradient Descent (L1 Regulariser)#
########################################################
NUM_Epochs <- 1000
LearningRate <- 1e-4
D <- NCOL(XTrain)
N <- NROW(XTrain)
L1 <- 0.1
Costs <- NULL

W <- rnorm(D, 0, sqrt(1/D)) #Randomly Initialising the Weights

for (i in 1:NUM_Epochs) {
  Yhat <- (XTrain %*% W)[,1]
  Residuals <- Yhat - YTrain
  W <- W - LearningRate * ((t(XTrain) %*% Residuals) + as.vector(L1*sign(W))) #Update W

  Costs <- c(Costs, ((t(Residuals) %*% Residuals) + sum(L1*W)) / N)
}

Yhat <- (XTest %*% W)[,1]
R2(YTest, Yhat)

#Visualising the Costs
ggplot() + geom_line(aes(x = 1:NROW(Costs), y = Costs))

Yhat[1]
YTest[1]
