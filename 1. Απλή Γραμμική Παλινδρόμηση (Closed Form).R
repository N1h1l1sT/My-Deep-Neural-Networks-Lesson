#Ο Κώδικας σε αυτό το script είναι μέρος του μαθήματος "Νευρωνικά Δίκτυα 10: Κώδικας Γραμμικής Παλινδρόμησης"
#https://www.youtube.com/watch?v=t7PNIfcTxeU
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

########################
#1 Independent Variable#
########################
r <- 0.90

DataDF <- mvrnorm(
  n = 100,
  mu = c(50, 50),
  Sigma = matrix(c(1, r, r, 1), nrow = 2),
  empirical = TRUE
) %>%
  as_tibble()

X <- DataDF %>% pull(1)
Y <- DataDF %>% pull(2)
XTrain <- X[1:(NROW(X)*0.8)]
YTrain <- Y[1:(NROW(X)*0.8)]
XTest <- X[((NROW(X)*0.8)+1):NROW(X)]
YTest <- Y[((NROW(X)*0.8)+1):NROW(X)]

#Visualising the Dataset
ggplot(DataDF, aes(x = V1, y = V2, color = V2)) +
  geom_point()

#############################################################
#Simple Univariate Linear Regression, Solving Yhat = WXi + b#
#############################################################

YXMean <- mean(YTrain * XTrain)
YMean <- mean(YTrain)
XMean <- mean(XTrain)
XSqMean <- mean(XTrain ^ 2)
XMeanSQ <- XMean ^ 2
N <- NROW(XTrain)

W <- (YXMean - YMean * XMean) / (XSqMean - XMeanSQ)
b <- (XSqMean * YMean - YXMean * XMean) / (XSqMean - XMeanSQ)
Yhat <- W * XTest + b
Yhat[1]
YTest[1]

R2(YTest, Yhat)

XSum <- sum(XTrain)
XSqSum <- sum(XTrain ^ 2)
YSum <- sum(YTrain)
XYSum <- sum(XTrain * YTrain)
XSumSq <- XSum ^ 2
W <- ((YSum * XSum) - (N * XYSum)) / (XSumSq - (N * XSqSum))
b <- ((XYSum * XSum) - (XSqSum * YSum)) / (XSumSq - (N * XSqSum))
Yhat <- W * XTest + b

R2(YTest, Yhat)
