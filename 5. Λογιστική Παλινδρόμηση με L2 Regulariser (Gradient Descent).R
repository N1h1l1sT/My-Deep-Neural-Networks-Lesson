#Ο Κώδικας σε αυτό το script είναι μέρος του μαθήματος "Νευρωνικά Δίκτυα 17: Κώδικας Λογιστικής Παλινδρόμησης με L2 Regulariser"
#https://www.youtube.com/watch?v=jtO3jx6dp-U
library(dplyr)
library(MASS)
library(ggplot2)

###########
#Functions#
###########
sigmoid <- function(z) 1/(1 + exp(-z))

##########################################
#Multiple Independent w/ Binary Dependent#
##########################################
N <- 2000
D <- 40
X <- matrix(runif(N*D, -5, 5), ncol = D) #Uniformly Distributed -5 to 5
TrueWeights <- c(1, 0.4, -0.6, rep(0, D-3))
Y <- round(sigmoid((X %*% TrueWeights) + rnorm(N, 0, sqrt(0.5))))[,1]
XTrain <- X[1:(NROW(X)*0.8),]
YTrain <- Y[1:(NROW(X)*0.8)]
XTest <- X[((NROW(X)*0.8)+1):NROW(X),]
YTest <- Y[((NROW(X)*0.8)+1):NROW(X)]

#Visualising the Dataset
library(gg3D)
ggplot(X %>% as_tibble(), aes(x = V1, y = V2, z = V3, colour = as.factor(Y))) +
  axes_3D() +
  stat_3D() +
  labs_3D(labs = c("V1", "V2", "V3")) +
  theme_void()

##########################################################
#Logistic Regression w/ Gradient Descent (L2 Regulariser)#
##########################################################
NUM_Epochs <- 10000
LearningRate <- 1e-6
D <- NCOL(XTrain)
N <- NROW(XTrain)
L2 <- 0.1
Costs <- NULL

W <- rnorm(D, 0, sqrt(1/D)) #Randomly Initialising the Weights

for (i in 1:NUM_Epochs) {
  Yhat <- sigmoid((XTrain %*% W)[,1])
  Residuals <- Yhat - YTrain
  W <- W - LearningRate * ((t(XTrain) %*% Residuals) + as.vector(L2*W)) #Update W

  Costs <- c(Costs, -mean(YTrain * log(Yhat) + (1 - YTrain) * log(1 - Yhat)) + mean(L2*W))
}

Yhat <- sigmoid((XTest %*% W)[,1])
mean(YTest == round(Yhat))

#Visualising the Costs
ggplot() + geom_line(aes(x = 1:NROW(Costs), y = Costs))


YTest[250]
round(Yhat[250])
