#Ο Κώδικας σε αυτό το script είναι μέρος του μαθήματος "Νευρωνικά Δίκτυα 22: Κώδικας Νευρωνικού 2 επιπέδων"
#https://www.youtube.com/watch?v=ogWCrq_AXqQ
library(dplyr)
library(MASS)
library(ggplot2)

`%+%` <- function(Mat, Vec) sweep(x = Mat, MARGIN = 2, Vec, FUN = "+")

###############################################
#Multiple Independent w/ Categorical Dependent#
###############################################
NPerClass <- 1000
X_Class1 <- sweep(x = matrix(rnorm(NPerClass * 2), ncol = 2), MARGIN = 2, c(0, -2), FUN = "+")
X_Class2 <- sweep(x = matrix(rnorm(NPerClass * 2), ncol = 2), MARGIN = 2, c(2, 2), FUN = "+")
X_Class3 <- sweep(x = matrix(rnorm(NPerClass * 2), ncol = 2), MARGIN = 2, c(-2, 2), FUN = "+")
X <- rbind(X_Class1, X_Class2, X_Class3)
Y <- c(rep(1, NPerClass), rep(2, NPerClass), rep(3, NPerClass)) %>% as.factor()
Ind <- sample.int(NROW(X))
X <- X[Ind,]
Y <- Y[Ind]
XTrain <- X[1:(NROW(X)*0.8),]
YTrain <- Y[1:(NROW(X)*0.8)]
XTest <- X[((NROW(X)*0.8)+1):NROW(X),]
YTest <- Y[((NROW(X)*0.8)+1):NROW(X)]

N <- NROW(X)
D <- NCOL(X)
K <- Y %>% unique() %>% NROW()

Target <- matrix(rep(0, N * K), ncol = K)
for (i in 1:N) {
  Target[i, Y[i]] <- 1
}
TTrain <- Target[1:(NROW(X)*0.8),]
TTest <- Target[((NROW(X)*0.8)+1):NROW(X),]

#Visualising the Dataset
ggplot() +
  geom_point(aes(x = X[,1], y = X[,2], colour = Y), alpha = 0.4, size = 2)


####################
## Neural Network ##
####################
Sigmoid <- function(a) 1 / (1 + exp(-a))

Softmax <- function(a) {
  Y <- t(apply(exp(a), 1, function(x) {x/sum(x)}))
  return(Y)
}

Cost <- function(Target, Y) {
  N <- NROW(Target)
  return(-sum(Target * log(Y)) / N)
}

Forward <- function(X, W1, b1, W2, b2) {
  Z1 <- Sigmoid((X %*% W1) %+% b1)
  Y <- Softmax((Z1 %*% W2) %+% b2)

  return(
    list(
      Y = Y,
      Z1 = Z1
    )
  )
}

Derivative_W2 <- function(Z1, Target, Y) {
  # N <- NROW(Target)
  # K <- NCOL(Target)
  # M1 <- NCOL(Z1)
  #
  # Res <- matrix(rep(0, M1 * K), ncol = K)
  # for (n in 1:N) {
  #   for (m1 in 1:M1) {
  #     for (k in 1:K) {
  #       Res[m1, k] <- Res[m1, k] + (-(Target[n, k] - Y[n, k]) * Z1[n, m1])
  #     }
  #   }
  # }
  #
  # return(Res)
  return(-(t(Z1) %*% (Target - Y)))
}

Derivative_b2 <- function(Target, Y) {
  # N <- NROW(Target)
  # K <- NCOL(Target)
  #
  # Res <- vector("numeric", K)
  # for (n in 1:N) {
  #   for (k in 1:K) {
  #     Res[k] <- Res[k] + Target[n, k] - Y[n, k]
  #   }
  # }
  #
  # return(Res)
  return(apply(-(Target - Y), 2, sum))
}

Derivative_W1 <- function(X, Z1, Target, Y, W2) {
  N <- NROW(X)
  D <- NCOL(X)
  M1 <- NROW(W2)
  K <- NCOL(W2)

  Res <- matrix(rep(0, D * M1), ncol = M1)

  # for (n in 1:N) {
  #   for (d in 1:D) {
  #     for (m1 in 1:M1) {
  #       for (k in 1:K) {
  #         Res[d, m1] <- Res[d, m1] + (-(Target[n, k] - Y[n, k]) * W2[m1, k] * Z1[n, m1] * (1 - Z1[n, m1]) * X[n, d])
  #       }
  #     }
  #   }
  # }
  #
  # return(Res)
  return(-(t(X) %*% (((Target - Y) %*% t(W2)) * (Z1 * (1-Z1)))))
}

Derivative_b1 <- function(Z1, Target, Y, W2) {
  Res <- -(((Target - Y) %*% t(W2)) * (Z1 * (1-Z1)))
  return(apply(Res, 2, sum))
}

Epochs <- 5000

M1 <- 10 #Number of Hidden Units in Hidden Layer
W1 <- matrix(rnorm(D * M1), ncol = M1)
b1 <- rnorm(M1)
W2 <- matrix(rnorm(M1 * K), ncol = K)
b2 <- rnorm(K)

LearningRate <- 1e-5
TrainCosts <- NULL
TestCosts <- NULL

system.time(
  for (epoch in 1:Epochs) {
    Result <- Forward(XTrain, W1, b1, W2, b2)

    if (epoch %% 100 == 0) {
      cost <- Cost(TTrain, Result$Y)
      TrainCosts <- c(TrainCosts, cost)

      print(paste0("Cost: ", cost, ", Accuracy: ", mean(YTrain == apply(Result$Y, 1, which.max))))

      TestCosts <- c(TestCosts, Cost(TTest, Forward(XTest, W1, b1, W2, b2)$Y))
    }

    #Gradient Descent
    W2 <- W2 - LearningRate * Derivative_W2(Result$Z1, TTrain, Result$Y)
    b2 <- b2 - LearningRate * Derivative_b2(TTrain, Result$Y)
    W1 <- W1 - LearningRate * Derivative_W1(XTrain, Result$Z1, TTrain, Result$Y, W2)
    b1 <- b1 - LearningRate * Derivative_b1(Result$Z1, TTrain, Result$Y, W2)
  }
)

TestRes <- Forward(XTest, W1, b1, W2, b2)$Y
print(paste0("Accuracy: ", mean(YTest == apply(TestRes, 1, which.max))))

YTest[1]
apply(TestRes, 1, which.max)[1]

#Visualising the Costs
ggplot() +
  geom_line(aes(1:NROW(TrainCosts), TrainCosts), colour = "green") +
  geom_line(aes(1:NROW(TestCosts), TestCosts), colour = "red")
