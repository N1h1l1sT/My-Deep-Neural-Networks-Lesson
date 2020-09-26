#Ο Κώδικας σε αυτό το script είναι μέρος του μαθήματος "Νευρωνικά Δίκτυα 26: Κώδικας Backpropagation για Βαθύ Νευρωνικό Δίκτυο"
#https://www.youtube.com/watch?v=DROpEDFaadI
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
#Given by User
M <- c(50, 25, K) #Number of Hidden Units in Hidden Layers
Activations <- c("sigmoid", "relu", "softmax")

Cost <- function(Target, Y) -sum(Target * log(Y)) / NROW(Target)
Sigmoid <- function(a) 1 / (1 + exp(-a))
Softmax <- function(a) t(apply(exp(a), 1, function(x) {x/sum(x)}))
relu <- function(a) a * as.numeric(a > 0)

Sigma <- function(a, NonLinearity) {
  NonLinearity <- tolower(NonLinearity)
  if (NonLinearity %in% c("sigmoid", "s")) {
    a <- Sigmoid(a)
  } else if (NonLinearity %in% c("tanh")) {
    a <- tanh(a)
  } else if (NonLinearity %in% c("relu")) {
    a <- relu(a)
  } else if (NonLinearity %in% c("softmax")) {
    a <- Softmax(a)
  }
  return(a)
}

Forward <- function(X, WeightsAndActFunList) {
  Result <- list(X)

  for (i in 1:NROW(WeightsAndActFunList)) {
    Result[[i+1]] <- (Result[[i]] %*% WeightsAndActFunList[[i]][["W"]]) %+% WeightsAndActFunList[[i]][["b"]]
    Result[[i+1]] <- Sigma(Result[[i+1]], WeightsAndActFunList[[i]][["ActivationFunOnNextLayer"]])
  }

  return(Result)
}

derivative_w_last_layer <- function(Y, Target) Y - Target
derivative_b_last_layer <- function(Y, Target) apply(Y - Target, 2, sum)


Epochs <- 2000
L2 <- 0.1
LearningRate <- 1e-5
TrainCosts <- NULL
TestCosts <- NULL

M1 <- D #M0 basically
ANN_Info <- NULL

for (l in 1:NROW(M)) {
  M2 <- M[[l]]
  ANN_Info[[paste0("Layer", l)]][["W"]] <- matrix((rnorm(M1 * M2) / sqrt(M1)), ncol = M2)
  ANN_Info[[paste0("Layer", l)]][["b"]] <- rnorm(M2)
  ANN_Info[[paste0("Layer", l)]][["ActivationFunOnNextLayer"]] <- Activations[[l]]
  M1 <- M2
}
rm(l)
rm(M1)
rm(M2)

for (epoch in 1:Epochs) {
  Result <- Forward(XTrain, ANN_Info)

  if (epoch %% 100 == 0) {
    cost <- Cost(TTrain, Result[[NROW(Result)]])
    TrainCosts <- c(TrainCosts, cost)

    print(paste0("Cost: ", cost, ", Accuracy: ", mean(YTrain == apply(Result[[NROW(Result)]], 1, which.max))))

    TestResult <- Forward(XTest, ANN_Info)
    TestCosts <- c(TestCosts, Cost(TTest, TestResult[[NROW(TestResult)]]))
  }

  #Delta L
  ANN_Info[[paste0("Layer", NROW(ANN_Info))]][["DeltaW"]] <- derivative_w_last_layer(Result[[NROW(Result)]], TTrain)
  ANN_Info[[paste0("Layer", NROW(ANN_Info))]][["Deltab"]] <- derivative_b_last_layer(Result[[NROW(Result)]], TTrain)

  ANN_Info[[paste0("Layer", NROW(ANN_Info))]][["DerW"]] <- t(Result[[NROW(ANN_Info)]]) %*% ANN_Info[[paste0("Layer", NROW(ANN_Info))]][["DeltaW"]] + L2 * ANN_Info[[paste0("Layer", NROW(ANN_Info))]][["W"]]
  ANN_Info[[paste0("Layer", NROW(ANN_Info))]][["Derb"]] <- ANN_Info[[paste0("Layer", NROW(ANN_Info))]][["Deltab"]] + L2 * ANN_Info[[paste0("Layer", NROW(ANN_Info))]][["b"]]

  ANN_Info[[paste0("Layer", NROW(ANN_Info))]][["W"]] <- ANN_Info[[paste0("Layer", NROW(ANN_Info))]][["W"]] - LearningRate * ANN_Info[[paste0("Layer", NROW(ANN_Info))]][["DerW"]]
  ANN_Info[[paste0("Layer", NROW(ANN_Info))]][["b"]] <- ANN_Info[[paste0("Layer", NROW(ANN_Info))]][["b"]] - LearningRate * ANN_Info[[paste0("Layer", NROW(ANN_Info))]][["Derb"]]

  #Gradient Descent
  for (l in ((NROW(ANN_Info)-1):1)) {
    if (ANN_Info[[paste0("Layer", l)]][["ActivationFunOnNextLayer"]] %in% c("sigmoid", "s")) {
      ANN_Info[[paste0("Layer", l)]][["DeltaW"]] <- ANN_Info[[paste0("Layer", l+1)]][["DeltaW"]] %*% t(ANN_Info[[paste0("Layer", l+1)]][["W"]]) * (Result[[l+1]] * (1 - Result[[l+1]]))
    } else if (ANN_Info[[paste0("Layer", l)]][["ActivationFunOnNextLayer"]] == "tanh") {
      ANN_Info[[paste0("Layer", l)]][["DeltaW"]] <- ANN_Info[[paste0("Layer", l+1)]][["DeltaW"]] %*% t(ANN_Info[[paste0("Layer", l+1)]][["W"]]) * (1 - (Result[[l+1]] ^ 2))
    } else if (ANN_Info[[paste0("Layer", l)]][["ActivationFunOnNextLayer"]] == "relu") {
      ANN_Info[[paste0("Layer", l)]][["DeltaW"]] <- ANN_Info[[paste0("Layer", l+1)]][["DeltaW"]] %*% t(ANN_Info[[paste0("Layer", l+1)]][["W"]]) * sign(Result[[l+1]])
    }
    ANN_Info[[paste0("Layer", l)]][["Deltab"]] <- apply(ANN_Info[[paste0("Layer", l)]][["DeltaW"]], 2, sum)

    ANN_Info[[paste0("Layer", l)]][["DerW"]] <- t(Result[[l]]) %*% ANN_Info[[paste0("Layer", l)]][["DeltaW"]] + L2 * ANN_Info[[paste0("Layer", l)]][["W"]]
    ANN_Info[[paste0("Layer", l)]][["Derb"]] <- ANN_Info[[paste0("Layer", l)]][["Deltab"]]

    ANN_Info[[paste0("Layer", l)]][["W"]] <- ANN_Info[[paste0("Layer", l)]][["W"]] - LearningRate * ANN_Info[[paste0("Layer", l)]][["DerW"]]
    ANN_Info[[paste0("Layer", l)]][["b"]] <- ANN_Info[[paste0("Layer", l)]][["b"]] - LearningRate * ANN_Info[[paste0("Layer", l)]][["Derb"]]
  }
}


TestRes <- Forward(XTest, ANN_Info)
print(paste0("Accuracy: ", mean(YTest == apply(TestRes[[NROW(TestRes)]], 1, which.max))))

#Visualising the Costs
ggplot() +
  geom_line(aes(1:NROW(TrainCosts), TrainCosts), colour = "green") +
  geom_line(aes(1:NROW(TestCosts), TestCosts), colour = "red")


YTest[1:10]
apply(TestRes[[NROW(TestRes)]][1:10,], 1, which.max)
