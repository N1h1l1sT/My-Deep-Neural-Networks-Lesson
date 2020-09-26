#Ο Κώδικας σε αυτό το script είναι μέρος του μαθήματος "Νευρωνικά Δίκτυα 29: Κώδικας Convolution"
#https://www.youtube.com/watch?v=B7Pay3-MSMM
library(dplyr)
library(MASS)
library(ggplot2)
library(png)

Convolution <- function(X, W) {
  Hi <- NROW(X)
  Wi <- NCOL(X)
  FH <- NROW(W)
  FW <- NCOL(W)

  Ho <- Hi - FH + 1
  Wo <- Wi - FW + 1

  Y <- array(0, dim = c(Ho, Wo))

  for (ho in 1:Ho) {
    for (wo in 1:Wo) {
      Y[ho, wo] <- sum(X[ho:(ho + FH - 1), wo:(wo + FW - 1)] * W)
    }
  }

  return(Y)
}

Millena <- readPNG("C:/Put/Your/Own/Path/Here/Millena.png")
GreyMillena <- (Millena[,,1] + Millena[,,2] + Millena[,,3])/3
image(t(GreyMillena %>% apply(2, rev)), col = gray.colors(64))

W <- matrix(1/16 * c(1,2,1,2,4,2,1,2,1), ncol = 3) #Gaussian Filter

Y <- Convolution(GreyMillena, W)
image(t(Y %>% apply(2, rev)), col = gray.colors(64))

