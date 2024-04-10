# Computing the matrix P of success probabilities
matrix_P <- function(Th,A,B,C){
  n <- length(Th) # Number of examinees
  k <- length(A) # Number of items
  P <- matrix(numeric(n*k), nrow = n)
  for(j in 1:k){
    for(i in 1:n){
      P[i,j] <- C[j] + (1 - C[j])/(1 + exp(-A[j] * (Th[i] - B[j])))
    }
  }
  return(P)
}


library(truncnorm)

# A
sim_A <- function(a = 0.1, b = 0.3, mean = 0.2, sd =0.1, n = 100, distr = "tnormal"){
  if(distr == "uniform"){
    A <- runif(n = n, min = a, max = b)
  }
  else{
    A <- rtruncnorm(n = n, a = 0, b = Inf, mean = mean, sd = sd)
  }
  return(A)
}

#B
sim_B <- function(a = 0, b = 300, mean = 0, sd = 1, n = 100, distr = "normal"){
  if(distr == "uniform"){
    v <- runif(n = n, min = a, max = b)
  }
  else{
    v <- rnorm(n = n, mean = mean, sd = sd)
  }
  return(v)
}
# Theta 
sim_Theta <- function(a = 0, b = 300, mean = 150, sd = 15, n = 1000, distr = "normal"){
  if(distr == "uniform"){
    v <- runif(n = n, min = a, max = b)
  }
  else{
    v <- rnorm(n = n, mean = mean, sd = sd)
  }
  return(v)
}

# C
sim_C <- function(setzero = FALSE, a = 0, b = 0.5,
                  mean = 0.1, sd = 0.05, n = 100, distr = "tnormal"){
  if(setzero == TRUE){
    C <- numeric(n)
  }
  else{
  if(distr == "uniform"){
    C <- runif(n = n, min = a, max = b)
  }
  else{
    C <- rtruncnorm(n = n, a = a, b = b, mean = mean, sd = sd)
  }
  }
  return(C)
}

# Computing of the labels based on the matrix P
sample_Labels <- function(P, Label = "0,1"){
  L <- apply(P, c(1,2), function(x) rbinom(1,1,x))
  if(Label == "-1,1"){
    L <- L * 2 - 1
  }
  return(L)
}

library(mirt)
# Set the seed for the random numbers generator
set.seed(2435)

#values of k (number of items) and N (number of persons) to vary
k_range <- c(100) # set here the number of items #c(100, 200, 500)
N_range <- c(50000) # set here the number of examinees #c(100000, 200000,500000)

#number of replications -> change this if you want to replicate the same condition R times
R <- 1

design_sim <- expand.grid(N_items = k_range, N_persons = N_range, replication = 1:R) 

for (d in 1:nrow(design_sim)) {
  
  k <- design_sim[d, "N_items"]
  N <- design_sim[d, "N_persons"]
  
  A <- sim_A(a=0,b=6, mean = 2.75,sd=0.3, n=k)
  B <- sim_B(mean = 0, sd = 1, n=k)
  C <- sim_C(sd=0.1, n=k, setzero = FALSE) # this line is used to generate C in 3PL models
  #C <- rep(.25, k) # this line sets the fixed value for the guessing parameter 
  Theta <- rnorm(N)
  P <- matrix_P(Theta,A,B,C) # or use the function simdata() from mirt?
  Labels <- sample_Labels(P)
  
  # save P as CSV file
  fileend <- paste0("k", as.character(design_sim[d, "N_items"]),
                    "_N", as.character(design_sim[d, "N_persons"]),
                    "_r", as.character(design_sim[d, "replication"]))
  
  write.csv(cbind(A,B,C), paste0("True_pars_", fileend, ".csv"), row.names = FALSE)
  write.csv(Theta, paste0("True_thetas_", fileend, ".csv"), row.names = FALSE)
  write.csv2(Labels, paste0("Labels_", fileend, ".csv"))
  # write.csv2(P, paste0("MatrixP_", fileend, ".csv"))
}