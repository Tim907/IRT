import numpy as np

#next three lines added to specify the seed
import random
random.seed(2435)
#next lines are there for freeing the unused space
import gc


def matrix_P(Th, A, B, C):
    n = len(Th)  # Number of examinees
    k = len(A)  # Number of items
    #P = np.zeros((n, k))
    P = np.empty(shape=(n, k), dtype = float)
    for j in range(k):
        for i in range(n):
            P[i][j] = C[j] + (1 - C[j]) / (1 + np.exp(- A[j] * (Th[i] - B[j])))
    return (P)


# A
from scipy.stats import truncnorm
from scipy.stats import uniform


def sim_A(a=0.1, b=0.3, mean=0.2, sd=0.1, n=100, distr="tnormal"):
    if distr == "unif":
        A = uniform.rvs(a, b, size=n)
    else:
        A = truncnorm.rvs(a=a, b=b, loc=mean, scale=sd, size=n)
    return A


# B
def sim_B(a=0.1, b=0.3, mean=0.2, sd=0.1, n=100, distr="normal"):
    if distr == "unif":
        B = uniform.rvs(a, b, size=n)
    else:
        B = np.random.normal(loc=mean, scale=sd, size=n)
    return B


# Theta
def sim_Theta(a=0.1, b=0.3, mean=0.2, sd=0.1, n=1000, distr="normal"):
    if distr == "unif":
        Th = uniform.rvs(a, b, size=n)
    else:
        Th = np.random.normal(loc=mean, scale=sd, size=n)
    return Th


# C
def sim_C(setzero=False, a=0, b=0.5, mean=0.1, sd=0.05, n=100, distr="tnormal"):
    if setzero == True:
        C = np.zeros(n)
    elif distr == "uniform":
        C = uniform.rvs(a, b, size=n)
    else:
        C = np.random.normal(loc=mean, scale=sd, size=n)
    return C

# Computing of Labels

from scipy.stats import bernoulli

def sample_Labels(P,n,k):
    L = np.empty(shape=(n, k), dtype = int)
    for j in range(k):
        for i in range(n):
            L[i][j] = bernoulli.rvs(P[i][j], size=1)[0]
        if ((j+1)%100==0):
            print(j+1)
    return L



# values of k (number of items) and N (number of persons) to vary
k_range = [100] #[20, 50]
N_range = [50000]

import pandas as pd

#this was a loop
k = k_range[0] 
N = N_range[0] 

A = sim_A(a=0, b=6, mean=2.75, sd=0.3, n=k)
B = sim_B(mean=0, sd=1, n=k)
C = sim_C(sd=0.2, n=k, setzero=True)
Theta = np.random.normal(loc=0, scale=1, size=N)
P = matrix_P(Theta, A, B, C)

ABC = pd.concat([pd.DataFrame(A), pd.DataFrame(B), pd.DataFrame(C)], axis=1)
fileend = "k" + str(k) + "_N" + str(N) + "_r" + str(1)
ABC.to_csv("True_pars_" + fileend + ".csv", sep = ';') # have added , sep = ';' as the parameter
del ABC
gc.collect()
Theta = pd.DataFrame(Theta)
Theta.to_csv("True_thetas_"+fileend+".csv", sep = ';')

#print(P)

del Theta, A, B, C
gc.collect()
#from here on ABCTheta are not used

Labels = sample_Labels(P,N,k)
#Labels = sample_Labels(P) old text
#since from this point P is no longer used, delete it
del P
gc.collect()

Labels = pd.DataFrame(Labels)

# P = pd.DataFrame(P)
Labels.to_csv("Labels_"+fileend+".csv", sep = ';')
# P.to_csv("MatrixP"+fileend+".csv", sep = ';')



