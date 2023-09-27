#!/bin/bash

# a bash script to compute the optimal epsilon for each entry in the optimal_lambdas.csv file

# Read in the optimal_lambdas.csv file
# The first line is the header, so skip it
# the columns are: alpha, epsilon, tau, lambda
# if epsilon is 0, then we run the optimal_epsilon.py alpha lam tau file.
# skip else

import os

optimal_lambdas_filename = "optimal_lambdas.csv"
optimal_lambdas_filename = "fashion_mnist_optimal_lambdas.csv"

# read the optimal_lambdas.csv file and store the results in a dictionary
knwon_lambdas = {}
with open(optimal_lambdas_filename,"r") as f:
    lines = f.readlines()
    for line in lines[1:]:
        # remove the newline character
        line = line[:-1]
        alpha, epsilon, tau, lam = line.split(",")
        knwon_lambdas[(alpha,epsilon,tau)] = lam


alphas = [0.1,0.5,1,2,5,10,20,50,100]
alphas = [12000/784]
epsilon = 0.0
# taus = [0.1,0.5,1,1.5,2.0,2.5]
taus = [0]

# print(knwon_lambdas)

for alpha in alphas:
    for tau in taus:
        query = (str(float(alpha)),str(float(epsilon)),str(float(tau)))
        # print(query)
        if query in knwon_lambdas:
            print(f"Skipping alpha {alpha} epsilon {epsilon} tau {tau} as we already know the optimal lambda")
            continue
        else:
            print(f"Running optimal_choice.py {alpha} {epsilon} {tau} {optimal_lambdas_filename}")
            os.system(f"python3 optimal_choice.py {alpha} {epsilon} {tau} {optimal_lambdas_filename}")
