#!/bin/bash

# a bash script to compute the optimal epsilon for each entry in the optimal_lambdas.csv file

# Read in the optimal_lambdas.csv file
# The first line is the header, so skip it
# the columns are: alpha, epsilon, tau, lambda
# if epsilon is 0, then we run the optimal_epsilon.py alpha lam tau file.
# skip else

import os

# read the optimal_epsilons.csv file and store the results in a dictionary
knwon_epsilons = {}
with open("optimal_epsilons.csv","r") as f:
    lines = f.readlines()
    for line in lines[1:]:
        # remove the newline character
        line = line[:-1]
        alpha, epsilon, tau, lam = line.split(",")
        knwon_epsilons[(alpha,lam,tau)] = epsilon

with open("optimal_lambdas.csv","r") as f:
    lines = f.readlines()
    for line in lines[1:]:
        # remove the newline character
        line = line[:-1]
        alpha, epsilon, tau, lam = line.split(",")
        if float(epsilon) == 0:
            # check if we already know the optimal epsilon
            if (alpha,lam,tau) in knwon_epsilons:
                print(f"Skipping alpha {alpha} lam {lam} tau {tau} as we already know the optimal epsilon")
                continue
            print(f"Running optimal_epsilon.py {alpha} {lam} {tau}")
            os.system(f"python3 optimal_epsilon.py {alpha} {lam} {tau}")

