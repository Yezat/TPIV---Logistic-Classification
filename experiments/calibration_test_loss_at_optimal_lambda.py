"""
Compute calibration and test_loss at optimal lambda
"""
from distutils.log import debug
from pickletools import optimize
from py import process
from sacred import Experiment
from sklearn.manifold import trustworthiness
import time


# import core

import os
import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 
from calibration import *
from experiment_information import ExperimentInformation
from experiment_information import NumpyEncoder
from util import error

# plot imports
from matplotlib import collections
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as mpimg
import matplotlib.colors as colors
from matplotlib import rcParams
import json
from mpl_toolkits.mplot3d import Axes3D
rcParams.update({'figure.autolayout': True})
from gradient_descent import *
import scipy as sp
from plot_optimal_choices import extract_parameters_from_filename
from experiment_information import get_experiment_information


if __name__ == "__main__":
    filenames = ["optimal_lambdas_min_n_300_max_n_5000_d_1000_tau_0.5_epsilon_0.01","optimal_lambdas_min_n_300_max_n_5000_d_1000_tau_0.5_epsilon_0.05","optimal_lambdas_min_n_300_max_n_5000_d_1000_tau_0.5_epsilon_0.0","optimal_lambdas_min_n_300_max_n_5000_d_1000_tau_0.5_epsilon_0.02","optimal_lambdas_min_n_300_max_n_5000_d_1000_tau_0.5_epsilon_0.03","optimal_lambdas_min_n_300_max_n_5000_d_1000_tau_0.5_epsilon_0.04"]

    # filenames = ["optimal_lambdas_min_n_300_max_n_5000_d_1000_tau_0.5_epsilon_0.01"]


    p = 0.75
    result_object = {}

    for filename in filenames:
        min_n, max_n, d, tau, epsilon = extract_parameters_from_filename(filename)
        

        with open(f"../data/{filename}.json", "r") as f:
            data = json.load(f)
            optimal_lambdas = data["optimal_lambdas"]
            optimal_lambdas_std = data["optimal_lambdas_std"]
            ns = np.array(data["ns"])
            alphas = ns/d

            result_object["epsilon_" + str(epsilon)] = {}
            result_object["epsilon_" + str(epsilon)]["alphas"] = alphas
            result_object["epsilon_" + str(epsilon)]["optimal_lambdas"] = optimal_lambdas
            result_object["epsilon_" + str(epsilon)]["optimal_lambdas_std"] = optimal_lambdas_std

            for i in range(len(optimal_lambdas)):
            
                result_object["epsilon_" + str(epsilon)]["alpha_" + str(alphas[i])] = []                

                print("Computing at alpha:", alphas[i], "for epsilon", epsilon, "optimal lambda", optimal_lambdas[i])

                lam = optimal_lambdas[i]

                for x in range(10):
                    w = sample_weights(d)
                    Xtrain, y = sample_training_data(w,d,ns[i],tau)
                    Xtest,ytest = sample_training_data(w,d,20000,tau)


                    w_gd = gd(Xtrain,y,lam,epsilon,debug=False)
                    
                    information = get_experiment_information(Xtest,w_gd,tau,y,Xtrain,w,ytest,d,"gd")
                    information.lam = lam
                    information.epsilon = epsilon                    
                    information.calibration = calc_calibration_analytical(information.rho,p,information.m,information.q,tau,debug=False)
                    result_object["epsilon_" + str(epsilon)]["alpha_" + str(alphas[i])].append(information)
    

    fname = f"../data/calibration_and_test_loss_at_optimal_lambdas_d_{d}_tau_{tau}_p_{p}.json"
    with open(fname, "w") as f:
        json.dump(result_object, f, cls=NumpyEncoder)
    print("Saved in ", fname)
            
