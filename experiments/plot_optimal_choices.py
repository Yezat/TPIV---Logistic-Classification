"""
Plot the optimal choice of lambda given a setting including epsilon, alpha, d, tau etc...
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

def extract_parameters_from_filename(filename):
    """
    Extract the parameters from the filename
    """
    splits = filename.split("_")
    min_n = int(splits[4])
    max_n = int(splits[7])
    d = int(splits[9])
    tau = float(splits[11])
    epsilon = float(splits[13])
    return min_n,max_n,d,tau,epsilon


if __name__ == "__main__":
    
    # filenames = ["optimal_lambdas_min_n_300_max_n_5000_d_1000_tau_0.5_epsilon_0.01","optimal_lambdas_min_n_300_max_n_5000_d_1000_tau_0.5_epsilon_0.05","optimal_lambdas_min_n_300_max_n_5000_d_1000_tau_0.5_epsilon_0.0","optimal_lambdas_min_n_300_max_n_5000_d_1000_tau_0.5_epsilon_0.02","optimal_lambdas_min_n_300_max_n_5000_d_1000_tau_0.5_epsilon_0.03","optimal_lambdas_min_n_300_max_n_5000_d_1000_tau_0.5_epsilon_0.04"]

    filenames = ["optimal_lambdas_min_n_300_max_n_5000_d_1000_tau_0.5_epsilon_0.01","optimal_lambdas_min_n_300_max_n_5000_d_1000_tau_0.5_epsilon_0.0","optimal_lambdas_min_n_300_max_n_5000_d_1000_tau_0.5_epsilon_0.02"]

    fig,ax = plt.subplots(1,1,figsize=(16,9))

    handles = []
    optimal_lams = []
    optimal_lams_std = []
    alphas = []
    epsilons = []
    for filename in filenames:
        min_n, max_n, d, tau, epsilon = extract_parameters_from_filename(filename)

        with open(f"../data/{filename}.json", "r") as f:
            data = json.load(f)
            optimal_lambdas = data["optimal_lambdas"]
            optimal_lambdas_std = data["optimal_lambdas_std"]
            ns = np.array(data["ns"])
            h1 = ax.errorbar(ns/d,optimal_lambdas,yerr=optimal_lambdas_std,label="$\lambda_{loss}$ $\\varepsilon$ "+str(epsilon))
            handles.append(h1)
            optimal_lams.append(optimal_lambdas[-1])
            optimal_lams_std.append(optimal_lambdas_std[-1])
            alphas.append(ns[-1]/d)
            epsilons.append(epsilon)
    
    ax.legend(handles=handles)
    # ax.set_xlabel("$\\alpha$")
    # ax.set_ylabel("$\lambda$")

    font_size = 20
    font = {'size':font_size}

    plt.title("Optimal $\lambda$",fontdict=font)
    plt.xlabel("$\\alpha$",fontdict=font,labelpad=font_size)
    plt.ylabel("$\lambda$",fontdict=font,labelpad=font_size)
    plt.setp(ax.get_xticklabels(),fontsize=font_size)
    plt.setp(ax.get_yticklabels(),fontsize=font_size)
    plt.legend(prop={'size': font_size})

    plt.savefig(f"../assets/combined_optimal_lambdas.pdf")
    plt.show()
    print(optimal_lams)
    print(optimal_lams_std)
    print(epsilons)
    print(alphas)
