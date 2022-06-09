"""
Plot calibration and test_loss at optimal lambda
"""
from distutils.log import debug
from http.client import CONTINUE
from pickletools import optimize
from re import A
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

def plot_for_key(key,at_epsilons):
    fig, ax = plt.subplots(1,1,figsize=(16,9))
    handles = []

    for epsilon in at_epsilons:
        
        # epsilon = k.split("_")[-1]
        # print("epsilon",epsilon,at_epsilons)
        # if epsilon not in at_epsilons:
        # continue
        k = "epsilon_"+str(epsilon)
        
        alphas = np.array(data_at_optimal_lambdas[k]["alphas"])
        

        losses = []
        losses_std = []
        for alpha in alphas:

            ls = []
            informations = data_at_optimal_lambdas[k]["alpha_{}".format(alpha)]
            for information in informations:
                ls.append(information[key])
            losses.append(np.mean(ls))
            losses_std.append(np.std(ls))
        
        h = ax.errorbar(alphas, losses,yerr=losses_std, label=f"$\\varepsilon$={epsilon}")
        handles.append(h)
    
    ax.legend(handles=handles)
    font_size = 20
    font = {'size':font_size}

    plt.title(f"{key} at optimal lambda",fontdict=font)
    plt.xlabel("$\\alpha$",fontdict=font,labelpad=font_size)
    if key == "calibration":
        plt.ylabel("$\Delta p$",fontdict=font,labelpad=font_size)
    else:
        plt.ylabel(key,fontdict=font,labelpad=font_size)
    plt.setp(ax.get_xticklabels(),fontsize=font_size)
    plt.setp(ax.get_yticklabels(),fontsize=font_size)
    plt.legend(prop={'size': font_size})
    plt.savefig(fname=f"../assets/{key}_at_optimal_lambda.pdf")
    plt.show()



if __name__ == "__main__":
    d = 1000
    tau = 0.5
    p = 0.75
    filename = f"../data/calibration_and_test_loss_at_optimal_lambdas_d_{d}_tau_{tau}_p_{p}.json"

    data_at_optimal_lambdas = {}

    with open(filename, "r") as f:
        data_at_optimal_lambdas = json.load(f)
    
    at_epsilons = [0.00,0.01,0.02]
    
    # plot calibration at optimal lambda
    plot_for_key("test_loss",at_epsilons)

    # plot test loss at optimal lambda
    plot_for_key("calibration",at_epsilons)
    plot_for_key("generalization_error",at_epsilons)