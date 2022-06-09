"""
Compute the optimal choice of lambda given a setting including epsilon, alpha, d, tau etc...
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

ex = Experiment('Optimal choice of lambda')

@ex.capture()
def minimize_lambda(n,d,epsilon,tau,tol):
    """
    Compute the optimal lambda given a setting including epsilon, alpha, d, tau etc...

    parameters:
    n: number of samples
    d: dimension of the data
    epsilon: adversarial parameter
    tau: noise level
    """
    print("Computing optimal lambda",n,d,epsilon,tau)
    
    res = sp.optimize.minimize_scalar(lambda l : minimizer_function(l,n, d, epsilon,tau),method="bounded", bounds=[1e-3,0.5],options={'xatol': tol,'maxiter':100})
    # res = sp.optimize.minimize(minimizer_function, 2, args=(w,n,d,epsilon,tau),bounds = ([0,5],), method='L-BFGS-B', options={'maxiter': 100,'ftol':0.2, 'disp': False})
    # res = sp.optimize.minimize(minimizer_function, 1, args=(w,n,d,epsilon,tau),bounds = ([0,5],), method='BFGS', options={'maxiter': 100,'gtol':1e-05, 'disp': False})
    print("Minimized", "success:",res.success,"message",res.message)
    if not res.success:
        raise Exception("Optimization of lambda failed " + str(res.message))
    return res.x

@ex.capture()
def minimizer_function(lam,n,d,epsilon,tau,test_size_factor, repetitions_in_minimize,method):
    los = []
    
    for i in range(repetitions_in_minimize):    
        w = sample_weights(d)
        Xtrain, y = sample_training_data(w,d,n,tau)

        w_gd = np.empty(w.shape,dtype=w.dtype)

        if method == "gd":
            w_gd = gd(Xtrain,y,lam,epsilon)
        elif method == "L-BFGS-B":
            w_gd = lbfgs(w,Xtrain,y,lam,epsilon)
        else:
            raise Exception("Unknown method " + method)                      
        
        Xtest,ytest = sample_training_data(w,d,test_size_factor*n,tau)
        
        # Note epsilon must be zero, that way we compute the correct loss...
        lo = loss_per_sample(ytest,Xtest@w_gd,epsilon=0,w=w_gd).mean()
        los.append(lo)
    
    lo = np.mean(los)
    print("loss",lo,"std",np.std(los),"parameters",lam,n,d,epsilon,tau)
    return lo

@ex.config
def my_config():
    tau = 0.5
    epsilon = 0.0
    d = 300
    number_of_runs = 2
    min_alpha = 1
    max_alpha = 5
    number_of_repeated_measurements = 3
    repetitions_in_minimize = 20
    test_size_factor = 5
    method = "gd"
    tol=1e-3

@ex.automain
def my_main(d,min_alpha,max_alpha,number_of_runs,epsilon,tau,number_of_repeated_measurements,method):
    
    start = time.time()

    n = np.linspace(min_alpha*d,max_alpha*d,number_of_runs,dtype=int)
        
    parameters = f"min_n_{n[0]}_max_n_{n[-1]}_d_{d}_tau_{tau}_epsilon_{epsilon}"

    filename =f"optimal_lambdas_{parameters}"

    optimal_lambdas = []
    optimal_lambdas_std = []
    for n_ in n:
        ls = [] 
        for i in range(number_of_repeated_measurements):
            m = minimize_lambda(n_,d,epsilon,tau)    
            ls.append(m)
        optimal_lambdas.append(np.mean(ls))
        optimal_lambdas_std.append(np.std(ls))
        

    print("optimal lambdas",optimal_lambdas)
    print("optimal lambdas std",optimal_lambdas_std)
    print("alphas",n/d)
    fig,ax = plt.subplots()
    h1 = ax.errorbar(n/d,optimal_lambdas,yerr=optimal_lambdas_std,label="$\lambda_{loss}$ epsilon "+str(epsilon)+" tau "+str(tau))
    ax.legend(handles=[h1])
    ax.set_xlabel("$\\alpha$")
    ax.set_ylabel("$\lambda$")
    plt.savefig(f"../assets/{filename}.pdf")
    end = time.time()
    print("Time Elapsed",end-start)
    plt.show()

    result = {}
    result["optimal_lambdas"] = optimal_lambdas
    result["optimal_lambdas_std"] = optimal_lambdas_std
    result["ns"] = n
    with open(f"../data/{filename}.json","w") as f:
        json.dump(result,f,cls=NumpyEncoder)
    print("Done with computing optimal lambdas. Filename=",filename)

    