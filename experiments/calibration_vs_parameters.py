"""
An experiment to plot calibration vs alpha
"""
from distutils.log import debug
from py import process
from datetime import datetime
import random
from sacred import Experiment

# import core
import os
import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 
from calibration import *
from experiment_information import ExperimentInformation, NumpyEncoder, get_experiment_information

# plot imports
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
import json
rcParams.update({'figure.autolayout': True})
from gradient_descent import *
from util import error

ex = Experiment('Calibration vs Parameters using GD on Adversarial Problem')

@ex.config
def my_config():
    lams = [10**-5]
    taus = [2]
    ps = [0.75]
    epsilons = [0.0,0.01,0.02,0.03,0.04,0.05]
    d = 300
    n_test = 20000
    number_of_runs = 40
    min_alpha = 0.2
    max_alpha = 5
    number_of_repeated_measurements = 10
    # methods = ["gd","L-BFGS-B","BFGS","CG"]
    methods = ["gd"]
    debug = False
    filename = "some_filename"

@ex.capture()
def get_configuration_dict(lams,taus,ps,epsilons,d,n_test,number_of_runs,min_alpha,max_alpha,number_of_repeated_measurements,methods):
    return {'lams':lams,'taus':taus,'ps':ps,'epsilons':epsilons,'d':d,'n_test':n_test,'number_of_runs':number_of_runs,'min_alpha':min_alpha,'max_alpha':max_alpha,'number_of_repeated_measurements':number_of_repeated_measurements,'methods':methods}

@ex.capture
def get_calibration(lam,tau,p,epsilon,d,n_test,n_train,method, debug):
    """
    # computes one calibration value

    w: weights of the model
    n: size of the test and train sets
    """    
    w = sample_weights(d)
    for i in range(10):
        try:

            Xtrain, y = sample_training_data(w,d,n_train,tau)
            Xtest,ytest = sample_training_data(w,d,n_test,tau)

            w_gd = np.empty(w.shape,dtype=w.dtype)
            if method == "gd":
                w_gd = gd(Xtrain,y,lam,epsilon,debug=debug)
            elif method == "L-BFGS-B":
                w_gd = lbfgs(Xtrain,y,lam,epsilon)
            else:
                raise Exception(f"Method {method} not implemented")
            
            information = get_experiment_information(Xtest,w_gd,tau,y,Xtrain,w,ytest,d,method)

            information.epsilon = epsilon
            information.lam = lam

            information.calibration = calc_calibration_analytical(information.rho,p,information.m,information.q,tau,debug=debug)

            if debug:
                print(json.dumps(information,cls=NumpyEncoder))

            return information
        except Exception as e:
            if i == 8:
                raise e    
    


# Note this function must be at the bottom of the file
@ex.automain
def my_main(filename,lams, taus, ps, epsilons, d, n_test, number_of_runs,number_of_repeated_measurements,min_alpha,max_alpha, methods):
    
        
    alphas = np.linspace(min_alpha,max_alpha,number_of_runs)
    print(f"Alphas",alphas)

    result_information = {}
    
    result_information["config"] = get_configuration_dict()
    print(f"Config",result_information["config"])
    result_information["timestamp"] = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    result_information["alphas"] = alphas

    for alpha in alphas:
        print( f"Alpha: {alpha}")
        result_information[f"alpha_{alpha}"] = {}
        for tau in taus:
            print( f"Tau: {tau}")
            result_information[f"alpha_{alpha}"][f"tau_{tau}"] = {}
            for lam in lams:
                print( f"Lambda: {lam}")
                result_information[f"alpha_{alpha}"][f"tau_{tau}"][f"lam_{lam}"] = {}
                for epsilon in epsilons:
                    print( f"Epsilon: {epsilon}")
                    result_information[f"alpha_{alpha}"][f"tau_{tau}"][f"lam_{lam}"][f"epsilon_{epsilon}"] = {}
                    for p in ps:
                        # print( f"P: {p}")
                        result_information[f"alpha_{alpha}"][f"tau_{tau}"][f"lam_{lam}"][f"epsilon_{epsilon}"][f"p_{p}"] = {}
                        for method in methods:
                            # print( f"Method: {method}")
                            result_information[f"alpha_{alpha}"][f"tau_{tau}"][f"lam_{lam}"][f"epsilon_{epsilon}"][f"p_{p}"][method] = {}
                            for i in range(number_of_repeated_measurements):
                                # print( f"Measurement: {i}")
                                result_information[f"alpha_{alpha}"][f"tau_{tau}"][f"lam_{lam}"][f"epsilon_{epsilon}"][f"p_{p}"][method][f"run_{i}"] = get_calibration(lam,tau,p,epsilon,d,n_test,int(alpha*d),method)
    
    filename = f"{filename}_exp_info"

    print(f"filename=\"{filename}\"")
    with open(f"../data/{filename}.json",'w') as f:
            json.dump(result_information,f,cls=NumpyEncoder)
    print("Saved")
    
    
    