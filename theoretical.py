"""
This module contains functions that can predict probabilities and labels given data, weights and noise levels
"""
from scipy.special import erfc
import numpy as np


# plot imports
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams

from erm import *
rcParams.update({'figure.autolayout': True})

from data import *
from gradient_descent import *



def predict(X, weights, tau):
    """
    this function predicts n labels y of dimension (d,)  given test data X (n,d)

    X: test data X (n,d)
    weights: (d,)
    tau: noise level
    """
    return np.sign(predict_proba(X,weights,tau) - 0.5)

def predict_proba(X, weights, tau):
    """
    this function predicts probabilities for n labels y of dimension (d,)  given test data X (n,d)

    X: test data X (n,d)
    weights: (d,)
    tau: noise level
    """
    val = 0.5
    argument = - (1/np.sqrt(2 * tau**2)) * ( X @ weights)
    
    argument[argument > 20] = 20
    argument[argument < -20] = -20 
    val = erfc( argument)/2
    if np.isnan(val).any():
        print("WARNING: NAN in predict_proba")
    return val

def predict_erm_proba(X,weights,debug=False):
    val = 0.5
    argument = -X@weights
    
    argument[argument > 20] = 20
    argument[argument < -20] = -20 
    if debug:
        print(np.min(argument), np.max(argument), np.mean(argument),np.median(argument),np.std(argument),np.linalg.norm(argument,2))
    val = 1/(1+np.exp(argument))
    if np.isnan(val).any():
        print("WARNING: NAN in predict_erm_proba")
    return val


def predict_erm(X,weights):
    return np.sign(predict_erm_proba(X,weights) - 0.5)
