"""
This module contains functions that can predict probabilities and labels given data, weights and noise levels
"""
import numpy as np
from gradient_descent import *


def predict_erm_proba(X,weights):
    argument = X@weights
    
    return sigmoid(argument)

def sigmoid(argument):
    argument = -argument
    mask = argument > 0
    val = np.empty_like(argument)
    val[mask] = np.exp(-argument[mask])/(1+np.exp(-argument[mask]))
    val[~mask] = 1/(1+np.exp(argument[~mask]))
    return val


def predict_erm(X,weights):
    return np.sign(predict_erm_proba(X,weights) - 0.5)
