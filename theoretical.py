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
