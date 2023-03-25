"""
This module contains code for custom gradient descent
"""
from logging import exception
from scipy.special import erfc
import numpy as np

import traceback
# plot imports
from matplotlib import collections
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as mpimg
import matplotlib.colors as colors
from matplotlib import rcParams
import json
from mpl_toolkits.mplot3d import Axes3D
from zmq import XPUB

from erm import *
rcParams.update({'figure.autolayout': True})

from data import *
import theoretical
import math
from scipy.optimize import basinhopping
from scipy.optimize import minimize
# from sklearn.linear_model import LinearModelLoss
# from sklearn._loss import LinearModelLoss
from sklearn.linear_model._linear_loss import LinearModelLoss
from sklearn.utils.validation import _num_samples
import sklearn.utils.validation as sk_validation
from sklearn._loss import HalfBinomialLoss
from sklearn.linear_model._logistic import _logistic_regression_path
import inspect as i
from scipy.special import logit

def loss_gradient(w,X,y,lam,epsilon):
    return total_loss(w,X,y,lam,epsilon), total_gradient(w,X,y,lam,epsilon)

def total_loss(w,X,y,lam,epsilon):
    loss = loss_per_sample(y,X@w,epsilon,w)
    loss = loss.sum()
    loss += 0.5 * lam * (w @ w)
    return loss

def loss_per_sample(y,raw_prediction, epsilon, w):
    raw_prediction = -y*raw_prediction + epsilon * np.linalg.norm(w)
    raw_prediction[raw_prediction > 20] = 20
    raw_prediction[raw_prediction < -20] = -20
    return np.log(1+np.exp(raw_prediction))

def total_gradient(w,X,y,lam,epsilon):
    grad = np.empty_like(w, dtype=w.dtype)
    grad_per_sample,epsilon_part = gradient_per_sample(w,X,y,epsilon)
    grad = grad_per_sample.T @ X  
    grad += epsilon_part
    grad += lam * w
    return grad

def gradient_per_sample(w,X,y,epsilon):
    p = y*(X@w) - epsilon * np.linalg.norm(w)
    p[p > 20] = 20
    p[p < -20] = -20
    b = 1/(1+np.exp(p))
    c = epsilon*w/np.linalg.norm(w)
    d = np.outer(b,c).sum(axis=0)
    return -y*b, d


def lbfgs(X,y,lam,epsilon,method="L-BFGS-B"):
    n,d = X.shape
    res = minimize(loss_gradient,sample_weights(d),args=(X, y, lam, epsilon),jac=True,method=method,options={'maxiter':100}) #"iprint": iprint,
    print("Minimized", "success:",res.success,"message",res.message)
    if not res.success:
        print("Optimization of Loss failed " + str(res.message))
    w = res.x
    return w

def gd(X,y,lam,epsilon, debug = False):
    n,d = X.shape
    w0 = sample_weights(d)
    wt = sample_weights(d)

    n_iter = 0 
    gradient_norm = 1
    loss_difference = 1
    training_error = 1
    learning_rate = 1000000

    last_loss = total_loss(w0,X,y,lam,epsilon)

    while gradient_norm > 10**-9 and loss_difference > 10**-9 and training_error != 0 and n_iter < 10000:
        # if debug:
        #     print("iteration: ",n_iter," loss: ",last_loss,"gradient_norm",gradient_norm,"learning_rate",learning_rate,"epsilon",epsilon,"lam",lam)

        
        g = total_gradient(w0,X,y,lam,epsilon)
        wt = w0 - learning_rate *g
        gradient_norm = np.linalg.norm(g)

        # if debug:
        #     print_loss(w0,total_loss,g,learning_rate,lam,X,y,epsilon)


        new_loss = total_loss(wt,X,y,lam,epsilon)
        if new_loss > last_loss:
            learning_rate *= 0.4
            continue
        loss_difference = last_loss - new_loss
        last_loss = new_loss
        

        w0 = wt
        training_error = 0.25*np.mean((y-np.sign(X@wt))**2)
        n_iter += 1

    if debug:
        print("GD converged after ",n_iter," iterations", "loss: ",last_loss,"gradient_norm",gradient_norm,"learning_rate",learning_rate,"epsilon",epsilon,"lam",lam,"alpha",n/d)
    return wt



def print_loss(w0,loss_fct,gradient,learning_rate,lam,X,y,epsilon):
    ts = np.linspace(-learning_rate,learning_rate,100)

    loss_gradient = []
    
    for t in ts:
        loss_gradient.append(loss_fct(w0+t*gradient,X,y,lam,epsilon))
    
    fig,ax = plt.subplots()
    plt_loss_over_time, = ax.plot(ts,loss_gradient,label="loss")
    # ax.scatter(0,loss(wt,X,y,lam,epsilon),label="loss at solution")
    ax.legend(handles=[plt_loss_over_time])
    plt.title("loss")
    plt.xlabel("gradient_direction")
    plt.ylabel("loss")
    plt.show()

