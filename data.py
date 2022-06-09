"""
This module provides various sampling functions
"""

from matplotlib import pyplot as plt
import numpy as np


def get_iid_gaussian(d, number_of_samples=1, dimension_adjusted_variance = False ):
    """
    returns a (n,d) dimensional normal sample

    d: dimension
    number_of_samples: number of d dimensional samples to produce
    dimension_adjusted_variance: divide variance by dimension
    """
    variance_term = 1 if not dimension_adjusted_variance else d
    return np.random.normal(0.,1./np.sqrt(variance_term),size=(number_of_samples,d))

def sample_weights(d):
    """
    returns a sample weight of dimension (d,)
    Weights are iid gaussians with mean 0 and variance 1

    d: dimension
    """
    return np.reshape(get_iid_gaussian(d), (d,))

def sample_training_data(weights, d, n,tau):
    """
    Samples training data
    Generates X iid gaussians with mean 0 and variance 1/d.
    Uses these and the weights to generate labels y with a noise level of tau.
    returns the X, the y

    parameters:
    d: dimension
    n: number of training data
    weights: (d,) dimensional vector of weights
    tau: noise level >= 0
    """
    X = get_iid_gaussian(d,n,True)
    y = generate_labels(weights,X,tau)
    return X,y

def sample_test_data(d, n):
    """
    Samples test data
    Generates X iid gaussians with mean 0 and variance 1/d.

    parameters:
    d: dimension
    n: number of training data

    returns the X (n,d)
    """
    return get_iid_gaussian(d,n,True)

def sample_noise(n):
    """
    returns sample noise of dimension (n,)
    an iid gaussian with mean 0 and variance 1

    n: dimension
    """
    return sample_weights(n)

def generate_labels(weights, xtest, tau):
    """
    generates labels for test data 
    If noisy, then the labels are generated with noise level tau.
    The labels are generated according to sign(xtest @ weights + tau * noise)
    The noise is iid gaussian with mean 0 and variance 1

    weights: (d,)
    xtest: test data (n,d)
    tau: noise level
    """
    #generate noise:
    n = xtest.shape[0]
    noise = sample_noise(n)
    return np.sign(xtest @ weights + tau * noise)


if (__name__ == "__main__"):
    ds = np.linspace(20,8000,10)
    weights = []
    first_param = []
    second_param = []
    for d in ds:
        print("d = ",d)
        w = sample_weights(int(d))
        X,y = sample_training_data(w,int(d),5000,2)
        weights.append(np.linalg.norm(w,2)/np.sqrt(d))
        first_param.append(np.mean( y*(X@w)) )
        
        p = X @ w # shape (n,)

        epsilon = 0.
        exp_term = np.exp( y*p - epsilon * np.linalg.norm(w,2)/np.sqrt(d) ) # shape (n,)
        b = ( 1/ ( 1 + exp_term) ) # shape (n,)
        second_param.append(np.mean((b * y)@X/5000) )
    fig, ax = plt.subplots()
    handle1, = ax.plot(ds,weights,label="weights")
    handle2, = ax.plot(ds,first_param,label="first parameter")
    handle3, = ax.plot(ds,second_param,label="second parameter")
    ax.legend(handles=[handle1,handle2])
    ax.set_xlabel("dimension")
    ax.set_ylabel("weight norm")
    plt.show()



