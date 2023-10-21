import numpy as np
from scipy.special import erfc
import json
from theoretical import sigmoid
import theoretical

def error(y, yhat):
    return 0.25*np.mean((y-yhat)**2)

def adversarial_error(y, Xtest, w_gd, epsilon):
    d = Xtest.shape[1]
    y_hat = np.sign( sigmoid( Xtest@w_gd - y*epsilon * np.sqrt( w_gd@w_gd /d)  ) - 0.5)

    # y_hat_prime = np.sign(-Xtest @ w_gd + y * epsilon * np.sqrt(np.linalg.norm(w_gd, 2) / d))

    # assert np.all(y_hat == y_hat_prime)

    return error(y, y_hat)

def random_positive_definite_matrix(d, scaling = 50, variance = 2):
    """
    Returns a random positive definite matrix of size d x d
    """
    A = np.random.normal(0,variance,size=(d,d))
    return A.T @ A + scaling * np.eye(d)

def power_law_diagonal_matrix(d,alpha = 1.1):
    """
    Returns a diagonal matrix with a spectra that is power-law distributed
    """
    return np.diag([d/(k+1)**alpha for k in range(d)])

def sigma_star(x):
    """
    Returns 0.5 * erfc(-x/sqrt(2))
    """
    return 0.5 * erfc(-x/np.sqrt(2))

def logistic_function(x):
    """
    Returns the logistic function of x
    """
    return 1/(1+np.exp(-x))

def generalization_error(rho_w_star,m,q, tau):
    """
    Returns the generalization error in terms of the overlaps
    """
    return np.arccos(m / np.sqrt( (rho_w_star + tau**2 ) * q ) )/np.pi

def sample_interesting_stuff(tau):
    # let's say we stumble upon computing the following expecation value:
    # E[ X * w_star * sign(X * w_star + tau*xi) * epsilon * sqrt(1/d) * sqrt(norm(w_gd)) ]
    # where X is a random vector with iid entries, xi is a random variable with standard normal distribution
    # for now we ignore the multiplicative factor epsilon * sqrt(1/d) * sqrt(norm(w_gd))
    
    # Let's sample X and xi
    d = 100000
    n = 1000
    X = np.random.normal(size=(n,d))
    xi = np.random.normal(size=(n,1))
    w = np.random.normal(size=(d,1))
    expecation_value = np.mean( X @ w * np.sign(X @ w + tau*xi) ) / np.sqrt(d)
    return expecation_value


if __name__ == "__main__":
    print(sample_interesting_stuff(1.0))
