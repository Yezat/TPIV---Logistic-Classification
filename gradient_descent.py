"""
This module contains code for custom gradient descent
"""
import numpy as np
import logging
import time
import numpy as np
from data import *
from scipy.optimize import minimize
from sklearn.utils.validation import check_array, check_consistent_length
from scipy.linalg import norm
from scipy.linalg import eigvalsh
from scipy.sparse.linalg import eigsh
import mpmath

"""
sklearn - expects labels as -1 and 1.
"""
def preprocessing(coef, X, y, lam, epsilon):
    # heavily inspired by the sklearn code, with hopefully all the relevant bits copied over to make it work using lbfgs
    solver = "lbfgs"
    X = check_array(
            X,
            accept_sparse="csr",
            dtype=np.float64,
            accept_large_sparse= not solver in ["liblinear", "sag", "saga"],
        )
    y = check_array(y, ensure_2d=False, dtype=None)
    check_consistent_length(X, y)

    _, n_features = X.shape
    fit_intercept = False

    w0 = np.zeros(n_features + int(fit_intercept), dtype=X.dtype)
    mask = y == 1
    y_bin = np.ones(y.shape, dtype=X.dtype)
    y_bin[~mask] = 0.0

    if coef.size not in (n_features, w0.size):
            raise ValueError(
                "Initialization coef is of shape %d, expected shape %d or %d"
                % (coef.size, n_features, w0.size)
            )
    w0[: coef.size] = coef

    target = y_bin
    return w0, X, target, lam, epsilon

def sklearn_optimize(coef,X,y,lam,epsilon, covariance_prior = None):
    w0, X,target, lam, epsilon = preprocessing(coef, X, y, lam, epsilon)

    func = loss_gradient 

    l2_reg_strength = lam
    n_threads = 1

    if covariance_prior is None:
        covariance_prior = np.eye(X.shape[1])

    method = "L-BFGS-B"
    

    opt_res = minimize(
                func,
                w0,
                method=method,
                jac=True,
                args=(X, target, l2_reg_strength, epsilon,covariance_prior, n_threads),
                options={"maxiter": 1000, "disp": False},
            )
    
    w0, _ = opt_res.x, opt_res.fun
    return w0


def log1pexp(x):
    out = np.zeros_like(x)
    idx0 = x <= -37
    out[idx0] = np.exp(x[idx0])
    idx1 = (x > -37) & (x <= -2)
    out[idx1] = np.log1p(np.exp(x[idx1]))
    idx2 = (x > -2) & (x <= 18)
    out[idx2] = np.log(1. + np.exp(x[idx2]))
    idx3 = (x > 18) & (x <= 33.3)
    out[idx3] = x[idx3] + np.exp(-x[idx3])
    idx4 = x > 33.3
    out[idx4] = x[idx4]
    return out

def stable_loss(z,e,y):
    return -y*z + y*e + (1-y)*log1pexp(z+e) + y*log1pexp(z-e)

def stable_sigmoid(x):
    out = np.zeros_like(x)
    idx = x <= 0
    out[idx] = np.exp(x[idx]) / (1 + np.exp(x[idx]))
    idx = x > 0
    out[idx] = 1 / (1 + np.exp(-x[idx]))
    return out

def stable_gradient(z,e,y):
    opt_attack_term = (1-y)*stable_sigmoid(z+e) + y*stable_sigmoid(-z+e)
    data_term = (1-y)*stable_sigmoid(z+e) - y*stable_sigmoid(-z+e)
    return opt_attack_term, data_term





def loss_gradient(coef, X, y,l2_reg_strength, epsilon, covariance_prior, n_threads=1):
    n_features = X.shape[1]
    weights = coef
    raw_prediction = X @ weights / np.sqrt(n_features)    

    adv_factor = epsilon * np.sqrt(weights @ weights) / np.sqrt(n_features)

    loss_out = stable_loss(raw_prediction,adv_factor,y)
    epsilon_gradient_out,gradient_out = stable_gradient(raw_prediction,adv_factor,y)

    loss,grad_per_sample = loss_out,gradient_out

    loss = loss.sum()

    adv_correction_factor = epsilon * weights / ( np.sqrt(weights @ weights) * np.sqrt(n_features) ) 
    adv_grad_summand = np.outer(epsilon_gradient_out, adv_correction_factor).sum(axis=0)


    # if epsilon is zero, assert that the norm of adv_grad_summand is zero
    if epsilon == 0:
        assert np.linalg.norm(adv_grad_summand) == 0

    l2_reg_strength /= 2

    loss +=  l2_reg_strength * (weights @ covariance_prior @ weights)
    grad = np.empty_like(coef, dtype=weights.dtype)

    grad[:n_features] = X.T @ grad_per_sample / np.sqrt(n_features) +  l2_reg_strength * ( covariance_prior + covariance_prior.T) @ weights + adv_grad_summand

    return loss, grad

def training_loss(w,X,y,lam,epsilon,covariance_prior = None):
    from state_evolution import adversarial_loss
    z = X@w
    if covariance_prior is None:
        covariance_prior = np.eye(X.shape[1])
    return (adversarial_loss(y,z,epsilon/np.sqrt(X.shape[1]),w@w).sum() + 0.5 * lam * w@covariance_prior@w )/X.shape[0]

def pure_training_loss(w,X,y,epsilon):
    from state_evolution import adversarial_loss
    z = X@w/np.sqrt(X.shape[1])
    return (adversarial_loss(y,z,epsilon*np.sqrt(w@w)/np.sqrt(X.shape[1])).sum())/X.shape[0]

def stable_cosh(x):
    out = np.zeros_like(x)
    idx = x <= 0
    out[idx] = np.exp(x[idx]) / (1 + np.exp(2*x[idx]) + 2*np.exp(x[idx]))
    idx = x > 0
    out[idx] = np.exp(-x[idx]) / (1 + np.exp(-2*x[idx]) + 2*np.exp(-x[idx]))
    return out

def compute_hessian(X,y,theta,epsilon, lam, Sigma_w):
    X = X / np.sqrt(X.shape[1])
    raw_prediction = X.dot(theta)

    # B - Optimal Attack ()
    B = epsilon * np.linalg.norm(theta) / np.sqrt(X.shape[1])

    # C and C_prime (n,)
    C = raw_prediction + B
    C_prime = raw_prediction - B

    # H - Derivative of Optimal Attack (d,)
    H = epsilon * theta / (np.linalg.norm(theta) * np.sqrt(X.shape[1]))

    # dH - Hessian of Optimal Attack (d,d)
    dH = np.eye(X.shape[1]) * epsilon / (np.linalg.norm(theta) * np.sqrt(X.shape[1])) - epsilon*np.outer(theta, theta) / (np.linalg.norm(theta) ** 3 * np.sqrt(X.shape[1]))

    # dH term
    vec = (1-y) * stable_sigmoid(C) + y * stable_sigmoid(-C_prime) # (n,)
    hessian = vec.sum() * dH

    # dC term and dC_prime term
    vecC = (1-y) * stable_cosh(C) # (n,)
    vecC_prime = y * stable_cosh(C_prime) # (n,)

    # Shift X by H
    X_plus = X + H
    X_minus = X - H

    # dC term
    act = np.multiply(X_plus.T, vecC) # (d,n)
    hessian += np.einsum('ij,ik->jk', X_plus, act.T) # (d,d)

    # dC_prime term
    act = np.multiply(X_minus.T, vecC_prime) # (d,n)
    hessian += np.einsum('ij,ik->jk', X_minus, act.T) # (d,d)

    # Regularization
    hessian += lam/2 * (Sigma_w + Sigma_w.T)

    return hessian

def min_eigenvalue_hessian(X,y,theta,epsilon, lam, Sigma_w):
    hessian = compute_hessian(X,y,theta,epsilon, lam, Sigma_w)
    # return np.min(eigvalsh(hessian))
    return eigsh(hessian, k=1, which='SA')[0][0]

