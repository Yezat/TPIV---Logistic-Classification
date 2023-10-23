"""
This module contains code for obtaining the loss, the gradient and the hessian of a given problem and an ERM estimator.
"""
import numpy as np
from scipy.optimize import minimize
from sklearn.utils.validation import check_array, check_consistent_length
from scipy.sparse.linalg import eigsh
from helpers import sigmoid, log1pexp, stable_cosh_squared, adversarial_loss, Task
from scipy.special import erfc
from data_model import *


"""
------------------------------------------------------------------------------------------------------------------------
    Optimizer
------------------------------------------------------------------------------------------------------------------------
"""

def run_optimizer(task : Task, data_model: AbstractDataModel, data:DataSet, logger):
    
    w_gd = sklearn_optimize(np.random.normal(0,1,(task.d,)),data.X,data.y,task.lam,task.epsilon, data_model.Sigma_w, data_model.Sigma_delta, logger)

    return w_gd


"""
Preprocesses the data for the sklearn optimizer.
"""
def preprocessing(coef, X, y, lam, epsilon):
    # sklearn - expects labels as -1 and 1.
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

def sklearn_optimize(coef,X,y,lam,epsilon, covariance_prior = None, sigma_delta = None, logger = None):
    w0, X,target, lam, epsilon = preprocessing(coef, X, y, lam, epsilon)

    func = loss_gradient 

    l2_reg_strength = lam

    if covariance_prior is None:
        covariance_prior = np.eye(X.shape[1])
    if sigma_delta is None:
        sigma_delta = np.eye(X.shape[1])

    method = "L-BFGS-B"    

    opt_res = minimize(
                func,
                w0,
                method=method,
                jac=True,
                args=(X, target, l2_reg_strength, epsilon,covariance_prior, sigma_delta),
                options={"maxiter": 1000, "disp": False},
            )
    
    w0, _ = opt_res.x, opt_res.fun
    return w0


def loss_gradient(coef, X, y,l2_reg_strength, epsilon, covariance_prior, sigma_delta):
    n_features = X.shape[1]
    weights = coef
    raw_prediction = X @ weights / np.sqrt(n_features)    

    l2_reg_strength /= 2

    wSw = weights.dot(sigma_delta@weights)
    nww = np.sqrt(weights@weights)

    optimal_attack = epsilon/np.sqrt(n_features) *  wSw / nww 

    loss = compute_loss(raw_prediction,optimal_attack,y)
    loss = loss.sum()
    loss +=  l2_reg_strength * (weights @ covariance_prior @ weights)


    epsilon_gradient_per_sample,gradient_per_sample = compute_gradient(raw_prediction,optimal_attack,y)   

    derivative_optimal_attack = epsilon/np.sqrt(n_features) * ( 2*sigma_delta@weights / nww  - ( wSw / nww**3 ) * weights )

    adv_grad_summand = np.outer(epsilon_gradient_per_sample, derivative_optimal_attack).sum(axis=0)


    # if epsilon is zero, assert that the norm of adv_grad_summand is zero
    if epsilon == 0:
        assert np.linalg.norm(adv_grad_summand) == 0    

    
    grad = np.empty_like(coef, dtype=weights.dtype)
    grad[:n_features] = X.T @ gradient_per_sample / np.sqrt(n_features) +  l2_reg_strength * ( covariance_prior + covariance_prior.T) @ weights + adv_grad_summand

    return loss, grad

"""
------------------------------------------------------------------------------------------------------------------------
    Loss
------------------------------------------------------------------------------------------------------------------------
"""


def compute_loss(z,e,y):
    return -y*z + y*e + (1-y)*log1pexp(z+e) + y*log1pexp(z-e)

def training_loss(w,X,y,lam,epsilon,covariance_prior = None):
    z = X@w
    if covariance_prior is None:
        covariance_prior = np.eye(X.shape[1])
    return (adversarial_loss(y,z,epsilon/np.sqrt(X.shape[1]),w@w).sum() + 0.5 * lam * w@covariance_prior@w )/X.shape[0]

def pure_training_loss(w,X,y,epsilon, Sigma_delta):
    z = X@w/np.sqrt(X.shape[1])
    attack = epsilon/np.sqrt(X.shape[1]) * ( w.dot(Sigma_delta@w) / np.sqrt(w@w)  )
    return (adversarial_loss(y,z,attack).sum())/X.shape[0]


"""
------------------------------------------------------------------------------------------------------------------------
    Gradient
------------------------------------------------------------------------------------------------------------------------
"""


def compute_gradient(z,e,y):
    opt_attack_term = (1-y)*sigmoid(z+e) + y*sigmoid(-z+e)
    data_term = (1-y)*sigmoid(z+e) - y*sigmoid(-z+e)
    return opt_attack_term, data_term


"""
------------------------------------------------------------------------------------------------------------------------
    Hessian
------------------------------------------------------------------------------------------------------------------------
"""


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
    vec = (1-y) * sigmoid(C) + y * sigmoid(-C_prime) # (n,)
    hessian = vec.sum() * dH

    # dC term and dC_prime term
    vecC = (1-y) * stable_cosh_squared(C) # (n,)
    vecC_prime = y * stable_cosh_squared(C_prime) # (n,)

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


"""
------------------------------------------------------------------------------------------------------------------------
    Errors
------------------------------------------------------------------------------------------------------------------------
"""
def error(y, yhat):
    return 0.25*np.mean((y-yhat)**2)

def adversarial_error(y, Xtest, w_gd, epsilon, Sigma_delta):
    d = Xtest.shape[1]
    wSw = w_gd.dot(Sigma_delta@w_gd)
    nww = np.sqrt(w_gd@w_gd)
    y_hat = np.sign( sigmoid( Xtest@w_gd/np.sqrt(d) - y*epsilon/np.sqrt(d) * wSw/nww  ) - 0.5)

    # y_hat_prime = np.sign(-Xtest @ w_gd + y * epsilon * np.sqrt(np.linalg.norm(w_gd, 2) / d))

    # assert np.all(y_hat == y_hat_prime)

    return error(y, y_hat)


"""
------------------------------------------------------------------------------------------------------------------------
    Calibration
------------------------------------------------------------------------------------------------------------------------
"""
def compute_experimental_teacher_calibration(p, w, werm, Xtest, sigma):
    try:

        
        # size of bins where we put the probas
        n, d = Xtest.shape
        dp = 0.025
        
        def probit(lf, sigma):
            return 0.5 * erfc(- lf / np.sqrt(2 * sigma**2))
        

        Ypred = sigmoid(Xtest @ werm / np.sqrt(d))
        

        index = [i for i in range(n) if p - dp <= Ypred[i] <= p + dp]
        

        if sigma == 0:
            teacher_probabilities = np.array([np.heaviside(Xtest[i] @ w / np.sqrt(d),0.5) for i in index]) 
        else:
            teacher_probabilities = np.array([probit(Xtest[i] @ w / np.sqrt(d),sigma) for i in index]) 

        if len(teacher_probabilities) == 0:
            return p

        return p - np.mean(teacher_probabilities)
    

    except Exception as e:
        # probably a mean of empty slice... is it an exception though?
        print(e)
        return np.nan
    
"""
------------------------------------------------------------------------------------------------------------------------
    Predictions
------------------------------------------------------------------------------------------------------------------------
"""
def predict_erm(X,weights):
    return np.sign(predict_erm_probability(X,weights) - 0.5)

def predict_erm_probability(X,weights):
    argument = X@weights/np.sqrt(X.shape[1])
    
    return sigmoid(argument)