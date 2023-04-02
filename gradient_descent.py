"""
This module contains code for custom gradient descent
"""
from logging import exception
from scipy.special import erfc, logsumexp
import numpy as np
import logging
import time
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
from sklearn.utils.validation import _num_samples, check_array, check_consistent_length, _check_sample_weight
import sklearn.utils.validation as sk_validation
from sklearn._loss import HalfBinomialLoss
from sklearn.linear_model._logistic import _logistic_regression_path
import inspect as i
from scipy.special import logit
from scipy.linalg import norm
import xyz as skloss

"""
sklearn
"""
def sklearn_optimize(coef,X,y,lam,epsilon):
    solver = "lbfgs"
    X = check_array(
            X,
            accept_sparse="csr",
            dtype=np.float64,
            accept_large_sparse= not solver in ["liblinear", "sag", "saga"],
        )
    y = check_array(y, ensure_2d=False, dtype=None)
    check_consistent_length(X, y)

    n_samples, n_features = X.shape
    fit_intercept = False

    w0 = np.zeros(n_features + int(fit_intercept), dtype=X.dtype)
    mask = y == 1
    y_bin = np.ones(y.shape, dtype=X.dtype)
    mask_classes = np.array([0, 1])
    y_bin[~mask] = 0.0

    if coef.size not in (n_features, w0.size):
            raise ValueError(
                "Initialization coef is of shape %d, expected shape %d or %d"
                % (coef.size, n_features, w0.size)
            )
    w0[: coef.size] = coef

    target = y_bin

    loss = LinearModelLoss(
                base_loss=HalfBinomialLoss(), fit_intercept=fit_intercept
            )
    func = loss.loss_gradient
    func = c_inspired_loss_gradient # TODO, for 

    sample_weight = None
    sample_weight = _check_sample_weight(sample_weight, X, dtype=X.dtype, copy=True)
    l2_reg_strength = lam
    n_threads = 1

    warm_start_sag = {"coef": np.expand_dims(w0, axis=1)}

    opt_res = minimize(
                func,
                w0,
                method="L-BFGS-B",
                jac=True,
                args=(X, target, epsilon, sample_weight, l2_reg_strength, n_threads)
            )
    # opt_res = minimize(
    #             func,
    #             w0,
    #             method="L-BFGS-B",
    #             jac=True,
    #             args=(X, target, sample_weight, l2_reg_strength, n_threads)
    #         )

    w0, loss = opt_res.x, opt_res.fun
    return w0

# Numerically stable version of log(1 + exp(x)) for double precision, see Eq. (10) of
# https://cran.r-project.org/web/packages/Rmpfr/vignettes/log1mexp-note.pdf
# Note: The only important cutoff is at x = 18. All others are to save computation
# time. Compared to the reference, we add the additional case distinction x <= -2 in
# order to use log instead of log1p for improved performance. As with the other
# cutoffs, this is accurate within machine precision of double.
def log1pexp(x: float) -> float:
    # test if x is a numpy array
    if isinstance(x, np.ndarray):
        # create an empty array of size x to fill
        r = np.empty_like(x)
        # create a mask for all entries where x <= -37
        less_than_minus_37 = x <= -37
        # replace entries with np.exp(x) where the mask is true
        np.putmask(r, less_than_minus_37, np.exp(x))
        # create a mask for all entries where x <= -2
        less_than_minus_2 = x <= -2
        # replace entries with np.log1p(np.exp(x)) where the mask is true
        np.putmask(r, less_than_minus_2, np.log1p(np.exp(x)))
        # create a mask for all entries where x <= 18
        less_than_18 = x <= 18
        # replace entries with np.log(1. + np.exp(x)) where the mask is true
        np.putmask(r, less_than_18, np.log(1. + np.exp(x)))
        # create a mask for all entries where x <= 33.3
        less_than_33_3 = x <= 33.3
        # replace entries with x + np.exp(-x) where the mask is true
        np.putmask(r, less_than_33_3, x + np.exp(-x))
        # replace all other entries with x
        np.putmask(r, ~less_than_minus_37 & ~less_than_minus_2 & ~less_than_18 & ~less_than_33_3, x)
        return r
    else:
        if x <= -37:
            return np.exp(x)
        elif x <= -2:
            return np.log1p(np.exp(x)) # according to the above paper, log1p is part of the c language standard.
        elif x <= 18:
            return np.log(1. + np.exp(x))
        elif x <= 33.3:
            return x + np.exp(-x)
        else:
            return x


def _w_intercept_raw(coef, X):
    fit_intercept = False
    is_multiclass = False
    n_classes = 1
    if not is_multiclass:
        if fit_intercept:
            intercept = coef[-1]
            weights = coef[:-1]
        else:
            intercept = 0.0
            weights = coef
        raw_prediction = X @ weights + intercept
    else:
        # reshape to (n_classes, n_dof)
        if coef.ndim == 1:
            weights = coef.reshape((n_classes, -1), order="F")
        else:
            weights = coef
        if fit_intercept:
            intercept = weights[:, -1]
            weights = weights[:, :-1]
        else:
            intercept = 0.0
        raw_prediction = X @ weights.T + intercept  # ndarray, likely C-contiguous

    return weights, intercept, raw_prediction

def c_inspired_loss_gradient(coef, X, y, epsilon, sample_weight=None, l2_reg_strength=0.0, n_threads=1):
    n_features, n_classes = X.shape[1], 1
    fit_intercept = False
    n_dof = n_features + int(fit_intercept)
    weights, intercept, raw_prediction = _w_intercept_raw(coef, X)

    # loss, grad_per_sample = just_loss_gradient(
    #     y_true=y,
    #     raw_prediction=raw_prediction
    # )
    half = skloss.CyHalfBinomialLoss()

    loss_out = None
    gradient_out = None
    adversarial_gradient_out = None
    if loss_out is None:
        if gradient_out is None:
            loss_out = np.empty_like(y)
            gradient_out = np.empty_like(raw_prediction)
        else:
            loss_out = np.empty_like(y, dtype=gradient_out.dtype)
    elif gradient_out is None:
        gradient_out = np.empty_like(raw_prediction, dtype=loss_out.dtype)
    if adversarial_gradient_out is None:
        adversarial_gradient_out = np.empty_like(raw_prediction, dtype=gradient_out.dtype)

    # Be graceful to shape (n_samples, 1) -> (n_samples,)
    if raw_prediction.ndim == 2 and raw_prediction.shape[1] == 1:
        raw_prediction = raw_prediction.squeeze(1)
    if gradient_out.ndim == 2 and gradient_out.shape[1] == 1:
        gradient_out = gradient_out.squeeze(1)
    if adversarial_gradient_out.ndim == 2 and adversarial_gradient_out.shape[1] == 1:
        adversarial_gradient_out = adversarial_gradient_out.squeeze(1)

    half.loss_gradient( y_true=y,
        raw_prediction=raw_prediction,
        adversarial_norm = epsilon * np.sqrt(weights @ weights),
        sample_weight=sample_weight,
        loss_out=loss_out,
        gradient_out=gradient_out,
        adversarial_gradient_out = adversarial_gradient_out,   
        n_threads=n_threads,
    )
    loss,grad_per_sample, adv_grad_per_sample = loss_out,gradient_out, adversarial_gradient_out


    loss = loss.sum()

    adv_correction_factor = epsilon * weights / np.sqrt(weights @ weights)
    adv_grad_summand = np.outer(adv_grad_per_sample, adv_correction_factor).sum(axis=0)


    # if epsilon is zero, assert that the norm of adv_grad_summand is zero
    if epsilon == 0:
        assert np.linalg.norm(adv_grad_summand) == 0

    loss += 0.5 * l2_reg_strength * (weights @ weights)
    grad = np.empty_like(coef, dtype=weights.dtype)
    grad[:n_features] = X.T @ grad_per_sample + l2_reg_strength * weights + adv_grad_summand
    if fit_intercept:
        grad[-1] = grad_per_sample.sum()

    return loss, grad

def just_loss_gradient(y_true, raw_prediction):
    return half_binomial_loss(y_true, raw_prediction), half_binomial_loss_gradient(y_true, raw_prediction)

def half_binomial_loss(y_true, raw_predictions):
    return log1pexp(raw_predictions) - y_true * raw_predictions

def half_binomial_loss_gradient(y_true, raw_predictions):
    exp_tmp = np.exp(-raw_predictions)
    return ((1 - y_true) - y_true * exp_tmp) / (1 + exp_tmp)



def loss_gradient(w,X,y,lam,epsilon):
    # print("loss_gradient")
    d = w.shape[0]
    loss = total_loss(w,X,y,lam,epsilon,d)
    gradient = total_gradient(w,X,y,lam,epsilon,d)
    # print("loss",loss, "gradient", norm(gradient,2))
    return loss, gradient

def total_loss(w, X, y, lam, epsilon, d):
    # loss(x_i) = log(1 + exp(raw_pred_i)) - y_true_i * raw_pred_i from sklearn document
    raw_predictions = X@w
    loss = np.log(1 + np.exp(raw_predictions)) - y * raw_predictions
    n = X.shape[0]
    loss = loss.sum() /n
    loss += 0.5 * lam * (w @ w) / np.sqrt(d)
    return loss

    # raw_predictions = X@w
    # l2_norm_w = norm(w,2)
    # loss = -y*raw_predictions + y * epsilon * l2_norm_w 
    # log1 = np.empty_like(raw_predictions, dtype=raw_predictions.dtype)
    # log1.fill(np.log(1))
    # loss += (1-y) * logsumexp(np.array([log1,raw_predictions + epsilon * l2_norm_w]),axis=0)
    # loss += y * logsumexp(np.array([log1,raw_predictions - epsilon * l2_norm_w]),axis=0)
    # loss = loss.sum()
    # loss += 0.5 * lam * (w @ w)    
    # return loss

def total_gradient(w, X, y, lam, epsilon, d):
    raw_predictions = X@w
    gradient = 1/(1+np.exp(-raw_predictions))*X.T - y*X.T
    return gradient.sum(axis=1)/X.shape[0] + lam * w/np.sqrt(d)

    # pointwise_gradient = gradient_pointwise(w,X,y,epsilon,d)
    # epsilon_pointwise = gradient_epsilon_pointwise(w,X,y,epsilon,d)
    # total_gradient = X.T @ pointwise_gradient
    # eps_factor =  epsilon
    # eps_factor *= w / norm(w,2)
    # epsilon_part = np.outer(eps_factor, epsilon_pointwise)
    # epsilon_part = epsilon_part.sum(axis=1)
    # n = X.shape[0]
    # return (total_gradient + epsilon_part) /n + lam * w

def gradient_pointwise(w, X, y, epsilon, d):
    result = -y
    raw_predictions = X@w
    l2_norm_w = norm(w,2)
    result += (1-y) * (1/(1+np.exp(-raw_predictions - epsilon * l2_norm_w)))
    result += y * (1/(1+np.exp(-raw_predictions + epsilon * l2_norm_w)))
    return result

def gradient_epsilon_pointwise(w, X, y, epsilon, d):
    result = y
    raw_predictions = X@w
    l2_norm_w = norm(w,2)
    result += (1-y) * (1/(1+np.exp(-raw_predictions - epsilon * l2_norm_w)))
    result -= y * (1/(1+np.exp(-raw_predictions + epsilon * l2_norm_w)))
    return result

# def total_loss(w,X,y,lam,epsilon):
#     d = X.shape[1]
#     loss = loss_per_sample(y,X@w,epsilon,w)
#     loss = loss.sum()
#     d = X.shape[1]
#     loss += 0.5 * lam * (w @ w) / np.sqrt(d)
#     return loss

def loss_per_sample(y,raw_prediction, epsilon, w): # TODO is bein used in ERMInformation... fix thids..
    d = w.shape[0]
    raw_prediction = -y*raw_prediction + epsilon * norm(w,2)
    # create a vector of np.log(1) of the same size as raw_prediction
    log1 = np.empty_like(raw_prediction, dtype=raw_prediction.dtype)
    log1.fill(np.log(1))
    return logsumexp(np.array([log1,raw_prediction]),axis=1)

# def total_gradient(w,X,y,lam,epsilon):
#     grad = np.empty_like(w, dtype=w.dtype)
#     grad_per_sample,epsilon_part = gradient_per_sample(w,X,y,epsilon)
#     grad = grad_per_sample.T @ X  
#     d = X.shape[1]
#     grad += epsilon_part    
#     grad += lam * w / np.sqrt(d)
#     return grad

# def gradient_per_sample(w,X,y,epsilon):
#     d = X.shape[1]
#     p = y*(X@w) - epsilon * norm(w,2) / (d**0.25)
#     b = 1/(1+np.exp(p))
#     c = epsilon*w/(norm(w,2) * (d**0.25))
#     d = np.outer(b,c).sum(axis=0)
#     return -y*b, d


def lbfgs(X,y,lam,epsilon,logger, method="L-BFGS-B"):
    # change y to be in {0,1}
    y = (y+1)/2

    n,d = X.shape
    res = minimize(loss_gradient,sample_weights(d),args=(X, y, lam, epsilon),jac=True,method=method,options={'maxiter':100}) 
    logger.info(f"Minimized {res.success} {res.message}")
    if not res.success:
        logger.info(f"OPTIMIZATION FAILED: {res.message} {res.status}")
    w = res.x
    return w

def gd(X,y,lam,epsilon, logger, debug = False):
    # change y to be in {0,1}
    y = (y+1)/2

    n,d = X.shape
    w0 = sample_weights(d)
    wt = sample_weights(d)

    n_iter = 0 
    gradient_norm = 1
    loss_difference = 1
    training_error = 1
    learning_rate = 1000000

    last_loss = total_loss(w0,X,y,lam,epsilon,d)

    while gradient_norm > 10**-9 and loss_difference > 10**-9 and training_error != 0 and n_iter < 10000:
        if debug:
            print("iteration: ",n_iter," loss: ",last_loss,"gradient_norm",gradient_norm,"learning_rate",learning_rate,"epsilon",epsilon,"lam",lam)

        
        g = total_gradient(w0,X,y,lam,epsilon,d)
        wt = w0 - learning_rate *g
        wp = w0 + learning_rate *g
        gradient_norm = norm(g,2)

        if debug:
            print_loss(w0,total_loss,g,learning_rate,lam,X,y,epsilon)


        new_loss = total_loss(wt,X,y,lam,epsilon,d)
        new_loss_p = total_loss(wp,X,y,lam,epsilon,d)
        if new_loss > last_loss:
            if new_loss_p > last_loss:
                learning_rate *= 0.4
                continue
            else:
                wt = wp
                new_loss = new_loss_p
        loss_difference = last_loss - new_loss
        last_loss = new_loss
        

        w0 = wt
        training_error = 0.25*np.mean((y-np.sign(X@wt))**2)
        n_iter += 1

    logger.debug(f"GD converged after {n_iter} iterations loss: {last_loss} gradient_norm {gradient_norm} learning_rate {learning_rate} epsilon {epsilon} lam {lam} alpha {n/d}")
    return wt



def print_loss(w0,loss_fct,gradient,learning_rate,lam,X,y,epsilon):
    ts = np.linspace(-learning_rate,learning_rate,100)

    loss_gradient = []
    d = X.shape[1]
    
    for t in ts:
        loss_gradient.append(loss_fct(w0+t*gradient,X,y,lam,epsilon,d))
    
    fig,ax = plt.subplots()
    plt_loss_over_time, = ax.plot(ts,loss_gradient,label="loss")
    # ax.scatter(0,loss(wt,X,y,lam,epsilon),label="loss at solution")
    ax.legend(handles=[plt_loss_over_time])
    plt.title("loss")
    plt.xlabel("gradient_direction")
    plt.ylabel("loss")
    plt.show()


if __name__ == "__main__":
    # TODO: try coding this stuff in torch

    logging.basicConfig(level=logging.DEBUG)

    alpha = 5
    d = 1000
    w = sample_weights(d)
    # method = "L-BFGS-B"
    method = "sklearn"
    # method = "gd"
    tau = 0 # stuff works fine with high enough noise level?
    epsilon = 0
    lam = 1

    #TODO: for alpha 0.2, d 1000, tau 2, epsilon 0, lam 1e-4 things don't work quite as well...
    # maybe it's worth coding stuff up in c anyway...

    start = time.time()
    print("Starting experiment with alpha = ",alpha," d = ",d," method = ",method," tau = ",tau," lam = ",lam," epsilon = ",epsilon)

    # generate data
    Xtrain, y = sample_training_data(w,d,int(alpha * d),tau)
    n_test = 100000
    Xtest,ytest = sample_training_data(w,d,n_test,tau)



    w_gd = np.empty(w.shape,dtype=w.dtype)
    if method == "gd":
        w_gd = gd(Xtrain,y,lam,epsilon,logging, debug=False)
    elif method == "sklearn":
        w_gd = sklearn_optimize(sample_weights(d),Xtrain,y,lam,epsilon)
        print(w_gd.shape)
    elif method == "L-BFGS-B":
        w_gd = lbfgs(Xtrain,y,lam,epsilon,logging)

    # compare to LogisticRegression
    clf = LogisticRegression(random_state=0, solver='lbfgs',max_iter=1000,C=1/lam).fit(Xtrain, y)
    w_lr = clf.coef_.flatten()
    # evaluate the norm of the coefficients both ways
    print("norm(w_gd,2)",norm(w_gd,2))
    print("norm(w_lr,2)",norm(w_lr,2))
    # compute the angle between the coefficients
    print("angle between coefficients",np.arccos(w_gd@w_lr/(norm(w_gd,2)*norm(w_lr,2))))
    # evaluate the difference between the coefficients both ways
    print("norm(w_gd-w_lr,2)",norm(w_gd-w_lr,2))
    # evaluate total loss both ways
    print("total_loss(w_gd,Xtrain,y,lam,epsilon)",total_loss(w_gd,Xtrain,y,lam,epsilon,d))
    print("total_loss(w_lr,Xtrain,y,lam,epsilon)",total_loss(w_lr,Xtrain,y,lam,epsilon,d))
    # evaluate the gradient norm both ways
    print("norm(total_gradient(w_gd,Xtrain,y,lam,epsilon),2)",norm(total_gradient(w_gd,Xtrain,y,lam,epsilon,d),2))
    print("norm(total_gradient(w_lr,Xtrain,y,lam,epsilon),2)",norm(total_gradient(w_lr,Xtrain,y,lam,epsilon,d),2))

    end = time.time()
    duration = end - start

    from experiment_information import ERMExperimentInformation
    erm_information = ERMExperimentInformation("my_erm_minimizer_tests",duration,Xtest,w_gd,tau,y,Xtrain,w,ytest,d,method,epsilon,lam)
    print("erm_information.generalization_error_erm, ", erm_information.generalization_error_erm)
    print("erm_information.generalization_error_overlap, ", erm_information.generalization_error_erm)
    
    # obtain experiment information for w_lr
    erm_information_lr = ERMExperimentInformation("my_erm_minimizer_tests",duration,Xtest,w_lr,tau,y,Xtrain,w,ytest,d,method,epsilon,lam)
    print("erm_information_lr.generalization_error_erm, ", erm_information_lr.generalization_error_erm)
    print("erm_information_lr.generalization_error_overlap, ", erm_information_lr.generalization_error_erm)

    # print dtypes of the weights, Xtrain, y
    print("w_gd.dtype",w_gd.dtype)
    print("w_lr.dtype",w_lr.dtype)
    print("Xtrain.dtype",Xtrain.dtype)
    print("y.dtype",y.dtype)

    """
    # they change the labels to if solver in ["lbfgs", "newton-cg", "newton-cholesky"]:
            # HalfBinomialLoss, used for those solvers, represents y in [0, 1] instead
            # of in [-1, 1].
            mask_classes = np.array([0, 1])
            y_bin[~mask] = 0.0
    
    # they use different weight matrix
    if n_classes == 1:
                w0[0, : coef.shape[1]] = -coef
                w0[1, : coef.shape[1]] = coef
                (1,d)

    # they use different loss function
    loss = LinearModelLoss(
                base_loss=HalfBinomialLoss(), fit_intercept=fit_intercept
            )
            func = loss.loss_gradient
    
    # check out the half binomial loss
    # and how this changes the linermodelloss
    # and how this results in the loss.loss_gradient function

    warm_start_sag = {"coef": np.expand_dims(w0, axis=1)} # is this important?

    opt_res = optimize.minimize(
                func,
                w0,
                method="L-BFGS-B",
                jac=True,
                args=(X, target, sample_weight, l2_reg_strength, n_threads),
                options={"iprint": iprint, "gtol": tol, "maxiter": max_iter},
            )
    
    # what exactly is this doing?
    n_iter_i = _check_optimize_result(
                solver,
                opt_res,
                max_iter,
                extra_warning_msg=_LOGISTIC_SOLVER_CONVERGENCE_MSG,
            )
    """