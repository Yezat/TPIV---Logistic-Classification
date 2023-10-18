"""
This module contains code for custom gradient descent
"""
import numpy as np
import logging
import time
import numpy as np
from erm import *
from data import *
from scipy.optimize import minimize
from sklearn.utils.validation import check_array, check_consistent_length, _check_sample_weight
from scipy.linalg import norm
import adversarial_loss_gradient as skloss
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

    n_samples, n_features = X.shape
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

    # if epsilon > 1 and lam >= 1:
    #     method = "Newton-CG"
    # else:
    #     method = "L-BFGS-B"
    method = "L-BFGS-B"
    

    opt_res = minimize(
                func,
                w0,
                method=method,
                jac=True,
                args=(X, target, l2_reg_strength, epsilon,covariance_prior, n_threads),
                options={"maxiter": 1000, "disp": False},
            )
    
    w0, loss = opt_res.x, opt_res.fun
    return w0




def mpmath_loss_gradient(y_true, raw_prediction, adversarial_norm):
    """
    Computes the loss and the gradient of the loss function.
    Parameters
    ----------
    y_true : ndarray, shape (n_samples,)
        The true labels.
    raw_prediction : ndarray, shape (n_samples,)
        The raw predictions.
    adversarial_norm : float
        The strength of the adversarial perturbation.
    Returns
    -------
    loss : float
        The loss.
    gradient : ndarray, shape (n_features,)
        The gradient of the loss.
    """
    y = y_true
    z = raw_prediction
    e = adversarial_norm




    e = mpmath.mpf(e)

    losses = []
    gradients = []
    epsilon_gradients = []

    for idx, el in enumerate(zip(y,z)):
        label = mpmath.mpf(el[0])
        raw_prediction = mpmath.mpf(el[1])


        loss = -label*raw_prediction + label*e + (1-label)*mpmath.log(1+mpmath.exp(raw_prediction+e)) + label*mpmath.log(1+mpmath.exp(raw_prediction-e))
        C = raw_prediction + adversarial_norm
        C_prime = raw_prediction - adversarial_norm    
        gradient = (1-label) / ( 1 + mpmath.exp(-C) ) - label/(1 + mpmath.exp(C_prime))
        epsilon_gradient = (1-label) / ( 1 + mpmath.exp(-C) ) + label/(1 + mpmath.exp(C_prime))

        # convert the mpmath mpf back to numpy float64
        loss = np.float64(loss)
        gradient = np.float64(gradient)
        epsilon_gradient = np.float64(epsilon_gradient)

        losses.append(loss)
        gradients.append(gradient)
        epsilon_gradients.append(epsilon_gradient)

    # convert to numpy arrays
    losses = np.array(losses)
    gradients = np.array(gradients)
    epsilon_gradients = np.array(epsilon_gradients)

    return losses, gradients, epsilon_gradients


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
    half = skloss.CyHalfBinomialLoss()
    # half = skloss_original.CyHalfBinomialLoss()
    

    loss_out = np.empty_like(y)
    gradient_out = np.empty_like(raw_prediction)
    epsilon_gradient_out = np.empty_like(raw_prediction)

    # Be graceful to shape (n_samples, 1) -> (n_samples,)
    if raw_prediction.ndim == 2 and raw_prediction.shape[1] == 1:
        raw_prediction = raw_prediction.squeeze(1)
    if gradient_out.ndim == 2 and gradient_out.shape[1] == 1:
        gradient_out = gradient_out.squeeze(1)
        epsilon_gradient_out = epsilon_gradient_out.squeeze(1)        

    # half.loss_gradient( y_true=y,
    #     raw_prediction=raw_prediction,    
    #     adversarial_norm = epsilon * np.sqrt(weights @ weights) / np.sqrt(n_features),
    #     # sample_weight=sample_weight,
    #     loss_out=loss_out,
    #     gradient_out=gradient_out,
    #     epsilon_gradient_out = epsilon_gradient_out,
    #     n_threads=n_threads,
    # )   

    # skloss.mp_loss_gradient( y_true=y,
    #     raw_prediction=raw_prediction,    
    #     adversarial_norm = epsilon * np.sqrt(weights @ weights) / np.sqrt(n_features),
    #     # sample_weight=sample_weight,
    #     loss_out=loss_out,
    #     gradient_out=gradient_out,
    #     epsilon_gradient_out = epsilon_gradient_out
    # )   

    # loss_out,gradient_out,epsilon_gradient_out = mpmath_loss_gradient(y,raw_prediction,epsilon * np.sqrt(weights @ weights) / np.sqrt(n_features))

    loss_out = stable_loss(raw_prediction,epsilon * np.sqrt(weights @ weights) / np.sqrt(n_features),y)
    epsilon_gradient_out,gradient_out = stable_gradient(raw_prediction,epsilon * np.sqrt(weights @ weights) / np.sqrt(n_features),y)

    loss,grad_per_sample = loss_out,gradient_out

    loss = loss.sum()

    adv_correction_factor = epsilon * weights / ( np.sqrt(weights @ weights) * np.sqrt(n_features)) 
    adv_grad_summand = np.outer(epsilon_gradient_out, adv_correction_factor).sum(axis=0)


    # if epsilon is zero, assert that the norm of adv_grad_summand is zero
    if epsilon == 0:
        assert np.linalg.norm(adv_grad_summand) == 0

    l2_reg_strength = l2_reg_strength / 2

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
    z = X@w
    return (adversarial_loss(y,z,epsilon,np.sqrt(w@w)).sum())/X.shape[0]


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    alpha = 5
    d = 1000
    w = sample_weights(d)
    method = "L-BFGS-B"
    method = "sklearn"
    # method = "gd"
    tau = 1
    epsilon = 1.9
    # epsilon = 1.34
    lam = 1

    start = time.time()
    print("Starting experiment with alpha = ",alpha," d = ",d," method = ",method," tau = ",tau," lam = ",lam," epsilon = ",epsilon)

    # # generate data
    Xtrain, y = sample_training_data(w,d,int(alpha * d),tau)
    n_test = 100000
    Xtest,ytest = sample_training_data(w,d,n_test,tau)

    # print("w",w)
    # result = "{"
    # for i in range(int(alpha*d)):
    #     result += "{" + str(Xtrain[i][0]) + "," + str(Xtrain[i][1]) + "," + str(y[i]) + "}"
    # result += "}"
    # print(result)



    w_gd = np.empty(w.shape,dtype=w.dtype)
    w_gd = sklearn_optimize(sample_weights(d),Xtrain,y,lam,epsilon)
    print(w_gd.shape)


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
    print("total_loss(w_gd,Xtrain,y,lam,epsilon)",loss_gradient(w_gd,Xtrain,y,lam,epsilon)[0])
    print("total_loss(w_lr,Xtrain,y,lam,epsilon)",loss_gradient(w_lr,Xtrain,y,lam,epsilon)[0])
    # evaluate the gradient norm both ways
    print("norm(total_gradient(w_gd,Xtrain,y,lam,epsilon),2)",norm(loss_gradient(w_gd,Xtrain,y,lam,epsilon)[1],2))
    print("norm(total_gradient(w_lr,Xtrain,y,lam,epsilon),2)",norm(loss_gradient(w_lr,Xtrain,y,lam,epsilon)[1],2))

    end = time.time()
    duration = end - start

    print("duration",duration)

    from experiment_information import ERMExperimentInformation
    erm_information = ERMExperimentInformation("my_erm_minimizer_tests",duration,Xtest,w_gd,tau,y,Xtrain,w,ytest,d,method,epsilon,lam,None,None)
    print("erm_information.generalization_error_erm, ", erm_information.generalization_error_erm)
    print("erm_information.generalization_error_overlap, ", erm_information.generalization_error_erm)
    
    # obtain experiment information for w_lr
    erm_information_lr = ERMExperimentInformation("my_erm_minimizer_tests",duration,Xtest,w_lr,tau,y,Xtrain,w,ytest,d,method,epsilon,lam,None,None)
    print("erm_information_lr.generalization_error_erm, ", erm_information_lr.generalization_error_erm)
    print("erm_information_lr.generalization_error_overlap, ", erm_information_lr.generalization_error_erm)

    # let's compute the training loss for both
    print("training_loss(w_gd,Xtrain,y,lam,epsilon)",pure_training_loss(w_gd,Xtrain,y,lam,epsilon))
    print("training_loss(w_lr,Xtrain,y,lam,epsilon)",pure_training_loss(w_lr,Xtrain,y,lam,epsilon))

    # print the overlaps
    print("Q ", erm_information.Q)
    print("Q lr", erm_information_lr.Q)
    print("m", erm_information.m)
    print("m lr", erm_information_lr.m)