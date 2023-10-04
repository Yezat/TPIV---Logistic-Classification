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

def sklearn_optimize(coef,X,y,lam,epsilon):
    w0, X,target, lam, epsilon = preprocessing(coef, X, y, lam, epsilon)

    func = loss_gradient 

    sample_weight = None
    l2_reg_strength = lam
    n_threads = 1

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
                args=(X, target, l2_reg_strength, epsilon, sample_weight, n_threads),
                options={"maxiter": 1000, "disp": False},
            )
    
    w0, loss = opt_res.x, opt_res.fun
    return w0


def loss_gradient(coef, X, y,l2_reg_strength, epsilon, sample_weight=None, n_threads=1):
    n_features, n_classes = X.shape[1], 1
    fit_intercept = False
    weights = coef
    raw_prediction = X @ weights

    half = skloss.CyHalfBinomialLoss()
    # half = skloss_original.CyHalfBinomialLoss()

    loss_out = None
    gradient_out = None
    if loss_out is None:
        if gradient_out is None:
            loss_out = np.empty_like(y)
            gradient_out = np.empty_like(raw_prediction)
        else:
            loss_out = np.empty_like(y, dtype=gradient_out.dtype)
    elif gradient_out is None:
        gradient_out = np.empty_like(raw_prediction, dtype=loss_out.dtype)
    

    # Be graceful to shape (n_samples, 1) -> (n_samples,)
    if raw_prediction.ndim == 2 and raw_prediction.shape[1] == 1:
        raw_prediction = raw_prediction.squeeze(1)
    if gradient_out.ndim == 2 and gradient_out.shape[1] == 1:
        gradient_out = gradient_out.squeeze(1)

    half.loss_gradient( y_true=y,
        raw_prediction=raw_prediction,    
        adversarial_norm = epsilon * np.sqrt(weights @ weights) / np.sqrt(n_features),    
        # sample_weight=sample_weight,
        loss_out=loss_out,
        gradient_out=gradient_out,
        n_threads=n_threads,
    )
    
    

    loss,grad_per_sample = loss_out,gradient_out


    loss = loss.sum()

    adv_correction_factor = epsilon * weights / ( np.sqrt(weights @ weights) * np.sqrt(n_features))
    adv_grad_summand = np.outer(grad_per_sample, adv_correction_factor).sum(axis=0)


    # if epsilon is zero, assert that the norm of adv_grad_summand is zero
    if epsilon == 0:
        assert np.linalg.norm(adv_grad_summand) == 0

    loss += 0.5 * l2_reg_strength * (weights @ weights)
    grad = np.empty_like(coef, dtype=weights.dtype)
    grad[:n_features] = X.T @ grad_per_sample + l2_reg_strength * weights + adv_grad_summand
    if fit_intercept:
        grad[-1] = grad_per_sample.sum()

    return loss, grad


def pure_training_loss(w,X,y,lam,epsilon):
    from state_evolution import adversarial_loss
    z = X@w
    return (adversarial_loss(y,z,epsilon/np.sqrt(X.shape[1]),w@w).sum() + 0.5 * lam * (w@w))/X.shape[0]


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