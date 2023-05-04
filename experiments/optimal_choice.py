"""
Compute the optimal choice of lambda given a setting including epsilon, alpha, d, tau etc...
"""
import os
import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 
from calibration import *
from scipy.optimize import minimize_scalar
from sweep import run_state_evolution
import logging

def minimize_lambda(alpha,epsilon,tau,tol, logger):
    """
    Compute the optimal lambda given a setting including epsilon, alpha, d, tau etc...

    parameters:
    alpha: alpha - sampling ratio
    epsilon: adversarial parameter
    tau: noise level
    """
    logger.info(f"Computing optimal lambda {alpha} {epsilon} {tau}")
    
    res = minimize_scalar(lambda l : minimizer_function(l,alpha, epsilon,tau, logger),method="bounded", bounds=[1e-3,1e2],options={'xatol': tol,'maxiter':100})
    logger.info(f"Minimized success: {res.success} message {res.message}")
    if not res.success:
        raise Exception("Optimization of lambda failed " + str(res.message))
    return res.x

def minimizer_function(lam,alpha,epsilon,tau, logger):
    # use the state evolution to compute the generalization error

    info = run_state_evolution(logger,"anyid", alpha,epsilon,lam,tau,None,None)

    return info.generalization_error



if __name__ == "__main__":
    # intitialize the logger
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # read the setting from the command line
    try:
        alpha = float(sys.argv[1])
        epsilon = float(sys.argv[2])
        tau = float(sys.argv[3])
    except:
        logger.info("Usage: optimal_choice.py alpha epsilon tau")
        logger.info("Default values if no parameters provided: alpha = 5, epsilon = 0.5, tau = 1")

    if alpha is None:
        alpha = 5
    if epsilon is None:
        epsilon = 0.5
    if tau is None:
        tau = 1
    
    # compute the optimal lambda
    lam = minimize_lambda(alpha,epsilon,tau,1e-4,logger)
    logger.info(f"Optimal lambda in alpha {alpha} epsilon {epsilon} tau {tau} is {lam}")

    filename = "optimal_lambdas.csv"
    # if the file does not exist, add a header
    if not os.path.isfile(filename):
        with open(filename,"w") as f:
            f.write("alpha,epsilon,tau,lambda\n")

    # append the result to the file
    with open(filename,"a") as f:
        f.write(f"{alpha},{epsilon},{tau},{lam}\n")

