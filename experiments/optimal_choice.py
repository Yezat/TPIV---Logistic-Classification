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
from experiments.sweep import run_state_evolution
import logging
from data_model import GaussianDataModel, FashionMNISTDataModel, DataModelType

def minimize_lambda(alpha,epsilon,tau,tol, logger, optimal_lambda_filename):
    """
    Compute the optimal lambda given a setting including epsilon, alpha, d, tau etc...

    parameters:
    alpha: alpha - sampling ratio
    epsilon: adversarial parameter
    tau: noise level
    """
    logger.info(f"Computing optimal lambda {alpha} {epsilon} {tau}")

    if optimal_lambda_filename is None:
        raise Exception("optimal_lambda_filename cannot be None")
    elif optimal_lambda_filename == "optimal_lambdas.csv":
        data_model = GaussianDataModel(100)
    elif optimal_lambda_filename == "fashion_mnist_optimal_lambdas.csv":
        data_model = FashionMNISTDataModel()
    else:
        raise Exception("Unknown optimal_lambda_filename")

    res = minimize_scalar(lambda l : minimizer_function(l,alpha, epsilon,tau, logger,data_model),method="bounded", bounds=[1e-3,1e2],options={'xatol': tol,'maxiter':100})
    logger.info(f"Minimized success: {res.success} message {res.message}")
    if not res.success:
        raise Exception("Optimization of lambda failed " + str(res.message))
    return res.x

def minimizer_function(lam,alpha,epsilon,tau, logger,data_model):
    # use the state evolution to compute the generalization error

    info = run_state_evolution(logger,"anyid", alpha,epsilon,lam,tau,None,None,data_model=data_model)

    return info.generalization_error

def get_optimal_lambda(alpha, epsilon, tau, logger, optimal_lambdas_filename="optimal_lambdas.csv"):

    knwon_lambdas = {}
    with open(optimal_lambdas_filename,"r") as f:
        lines = f.readlines()
        for line in lines[1:]:
            # remove the newline character
            line = line[:-1]
            alpha_2, epsilon_2, tau_2, lam_2 = line.split(",")
            knwon_lambdas[(alpha_2,epsilon_2,tau_2)] = lam_2


    query = (str(float(alpha)),str(float(epsilon)),str(float(tau)))
    logger.info(f"Querying optimal lambda in alpha {alpha} epsilon {epsilon} tau {tau}")
    if query in knwon_lambdas:
        logger.info(f"Skipping alpha {alpha} epsilon {epsilon} tau {tau} as we already know the optimal lambda {knwon_lambdas[query]}")
        return float(knwon_lambdas[query])

    lam = minimize_lambda(alpha,epsilon,tau,1e-4,logger,optimal_lambdas_filename)
    logger.info(f"Optimal lambda in alpha {alpha} epsilon {epsilon} tau {tau} is {lam}")

    filename = optimal_lambdas_filename
    # if the file does not exist, add a header
    if not os.path.isfile(filename):
        with open(filename,"w") as f:
            f.write("alpha,epsilon,tau,lambda\n")

    # append the result to the file
    with open(filename,"a") as f:
        f.write(f"{alpha},{epsilon},{tau},{lam}\n")
    return lam



if __name__ == "__main__":
    # intitialize the logger
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    alpha = None
    epsilon = None
    tau = None
    filename = None

    # read the setting from the command line
    try:
        alpha = float(sys.argv[1])
        epsilon = float(sys.argv[2])
        tau = float(sys.argv[3])
        filename = sys.argv[4]
    except:
        logger.info("Usage: optimal_choice.py alpha epsilon tau filename")
        logger.info("Default values if no parameters provided: alpha = 5, epsilon = 0.5, tau = 1, filename=optimal_lambdas.csv")

    if alpha is None:
        alpha = 12000/784
    if epsilon is None:
        epsilon = 0
    if tau is None:
        tau = 0
    if filename is None:
        filename = "fashion_mnist_optimal_lambdas.csv"
    
    # compute the optimal lambda
    get_optimal_lambda(alpha,epsilon,tau,logger,optimal_lambdas_filename=filename)    

