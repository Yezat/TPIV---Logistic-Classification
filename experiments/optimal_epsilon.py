"""
Compute the optimal choice of epsilon given a setting including alpha, d, tau, lam etc...
The idea is to pick the optimum in terms of the minimal integral over the calibration
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

def minimize_epsilon(alpha,lam,tau,tol, logger):
    """
    Compute the optimal epsilon given a setting including epsilon, alpha, d, tau etc...

    parameters:
    alpha: alpha - sampling ratio
    epsilon: adversarial parameter
    tau: noise level
    """
    logger.info(f"Computing optimal epsilon {alpha} {lam} {tau}")

    ps = np.linspace(0.01,0.99,1000)
    
    res = minimize_scalar(lambda e : minimizer_function(lam,alpha, e,tau, ps, logger),method="bounded", bounds=[0,10.0],options={'xatol': tol,'maxiter':200})
    logger.info(f"Minimized success: {res.success} message {res.message}")
    if not res.success:
        raise Exception("Optimization of epsilon failed " + str(res.message))
    return res.x



def minimizer_function(lam,alpha,epsilon,tau, ps, logger):
    # use the state evolution to compute the generalization error

    info = run_state_evolution(logger,"anyid", alpha,epsilon,lam,tau,None,ps, log=False)
    
    total_calibration = np.abs(np.array(info.calibrations.calibrations)).sum()

    logger.info(f"Absolute integrated calibration for epsilon {epsilon} is {total_calibration}")

    return total_calibration



if __name__ == "__main__":
    # intitialize the logger
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # read the setting from the command line
    try:
        alpha = float(sys.argv[1])
        lam = float(sys.argv[2])
        tau = float(sys.argv[3])
    except:
        logger.info("Usage: optimal_epsilon.py alpha lam tau")
        logger.info("Default values if no parameters provided: alpha = 5, lam = 0.5, tau = 1")

    if alpha is None:
        alpha = 5
    if lam is None:
        lam = 0.5
    if tau is None:
        tau = 1
    
    # compute the optimal epsilon
    epsilon = minimize_epsilon(alpha,lam,tau,1e-6,logger)
    logger.info(f"Optimal epsilon in alpha {alpha} lam {lam} tau {tau} is {epsilon}")

    filename = "optimal_epsilons.csv"
    # if the file does not exist, add a header
    if not os.path.isfile(filename):
        with open(filename,"w") as f:
            f.write("alpha,epsilon,tau,lambda\n")

    # append the result to the file
    with open(filename,"a") as f:
        f.write(f"{alpha},{epsilon},{tau},{lam}\n")

