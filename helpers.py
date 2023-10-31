import numpy as np
from scipy.special import erfc
from data_model import DataModelType
import os


"""
------------------------------------------------------------------------------------------------------------------------
    Matrix Generation
------------------------------------------------------------------------------------------------------------------------
"""

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

"""
------------------------------------------------------------------------------------------------------------------------
    Numerics
------------------------------------------------------------------------------------------------------------------------
"""

def stable_cosh(x):
    out = np.zeros_like(x)
    idx = x <= 0
    out[idx] = np.exp(x[idx]) / (1 + np.exp(2*x[idx]))
    idx = x > 0
    out[idx] = np.exp(-x[idx]) / (1 + np.exp(-2*x[idx]))
    return out

def stable_cosh_squared(x):
    out = np.zeros_like(x)
    idx = x <= 0
    out[idx] = np.exp(x[idx]) / (1 + np.exp(2*x[idx]) + 2*np.exp(x[idx]))
    idx = x > 0
    out[idx] = np.exp(-x[idx]) / (1 + np.exp(-2*x[idx]) + 2*np.exp(-x[idx]))
    return out

def sigmoid(x):
    out = np.zeros_like(x)
    idx = x <= 0
    out[idx] = np.exp(x[idx]) / (1 + np.exp(x[idx]))
    idx = x > 0
    out[idx] = 1 / (1 + np.exp(-x[idx]))
    return out

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

"""
------------------------------------------------------------------------------------------------------------------------
    Losses and Activations
------------------------------------------------------------------------------------------------------------------------
"""

def sigma_star(x):
    """
    Returns 0.5 * erfc(-x/sqrt(2))
    """
    return 0.5 * erfc(-x/np.sqrt(2))


def adversarial_loss(y,z, epsilon_term):
    return log1pexp(-y*z + epsilon_term)

def second_derivative_loss(y: float, z: float, epsilon_term: float) -> float:
    return y**2 * stable_cosh(0.5*y*z - 0.5*epsilon_term)**(2)

def gaussian(x : float, mean : float = 0, var : float = 1) -> float:
    '''
    Gaussian measure
    '''
    return np.exp(-.5 * (x-mean)**2 / var)/np.sqrt(2*np.pi*var)
    
"""
------------------------------------------------------------------------------------------------------------------------
    Helper Classes
------------------------------------------------------------------------------------------------------------------------
"""



class Task:
    def __init__(self, id, experiment_id, method, alpha, epsilon, test_against_epsilon, lam, tau,d,ps, dp, data_model_type: DataModelType, compute_hessian: bool = False):
        self.id = id
        self.experiment_id = experiment_id
        self.method = method
        self.alpha = alpha
        self.epsilon = epsilon
        self.test_against_epsilon = test_against_epsilon
        self.lam = lam
        self.tau = tau
        self.d = d
        self.gamma = 1
        self.result = None
        self.ps = ps
        self.dp = dp
        self.data_model_type: DataModelType = data_model_type
        self.compute_hessian: bool = compute_hessian

    def __str__(self):
        return f"Task {self.id} with method {self.method} and alpha={self.alpha}, epsilon={self.epsilon}, test_against_epsilon={self.test_against_epsilon}, lambda={self.lam}, tau={self.tau}, d={self.d}, and data model {self.data_model_type.name}"


"""
------------------------------------------------------------------------------------------------------------------------
    Optimal Lambda Helpers
------------------------------------------------------------------------------------------------------------------------
"""

class OptimalLambdaResult():
    def __init__(self, alpha, epsilon, tau, optimal_lambda, data_model_type, data_model_name):
        self.alpha = alpha
        self.epsilon = epsilon
        self.tau = tau
        self.optimal_lambda = optimal_lambda
        self.data_model_type = data_model_type
        self.data_model_name = data_model_name

    def to_csv_line(self):
        # round all the results to 8 digits
        return f"{self.alpha:.8f},{self.epsilon:.8f},{self.tau:.8f},{self.optimal_lambda:.8f},{self.data_model_type.name},{self.data_model_name}"
    
    def from_csv_line(self, line):
        # remove the line break
        line = line[:-1]
        alpha,epsilon,tau,lam,data_model_type,data_model_name = line.split(",")
        return OptimalLambdaResult(float(alpha),float(epsilon),float(tau),float(lam),DataModelType[data_model_type],data_model_name)
    
    def get_csv_header(self):
        return "alpha,epsilon,tau,lambda,data_model_type,data_model_name"

    def get_csv_filename(self):
        return "optimal_lambdas.csv"
    
    def get_key(self):
        # the key of a result is the tuple (alpha,epsilon,tau, data_model_type, data_model_name) returned as a string
        # round all the results to 8 digits
        return f"{self.alpha:.8f},{self.epsilon:.8f},{self.tau:.8f},{self.data_model_type.name},{self.data_model_name}"

    def get_target(self):
        return self.optimal_lambda

class OptimalAdversarialLambdaResult():
    def __init__(self, alpha, epsilon, test_epsilon, tau, optimal_lambda, data_model_type, data_model_name):
        self.alpha = alpha
        self.epsilon = epsilon
        self.test_epsilon = test_epsilon
        self.tau = tau
        self.optimal_lambda = optimal_lambda
        self.data_model_type = data_model_type
        self.data_model_name = data_model_name

    def to_csv_line(self):
        # round all the results to 8 digits
        return f"{self.alpha:.8f},{self.epsilon:.8f},{self.test_epsilon:.8f},{self.tau:.8f},{self.optimal_lambda:.8f},{self.data_model_type.name},{self.data_model_name}"
    
    def from_csv_line(self, line):
        # remove the line break
        line = line[:-1]
        alpha,epsilon,test_epsilon,tau,lam,data_model_type,data_model_name = line.split(",")
        return OptimalAdversarialLambdaResult(float(alpha),float(epsilon), float(test_epsilon),float(tau),float(lam),DataModelType[data_model_type],data_model_name)
    
    def get_csv_header(self):
        return "alpha,epsilon,test_epsilon,tau,lambda,data_model_type,data_model_name"

    def get_csv_filename(self):
        return "optimal_adversarial_lambdas.csv"
    
    def get_key(self):
        # the key of a result is the tuple (alpha,epsilon,tau, data_model_type, data_model_name) returned as a string
        # round all the results to 8 digits
        return f"{self.alpha:.8f},{self.epsilon:.8f},{self.test_epsilon:.8f},{self.tau:.8f},{self.data_model_type.name},{self.data_model_name}"

    def get_target(self):
        return self.optimal_lambda


class OptimalEpsilonResult():
    def __init__(self, alpha, optimal_epsilon, tau, lam, data_model_type, data_model_name):
        self.alpha = alpha
        self.optimal_epsilon = optimal_epsilon
        self.tau = tau
        self.lam = lam
        self.data_model_type = data_model_type
        self.data_model_name = data_model_name

    def to_csv_line(self):
        # round all the results to 8 digits
        return f"{self.alpha:.8f},{self.optimal_epsilon:.8f},{self.tau:.8f},{self.lam:.8f},{self.data_model_type.name},{self.data_model_name}"
    
    def from_csv_line(self, line):
        # remove the line break
        line = line[:-1]
        alpha,epsilon,tau,lam,data_model_type,data_model_name = line.split(",")
        return OptimalEpsilonResult(float(alpha),float(epsilon),float(tau),float(lam),DataModelType[data_model_type],data_model_name)
    
    def get_csv_header(self):
        return "alpha,epsilon,tau,lambda,data_model_type,data_model_name"

    def get_csv_filename(self):
        return "optimal_epsilons.csv"
    
    def get_key(self):
        # the key of a result is the tuple (alpha,epsilon,tau, data_model_type, data_model_name) returned as a string
        # round all the results to 8 digits
        return f"{self.alpha:.8f},{self.lam:.8f},{self.tau:.8f},{self.data_model_type.name},{self.data_model_name}"

    def get_target(self):
        return self.optimal_epsilon

"""
------------------------------------------------------------------------------------------------------------------------
    CSV Helpers
------------------------------------------------------------------------------------------------------------------------
"""

def append_object_to_csv(obj):
    # if the file does not exist, create it and append a header
    if not os.path.isfile(obj.get_csv_filename()):
        with open(obj.get_csv_filename(),"w") as f:
            f.write(obj.get_csv_header() + "\n")
    with open(obj.get_csv_filename(),"a") as f:
        f.write(obj.to_csv_line() + "\n")

def load_csv_to_object_dictionary(obj, path = ""):
    filename = path + obj.get_csv_filename()
    # if the file does not exist, return an empty dictionary
    if not os.path.isfile(filename):
        return {}
    with open(filename,"r") as f:
        lines = f.readlines()
        # remove the header
        lines = lines[1:]
        # create a dictionary
        dictionary = {}
        for line in lines:
            # create the object
            obj = obj.from_csv_line(line)
            # add it to the dictionary
            dictionary[obj.get_key()] = obj.get_target()
        return dictionary 
    


