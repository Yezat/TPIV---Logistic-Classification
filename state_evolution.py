import numpy as np
from typing import Tuple
from scipy.integrate import quad, dblquad
from scipy.special import erfc, erf, logit, owens_t
from helpers import *
from data_model import *
from scipy.optimize import root_scalar
import numba as nb
import math

"""
------------------------------------------------------------------------------------------------------------------------
    Helper Classes
------------------------------------------------------------------------------------------------------------------------
"""
class OverlapSet():
    def __init__(self) -> None:
        self.INITIAL_CONDITION = (1e-1,1e-1,1e-1,1e-1,1e-1,1e-1,1e-1)

        self.m = self.INITIAL_CONDITION[0]
        self.q = self.INITIAL_CONDITION[1]
        self.sigma = self.INITIAL_CONDITION[2]
        self.A = self.INITIAL_CONDITION[3]
        self.N = self.INITIAL_CONDITION[4]
        self.P = self.INITIAL_CONDITION[5]
        self.F = self.INITIAL_CONDITION[6]
        
        self.m_hat = 0
        self.q_hat = 0
        self.sigma_hat = 0
        self.A_hat = 0
        self.N_hat = 0
        self.F_hat = 0
        self.P_hat = 0

        self.BLEND_FPE = 0.75
        self.TOL_FPE = 1e-4
        self.MIN_ITER_FPE = 10
        self.MAX_ITER_FPE = 5000
        self.INT_LIMS = 10.0

    def log_overlaps(self, logger):
        logger.info(f"m: {self.m}, q: {self.q}, sigma: {self.sigma}, P: {self.P}, N: {self.N}, A: {self.A}, F: {self.F}")
        logger.info(f"m_hat: {self.m_hat}, q_hat: {self.q_hat}, sigma_hat: {self.sigma_hat}, P_hat: {self.P_hat}, N_hat: {self.N_hat}, A_hat: {self.A_hat}, F_hat: {self.F_hat}")

    def update_overlaps(self, m,q,sigma,A,N, P , F):
        self.n_m = damped_update(m, self.m, self.BLEND_FPE)
        self.n_q = damped_update(q, self.q, self.BLEND_FPE)
        self.n_sigma = damped_update(sigma, self.sigma, self.BLEND_FPE)
        self.n_A = damped_update(A, self.A, self.BLEND_FPE)
        self.n_N = damped_update(N, self.N, self.BLEND_FPE)
        self.n_P = damped_update(P, self.P, self.BLEND_FPE)
        self.n_F = damped_update(F, self.F, self.BLEND_FPE)

        # Compute the error
        err = max([abs(self.n_m - self.m), abs(self.n_q - self.q), abs(self.n_sigma - self.sigma), abs(self.n_A - self.A), abs(self.n_N - self.N), abs(self.n_P - self.P), abs(self.n_F - self.F)])

        # Update the overlaps
        self.m = self.n_m
        self.q = self.n_q
        self.sigma = self.n_sigma
        self.A = self.n_A
        self.N = self.n_N
        self.P = self.n_P
        self.F = self.n_F

        return err
        


"""
------------------------------------------------------------------------------------------------------------------------
    Proximals
------------------------------------------------------------------------------------------------------------------------
"""

import ctypes
# if ./brentq.so exists load it, otherwise load ../brentq.so
try:
    DSO = ctypes.CDLL('./brentq.so')
except: 
    DSO = ctypes.CDLL('../brentq.so')


# Add typing information
c_func = DSO.brentq
c_func.restype = ctypes.c_double
c_func.argtypes = [ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_int, ctypes.c_double, ctypes.c_double, ctypes.c_double]

COMPUTE_APPROXIMATE_PROXIMAL = False

@nb.njit
def evaluate_proximal(V: float, y: float, epsilon_term: float, w: float) -> float:


    if COMPUTE_APPROXIMATE_PROXIMAL:
        return w + y * V * np.exp( -y*w + epsilon_term ) / (1 + np.exp( -y*w + epsilon_term ) )

    # if epsilon_term > 0:
    #     return w + y * V * np.exp( -y*w + epsilon_term ) / (1 + np.exp( -y*w + epsilon_term ) )

    if y == 0:
        return w

    w_prime = w - epsilon_term / y
    z = c_func(-50000000, 50000000, 10e-10, 10e-10, 500, y, V, w_prime)
    return z + epsilon_term / y


"""
------------------------------------------------------------------------------------------------------------------------
    Hat Overlap Equations
------------------------------------------------------------------------------------------------------------------------
"""
@nb.njit
def _m_hat_integrand(xi: float, y: float, m: float, q: float, rho: float, tau: float, epsilon: float, P: float, N: float, sigma: float) -> float:
    e = m * m / (rho * q)
    w_0 = np.sqrt(rho*e) * xi
    V_0 = rho * (1-e)

    # z_out_0 and f_out_0 simplify together as the erfc cancels. See computation
    w = np.sqrt(q) * xi

    partial_prox =  evaluate_proximal(sigma,y,epsilon*P/np.sqrt(N),w) - w

    return partial_prox * gaussian(w_0,0,V_0+tau**2) * gaussian(xi,0,1)

def logistic_m_hat_func(overlaps: OverlapSet, rho: float, alpha: float, epsilon: float, tau:float, int_lims: float = 20.0):

    Iplus = quad(lambda xi: _m_hat_integrand(xi,1, overlaps.m, overlaps.q, rho, tau, epsilon, overlaps.P, overlaps.N, overlaps.sigma),-int_lims,int_lims,limit=500)[0]
    Iminus = quad(lambda xi: _m_hat_integrand(xi,-1, overlaps.m, overlaps.q, rho, tau, epsilon, overlaps.P, overlaps.N, overlaps.sigma),-int_lims,int_lims,limit=500)[0]
    return alpha / overlaps.sigma * (Iplus - Iminus)

@nb.njit
def _q_hat_integrand(xi: float, y: float, m: float, q: float, rho: float, tau: float, epsilon: float, P: float, N: float, sigma: float) -> float:
    e = m * m / (rho * q)
    w_0 = np.sqrt(rho*e) * xi
    V_0 = rho * (1-e)

    z_0 = math.erfc((-y * w_0) / np.sqrt(2*(tau**2 + V_0)))

    w = np.sqrt(q) * xi
    proximal = evaluate_proximal(sigma,y,epsilon*P/np.sqrt(N),w)
    partial_proximal = ( proximal - w ) ** 2

    return z_0 * (partial_proximal/ (sigma ** 2) ) * gaussian(xi,0,1)

def logistic_q_hat_func(overlaps: OverlapSet, rho: float, alpha: float, epsilon: float, tau:float, int_lims: float = 20.0, logger = None):

    Iplus = quad(lambda xi: _q_hat_integrand(xi,1, overlaps.m, overlaps.q, rho, tau, epsilon, overlaps.P, overlaps.N, overlaps.sigma),-int_lims,int_lims,limit=500)[0]
    Iminus = quad(lambda xi: _q_hat_integrand(xi,-1, overlaps.m, overlaps.q, rho, tau, epsilon, overlaps.P, overlaps.N, overlaps.sigma),-int_lims,int_lims,limit=500)[0]

    return  0.5 * alpha * (Iplus + Iminus)

"""
Derivative of f_out
"""
@nb.njit
def alternative_derivative_f_out(xi: float, y: float, m: float, q: float, sigma: float, epsilon: float, P: float, N: float) -> float:
    w = np.sqrt(q) * xi
    proximal = evaluate_proximal(sigma,y,epsilon*P/np.sqrt(N),w)

    second_derivative = numba_second_derivative_loss(y,proximal,epsilon*P/np.sqrt(N))

    return second_derivative / ( 1 + sigma * second_derivative) # can be seen from aubin (45)

@nb.njit
def _sigma_hat_integrand(xi: float, y: float, m: float, q: float, rho: float, tau: float, epsilon: float, P: float, N: float, sigma: float) -> float:
    z_0 = math.erfc(  ( (-y * m) / np.sqrt(q) * xi) / np.sqrt(2*(tau**2 + (rho - m**2/q))))

    derivative_f_out = alternative_derivative_f_out(xi,y,m,q,sigma,epsilon,P,N)

    return z_0 * ( derivative_f_out ) * gaussian(xi,0,1)
    

def logistic_sigma_hat_func(overlaps: OverlapSet, rho: float, alpha: float, epsilon: float, tau: float, int_lims: float = 20.0, logger = None):
 
    Iplus = quad(lambda xi: _sigma_hat_integrand(xi,1, overlaps.m, overlaps.q, rho, tau, epsilon, overlaps.P, overlaps.N, overlaps.sigma) , -int_lims, int_lims, limit=500)[0]
    Iminus = quad(lambda xi: _sigma_hat_integrand(xi,-1, overlaps.m, overlaps.q, rho, tau, epsilon, overlaps.P, overlaps.N, overlaps.sigma) , -int_lims, int_lims, limit=500)[0]

    return  0.5 * alpha * (Iplus + Iminus)



@nb.njit
def _P_hat_integrand(xi: float, y: float, m: float, q: float, rho: float, tau: float, epsilon: float, P: float, N: float, sigma: float) -> float:
    e = m * m / (rho * q)
    w_0 = np.sqrt(rho*e) * xi
    V_0 = rho * (1-e)

    z_0 = math.erfc((-y * w_0) / np.sqrt(2*(tau**2 + V_0)))

    w = np.sqrt(q) * xi

    z_star = evaluate_proximal(sigma,y,epsilon*P/np.sqrt(N),w)

    m_derivative = -(z_star - w)/sigma


    m_derivative *= -y*epsilon / np.sqrt(N)

    return z_0 * m_derivative * gaussian(xi,0,1)
    
def logistic_P_hat_func(overlaps: OverlapSet, rho: float, alpha: float, epsilon: float, tau:float, int_lims: float = 20.0):
    Iplus = quad(lambda xi: _P_hat_integrand(xi,1, overlaps.m, overlaps.q, rho, tau, epsilon, overlaps.P, overlaps.N, overlaps.sigma),-int_lims,int_lims,limit=500)[0]
    Iminus = quad(lambda xi: _P_hat_integrand(xi,-1, overlaps.m, overlaps.q, rho, tau, epsilon, overlaps.P, overlaps.N, overlaps.sigma),-int_lims,int_lims,limit=500)[0]

    return alpha * (Iplus + Iminus)

@nb.njit
def _N_hat_integrand(xi: float, y: float, m: float, q: float, rho: float, tau: float, epsilon: float, P: float, N: float, sigma: float) -> float:
    e = m * m / (rho * q)
    w_0 = np.sqrt(rho*e) * xi
    V_0 = rho * (1-e)
    z_0 = math.erfc((-y * w_0) / np.sqrt(2*(tau**2 + V_0)))
    w = np.sqrt(q) * xi

    z_star = evaluate_proximal(sigma,y,epsilon*P/np.sqrt(N),w)    


    m_derivative = -(z_star - w)/sigma


    m_derivative *= y*0.5*epsilon*P/(N**(3/2))
    return z_0 * m_derivative * gaussian(xi,0,1)

def logistic_N_hat_func(overlaps: OverlapSet, rho: float, alpha: float, epsilon: float, tau:float, int_lims: float = 20.0):
    Iplus = quad(lambda xi: _N_hat_integrand(xi,1, overlaps.m, overlaps.q, rho, tau, epsilon, overlaps.P, overlaps.N, overlaps.sigma),-int_lims,int_lims,limit=500)[0]
    Iminus = quad(lambda xi: _N_hat_integrand(xi,-1, overlaps.m, overlaps.q, rho, tau, epsilon, overlaps.P, overlaps.N, overlaps.sigma),-int_lims,int_lims,limit=500)[0]
    return alpha * (Iplus + Iminus)


"""
------------------------------------------------------------------------------------------------------------------------
    Ridge Equations
------------------------------------------------------------------------------------------------------------------------
"""

# TODO adapt to tau, the noisy model

def ridge_m_hat_func(task,overlaps,data_model,logger):
    adv = (1 + task.epsilon * overlaps.P/np.sqrt(overlaps.N))
    return adv*task.alpha/np.sqrt(data_model.gamma) * 1/(1+overlaps.sigma) * np.sqrt(2/(np.pi*data_model.rho))

def ridge_q_hat_func(task,overlaps,data_model,logger):
    adv = (1 + task.epsilon * overlaps.P/np.sqrt(overlaps.N))
    return task.alpha * (adv**2 + overlaps.q - 2*overlaps.m*adv*np.sqrt(2/(np.pi*data_model.rho))) / (1+overlaps.sigma)**2

def ridge_sigma_hat_func(task,overlaps,data_model,logger):
    return task.alpha/(1 + overlaps.sigma)

def ridge_P_hat_func(task,overlaps,data_model,logger):
    adv = (1 + task.epsilon * overlaps.P/np.sqrt(overlaps.N))
    result = task.alpha * ( -adv + overlaps.m*np.sqrt(2)/np.sqrt(np.pi*data_model.rho)  ) * 1/(1 + overlaps.sigma)
    result *= task.epsilon / np.sqrt(overlaps.N)
    return -result*2

def ridge_N_hat_func(task,overlaps,data_model,logger):
    adv = (1 + task.epsilon * overlaps.P/np.sqrt(overlaps.N))
    result = task.alpha * ( -adv + overlaps.m*np.sqrt(2)/np.sqrt(np.pi*data_model.rho)  ) * 1/(1 + overlaps.sigma)
    result *= -0.5*task.epsilon*overlaps.P/(overlaps.N**(3/2))
    return -result*2

"""
------------------------------------------------------------------------------------------------------------------------
    Overlap Equations
------------------------------------------------------------------------------------------------------------------------
"""


def var_hat_func(task, overlaps, data_model, logger=None):
    if task.problem_type == ProblemType.Ridge:
        overlaps.m_hat = ridge_m_hat_func(task,overlaps,data_model,logger)
        overlaps.q_hat = ridge_q_hat_func(task,overlaps,data_model,logger)
        overlaps.sigma_hat = ridge_sigma_hat_func(task,overlaps,data_model,logger)
        overlaps.P_hat = ridge_P_hat_func(task,overlaps,data_model,logger)
        overlaps.N_hat = ridge_N_hat_func(task,overlaps,data_model,logger)
    elif task.problem_type == ProblemType.Logistic or task.problem_type == ProblemType.EquivalentLogistic or task.problem_type == ProblemType.PerturbedBoundaryLogistic or task.problem_type == ProblemType.PerturbedBoundaryCoefficientLogistic:
        overlaps.m_hat = logistic_m_hat_func(overlaps,data_model.rho,task.alpha,task.epsilon,task.tau,overlaps.INT_LIMS)/np.sqrt(data_model.gamma)
        overlaps.q_hat = logistic_q_hat_func(overlaps,data_model.rho,task.alpha,task.epsilon,task.tau,overlaps.INT_LIMS, logger)
        overlaps.sigma_hat = logistic_sigma_hat_func(overlaps,data_model.rho,task.alpha,task.epsilon,task.tau,overlaps.INT_LIMS,logger=logger)
        overlaps.P_hat = logistic_P_hat_func(overlaps,data_model.rho,task.alpha,task.epsilon,task.tau,overlaps.INT_LIMS)
        overlaps.N_hat = logistic_N_hat_func(overlaps,data_model.rho,task.alpha,task.epsilon,task.tau,overlaps.INT_LIMS)
    else:
        raise Exception(f"var_hat_func - problem_type {task.problem_type} not implemented")
    return overlaps

def var_func(task, overlaps, data_model, logger, slice_from = None, slice_to = None):

    if slice_to is None:
        slice_to = data_model.d
    if slice_from is None:
        slice_from = 0

    Lambda = task.lam * data_model.spec_Sigma_w[slice_from: slice_to] + overlaps.sigma_hat * data_model.spec_Sigma_x[slice_from: slice_to] + overlaps.P_hat * data_model.spec_Sigma_delta[slice_from: slice_to] + overlaps.N_hat * np.ones(slice_to-slice_from)
    H = data_model.spec_Sigma_x[slice_from: slice_to] * overlaps.q_hat + overlaps.m_hat**2 * data_model.spec_PhiPhit[slice_from: slice_to]

    
    sigma = np.mean(data_model.spec_Sigma_x[slice_from: slice_to]/Lambda)       
    q = np.mean((H * data_model.spec_Sigma_x[slice_from: slice_to]) / Lambda**2) 
    m = overlaps.m_hat/np.sqrt(data_model.gamma) * np.mean(data_model.spec_PhiPhit[slice_from: slice_to]/Lambda)
    
    P = np.mean( H * data_model.spec_Sigma_delta[slice_from: slice_to] / Lambda**2)

    N = np.mean( H * np.ones(slice_to-slice_from) / Lambda**2)

    A = np.mean( H * data_model.spec_Sigma_upsilon[slice_from: slice_to] / Lambda**2)
    F = 0.5* overlaps.m_hat * np.mean( data_model.spec_FTerm[slice_from: slice_to]/ Lambda)

    return m, q, sigma, A, N, P, F

"""
------------------------------------------------------------------------------------------------------------------------
    Logistic Observables
------------------------------------------------------------------------------------------------------------------------
"""

@nb.njit
def _training_error_integrand(xi: float, y: float, m: float, q: float, rho: float, tau: float, epsilon: float, P: float, N: float, sigma: float) -> float:
    e = m * m / (rho * q)
    w_0 = np.sqrt(rho*e) * xi
    V_0 = rho * (1-e)

    z_0 = math.erfc((-y * w_0) / np.sqrt(2*(tau**2 + V_0)))

    # z_out_0 and f_out_0 simplify together as the erfc cancels. See computation
    w = np.sqrt(q) * xi

    proximal = evaluate_proximal(sigma,y,epsilon*P/np.sqrt(N),w)

    activation = np.sign(proximal)

    return z_0* gaussian(xi,0,1) * (activation != y)

@nb.njit
def _training_loss_integrand(xi: float, y: float, q: float, m: float, rho: float, tau: float, epsilon: float, P: float, N: float, sigma: float) -> float:
    w = np.sqrt(q) * xi
    z_0 = math.erfc(  ( (-y * m * xi) / np.sqrt(q) ) / np.sqrt(2*(tau**2 + (rho - m**2/q))))

    proximal = evaluate_proximal(sigma,y,epsilon*P/np.sqrt(N),w)

    l = log1pexp_numba(-y*proximal + epsilon*P/np.sqrt(N))

    return z_0 * l * gaussian(xi,0,1)


@nb.njit
def _test_loss_integrand(xi: float, y: float, m: float, q: float, rho: float, tau: float, epsilon: float, A: float, N: float) -> float:
    e = m * m / (rho * q)
    w_0 = np.sqrt(rho*e) * xi
    V_0 = rho * (1-e)


    z_0 = math.erfc((-y * w_0) / np.sqrt(2*(tau**2 + V_0)))
    
    w = np.sqrt(q) * xi

    loss_value = log1pexp_numba( - y *w + epsilon*A/np.sqrt(N))

    return z_0 * gaussian(xi,0,1) * loss_value

class LogisticObservables:

    @staticmethod
    def training_loss_logistic_with_regularization(task: Task, overlaps: OverlapSet, data_model: AbstractDataModel, int_lims: float):
        return LogisticObservables.training_loss(task, overlaps, data_model, int_lims) + (task.lam/(2*task.alpha)) * overlaps.q

    @staticmethod
    def training_loss(task: Task, overlaps: OverlapSet, data_model: AbstractDataModel, int_lims: float):
        I1 = quad(lambda xi: _training_loss_integrand(xi,1, overlaps.q, overlaps.m, data_model.rho, task.tau, task.epsilon, overlaps.P, overlaps.N, overlaps.sigma) , -int_lims, int_lims, limit=500)[0]
        I2 = quad(lambda xi: _training_loss_integrand(xi,-1, overlaps.q, overlaps.m, data_model.rho, task.tau, task.epsilon, overlaps.P, overlaps.N, overlaps.sigma) , -int_lims, int_lims, limit=500)[0]
        return (I1 + I2)/2

    
    @staticmethod
    def training_error(task: Task, overlaps: OverlapSet, data_model: AbstractDataModel, int_lims: float):
        Iplus = quad(lambda xi: _training_error_integrand(xi,1,overlaps.m,overlaps.q,data_model.rho,task.tau,task.epsilon,overlaps.P,overlaps.N,overlaps.sigma),-int_lims,int_lims,limit=500)[0]
        Iminus = quad(lambda xi: _training_error_integrand(xi,-1,overlaps.m,overlaps.q,data_model.rho,task.tau,task.epsilon,overlaps.P,overlaps.N,overlaps.sigma),-int_lims,int_lims,limit=500)[0]
        return (Iplus + Iminus) * 0.5

    
    @staticmethod
    def test_loss(task: Task, overlaps: OverlapSet, data_model: AbstractDataModel, epsilon: float, int_lims: float):

        Iplus = quad(lambda xi: _test_loss_integrand(xi,1,overlaps.m,overlaps.q,data_model.rho,task.tau,epsilon,overlaps.A,overlaps.N),-int_lims,int_lims,limit=500)[0]
        Iminus = quad(lambda xi: _test_loss_integrand(xi,-1,overlaps.m,overlaps.q,data_model.rho,task.tau,epsilon,overlaps.A,overlaps.N),-int_lims,int_lims,limit=500)[0]
        return (Iplus + Iminus) * 0.5

    


"""
------------------------------------------------------------------------------------------------------------------------
    Ridge Observables
------------------------------------------------------------------------------------------------------------------------
"""
class RidgeObservables:
    # TODO compute and then implement...

    @staticmethod
    def training_loss(task: Task, overlaps: OverlapSet, data_model: AbstractDataModel, int_lims: float):
        return None

    @staticmethod
    def training_error(task: Task, overlaps: OverlapSet, data_model: AbstractDataModel, int_lims: float):
        return None

    @staticmethod
    def test_loss(task: Task, overlaps: OverlapSet, data_model: AbstractDataModel, epsilon: float, int_lims: float):
        return None

"""
------------------------------------------------------------------------------------------------------------------------
    General Observables
------------------------------------------------------------------------------------------------------------------------
"""

def generalization_error(rho,m,q, tau):
    """
    Returns the generalization error in terms of the overlaps
    """
    return np.arccos(m / np.sqrt( (rho + tau**2 ) * q ) )/np.pi


@nb.njit
def teacher_error(nu: float, tau: float, rho: float) -> float:
    return np.exp(- (nu**2) / (2*rho)) * (1 + math.erf( nu / (np.sqrt(2) * tau ) ))

def compute_data_model_angle(data_model: AbstractDataModel, overlaps: OverlapSet, tau):
    L = overlaps.sigma_hat * data_model.spec_Sigma_x + overlaps.P_hat * data_model.spec_Sigma_delta + overlaps.N_hat * np.ones(data_model.d)
    return np.sum(data_model.spec_PhiPhit / L) / np.sqrt( ( data_model.d * tau**2 + data_model.d * data_model.rho) * np.sum(data_model.spec_PhiPhit * data_model.spec_Sigma_x / L**2) )

def compute_data_model_attackability(data_model: AbstractDataModel, overlaps: OverlapSet):
    L = overlaps.sigma_hat * data_model.spec_Sigma_x + overlaps.P_hat * data_model.spec_Sigma_delta + overlaps.N_hat * np.ones(data_model.d)
    return np.sum(data_model.spec_PhiPhit * data_model.spec_Sigma_upsilon / L**2 ) / np.sqrt( np.sum(data_model.spec_PhiPhit / L**2) * np.sum(data_model.spec_PhiPhit * data_model.spec_Sigma_x / L**2) )

def asymptotic_adversarial_generalization_error(data_model: AbstractDataModel, overlaps: OverlapSet, epsilon, tau):

    angle = compute_data_model_angle(data_model, overlaps, tau)
    attackability = compute_data_model_attackability(data_model, overlaps)

    a = angle/ np.sqrt(1 - angle**2)

    b = epsilon * attackability

    owen = 2 * owens_t(a*b , 1/a)

    erferfc = 0.5 * erf(b/np.sqrt(2)) * erfc(-a*b/np.sqrt(2))

    gen_error = owen + erferfc 

    return gen_error

def adversarial_generalization_error_overlaps_teacher(overlaps: OverlapSet, task: Task, data_model: AbstractDataModel, epsilon: float):

    # if tau is not zero, we can use the simpler formula
    if task.tau >= 1e-10:
    
        I = quad(lambda nu: teacher_error(nu, task.tau, data_model.rho), epsilon * overlaps.F / np.sqrt(overlaps.N), np.inf )[0]
        return 1 - I / (np.sqrt(2 * np.pi * data_model.rho))


    return erf( epsilon * overlaps.F / np.sqrt( 2 * data_model.rho * overlaps.N ) )


def adversarial_generalization_error_overlaps(overlaps: OverlapSet, task: Task, data_model: AbstractDataModel, epsilon:float):

    a = overlaps.m/np.sqrt((overlaps.q* (data_model.rho + task.tau**2 ) - overlaps.m**2))

    b = epsilon * overlaps.A / np.sqrt(overlaps.N* overlaps.q)

    owen = 2 * owens_t(a*b , 1/a)

    erferfc = 0.5 * erf(b/np.sqrt(2)) * erfc(-a*b/np.sqrt(2))

    gen_error = owen + erferfc 


    angle = generalization_error(data_model.rho,overlaps.m,overlaps.q,task.tau)

    def integrand(xi):
        return (1/np.sqrt(np.pi*2)) * np.exp(-(xi**2)/(2*data_model.rho)) * (1 + erf(xi/(np.sqrt(2)*task.tau))) * gaussian(xi,0,1)

    return gen_error

def first_term_fair_error(overlaps, data_model, gamma, epsilon):
    V = (data_model.rho )*overlaps.q - overlaps.m**2
    gamma_max = gamma+epsilon*overlaps.F/np.sqrt(overlaps.N)

    # first term
    def erfc_term(nu):
        return np.exp((-(nu**2))/(2*(data_model.rho ))) * erfc( ( overlaps.F*overlaps.m*nu + overlaps.A*(data_model.rho )*( gamma - nu ) ) / (overlaps.F * np.sqrt( 2 * (data_model.rho ) * V )))
    
    def erf_term(nu):
        return np.exp((-(nu**2))/(2 * (data_model.rho ))) * (1 + erf( ( overlaps.F * overlaps.m * nu - overlaps.A * (data_model.rho ) * ( nu + gamma) ) / (overlaps.F * np.sqrt(2 * (data_model.rho ) * V)) ))

    first_term = quad(lambda nu: erfc_term(nu),gamma,gamma_max,limit=500)[0]
    first_term += quad(lambda nu: erf_term(nu),-gamma_max,-gamma,limit=500)[0]
    first_term /= (2*np.sqrt(2*np.pi * (data_model.rho )))
    return first_term

def second_term_fair_error(overlaps, data_model, gamma, epsilon):
    V = (data_model.rho )*overlaps.q - overlaps.m**2
    gamma_max = gamma+epsilon*overlaps.F/np.sqrt(overlaps.N)

    # second term
    def second_integral(nu):
        return erfc((-epsilon*overlaps.A*(data_model.rho ) + np.sqrt(overlaps.N)*overlaps.m*nu)/np.sqrt(overlaps.N*2*(data_model.rho ) * V)) * np.exp(-(nu)**2 / (2*(data_model.rho )))

    result2 = quad(lambda nu: second_integral(nu),gamma_max,np.inf,limit=500)
    second_term = result2[0]
    second_term /= np.sqrt(2*np.pi * (data_model.rho ))

    return second_term

def third_term_fair_error(overlaps, data_model, gamma, epsilon):
    V = (data_model.rho )*overlaps.q - overlaps.m**2
    gamma_max = gamma+epsilon*overlaps.F/np.sqrt(overlaps.N)

    # third term
    def third_integral(nu):
        return np.exp(-(nu)**2/(2*(data_model.rho )) ) * erfc( overlaps.m*nu / np.sqrt(2*(data_model.rho ) * V))
    result3 = quad(lambda nu: third_integral(nu),0,gamma,limit=500)
    third_term = result3[0]
    third_term /= np.sqrt(2*np.pi * (data_model.rho ))

    return third_term

def fair_adversarial_error_overlaps(overlaps, data_model, gamma, epsilon, logger=None):
    
    first_term = first_term_fair_error(overlaps, data_model, gamma, epsilon)

    second_term = second_term_fair_error(overlaps, data_model, gamma, epsilon)

    third_term = third_term_fair_error(overlaps, data_model, gamma, epsilon)
    
    return first_term + second_term + third_term

def overlap_calibration(rho,p,m,q_erm,tau, debug = False):
    """
    Analytical calibration for a given probability p, overlaps m and q and a given noise level tau
    Given by equation 23 in 2202.03295.
    Returns the calibration value.

    p: probability between 0 and 1
    m: overlap between teacher and student
    q_erm: student overlap
    tau: noise level
    """
    logi = logit(p)
    m_q_ratio = m/(q_erm)

    num = (logi )* m_q_ratio 
    if debug:
        print("tau",tau,"m**2",m**2,"q_erm",q_erm,"m**2/q_erm",m**2/q_erm)
    denom = np.sqrt(rho - (m**2)/(q_erm) + (tau)**2)
    if debug:
        print("logi",logi,"m_q_ratio",m_q_ratio,"num",num,"denom",denom)
    return p - sigma_star( ( num ) / ( denom )  )


"""
------------------------------------------------------------------------------------------------------------------------
    Fixed Point Iteration
------------------------------------------------------------------------------------------------------------------------
"""

def damped_update(new, old, damping):
    return damping * new + (1 - damping) * old


def fixed_point_finder(
    logger,
    my_data_model: AbstractDataModel,
    task: Task,
    log: bool = True,
):
    
    overlaps = OverlapSet()  

    
    err = 1.0
    iter_nb = 0
    
    while err > overlaps.TOL_FPE or iter_nb < overlaps.MIN_ITER_FPE:
        if iter_nb % 5000 == 0 and log:
            logger.info(f"iter_nb: {iter_nb}, err: {err}")
            overlaps.log_overlaps(logger)


        overlaps = var_hat_func(task, overlaps, my_data_model, logger)

        new_m, new_q, new_sigma, new_A, new_N, new_P, new_F = var_func(task, overlaps, my_data_model, logger)

        err = overlaps.update_overlaps(new_m, new_q, new_sigma, new_A, new_N, new_P, new_F)       

        iter_nb += 1
        if iter_nb > overlaps.MAX_ITER_FPE:
            raise Exception("fixed_point_finder - reached max_iterations")
    return overlaps



