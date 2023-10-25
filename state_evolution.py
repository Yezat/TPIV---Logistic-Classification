import numpy as np
from typing import Tuple
from scipy.integrate import quad
from scipy.special import erfc
from helpers import *
from data_model import *
from scipy.optimize import root_scalar
from scipy.special import logit

"""
------------------------------------------------------------------------------------------------------------------------
    Helper Classes
------------------------------------------------------------------------------------------------------------------------
"""
class OverlapSet():
    def __init__(self) -> None:
        self.INITIAL_CONDITION = (0.1,0.1,0.5,0.1,0.1,0.1,0.1)

        self.m = self.INITIAL_CONDITION[0]
        self.q = self.INITIAL_CONDITION[1]
        self.sigma = self.INITIAL_CONDITION[2]
        self.A = self.INITIAL_CONDITION[3]
        self.N = self.INITIAL_CONDITION[4]
        self.a = self.INITIAL_CONDITION[5]
        self.n = self.INITIAL_CONDITION[6]
        
        self.m_hat = 0
        self.q_hat = 0
        self.sigma_hat = 0
        self.A_hat = 0
        self.N_hat = 0
        self.a_hat = 0
        self.n_hat = 0

        self.BLEND_FPE = 0.75
        self.TOL_FPE = 1e-4
        self.MIN_ITER_FPE = 10
        self.MAX_ITER_FPE = 5000
        self.INT_LIMS = 10.0

    def log_overlaps(self, logger):
        logger.info(f"m: {self.m}, q: {self.q}, sigma: {self.sigma}, A: {self.A}, N: {self.N}, a: {self.a}, n: {self.n}")
        logger.info(f"m_hat: {self.m_hat}, q_hat: {self.q_hat}, sigma_hat: {self.sigma_hat}, A_hat: {self.A_hat}, N_hat: {self.N_hat}, a_hat: {self.a_hat}, n_hat: {self.n_hat}")

    def update_overlaps(self, m,q,sigma,A,N,a,n):
        self.n_m = damped_update(m, self.m, self.BLEND_FPE)
        self.n_q = damped_update(q, self.q, self.BLEND_FPE)
        self.n_sigma = damped_update(sigma, self.sigma, self.BLEND_FPE)
        self.n_A = damped_update(A, self.A, self.BLEND_FPE)
        self.n_N = damped_update(N, self.N, self.BLEND_FPE)
        self.n_a = damped_update(a, self.a, self.BLEND_FPE)
        self.n_n = damped_update(n, self.n, self.BLEND_FPE)

        # Compute the error
        err = max([abs(self.n_m - self.m), abs(self.n_q - self.q), abs(self.n_sigma - self.sigma), abs(self.n_A - self.A), abs(self.n_N - self.N), abs(self.n_a - self.a), abs(self.n_n - self.n)])

        # Update the overlaps
        self.m = self.n_m
        self.q = self.n_q
        self.sigma = self.n_sigma
        self.A = self.n_A
        self.N = self.n_N
        self.a = self.n_a
        self.n = self.n_n

        return err
        


"""
------------------------------------------------------------------------------------------------------------------------
    Proximals
------------------------------------------------------------------------------------------------------------------------
"""
def optim(z,y,V,w_prime):
    a = y*z
    if a <= 0:
        return y*V/(1+ np.exp(y*z)) + w_prime - z
    else:
        return y*V*np.exp(-y*z)/(1+ np.exp(-y*z)) + w_prime - z

def proximal_logistic_root_scalar(V: float, y: float, epsilon_term: float, w:float) -> float:
    if y == 0:
        return w
    try:
        w_prime = w - epsilon_term / y
        result = root_scalar(lambda z: optim(z,y,V,w_prime) , bracket=[-50000000,50000000],xtol=10e-10,rtol=10e-10)
        z = result.root
        return z + epsilon_term / y
    except Exception as e:
        # print all parameters
        print("V: ", V, "y: ", y, "epsilon_term:", epsilon_term, "w: ", w)
        raise e



"""
------------------------------------------------------------------------------------------------------------------------
    Hat Overlap Equations
------------------------------------------------------------------------------------------------------------------------
"""

def m_hat_func(m: float, q: float, sigma: float, A: float, N: float, a:float, n:float, rho: float, alpha: float, epsilon: float, tau:float, lam: float, int_lims: float = 20.0):

    def integrand(xi, y):
        e = m * m / (rho * q)
        w_0 = np.sqrt(rho*e) * xi
        V_0 = rho * (1-e)

        # z_out_0 and f_out_0 simplify together as the erfc cancels. See computation
        w = np.sqrt(q) * xi

        partial_prox =  proximal_logistic_root_scalar(sigma,y,epsilon*a/np.sqrt(n),w) - w

        return partial_prox * gaussian(w_0,0,V_0+tau**2) * gaussian(xi)

    Iplus = quad(lambda xi: integrand(xi,1),-int_lims,int_lims,limit=500)[0]
    Iminus = quad(lambda xi: integrand(xi,-1),-int_lims,int_lims,limit=500)[0]
    return alpha / sigma * (Iplus - Iminus)

def q_hat_func(m: float, q: float, sigma: float, A: float, N: float, a:float, n:float, rho: float, alpha: float, epsilon: float, tau:float, lam: float, int_lims: float = 20.0):

    def integrand(xi, y):
        e = m * m / (rho * q)
        w_0 = np.sqrt(rho*e) * xi
        V_0 = rho * (1-e)

        z_0 = erfc((-y * w_0) / np.sqrt(2*(tau**2 + V_0)))

        w = np.sqrt(q) * xi
        proximal = proximal_logistic_root_scalar(sigma,y,epsilon*a/np.sqrt(n),w)
        partial_proximal = ( proximal - w ) ** 2

        return z_0 * (partial_proximal/ (sigma ** 2) ) * gaussian(xi)

    Iplus = quad(lambda xi: integrand(xi,1),-int_lims,int_lims,limit=500)[0]
    Iminus = quad(lambda xi: integrand(xi,-1),-int_lims,int_lims,limit=500)[0]

    return  0.5 * alpha * (Iplus + Iminus)

def sigma_hat_func(m: float, q: float, sigma: float, A: float, N: float, a:float, n:float, rho: float, alpha: float, epsilon: float, tau: float, lam: float, int_lims: float = 20.0, logger = None):

    """
    Derivative of f_out
    """
    def alternative_derivative_f_out(xi,y):
        w = np.sqrt(q) * xi
        proximal = proximal_logistic_root_scalar(sigma,y,epsilon*a/np.sqrt(n),w)

        second_derivative = second_derivative_loss(y,proximal,epsilon*a/np.sqrt(n))

        return second_derivative / ( 1 + sigma * second_derivative)

    def integrand(xi, y):
        z_0 = erfc(  ( (-y * m) / np.sqrt(q) * xi) / np.sqrt(2*(tau**2 + (rho - m**2/q))))

        derivative_f_out = alternative_derivative_f_out(xi,y)

        return z_0 * ( derivative_f_out ) * gaussian(xi)

 
    Iplus = quad(lambda xi: integrand(xi,1) , -int_lims, int_lims, limit=500)[0]
    Iminus = quad(lambda xi: integrand(xi,-1) , -int_lims, int_lims, limit=500)[0]

    return  0.5 * alpha * (Iplus + Iminus)


def A_hat_func(m: float, q: float, sigma: float, A: float, N: float, a:float, n:float, rho_w_star: float, alpha: float, epsilon: float, tau:float, lam: float, int_lims: float = 20.0):
    return 0
def N_hat_func(m: float, q: float, sigma: float, A: float, N: float, a:float, n:float, rho_w_star: float, alpha: float, epsilon: float, tau:float, lam: float, int_lims: float = 20.0):
    return 0

def a_hat_func(m: float, q: float, sigma: float, A: float, N: float, a:float, n:float, rho_w_star: float, alpha: float, epsilon: float, tau:float, lam: float, int_lims: float = 20.0):

    def integrand(xi, y):
        e = m * m / (rho_w_star * q)
        w_0 = np.sqrt(rho_w_star*e) * xi
        V_0 = rho_w_star * (1-e)

        z_0 = erfc((-y * w_0) / np.sqrt(2*(tau**2 + V_0)))

        w = np.sqrt(q) * xi

        z_star = proximal_logistic_root_scalar(sigma,y,epsilon*a/np.sqrt(n),w)
        
        arg = y*z_star - epsilon * a/np.sqrt(n)
        m_derivative = sigmoid(-arg)
    

        m_derivative *= epsilon / np.sqrt(n)

        return z_0 * m_derivative * gaussian(xi)

    Iplus = quad(lambda xi: integrand(xi,1),-int_lims,int_lims,limit=500)[0]
    Iminus = quad(lambda xi: integrand(xi,-1),-int_lims,int_lims,limit=500)[0]

    return alpha * (Iplus + Iminus)


def n_hat_func(m: float, q: float, sigma: float, A: float, N: float, a:float, n:float, rho_w_star: float, alpha: float, epsilon: float, tau:float, lam: float, int_lims: float = 20.0):
    def integrand(xi, y):
        e = m * m / (rho_w_star * q)
        w_0 = np.sqrt(rho_w_star*e) * xi
        V_0 = rho_w_star * (1-e)

        z_0 = erfc((-y * w_0) / np.sqrt(2*(tau**2 + V_0)))

        w = np.sqrt(q) * xi

        z_star = proximal_logistic_root_scalar(sigma,y,epsilon*a/np.sqrt(n),w)
        

        arg = y*z_star - epsilon * a/np.sqrt(n)
        m_derivative = sigmoid(-arg)

        m_derivative *= -0.5*epsilon*a/(n**(3/2))

        return z_0 * m_derivative * gaussian(xi)

    Iplus = quad(lambda xi: integrand(xi,1),-int_lims,int_lims,limit=500)[0]
    Iminus = quad(lambda xi: integrand(xi,-1),-int_lims,int_lims,limit=500)[0]

    return alpha * (Iplus + Iminus)

def var_hat_func(task, overlaps, data_model, logger=None):
    overlaps.m_hat = m_hat_func(overlaps.m, overlaps.q, overlaps.sigma, overlaps.A, overlaps.N,overlaps.a,overlaps.n,data_model.rho,task.alpha,task.epsilon,task.tau,task.lam,overlaps.INT_LIMS)/np.sqrt(data_model.gamma)
    overlaps.q_hat = q_hat_func(overlaps.m, overlaps.q, overlaps.sigma, overlaps.A, overlaps.N,overlaps.a,overlaps.n,data_model.rho,task.alpha,task.epsilon,task.tau,task.lam,overlaps.INT_LIMS)
    overlaps.sigma_hat = sigma_hat_func(overlaps.m, overlaps.q, overlaps.sigma, overlaps.A, overlaps.N,overlaps.a,overlaps.n,data_model.rho,task.alpha,task.epsilon,task.tau,task.lam,overlaps.INT_LIMS,logger=logger)
    overlaps.A_hat = A_hat_func(overlaps.m, overlaps.q, overlaps.sigma, overlaps.A, overlaps.N,overlaps.a,overlaps.n,data_model.rho,task.alpha,task.epsilon,task.tau,task.lam,overlaps.INT_LIMS)
    overlaps.N_hat = N_hat_func(overlaps.m, overlaps.q, overlaps.sigma, overlaps.A, overlaps.N,overlaps.a,overlaps.n,data_model.rho,task.alpha,task.epsilon,task.tau,task.lam,overlaps.INT_LIMS)
    overlaps.a_hat = a_hat_func(overlaps.m, overlaps.q, overlaps.sigma, overlaps.A, overlaps.N,overlaps.a,overlaps.n,data_model.rho,task.alpha,task.epsilon,task.tau,task.lam,overlaps.INT_LIMS)
    overlaps.n_hat = n_hat_func(overlaps.m, overlaps.q, overlaps.sigma, overlaps.A, overlaps.N,overlaps.a,overlaps.n,data_model.rho,task.alpha,task.epsilon,task.tau,task.lam,overlaps.INT_LIMS)
    return overlaps



"""
------------------------------------------------------------------------------------------------------------------------
    Overlap Equations
------------------------------------------------------------------------------------------------------------------------
"""
def var_func(task, overlaps, data_model, logger):

    C = overlaps.A_hat + overlaps.a_hat
    M = overlaps.N_hat + overlaps.n_hat

    Lambda = task.lam * data_model.spec_Sigma_w + overlaps.sigma_hat * data_model.spec_Sigma_x + C * data_model.spec_Sigma_delta + M * np.ones(data_model.d)
    H = data_model.spec_Sigma_x * overlaps.q_hat + overlaps.m_hat**2 * data_model.spec_PhiPhit 

    
    sigma = np.mean(data_model.spec_Sigma_x/Lambda)       
    q = np.mean((H * data_model.spec_Sigma_x) / Lambda**2) 
    m = overlaps.m_hat/np.sqrt(data_model.gamma) * np.mean(data_model.spec_PhiPhit/Lambda)
    
    
    a = np.mean( H * data_model.spec_Sigma_delta / Lambda**2)
    A = a - np.mean( data_model.spec_Sigma_delta / Lambda)
    
    n = np.mean( H * np.ones(data_model.d) / Lambda**2)
    N = n - np.mean( np.ones(data_model.d) / Lambda)


    return m, q, sigma, A, N, a, n

"""
------------------------------------------------------------------------------------------------------------------------
    Observables
------------------------------------------------------------------------------------------------------------------------
"""
def training_loss_logistic(m: float, q: float, sigma: float, A: float, N:float, rho: float, alpha: float, tau: float, epsilon: float, lam: float , int_lims: float = 20.0):
    return pure_training_loss_logistic(m,q,sigma,A,N,rho,alpha,tau,epsilon,lam,int_lims) + (lam/(2*alpha)) * q


def pure_training_loss_logistic(m: float, q: float, sigma: float, A: float, N: float,a:float,n:float, rho: float, alpha: float, tau: float, epsilon: float, lam: float , int_lims: float = 20.0):

    def integrand(xi,y):
        w = np.sqrt(q) * xi
        z_0 = erfc(  ( (-y * m * xi) / np.sqrt(q) ) / np.sqrt(2*(tau**2 + (rho - m**2/q))))

        proximal = proximal_logistic_root_scalar(sigma,y,epsilon*a/np.sqrt(n),w)

        l = adversarial_loss(y,proximal, epsilon*a/np.sqrt(n))

        return z_0 * l * gaussian(xi)


    I1 = quad(lambda xi: integrand(xi,1) , -int_lims, int_lims, limit=500)[0]
    I2 = quad(lambda xi: integrand(xi,-1) , -int_lims, int_lims, limit=500)[0]
    return (I1 + I2)/2

def training_error_logistic(m: float, q: float, sigma: float, A: float, N: float,a:float,n:float, rho: float, alpha: float, tau: float, epsilon: float, lam: float, int_lims: float = 20.0):

    def integrand(xi, y):
        e = m * m / (rho * q)
        w_0 = np.sqrt(rho*e) * xi
        V_0 = rho * (1-e)

        z_0 = erfc((-y * w_0) / np.sqrt(2*(tau**2 + V_0)))

        # z_out_0 and f_out_0 simplify together as the erfc cancels. See computation
        w = np.sqrt(q) * xi

        proximal = proximal_logistic_root_scalar(sigma,y,epsilon*a/np.sqrt(n),w)

        activation = np.sign(proximal)

        return z_0* gaussian(xi) * (activation != y)

    Iplus = quad(lambda xi: integrand(xi,1),-int_lims,int_lims,limit=500)[0]
    Iminus = quad(lambda xi: integrand(xi,-1),-int_lims,int_lims,limit=500)[0]
    return (Iplus + Iminus) * 0.5

def adversarial_generalization_error_logistic(m: float, q: float, rho: float, tau: float, epsilon_term: float, int_lims: float = 20.0):
    def integrand(xi, y):
        e = m * m / (rho * q)
        w_0 = np.sqrt(rho*e) * xi
        V_0 = rho * (1-e)

        z_0 = erfc((-y * w_0) / np.sqrt(2*(tau**2 + V_0)))
        # z_out_0 and f_out_0 simplify together as the erfc cancels. See computation
        w = np.sqrt(q) * xi


        activation = np.sign(w - y*epsilon_term)

        return z_0 * gaussian(xi) * (activation != y)

    Iplus = quad(lambda xi: integrand(xi,1),-int_lims,int_lims,limit=500)[0]
    Iminus = quad(lambda xi: integrand(xi,-1),-int_lims,int_lims,limit=500)[0]
    return (Iplus + Iminus) * 0.5


def test_loss_overlaps(m: float, q: float, rho: float, tau: float, sigma: float, epsilon_term: float, int_lims: float = 20.0):
    def integrand(xi, y):
        e = m * m / (rho * q)
        w_0 = np.sqrt(rho*e) * xi
        V_0 = rho * (1-e)

        z_0 = erfc((-y * w_0) / np.sqrt(2*(tau**2 + V_0)))
        # z_out_0 and f_out_0 simplify together as the erfc cancels. See computation
        w = np.sqrt(q) * xi

        loss_value = adversarial_loss(y,w,epsilon_term)

        return z_0 * gaussian(xi) * loss_value

    Iplus = quad(lambda xi: integrand(xi,1),-int_lims,int_lims,limit=500)[0]
    Iminus = quad(lambda xi: integrand(xi,-1),-int_lims,int_lims,limit=500)[0]
    return (Iplus + Iminus) * 0.5



def generalization_error(rho_w_star,m,q, tau):
    """
    Returns the generalization error in terms of the overlaps
    """
    return np.arccos(m / np.sqrt( (rho_w_star + tau**2 ) * q ) )/np.pi


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
    m_q_ratio = m/(q_erm )

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
    task: Task
):
    
    overlaps = OverlapSet()  

    
    err = 1.0
    iter_nb = 0
    
    while err > overlaps.TOL_FPE or iter_nb < overlaps.MIN_ITER_FPE:
        if iter_nb % 10 == 0:
            logger.info(f"iter_nb: {iter_nb}, err: {err}")
            overlaps.log_overlaps(logger)


        overlaps = var_hat_func(task, overlaps, my_data_model, logger)

        new_m, new_q, new_sigma, new_A, new_N, new_a, new_n = var_func(task, overlaps, my_data_model, logger)

        err = overlaps.update_overlaps(new_m, new_q, new_sigma, new_A, new_N, new_a, new_n)


        iter_nb += 1
        if iter_nb > overlaps.MAX_ITER_FPE:
            raise Exception("fixed_point_finder - reached max_iterations")
    return overlaps

