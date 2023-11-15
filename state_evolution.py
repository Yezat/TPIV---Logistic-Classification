import numpy as np
from typing import Tuple
from scipy.integrate import quad, dblquad
from scipy.special import erfc, erf, logit, owens_t
from helpers import *
from data_model import *
from scipy.optimize import root_scalar

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

def logistic_m_hat_func(overlaps: OverlapSet, rho: float, alpha: float, epsilon: float, tau:float, lam: float, int_lims: float = 20.0):

    def integrand(xi, y):
        e = overlaps.m * overlaps.m / (rho * overlaps.q)
        w_0 = np.sqrt(rho*e) * xi
        V_0 = rho * (1-e)

        # z_out_0 and f_out_0 simplify together as the erfc cancels. See computation
        w = np.sqrt(overlaps.q) * xi

        partial_prox =  proximal_logistic_root_scalar(overlaps.sigma,y,epsilon*overlaps.P/np.sqrt(overlaps.N),w) - w

        return partial_prox * gaussian(w_0,0,V_0+tau**2) * gaussian(xi)

    Iplus = quad(lambda xi: integrand(xi,1),-int_lims,int_lims,limit=500)[0]
    Iminus = quad(lambda xi: integrand(xi,-1),-int_lims,int_lims,limit=500)[0]
    return alpha / overlaps.sigma * (Iplus - Iminus)

def logistic_q_hat_func(overlaps: OverlapSet, rho: float, alpha: float, epsilon: float, tau:float, lam: float, int_lims: float = 20.0):

    def integrand(xi, y):
        e = overlaps.m * overlaps.m / (rho * overlaps.q)
        w_0 = np.sqrt(rho*e) * xi
        V_0 = rho * (1-e)

        z_0 = erfc((-y * w_0) / np.sqrt(2*(tau**2 + V_0)))

        w = np.sqrt(overlaps.q) * xi
        proximal = proximal_logistic_root_scalar(overlaps.sigma,y,epsilon*overlaps.P/np.sqrt(overlaps.N),w)
        partial_proximal = ( proximal - w ) ** 2

        return z_0 * (partial_proximal/ (overlaps.sigma ** 2) ) * gaussian(xi)

    Iplus = quad(lambda xi: integrand(xi,1),-int_lims,int_lims,limit=500)[0]
    Iminus = quad(lambda xi: integrand(xi,-1),-int_lims,int_lims,limit=500)[0]

    return  0.5 * alpha * (Iplus + Iminus)

def logistic_sigma_hat_func(overlaps: OverlapSet, rho: float, alpha: float, epsilon: float, tau: float, lam: float, int_lims: float = 20.0, logger = None):

    """
    Derivative of f_out
    """
    def alternative_derivative_f_out(xi,y):
        w = np.sqrt(overlaps.q) * xi
        proximal = proximal_logistic_root_scalar(overlaps.sigma,y,epsilon*overlaps.P/np.sqrt(overlaps.N),w)

        second_derivative = second_derivative_loss(y,proximal,epsilon*overlaps.P/np.sqrt(overlaps.N))

        return second_derivative / ( 1 + overlaps.sigma * second_derivative)

    def integrand(xi, y):
        z_0 = erfc(  ( (-y * overlaps.m) / np.sqrt(overlaps.q) * xi) / np.sqrt(2*(tau**2 + (rho - overlaps.m**2/overlaps.q))))

        derivative_f_out = alternative_derivative_f_out(xi,y)

        return z_0 * ( derivative_f_out ) * gaussian(xi)

 
    Iplus = quad(lambda xi: integrand(xi,1) , -int_lims, int_lims, limit=500)[0]
    Iminus = quad(lambda xi: integrand(xi,-1) , -int_lims, int_lims, limit=500)[0]

    return  0.5 * alpha * (Iplus + Iminus)


def logistic_P_hat_func(overlaps: OverlapSet, rho: float, alpha: float, epsilon: float, tau:float, lam: float, int_lims: float = 20.0):

    def integrand(xi, y):
        e = overlaps.m * overlaps.m / (rho * overlaps.q)
        w_0 = np.sqrt(rho*e) * xi
        V_0 = rho * (1-e)

        z_0 = erfc((-y * w_0) / np.sqrt(2*(tau**2 + V_0)))

        w = np.sqrt(overlaps.q) * xi

        z_star = proximal_logistic_root_scalar(overlaps.sigma,y,epsilon*overlaps.P/np.sqrt(overlaps.N),w)
        
        arg = y*z_star - epsilon * overlaps.P/np.sqrt(overlaps.N)
        m_derivative = sigmoid(-arg)
    

        m_derivative *= epsilon / np.sqrt(overlaps.N)

        return z_0 * m_derivative * gaussian(xi)

    Iplus = quad(lambda xi: integrand(xi,1),-int_lims,int_lims,limit=500)[0]
    Iminus = quad(lambda xi: integrand(xi,-1),-int_lims,int_lims,limit=500)[0]

    return alpha * (Iplus + Iminus)


def logistic_N_hat_func(overlaps: OverlapSet, rho: float, alpha: float, epsilon: float, tau:float, lam: float, int_lims: float = 20.0):
    def integrand(xi, y):
        e = overlaps.m * overlaps.m / (rho * overlaps.q)
        w_0 = np.sqrt(rho*e) * xi
        V_0 = rho * (1-e)

        z_0 = erfc((-y * w_0) / np.sqrt(2*(tau**2 + V_0)))

        w = np.sqrt(overlaps.q) * xi

        z_star = proximal_logistic_root_scalar(overlaps.sigma,y,epsilon*overlaps.P/np.sqrt(overlaps.N),w)
        

        arg = y*z_star - epsilon * overlaps.P/np.sqrt(overlaps.N)
        m_derivative = sigmoid(-arg)

        m_derivative *= -0.5*epsilon*overlaps.P/(overlaps.N**(3/2))

        return z_0 * m_derivative * gaussian(xi)

    Iplus = quad(lambda xi: integrand(xi,1),-int_lims,int_lims,limit=500)[0]
    Iminus = quad(lambda xi: integrand(xi,-1),-int_lims,int_lims,limit=500)[0]

    return alpha * (Iplus + Iminus)


"""
------------------------------------------------------------------------------------------------------------------------
    Ridge Equations
------------------------------------------------------------------------------------------------------------------------
"""

        # Vhat = self.alpha * 1/(1+V)
        # qhat = self.alpha * (1 + q - 2*m*np.sqrt(2/(np.pi*self.data_model.rho))) / (1+V)**2
        # mhat = self.alpha/np.sqrt(self.data_model.gamma) * 1/(1+V) * np.sqrt(2/(np.pi*self.data_model.rho))

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
    elif task.problem_type == ProblemType.Logistic:
        overlaps.m_hat = logistic_m_hat_func(overlaps,data_model.rho,task.alpha,task.epsilon,task.tau,task.lam,overlaps.INT_LIMS)/np.sqrt(data_model.gamma)
        overlaps.q_hat = logistic_q_hat_func(overlaps,data_model.rho,task.alpha,task.epsilon,task.tau,task.lam,overlaps.INT_LIMS)
        overlaps.sigma_hat = logistic_sigma_hat_func(overlaps,data_model.rho,task.alpha,task.epsilon,task.tau,task.lam,overlaps.INT_LIMS,logger=logger)
        overlaps.P_hat = logistic_P_hat_func(overlaps,data_model.rho,task.alpha,task.epsilon,task.tau,task.lam,overlaps.INT_LIMS)
        overlaps.N_hat = logistic_N_hat_func(overlaps,data_model.rho,task.alpha,task.epsilon,task.tau,task.lam,overlaps.INT_LIMS)
    else:
        raise Exception(f"var_hat_func - problem_type {task.problem_type} not implemented")
    return overlaps

def var_func(task, overlaps, data_model, logger):

    Lambda = task.lam * data_model.spec_Sigma_w + overlaps.sigma_hat * data_model.spec_Sigma_x + overlaps.P_hat * data_model.spec_Sigma_delta + overlaps.N_hat * np.ones(data_model.d)
    H = data_model.spec_Sigma_x * overlaps.q_hat + overlaps.m_hat**2 * data_model.spec_PhiPhit 

    
    sigma = np.mean(data_model.spec_Sigma_x/Lambda)       
    q = np.mean((H * data_model.spec_Sigma_x) / Lambda**2) 
    m = overlaps.m_hat/np.sqrt(data_model.gamma) * np.mean(data_model.spec_PhiPhit/Lambda)
    
    P = np.mean( H * data_model.spec_Sigma_delta / Lambda**2)

    N = np.mean( H * np.ones(data_model.d) / Lambda**2)

    A = np.mean( H * data_model.spec_Sigma_upsilon / Lambda**2)
    F = 0.5* overlaps.m_hat * np.mean( data_model.spec_FTerm / Lambda)

    return m, q, sigma, A, N, P, F

"""
------------------------------------------------------------------------------------------------------------------------
    Logistic Observables
------------------------------------------------------------------------------------------------------------------------
"""

class LogisticObservables:

    @staticmethod
    def training_loss_logistic_with_regularization(task: Task, overlaps: OverlapSet, data_model: AbstractDataModel, int_lims: float):
        return LogisticObservables.training_loss(task, overlaps, data_model, int_lims) + (task.lam/(2*task.alpha)) * overlaps.q

    @staticmethod
    def training_loss(task: Task, overlaps: OverlapSet, data_model: AbstractDataModel, int_lims: float):

        def integrand(xi,y):
            w = np.sqrt(overlaps.q) * xi
            z_0 = erfc(  ( (-y * overlaps.m * xi) / np.sqrt(overlaps.q) ) / np.sqrt(2*(task.tau**2 + (data_model.rho - overlaps.m**2/overlaps.q))))

            proximal = proximal_logistic_root_scalar(overlaps.sigma,y,task.epsilon*overlaps.P/np.sqrt(overlaps.N),w)

            l = adversarial_loss(y,proximal, task.epsilon*overlaps.P/np.sqrt(overlaps.N))

            return z_0 * l * gaussian(xi)

        I1 = quad(lambda xi: integrand(xi,1) , -int_lims, int_lims, limit=500)[0]
        I2 = quad(lambda xi: integrand(xi,-1) , -int_lims, int_lims, limit=500)[0]
        return (I1 + I2)/2

    @staticmethod
    def training_error(task: Task, overlaps: OverlapSet, data_model: AbstractDataModel, int_lims: float):
        def integrand(xi, y):
            e = overlaps.m * overlaps.m / (data_model.rho * overlaps.q)
            w_0 = np.sqrt(data_model.rho*e) * xi
            V_0 = data_model.rho * (1-e)

            z_0 = erfc((-y * w_0) / np.sqrt(2*(task.tau**2 + V_0)))

            # z_out_0 and f_out_0 simplify together as the erfc cancels. See computation
            w = np.sqrt(overlaps.q) * xi

            proximal = proximal_logistic_root_scalar(overlaps.sigma,y,task.epsilon*overlaps.P/np.sqrt(overlaps.N),w)

            activation = np.sign(proximal)

            return z_0* gaussian(xi) * (activation != y)

        Iplus = quad(lambda xi: integrand(xi,1),-int_lims,int_lims,limit=500)[0]
        Iminus = quad(lambda xi: integrand(xi,-1),-int_lims,int_lims,limit=500)[0]
        return (Iplus + Iminus) * 0.5

    @staticmethod
    def test_loss(task: Task, overlaps: OverlapSet, data_model: AbstractDataModel, int_lims: float):
        def integrand(xi, y):
            e = overlaps.m * overlaps.m / (data_model.rho * overlaps.q)
            w_0 = np.sqrt(data_model.rho*e) * xi
            V_0 = data_model.rho * (1-e)

            z_0 = erfc((-y * w_0) / np.sqrt(2*(task.tau**2 + V_0)))
            # z_out_0 and f_out_0 simplify together as the erfc cancels. See computation
            w = np.sqrt(overlaps.q) * xi

            loss_value = adversarial_loss(y,w,task.test_against_epsilon*overlaps.A/np.sqrt(overlaps.N))

            return z_0 * gaussian(xi) * loss_value

        Iplus = quad(lambda xi: integrand(xi,1),-int_lims,int_lims,limit=500)[0]
        Iminus = quad(lambda xi: integrand(xi,-1),-int_lims,int_lims,limit=500)[0]
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
    def test_loss(task: Task, overlaps: OverlapSet, data_model: AbstractDataModel, int_lims: float):
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


def adversarial_generalization_error_overlaps_teacher(overlaps: OverlapSet, task: Task, data_model: AbstractDataModel):
    return erf( task.test_against_epsilon * overlaps.F / np.sqrt( 2 * data_model.rho * overlaps.N ) )


def adversarial_generalization_error_overlaps(overlaps: OverlapSet, task: Task, data_model: AbstractDataModel):

    a = overlaps.m/np.sqrt((overlaps.q* data_model.rho - overlaps.m**2))

    b = task.test_against_epsilon * overlaps.A / np.sqrt(overlaps.N* overlaps.q)

    owen = 2 * owens_t(a*b , 1/a)

    erferfc = 0.5 * erf(b/np.sqrt(2)) * erfc(-a*b/np.sqrt(2))

    gen_error = owen + erferfc 

    return gen_error



def fair_adversarial_error_overlaps(overlaps: OverlapSet, task: Task, data_model: AbstractDataModel, gamma, logger=None):
    
    
    V = data_model.rho*overlaps.q - overlaps.m**2
    gamma_max = gamma+task.test_against_epsilon*overlaps.F/np.sqrt(overlaps.N)
    gamma_star = max(gamma, gamma_max)

    # first term
    def first_integrand(nu, y):
        return erfc( ( y*overlaps.m*overlaps.F*nu - y* data_model.rho * np.sqrt(overlaps.N) * ( np.abs(nu) - gamma ) ) / (overlaps.F * np.sqrt(2*V*data_model.rho))  ) * np.exp(-(nu)**2 / 2)
    
    r1 = quad(lambda nu: first_integrand(nu,-1),-gamma_max,-gamma,limit=500)
    first_term = r1[0]
    r2 = quad(lambda nu: first_integrand(nu,1),gamma,gamma_max,limit=500)
    first_term += r2[0]
    first_term /= (2*np.sqrt(2*np.pi * data_model.rho))
    

    # second term
    def second_integral(nu):
        return erfc((-task.test_against_epsilon*np.sqrt(overlaps.N)*data_model.rho + overlaps.m*nu)/np.sqrt(2*data_model.rho * V)) * np.exp(-(nu)**2 / (2*data_model.rho))


    
    result2 = quad(lambda nu: second_integral(nu),gamma_star,np.inf,limit=500)
    second_term = result2[0]
    second_term /= np.sqrt(2*np.pi * data_model.rho)    



    # third term
    def third_integral(nu):
        return np.exp(-(nu)**2/(2*data_model.rho) ) * erfc( overlaps.m*nu / np.sqrt(2*data_model.rho * V))
    result3 = quad(lambda nu: third_integral(nu),0,gamma,limit=500)
    third_term = result3[0]
    third_term /= np.sqrt(2*np.pi * data_model.rho)    
    
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
        if iter_nb % 10 == 0 and log:
            logger.info(f"iter_nb: {iter_nb}, err: {err}")
            overlaps.log_overlaps(logger)


        overlaps = var_hat_func(task, overlaps, my_data_model, logger)

        new_m, new_q, new_sigma, new_A, new_N, new_P, new_F = var_func(task, overlaps, my_data_model, logger)

        err = overlaps.update_overlaps(new_m, new_q, new_sigma, new_A, new_N, new_P, new_F)


        iter_nb += 1
        if iter_nb > overlaps.MAX_ITER_FPE:
            raise Exception("fixed_point_finder - reached max_iterations")
    return overlaps

