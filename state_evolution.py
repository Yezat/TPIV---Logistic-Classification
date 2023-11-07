import numpy as np
from typing import Tuple
from scipy.integrate import quad
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
        self.INITIAL_CONDITION = (1e-1,1e-1,1e-1,1e-1,1e-1)

        self.m = self.INITIAL_CONDITION[0]
        self.q = self.INITIAL_CONDITION[1]
        self.sigma = self.INITIAL_CONDITION[2]
        self.A = self.INITIAL_CONDITION[3]
        self.N = self.INITIAL_CONDITION[4]
        
        self.m_hat = 0
        self.q_hat = 0
        self.sigma_hat = 0
        self.A_hat = 0
        self.N_hat = 0

        self.BLEND_FPE = 0.75
        self.TOL_FPE = 1e-4
        self.MIN_ITER_FPE = 10
        self.MAX_ITER_FPE = 5000
        self.INT_LIMS = 10.0

    def log_overlaps(self, logger):
        logger.info(f"m: {self.m}, q: {self.q}, sigma: {self.sigma}, A: {self.A}, N: {self.N}")
        logger.info(f"m_hat: {self.m_hat}, q_hat: {self.q_hat}, sigma_hat: {self.sigma_hat}, A_hat: {self.A_hat}, N_hat: {self.N_hat}")

    def update_overlaps(self, m,q,sigma,A,N):
        self.n_m = damped_update(m, self.m, self.BLEND_FPE)
        self.n_q = damped_update(q, self.q, self.BLEND_FPE)
        self.n_sigma = damped_update(sigma, self.sigma, self.BLEND_FPE)
        self.n_A = damped_update(A, self.A, self.BLEND_FPE)
        self.n_N = damped_update(N, self.N, self.BLEND_FPE)

        # Compute the error
        err = max([abs(self.n_m - self.m), abs(self.n_q - self.q), abs(self.n_sigma - self.sigma), abs(self.n_A - self.A), abs(self.n_N - self.N)])

        # Update the overlaps
        self.m = self.n_m
        self.q = self.n_q
        self.sigma = self.n_sigma
        self.A = self.n_A
        self.N = self.n_N

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

def m_hat_func(overlaps: OverlapSet, rho: float, alpha: float, epsilon: float, tau:float, lam: float, int_lims: float = 20.0):

    def integrand(xi, y):
        e = overlaps.m * overlaps.m / (rho * overlaps.q)
        w_0 = np.sqrt(rho*e) * xi
        V_0 = rho * (1-e)

        # z_out_0 and f_out_0 simplify together as the erfc cancels. See computation
        w = np.sqrt(overlaps.q) * xi

        partial_prox =  proximal_logistic_root_scalar(overlaps.sigma,y,epsilon*overlaps.A/np.sqrt(overlaps.N),w) - w

        return partial_prox * gaussian(w_0,0,V_0+tau**2) * gaussian(xi)

    Iplus = quad(lambda xi: integrand(xi,1),-int_lims,int_lims,limit=500)[0]
    Iminus = quad(lambda xi: integrand(xi,-1),-int_lims,int_lims,limit=500)[0]
    return alpha / overlaps.sigma * (Iplus - Iminus)

def q_hat_func(overlaps: OverlapSet, rho: float, alpha: float, epsilon: float, tau:float, lam: float, int_lims: float = 20.0):

    def integrand(xi, y):
        e = overlaps.m * overlaps.m / (rho * overlaps.q)
        w_0 = np.sqrt(rho*e) * xi
        V_0 = rho * (1-e)

        z_0 = erfc((-y * w_0) / np.sqrt(2*(tau**2 + V_0)))

        w = np.sqrt(overlaps.q) * xi
        proximal = proximal_logistic_root_scalar(overlaps.sigma,y,epsilon*overlaps.A/np.sqrt(overlaps.N),w)
        partial_proximal = ( proximal - w ) ** 2

        return z_0 * (partial_proximal/ (overlaps.sigma ** 2) ) * gaussian(xi)

    Iplus = quad(lambda xi: integrand(xi,1),-int_lims,int_lims,limit=500)[0]
    Iminus = quad(lambda xi: integrand(xi,-1),-int_lims,int_lims,limit=500)[0]

    return  0.5 * alpha * (Iplus + Iminus)

def sigma_hat_func(overlaps: OverlapSet, rho: float, alpha: float, epsilon: float, tau: float, lam: float, int_lims: float = 20.0, logger = None):

    """
    Derivative of f_out
    """
    def alternative_derivative_f_out(xi,y):
        w = np.sqrt(overlaps.q) * xi
        proximal = proximal_logistic_root_scalar(overlaps.sigma,y,epsilon*overlaps.A/np.sqrt(overlaps.N),w)

        second_derivative = second_derivative_loss(y,proximal,epsilon*overlaps.A/np.sqrt(overlaps.N))

        return second_derivative / ( 1 + overlaps.sigma * second_derivative)

    def integrand(xi, y):
        z_0 = erfc(  ( (-y * overlaps.m) / np.sqrt(overlaps.q) * xi) / np.sqrt(2*(tau**2 + (rho - overlaps.m**2/overlaps.q))))

        derivative_f_out = alternative_derivative_f_out(xi,y)

        return z_0 * ( derivative_f_out ) * gaussian(xi)

 
    Iplus = quad(lambda xi: integrand(xi,1) , -int_lims, int_lims, limit=500)[0]
    Iminus = quad(lambda xi: integrand(xi,-1) , -int_lims, int_lims, limit=500)[0]

    return  0.5 * alpha * (Iplus + Iminus)


def A_hat_func(overlaps: OverlapSet, rho: float, alpha: float, epsilon: float, tau:float, lam: float, int_lims: float = 20.0):

    def integrand(xi, y):
        e = overlaps.m * overlaps.m / (rho * overlaps.q)
        w_0 = np.sqrt(rho*e) * xi
        V_0 = rho * (1-e)

        z_0 = erfc((-y * w_0) / np.sqrt(2*(tau**2 + V_0)))

        w = np.sqrt(overlaps.q) * xi

        z_star = proximal_logistic_root_scalar(overlaps.sigma,y,epsilon*overlaps.A/np.sqrt(overlaps.N),w)
        
        arg = y*z_star - epsilon * overlaps.A/np.sqrt(overlaps.N)
        m_derivative = sigmoid(-arg)
    

        m_derivative *= epsilon / np.sqrt(overlaps.N)

        return z_0 * m_derivative * gaussian(xi)

    Iplus = quad(lambda xi: integrand(xi,1),-int_lims,int_lims,limit=500)[0]
    Iminus = quad(lambda xi: integrand(xi,-1),-int_lims,int_lims,limit=500)[0]

    return alpha * (Iplus + Iminus)


def N_hat_func(overlaps: OverlapSet, rho: float, alpha: float, epsilon: float, tau:float, lam: float, int_lims: float = 20.0):
    def integrand(xi, y):
        e = overlaps.m * overlaps.m / (rho * overlaps.q)
        w_0 = np.sqrt(rho*e) * xi
        V_0 = rho * (1-e)

        z_0 = erfc((-y * w_0) / np.sqrt(2*(tau**2 + V_0)))

        w = np.sqrt(overlaps.q) * xi

        z_star = proximal_logistic_root_scalar(overlaps.sigma,y,epsilon*overlaps.A/np.sqrt(overlaps.N),w)
        

        arg = y*z_star - epsilon * overlaps.A/np.sqrt(overlaps.N)
        m_derivative = sigmoid(-arg)

        m_derivative *= -0.5*epsilon*overlaps.A/(overlaps.N**(3/2))

        return z_0 * m_derivative * gaussian(xi)

    Iplus = quad(lambda xi: integrand(xi,1),-int_lims,int_lims,limit=500)[0]
    Iminus = quad(lambda xi: integrand(xi,-1),-int_lims,int_lims,limit=500)[0]

    return alpha * (Iplus + Iminus)

def var_hat_func(task, overlaps, data_model, logger=None):
    overlaps.m_hat = m_hat_func(overlaps,data_model.rho,task.alpha,task.epsilon,task.tau,task.lam,overlaps.INT_LIMS)/np.sqrt(data_model.gamma)
    overlaps.q_hat = q_hat_func(overlaps,data_model.rho,task.alpha,task.epsilon,task.tau,task.lam,overlaps.INT_LIMS)
    overlaps.sigma_hat = sigma_hat_func(overlaps,data_model.rho,task.alpha,task.epsilon,task.tau,task.lam,overlaps.INT_LIMS,logger=logger)
    overlaps.A_hat = A_hat_func(overlaps,data_model.rho,task.alpha,task.epsilon,task.tau,task.lam,overlaps.INT_LIMS)
    overlaps.N_hat = N_hat_func(overlaps,data_model.rho,task.alpha,task.epsilon,task.tau,task.lam,overlaps.INT_LIMS)
    return overlaps



"""
------------------------------------------------------------------------------------------------------------------------
    Overlap Equations
------------------------------------------------------------------------------------------------------------------------
"""
def var_func(task, overlaps, data_model, logger):

    Lambda = task.lam * data_model.spec_Sigma_w + overlaps.sigma_hat * data_model.spec_Sigma_x + overlaps.A_hat * data_model.spec_Sigma_delta + overlaps.N_hat * np.ones(data_model.d)
    H = data_model.spec_Sigma_x * overlaps.q_hat + overlaps.m_hat**2 * data_model.spec_PhiPhit 

    
    sigma = np.mean(data_model.spec_Sigma_x/Lambda)       
    q = np.mean((H * data_model.spec_Sigma_x) / Lambda**2) 
    m = overlaps.m_hat/np.sqrt(data_model.gamma) * np.mean(data_model.spec_PhiPhit/Lambda)
    
    A = np.mean( H * data_model.spec_Sigma_delta / Lambda**2)

    N = np.mean( H * np.ones(data_model.d) / Lambda**2)

    return m, q, sigma, A, N

"""
------------------------------------------------------------------------------------------------------------------------
    Observables
------------------------------------------------------------------------------------------------------------------------
"""
def training_loss_logistic(m: float, q: float, sigma: float, A: float, N:float, rho: float, alpha: float, tau: float, epsilon: float, lam: float , int_lims: float = 20.0):
    return pure_training_loss_logistic(m,q,sigma,A,N,rho,alpha,tau,epsilon,lam,int_lims) + (lam/(2*alpha)) * q


def pure_training_loss_logistic(overlaps: OverlapSet, rho: float, alpha: float, tau: float, epsilon: float, lam: float , int_lims: float = 20.0):

    def integrand(xi,y):
        w = np.sqrt(overlaps.q) * xi
        z_0 = erfc(  ( (-y * overlaps.m * xi) / np.sqrt(overlaps.q) ) / np.sqrt(2*(tau**2 + (rho - overlaps.m**2/overlaps.q))))

        proximal = proximal_logistic_root_scalar(overlaps.sigma,y,epsilon*overlaps.A/np.sqrt(overlaps.N),w)

        l = adversarial_loss(y,proximal, epsilon*overlaps.A/np.sqrt(overlaps.N))

        return z_0 * l * gaussian(xi)


    I1 = quad(lambda xi: integrand(xi,1) , -int_lims, int_lims, limit=500)[0]
    I2 = quad(lambda xi: integrand(xi,-1) , -int_lims, int_lims, limit=500)[0]
    return (I1 + I2)/2

def training_error_logistic(overlaps: OverlapSet, rho: float, alpha: float, tau: float, epsilon: float, lam: float, int_lims: float = 20.0):

    def integrand(xi, y):
        e = overlaps.m * overlaps.m / (rho * overlaps.q)
        w_0 = np.sqrt(rho*e) * xi
        V_0 = rho * (1-e)

        z_0 = erfc((-y * w_0) / np.sqrt(2*(tau**2 + V_0)))

        # z_out_0 and f_out_0 simplify together as the erfc cancels. See computation
        w = np.sqrt(overlaps.q) * xi

        proximal = proximal_logistic_root_scalar(overlaps.sigma,y,epsilon*overlaps.A/np.sqrt(overlaps.N),w)

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

def adversarial_generalization_error_overlaps(overlaps: OverlapSet, task: Task, data_model: AbstractDataModel):
    # gen_error = generalization_error(data_model.rho,overlaps.m,overlaps.q,task.tau)

    a = overlaps.m/np.sqrt((overlaps.q* data_model.rho - overlaps.m**2))

    b = task.test_against_epsilon * overlaps.A / np.sqrt(overlaps.N* overlaps.q)

    owen = 2 * owens_t(a*b , 1/a)

    erferfc = 0.5 * erf(b/np.sqrt(2)) * erfc(-a*b/np.sqrt(2))

    # cot = np.arctan(-1/a)/np.pi

    # print("owen",owen,"erferfc",erferfc,"cot",cot,"gen_error",gen_error)

    gen_error = owen + erferfc 


    # # let's try the crazy formula
    # hat_ratio = 1 / (np.sqrt( data_model.rho - 1 + data_model.rho * overlaps.q_hat/overlaps.m_hat**2  ))

    # # assert that hatm/sqrt(hatq) >= 0
    # assert overlaps.m_hat / np.sqrt(overlaps.q_hat) >= 0, f"hat_ratio: {hat_ratio}, q_hat: {overlaps.q_hat}, m_hat: {overlaps.m_hat}, rho: {data_model.rho}"

    # # assert a precision of 1e-3 between a and hat_ratio
    # assert abs(a - hat_ratio) < 1e-3, f"a: {a}, hat_ratio: {hat_ratio}, q_hat: {overlaps.q_hat}, m_hat: {overlaps.m_hat}, rho: {data_model.rho}"


    # alternative_2 =  0.5 + 0.5*erf(b/np.sqrt(2)) - 2 * owens_t( b, a)

    # assert abs(gen_error - alternative_2) < 1e-15, f"gen_error: {gen_error}, alternative_2: {alternative_2}, m: {overlaps.m}, q: {overlaps.q}"



    # # Let's code up A78 and see if it matches numerically the other expressions
    # alternative_3 = np.arccos(overlaps.m / np.sqrt( data_model.rho * overlaps.q ))/np.pi

    # def integrand(xi):
    #     return np.exp(-xi**2/2) * erfc( - overlaps.m * xi / np.sqrt(2 * (overlaps.q * data_model.rho - overlaps.m**2) ) ) / np.sqrt(2*np.pi)

    # alternative_3 += quad(integrand,0,task.test_against_epsilon * overlaps.A / np.sqrt(overlaps.N* overlaps.q),limit=500)[0]

    # assert abs(gen_error - alternative_3) < 1e-15, f"gen_error: {gen_error}, alternative_3: {alternative_3}, m: {overlaps.m}, q: {overlaps.q}"


    # assert abs(alternative_2 - alternative_3) < 1e-15, f"alternative_2: {alternative_2}, alternative_3: {alternative_3}, m: {overlaps.m}, q: {overlaps.q}"

    return gen_error

def adversarial_generalization_error_overlaps_test(overlaps: OverlapSet, task: Task, data_model: AbstractDataModel):

    gen_error = generalization_error(data_model.rho,overlaps.m,overlaps.q,task.tau)

    def solution(a,b):
        r = -4*np.pi*np.sqrt(a**2)*owens_t(np.sqrt(2)*np.sqrt(a**2)*b,1/np.sqrt(a**2))
        r += np.pi*a*erf(b)*erfc(a*b)
        r += 2*a*np.arctan(1/a)
        r /= 2*np.sqrt(np.pi)*a
        return r

    eg = task.test_against_epsilon * overlaps.A / np.sqrt(overlaps.N* overlaps.q)

    a = -overlaps.m / np.sqrt((overlaps.q* data_model.rho - overlaps.m**2))
    b = eg/np.sqrt(2)
    
    gen_error += solution(a,b)/np.sqrt(np.pi)

    return gen_error

def adversarial_generalization_error_overlaps_teacher(overlaps: OverlapSet, task: Task, data_model: AbstractDataModel):
    def integrand(xi, y):
        e = overlaps.m * overlaps.m / (data_model.rho * overlaps.q)
        w_0 = np.sqrt(data_model.rho*e) * xi
        V_0 = data_model.rho * (1-e)

        z_0 = erfc((-y * w_0) / np.sqrt(2*(task.tau**2 + V_0)))
        # z_out_0 and f_out_0 simplify together as the erfc cancels. See computation
        w = np.sqrt(overlaps.q) * xi

        epsilon_term = task.test_against_epsilon*overlaps.A/np.sqrt(overlaps.N)

        activation = np.sign(w - y*epsilon_term)

        proximal = proximal_logistic_root_scalar(overlaps.sigma,y,epsilon_term,w)

        teacher_activation = np.sign(proximal)

        return z_0 * gaussian(xi) * (activation != teacher_activation)

    int_lims = 20

    Iplus = quad(lambda xi: integrand(xi,1),-int_lims,int_lims,limit=500)[0]
    Iminus = quad(lambda xi: integrand(xi,-1),-int_lims,int_lims,limit=500)[0]
    return (Iplus + Iminus) * 0.5



def robustness_overlaps(m: float, q: float, rho: float, tau: float, epsilon_term: float, int_lims: float = 20.0):
    def integrand(xi, y):
        e = m * m / (rho * q)
        w_0 = np.sqrt(rho*e) * xi
        V_0 = rho * (1-e)

        z_0 = erfc((-y * w_0) / np.sqrt(2*(tau**2 + V_0)))
        # z_out_0 and f_out_0 simplify together as the erfc cancels. See computation
        w = np.sqrt(q) * xi


        attacked_activation = np.sign(w - y*epsilon_term)
        activation = np.sign(w)

        return z_0 * gaussian(xi) * (activation == attacked_activation)

    Iplus = quad(lambda xi: integrand(xi,1),-int_lims,int_lims,limit=500)[0]
    Iminus = quad(lambda xi: integrand(xi,-1),-int_lims,int_lims,limit=500)[0]
    return (Iplus + Iminus) * 0.5


def robustness_overlaps_teacher(overlaps, rho: float, tau: float, epsilon_term: float, int_lims: float = 20.0):
    def integrand(xi, y):
        e = overlaps.m * overlaps.m / (rho * overlaps.q)
        w_0 = np.sqrt(rho*e) * xi
        V_0 = rho * (1-e)

        z_0 = erfc((-y * w_0) / np.sqrt(2*(tau**2 + V_0)))
        # z_out_0 and f_out_0 simplify together as the erfc cancels. See computation
        w = np.sqrt(overlaps.q) * xi


        attacked_activation = np.sign(w - y*epsilon_term)
        

        proximal = proximal_logistic_root_scalar(overlaps.sigma,y,epsilon_term,w)

        activation = np.sign(proximal)

        return z_0 * gaussian(xi) * (y == attacked_activation)

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



def generalization_error(rho,m,q, tau):
    """
    Returns the generalization error in terms of the overlaps
    """
    return np.arccos(m / np.sqrt( (rho + tau**2 ) * q ) )/np.pi


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

        new_m, new_q, new_sigma, new_A, new_N = var_func(task, overlaps, my_data_model, logger)

        err = overlaps.update_overlaps(new_m, new_q, new_sigma, new_A, new_N)


        iter_nb += 1
        if iter_nb > overlaps.MAX_ITER_FPE:
            raise Exception("fixed_point_finder - reached max_iterations")
    return overlaps

