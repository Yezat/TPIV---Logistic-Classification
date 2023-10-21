import numpy as np
from typing import Tuple
from scipy.integrate import quad
from scipy.special import erfc
from util import *
from data_model import *
from scipy.optimize import root_scalar
import mygrad as mg

"""
Proximal from root scalar logistic
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

def log1pexp(x):
    """Compute log(1+exp(x)) componentwise."""
    # inspired from sklearn and https://cran.r-project.org/web/packages/Rmpfr/vignettes/log1mexp-note.pdf
    # and http://fa.bianp.net/blog/2019/evaluate_logistic/
    out = np.zeros_like(x)
    idx0 = x <= -37
    out[idx0] = np.exp(x[idx0])
    idx1 = (x > -37) & (x <= -2)
    out[idx1] = np.log1p(np.exp(x[idx1]))
    idx2 = (x > -2) & (x <= 18)
    out[idx2] = np.log(1. + np.exp(x[idx2]))
    idx3 = (x > 18) & (x <= 33.3)
    out[idx3] = np.exp(-x[idx3]) + x[idx3]
    idx4 = x > 33.3
    out[idx4] = x[idx4]
    return out

def stable_sigmoid(x):
    out = np.zeros_like(x)
    idx = x <= 0
    out[idx] = np.exp(x[idx]) / (1 + np.exp(x[idx]))
    idx = x > 0
    out[idx] = 1 / (1 + np.exp(-x[idx]))
    return out

def adversarial_loss(y,z, epsilon_term):
    return log1pexp(-y*z + epsilon_term)

def stable_cosh(x):
    out = np.zeros_like(x)
    idx = x <= 0
    out[idx] = np.exp(x[idx]) / (1 + np.exp(2*x[idx]))
    idx = x > 0
    out[idx] = np.exp(-x[idx]) / (1 + np.exp(-2*x[idx]))
    return out

def second_derivative_loss(y: float, z: float, epsilon_term: float) -> float:
    return y**2 * stable_cosh(0.5*y*z - 0.5*epsilon_term)**(2)

def gaussian(x : float, mean : float = 0, var : float = 1) -> float:
    '''
    Gaussian measure
    '''
    return np.exp(-.5 * (x-mean)**2 / var)/np.sqrt(2*np.pi*var)

def moreau_envelope(sigma: float, z:float, y: float, epsilon_term: float, w: float) -> float:
    return adversarial_loss(y,z,epsilon_term) + ((z-w)**2)/(2*sigma)

def moreau_derivative(sigma: float,y: float,epsilon_term: float,w: float) -> float:
    lim = 1
    evaluations = 10
    epsilon_range = np.linspace(epsilon_term -lim, epsilon_term+lim,evaluations)
    # get the index of the actual epsilon_term
    index = evaluations // 2
    zs = np.array([ proximal_logistic_root_scalar(sigma,y,e,w) for e in epsilon_range])
    values = moreau_envelope(sigma,zs,y,epsilon_range,w)
    gradients = np.gradient(values)
    return gradients[index]
    # eps_tensor = np.linspace(epsilon_term -lim, epsilon_term+lim,evaluations)
    # eps_tensor = mg.tensor([epsilon_term])
    # def f(eps):
    #     zs = np.array([ proximal_logistic_root_scalar(sigma,y,e,w) for e in eps])
    #     values = moreau_envelope(sigma,zs,y,eps,w)
    #     return values
    # k = f(eps_tensor)
    # k.backward()
    # return eps_tensor.grad[0]

    


"""
---------------------- Hat Equations ----------------------
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


        # m_derivative = moreau_derivative(sigma,y,epsilon*a/np.sqrt(n),w)
        # m_derivative *= epsilon*a/np.sqrt(n)


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

        # m_derivative = moreau_derivative(sigma,y,epsilon*a/np.sqrt(n),w)
        # m_derivative *= epsilon/np.sqrt(n)

        z_star = proximal_logistic_root_scalar(sigma,y,epsilon*a/np.sqrt(n),w)
        # m_derivative = moreau_envelope(sigma,z_star,y,epsilon*a/np.sqrt(n),w)
        
        arg = y*z_star - epsilon * a/np.sqrt(n)
        m_derivative = stable_sigmoid(-arg)
        
        # m_derivative = (stable_sigmoid(-y*z_star + epsilon * a / np.sqrt(n)) + (z_star-w)/sigma)

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

        # m_derivative = moreau_derivative(sigma,y,epsilon*a/np.sqrt(n),w)
        # m_derivative *= -0.5*epsilon*a/(n**(3/2))

        z_star = proximal_logistic_root_scalar(sigma,y,epsilon*a/np.sqrt(n),w)
        # m_derivative = moreau_envelope(sigma,z_star,y,epsilon*a/np.sqrt(n),w)
        

        arg = y*z_star - epsilon * a/np.sqrt(n)
        m_derivative = stable_sigmoid(-arg)

        # m_derivative = (stable_sigmoid(-y*z_star + epsilon * a / np.sqrt(n)) + (z_star-w)/sigma)

        m_derivative *= -0.5*epsilon*a/(n**(3/2))

        return z_0 * m_derivative * gaussian(xi)

    Iplus = quad(lambda xi: integrand(xi,1),-int_lims,int_lims,limit=500)[0]
    Iminus = quad(lambda xi: integrand(xi,-1),-int_lims,int_lims,limit=500)[0]

    return alpha * (Iplus + Iminus)

def var_hat_func(m, q, sigma, A, N,a,n, rho_w_star, alpha, epsilon, tau, lam, int_lims, gamma, logger=None):
    m_hat = m_hat_func(m, q, sigma, A, N,a,n,rho_w_star,alpha,epsilon,tau,lam,int_lims)/np.sqrt(gamma)
    q_hat = q_hat_func(m, q, sigma, A, N,a,n, rho_w_star,alpha,epsilon,tau,lam,int_lims)
    sigma_hat = sigma_hat_func(m, q, sigma, A, N,a,n,rho_w_star,alpha,epsilon,tau,lam,int_lims,logger=logger)
    A_hat = A_hat_func(m, q, sigma, A, N,a,n,rho_w_star,alpha,epsilon,tau,lam,int_lims)
    N_hat = N_hat_func(m, q, sigma, A, N,a,n,rho_w_star,alpha,epsilon,tau,lam,int_lims)
    a_hat = a_hat_func(m, q, sigma, A, N,a,n,rho_w_star,alpha,epsilon,tau,lam,int_lims)
    n_hat = n_hat_func(m, q, sigma, A, N,a,n,rho_w_star,alpha,epsilon,tau,lam,int_lims)
    return m_hat, q_hat, sigma_hat, A_hat, N_hat, a_hat, n_hat



"""
---------------------- Overlap Equations ----------------------
"""

def var_func(m_hat, q_hat, sigma_hat, A_hat, N_hat,a_hat, n_hat, rho, lam, data_model, logger):

    
    C = A_hat + a_hat
    M = N_hat + n_hat

    Lambda_clean = lam * data_model.spec_Sigma_w + sigma_hat * data_model.spec_Sigma_x 
    H_clean = data_model.spec_Sigma_x * q_hat + m_hat**2 * data_model.spec_PhiPhit 

    Lambda = lam * data_model.spec_Sigma_w + sigma_hat * data_model.spec_Sigma_x + C * data_model.spec_Sigma_delta + M * np.ones(data_model.d)
    H = data_model.spec_Sigma_x * q_hat + m_hat**2 * data_model.spec_PhiPhit + a_hat * data_model.spec_Sigma_delta + n_hat * np.ones(data_model.d)  

    
    sigma = np.mean(data_model.spec_Sigma_x/Lambda)       
    q = np.mean((H_clean * data_model.spec_Sigma_x) / Lambda**2) 
    m = m_hat/np.sqrt(data_model.gamma) * np.mean(data_model.spec_PhiPhit/Lambda)
    
    
    
    Lambda_a = lam * data_model.spec_Sigma_w + sigma_hat * data_model.spec_Sigma_x 
    H_a = data_model.spec_Sigma_delta * q_hat + m_hat**2 * data_model.spec_PhiPhit 
    
    a = np.mean( H_clean * data_model.spec_Sigma_delta / Lambda**2)
    A = a - np.mean( data_model.spec_Sigma_delta / Lambda)

    
    
    Lambda_n = lam * data_model.spec_Sigma_w + sigma_hat * data_model.spec_Sigma_x 
    H_n = data_model.spec_Sigma_x * q_hat + m_hat**2 * data_model.spec_PhiPhit 
    
    n = np.mean( H_clean * np.ones(data_model.d) / Lambda**2)
    N = n - np.mean( np.ones(data_model.d) / Lambda)


    return m, q, sigma, A, N, a, n


"""
---------------------- Observables ----------------------
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
    Q = q
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
    Q = q
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



def damped_update(new, old, damping):
    return damping * new + (1 - damping) * old

BLEND_FPE = 0.75
TOL_FPE = 1e-4
MIN_ITER_FPE = 10
MAX_ITER_FPE = 5000
INT_LIMS = 10.0
INITIAL_CONDITION = (0.1,0.1,0.5,0.1,0.1,0.1,0.1)


def fixed_point_finder(
    logger,
    my_data_model: AbstractDataModel,
    rho_w_star: float,
    alpha: float,
    epsilon: float,
    tau: float,
    lam: float,
    abs_tol: float = TOL_FPE,
    min_iter: int = MIN_ITER_FPE,
    max_iter: int = MAX_ITER_FPE,
    blend_fpe: float = BLEND_FPE,
    int_lims: float = INT_LIMS,
    initial_condition: Tuple[float, float, float] = INITIAL_CONDITION,
    log = True,

):
    rho_w_star = my_data_model.rho
    gamma = my_data_model.gamma
    m, q, sigma, A, N, a, n = initial_condition[0], initial_condition[1], initial_condition[2], initial_condition[3], initial_condition[4], initial_condition[5], initial_condition[6]
    err = 1.0
    iter_nb = 0
    m_hat = 0
    q_hat = 0
    sigma_hat = 0
    A_hat = 0
    N_hat = 0
    a_hat = 0
    n_hat = 0
    while err > abs_tol or iter_nb < min_iter:
        if iter_nb % 10 == 0 and log:
            logger.info(f"iter_nb: {iter_nb}, err: {err}")
            logger.info(f"m: {m}, q: {q}, sigma: {sigma}, A: {A}, N: {N}, a: {a}, n: {n}")
            logger.info(f"m_hat: {m_hat}, q_hat: {q_hat}, sigma_hat: {sigma_hat}, A_hat: {A_hat}, N_hat: {N_hat}, a_hat: {a_hat}, n_hat: {n_hat}")


        m_hat, q_hat, sigma_hat, A_hat, N_hat, a_hat, n_hat = var_hat_func(m, q, sigma, A, N, a, n, rho_w_star, alpha, epsilon, tau, lam, int_lims, gamma, logger=logger)

        new_m, new_q, new_sigma, new_A, new_N, new_a, new_n = var_func(m_hat, q_hat, sigma_hat, A_hat, N_hat,a_hat,n_hat, rho_w_star, lam, my_data_model, logger)


        n_m = damped_update(new_m, m, blend_fpe)
        n_q = damped_update(new_q, q, blend_fpe)
        n_sigma = damped_update(new_sigma, sigma, blend_fpe)
        n_A = damped_update(new_A, A, blend_fpe)
        n_N = damped_update(new_N, N, blend_fpe)
        n_a = damped_update(new_a, a, blend_fpe)
        n_n = damped_update(new_n, n, blend_fpe)


        err = max([abs(n_m - m), abs(n_q - q), abs(n_sigma - sigma), abs(n_A - A), abs(n_N - N), abs(n_a - a), abs(n_n - n)])
        m, q, sigma, A, N, a, n = n_m, n_q, n_sigma, n_A, n_N, n_a, n_n

        iter_nb += 1
        if iter_nb > max_iter:
            raise Exception("fixed_point_finder - reached max_iterations")
    return m, q, sigma, A, N, a, n, sigma_hat, q_hat, m_hat, A_hat, N_hat, a_hat, n_hat

