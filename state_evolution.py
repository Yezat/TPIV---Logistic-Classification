import numpy as np
from typing import Tuple
from scipy.integrate import quad, dblquad, tplquad
from scipy.optimize import fixed_point, root, root_scalar, minimize_scalar, minimize
from scipy.special import erfc,erf, logsumexp
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
from erm import *
from util import *
from data import *
import warnings
import logging
import time
from proximal import proximal_logistic_root_scalar


def gaussian(x : float, mean : float = 0, var : float = 1) -> float:
    '''
    Gaussian measure
    '''
    return np.exp(-.5 * (x-mean)**2 / var)/np.sqrt(2*np.pi*var)

def second_derivative_loss(y: float, z: float, Q: float, epsilon: float) -> float:
    return y**2 / (2 * np.cosh(y*z - epsilon * np.sqrt(Q)) + 2)
    
def derivative_proximal(V: float, y: float, z: float, Q: float, epsilon: float) -> float:
    return 1/(1 + V * second_derivative_loss(y,z,Q,epsilon))

def m_hat_func(m: float, q: float, sigma: float, rho_w_star: float, alpha: float, epsilon: float, tau:float, int_lims: float = 20.0):
    
    logging.info("m_hat_func")
    # print parameters
    logging.info("m: %s, q: %s, sigma: %s, rho_w_star: %s, alpha: %s, epsilon: %s, tau: %s, int_lims: %s", m, q, sigma, rho_w_star, alpha, epsilon, tau, int_lims)

    def integrand_plus(xi):
        e = m * m / (rho_w_star * q)
        w_0 = np.sqrt(rho_w_star*e) * xi
        V_0 = rho_w_star + (1-e)

        # z_out_0 and f_out_0 simplify together as the erfc cancels. See computation
        w = np.sqrt(q) * xi

        partial_prox =  proximal_logistic_root_scalar(sigma,1,sigma+q,epsilon,w) - w 

        return partial_prox * gaussian(w_0,0,V_0+tau**2) * gaussian(xi) + 1/np.sqrt( 2*np.pi*( V_0+tau**2))
    
    def integrand_minus(xi):
        e = m * m / (rho_w_star * q)
        w_0 = np.sqrt(rho_w_star*e) * xi
        V_0 = rho_w_star + (1-e)

        # z_out_0 and f_out_0 simplify together as the erfc cancels. See computation
        w = np.sqrt(q) * xi

        partial_prox =  proximal_logistic_root_scalar(sigma,-1,sigma+q,epsilon,w) - w 

        return partial_prox * gaussian(w_0,0,V_0+tau**2) * gaussian(xi) + 1/np.sqrt( 2*np.pi*( V_0+tau**2))

    Iplus = quad(lambda xi: integrand_plus(xi),-int_lims,int_lims,limit=500)[0]
    Iminus = quad(lambda xi: integrand_minus(xi),-int_lims,int_lims,limit=500)[0]
    return alpha / sigma * (Iplus - Iminus)

def q_hat_func(m: float, q: float, sigma: float, rho_w_star: float, alpha: float, epsilon: float, tau:float, int_lims: float = 20.0):
    logging.info("q_hat_func")
    def integrand_plus(xi):
        e = m * m / (rho_w_star * q)
        w_0 = np.sqrt(rho_w_star*e) * xi
        V_0 = rho_w_star + (1-e)

        
        z_0 = erfc(- (w_0) / np.sqrt(2*(tau**2 + V_0)))

        w = np.sqrt(q) * xi
        partial_proximal = ( proximal_logistic_root_scalar(sigma,1,sigma+q,epsilon,w) - w ) ** 2

        return z_0 * partial_proximal * gaussian(xi)
    
    def integrand_minus(xi):
        e = m * m / (rho_w_star * q)
        w_0 = np.sqrt(rho_w_star*e) * xi
        V_0 = rho_w_star + (1-e)

        z_0 = erfc((w_0) / np.sqrt(2*(tau**2 + V_0)))

        w = np.sqrt(q) * xi
        partial_proximal = ( proximal_logistic_root_scalar(sigma,-1,sigma+q,epsilon,w) - w ) ** 2

        return z_0 * partial_proximal * gaussian(xi)

    Iplus = quad(lambda xi: integrand_plus(xi),-int_lims,int_lims,limit=500)[0]
    Iminus = quad(lambda xi: integrand_minus(xi),-int_lims,int_lims,limit=500)[0]

    return alpha / (sigma ** 2)  * 0.5 * (Iplus + Iminus)

def sigma_hat_func(m: float, q: float, sigma: float, rho_w_star: float, alpha: float, tau: float, epsilon: float, int_lims: float = 20.0):
    logging.info("sigma_hat_func")
    def integrand_plus(xi):
        # todo: simplify
        
        z_0 = erfc( - ( m * np.sqrt(q) * xi) / np.sqrt(2*(tau**2 + (rho_w_star - m**2/q))))

        w = np.sqrt(q) * xi
        proximal = proximal_logistic_root_scalar(sigma,1,sigma+q,epsilon,w)

        derivative_proximal = 1/(1 + sigma * second_derivative_loss(1,proximal,sigma+1,epsilon))

        derivative_f_out =  1/sigma * (derivative_proximal - 1)

        return z_0 * ( derivative_f_out + epsilon * (1/np.sqrt(sigma+q))  ) * gaussian(xi)
    
    def integrand_minus(xi):
        z_0 = erfc(  ( m * np.sqrt(q) * xi) / np.sqrt(2*(tau**2 + (rho_w_star - m**2/q))))

        w = np.sqrt(q) * xi
        proximal = proximal_logistic_root_scalar(sigma,-1,sigma+q,epsilon,w)

        derivative_proximal = 1/(1 + sigma * second_derivative_loss(-1,proximal,sigma+1,epsilon))

        derivative_f_out =  1/sigma * (derivative_proximal - 1)

        return z_0 * ( derivative_f_out + epsilon * (1/np.sqrt(sigma+q))  ) * gaussian(xi)

    Iplus = quad(lambda xi: integrand_plus(xi),-int_lims,int_lims,limit=500)[0]
    Iminus = quad(lambda xi: integrand_minus(xi),-int_lims,int_lims,limit=500)[0]

    return -alpha * 0.5 * (Iplus + Iminus)

# m,q,sigma -> see application
def var_hat_func(m, q, sigma, rho_w_star, alpha, epsilon, tau, int_lims):
    logging.info("var_hat_func")
    logging.info("m: %s, q: %s, sigma: %s, rho_w_star: %s, alpha: %s, epsilon: %s, tau: %s, int_lims: %s", m, q, sigma, rho_w_star, alpha, epsilon, tau, int_lims)
    m_hat = m_hat_func(m, q, sigma,rho_w_star,alpha,epsilon,tau,int_lims)
    q_hat = q_hat_func(m, q, sigma, rho_w_star,alpha,epsilon,tau,int_lims)
    sigma_hat = sigma_hat_func(m, q, sigma,rho_w_star,alpha,tau,epsilon,int_lims)
    return m_hat, q_hat, sigma_hat

def var_func(m_hat, q_hat, sigma_hat, rho_w_star, lam):
    sigma = 1 / (lam + sigma_hat)
    q = (rho_w_star * m_hat**2 + q_hat) / (lam + sigma_hat) ** 2
    m = (rho_w_star * m_hat) / (lam + sigma_hat)
    return m, q, sigma

def damped_update(new, old, damping):
    return damping * new + (1 - damping) * old

BLEND_FPE = 0.75
TOL_FPE = 1e-6
MIN_ITER_FPE = 10
MAX_ITER_FPE = 5000
INT_LIMS = 20.0
INITIAL_CONDITION = (0.5,0.5,0.5)

def fixed_point_finder(
    logger,
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
):
    m, q, sigma = initial_condition[0], initial_condition[1], initial_condition[2]
    err = 1.0
    iter_nb = 0
    while err > abs_tol or iter_nb < min_iter:
        logger.info(f"iter_nb: {iter_nb}, err: {err}")

        m_hat, q_hat, sigma_hat = var_hat_func(m, q, sigma, rho_w_star, alpha, epsilon, tau, int_lims)

        new_m, new_q, new_sigma = var_func(m_hat, q_hat, sigma_hat, rho_w_star, lam)

        logger.info(f"m: {m}, q: {q}, sigma: {sigma}")

        err = max([abs(new_m - m), abs(new_q - q), abs(new_sigma - sigma)])

        m = damped_update(new_m, m, blend_fpe)
        q = damped_update(new_q, q, blend_fpe)
        sigma = damped_update(new_sigma, sigma, blend_fpe)

        iter_nb += 1
        if iter_nb > max_iter:
            raise Exception("fixed_point_finder - reached max_iterations")
    return m, q, sigma

if __name__ == "__main__":
    d = 300
    n = 600
    n_test = 100000
    w = sample_weights(d)
    tau = 0
    lam = 1
    epsilon = 0.0
    logging.basicConfig(level=logging.INFO)
    

    # V:  0.5 y:  -17.30126733377969 Q:  1.0 epsilon:  0 w:  4.764398807031843
    # proximal(0.5, -17.30126733377969, 1.0, 0, 4.764398807031843, debug=True)


    # y: -19.478130570343435 w:  164.80196123904906 V:  1.5 tau:  2
    # f_out_0(-19.478130570343435, 164.80196123904906, 1.5, 2)


    # y:  19.478130570343435 z:  [68.17374282] Q:  1.0 epsilon:  0.04 V:  0.5 w:  164.80196123904906
    # function_fixed_point(19.478130570343435, 68.17374282, 1.0, 0.04,0.5, 164.80196123904906)
    # proximal(0.5, 15.478130570343435, 1.0, 0.04, 164.80196123904906, debug=True)
    # y = 19.478130570343435
    # z = 68.17374282
    # Q = 1.0
    # epsilon = 0.04
    # V = 0.5
    # w = 164.80196123904906
    # argmin = np.inf
    # min = np.inf
    # for z in np.linspace(-100,100,100):
    #     argument = y*z - epsilon * np.sqrt(Q)
    #     print("argument", argument)
    #     e = np.exp(argument)
    #     print("e", e)
    #     r = y*V/(1+ np.exp(y*z - epsilon * np.sqrt(Q))) + w -z
    #     print("result", r)
    #     if r < min:
    #         min = r
    #         argmin = z

    # print("argmin", argmin)
    # print("min", min)

    start = time.time()
    m,q,sigma = fixed_point_finder(logging,rho_w_star=1,alpha=n/d,epsilon=epsilon,tau=tau,lam=lam,abs_tol=TOL_FPE,min_iter=MIN_ITER_FPE,max_iter=MAX_ITER_FPE,blend_fpe=BLEND_FPE,int_lims=INT_LIMS,initial_condition=INITIAL_CONDITION)
    print("m: ", m)
    print("q: ", q)
    print("sigma: ", sigma)
    print("Generalization error", generalization_error(1,m,sigma+q,tau))
    print("time", time.time() - start)

    
    # Xtrain, y = sample_training_data(w,d,n,tau)
    # Xtest = sample_test_data(d,n_test)  
    # clf = get_logistic_regressor(Xtrain,y,0)

    # f_erm = predict_proba_on_logistic_regressor(clf,Xtest)
    
    # fig,ax = plt.subplots()
    # plt_erm, = ax.plot(f_erm, label="$f_{erm}$")
    # ax.legend(handles=[plt_erm])
    # plt.title("$f_{erm}$")
    # plt.xlabel("Samples")
    # plt.ylabel("$f_{erm}$")
    # plt.show()