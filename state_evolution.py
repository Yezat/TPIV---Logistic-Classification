import numpy as np
from typing import Tuple
from scipy.integrate import quad, dblquad
from scipy.optimize import fixed_point, root, root_scalar, minimize_scalar, minimize
from scipy.special import erfc,erf, logsumexp
import matplotlib.pyplot as plt
from erm import *
from util import *
from data import *
import warnings
import logging
import time


def gaussian(x : float, mean : float = 0, var : float = 1) -> float:
    '''
    Gaussian measure
    '''
    return np.exp(-.5 * (x-mean)**2 / var)/np.sqrt(2*np.pi*var)

def p_out(y: float, z: float, Q: float, epsilon: float = 0.0) -> float:
    return 1 + np.exp(-y*z + epsilon * np.sqrt(Q))

def second_derivative_loss(y: float, z: float, Q: float, epsilon: float) -> float:
    return y**2 / (4 * np.cosh(y*z/2 - epsilon/2 * np.sqrt(Q)))
    
def first_derivative_loss(y: float, z: float, Q: float, epsilon: float) -> float:
    return -y / (1 + np.exp(y*z - epsilon * np.sqrt(Q)))

def derivative_proximal(V: float, y: float, z: float, Q: float, epsilon: float) -> float:
    return 1/(1 + V * second_derivative_loss(y,z,Q,epsilon))

def derivative_f_out(V: float, y: float, Q: float, epsilon: float, w: float) -> float:
    z = proximal(V,y,Q,epsilon,w)
    return 1/V * (derivative_proximal(V,y,z,Q,epsilon) - 1)

def function_fixed_point(y: float, z: float, Q: float, epsilon: float, V: float, w: float) -> float:
    return y*V/(1+ np.exp(y*z - epsilon * np.sqrt(Q))) + w
    
def proximal_loss(y: float, z: float, epsilon: float, Q: float, V: float, w: float) -> float:
    if type(z) == np.ndarray:
        z = z[0]
    return logsumexp([np.log(1),-y*z + epsilon * np.sqrt(Q)]) + V/2 * (z - w)**2

def fixed_point_iteration(y: float, z: float, Q: float, epsilon: float, V: float, w: float):
    z0 = 0
    z1 = function_fixed_point(y,z0,Q,epsilon,V,w)
    while np.abs(z1 - z0) > 1e-6:
        z0 = z1
        z1 = function_fixed_point(y,z0,Q,epsilon,V,w)
    return z1

#https://en.wikipedia.org/wiki/Fixed-point_iteration
#https://math.stackexchange.com/questions/1683654/proximal-operator-for-the-logistic-loss-function
#https://web.stanford.edu/~boyd/papers/pdf/prox_algs.pdf
# See chapter 6 on how to minimize the f(x) + (1/2) * V * (x - w)**2 (equation 6.1)
# Just before Chapter 3.4 there is an approximation given by w - V * f'(w)
def proximal(V: float, y: float, Q: float, epsilon: float, w:float) -> float:
    # logging.debug("proximal:")
    # logging.info("V: %s, y: %s, Q: %s, epsilon: %s, w: %s", V, y, Q, epsilon, w)
    z = -0.004

    # start  = time.time()
    # fixed_point(lambda z: function_fixed_point(y,z,Q,epsilon,V,w),z,maxiter=100000) #Failed to coverge after 500 iterations, we sometimes get unstable fixed points.
    # end = time.time()
    # logging.info("Fixed point iteration took %s seconds, result is %s", end - start, z)
    
    start = time.time()
    result = root(lambda z: y*V/(1+ np.exp(y*z - epsilon * np.sqrt(Q))) + w -z ,0) # one full iteration 150 seconds (got 0.07)
    z = result.x[0]
    end = time.time()
    # logging.info("Root finding took %s seconds", end - start)
        
    # result = root_scalar(lambda z: y*V/(1+ np.exp(y*z - epsilon * np.sqrt(Q))) + w - z, fprime= lambda z: (y**2)*V/(2 + 2*np.cosh(-y*z + epsilon * np.sqrt(Q))) - 1,x0=0,method="newton") # got a warning about the maximum number of subdivisions... Crazy slow, I stopped it
    # z = result.root 
    # z = fixed_point_iteration(y,z,Q,epsilon,V,w)  # insanely slow too, I stopped it

    # result = root(lambda z: proximal_loss(y,z,epsilon,Q,V,w), 0, method='hybr')
    # result = minimize_scalar(lambda z: proximal_loss(y,z,epsilon,Q,V,w),method="Brent")

    # result = minimize(lambda z: proximal_loss(y,z,epsilon,Q,V,w),0)
    

    # approximation:
    # z = w - V * first_derivative_loss(y,w,Q,epsilon)
    # logging.debug("result: %s", z)
    return z

def f_out(V: float, w: float, y: float, Q: float, epsilon: float) -> float:
    return 1/V * ( proximal(V,y,Q,epsilon,w) - w ) # directly write root.

def f_out_0(y: float, w: float, V: float, tau: float) -> float:
    with warnings.catch_warnings():
        warnings.filterwarnings('error')
        try:
            return 2*y * gaussian(w*y,0,V+tau**2) / (erfc(-w*y/np.sqrt(2*(V+tau**2))))
        except Warning:
            logging.warning("Warning in f_out_0")
            logging.warning("y: %s, w: %s, V: %s, tau: %s", y, w, V, tau)
            raise Exception("Warning in f_out_0")


def z_out(y: float, mean: float, var: float, Q: float, epsilon: float = 0.0, int_lims: float = 20.0) -> float:
    def integrand(z):
        return p_out(y,z,Q,epsilon=epsilon) * gaussian(z,mean,var)
    
    return quad(integrand, -int_lims, int_lims, epsabs=1e-10, epsrel=1e-10, limit=200)[0]

# def z_out_0(y: float, m: float, q: float, xi: float,tau: float, rho_w_star:float) -> float:
    # return 0.5 * erfc( - (y * m * np.sqrt(q) * xi) / np.sqrt(2*(tau**2 + (rho_w_star - m**2/q))))

# we write z_ou_0 as in (56) of the uncertainty paper
def z_out_0(y: float, w: float, V: float, tau: float) -> float:
    return 0.5 * erfc(- (y * w) / np.sqrt(2*(tau**2 + V)))

def eta(m: float, q: float, rho_w_star: float) -> float:
    return m * m / (rho_w_star * q)

def m_hat_func(m: float, q: float, sigma: float, rho_w_star: float, alpha: float, epsilon: float, tau:float, int_lims: float = 20.0):
    
    logging.info("m_hat_func")
    # print parameters
    logging.info("m: %s, q: %s, sigma: %s, rho_w_star: %s, alpha: %s, epsilon: %s, tau: %s, int_lims: %s", m, q, sigma, rho_w_star, alpha, epsilon, tau, int_lims)

    def integrand(y, xi):
        e = eta(m,q,rho_w_star)
        w_0 = np.sqrt(rho_w_star*e) * xi

        V_0 = rho_w_star + (1-e)

        # z_out_0 and f_out_0 simplify together as the erfc cancels. See computation

        f_0 = y * gaussian(w_0*y,0,V_0+tau**2)

        f_o = f_out(sigma, np.sqrt(q) * xi, y, sigma+q,epsilon)

        return f_0 * f_o * gaussian(xi)

    return alpha * dblquad(integrand,-np.inf,np.inf,-int_lims,int_lims,epsabs=1e-10,epsrel=1e-10)[0]

def q_hat_func(m: float, q: float, sigma: float, rho_w_star: float, alpha: float, epsilon: float, tau:float, int_lims: float = 20.0):
    logging.info("q_hat_func")
    def integrand(y, xi):

        e = eta(m,q,rho_w_star)
        w_0 = np.sqrt(rho_w_star*e) * xi
        V_0 = rho_w_star + (1-e)

        z_0 = z_out_0(y,w_0,V_0,tau)
        f_o = f_out(sigma, np.sqrt(q) * xi, y, sigma+q,epsilon)

        return z_0 * f_o**2 * gaussian(xi)
        

    return alpha * dblquad(integrand,-np.inf,np.inf,-int_lims,int_lims,epsabs=1e-10,epsrel=1e-10)[0]

def sigma_hat_func(m: float, q: float, sigma: float, rho_w_star: float, alpha: float, tau: float, epsilon: float, int_lims: float = 20.0):
    logging.info("sigma_hat_func")
    def integrand(y, xi):
        e = eta(m,q,rho_w_star)
        w_0 = np.sqrt(rho_w_star*e) * xi
        V_0 = rho_w_star + (1-e)

        z_0 = z_out_0(y,w_0,V_0,tau)

        f_o_prime = derivative_f_out(sigma,y,sigma+q,epsilon,np.sqrt(q) * xi)

        return z_0 * (f_o_prime + epsilon * (1/np.sqrt(sigma+q)) *(1 - np.sqrt(q)*xi) ) * gaussian(xi)
    
    return -alpha * dblquad(integrand,-np.inf,np.inf,-int_lims,int_lims,epsabs=1e-10,epsrel=1e-10)[0]

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
TOL_FPE = 1e-4
MIN_ITER_FPE = 20
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

        # logging.info("m_hat: %s, q_hat: %s, sigma_hat: %s", m_hat, q_hat, sigma_hat)

        new_m, new_q, new_sigma = var_func(m_hat, q_hat, sigma_hat, rho_w_star, lam)

        # logger.info(f"new_m: %s, new_q: %s, new_sigma: %s", new_m, new_q, new_sigma)
        logger.info(f"m: {m}, q: {q}, sigma: {sigma}")

        err = max([abs(new_m - m), abs(new_q - q), abs(new_sigma - sigma)])

        # if iter_nb % 100 == 0:
        #     logging.info("iter_nb: %s, err: %s", iter_nb, err)

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
    p = 0.75
    dp = 0.1
    tau = 2
    lam = 1e-5
    epsilon = 0.04
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
    print("Generalization error", generalization_error(1,m,sigma+q))
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