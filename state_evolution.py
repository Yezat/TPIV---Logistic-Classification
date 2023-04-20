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
from proximal import proximal_logistic_root_scalar, proximal_2_logistic_root_scalar

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

def loss(z):
    return log1pexp(-z)

def adversarial_loss(y,z, epsilon, Q):
    return log1pexp(-y*z + epsilon * np.sqrt(Q))

def gaussian(x : float, mean : float = 0, var : float = 1) -> float:
    '''
    Gaussian measure
    '''
    return np.exp(-.5 * (x-mean)**2 / var)/np.sqrt(2*np.pi*var)

def second_derivative_loss(y: float, z: float, Q: float, epsilon: float) -> float:
    return y**2 / (2 * np.cosh(0.5*y*z - 0.5*epsilon * np.sqrt(Q)))**2
    
def derivative_proximal(V: float, y: float, z: float, Q: float, epsilon: float) -> float:
    return 1/(1 + V * second_derivative_loss(y,z,Q,epsilon))

def m_hat_func(m: float, q: float, sigma: float, rho_w_star: float, alpha: float, epsilon: float, tau:float, lam: float, int_lims: float = 20.0):
    Q = q
    def integrand(xi, y):
        e = m * m / (rho_w_star * q)
        w_0 = np.sqrt(rho_w_star*e) * xi
        V_0 = rho_w_star * (1-e)

        # z_out_0 and f_out_0 simplify together as the erfc cancels. See computation
        w = np.sqrt(q) * xi

        partial_prox =  proximal_logistic_root_scalar(sigma,y,Q,epsilon,w) - w 

        return partial_prox * gaussian(w_0,0,V_0+tau**2) * gaussian(xi)

    Iplus = quad(lambda xi: integrand(xi,1),-int_lims,int_lims,limit=500)[0]
    Iminus = quad(lambda xi: integrand(xi,-1),-int_lims,int_lims,limit=500)[0]
    return alpha / sigma * (Iplus - Iminus)

def q_hat_func(m: float, q: float, sigma: float, rho_w_star: float, alpha: float, epsilon: float, tau:float, lam: float, int_lims: float = 20.0):
    Q = q
    def integrand(xi, y):
        e = m * m / (rho_w_star * q)
        w_0 = np.sqrt(rho_w_star*e) * xi
        V_0 = rho_w_star * (1-e)

        z_0 = erfc((-y * w_0) / np.sqrt(2*(tau**2 + V_0)))

        w = np.sqrt(q) * xi
        proximal = proximal_logistic_root_scalar(sigma,y,Q,epsilon,w)
        partial_proximal = ( proximal - w ) ** 2


        # approximating gaussian integrals with hermite polynomials ( does not work in q)
        # x, herm = np.polynomial.hermite.hermgauss(50)
        # const = np.pi**-0.5
        # translated_argument = np.sqrt(2 * sigma) * x + w

        # numerator = np.sum(herm * const * (1 + np.exp(- y * translated_argument + epsilon * np.sqrt(Q))) ** -2)
        # denominator = np.sum(herm * const * (1 + np.exp(- y * translated_argument + epsilon * np.sqrt(Q))) ** -1)

        # if numerator == 0.0:
        #     denominator = 1
        # else:
        #     assert denominator != 0, f"denominator is zero, numerator = {numerator}, sigma = {sigma}, y = {y}, Q = {Q}, epsilon = {epsilon}, w = {w}, xi = {xi}"

        # z_out_thing = numerator / denominator

        # epsilon_term = ( z_out_thing - 1 ) / (2 * np.sqrt(Q))
        # epsilon_term *= epsilon


        # the thing with the derivative of the moreau-yosida regularization ( a bit unstable apparently... but seems correct for high tau and lam...)
        # z_star = proximal
        # arg = y*z_star - epsilon * np.sqrt(Q)
        # if arg <= 0:
        #     numerator = z_star - w - y/(1 + np.exp(arg))
        # else:
        #     numerator = z_star - w - (y * np.exp(-arg))/(1 + np.exp(-arg))
        # denominator = 4 + 2 * np.cosh(y*z_star - epsilon * np.sqrt(Q))
        # epsilon_term = numerator / denominator
        # epsilon_term *= y * epsilon / np.sqrt(Q)

        z_star = proximal
        arg = y*z_star - epsilon * np.sqrt(Q)
        cosh = 4 + 4 *np.cosh(arg) # TODO: double check if there is a 4 in front of the cosh
        first = y*(w - z_star) / cosh
        if arg <= 0:
            second = sigma / ((1 + np.exp(arg)) * cosh)
        else:
            second = sigma * np.exp(-arg) / ((1 + np.exp(-arg)) * cosh)
        epsilon_term = (first + second) * epsilon / np.sqrt(Q)
       

        # Taking the zero temperature limit and then computing the derivative
        # z_star = proximal
        # arg = y*z_star - epsilon * np.sqrt(Q)
        # if arg <= 0:
        #     epsilon_term = -1 / (1 + np.exp(arg))
        # else:
        #     epsilon_term = -np.exp(-arg) / (1 + np.exp(-arg))
        # epsilon_term *= epsilon / (2 * np.sqrt(Q))
        # epsilon_term -= ( (z_star - w)**2 )/( 2 * sigma**2)
        # epsilon_term += xi * (z_star-w) / ( 2 * sigma * np.sqrt(Q) )

        # # taking zero temperature limit after derivative
        # z_star = proximal
        # arg = y*z_star - epsilon * np.sqrt(Q)
        # if arg <= 0:
        #     epsilon_term = -1 / (1 + np.exp(arg))
        # else:
        #     epsilon_term = -np.exp(-arg) / (1 + np.exp(-arg))
        # epsilon_term *= epsilon / (2 * np.sqrt(Q))     

        # the thing with the derivative of the moreau-yosida regularization (maybe this guy is wrong)
        # z_star = proximal
        # arg = y*z_star - epsilon * np.sqrt(Q)
        # easy = y*(z_star - w) / (4 + 2 * np.cosh(arg))
        # hard = 1 / ( 5 + 5*np.exp(arg) + np.exp(2*arg) + np.exp(-arg))
        # epsilon_term = (epsilon / np.sqrt(Q)) * (easy + hard)

        return z_0 * (partial_proximal/ (sigma ** 2) + epsilon_term ) * gaussian(xi)

    Iplus = quad(lambda xi: integrand(xi,1),-int_lims,int_lims,limit=500)[0]
    Iminus = quad(lambda xi: integrand(xi,-1),-int_lims,int_lims,limit=500)[0]

    return alpha * 0.5 * (Iplus + Iminus)

def sigma_hat_func(m: float, q: float, sigma: float, rho_w_star: float, alpha: float, epsilon: float, tau: float, lam: float, int_lims: float = 20.0):
    Q = q
    def integrand(xi, y):
        z_0 = erfc(  ( (-y * m) / np.sqrt(q) * xi) / np.sqrt(2*(tau**2 + (rho_w_star - m**2/q))))

        w = np.sqrt(q) * xi
        proximal = proximal_logistic_root_scalar(sigma,y,Q,epsilon,w)

        derivative_proximal = 1/(1 + sigma * second_derivative_loss(y,proximal,Q,epsilon))

        derivative_f_out =  1/sigma * (derivative_proximal -1)

        

        # this is the saddle_point approximation
        # proximal_2 = proximal_2_logistic_root_scalar(sigma,y,Q,epsilon,w)
        # denominator =  np.sqrt(1 + 2*sigma* second_derivative_loss(y,proximal_2,Q,epsilon)  )
        # numerator = np.sqrt(1 + sigma* second_derivative_loss(y,proximal,Q,epsilon))        
        # arg_g = - ( ( (proximal_2-w)**2)/(2*sigma) + 2 * adversarial_loss(y,proximal_2,epsilon,Q) )
        # arg_f = ( ( (proximal-w)**2)/(2*sigma) + adversarial_loss(y,proximal,epsilon,Q) )
        # arg = arg_g + arg_f
        # z_out_thing = np.exp(arg) * numerator / denominator

        


        return z_0 * ( derivative_f_out ) * gaussian(xi)

    Iplus = quad(lambda xi: integrand(xi,1),-int_lims,int_lims,limit=500)[0]
    Iminus = quad(lambda xi: integrand(xi,-1),-int_lims,int_lims,limit=500)[0]

    return -alpha * 0.5 * (Iplus + Iminus)



def training_error_logistic(m: float, q: float, sigma: float, rho_w_star: float, alpha: float, tau: float, epsilon: float, lam: float , int_lims: float = 20.0):
    Q = q
    def integrand(xi,y):
        w = np.sqrt(q) * xi
        z_0 = erfc(  ( (-y * m * xi) / np.sqrt(q) ) / np.sqrt(2*(tau**2 + (rho_w_star - m**2/q))))
        
        proximal = proximal_logistic_root_scalar(sigma,y,Q,epsilon,w)
        
        l = adversarial_loss(y,proximal, epsilon, Q)

        return z_0 * l * gaussian(xi)


    I1 = quad(lambda ξ: integrand(ξ,1) , -int_lims, int_lims, limit=500)[0]
    I2 = quad(lambda ξ: integrand(ξ,-1) , -int_lims, int_lims, limit=500)[0]
    return (I1 + I2)/2 + (lam/(2*alpha)) * q



# m,q,sigma -> see application
def var_hat_func(m, q, sigma, rho_w_star, alpha, epsilon, tau, lam, int_lims):
    # logging.info("var_hat_func")
    # logging.info("m: %s, q: %s, sigma: %s, rho_w_star: %s, alpha: %s, epsilon: %s, tau: %s, int_lims: %s", m, q, sigma, rho_w_star, alpha, epsilon, tau, int_lims)
    m_hat = m_hat_func(m, q, sigma,rho_w_star,alpha,epsilon,tau,lam,int_lims)
    q_hat = q_hat_func(m, q, sigma, rho_w_star,alpha,epsilon,tau,lam,int_lims)
    sigma_hat = sigma_hat_func(m, q, sigma,rho_w_star,alpha,epsilon,tau,lam,int_lims)
    return m_hat, q_hat, sigma_hat

def var_func(m_hat, q_hat, sigma_hat, rho_w_star, lam):
    sigma = 1 / (lam + sigma_hat)
    q = (rho_w_star * m_hat**2 + q_hat) / (lam + sigma_hat)** 2
    m = (rho_w_star * m_hat) / (lam + sigma_hat)
    return m, q, sigma

def damped_update(new, old, damping):
    return damping * new + (1 - damping) * old

BLEND_FPE = 0.75
TOL_FPE = 1e-4
MIN_ITER_FPE = 10
MAX_ITER_FPE = 5000
INT_LIMS = 10.0
INITIAL_CONDITION = (0.1,0.1,0.9)

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
        if iter_nb % 10 == 0:
            logger.info(f"iter_nb: {iter_nb}, err: {err}")
            logger.info(f"m: {m}, q: {q}, sigma: {sigma}")

        m_hat, q_hat, sigma_hat = var_hat_func(m, q, sigma, rho_w_star, alpha, epsilon, tau, lam, int_lims)

        new_m, new_q, new_sigma = var_func(m_hat, q_hat, sigma_hat, rho_w_star, lam)

        
        n_m = damped_update(new_m, m, blend_fpe)
        n_q = damped_update(new_q, q, blend_fpe)
        n_sigma = damped_update(new_sigma, sigma, blend_fpe)
        

        err = max([abs(n_m - m), abs(n_q - q), abs(n_sigma - sigma)])        
        m, q, sigma = n_m, n_q, n_sigma

        iter_nb += 1
        if iter_nb > max_iter:
            raise Exception("fixed_point_finder - reached max_iterations")
    return m, q, sigma

if __name__ == "__main__":
    d = 1500
    alpha = 3.0
    n = 1000
    n_test = 100000
    w = sample_weights(d)
    tau = 2
    lam = 1
    epsilon = 0.7
    logging.basicConfig(level=logging.INFO)

    # alpha=1.0, epsilon=0.0, lambda=1e-05, tau=2, d=1000, gen_error=nan
    

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
    print("Generalization error", generalization_error(1,m,q,tau))
    print("Training error",training_error_logistic(m,q,sigma,1,n/d,tau,epsilon, lam) )
    print("time", time.time() - start)

    

    # start = time.time()

    # # generate ground truth
    # w = sample_weights(d)

    # # generate data
    # Xtrain, y = sample_training_data(w,d,int(alpha * d),tau)
    # n_test = 100000
    # Xtest,ytest = sample_training_data(w,d,n_test,tau)

    # w_gd = np.empty(w.shape,dtype=w.dtype)

    # from experiment_information import *
    # from gradient_descent import gd, lbfgs, sklearn_optimize
    # w_gd = sklearn_optimize(sample_weights(d),Xtrain,y,lam,epsilon)


    # end = time.time()
    # duration = end - start
    # erm_information = ERMExperimentInformation("blabla",duration,Xtest,w_gd,tau,y,Xtrain,w,ytest,d,"sklearn",epsilon,lam)
    # print("erm m", erm_information.m)
    # print("erm q", erm_information.Q)    
    # print("erm generalization error", erm_information.generalization_error_erm)
    # print("erm training error", erm_information.training_loss)

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