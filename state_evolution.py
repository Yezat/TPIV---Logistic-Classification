import numpy as np
from typing import Tuple
from scipy.integrate import quad, fixed_quad, romberg
from scipy.special import erfc
from erm import *
from util import *
from data import *
import logging
from scipy.optimize import root_scalar
import time

"""
Proximal from root scalar logistic
"""
def optim(z,y,V,w_prime):
    a = y*z
    if a <= 0:
        return y*V/(1+ np.exp(y*z)) + w_prime - z
    else:
        return y*V*np.exp(-y*z)/(1+ np.exp(-y*z)) + w_prime - z

def proximal_logistic_root_scalar(V: float, y: float, Q: float, epsilon: float, w:float) -> float:
    if y == 0:
        return w
    try:
        w_prime = w - epsilon * np.sqrt(Q) / y
        result = root_scalar(lambda z: optim(z,y,V,w_prime) , bracket=[-50000000,50000000],xtol=10e-10,rtol=10e-10) 
        z = result.root
        return z + epsilon * np.sqrt(Q) / y
    except Exception as e:
        # print all parameters
        print("V: ", V, "y: ", y, "Q: ", Q, "epsilon: ", epsilon, "w: ", w)        
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

def adversarial_loss(y,z, epsilon, Q):
    return log1pexp(-y*z + epsilon * np.sqrt(Q))

def gaussian(x : float, mean : float = 0, var : float = 1) -> float:
    '''
    Gaussian measure
    '''
    return np.exp(-.5 * (x-mean)**2 / var)/np.sqrt(2*np.pi*var)

def second_derivative_loss(y: float, z: float, Q: float, epsilon: float) -> float:
    return y**2 / (2 * np.cosh(0.5*y*z - 0.5*epsilon * np.sqrt(Q)))**2

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

        z_star = proximal
        arg = y*z_star - epsilon * np.sqrt(Q)
        cosh = 4 + 4 *np.cosh(arg) 
        # cosh /= sigma # TODO: go in calmth trough derivation again and then fix to whatever turns out to be right.
        first = y*(w - z_star) / ( cosh)
        if arg <= 0:
            second = sigma / ((1 + np.exp(arg)) * cosh)
        else:
            second = sigma * np.exp(-arg) / ((1 + np.exp(-arg)) * cosh)
        epsilon_term = (first + second) * epsilon / np.sqrt(Q)
       

        return z_0 * (partial_proximal/ (sigma ** 2) + epsilon_term ) * gaussian(xi)

    Iplus = quad(lambda xi: integrand(xi,1),-int_lims,int_lims,limit=500)[0]
    Iminus = quad(lambda xi: integrand(xi,-1),-int_lims,int_lims,limit=500)[0]

    return alpha * 0.5 * (Iplus + Iminus)

def sigma_hat_func(m: float, q: float, sigma: float, rho_w_star: float, alpha: float, epsilon: float, tau: float, lam: float, int_lims: float = 20.0):
    Q = q

    """
    Original derivative of f_out (needs a minus in front of alpha * Iplus...)
    """
    def get_derivative_f_out(xi,y):
        w = np.sqrt(q) * xi
        proximal = proximal_logistic_root_scalar(sigma,y,Q,epsilon,w)

        derivative_proximal = 1/(1 + sigma * second_derivative_loss(y,proximal,Q,epsilon))

        derivative_f_out =  1/sigma * (derivative_proximal -1)       
        return derivative_f_out
    
    """
    Alternative derivative of f_out
    """
    def alternative_derivative_f_out(xi,y):
        w = np.sqrt(q) * xi
        proximal = proximal_logistic_root_scalar(sigma,y,Q,epsilon,w)

        second_derivative = second_derivative_loss(y,proximal,Q,epsilon)

        return second_derivative / ( 1 + sigma * second_derivative)

    def integrand(xi, y):
        z_0 = erfc(  ( (-y * m) / np.sqrt(q) * xi) / np.sqrt(2*(tau**2 + (rho_w_star - m**2/q))))

        # derivative_f_out = get_derivative_f_out(xi,y)

        derivative_f_out = alternative_derivative_f_out(xi,y)

        return z_0 * ( derivative_f_out ) * gaussian(xi)
    
    # Idea:
    # I suspect the second_derivative_loss to concentrate around the proximal = y*epsilon*sqrt(Q) 
    # So we now around what value as a function of the proximal the integrand will concentrate to
    # The proximal depends on xi as a function of w = xi * sqrt(q)
    # hence we can find the value of xi where the proximal is equal to y*epsilon*sqrt(Q)
    # this should be doable by root finding
    # it has to be seen whether we need to dynamically reduce the limits around this point as long as the integral is zero...

    def find_concentration_point(y):
        try:
            # find the point where the proximal is equal to y*epsilon*sqrt(Q) 
            concentration_point = root_scalar(lambda xi: proximal_logistic_root_scalar(sigma,y,Q,epsilon,np.sqrt(q)*xi) - y*epsilon*np.sqrt(Q),bracket=[-1,1],method='brentq')
                
            return concentration_point.root
        except:
            # if the root finding fails, just return None
            return None
    
    """
    Reduce limit around concentration point (does not work in large alpha currently...)
    """

    # def reduce_limit(y):
    #     concentration_point = find_concentration_point(y)
    #     # print the concentration point
    #     # print("concentration point: ", concentration_point, " for y = ", y)
    #     if concentration_point is None:
    #         return quad(lambda xi: integrand(xi,y),-int_lims,int_lims,limit=500)[0]

    #     # print the integrand at the concentration point

    #     min_integral = integrand(concentration_point,y)
    #     # print("integrand at concentration point: ", min_integral, " for y = ", y)

    #     # if min_integral < lam:
    #     #     return quad(lambda xi: integrand(xi,y),-int_lims,int_lims,limit=500)[0]

    #     lim = int_lims

    #     # compute the integral at the concentration point
    #     I = quad(lambda xi: integrand(xi,y),concentration_point-lim,concentration_point+lim,limit=500)[0]
    #     # print the integral
    #     # print("integral at concentration point: ", I, " for y = ", y)
    #     # if the integral is zero, reduce the limits
                
    #     max_iter = 100
    #     c = 0
    #     while I < min_integral and c < max_iter:
    #         lim *= 0.5
    #         c += 1
    #         I = quad(lambda xi: integrand(xi,y),concentration_point-lim,concentration_point+lim,limit=500)[0]
    #         # print the integral with very high precision
    #         # print("integral at concentration point: ", I, " for y = ", y, " with lim = ", lim, " and min_integral = ", min_integral, " concentration point: ", concentration_point)
            
    #     if c == max_iter and I < min_integral:
    #         I = min_integral
            
    #     # print the final integral
    #     # print("final integral at concentration point: ", I, " for y = ", y, " with lim = ", lim, " and min_integral = ", min_integral, " concentration point: ", concentration_point)

    #     # print the limits
    #     # print("limits: ", concentration_point-lim, concentration_point+lim, " for y = ", y)
    #     return I


    
    """
    Reduce limit from earlier version
    """

    def reduce_limit(y):
        step = 1
        left_lim = -int_lims
        while get_derivative_f_out(left_lim+step,y) == 0 and np.abs(left_lim - y) > 1.5*step:
            left_lim += step
            if np.abs(left_lim) <= 2:
                step = 0.1

        step = 1
        right_lim = int_lims
        while get_derivative_f_out(right_lim-step,y) == 0 and np.abs(y-right_lim) > 1.5*step:
            right_lim -= step 
            if np.abs(right_lim) <= 2:
                step = 0.1
        return left_lim, right_lim

    left_lim_plus, right_lim_plus = reduce_limit(1)
    left_lim_minus, right_lim_minus = reduce_limit(-1)

    Iplus = quad(lambda xi: integrand(xi,1),left_lim_plus,right_lim_plus,limit=500)[0]
    Iminus = quad(lambda xi: integrand(xi,-1),left_lim_minus,right_lim_minus,limit=500)[0]

    """
    Optional reduce limit improvement by using the concentration point
    """
    if Iplus + Iminus < 1e-15:
        concentration_point = find_concentration_point(1)
        Iplus = quad(lambda xi: integrand(xi,1),concentration_point-0.1,concentration_point+0.1,limit=500)[0]
        concentration_point = find_concentration_point(-1)
        Iminus = quad(lambda xi: integrand(xi,-1),concentration_point-0.1,concentration_point+0.1,limit=500)[0]

        concentration_point = find_concentration_point(1)
        Iplus = integrand(concentration_point,1)
        Iplus_quad = quad(lambda xi: integrand(xi,1),concentration_point-0.1,concentration_point+0.1,limit=500)[0]
        print("concentration point: ", concentration_point)
        print("Iplus: ", Iplus)        
        print("Iplus_quad: ", Iplus_quad)

        for lim in np.linspace(0.01,1,10):
            Iplus = quad(lambda xi: integrand(xi,1),concentration_point-lim,concentration_point+lim,limit=500)[0]
            print("Iplus: ", Iplus, " for lim = ", lim)
        
        concentration_point = find_concentration_point(-1)
        Iminus = integrand(concentration_point,-1)
        Iminus_quad = quad(lambda xi: integrand(xi,-1),concentration_point-0.1,concentration_point+0.1,limit=500)[0]
        print("concentration point: ", concentration_point)
        print("Iminus: ", Iminus)
        
        print("Iminus_quad: ", Iminus_quad)

        for lim in np.linspace(0.01,1,10):
            Iminus = quad(lambda xi: integrand(xi,-1),concentration_point-lim,concentration_point+lim,limit=500)[0]
            print("Iminus: ", Iminus, " for lim = ", lim)

        raise Exception("Iplus + Iminus < 1e-15")

    """
    Full int_lims
    """
    # Iplus = quad(lambda xi: integrand(xi,1) , -int_lims, int_lims, limit=500)[0]
    # Iminus = quad(lambda xi: integrand(xi,-1) , -int_lims, int_lims, limit=500)[0]

    return alpha * 0.5 * (Iplus + Iminus)



def training_loss_logistic(m: float, q: float, sigma: float, rho_w_star: float, alpha: float, tau: float, epsilon: float, lam: float , int_lims: float = 20.0):
    Q = q
    def integrand(xi,y):
        w = np.sqrt(q) * xi
        z_0 = erfc(  ( (-y * m * xi) / np.sqrt(q) ) / np.sqrt(2*(tau**2 + (rho_w_star - m**2/q))))
        
        proximal = proximal_logistic_root_scalar(sigma,y,Q,epsilon,w)
        
        l = adversarial_loss(y,proximal, epsilon, Q)

        return z_0 * l * gaussian(xi)


    I1 = quad(lambda xi: integrand(xi,1) , -int_lims, int_lims, limit=500)[0]
    I2 = quad(lambda xi: integrand(xi,-1) , -int_lims, int_lims, limit=500)[0]
    return (I1 + I2)/2 + (lam/(2*alpha)) * q

def training_error_logistic(m: float, q: float, sigma: float, rho_w_star: float, alpha: float, tau: float, epsilon: float, lam: float, int_lims: float = 20.0):
    Q = q
    def integrand(xi, y):
        e = m * m / (rho_w_star * q)
        w_0 = np.sqrt(rho_w_star*e) * xi
        V_0 = rho_w_star * (1-e)

        z_0 = erfc((-y * w_0) / np.sqrt(2*(tau**2 + V_0)))

        # z_out_0 and f_out_0 simplify together as the erfc cancels. See computation
        w = np.sqrt(q) * xi

        proximal = proximal_logistic_root_scalar(sigma,y,Q,epsilon,w)

        activation = np.sign(proximal)       

        return z_0* gaussian(xi) * (activation != y)

    Iplus = quad(lambda xi: integrand(xi,1),-int_lims,int_lims,limit=500)[0]
    Iminus = quad(lambda xi: integrand(xi,-1),-int_lims,int_lims,limit=500)[0]
    return (Iplus + Iminus) * 0.5

def adversarial_generalization_error_logistic(m: float, q: float, rho_w_star: float, tau: float, epsilon: float, int_lims: float = 20.0):
    Q = q
    def integrand(xi, y):
        e = m * m / (rho_w_star * q)
        w_0 = np.sqrt(rho_w_star*e) * xi
        V_0 = rho_w_star * (1-e)

        z_0 = erfc((-y * w_0) / np.sqrt(2*(tau**2 + V_0)))
        # z_out_0 and f_out_0 simplify together as the erfc cancels. See computation
        w = np.sqrt(q) * xi

        

        activation = np.sign(w - y*epsilon * np.sqrt(q))
        # 
        

        return z_0 * gaussian(xi) * (activation != y)

    Iplus = quad(lambda xi: integrand(xi,1),-int_lims,int_lims,limit=500)[0]
    Iminus = quad(lambda xi: integrand(xi,-1),-int_lims,int_lims,limit=500)[0]
    return (Iplus + Iminus) * 0.5




def var_hat_func(m, q, sigma, rho_w_star, alpha, epsilon, tau, lam, int_lims):
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
    log = True,
):
    m, q, sigma = initial_condition[0], initial_condition[1], initial_condition[2]
    err = 1.0
    iter_nb = 0
    m_hat = 0
    q_hat = 0
    sigma_hat = 0
    while err > abs_tol or iter_nb < min_iter:
        if iter_nb % 1 == 0 and log:
            logger.info(f"iter_nb: {iter_nb}, err: {err}")
            logger.info(f"m: {m}, q: {q}, sigma: {sigma}")
            logger.info(f"m_hat: {m_hat}, q_hat: {q_hat}, sigma_hat: {sigma_hat}")

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
    return m, q, sigma, sigma_hat, q_hat, m_hat

if __name__ == "__main__":
    d = 1000
    alpha = 1000
    n = int(alpha * d)
    n_test = 100000
    w = sample_weights(d)
    tau = 0.1
    lam = 0.01
    epsilon = 1.0 # or 0.5 don't work...
    logging.basicConfig(level=logging.INFO)

    # TODO: this should work...
    # 0.41142029787528417, epsilon=0.75, lambda=0.01, tau = 0.1

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


    # INFO:root:m: 1064.6272005520711, q: 1147638.4814880975, sigma: 13.409680485796514
    # INFO:root:m_hat: 79.39244361550637, q_hat: 54.10770407302894, sigma_hat: 0.06230981214041529
    # m = 1064.6272005520711
    # q = 1147638.4814880975
    # sigma = 13.409680485796514
    # sigma_hat = -0.0
    # sigma_hat = sigma_hat_func(m,q,sigma,1,alpha,epsilon,tau,lam,10)

    # print("sigma_hat", sigma_hat)

    # def integrand(xi, y, eps):
    #     z_0 = erfc(  ( (-y * m) / np.sqrt(q) * xi) / np.sqrt(2*(tau**2 + (1 - m**2/q))))

    #     w = np.sqrt(q) * xi
    #     proximal = proximal_logistic_root_scalar(sigma,y,q,eps,w)

    #     derivative_proximal = 1/(1 + sigma * second_derivative_loss(y,proximal,q,eps))

    #     derivative_f_out =  1/sigma * (derivative_proximal -1)       

    #     return z_0 * ( derivative_f_out ) * gaussian(xi)

    
    
    # y = 1
    # print("y", y)
    
    # for xi in np.linspace(-2,2,100):
    #     for epsilon in [0.75]:
    #         # print("epsilon", epsilon)            
    #         # print("integrand", integrand(xi,y,epsilon))
    #         z_0 = erfc(  ( (-y * m) / np.sqrt(q) * xi) / np.sqrt(2*(tau**2 + (1 - m**2/q))))
    #         # print("z_0", z_0)
    #         # print("gaussian xi", gaussian(xi))
    #         w = np.sqrt(q) * xi
    #         proximal = proximal_logistic_root_scalar(sigma,y,q,epsilon,w)

    #         derivative_proximal = 1/(1 + sigma * second_derivative_loss(y,proximal,q,epsilon))

    #         derivative_f_out =  1/sigma * (derivative_proximal -1) 
    #         if derivative_f_out != 0:
    #             print("derivative f_out non zero at xi", xi)
    #             print("derivative_f_out", derivative_f_out)


    # for epsilon in [0.69,0.7]:
    #     print("epsilon", epsilon)
    #     I = quad(lambda xi: integrand(xi,y,epsilon),-1,1,limit=500)[0]
    #     print("I", I)
    # I = quad(lambda xi: integrand(xi,1,epsilon),-INT_LIMS,INT_LIMS,limit=500)[0]
    # print("I", I)

    # m_hat, q_hat, sigma_hat = var_hat_func(m, q, sigma, 1, alpha, epsilon, tau, lam, INT_LIMS)
    # print("m_hat", m_hat)
    # print("q_hat", q_hat)
    # print("sigma_hat", sigma_hat)

    start = time.time()
    m,q,sigma, sigma_hat, q_hat, m_hat = fixed_point_finder(logging,rho_w_star=1,alpha=n/d,epsilon=epsilon,tau=tau,lam=lam,abs_tol=TOL_FPE,min_iter=MIN_ITER_FPE,max_iter=MAX_ITER_FPE,blend_fpe=BLEND_FPE,int_lims=INT_LIMS,initial_condition=INITIAL_CONDITION)
    print("m: ", m)
    print("q: ", q)
    print("sigma: ", sigma)
    print("Generalization error", generalization_error(1,m,q,tau))
    print("Training error",training_loss_logistic(m,q,sigma,1,n/d,tau,epsilon, lam) )
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