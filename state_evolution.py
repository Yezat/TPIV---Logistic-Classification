import numpy as np
from typing import Tuple
from scipy.integrate import quad, dblquad
from scipy.optimize import fixed_point, root
from scipy.special import erfc
import matplotlib.pyplot as plt
from erm import *
from util import *
from data import *

BLEND_FPE = 0.75
TOL_FPE = 1e-7
MIN_ITER_FPE = 100
MAX_ITER_FPE = 5000

def gaussian(x : float, mean : float = 0, var : float = 1) -> float:
    '''
    Gaussian measure
    '''
    return np.exp(-.5 * (x-mean)**2 / var)/np.sqrt(2*np.pi*var)

def p_out(y: float, z: float, Q: float, epsilon: float = 0.0) -> float:
    return 1 + np.exp(-y*z + epsilon * np.sqrt(Q))

def second_derivative_loss(y: float, z: float, Q: float, epsilon: float) -> float:
    return y**2 / (4 * np.cosh(y*z/2 - epsilon/2 * np.sqrt(Q)))

def derivative_proximal(V: float, y: float, z: float, Q: float, epsilon: float) -> float:
    return 1/(1 + V * second_derivative_loss(y,z,Q,epsilon))

def derivative_f_out(V: float, y: float, Q: float, epsilon: float, w: float) -> float:
    z = proximal(V,y,Q,epsilon,w)
    return 1/V * (derivative_proximal(V,y,z,Q,epsilon) - 1)

def function_fixed_point(y: float, z: float, Q: float, epsilon: float, V: float, w: float) -> float:
    return y*V/(1+ np.exp(y*z - epsilon * np.sqrt(Q))) + w -z

#https://en.wikipedia.org/wiki/Fixed-point_iteration
#https://math.stackexchange.com/questions/1683654/proximal-operator-for-the-logistic-loss-function
def proximal(V: float, y: float, Q: float, epsilon: float, w:float, debug = False) -> float:
    if debug:
        print("proximal:")
        print("V: ", V, "y: ", y, "Q: ", Q, "epsilon: ", epsilon, "w: ", w)
    z = 0.0
    # fixed_point(lambda z: function_fixed_point(y,z,Q,epsilon,V,w), z)
    result = root(lambda z: function_fixed_point(y,z,Q,epsilon,V,w), z)
    z = result.x[0]
    if debug:
        print("result: ", z)
    return z

def f_out(V: float, w: float, y: float, Q: float, epsilon: float) -> float:
    return 1/V * ( proximal(V,y,Q,epsilon,w) - w )

def f_out_0(y: float, w: float, V: float, tau: float) -> float:
    return 2*y * gaussian(w*y,0,V+tau**2) / (erfc(-w*y/np.sqrt(2*(V+tau**2))))

def z_out(y: float, mean: float, var: float, Q: float, epsilon: float = 0.0, int_lims: float = 20.0) -> float:
    def integrand(z):
        return p_out(y,z,Q,epsilon=epsilon) * gaussian(z,mean,var)
    
    return quad(integrand, -int_lims, int_lims, epsabs=1e-10, epsrel=1e-10, limit=200)[0]

def z_out_0(y: float, m: float, q: float, xi: float,tau: float, rho_w_star:float) -> float:
    return 0.5 * erfc( - (y * m * np.sqrt(q) * xi) / np.sqrt(2*(tau**2 + (rho_w_star - m**2/q))))

def eta(m: float, q: float, rho_w_star: float) -> float:
    return m * m / (rho_w_star * q)

def m_hat_func(m: float, q: float, sigma: float, rho_w_star: float, alpha: float, epsilon: float, tau:float, int_lims: float = 20.0):
    
    def integrand(y, xi):
        e = eta(m,q,rho_w_star)
        w_0 = np.sqrt(rho_w_star*e) * xi

        V_0 = rho_w_star + (1-e)

        z_0 = z_out_0(y,m,q,xi,tau,rho_w_star)

        f_0 = f_out_0(y,w_0,V_0,tau)

        f_o = f_out(sigma, np.sqrt(q) * xi, y, sigma+q,epsilon)

        return z_0 * f_0 * f_o * gaussian(xi)

    return alpha * dblquad(integrand,-np.inf,np.inf,-int_lims,int_lims,epsabs=1e-10,epsrel=1e-10)[0]

def q_hat_func(m: float, q: float, sigma: float, rho_w_star: float, alpha: float, epsilon: float, tau:float, int_lims: float = 20.0):
    
    def integrand(y, xi):

        z_0 = z_out_0(y,m,q,xi,tau,rho_w_star)
        f_o = f_out(sigma, np.sqrt(q) * xi, y, sigma+q,epsilon)

        return z_0 * f_o**2 * gaussian(xi)
        

    return alpha * dblquad(integrand,-np.inf,np.inf,-int_lims,int_lims,epsabs=1e-10,epsrel=1e-10)[0]

def sigma_hat_func(m: float, q: float, sigma: float, rho_w_star: float, alpha: float, tau: float, epsilon: float, int_lims: float = 20.0):
    
    def integrand(y, xi):
        z_0 = z_out_0(y,m,q,xi,tau,rho_w_star)

        f_o_prime = derivative_f_out(sigma,y,sigma+q,epsilon,np.sqrt(q) * xi)

        return z_0 * (f_o_prime + epsilon * (1/np.sqrt(sigma+q)) *(1 - np.sqrt(q)*xi) ) * gaussian(xi)
    
    return -alpha * dblquad(integrand,-np.inf,np.inf,-int_lims,int_lims,epsabs=1e-10,epsrel=1e-10)[0]

# m,q,sigma -> see application

def var_hat_func(m, q, sigma, rho_w_star, alpha, epsilon, tau, int_lims):
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

def fixed_point_finder(
    initial_condition: Tuple[float, float, float],
    rho_w_star: float,
    alpha: float,
    epsilon: float,
    tau: float,
    lam: float,
    int_lims: float = 20.0,
    abs_tol: float = TOL_FPE,
    min_iter: int = MIN_ITER_FPE,
    max_iter: int = MAX_ITER_FPE,
):
    m, q, sigma = initial_condition[0], initial_condition[1], initial_condition[2]
    err = 1.0
    iter_nb = 0
    while err > abs_tol or iter_nb < min_iter:
        print("iter_nb", iter_nb, "err", err)

        m_hat, q_hat, sigma_hat = var_hat_func(m, q, sigma, rho_w_star, alpha, epsilon, tau, int_lims)

        print("m_hat", m_hat, "q_hat", q_hat, "sigma_hat", sigma_hat)

        new_m, new_q, new_sigma = var_func(m_hat, q_hat, sigma_hat, rho_w_star, lam)

        print("new_m", new_m, "new_q", new_q, "new_sigma", new_sigma)

        err = max([abs(new_m - m), abs(new_q - q), abs(new_sigma - sigma)])

        if iter_nb % 100 == 0:
            print("\t", err)

        m = damped_update(new_m, m, BLEND_FPE)
        q = damped_update(new_q, q, BLEND_FPE)
        sigma = damped_update(new_sigma, sigma, BLEND_FPE)

        iter_nb += 1
        if iter_nb > max_iter:
            raise Exception("fixed_point_finder - reached max_iterations")

    return m, q, sigma



if __name__ == "__main__":
    d = 300
    n = 3000
    n_test = 100000
    w = sample_weights(d)
    p = 0.75
    dp = 0.1
    tau = 2
    lam = 0.01
    epsilon = 0
    

    # V:  0.5 y:  -17.30126733377969 Q:  1.0 epsilon:  0 w:  4.764398807031843
    # proximal(0.5, -17.30126733377969, 1.0, 0, 4.764398807031843, debug=True)


    m,q,sigma = fixed_point_finder((0.5,0.5,0.5),rho_w_star=1,alpha=n/d,epsilon=epsilon,tau=tau,lam=lam)

    print("m: ", m)
    print("q: ", q)
    print("sigma: ", sigma)
    print("Generalization error", generalization_error(1,m,sigma+q))
    
    
    
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