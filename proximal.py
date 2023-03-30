import numpy as np
from scipy.optimize import minimize, root, fixed_point, root_scalar, minimize_scalar
import time
import math

#https://en.wikipedia.org/wiki/Fixed-point_iteration
#https://math.stackexchange.com/questions/1683654/proximal-operator-for-the-logistic-loss-function
#https://web.stanford.edu/~boyd/papers/pdf/prox_algs.pdf

"""
Proximal from paper
"""
# http://proximity-operator.net/code/matlab/scalar/prox_logit.m
# http://proximity-operator.net/code/matlab/scalar/prox_entropy_symm.m
# http://proximity-operator.net/code/matlab/scalar/fun_logit.m
# TODO: look at the link above and the paper referenced on tha page of the proximal operator of the logit function
# In particular, I wonder what is happening for negative values of x. 

def prox_entropy_symm(x, gamma):
    # This procedure computes the proximity operator of the function:
    # f(x) = gamma * ( x * log(x) + (1-x) * log(1-x) )
    # When the input 'x' is an array, the output 'p' is computed element-wise.
    # INPUTS
    # x - ND array
    # gamma - positive, scalar or ND array with the same size as 'x'
    
    # check input
    if np.any(gamma <= 0) or (not np.isscalar(gamma) and np.any(np.shape(gamma) != np.shape(x))):
        raise ValueError("'gamma' must be positive and either scalar or the same size as 'x'")
    
    limit = 8
    w = np.zeros(np.shape(x))
    igamma = 1./gamma
    c = x*igamma - np.log(gamma)
    z = np.exp(c)
    r = np.exp(x*igamma)
    
    # ASYMPTOTIC APPROX
    approx = 1 - np.exp((1-x)*igamma)
    
    # INITIALIZATION
    # Case 1: gamma <= 1/30
    w[(z>1) & (gamma <= 1/30)] = c[(z>1) & (gamma <= 1/30)] - np.log(c[(z>1) & (gamma <= 1/30)])
    
    # Case 2: gamma > 1/30
    w[(z>1) & (gamma > 1/30)] = igamma*approx[(z>1) & (gamma > 1/30)]
    
    # RUN HALEY'S METHOD FOR SOLVING w = W_{exp(x/gamma)}(exp(x/gamma)/gamma)
    # where W_r(x) is the generalized Lambert function
    maxiter = 20
    testend = np.zeros(np.shape(x))
    prec = 1e-8
    
    for it in range(maxiter):
        e = np.exp(w)
        y = w*e + r*w - z
        v = e*(1 + w) + r
        u = e*(2 + w)
        wnew = w - y/(v - y*u/(2*v))
        testend[(np.abs(wnew-w)/np.abs(w) < prec) & (testend == 0)] = 1
        idx_update = np.where((np.abs(wnew-w)/np.abs(w) >= prec) & (testend == 0))
        w[idx_update] = wnew[idx_update]
        
        if np.sum(testend)==len(w):
            break
    
    p = gamma*w
    
    # ASYMPTOTIC DVP
    p[(c>limit) & (gamma > 1)] = approx[(c>limit) & (gamma > 1)]
    
    # FINAL TRESHOLD TO AVOID NUMERICAL ISSUES FOR SMALL GAMMA
    p = np.minimum(p,1)
    
    return p


def prox_logit(x, gamma):
    """
    This procedure computes the proximity operator of the function:
    f(x) = gamma * log( 1 + exp(x) )
    When the input 'x' is an array, the output 'p' is computed element-wise.
    """
    # check input
    if not np.isscalar(gamma) or gamma <= 0:
        raise ValueError("'gamma' must be positive and either scalar or the same size as 'x'")
    
    # Use Moreau's decomposition formula
    p = x - gamma * prox_entropy_symm(x / gamma, 1 / gamma)
    
    return p


def proximal_from_paper(V: float, y: float, Q: float, epsilon: float, w:float, debug=False) -> float:
    if y == 0:
        return w
    
    b = + epsilon * np.sqrt(Q)
    gamma = y**2 * V
    if debug:
        print("gamma: ", gamma, "b: ", b, "w: ", w, "y: ", y, "V: ", V, "Q: ", Q, "epsilon: ", epsilon)
    return -1/y * (prox_logit( np.array([-y*w + b]),gamma ) - b)
    
"""
Proximal from paper using equation 58 (Probably wrong, I do not understand)
"""
def equation_58(gamma, v):
    return v + gamma * ( 1 - np.exp(gamma + v) + (1-gamma)*np.exp(2*(gamma + v))) + np.heaviside(np.exp(2*v),0)

def proximal_from_equation_58(V: float, y: float, Q: float, epsilon: float, w:float, debug=False) -> float:
    if y == 0:
        return w
    
    b = + epsilon * np.sqrt(Q)
    gamma = y**2 * V
    if debug:
        print("gamma: ", gamma, "b: ", b, "w: ", w, "y: ", y, "V: ", V, "Q: ", Q, "epsilon: ", epsilon)
    return -1/y * (equation_58(gamma, -y*w + b) - b)


"""
Proximal from root
"""
def proximal_root(V: float, y: float, Q: float, epsilon: float, w:float) -> float:
    result = root(lambda z: y*V/(1+ np.exp(y*z - epsilon * np.sqrt(Q))) + w -z ,0) 
    return result.x[0]

""" 
Proximal from logistic root
"""
def proximal_logistic_root(V: float, y: float, Q: float, epsilon: float, w:float) -> float:
    if y == 0:
        return w

    w_prime = w - epsilon * np.sqrt(Q) / y
    result = root(lambda z: y*V/(1+ np.exp(y*z)) + w_prime - z ,0) 
    z = result.x[0]
    return z + epsilon * np.sqrt(Q) / y
    
"""
Proximal from minimize scalar
"""
def proximal_minimize_scalar(V: float, y: float, Q: float, epsilon: float, w: float, debug = False) -> float:
    return minimize_scalar(lambda z: (z-w)**2/(2*V) + np.log(1 + np.exp(-y*z + epsilon * np.sqrt(Q))))['x']

def proximal_logistic_minimize_scalar(V: float, y: float, Q: float, epsilon: float, w: float, debug = False) -> float:
    if y == 0:
        return w
    w_prime = w - epsilon * np.sqrt(Q) / y
    return minimize_scalar(lambda z: (z-w_prime)**2/(2*V) + np.log(1 + np.exp(-y*z)))['x'] + epsilon * np.sqrt(Q) / y


"""
Proximal from root scalar
"""
def proximal_root_scalar(V: float, y: float, Q: float, epsilon: float, w:float) -> float:
    result = root_scalar(lambda z: y*V/(1+ np.exp(y*z - epsilon * np.sqrt(Q))) + w -z , bracket=[-5000,5000]) 
    return result.root

"""
Proximal from root scalar logistic
"""
def proximal_logistic_root_scalar(V: float, y: float, Q: float, epsilon: float, w:float) -> float:
    if y == 0:
        return w
    try:
        w_prime = w - epsilon * np.sqrt(Q) / y
        result = root_scalar(lambda z: y*V/(1+ np.exp(y*z)) + w_prime - z , bracket=[-5000,5000]) 
        z = result.root
        return z + epsilon * np.sqrt(Q) / y
    except Exception as e:
        # print all parameters
        print("V: ", V, "y: ", y, "Q: ", Q, "epsilon: ", epsilon, "w: ", w)        
        raise e



"""
Proximal from fixed_point
"""
def proximal_fixed_point(V: float, y: float, Q: float, epsilon: float, w:float) -> float:
    return fixed_point(lambda z: y*V/(1+ np.exp(y*z - epsilon * np.sqrt(Q))) + w ,0)
    
"""
Proximal from logistic fixed_point
"""
def proximal_logistic_fixed_point(V: float, y: float, Q: float, epsilon: float, w:float) -> float:
    if y == 0:
        return w

    w_prime = w - epsilon * np.sqrt(Q) / y
    return fixed_point(lambda z: y*V/(1+ np.exp(y*z)) + w_prime ,0) + epsilon * np.sqrt(Q) / y

"""
Complete/ Full Proximal
"""

def full_proximal(z: float, V: float, y: float, Q: float, epsilon: float, w: float):
    return np.log(1 + np.exp(-y*z + epsilon * np.sqrt(Q))) + 1/(2*V) * (z - w)**2

def full_proximal_gradient(z: float, V: float, y: float, Q: float, epsilon: float, w: float):
    return -y / (1 + np.exp(y * z - epsilon * np.sqrt(Q))) + 1/V * (z - w)

def complete_proximal(z: float, V: float, y: float, Q: float, epsilon: float, w: float):
    return full_proximal(z, V, y, Q, epsilon, w), full_proximal_gradient(z, V, y, Q, epsilon, w)

def lbfgs_proximal_solver(V: float, y: float, Q: float, epsilon: float, w: float, debug = False):
    res = minimize(complete_proximal,0,args=(V, y, Q, epsilon,w),jac=True,method="L-BFGS-B",options={'maxiter':500}) 
    if debug:
        print("Minimized", "success:",res.success,"message",res.message)
        if not res.success:
            print("Optimization of Loss failed " + str(res.message))

        delta = 5
        print("Delta", delta)
        # evaluate the full_proximal function at the solution and plus minus delta and print the results
        print("Proximal at solution", full_proximal(res.x, V, y, Q, epsilon, w))
        print("Proximal at solution + delta", full_proximal(res.x + delta, V, y, Q, epsilon, w))
        print("Proximal at solution - delta", full_proximal(res.x - delta, V, y, Q, epsilon, w))
    return res.x[0]

"""
Logistic Proximal
"""
def logistic_proximal(z: float, V: float, y: float, w: float):
    return np.log(1 + np.exp(-y*z)) + 1/(2*V) * (z - w)**2

def logistic_proximal_gradient(z: float, V: float, y: float, w: float):
    return -y / (1 + np.exp(y * z)) + 1/V * (z - w)

def complete_logistic_proximal(z: float, V: float, y: float, w: float):
    return logistic_proximal(z, V, y, w), logistic_proximal_gradient(z, V, y, w)

def lbfgs_logistic_proximal_solver(V: float, y: float, w: float, debug = False):
    res = minimize(complete_logistic_proximal,0,args=(V, y, w),jac=True,method="L-BFGS-B",options={'maxiter':500}) 
    if debug:
        print("Minimized", "success:",res.success,"message",res.message)
        if not res.success:
            print("Optimization of Loss failed " + str(res.message))

        delta = 5
        print("Delta", delta)
        # evaluate the full_proximal function at the solution and plus minus delta and print the results
        print("Proximal at solution", logistic_proximal(res.x, V, y, w))
        print("Proximal at solution + delta", logistic_proximal(res.x + delta, V, y, w))
        print("Proximal at solution - delta", logistic_proximal(res.x - delta, V, y, w))
    return res.x[0]

def proximal_from_logistic(V: float, y: float, Q: float, epsilon: float, w: float, debug = False):
    if y == 0:
        return w

    w_prime = w - epsilon * np.sqrt(Q) / y
    return lbfgs_logistic_proximal_solver(V,y,w_prime) + epsilon * np.sqrt(Q) / y

"""
Code to compare all proximal implementations
"""    

def compare_functions(functions, args=(), num_runs=1000):
    """
    Compare the runtime of multiple functions with the same arguments.

    :param functions: List of functions to compare
    :param args: Tuple of arguments to pass to the functions
    :param num_runs: Number of times to run each function
    :return: Dictionary with function name as key and average runtime as value
    """
    timing_results = {}
    for func in functions:
        timing_results[func.__name__] = 0

    number_of_trials = 0
    # initialize a dictionary for each function name with zero as value
    different_results = {}
    for func in functions:
        different_results[func.__name__] = 0

    for y in range(-1,1):
        for w in np.linspace(1,200,10):
            for V in np.linspace(0.01,2,5):
                for Q in np.linspace(0.01,1,5):
                    for epsilon in np.array([0,0.1,0.2]):
                        results = {}
                        number_of_trials += 1
                        for func in functions:                            
                            start_time = time.time()
                            for _ in range(num_runs):
                                result = func(V,y,Q,epsilon,w)
                            end_time = time.time()
                            avg_time = (end_time - start_time) / num_runs
                            results[func.__name__] = avg_time

                            # add up the results
                            timing_results[func.__name__] += avg_time

                            # Check if all functions return the same result
                            if 'output' not in results:
                                results['output'] = result
                            elif not math.isclose(results['output'], result, rel_tol=1e-4, abs_tol=1e-4):
                                print(f"Warning: Function {func.__name__} returned a different result. ")
                                print(f"Expected {results['output']}, got {result}")
                                print(f"V: {V}, y: {y}, Q: {Q}, epsilon: {epsilon}, w: {w}")
                                different_results[func.__name__] += 1
                                # run the function again 
                                func(V,y,Q,epsilon,w, debug=True)

    # divide by the number of trials to get the average
    for func in functions:
        timing_results[func.__name__] /= number_of_trials

    print(f"Number of trials: {number_of_trials}")
    print(f"Number of different results: {different_results}")

    return timing_results


if __name__ == "__main__":
    V = 0.5
    y = -19.478130570343435
    Q = 1.0
    epsilon = 0.04
    w = 4.764398807031843

    # computes the proximal by rewriting in terms of the logistic proximal
    start = time.time()
    print("Proximal from logistic", proximal_from_logistic(V, y, Q, epsilon, w))
    print("Time", time.time() - start)

    # computes the proximal directly (should get the same result as above)
    start = time.time()
    print("Proximal direct", lbfgs_proximal_solver(V, y, Q, epsilon, w))
    print("Time", time.time() - start)

    # computes the proximal using root
    start = time.time()
    print("Proximal root", proximal_root(V, y, Q, epsilon, w))
    print("Time", time.time() - start)

    # computes the proximal using root on the logistic proximal
    start = time.time()
    print("Proximal root on logistic", proximal_logistic_root(V, y, Q, epsilon, w))
    print("Time", time.time() - start)

    # computes the proximal using fixed_point
    start = time.time()
    print("Proximal fixed_point", proximal_fixed_point(V, y, Q, epsilon, w))
    print("Time", time.time() - start)

    # computes the proximal using fixed_point on the logistic proximal
    start = time.time()
    print("Proximal fixed_point on logistic", proximal_logistic_fixed_point(V, y, Q, epsilon, w))
    print("Time", time.time() - start)

    # computes the proximal from the method from the paper
    start = time.time()
    print("Proximal from paper", proximal_from_paper(V, y, Q, epsilon, w))
    print("Time", time.time() - start)

    # computes the proximal using root_scalar
    start = time.time()
    print("Proximal root_scalar", proximal_root_scalar(V, y, Q, epsilon, w))
    print("Time", time.time() - start)

    # computes the proximal using root_scalar on the logistic proximal
    start = time.time()
    print("Proximal root_scalar on logistic", proximal_logistic_root_scalar(V, y, Q, epsilon, w))
    print("Time", time.time() - start)

    # computes the proximal using minimize_scalar
    start = time.time()
    print("Proximal minimize_scalar", proximal_minimize_scalar(V, y, Q, epsilon, w))
    print("Time", time.time() - start)

    # computes the proximal using minimize_scalar on the logistic proximal
    start = time.time()
    print("Proximal minimize_scalar on logistic", proximal_logistic_minimize_scalar(V, y, Q, epsilon, w))
    print("Time", time.time() - start)


    # # computes the proximal from equation 58
    # start = time.time()
    # print("Proximal from equation 58", proximal_from_equation_58(V, y, Q, epsilon, w))
    # print("Time", time.time() - start)


    # compare the runtime of all proximal implementations
    all_functions = [proximal_root, lbfgs_proximal_solver,proximal_from_logistic, proximal_logistic_root, proximal_root_scalar, proximal_logistic_root_scalar, proximal_minimize_scalar, proximal_logistic_minimize_scalar]
    results = compare_functions(all_functions, num_runs=1)
    print(results)
    # the lbfgs solver has sometimes trouble finding the correct minima. The root and root_scalar methods find the better minima...