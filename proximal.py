import numpy as np
# http://proximity-operator.net/code/matlab/scalar/prox_logit.m

# TODO: look at the link above and the paper referenced on tha page of the proximal operator of the logit function
# In particular, I wonder what is happening for negative values of x. 
# def prox_logit(x, gamma):
#     """
#     This procedure computes the proximity operator of the function:
#     f(x) = gamma * log( 1 + exp(x) )
#     When the input 'x' is an array, the output 'p' is computed element-wise.
#     """
#     # check input
#     if any(gamma <= 0) or (not np.isscalar(gamma) and any(np.shape(gamma) != np.shape(x))):
#         raise ValueError("'gamma' must be positive and either scalar or the same size as 'x'")
    
#     # Use Moreau's decomposition formula
#     p = x - gamma * prox_entropy_symm(x / gamma, 1 / gamma)
    
#     return p
