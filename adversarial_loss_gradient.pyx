# cython: language_level=3


# Design:
# See https://github.com/scikit-learn/scikit-learn/issues/15123 for reasons.
# a) Merge link functions into loss functions for speed and numerical
#    stability, i.e. use raw_prediction instead of y_pred in signature.
# b) Pure C functions (nogil) calculate single points (single sample)
# c) Wrap C functions in a loop to get Python functions operating on ndarrays.
#   - Write loops manually---use Tempita for this.
#     Reason: There is still some performance overhead when using a wrapper
#     function "wrap" that carries out the loop and gets as argument a function
#     pointer to one of the C functions from b), e.g.
#     wrap(closs_half_poisson, y_true, ...)
#   - Pass n_threads as argument to prange and propagate option to all callers.
# d) Provide classes (Cython extension types) per loss (names start with Cy) in
#    order to have semantical structured objects.
#    - Member functions for single points just call the C function from b).
#      These are used e.g. in SGD `_plain_sgd`.
#    - Member functions operating on ndarrays, see c), looping over calls to C
#      functions from b).
# e) Provide convenience Python classes that compose from these extension types
#    elsewhere (see loss.py)
#    - Example: loss.gradient calls CyLoss.gradient but does some input
#      checking like None -> np.empty().
#
# Note: We require 1-dim ndarrays to be contiguous.

from cython.parallel import parallel, prange
import numpy as np
import mpmath as mp

from libc.math cimport exp, fabs, log, log1p, pow
from libc.stdlib cimport malloc, free

# # Fused types for y_true, y_pred, raw_prediction
# ctypedef fused Y_DTYPE_C:
#     double
#     float
cimport numpy as cnp
ctypedef cnp.npy_float64 Y_DTYPE_C

# Fused types for gradient and hessian
ctypedef fused G_DTYPE_C:
    double
    float


# Struct to return 2 doubles
ctypedef struct double_pair:
   double val1
   double val2

ctypedef struct double_triple:
    double val1
    double val2
    double val3

# -------------------------------------
# Helper functions
# -------------------------------------
# Numerically stable version of log(1 + exp(x)) for double precision, see Eq. (10) of
# https://cran.r-project.org/web/packages/Rmpfr/vignettes/log1mexp-note.pdf
# Note: The only important cutoff is at x = 18. All others are to save computation
# time. Compared to the reference, we add the additional case distinction x <= -2 in
# order to use log instead of log1p for improved performance. As with the other
# cutoffs, this is accurate within machine precision of double.
cdef inline double log1pexp(double x) noexcept nogil:
    if x <= -37:
        return exp(x)
    elif x <= -2:
        return log1p(exp(x))
    elif x <= 18:
        return log(1. + exp(x))
    elif x <= 33.3:
        return x + exp(-x)
    else:
        return x


cdef inline double_triple closs_grad_half_binomial(
    double y_true,
    double raw_prediction,
    double adversarial_norm
) noexcept nogil:
    cdef double_triple lg
    cdef double adversarial_plus = raw_prediction + adversarial_norm
    cdef double adversarial_minus = raw_prediction - adversarial_norm
    cdef double plus_term
    cdef double minus_term
    if adversarial_plus <= 0:
        lg.val2 = exp(adversarial_plus)  # used as temporary
        if adversarial_plus <= -37:
            lg.val1 = lg.val2 - y_true * adversarial_minus              # loss
        else:
            lg.val1 = log1p(lg.val2) - y_true * adversarial_minus       # loss

        # compute additive term for loss
        # here exp(adversarial_plus) will not overflow
        # if adversarial_minus <= 0:
        # exp(adversarial_minus) will not overflow  
        # else:
        # exp(adversarial_minus) will overflow
        # however, adversarial_minus is always smaller than adversarial_plus, so id adversarial_plus will not overflow, adversarial_minus will not overflow either
        lg.val1 += y_true * log( (1 + exp(adversarial_minus)) / (1 + exp(adversarial_plus)) )

        # compute x_mu_j term for gradient
        # adversarial_plus will not overflow
        if adversarial_minus <= 0:
            lg.val2 = (1-y_true) * exp(adversarial_plus) / (1 + exp(adversarial_plus)) - y_true / (1 + exp(adversarial_minus)) 
            lg.val3 = (1-y_true) * exp(adversarial_plus) / (1 + exp(adversarial_plus)) + y_true / (1 + exp(adversarial_minus)) 
        else:
            lg.val2 = (1-y_true) * exp(adversarial_plus) / (1 + exp(adversarial_plus)) - y_true * ( exp(-adversarial_minus) / (1 + exp(-adversarial_minus)) )
            lg.val3 = (1-y_true) * exp(adversarial_plus) / (1 + exp(adversarial_plus)) + y_true * ( exp(-adversarial_minus) / (1 + exp(-adversarial_minus)) )

    else:
        lg.val2 = exp(-adversarial_plus)  # used as temporary
        if adversarial_plus <= 18:
            # log1p(exp(x)) = log(1 + exp(x)) = x + log1p(exp(-x))
            lg.val1 = log1p(lg.val2) + (1 - y_true) * adversarial_minus  # loss
        else:
            lg.val1 = lg.val2 + (1 - y_true) * adversarial_minus         # loss
    
        # compute additive term for loss
        # here exp(adversarial_plus) can overflow, hence we use -adversarial_plus 

        # adversarial_norm is always positive, hence the exp(negative adversarial_norm) cannot overflow
        lg.val1 += y_true * log( (exp(-adversarial_plus) + exp(-2*adversarial_norm)) / (1 + exp(-adversarial_plus)) )
        # as you can see, adversarial_minus no longer shows up, so there is no second case distinction

        # compute x_mu_j term for gradient
        # adversarial_plus will overflow
        if adversarial_minus <= 0:
            lg.val2 = (1-y_true) / (1 + exp(-adversarial_plus)) - y_true * ( 1 / (1 + exp(adversarial_minus)) )
            lg.val3 = (1-y_true) / (1 + exp(-adversarial_plus)) + y_true * ( 1 / (1 + exp(adversarial_minus)) )
        else:
            lg.val2 = (1-y_true) / (1 + exp(-adversarial_plus)) - y_true * ( exp(-adversarial_minus) / (1 + exp(-adversarial_minus)) )
            lg.val3 = (1-y_true) / (1 + exp(-adversarial_plus)) + y_true * ( exp(-adversarial_minus) / (1 + exp(-adversarial_minus)) )

    return lg


cdef inline double_triple mp_closs_grad_half_binomial(
    double y_true,
    double raw_prediction,
    double adversarial_norm
):
    cdef double_triple lg
    cdef double adversarial_plus = raw_prediction + adversarial_norm
    cdef double adversarial_minus = raw_prediction - adversarial_norm
    cdef object y = mp.mpf(y_true)
    cdef object ap = mp.mpf(adversarial_plus)
    cdef object am = mp.mpf(adversarial_minus)
    cdef object rp = mp.mpf(raw_prediction)
    cdef object an = mp.mpf(adversarial_norm)

    cdef object lo = -y * rp + y*an + (1-y)*mp.log(1+mp.exp(rp+an)) + y * mp.log(1+mp.exp(rp-an))
    cdef object gr = (1-y) / (1 + mp.exp(-ap)) - y/(1+mp.exp(am))
    cdef object eg = (1-y) / (1 + mp.exp(-ap)) + y/(1+mp.exp(am))

    lg.val1 = np.float64(lo)
    lg.val2 = np.float64(gr)
    lg.val3 = np.float64(eg)

    return lg


def mp_loss_gradient(
        const Y_DTYPE_C[::1] y_true,          # IN
        const Y_DTYPE_C[::1] raw_prediction,  # IN
        const double adversarial_norm,        # IN
        G_DTYPE_C[::1] loss_out,              # OUT
        G_DTYPE_C[::1] gradient_out,          # OUT
        G_DTYPE_C[::1] epsilon_gradient_out          # OUT
    ):
        cdef:
            int i
            int n_samples = y_true.shape[0]
            double_triple dbl3

        for i in range(
                n_samples
            ):
                dbl3 = mp_closs_grad_half_binomial(y_true[i], raw_prediction[i], adversarial_norm)
                loss_out[i] = dbl3.val1
                gradient_out[i] = dbl3.val2
                epsilon_gradient_out[i] = dbl3.val3

        return np.asarray(loss_out), np.asarray(gradient_out), np.asarray(epsilon_gradient_out)


cdef class CyHalfBinomialLoss():
    """Half Binomial deviance loss with logit link.

    Domain:
    y_true in [0, 1]
    y_pred in (0, 1), i.e. boundaries excluded

    Link:
    y_pred = expit(raw_prediction)
    """

    def loss_gradient(
        self,
        const Y_DTYPE_C[::1] y_true,          # IN
        const Y_DTYPE_C[::1] raw_prediction,  # IN
        const double adversarial_norm,        # IN
        G_DTYPE_C[::1] loss_out,              # OUT
        G_DTYPE_C[::1] gradient_out,          # OUT
        G_DTYPE_C[::1] epsilon_gradient_out,          # OUT
        int n_threads=1
    ):
        cdef:
            int i
            int n_samples = y_true.shape[0]
            double_triple dbl3

        for i in prange(
                n_samples, schedule='static', nogil=True, num_threads=n_threads
            ):
                dbl3 = closs_grad_half_binomial(y_true[i], raw_prediction[i], adversarial_norm)
                loss_out[i] = dbl3.val1
                gradient_out[i] = dbl3.val2
                epsilon_gradient_out[i] = dbl3.val3

        return np.asarray(loss_out), np.asarray(gradient_out), np.asarray(epsilon_gradient_out)
