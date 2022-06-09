"""
This module calculates calibrations
"""
import theoretical
from erm import *
import math
from gradient_descent import *
from util import sigma_star, logistic_function
from scipy.special import logit

def calc_calibration_for_sklearn(Xtest, p, dp, clf, w, tau):
    """
    calculates the calibration for a given test set and a given probability p and a given precision dp
    on the given logistic regressor clf and given weights w. returns nan if no calibration could be calculated

    Xtest: test data (n,d)
    p: probability between 0 and 1
    dp: probability precision
    clf: logistic regressor
    w: weights of oracle
    tau: noise level of oracle
    """
    f_star = theoretical.predict_proba(Xtest,w,tau)
    f_erm = predict_proba_on_logistic_regressor(clf,Xtest)
    indices = np.logical_and(p-dp < f_erm,p+dp > f_erm).nonzero()[0]
    try:
        return p - np.mean(np.take(f_star,indices))
    except:
        return np.nan

def calc_calibration_gd(Xtest, p, dp, w_gd, w, tau, debug = False):
    """
    Calculates the calibration for a given test set and a given probability p and a given precision dp
    on the given logistic regressor clf and given weights w. 
    Takes the maximum of the erm prediction distribution to adapt the binning.
    Returns p in case no calibration could be calculated and prints a warning

    Xtest: test data (n,d)
    p: probability between 0 and 1
    dp: probability precision
    w_gd: weights of student
    w: weights of teacher
    tau: noise level of oracle
    debug: if true, prints debug information
    """
    f_star = theoretical.predict_proba(Xtest,w,tau)
    f_erm = theoretical.predict_erm_proba(Xtest,w_gd,debug=True)
    if debug:
        print("Calibration, F_ERM debug statements:",np.mean(f_erm),np.std(f_erm),np.var(f_erm),np.min(f_erm),np.max(f_erm))
        print("Calibration, F_STAR debug statements:",np.mean(f_star),np.std(f_star),np.var(f_star),np.min(f_star),np.max(f_star))
    indices = np.array([])
    
    assert(f_star.shape == f_erm.shape)


    bins = np.linspace(np.min(f_erm),np.max(f_erm),math.ceil(1/dp))
    

    bin_index = int(p*math.ceil(1/dp))

    
    lower = bins[bin_index-1]
    upper = bins[bin_index]
    
    indices = np.logical_and(lower < f_erm,upper > f_erm)
    indices = indices.nonzero()[0]

    mid = (upper-lower)/2 + lower
    adjustment = p-mid
    
    return p - np.mean(np.take(f_star,indices)) - adjustment

def calc_calibration_analytical(rho,p,m,q_erm,tau, debug = False):
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
    m_q_ratio = m/(q_erm )

    num = (logi )* m_q_ratio 
    if debug:
        print("tau",tau,"m**2",m**2,"q_erm",q_erm,"m**2/q_erm",m**2/q_erm)
    denom = np.sqrt(rho - (m**2)/(q_erm) + (tau)**2)
    if debug:
        print("logi",logi,"m_q_ratio",m_q_ratio,"num",num,"denom",denom)
    return p - sigma_star( ( num ) / ( denom )  )


        