"""
This module calculates calibrations
"""
from erm import *
from gradient_descent import *
from util import sigma_star
from scipy.special import erfc
from scipy.special import logit


def compute_experimental_teacher_calibration(p, w, werm, Xtest, sigma):
    try:

        
        # size of bins where we put the probas
        n, d = Xtest.shape
        dp = 0.025
        sigmoid = np.vectorize(lambda x : 1. / (1. + np.exp( -x )))
        Ypred = sigmoid(Xtest @ werm)

        index = [i for i in range(n) if p - dp <= Ypred[i] <= p + dp]
        def probit(lf, sigma):
            return 0.5 * erfc(- lf / np.sqrt(2 * sigma**2))

        return p - np.mean([probit(w @ Xtest[i], sigma) for i in index])
    except Exception as e:
        print(e)
        return np.nan



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


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    alpha = 2
    d = 1000
    method = "sklearn"
    # method = "gd"
    tau = 0.1
    epsilon = 0.8
    p = 0.75
    dp = 0.01
    lam = 10**-2

    repetitions = 10

    gd_calibs_analytical = np.empty(repetitions)
    gd_calibs_analytical[:] = np.nan
    clarte_calibs = np.empty(repetitions)
    clarte_calibs[:] = np.nan

    for i in range(repetitions):

        start = time.time()
        print("Starting experiment with alpha = ",alpha," d = ",d," method = ",method," tau = ",tau," lam = ",lam," epsilon = ",epsilon)

        w = sample_weights(d)
        # # generate data
        Xtrain, y = sample_training_data(w,d,int(alpha * d),tau)
        n_test = 100000
        Xtest,ytest = sample_training_data(w,d,n_test,tau)

        w_gd = np.empty(w.shape,dtype=w.dtype)
        w_gd = sklearn_optimize(sample_weights(d),Xtrain,y,lam,epsilon)
        print(w_gd.shape)

        end = time.time()
        duration = end - start
        print("Duration: ",duration)

        from experiment_information import ERMExperimentInformation
        erm_information = ERMExperimentInformation("my_erm_minimizer_tests",duration,Xtest,w_gd,tau,y,Xtrain,w,ytest,d,method,epsilon,lam,None,None)


        # # calculate calibration
        clarte_calib = compute_experimental_teacher_calibration(p,w,w_gd,Xtest,tau)
        overlap_calib = calc_calibration_analytical(erm_information.rho,p,erm_information.m,erm_information.Q,tau)

        print("Clarte calibration: ",clarte_calib)
        print("Overlap calibration: ",overlap_calib)

        gd_calibs_analytical[i] = overlap_calib
        clarte_calibs[i] = clarte_calib


    print("Clarte calibration mean: ",np.mean(clarte_calibs))
    print("Clarte calibration std: ",np.std(clarte_calibs))
    print("Analytical calibration mean: ",np.mean(gd_calibs_analytical))
    print("Analytical calibration std: ",np.std(gd_calibs_analytical))
