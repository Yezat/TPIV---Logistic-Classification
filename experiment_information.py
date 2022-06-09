import json
from gradient_descent import loss_per_sample
import theoretical
from util import error
import numpy as np


class ExperimentInformation:
    def __init__(self):
        self.generalization_error = 0.0
        self.chosen_minimizer = "unknown"
        self.training_error = 0.0
        self.q = 0.0
        self.m = 0.0
        self.rho = 0.0
        self.cosb = 0.0
        self.w = 0.0
        self.w_gd = 0.0
        self.epsilon = 0.0
        self.lam = 0.0
        self.calibration = 0.0
        self.d = 0.0
        self.tau = 0.0
        self.test_loss = 0.0

def get_experiment_information(Xtest,w_gd,tau,y,Xtrain,w,ytest,d,minimizer_name):
    debug_information = ExperimentInformation()
    yhat_gd = theoretical.predict_erm(Xtest,w_gd)
    debug_information.generalization_error = error(ytest,yhat_gd)
    yhat_gd_train = theoretical.predict_erm(Xtrain,w_gd)
    debug_information.test_loss = loss_per_sample(ytest,Xtest@w_gd,epsilon=0,w=w_gd).mean()
    debug_information.training_error = error(y,yhat_gd_train)
    debug_information.q = w_gd@w_gd / d
    debug_information.rho = w@w /d
    debug_information.m = w_gd@w / d
    debug_information.cosb = debug_information.m / np.sqrt(debug_information.q*debug_information.rho)
    debug_information.w = w
    debug_information.norm_w = np.linalg.norm(w,2)
    debug_information.w_gd = w_gd
    debug_information.norm_w_gd = np.linalg.norm(w_gd,2)
    debug_information.chosen_minimizer = minimizer_name
    debug_information.d = d
    debug_information.tau = tau
    return debug_information

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj,ExperimentInformation):
            return obj.__dict__
        if isinstance(obj,np.int32):
            return str(obj)
        return json.JSONEncoder.default(self, obj)

