import json
from gradient_descent import loss_per_sample
import theoretical
from util import error
import numpy as np
from _version import __version__
from typing import Tuple
from util import generalization_error
import datetime
import sqlite3

class StateEvolutionExperimentInformation:
    # define a constructor with all attributes
    def __init__(self, sigma: float, q: float, m: float, initial_condition: Tuple[float, float, float],alpha:float,epsilon:float,tau:float,lam:float,abs_tol:float,min_iter:int,max_iter:int,blend_fpe:float,int_lims:float):
        self.code_version = __version__
        rho_w_star = 1.0
        self.generalization_error: float = generalization_error(rho_w_star, m, sigma + q)
        # store current date and time
        self.date: datetime = datetime.datetime.now()
        self.sigma: float = sigma
        self.q: float = q
        self.m: float = m
        self.initial_condition: Tuple[float, float, float] = initial_condition
        self.rho_w_star: float = rho_w_star
        self.alpha: float = alpha
        self.epsilon: float = epsilon 
        self.tau: float = tau
        self.lam: float = lam 
        self.abs_tol: float = abs_tol 
        self.min_iter: int = min_iter
        self.max_iter: int = max_iter
        self.blend_fpe: float = blend_fpe
        self.int_lims: float = int_lims    

class ExperimentInformation:
    def __init__(self, Xtest: np.ndarray, w_gd: np.ndarray, tau: float, y: np.ndarray, Xtrain: np.ndarray, w: np.ndarray, ytest: np.ndarray, d: int, minimizer_name: str):
        self.code_version = __version__
        n = Xtrain.shape[0]
        yhat_gd = theoretical.predict_erm(Xtest,w_gd)
        self.generalization_error = error(ytest,yhat_gd)
        yhat_gd_train = theoretical.predict_erm(Xtrain,w_gd)
        self.date: datetime = datetime.datetime.now()
        self.chosen_minimizer = "unknown"
        self.training_error = error(y,yhat_gd_train)
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
        self.alpha = 0.0
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
    n = Xtrain.shape[0]
    debug_information.alpha = n /d
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
    


DB_NAME = "experiments.db"
STATE_EVOLUTION_TABLE = "state_evolution"
ERM_TABLE = "erm"

# How to change a column later:
# cursor.execute('ALTER TABLE experiments ADD COLUMN new_column TEXT')

class DatabaseConnection:
    def __init__(self):
        self.connection = sqlite3.connect(DB_NAME)
        self.cursor = self.connection.cursor()

        # test if both tables exist and create them if not
        self.cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{STATE_EVOLUTION_TABLE}'")
        if self.cursor.fetchone() is None:
            print("Creating table",STATE_EVOLUTION_TABLE,"...")
            self.cursor.execute(f"CREATE TABLE {STATE_EVOLUTION_TABLE} (code_version TEXT, date TEXT, sigma REAL, q REAL, m REAL, initial_condition TEXT, alpha REAL, epsilon REAL, tau REAL, lam REAL, abs_tol REAL, min_iter INTEGER, max_iter INTEGER, blend_fpe REAL, int_lims REAL, generalization_error REAL)")
            self.connection.commit()
        # For the erm table
        self.cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{ERM_TABLE}'")
        if self.cursor.fetchone() is None:
            print("Creating table",ERM_TABLE,"...")
            # TODO, this isn't correct yet and the ERM class will change...
            self.cursor.execute(f"CREATE TABLE {ERM_TABLE} (code_version TEXT, date TEXT, q REAL, rho REAL, m REAL, alpha REAL, epsilon REAL, tau REAL, lam REAL, abs_tol REAL, min_iter INTEGER, max_iter INTEGER, blend_fpe REAL, int_lims REAL, generalization_error REAL, final_state TEXT, final_state_error REAL)")
            self.connection.commit()

    def insertStateEvolutionExperimentInformation(self, experiment_information: StateEvolutionExperimentInformation):
        self.cursor.execute(f"INSERT")
        self.connection.commit()

    def insertERMEvolutionExperimentInformation(self, experiment_information: ExperimentInformation):
        self.cursor.execute(f"INSERT INTO {ERM_TABLE} VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", (experiment_information.code_version, experiment_information.date, experiment_information.sigma, experiment_information.q, experiment_information.m, json.dumps(experiment_information.initial_condition), experiment_information.alpha, experiment_information.epsilon, experiment_information.tau, experiment_information.lam, experiment_information.abs_tol, experiment_information.min_iter, experiment_information.max_iter, experiment_information.blend_fpe, experiment_information.int_lims, experiment_information.generalization_error, json.dumps(experiment_information.final_state), experiment_information.final_state_error))
        self.connection.commit()

    # how to implement a select
    # c.execute('SELECT * FROM experiments WHERE result > 0.5')
    # results = c.fetchall()
    # for row in results:
    #     print(row)

    def __enter__(self):
        return self.connection

    def __exit__(self):
        self.connection.close()

if __name__ == "__main__":
    v = StateEvolutionExperimentInformation()
    print(json.dumps(v.__dict__,cls=NumpyEncoder))