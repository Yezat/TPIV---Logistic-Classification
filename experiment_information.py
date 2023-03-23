import json
from gradient_descent import loss_per_sample
import theoretical
from util import error
import numpy as np
from _version import __version__
from typing import Tuple
from util import generalization_error
import datetime
import uuid
import pandas as pd
import sqlite3
from data import *

class ExperimentInformation:
    def __init__(self, experiment_id: str, state_evolution_repetitions: int, erm_repetitions: int, alphas: np.ndarray, epsilons: np.ndarray, lambdas: np.ndarray,tau: np.ndarray, d: int):
        self.experiment_id: str = experiment_id
        self.code_version: str = __version__
        self.date: datetime = datetime.datetime.now()
        self.state_evolution_repetitions: int = state_evolution_repetitions
        self.erm_repetitions: int = erm_repetitions
        self.alphas: np.ndarray = alphas
        self.epsilons: np.ndarray = epsilons
        self.lambdas: np.ndarray = lambdas
        self.tau: float = tau
        self.d: int = d
        self.completed: bool = False
        

class StateEvolutionExperimentInformation:
    # define a constructor with all attributes
    def __init__(self, experiment_id: str, sigma: float, q: float, m: float, initial_condition: Tuple[float, float, float],alpha:float,epsilon:float,tau:float,lam:float,abs_tol:float,min_iter:int,max_iter:int,blend_fpe:float,int_lims:float):
        self.id: str = str(uuid.uuid4())
        self.code_version: str = __version__
        self.experiment_id: str = experiment_id
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

class ERMExperimentInformation:
    def __init__(self, experiment_id: str, Xtest: np.ndarray, w_gd: np.ndarray, tau: float, y: np.ndarray, Xtrain: np.ndarray, w: np.ndarray, ytest: np.ndarray, d: int, minimizer_name: str, epsilon: float, lam: float):
        self.id: str = str(uuid.uuid4())
        self.code_version: str = __version__
        self.experiment_id: str = experiment_id
        self.q: float= w_gd@w_gd / d
        self.rho: float = w@w /d
        self.m: float = w_gd@w / d
        self.cosb: float = self.m / np.sqrt(self.q*self.rho)
        self.epsilon: float = epsilon
        self.lam: float = lam
        
        n: int = Xtrain.shape[0]
        yhat_gd = theoretical.predict_erm(Xtest,w_gd)
        self.generalization_error_erm: float = error(ytest,yhat_gd)
        self.generalization_error_overlap: float = generalization_error(self.rho,self.m,self.q)
        yhat_gd_train = theoretical.predict_erm(Xtrain,w_gd)
        self.date: datetime = datetime.datetime.now()
        self.chosen_minimizer: str = minimizer_name
        self.training_error: float = error(y,yhat_gd_train)        

        self.d: int = d
        self.tau: float = tau
        self.alpha: float = n/d
        self.test_loss: float = loss_per_sample(ytest,Xtest@w_gd,epsilon=epsilon,w=w_gd).mean()


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj,ERMExperimentInformation):
            return obj.__dict__
        if isinstance(obj,np.int32):
            return str(obj)
        return json.JSONEncoder.default(self, obj)
    


DB_NAME = "experiments.db"
STATE_EVOLUTION_TABLE = "state_evolution"
ERM_TABLE = "erm"
EXPERIMENTS_TABLE = "experiments"

# How to change a column later:
# cursor.execute('ALTER TABLE experiments ADD COLUMN new_column TEXT')

class DatabaseHandler:
    def __init__(self, logger, db_name=DB_NAME):
        self.connection = sqlite3.connect(db_name,timeout=15)
        self.cursor = self.connection.cursor()
        self.logger = logger

        # test if all tables exist and create them if not
        self.cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{STATE_EVOLUTION_TABLE}'")
        if self.cursor.fetchone() is None:
            self.logger(f"Creating table {STATE_EVOLUTION_TABLE} ...")
            self.cursor.execute(f"CREATE TABLE {STATE_EVOLUTION_TABLE} (id TEXT, code_version TEXT, experiment_id TEXT, date TEXT, sigma REAL, q REAL, m REAL, initial_condition TEXT, alpha REAL, epsilon REAL, tau REAL, lam REAL, abs_tol REAL, min_iter INTEGER, max_iter INTEGER, blend_fpe REAL, int_lims REAL, generalization_error REAL)")
            self.connection.commit()

        # For the experiments table
        self.cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{EXPERIMENTS_TABLE}'")
        if self.cursor.fetchone() is None:
            self.logger(f"Creating table {EXPERIMENTS_TABLE} ...")
            self.cursor.execute(f"CREATE TABLE {EXPERIMENTS_TABLE} (experiment_id TEXT,code_version TEXT, state_evolution_repetitions INTEGER, erm_repetitions INTEGER, date TEXT, alphas TEXT, epsilons TEXT, lambdas TEXT, tau REAL, d INTEGER, completed BOOLEAN)")
            self.connection.commit()

        # For the erm table
        self.cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{ERM_TABLE}'")
        if self.cursor.fetchone() is None:
            self.logger(f"Creating table {ERM_TABLE} ...")
            self.cursor.execute(f"CREATE TABLE {ERM_TABLE} (id TEXT, code_version TEXT, experiment_id TEXT, date TEXT, q REAL, m REAL, rho REAL, cosb REAL, epsilon REAL, lam REAL, generalization_error_erm REAL, generalization_error_overlap REAL, chosen_minimizer TEXT, training_error REAL, test_loss REAL, d INTEGER, tau REAL, alpha REAL)")
            self.connection.commit()

    def insert_experiment(self, experiment_information: ExperimentInformation):
        self.cursor.execute(f"INSERT INTO {EXPERIMENTS_TABLE} VALUES (?,?,?,?,?,?,?,?,?,?,?)", (experiment_information.experiment_id, experiment_information.code_version, experiment_information.date.isoformat(), experiment_information.state_evolution_repetitions, experiment_information.erm_repetitions, json.dumps(experiment_information.alphas,cls=NumpyEncoder), json.dumps(experiment_information.epsilons,cls=NumpyEncoder), json.dumps(experiment_information.lambdas,cls=NumpyEncoder), experiment_information.tau, experiment_information.d, experiment_information.completed))
        self.connection.commit()

    def set_experiment_completed(self, experiment_id: str):
        self.cursor.execute(f"UPDATE {EXPERIMENTS_TABLE} SET completed=1 WHERE experiment_id='{experiment_id}'")
        self.connection.commit()

    def insert_state_evolution(self, experiment_information: StateEvolutionExperimentInformation):
        self.cursor.execute(f"INSERT INTO {STATE_EVOLUTION_TABLE} VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", (experiment_information.id,experiment_information.code_version, experiment_information.experiment_id, experiment_information.date.isoformat(), experiment_information.sigma, experiment_information.q, experiment_information.m, json.dumps(experiment_information.initial_condition,cls=NumpyEncoder), experiment_information.alpha, experiment_information.epsilon, experiment_information.tau, experiment_information.lam, experiment_information.abs_tol, experiment_information.min_iter, experiment_information.max_iter, experiment_information.blend_fpe, experiment_information.int_lims, experiment_information.generalization_error))
        self.connection.commit()

    def insert_erm(self, experiment_information: ERMExperimentInformation):
        self.cursor.execute(f"INSERT INTO {ERM_TABLE} VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", (experiment_information.id,experiment_information.code_version, experiment_information.experiment_id, experiment_information.date.isoformat(), experiment_information.q, experiment_information.m, experiment_information.rho, experiment_information.cosb, experiment_information.epsilon, experiment_information.lam, experiment_information.generalization_error_erm, experiment_information.generalization_error_overlap, experiment_information.chosen_minimizer, experiment_information.training_error, experiment_information.test_loss, experiment_information.d, experiment_information.tau, experiment_information.alpha))
        self.connection.commit()

    def get_experiments(self):
        return pd.read_sql_query(f"SELECT * FROM {EXPERIMENTS_TABLE}", self.connection)

    def get_state_evolutions(self):
        return pd.read_sql_query(f"SELECT * FROM {STATE_EVOLUTION_TABLE}", self.connection)
    
    def get_erms(self):
        return pd.read_sql_query(f"SELECT * FROM {ERM_TABLE}", self.connection)

    # how to implement a select
    # make sure to return a pandas dataframe...
    # c.execute('SELECT * FROM experiments WHERE result > 0.5')
    # results = c.fetchall()
    # for row in results:
    #     print(row)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.connection.close()


def create_experiment_id():
    return str(uuid.uuid4())

if __name__ == "__main__":
    id = create_experiment_id()
    print(id)
    with DatabaseHandler() as dbHandler:
        # create random experiment information
        experiment_information = ExperimentInformation(id,10,10, [0.1,0.2,0.3], [0.1,0.2,0.3], [0.1,0.2,0.3], 0.1, 10)
        dbHandler.insert_experiment(experiment_information)
        # create random state evolution experiment information
        state_evolution_exp = StateEvolutionExperimentInformation(id,0.1, 0.1, 0.1, [0.1,0.2,0.4],4,3,1,3,4,5,23,34,234)
        dbHandler.insert_state_evolution(state_evolution_exp)

        # create random erm experiment information
        # sample test data
        w = sample_weights(10)
        X, y = sample_training_data(w,10,100,2)
        w2 = sample_weights(10)
        X2, y2 = sample_training_data(w2,10,100,2)
        erm_exp = ERMExperimentInformation(id,X,w,2,y,X2,w2,y2,10,"test",0.3,1e-4)
        dbHandler.insert_erm(erm_exp)

        # set experiment to completed
        dbHandler.set_experiment_completed(id)
    
