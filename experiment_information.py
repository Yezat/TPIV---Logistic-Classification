from gradient_descent import pure_training_loss
import theoretical
from state_evolution import pure_training_loss_logistic, training_error_logistic, adversarial_generalization_error_logistic
from util import error, adversarial_error
import numpy as np
from _version import __version__
from typing import Tuple
from util import generalization_error
import datetime
import uuid
from data import *
import json
import sqlite3
import pandas as pd
import logging
from data_model import DataModelType

class ExperimentInformation:
    def __init__(self, state_evolution_repetitions: int, erm_repetitions: int, alphas: np.ndarray, epsilons: np.ndarray, lambdas: np.ndarray, taus: np.ndarray, d: int, erm_methods: list, ps: np.ndarray, dp: float, dataModelType: DataModelType, p: int, experiment_name: str = ""):
        self.experiment_id: str = str(uuid.uuid4())
        self.experiment_name: str = experiment_name
        self.duration: float = 0.0
        self.code_version: str = __version__
        self.date: datetime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.state_evolution_repetitions: int = state_evolution_repetitions
        self.erm_repetitions: int = erm_repetitions
        self.alphas: np.ndarray = alphas
        self.epsilons: np.ndarray = epsilons
        self.lambdas: np.ndarray = lambdas
        self.taus: np.ndarray = taus
        self.ps: np.ndarray = ps
        self.dp: float = dp
        self.p: int = p
        self.d: int = d
        self.erm_methods: list = erm_methods
        self.completed: bool = False
        self.data_model_type: DataModelType = dataModelType

    # overwrite the to string method to print all attributes and their type
    def __str__(self):
        # return for each attribute the content and the type
        return "\n".join(["%s: %s (%s)" % (key, value, type(value)) for key, value in self.__dict__.items()])

class CalibrationResults:
    def __init__(self, ps: np.ndarray, calibrations: np.ndarray, dp: float):
        self.ps: np.ndarray = ps
        self.calibrations: np.ndarray = calibrations
        self.dp: float = dp # if this is None, we know we computed the calibration using the analytical expression
    # make this object json serializable
    def to_json(self):
        return json.dumps(self, sort_keys=True, indent=4, cls=NumpyEncoder)



class StateEvolutionExperimentInformation:
    # define a constructor with all attributes
    def __init__(self, experiment_id: str, duration: float, sigma: float, q: float, m: float, initial_condition: Tuple[float, float, float],alpha:float,epsilon:float,tau:float,lam:float,calibrations:CalibrationResults,abs_tol:float,min_iter:int,max_iter:int,blend_fpe:float,int_lims:float, sigma_hat: float, q_hat: float, m_hat: float, rho: float, A: float, N: float, A_hat: float, N_hat: float):
        self.id: str = str(uuid.uuid4())
        self.code_version: str = __version__
        self.duration: float = duration
        self.experiment_id: str = experiment_id
        rho_w_star = rho
        self.generalization_error: float = generalization_error(rho_w_star, m, q, tau)
        self.adversarial_generalization_error: float = adversarial_generalization_error_logistic(m,q,rho_w_star,tau,epsilon)
        self.training_loss: float = pure_training_loss_logistic(m,q,sigma,A,N,rho_w_star,alpha,tau,epsilon, lam)
        self.training_error: float = training_error_logistic(m,q,sigma,A,N,rho_w_star,alpha,tau,epsilon, lam)
        # store current date and time
        self.date: datetime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.sigma: float = sigma
        self.q: float = q
        self.m: float = m
        self.cosb: float = self.m / np.sqrt((self.q)*rho_w_star)
        self.initial_condition: Tuple[float, float, float] = initial_condition
        self.rho_w_star: float = rho_w_star
        self.alpha: float = alpha
        self.epsilon: float = epsilon
        self.tau: float = tau
        self.lam: float = lam
        self.calibrations: CalibrationResults = calibrations
        self.abs_tol: float = abs_tol
        self.min_iter: int = min_iter
        self.max_iter: int = max_iter
        self.blend_fpe: float = blend_fpe
        self.int_lims: float = int_lims
        self.sigma_hat : float = sigma_hat
        self.q_hat : float = q_hat
        self.m_hat : float = m_hat
        self.A : float = A
        self.N : float = N
        self.A_hat : float = A_hat
        self.N_hat : float = N_hat

class ERMExperimentInformation:
    def __init__(self, experiment_id: str, duration: float, Xtest: np.ndarray, w_gd: np.ndarray, tau: float, y: np.ndarray, Xtrain: np.ndarray, w: np.ndarray, ytest: np.ndarray, d: int, minimizer_name: str, epsilon: float, lam: float, analytical_calibrations: CalibrationResults, erm_calibrations: CalibrationResults, m: float, Q: float, rho: float, Sigma_w: np.ndarray, A: float, N: float):
        self.id: str = str(uuid.uuid4())
        self.duration : float = duration
        self.code_version: str = __version__
        self.experiment_id: str = experiment_id
        self.Q: float= Q
        self.m: float = m
        self.rho: float = rho
        self.cosb: float = self.m / np.sqrt(self.Q*self.rho)
        self.epsilon: float = epsilon
        self.lam: float = lam

        n: int = Xtrain.shape[0]
        yhat_gd = theoretical.predict_erm(Xtest,w_gd)
        # yhat_gd = np.sign(Xtest @ w_gd)
        self.generalization_error_erm: float = error(ytest,yhat_gd)
        self.adversarial_generalization_error_erm: float = adversarial_error(ytest,Xtest,w_gd,epsilon)
        self.generalization_error_overlap: float = generalization_error(self.rho,self.m,self.Q, tau)
        self.adversarial_generalization_error_overlap: float = adversarial_generalization_error_logistic(self.m,self.Q,self.rho,tau,epsilon)
        yhat_gd_train = theoretical.predict_erm(Xtrain,w_gd)
        self.date: datetime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.chosen_minimizer: str = minimizer_name
        self.training_error: float = error(y,yhat_gd_train)
        self.training_loss: float = pure_training_loss(w_gd,Xtrain,y,epsilon)
        self.d: int = d
        self.tau: float = tau
        self.alpha: float = n/d
        self.A: float = A
        self.N: float = N

        self.analytical_calibrations: CalibrationResults = analytical_calibrations
        self.erm_calibrations: CalibrationResults = erm_calibrations

        self.test_loss: float = pure_training_loss(w_gd,Xtest,ytest,epsilon)

    # overwrite the to string method to print all attributes and their type
    def __str__(self):
        # return for each attribute the content and the type
        return "\n".join(["%s: %s (%s)" % (key, value, type(value)) for key, value in self.__dict__.items()])

# How to change a column later:
# cursor.execute('ALTER TABLE experiments ADD COLUMN new_column TEXT')

DB_NAME = "experiments.db"
STATE_EVOLUTION_TABLE = "state_evolution"
ERM_TABLE = "erm"
EXPERIMENTS_TABLE = "experiments"


class DatabaseHandler:
    def __init__(self, logger, db_name=DB_NAME):
        self.connection = sqlite3.connect(db_name,timeout=15)
        self.cursor = self.connection.cursor()
        self.logger = logger

        # test if all tables exist and create them if not
        self.cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{STATE_EVOLUTION_TABLE}'")
        if self.cursor.fetchone() is None:
            self.logger.info(f"Creating table {STATE_EVOLUTION_TABLE} ...")
            self.cursor.execute(f'''
                CREATE TABLE {STATE_EVOLUTION_TABLE} (
                    id TEXT PRIMARY KEY,
                    code_version TEXT,
                    duration REAL,
                    experiment_id TEXT,
                    generalization_error REAL,
                    adversarial_generalization_error REAL,
                    training_loss REAL,
                    training_error REAL,
                    date TEXT,
                    sigma REAL,
                    q REAL,
                    m REAL,
                    cosb REAL,
                    initial_condition BLOB,
                    rho_w_star REAL,
                    alpha REAL,
                    epsilon REAL,
                    tau REAL,
                    lam REAL,
                    calibrations BLOB,
                    abs_tol REAL,
                    min_iter INTEGER,
                    max_iter INTEGER,
                    blend_fpe REAL,
                    int_lims REAL,
                    sigma_hat REAL,
                    q_hat REAL,
                    m_hat REAL,
                    A REAL,
                    N REAL,
                    A_hat REAL,
                    N_hat REAL
                )
            ''')
            self.connection.commit()

        # For the experiments table
        self.cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{EXPERIMENTS_TABLE}'")
        if self.cursor.fetchone() is None:
            self.logger.info(f"Creating table {EXPERIMENTS_TABLE} ...")
            self.cursor.execute(f'''CREATE TABLE {EXPERIMENTS_TABLE} (
                experiment_id TEXT PRIMARY KEY,
                experiment_name TEXT,
                duration REAL,
                code_version TEXT,
                date TEXT,
                state_evolution_repetitions INTEGER,
                erm_repetitions INTEGER,
                alphas BLOB,
                epsilons BLOB,
                lambdas BLOB,
                taus BLOB,
                ps BLOB,
                dp REAL,
                d INTEGER,
                p INTEGER,
                erm_methods BLOB,
                completed BOOLEAN,
                data_model_type TEXT
            )''')
            self.connection.commit()

        # For the erm table
        self.cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{ERM_TABLE}'")
        if self.cursor.fetchone() is None:
            self.logger.info(f"Creating table {ERM_TABLE} ...")
            self.cursor.execute(f'''
                CREATE TABLE {ERM_TABLE} (
                    id TEXT PRIMARY KEY,
                    duration REAL,
                    code_version TEXT,
                    experiment_id TEXT,
                    Q REAL,
                    rho REAL,
                    m REAL,
                    cosb REAL,
                    epsilon REAL,
                    lam REAL,
                    generalization_error_erm REAL,
                    generalization_error_overlap REAL,
                    adversarial_generalization_error_erm REAL,
                    adversarial_generalization_error_overlap REAL,
                    date TEXT,
                    chosen_minimizer TEXT,
                    training_error REAL,
                    training_loss REAL,
                    d INTEGER,
                    tau REAL,
                    alpha REAL,
                    analytical_calibrations BLOB,
                    erm_calibrations BLOB,
                    test_loss REAL,
                    A REAL,
                    N REAL
                )
            ''')
            self.connection.commit()

    def insert_experiment(self, experiment_information: ExperimentInformation):
        # self.logger.info(str(experiment_information))
        self.cursor.execute(f'''
        INSERT INTO {EXPERIMENTS_TABLE} VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''', (
            experiment_information.experiment_id,
            experiment_information.experiment_name,
            experiment_information.duration,
            experiment_information.code_version,
            experiment_information.date,
            experiment_information.state_evolution_repetitions,
            experiment_information.erm_repetitions,
            json.dumps(experiment_information.alphas, cls=NumpyEncoder),
            json.dumps(experiment_information.epsilons, cls=NumpyEncoder),
            json.dumps(experiment_information.lambdas, cls=NumpyEncoder),
            json.dumps(experiment_information.taus, cls=NumpyEncoder),
            json.dumps(experiment_information.ps, cls=NumpyEncoder),
            float(experiment_information.dp),
            experiment_information.d,
            experiment_information.p,
            json.dumps(experiment_information.erm_methods),
            experiment_information.completed,
            experiment_information.data_model_type.name,
            ))
        self.connection.commit()

    def complete_experiment(self, experiment_id: str, duration: float):
        self.cursor.execute(f"UPDATE {EXPERIMENTS_TABLE} SET completed=1, duration={duration} WHERE experiment_id='{experiment_id}'")
        self.connection.commit()

    def insert_state_evolution(self, experiment_information: StateEvolutionExperimentInformation):
        self.cursor.execute(f'''
        INSERT INTO {STATE_EVOLUTION_TABLE} VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''', (
            experiment_information.id,
            experiment_information.code_version,
            experiment_information.duration,
            experiment_information.experiment_id,
            experiment_information.generalization_error,
            experiment_information.adversarial_generalization_error,
            experiment_information.training_loss,
            experiment_information.training_error,
            experiment_information.date,
            experiment_information.sigma,
            experiment_information.q,
            experiment_information.m,
            experiment_information.cosb,
            json.dumps(experiment_information.initial_condition),
            experiment_information.rho_w_star,
            float(experiment_information.alpha),
            float(experiment_information.epsilon),
            float(experiment_information.tau),
            float(experiment_information.lam),
            experiment_information.calibrations.to_json(),
            experiment_information.abs_tol,
            experiment_information.min_iter,
            experiment_information.max_iter,
            experiment_information.blend_fpe,
            experiment_information.int_lims,
            experiment_information.sigma_hat,
            experiment_information.q_hat,
            experiment_information.m_hat,
            experiment_information.A,
            experiment_information.N,
            experiment_information.A_hat,
            experiment_information.N_hat
        ))
        self.connection.commit()

    def delete_incomplete_experiments(self):
        # get experiment ids from incomplete experiments
        self.cursor.execute(f"SELECT experiment_id FROM {EXPERIMENTS_TABLE} WHERE completed=0")
        incomplete_experiment_ids = [row[0] for row in self.cursor.fetchall()]

        # delete incomplete experiments from state evolution table
        for experiment_id in incomplete_experiment_ids:
            self.cursor.execute(f"DELETE FROM {STATE_EVOLUTION_TABLE} WHERE experiment_id='{experiment_id}'")

        # delete incomplete experiments from erm table
        for experiment_id in incomplete_experiment_ids:
            self.cursor.execute(f"DELETE FROM {ERM_TABLE} WHERE experiment_id='{experiment_id}'")

        # delete incomplete experiments from experiments table
        self.cursor.execute(f"DELETE FROM {EXPERIMENTS_TABLE} WHERE completed=0")
        self.connection.commit()

    def insert_erm(self, experiment_information: ERMExperimentInformation):
        # self.logger.info(str(experiment_information))
        self.cursor.execute(f'''
        INSERT INTO {ERM_TABLE} VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            experiment_information.id,
            experiment_information.duration,
            experiment_information.code_version,
            experiment_information.experiment_id,
            experiment_information.Q,
            experiment_information.rho,
            experiment_information.m,
            experiment_information.cosb,
            float(experiment_information.epsilon),
            float(experiment_information.lam),
            experiment_information.generalization_error_erm,
            experiment_information.generalization_error_overlap,
            experiment_information.adversarial_generalization_error_erm,
            experiment_information.adversarial_generalization_error_overlap,
            experiment_information.date,
            experiment_information.chosen_minimizer,
            experiment_information.training_error,
            experiment_information.training_loss,
            experiment_information.d,
            float(experiment_information.tau),
            float(experiment_information.alpha),
            experiment_information.analytical_calibrations.to_json(),
            experiment_information.erm_calibrations.to_json(),
            experiment_information.test_loss,
            experiment_information.A,
            experiment_information.N
        ))
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

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj,ERMExperimentInformation):
            return obj.__dict__
        if isinstance(obj, CalibrationResults):
            return obj.__dict__
        if isinstance(obj,np.int32):
            return str(obj)
        if isinstance(obj, DataModelType):
            return obj.name
        return json.JSONEncoder.default(self, obj)
    
class NumpyDecoder(json.JSONDecoder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def decode(self, s, _w=json.decoder.WHITESPACE.match):
        # Parse the JSON string into a Python object
        obj = super().decode(s, _w)

        # Check if the 'data_model_type' field is present in the object
        if 'data_model_type' in obj:
            # Get the value of 'data_model_type'
            data_model_type_str = obj['data_model_type']

            # Map the string value to the enumeration type
            data_model_type = DataModelType[data_model_type_str]

            # Replace the string value with the enumeration type
            obj['data_model_type'] = data_model_type

        return obj

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    with DatabaseHandler(logging) as dbHandler:
        # create random experiment information
        experiment_information = ExperimentInformation(5,5,[0.1,0.2,0.4],[0.1,0.2,0.4],[0.1,0.2,0.4],0.1,10,["test","test2"],"test")
        id = experiment_information.experiment_id
        dbHandler.insert_experiment(experiment_information)
        # create random state evolution experiment information
        state_evolution_exp = StateEvolutionExperimentInformation(id,100,0.1,0.1,0.1,(0.1,0.1,0.1),0.1,0.1,0.1,0.1,0.1,0,0,0,0)
        dbHandler.insert_state_evolution(state_evolution_exp)

        # create random erm experiment information
        # sample test data
        w = sample_weights(10)
        X, y = sample_training_data(w,10,100,2)
        w2 = sample_weights(10)
        X2, y2 = sample_training_data(w2,10,100,2)
        erm_exp = ERMExperimentInformation(id,100,X2,w2,0.1,y,X,w,y2,10,"test",0.1,0.1)
        dbHandler.insert_erm(erm_exp)

        # set experiment to completed
        # dbHandler.complete_experiment(id,100)
        dbHandler.delete_incomplete_experiments()

