from ERM import compute_experimental_teacher_calibration, adversarial_error_teacher, fair_adversarial_error_erm
from state_evolution import generalization_error, overlap_calibration, adversarial_generalization_error_overlaps, OverlapSet, adversarial_generalization_error_overlaps_teacher, LogisticObservables, RidgeObservables, fair_adversarial_error_overlaps
from helpers import *
from ERM import predict_erm, error, adversarial_error
import numpy as np
from _version import __version__
from typing import Tuple
import datetime
import uuid
from data import *
import json
import sqlite3
import pandas as pd
from data_model import *
import ast

# create an ExperimentType enum
class ExperimentType(Enum):
    Sweep = 0
    OptimalLambda = 1
    SweepAtOptimalLambda = 2
    OptimalEpsilon = 3
    OptimalLambdaAdversarialTestError = 4



class ExperimentInformation:
    def __init__(self, state_evolution_repetitions: int, erm_repetitions: int, alphas: np.ndarray, epsilons: np.ndarray, lambdas: np.ndarray, taus: np.ndarray, d: int, experiment_type: ExperimentType, ps: np.ndarray, dp: float, data_model_type: DataModelType, data_model_name: str, data_model_description: str, test_against_largest_epsilon: bool, problem_types: list[ProblemType],  gamma_fair_error: float, experiment_name: str = ""):
        self.experiment_id: str = str(uuid.uuid4())
        self.experiment_name: str = experiment_name
        self.duration: float = 0.0
        self.code_version: str = __version__
        self.date: datetime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.state_evolution_repetitions: int = state_evolution_repetitions
        self.erm_repetitions: int = erm_repetitions
        self.alphas: np.ndarray = alphas
        self.epsilons: np.ndarray = epsilons
        self.test_against_largest_epsilon: bool = test_against_largest_epsilon
        self.lambdas: np.ndarray = lambdas
        self.taus: np.ndarray = taus
        self.ps: np.ndarray = ps
        self.dp: float = dp
        self.d: int = d
        self.experiment_type: ExperimentType = experiment_type
        self.completed: bool = False
        self.data_model_type: DataModelType = data_model_type
        self.data_model_name: str = data_model_name
        self.data_model_description: str = data_model_description
        self.problem_types: list[ProblemType] = problem_types
        self.gamma_fair_error: float = gamma_fair_error

    @classmethod
    def fromdict(cls, d):
        return cls(**d)
    
    @classmethod
    def from_df(cls, df):
        experiment_dict = df.to_dict()
        # store experiment_id and code_version
        experiment_id = experiment_dict["experiment_id"]
        code_version = experiment_dict["code_version"]
        experiment_date = experiment_dict["date"]
        duration = experiment_dict["duration"]
        completed = experiment_dict["completed"]
        # remove the entries for experiment_id, code_version and date
        experiment_dict.pop("experiment_id")
        experiment_dict.pop("code_version")
        experiment_dict.pop("date")
        experiment_dict.pop("duration")
        experiment_dict.pop("completed")
        experiment = cls(**experiment_dict)
        # restore experiment_id and code_version
        experiment.experiment_id = experiment_id
        experiment.code_version = code_version
        experiment.date = experiment_date
        experiment.duration = duration
        experiment.completed = completed

        experiment.data_model_type = DataModelType[experiment.data_model_type]
        experiment.experiment_type = ExperimentType[experiment.experiment_type]
        experiment.problem_types = [ProblemType[problem_type] for problem_type in ast.literal_eval(experiment.problem_types)]


        return experiment

    # overwrite the to string method to print all attributes and their type
    def __str__(self):
        # return for each attribute the content and the type
        return "\n".join(["%s: %s (%s)" % (key, value, type(value)) for key, value in self.__dict__.items()])
    
    
    def get_data_model(self, logger, source_pickle_path = "../", delete_existing = False, Sigma_w = None, Sigma_delta = None, Sigma_upsilon = None, name: str = "", description: str = "", feature_ratios: np.ndarray = None, features_x: np.ndarray = None, features_theta: np.ndarray = None):
        """
        Instantiates a data model of the type specified in self.data_model_type and stores it to source_pickle_path,
        custom student prior covariances and adversarial training covariances can be specified
        parameters:
        logger: a logger object
        source_pickle_path: the path where the data model should be stored
        delete_existing: if true, the data model will be recomputed even if a pickle exists
        Sigma_w: the covariance matrix of the student prior
        Sigma_delta: the covariance matrix of the adversarial training 
        ----------------
        Both Sigma_w and Sigma_delta only have effect if the data model is newly created, that is either no pickle exists or delete_existing is True
        ----------------
        returns:
        the data model
        """
        data_model = None
        if self.data_model_type == DataModelType.VanillaGaussian:
            data_model = VanillaGaussianDataModel(self.d,logger,source_pickle_path=source_pickle_path,delete_existing=delete_existing, Sigma_w=Sigma_w, Sigma_delta=Sigma_delta, Sigma_upsilon=Sigma_upsilon, name=name, description=description)
        elif self.data_model_type == DataModelType.SourceCapacity:
            data_model = SourceCapacityDataModel(self.d, logger, source_pickle_path=source_pickle_path, delete_existing=delete_existing, Sigma_w=Sigma_w, Sigma_delta=Sigma_delta, Sigma_upsilon=Sigma_upsilon, name=name, description=description)
        elif self.data_model_type == DataModelType.MarginGaussian:
            data_model = MarginGaussianDataModel(self.d,logger, source_pickle_path=source_pickle_path, delete_existing=delete_existing, Sigma_w=Sigma_w, Sigma_delta=Sigma_delta, Sigma_upsilon=Sigma_upsilon, name=name, description=description)
        elif self.data_model_type == DataModelType.KFeaturesModel:
            data_model = KFeaturesModel(self.d,logger, source_pickle_path=source_pickle_path, delete_existing=delete_existing, Sigma_w=Sigma_w, Sigma_delta=Sigma_delta, Sigma_upsilon=Sigma_upsilon, name=name, description=description, feature_ratios = feature_ratios, features_x = features_x, features_theta = features_theta)
        else:
            raise Exception("Unknown DataModelType, did you remember to add the initialization?")
        return data_model

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
    def __init__(self, task,overlaps, data_model, logger):

        if task.problem_type == ProblemType.Ridge:
            observables = RidgeObservables()
        elif task.problem_type == ProblemType.Logistic:
            observables = LogisticObservables()
        else:
            raise Exception(f"Problem type {task.problem_type.name} not implemented")

        # let's compute and store the calibrations
        calibrations = []
        if task.ps is not None:
            for p in task.ps:
                calibrations.append(overlap_calibration(data_model.rho,p,overlaps.m,overlaps.q,task.tau))
        calibration_results = CalibrationResults(task.ps,calibrations,None)
        self.calibrations: CalibrationResults = calibration_results

        # store basic information
        self.problem_type: ProblemType = task.problem_type
        self.id: str = str(uuid.uuid4())
        self.code_version: str = __version__
        self.duration: float = None # the duration is set in sweep after all errors have been computed.
        self.experiment_id: str = task.experiment_id
        self.date: datetime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # State Evolution Parameters
        self.initial_condition: Tuple[float, float, float] = overlaps.INITIAL_CONDITION
        self.abs_tol: float = overlaps.TOL_FPE
        self.min_iter: int = overlaps.MIN_ITER_FPE
        self.max_iter: int = overlaps.MAX_ITER_FPE
        self.blend_fpe: float = overlaps.BLEND_FPE
        self.int_lims: float = overlaps.INT_LIMS

        # Experiment Parameters
        self.alpha: float = task.alpha
        self.epsilon: float = task.epsilon
        self.test_against_epsilon: float = task.test_against_epsilon
        self.tau: float = task.tau
        self.lam: float = task.lam

        # Generalization Error
        self.generalization_error: float = generalization_error(data_model.rho, overlaps.m, overlaps.q, task.tau)

        # Adversarial Generalization Error
        self.adversarial_generalization_error: float = adversarial_generalization_error_overlaps(overlaps, task, data_model)
        self.adversarial_generalization_error_teacher: float = adversarial_generalization_error_overlaps_teacher(overlaps, task, data_model)
        
        self.fair_adversarial_error: float = fair_adversarial_error_overlaps(overlaps, task, data_model,task.gamma_fair_error,logger)

        # Training Error
        self.training_error: float = observables.training_error(task, overlaps, data_model, self.int_lims)

        # Loss
        self.training_loss: float = observables.training_loss(task, overlaps, data_model, self.int_lims)
        self.test_loss: float = observables.test_loss(task, overlaps, data_model, self.int_lims)
        
        
        # Overlaps
        self.sigma: float = overlaps.sigma
        self.q: float = overlaps.q
        self.Q_self: float = overlaps.sigma + overlaps.q
        self.m: float = overlaps.m
        self.rho: float = data_model.rho
        self.A : float = overlaps.A
        self.N : float = overlaps.N
        self.P : float = overlaps.P
        self.F : float = overlaps.F
        

        # Hat overlaps
        self.sigma_hat : float = overlaps.sigma_hat
        self.q_hat : float = overlaps.q_hat
        self.m_hat : float = overlaps.m_hat
        self.A_hat : float = overlaps.A_hat
        self.N_hat : float = overlaps.N_hat

        
        # Angle
        self.angle: float = self.m / np.sqrt((self.q)*data_model.rho)
        
        # Robustness
        self.robustness: float = 1 - self.adversarial_generalization_error
        self.teacher_robustness: float = 1 - self.adversarial_generalization_error_teacher

class ERMExperimentInformation:
    def __init__(self, task, data_model, data: DataSet, weights, problem_instance, logger):

        # let's compute and store the overlaps      
        self.Q = weights.dot(data_model.Sigma_x@weights) / task.d
        self.A: float = weights.dot(data_model.Sigma_upsilon @ weights) / task.d
        self.N: float = weights.dot(weights) / task.d
        self.P: float = weights.dot(data_model.Sigma_delta@weights) / task.d
        

        # let's calculate the calibration
        analytical_calibrations = []
        erm_calibrations = []
        if data.theta is not None:           
            
            self.rho: float = data.theta.dot(data_model.Sigma_x@data.theta) / task.d
            self.m = weights.dot(data_model.Sigma_x@data.theta) / task.d
            self.F: float = weights.dot(data_model.Sigma_upsilon@data.theta) / task.d

            # We cannot compute the calibration if we don't know the ground truth.    
            if task.ps is not None:
                for p in task.ps:
                
                    analytical_calibrations.append(overlap_calibration(data_model.rho,p,self.m,self.Q,task.tau))        
                    erm_calibrations.append(compute_experimental_teacher_calibration(p,data.theta,weights,data.X_test,task.tau))
        else:
            self.rho: float = data_model.rho
            self.m: float = None

        analytical_calibrations_result = CalibrationResults(task.ps,analytical_calibrations,None)
        erm_calibrations_result = CalibrationResults(task.ps,erm_calibrations,task.dp)
        self.analytical_calibrations: CalibrationResults = analytical_calibrations_result
        self.erm_calibrations: CalibrationResults = erm_calibrations_result

        overlaps = OverlapSet()
        overlaps.A = self.A
        overlaps.N = self.N
        overlaps.m = self.m
        overlaps.q = self.Q
        overlaps.P = self.P
        overlaps.F = self.F

        # store basic information
        self.problem_type: ProblemType = task.problem_type
        self.id: str = str(uuid.uuid4())
        self.duration : float = None # the duration is set in sweep after all errors have been computed.
        self.code_version: str = __version__
        self.experiment_id: str = task.experiment_id
        self.date: datetime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.chosen_minimizer: str = "sklearn"

        # store the experiment parameters
        self.epsilon: float = task.epsilon
        self.test_against_epsilon: float = task.test_against_epsilon
        self.lam: float = task.lam
        self.d: int = task.d
        self.tau: float = task.tau
        n: int = data.X.shape[0]
        self.alpha: float = n/task.d

        # Angle
        self.angle: float = self.m / np.sqrt(self.Q*self.rho)

        # Generalization Error        
        yhat_gd = predict_erm(data.X_test,weights)
        self.generalization_error_erm: float = error(data.y_test,yhat_gd)
        self.generalization_error_overlap: float = generalization_error(self.rho,self.m,self.Q, task.tau)
        

        # Adversarial Generalization Error
        self.adversarial_generalization_error: float = adversarial_error(data.y_test,data.X_test,weights,task.test_against_epsilon, data_model.Sigma_upsilon)
        self.adversarial_generalization_error_teacher: float = adversarial_error_teacher(data.y_test,data.X_test,weights,data.theta,task.test_against_epsilon,data_model)
        self.adversarial_generalization_error_overlap: float = adversarial_generalization_error_overlaps(overlaps, task, data_model)
        self.fair_adversarial_error: float = fair_adversarial_error_erm(data.y_test,data.X_test,weights,data.theta,task.test_against_epsilon,task.gamma_fair_error,data_model, logger)
        
        # Training Error
        yhat_gd_train = predict_erm(data.X,weights)
        self.training_error: float = error(data.y,yhat_gd_train)       
        

        # Loss
        self.training_loss: float = problem_instance.training_loss(weights,data.X,data.y,task.epsilon,Sigma_delta=data_model.Sigma_delta)       
        self.test_loss: float = problem_instance.training_loss(weights,data.X_test,data.y_test,task.test_against_epsilon, Sigma_delta=data_model.Sigma_upsilon)

        # Robustness
        self.robustness: float = 1 - self.adversarial_generalization_error
        self.teacher_robustness: float = 1 - self.adversarial_generalization_error_teacher
        


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
                    problem_type TEXT,
                    experiment_id TEXT,
                    generalization_error REAL,
                    adversarial_generalization_error REAL,
                    adversarial_generalization_error_teacher REAL,
                    fair_adversarial_error REAL,
                    training_loss REAL,
                    training_error REAL,
                    date TEXT,
                    sigma REAL,
                    q REAL,
                    Q_self REAL,
                    m REAL,
                    angle REAL,
                    initial_condition BLOB,
                    rho REAL,
                    alpha REAL,
                    epsilon REAL,
                    test_against_epsilon REAL,
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
                    P REAL,
                    F REAL,
                    A_hat REAL,
                    N_hat REAL,
                    test_loss REAL,
                    robustness REAL,
                    teacher_robustness REAL
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
                problem_types TEXT,
                code_version TEXT,
                date TEXT,
                state_evolution_repetitions INTEGER,
                erm_repetitions INTEGER,
                alphas BLOB,
                epsilons BLOB,
                test_against_largest_epsilon BOOLEAN, 
                lambdas BLOB,
                taus BLOB,
                ps BLOB,
                dp REAL,
                d INTEGER,
                experiment_type Text,
                completed BOOLEAN,
                data_model_type TEXT,
                data_model_name TEXT,
                data_model_description TEXT,
                gamma_fair_error REAL
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
                    problem_type TEXT,
                    code_version TEXT,
                    experiment_id TEXT,
                    Q REAL,
                    rho REAL,
                    m REAL,
                    angle REAL,
                    epsilon REAL,
                    test_against_epsilon REAL,
                    lam REAL,
                    generalization_error_erm REAL,
                    generalization_error_overlap REAL,
                    adversarial_generalization_error REAL,
                    adversarial_generalization_error_teacher REAL,
                    adversarial_generalization_error_overlap REAL,
                    fair_adversarial_error REAL,
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
                    N REAL,
                    P REAL,
                    F REAL,
                    robustness REAL,
                    teacher_robustness REAL
                )
            ''')
            self.connection.commit()

    def insert_experiment(self, experiment_information: ExperimentInformation):
        # self.logger.info(str(experiment_information))
        self.cursor.execute(f'''
        INSERT INTO {EXPERIMENTS_TABLE} VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''', (
            experiment_information.experiment_id,
            experiment_information.experiment_name,
            experiment_information.duration,
            json.dumps([problem_type.name for problem_type in experiment_information.problem_types], cls=NumpyEncoder),
            experiment_information.code_version,
            experiment_information.date,
            experiment_information.state_evolution_repetitions,
            experiment_information.erm_repetitions,
            json.dumps(experiment_information.alphas, cls=NumpyEncoder),
            json.dumps(experiment_information.epsilons, cls=NumpyEncoder),
            experiment_information.test_against_largest_epsilon,
            json.dumps(experiment_information.lambdas, cls=NumpyEncoder),
            json.dumps(experiment_information.taus, cls=NumpyEncoder),
            json.dumps(experiment_information.ps, cls=NumpyEncoder),
            float(experiment_information.dp),
            experiment_information.d,
            experiment_information.experiment_type.name,
            experiment_information.completed,
            experiment_information.data_model_type.name,
            experiment_information.data_model_name,
            experiment_information.data_model_description,
            experiment_information.gamma_fair_error
            ))
        self.connection.commit()

    def complete_experiment(self, experiment_id: str, duration: float):
        self.cursor.execute(f"UPDATE {EXPERIMENTS_TABLE} SET completed=1, duration={duration} WHERE experiment_id='{experiment_id}'")
        self.connection.commit()

    def insert_state_evolution(self, experiment_information: StateEvolutionExperimentInformation):
        self.cursor.execute(f'''
        INSERT INTO {STATE_EVOLUTION_TABLE} VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''', (
            experiment_information.id,
            experiment_information.code_version,
            experiment_information.duration,
            experiment_information.problem_type.name,
            experiment_information.experiment_id,
            experiment_information.generalization_error,
            experiment_information.adversarial_generalization_error,
            experiment_information.adversarial_generalization_error_teacher,
            experiment_information.fair_adversarial_error,
            experiment_information.training_loss,
            experiment_information.training_error,
            experiment_information.date,
            experiment_information.sigma,
            experiment_information.q,
            experiment_information.Q_self,
            experiment_information.m,
            experiment_information.angle,
            json.dumps(experiment_information.initial_condition),
            experiment_information.rho,
            float(experiment_information.alpha),
            float(experiment_information.epsilon),
            float(experiment_information.test_against_epsilon),
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
            experiment_information.P,
            experiment_information.F,
            experiment_information.A_hat,
            experiment_information.N_hat,
            experiment_information.test_loss,
            experiment_information.robustness,
            experiment_information.teacher_robustness
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
        INSERT INTO {ERM_TABLE} VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            experiment_information.id,
            experiment_information.duration,
            experiment_information.problem_type.name,
            experiment_information.code_version,
            experiment_information.experiment_id,
            experiment_information.Q,
            experiment_information.rho,
            experiment_information.m,
            experiment_information.angle,
            float(experiment_information.epsilon),
            float(experiment_information.test_against_epsilon),
            float(experiment_information.lam),
            experiment_information.generalization_error_erm,
            experiment_information.generalization_error_overlap,
            experiment_information.adversarial_generalization_error,
            experiment_information.adversarial_generalization_error_teacher,
            experiment_information.adversarial_generalization_error_overlap,
            experiment_information.fair_adversarial_error,
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
            experiment_information.N,
            experiment_information.P,
            experiment_information.F,
            experiment_information.robustness,
            experiment_information.teacher_robustness
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
    # c.execute('SELECT * FROM experiments WHERE result > 0.0')
    # results = c.fetchall()
    # for row in results:
    #     print(row)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.connection.close()

"""
------------------------------------------------------------------------------------------------------------------------
    Experiment Serialization Helpers
------------------------------------------------------------------------------------------------------------------------
"""


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj,ERMExperimentInformation):
            return obj.__dict__
        if isinstance(obj, ExperimentInformation):
            experiment = obj.__dict__
            # remove the entries for experiment_id, code_version and date
            experiment.pop("experiment_id")
            experiment.pop("code_version")
            experiment.pop("date")
            experiment.pop("duration")
            experiment.pop("completed")
            return experiment
        if isinstance(obj, CalibrationResults):
            return obj.__dict__
        if isinstance(obj,np.int32):
            return str(obj)
        if isinstance(obj, Enum):
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

        # check if the 'experiment_type' field is present in the object
        if 'experiment_type' in obj:
            # Get the value of 'experiment_type'
            experiment_type_str = obj['experiment_type']

            # Map the string value to the enumeration type
            experiment_type = ExperimentType[experiment_type_str]

            # Replace the string value with the enumeration type
            obj['experiment_type'] = experiment_type

        # check if the 'problem_types' field is present in the object
        if 'problem_types' in obj:
            # Get the value of 'problem_types'
            problem_types_str = obj['problem_types']

            # Map the string value to the enumeration type
            problem_types = [ProblemType[problem_type_str] for problem_type_str in problem_types_str]

            # Replace the string value with the enumeration type
            obj['problem_types'] = problem_types

        # check if the 'problem_type' field is present in the object
        if 'problem_type' in obj:
            # Get the value of 'problem_type'
            problem_type_str = obj['problem_type']

            # Map the string value to the enumeration type
            problem_type = ProblemType[problem_type_str]

            # Replace the string value with the enumeration type
            obj['problem_type'] = problem_type

        return obj
