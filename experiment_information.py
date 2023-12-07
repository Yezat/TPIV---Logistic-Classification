from ERM import compute_experimental_teacher_calibration, adversarial_error_teacher, fair_adversarial_error_erm
from state_evolution import generalization_error, overlap_calibration, adversarial_generalization_error_overlaps, OverlapSet, adversarial_generalization_error_overlaps_teacher, LogisticObservables, RidgeObservables, fair_adversarial_error_overlaps, var_func
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
import time
import ast

# create an ExperimentType enum
class ExperimentType(Enum):
    Sweep = 0
    OptimalLambda = 1
    SweepAtOptimalLambda = 2
    OptimalEpsilon = 3
    OptimalLambdaAdversarialTestError = 4
    SweepAtOptimalLambdaAdversarialTestError = 5

class DataModelDefinition:
    def __init__(self,d : int, delete_existing: bool, normalize_matrices: bool, Sigma_w: np.ndarray, Sigma_delta: np.ndarray, Sigma_upsilon: np.ndarray, name: str, description: str, data_model_type: DataModelType, feature_ratios: np.ndarray = None, features_x: np.ndarray = None, features_theta: np.ndarray = None) -> None:
        self.delete_existing: bool = delete_existing
        self.normalize_matrices: bool = normalize_matrices
        self.Sigma_w: np.ndarray = Sigma_w
        self.Sigma_delta: np.ndarray = Sigma_delta
        self.Sigma_upsilon: np.ndarray = Sigma_upsilon
        self.name: str = name
        self.d = d
        self.description: str = description
        self.data_model_type: DataModelType = data_model_type
        self.feature_ratios: np.ndarray = feature_ratios
        self.features_x: np.ndarray = features_x
        self.features_theta: np.ndarray = features_theta

    @classmethod
    def from_name(cls, name: str, data_model_type: DataModelType, base_path = "../"):
        """
        Loads a data model definition from a json file based on the name and the data_model_type.
        """
        filepath = f"{base_path}data/data_model_definitions/{data_model_type.name}/{name}.json"
        with open(filepath) as f:
            data_model_definition_dict = json.load(f, cls=NumpyDecoder)

        result = cls(**data_model_definition_dict)

        # make sure Sigma_w, Sigma_delta and Sigma_upsilon are numpy arrays
        result.Sigma_w = np.array(result.Sigma_w)
        result.Sigma_delta = np.array(result.Sigma_delta)
        result.Sigma_upsilon = np.array(result.Sigma_upsilon)

        # same for feature_ratios, features_x and features_theta
        if result.feature_ratios is not None:
            result.feature_ratios = np.array(result.feature_ratios)
        if result.features_x is not None:
            result.features_x = np.array(result.features_x)
        if result.features_theta is not None:
            result.features_theta = np.array(result.features_theta)

        return result
    
    def store(self, base_path = "../"):
        """
        Stores the data model definition in a json file.
        """
        filepath = f"{base_path}data/data_model_definitions/{self.data_model_type.name}/{self.name}.json"
        # make sure the directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(self.__dict__, f, cls=NumpyEncoder)

class ExperimentInformation:
    def __init__(self, state_evolution_repetitions: int, erm_repetitions: int, alphas: np.ndarray, epsilons: np.ndarray, lambdas: np.ndarray, taus: np.ndarray, d: int, experiment_type: ExperimentType, ps: np.ndarray, dp: float, data_model_types: list[DataModelType], data_model_names: [str], data_model_descriptions: [str], test_against_epsilons: np.ndarray, problem_types: list[ProblemType],  gamma_fair_error: float, experiment_name: str = ""):
        self.experiment_id: str = str(uuid.uuid4())
        self.experiment_name: str = experiment_name
        self.duration: float = 0.0
        self.code_version: str = __version__
        self.date: datetime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.state_evolution_repetitions: int = state_evolution_repetitions
        self.erm_repetitions: int = erm_repetitions
        self.alphas: np.ndarray = alphas
        self.epsilons: np.ndarray = epsilons
        self.test_against_epsilons: np.ndarray = test_against_epsilons
        self.lambdas: np.ndarray = lambdas
        self.taus: np.ndarray = taus
        self.ps: np.ndarray = ps
        self.dp: float = dp
        self.d: int = d
        self.experiment_type: ExperimentType = experiment_type
        self.completed: bool = False
        self.data_model_types: [DataModelType] = data_model_types
        self.data_model_names: [str] = data_model_names
        self.data_model_descriptions: [str] = data_model_descriptions
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
        
        
        experiment.data_model_types = [DataModelType[data_model_type] for data_model_type in ast.literal_eval(experiment.data_model_types)]
        experiment.data_model_names = ast.literal_eval(experiment.data_model_names)
        experiment.data_model_descriptions = ast.literal_eval(experiment.data_model_descriptions)

        experiment.problem_types = [ProblemType[problem_type] for problem_type in ast.literal_eval(experiment.problem_types)]
        experiment.experiment_type = ExperimentType[experiment.experiment_type]


        return experiment

    # overwrite the to string method to print all attributes and their type
    def __str__(self):
        # return for each attribute the content and the type
        return "\n".join(["%s: %s (%s)" % (key, value, type(value)) for key, value in self.__dict__.items()])
    
    
    def _load_data_model(self, logger, name, model_type, source_pickle_path = "../"):
        """
        Loads an already existing data model from a pickle file based on the model_type and the name.
        You can pass a source_pickle_path to specify the path to the pickle file.
        ----------------
        returns:
        the data model
        """
        data_model = None
        # log the model type
        logger.info(f"Loading data model {name} of type {model_type.name} ...")
        if model_type == DataModelType.VanillaGaussian:
            data_model = VanillaGaussianDataModel(self.d,logger,source_pickle_path=source_pickle_path, name=name)
        elif model_type == DataModelType.SourceCapacity:
            data_model = SourceCapacityDataModel(self.d, logger, source_pickle_path=source_pickle_path,  name=name)
        elif model_type == DataModelType.MarginGaussian:
            data_model = MarginGaussianDataModel(self.d,logger, source_pickle_path=source_pickle_path,  name=name)
        elif model_type == DataModelType.KFeaturesModel:
            data_model = KFeaturesModel(self.d,logger, source_pickle_path=source_pickle_path,  name=name)
        else:
            raise Exception("Unknown DataModelType, did you remember to add the initialization?")
        return data_model

    def load_data_model_definitions(self, base_path = "../"):
        """
        Loads the data model definitions specified in the experiment_information based on existing json files.
        """
        data_model_definitions = []
        for data_model_type, data_model_name in zip(self.data_model_types, self.data_model_names):
            data_model_definitions.append(DataModelDefinition.from_name(data_model_name, data_model_type, base_path=base_path))
        return data_model_definitions


    def store_data_model_definitions(self, data_model_definitions: list[DataModelDefinition], base_path = "../"):
        """
        Stores the data model definitions in json files.
        """
        for data_model_definition in data_model_definitions:
            data_model_definition.store(base_path=base_path)


    def load_data_models(self, logger, source_pickle_path = "../"):
        """
        Loads all data models specified in the experiment_information based on existing pickle files.
        Creates a dictionary with key (data_model_type, data_model_name)
        """
        data_models = {}
        for data_model_type, data_model_name in zip(self.data_model_types, self.data_model_names):
            data_model = self._load_data_model(logger, data_model_name, data_model_type, source_pickle_path=source_pickle_path)
            data_models[(data_model_type, data_model_name)] = data_model
        return data_models

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
        self.data_model_type: DataModelType = data_model.model_type
        self.data_model_name: str = data_model.name
        self.data_model_description: str = data_model.description

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
        self.test_against_epsilons: np.ndarray = task.test_against_epsilons
        self.tau: float = task.tau
        self.lam: float = task.lam

        # Generalization Error
        self.generalization_error: float = generalization_error(data_model.rho, overlaps.m, overlaps.q, task.tau)

        # Adversarial Generalization Error
        self.adversarial_generalization_errors: np.ndarray = np.array([(eps,adversarial_generalization_error_overlaps(overlaps, task, data_model, eps) ) for eps in task.test_against_epsilons ] )
        self.adversarial_generalization_errors_teacher: np.ndarray = np.array([(eps,adversarial_generalization_error_overlaps_teacher(overlaps, task, data_model, eps) ) for eps in task.test_against_epsilons ])
        
        self.fair_adversarial_errors: float = np.array([(eps,fair_adversarial_error_overlaps(overlaps, data_model,task.gamma_fair_error,eps, logger) ) for eps in task.test_against_epsilons])

        # Training Error
        self.training_error: float = observables.training_error(task, overlaps, data_model, self.int_lims)

        # Loss
        self.training_loss: float = observables.training_loss(task, overlaps, data_model, self.int_lims)
        self.test_losses: np.ndarray = np.array([(eps,observables.test_loss(task, overlaps, data_model, eps, self.int_lims) ) for eps in task.test_against_epsilons ])
        
        
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

        # subspace overlaps
        self.subspace_overlaps = {}
        if data_model.model_type == DataModelType.KFeaturesModel:
            # it makes sense to talk about subspaces only if we have a KFeaturesModel
            feature_sizes = data_model.feature_sizes
        else:
            feature_sizes = [int(0)]
        d = task.d

        Sigmas = []
        ms = []
        Qs = []
        As = []
        Ns = []
        Ps = []
        Fs = []


        for i, size in enumerate(feature_sizes):
            slice_from = int(np.sum(feature_sizes[:i]))
            if i == len(feature_sizes)-1:
                slice_to = d
            else:
                slice_to = int(np.sum(feature_sizes[:i+1]))


            m, q, sigma, A, N, P, F = var_func(task, overlaps, data_model, logger, slice_from, slice_to)

            # log the computed overlaps
            logger.info(f"Subspace {i}: m={m}, q={q}, sigma={sigma}, A={A}, N={N}, P={P}, F={F}")
            
            Sigmas.append(sigma)
            ms.append(m)
            Qs.append(q)
            As.append(A)
            Ns.append(N)
            Ps.append(P)
            Fs.append(F)

        self.subspace_overlaps["Sigmas"] = Sigmas
        self.subspace_overlaps["ms"] = ms
        self.subspace_overlaps["qs"] = Qs
        self.subspace_overlaps["As"] = As
        self.subspace_overlaps["Ns"] = Ns
        self.subspace_overlaps["Ps"] = Ps
        self.subspace_overlaps["Fs"] = Fs

        # if there is only one subspace, add a copy of the first subspace to the second subspace
        if len(Sigmas) == 1:
            Sigmas.append(Sigmas[0])
            ms.append(ms[0])
            Qs.append(Qs[0])
            As.append(As[0])
            Ns.append(Ns[0])
            Ps.append(Ps[0])
            Fs.append(Fs[0])

        # now extract the subspace strengths (we assume there are two subspaces or one in case of vanilla gaussian data)
        subspace_strengths_X = [data_model.Sigma_x[0,0], data_model.Sigma_x[-1,-1]]
        subspace_strengths_delta = [data_model.Sigma_delta[0,0], data_model.Sigma_delta[-1,-1]]
        subspace_strengths_upsilon = [data_model.Sigma_upsilon[0,0], data_model.Sigma_upsilon[-1,-1]]
        
        # compute the ratio of each subspace overlap to its strength
        subspace_strength_ratios_Sigma = []
        subspace_strength_ratios_m = []
        subspace_strength_ratios_q = []
        subspace_strength_ratios_A = []
        subspace_strength_ratios_N = []
        subspace_strength_ratios_P = []
        subspace_strength_ratios_F = []


        for i,_ in enumerate(subspace_strengths_X):
            subspace_strength_ratios_Sigma.append(Sigmas[i]/subspace_strengths_X[i])
            subspace_strength_ratios_m.append(ms[i]/subspace_strengths_X[i])
            subspace_strength_ratios_q.append(Qs[i]/subspace_strengths_X[i])
            subspace_strength_ratios_A.append(As[i]/subspace_strengths_upsilon[i])
            subspace_strength_ratios_N.append(Ns[i]/1)
            subspace_strength_ratios_P.append(Ps[i]/subspace_strengths_delta[i])
            subspace_strength_ratios_F.append(Ps[i]/subspace_strengths_upsilon[i])
       
                   
        # for each of the subspace_strength_ratios, compute the ratio of the first to the second subspace and store it in a dict called subspace_overlaps_ratio
        self.subspace_overlaps_ratio = {}
        self.subspace_overlaps_ratio["Sigma"] = subspace_strength_ratios_Sigma[0]/subspace_strength_ratios_Sigma[1]
        self.subspace_overlaps_ratio["m"] = subspace_strength_ratios_m[0]/subspace_strength_ratios_m[1]
        self.subspace_overlaps_ratio["q"] = subspace_strength_ratios_q[0]/subspace_strength_ratios_q[1]
        self.subspace_overlaps_ratio["A"] = subspace_strength_ratios_A[0]/subspace_strength_ratios_A[1]
        self.subspace_overlaps_ratio["N"] = subspace_strength_ratios_N[0]/subspace_strength_ratios_N[1]
        self.subspace_overlaps_ratio["P"] = subspace_strength_ratios_P[0]/subspace_strength_ratios_P[1]
        self.subspace_overlaps_ratio["F"] = subspace_strength_ratios_F[0]/subspace_strength_ratios_F[1]
        



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
        self.data_model_type: DataModelType = data_model.model_type
        self.data_model_name: str = data_model.name
        self.data_model_description: str = data_model.description


        # store the experiment parameters
        self.epsilon: float = task.epsilon
        self.test_against_epsilons: float = task.test_against_epsilons
        self.lam: float = task.lam
        self.d: int = task.d
        self.tau: float = task.tau
        n: int = data.X.shape[0]
        self.alpha: float = task.alpha

        # Angle
        self.angle: float = self.m / np.sqrt(self.Q*self.rho)

        # Generalization Error        
        yhat_gd = predict_erm(data.X_test,weights)
        self.generalization_error_erm: float = error(data.y_test,yhat_gd)
        self.generalization_error_overlap: float = generalization_error(self.rho,self.m,self.Q, task.tau)
        

        # Adversarial Generalization Error
        self.adversarial_generalization_errors: np.ndarray = np.array( [(eps,adversarial_error(data.y_test,data.X_test,weights,eps, data_model.Sigma_upsilon)) for eps in task.test_against_epsilons ])
        self.adversarial_generalization_errors_teacher: np.ndarray = np.array( [(eps,adversarial_error_teacher(data.y_test,data.X_test,weights,data.theta,eps,data_model)) for eps in task.test_against_epsilons ])
        self.adversarial_generalization_errors_overlap: np.ndarray = np.array( [(eps,adversarial_generalization_error_overlaps(overlaps, task, data_model, eps)) for eps in task.test_against_epsilons ])
        self.fair_adversarial_errors: np.ndarray = np.array( [(eps,fair_adversarial_error_erm(data.X_test,weights,data.theta,eps,task.gamma_fair_error,data_model, logger)) for eps in task.test_against_epsilons ])
        
        # Training Error
        yhat_gd_train = predict_erm(data.X,weights)
        self.training_error: float = error(data.y,yhat_gd_train)       
        

        # Loss
        self.training_loss: float = problem_instance.training_loss(weights,data.X,data.y,task.epsilon,Sigma_delta=data_model.Sigma_delta)       
        self.test_losses: np.ndarray = np.array( [(eps,problem_instance.training_loss(weights,data.X_test,data.y_test,eps, Sigma_delta=data_model.Sigma_upsilon)) for eps in task.test_against_epsilons ])

        # subspace overlaps
        self.subspace_overlaps = {}
        if data_model.model_type == DataModelType.KFeaturesModel:
            # it makes sense to talk about subspaces only if we have a KFeaturesModel
            feature_sizes = data_model.feature_sizes
        else:
            feature_sizes = [int(0)]
        d = task.d

        rhos = []
        ms = []
        Qs = []
        As = []
        Ns = []
        Ps = []
        Fs = []


        for i, size in enumerate(feature_sizes):
            slice_from = int(np.sum(feature_sizes[:i]))
            if i == len(feature_sizes)-1:
                slice_to = d
            else:
                slice_to = int(np.sum(feature_sizes[:i+1]))
            
            size = slice_to - slice_from

            subspace_weights = weights[slice_from:slice_to]
            subspace_teacher_weights = data.theta[slice_from:slice_to]

            subspace_Sigma_x = data_model.Sigma_x[slice_from:slice_to,slice_from:slice_to]
            subspace_Sigma_upsilon = data_model.Sigma_upsilon[slice_from:slice_to,slice_from:slice_to]
            subspace_Sigma_delta = data_model.Sigma_delta[slice_from:slice_to,slice_from:slice_to]

            
            rho = subspace_teacher_weights.dot(subspace_Sigma_x@subspace_teacher_weights) / size
            m = subspace_teacher_weights.dot(subspace_Sigma_x@subspace_weights) / size
            F = subspace_teacher_weights.dot(subspace_Sigma_upsilon@subspace_weights) / size
            Q = subspace_weights.dot(subspace_Sigma_x@subspace_weights) / size
            A = subspace_weights.dot(subspace_Sigma_upsilon@subspace_weights) / size
            N = subspace_weights.dot(subspace_weights) / size
            P = subspace_weights.dot(subspace_Sigma_delta@subspace_weights) / size

            rhos.append(rho)
            ms.append(m)
            Qs.append(Q)
            As.append(A)
            Ns.append(N)
            Ps.append(P)
            Fs.append(F)

        self.subspace_overlaps["rhos"] = rhos
        self.subspace_overlaps["ms"] = ms
        self.subspace_overlaps["qs"] = Qs
        self.subspace_overlaps["As"] = As
        self.subspace_overlaps["Ns"] = Ns
        self.subspace_overlaps["Ps"] = Ps
        self.subspace_overlaps["Fs"] = Fs

        # if there is only one subspace, add a copy of the first subspace to the second subspace
        if len(ms) == 1:
            ms.append(ms[0])
            Qs.append(Qs[0])
            As.append(As[0])
            Ns.append(Ns[0])
            Ps.append(Ps[0])
            Fs.append(Fs[0])

        # now extract the subspace strengths (we assume there are two subspaces or one in case of vanilla gaussian data)
        subspace_strengths_X = [data_model.Sigma_x[0,0], data_model.Sigma_x[-1,-1]]
        subspace_strengths_delta = [data_model.Sigma_delta[0,0], data_model.Sigma_delta[-1,-1]]
        subspace_strengths_upsilon = [data_model.Sigma_upsilon[0,0], data_model.Sigma_upsilon[-1,-1]]

        # log the subspace strengths
        logger.info(f"Subspace strengths: X={subspace_strengths_X}, delta={subspace_strengths_delta}, upsilon={subspace_strengths_upsilon}")
        
        # compute the ratio of each subspace overlap to its strength
        subspace_strength_ratios_m = []
        subspace_strength_ratios_q = []
        subspace_strength_ratios_A = []
        subspace_strength_ratios_N = []
        subspace_strength_ratios_P = []
        subspace_strength_ratios_F = []



        for i,_ in enumerate(subspace_strengths_X):
            subspace_strength_ratios_m.append(ms[i]/subspace_strengths_X[i])
            subspace_strength_ratios_q.append(Qs[i]/subspace_strengths_X[i])
            subspace_strength_ratios_A.append(As[i]/subspace_strengths_upsilon[i])
            subspace_strength_ratios_N.append(Ns[i]/1)
            subspace_strength_ratios_P.append(Ps[i]/subspace_strengths_delta[i])
            subspace_strength_ratios_F.append(Ps[i]/subspace_strengths_upsilon[i])
       
                   
        # for each of the subspace_strength_ratios, compute the ratio of the first to the second subspace and store it in a dict called subspace_overlaps_ratio
        self.subspace_overlaps_ratio = {}
        self.subspace_overlaps_ratio["m"] = subspace_strength_ratios_m[0]/subspace_strength_ratios_m[1]
        self.subspace_overlaps_ratio["q"] = subspace_strength_ratios_q[0]/subspace_strength_ratios_q[1]
        self.subspace_overlaps_ratio["A"] = subspace_strength_ratios_A[0]/subspace_strength_ratios_A[1]
        self.subspace_overlaps_ratio["N"] = subspace_strength_ratios_N[0]/subspace_strength_ratios_N[1]
        self.subspace_overlaps_ratio["P"] = subspace_strength_ratios_P[0]/subspace_strength_ratios_P[1]
        self.subspace_overlaps_ratio["F"] = subspace_strength_ratios_F[0]/subspace_strength_ratios_F[1]

                


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
                    adversarial_generalization_errors REAL,
                    adversarial_generalization_errors_teacher REAL,
                    fair_adversarial_errors REAL,
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
                    test_against_epsilons BLOB,
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
                    test_losses REAL,
                    subspace_overlaps BLOB,
                    subspace_overlaps_ratio BLOB,
                    data_model_type TEXT,
                    data_model_name TEXT,
                    data_model_description TEXT
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
                test_against_epsilons BLOB, 
                lambdas BLOB,
                taus BLOB,
                ps BLOB,
                dp REAL,
                d INTEGER,
                experiment_type Text,
                completed BOOLEAN,
                data_model_types BLOB,
                data_model_names BLOB,
                data_model_descriptions BLOB,
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
                    q REAL,
                    rho REAL,
                    m REAL,
                    angle REAL,
                    epsilon REAL,
                    test_against_epsilons BLOB,
                    lam REAL,
                    generalization_error_erm REAL,
                    generalization_error_overlap REAL,
                    adversarial_generalization_errors BLOB,
                    adversarial_generalization_errors_teacher BLOB,
                    adversarial_generalization_errors_overlap BLOB,
                    fair_adversarial_errors BLOB,
                    date TEXT,
                    training_error REAL,
                    training_loss REAL,
                    d INTEGER,
                    tau REAL,
                    alpha REAL,
                    analytical_calibrations BLOB,
                    erm_calibrations BLOB,
                    test_losses BLOB,
                    A REAL,
                    N REAL,
                    P REAL,
                    F REAL,
                    subspace_overlaps BLOB,
                    subspace_overlaps_ratio BLOB,
                    data_model_type TEXT,
                    data_model_name TEXT,
                    data_model_description TEXT
                )
            ''')
            self.connection.commit()

    def insert_experiment(self, experiment_information: ExperimentInformation):
        for _ in range(3): # we try at most three times
            try:
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
                    json.dumps(experiment_information.test_against_epsilons, cls=NumpyEncoder),
                    json.dumps(experiment_information.lambdas, cls=NumpyEncoder),
                    json.dumps(experiment_information.taus, cls=NumpyEncoder),
                    json.dumps(experiment_information.ps, cls=NumpyEncoder),
                    float(experiment_information.dp),
                    experiment_information.d,
                    experiment_information.experiment_type.name,
                    experiment_information.completed,
                    json.dumps(experiment_information.data_model_types, cls=NumpyEncoder),
                    json.dumps(experiment_information.data_model_names, cls=NumpyEncoder),
                    json.dumps(experiment_information.data_model_descriptions, cls=NumpyEncoder),
                    experiment_information.gamma_fair_error
                    ))
                self.connection.commit()
                return 
            except sqlite3.OperationalError as e:
                # Check if the error is due to a busy database
                if "database is locked" in str(e):
                    # wait for a random amount of seconds between 1 and 5
                    sleep_time = np.random.randint(1,5)
                    self.logger.info(f"Database is locked, waiting for {sleep_time} seconds...")
                    time.sleep(sleep_time)
                else:
                    raise e
        raise Exception("Could not insert experiment into database")

    def complete_experiment(self, experiment_id: str, duration: float):
        self.cursor.execute(f"UPDATE {EXPERIMENTS_TABLE} SET completed=1, duration={duration} WHERE experiment_id='{experiment_id}'")
        # Set the date to the current date
        self.cursor.execute(f"UPDATE {EXPERIMENTS_TABLE} SET date='{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}' WHERE experiment_id='{experiment_id}'")
        self.connection.commit()

    def insert_state_evolution(self, experiment_information: StateEvolutionExperimentInformation):
        self.cursor.execute(f'''
        INSERT INTO {STATE_EVOLUTION_TABLE} VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''', (
            experiment_information.id,
            experiment_information.code_version,
            experiment_information.duration,
            experiment_information.problem_type.name,
            experiment_information.experiment_id,
            experiment_information.generalization_error,
            json.dumps(experiment_information.adversarial_generalization_errors, cls=NumpyEncoder),
            json.dumps(experiment_information.adversarial_generalization_errors_teacher, cls=NumpyEncoder),
            json.dumps(experiment_information.fair_adversarial_errors, cls=NumpyEncoder),
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
            json.dumps(experiment_information.test_against_epsilons),
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
            json.dumps(experiment_information.test_losses, cls=NumpyEncoder),
            json.dumps(experiment_information.subspace_overlaps, cls=NumpyEncoder),
            json.dumps(experiment_information.subspace_overlaps_ratio, cls=NumpyEncoder),
            experiment_information.data_model_type.name,
            experiment_information.data_model_name,
            experiment_information.data_model_description
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
        INSERT INTO {ERM_TABLE} VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
            json.dumps(experiment_information.test_against_epsilons),
            float(experiment_information.lam),
            experiment_information.generalization_error_erm,
            experiment_information.generalization_error_overlap,
            json.dumps(experiment_information.adversarial_generalization_errors, cls=NumpyEncoder),
            json.dumps(experiment_information.adversarial_generalization_errors_teacher, cls=NumpyEncoder),
            json.dumps(experiment_information.adversarial_generalization_errors_overlap, cls=NumpyEncoder),
            json.dumps(experiment_information.fair_adversarial_errors, cls=NumpyEncoder),
            experiment_information.date,
            experiment_information.training_error,
            experiment_information.training_loss,
            experiment_information.d,
            float(experiment_information.tau),
            float(experiment_information.alpha),
            experiment_information.analytical_calibrations.to_json(),
            experiment_information.erm_calibrations.to_json(),
            json.dumps(experiment_information.test_losses, cls=NumpyEncoder),
            experiment_information.A,
            experiment_information.N,
            experiment_information.P,
            experiment_information.F,
            json.dumps(experiment_information.subspace_overlaps, cls=NumpyEncoder),
            json.dumps(experiment_information.subspace_overlaps_ratio, cls=NumpyEncoder),
            experiment_information.data_model_type.name,
            experiment_information.data_model_name,
            experiment_information.data_model_description
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

        if 'data_model_types' in obj:
            # get the value of 'data_model_types'
            data_model_types_str = obj['data_model_types']

            # Map the string value to the enumeration type
            data_model_types = [DataModelType[data_model_type_str] for data_model_type_str in data_model_types_str]

            # Replace the string value with the enumeration type
            obj['data_model_types'] = data_model_types

        # check if the 'problem_type' field is present in the object
        if 'problem_type' in obj:
            # Get the value of 'problem_type'
            problem_type_str = obj['problem_type']

            # Map the string value to the enumeration type
            problem_type = ProblemType[problem_type_str]

            # Replace the string value with the enumeration type
            obj['problem_type'] = problem_type

        return obj
