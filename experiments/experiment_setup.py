import os
import inspect
import sys
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
from data_model import DataModelType
from experiment_information import NumpyEncoder, ExperimentInformation, ExperimentType, ProblemType
import json
import numpy as np
from helpers import *

import logging
logger = logging.getLogger()
# Make the logger log to console
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)

data_model_name = ""
data_model_description = "" # Don't remove this, the next definition is ment to be commented out...

feature_ratios = None
features_x = None
features_theta = None

"""
------------------------------------------------------------------------------------------------------------------------
    Start Here, and define your data_model
------------------------------------------------------------------------------------------------------------------------
"""
d = 1000

# data_model_type = DataModelType.VanillaGaussian
# data_model_name = "VanillaGaussian"
# data_model_description = "A Data-Model with Identity Gaussians for all the covariances."
# Sigma_w = np.eye(d)
# Sigma_delta = np.eye(d)
# Sigma_upsilon = np.eye(d)


"""
-------------------
"""

# data_model_type = DataModelType.MarginGaussian
# data_model_name = "DirectedMarginGaussian"
# data_model_description = "A Data-Model with Gaussian Mixtures with a certain margin towards the e1 teacher."
# Sigma_w = np.eye(d)
# Sigma_delta = np.zeros((d,d))
# Sigma_delta[0,0] = 1


"""
-------------------
"""

# data_model_type = DataModelType.SourceCapacity
# data_model_name = "SourceCapacity_IDStudentPrior"
# data_model_description = "Source Capacity data model with Identity Gaussian Prior and a teacher with alpha = 1.2, r = 0.3"
# Sigma_w = np.eye(d)
# Sigma_delta = np.eye(d)

"""
-------------------
"""

# data_model_type = DataModelType.SourceCapacity
# data_model_name = "SourceCapacity_PowerLawStudentPrior"
# data_model_description = "Source Capacity data model with alpha Prior with alpha=1.4 and a teacher with alpha = 1.2, r = 0.3"
# Sigma_w = power_law_diagonal_matrix(d, 1.4)
# Sigma_delta = np.eye(d)

"""
-------------------
"""


# data_model_type = DataModelType.SourceCapacity
# data_model_name = "SourceCapacity_IDStudentPriorFirstFeatureSigmaDelta"
# data_model_description = "Source Capacity data model with Identity Gaussian Prior and a teacher with alpha = 1.2, r = 0.3, Sigma Delta selecting the first feature only"
# Sigma_w = np.eye(d)
# Sigma_delta = np.diag(np.zeros(d))
# Sigma_delta[0,0] = 1

"""
-------------------
"""


# data_model_type = DataModelType.SourceCapacity
# data_model_name = "SourceCapacity_IDStudentPriorFirstFeatureSigmaDelta"
# data_model_description = "Source Capacity data model with Identity Gaussian Prior and a teacher with alpha = 1.2, r = 0.3, Sigma Delta selecting the first feature only"
# Sigma_w = np.eye(d)
# Sigma_delta = np.diag(np.zeros(d))
# Sigma_delta[0,0] = 1

"""
-------------------
"""


data_model_type = DataModelType.KFeaturesModel
feature_ratios = np.array([0.5,0.5])
features_x = np.array([2,1])
features_theta = np.array([1,1])
# data_model_name = f"KFeaturesModel_TwoFeatures_ProtectingFirst_AttackingSecond_{feature_ratios}_{features_x}_{features_theta}"
data_model_name = ""
data_model_description = "2 Features, Theta Identity, Sigma_delta Identity"
Sigma_w = np.eye(d)
Sigma_delta = np.eye(d)
# only attack the first half
Sigma_delta[0:int(d/2),0:int(d/2)] = 10*np.eye(int(d/2))
Sigma_upsilon = np.eye(d)
# only attack the second half
Sigma_upsilon[int(d/2):d,int(d/2):d] = 10*np.eye(int(d/2))

experiment_filename = "sweep_experiment.json"

"""
------------------------------------------------------------------------------------------------------------------------
    Next, define your experiment.
------------------------------------------------------------------------------------------------------------------------
"""


# # Create a SweepExperiment
def get_default_experiment():
    state_evolution_repetitions: int = 1
    erm_repetitions: int = 2
    alphas: np.ndarray = np.linspace(0.5,2.0,2) #np.linspace(0.1,10,15) #
    epsilons: np.ndarray = np.array([0,0.4]) # np.linspace(0,0.6,2) # np.array([0.0,0.2]) # np.array([0,0.1,0.3,0.4,0.5]) 
    lambdas: np.ndarray = np.array([0.01,10]) # np.logspace(-1,2,1) #np.concatenate([-np.logspace(-4,-1,10),np.logspace(-6,-3,2)])  #np.array([-0.0001])
    taus: np.ndarray = np.array([0])
    ps: np.ndarray = None # np.array([0.6,0.75,0.9])
    dp: float = 0.01
    # round the lambdas, epsilons and alphas for 4 digits
    alphas = np.round(alphas,4)
    test_against_epsilons: np.ndarray = np.array([0.2])
    epsilons = np.round(epsilons,4)
    lambdas = np.round(lambdas,4)
    test_against_epsilons = np.round(test_against_epsilons,4)
    experiment_type: ExperimentType = ExperimentType.OptimalLambdaAdversarialTestError    
    experiment_name: str = f"Sweep Alpha - {data_model_type.name} - {data_model_name} - {data_model_description}"
    problem_types: list[ProblemType] = [ProblemType.Logistic]
    gamma_fair_error: float = 0.01
    experiment = ExperimentInformation(state_evolution_repetitions,erm_repetitions,alphas,epsilons,lambdas,taus,d,experiment_type,ps,dp, data_model_type,data_model_name, data_model_description, test_against_epsilons,problem_types,gamma_fair_error, experiment_name)
    return experiment
experiment = get_default_experiment()

# Log the experiment object
logger.info(f"Experiment: {experiment}")

"""
------------------------------------------------------------------------------------------------------------------------
    Now let the experiment create the data_model and save the experiment to a json file.
------------------------------------------------------------------------------------------------------------------------
"""


try:
    # Force a creation of a new data_model
    experiment.get_data_model(logger,source_pickle_path="../",delete_existing=True, Sigma_w=Sigma_w, Sigma_delta=Sigma_delta, Sigma_upsilon=Sigma_upsilon, name=data_model_name, description=data_model_description, feature_ratios=feature_ratios, features_x=features_x, features_theta=features_theta)
except Exception as e:
    # if you overwrite an existing data_model, you will get an exception. Still overwrite an experiment definition to ensure that you can run the experiment with the existing data_model.
    logger.info(f"Exception '{e}' occured.")
    # Log the exception stack trace
    # logger.exception(e)

# use json dump to save the experiment parameters
with open(experiment_filename,"w") as f:
    # use the NumpyEncoder to encode numpy arrays
    # Let's produce some json
    json_string = json.dumps(experiment,cls= NumpyEncoder)
    # now write it
    f.write(json_string)

logger.info(f"Succesfully saved experiment to {experiment_filename}. You can run it even if we did not recreate the data_model!")

# # Start the MPI
# import subprocess
# # Run this command
# # mpiexec -n 2 python sweep.py sweep_experiment.json
# subprocess.run(["mpiexec","-n","4","python","sweep.py","sweep_experiment.json"])