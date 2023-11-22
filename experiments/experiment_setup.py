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

data_model_type = DataModelType.VanillaGaussian
data_model_name = "VanillaGaussian"
data_model_description = "A Data-Model with Identity Gaussians for all the covariances."
Sigma_w = np.eye(d)
Sigma_delta = np.eye(d)
Sigma_upsilon = np.eye(d)

########## Attacking the first half of the features ##########

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


# data_model_type = DataModelType.KFeaturesModel
# feature_ratios = np.array([0.5,0.5])
# features_x = np.array([10,1])
# features_theta = np.array([1,1])
# data_model_name = f"KFeaturesModel_TwoFeatures_ProtectingIdentity_AttackingIdentity_{feature_ratios}_{features_x}_{features_theta}"
# data_model_description = "2 Features, Theta Identity, Sigma_delta Identity"
# Sigma_w = np.eye(d)
# Sigma_delta = np.eye(d)
# Sigma_upsilon = np.eye(d)

######### Attacking the first half of the features ##########

# data_model_type = DataModelType.KFeaturesModel
# feature_ratios = np.array([0.5,0.5])
# features_x = np.array([10,1])
# features_theta = np.array([1,1])
# data_model_name = f"KFeaturesModel_TwoFeatures_ProtectingIdentity_AttackingFirstHalf_{feature_ratios}_{features_x}_{features_theta}"
# data_model_description = "2 Features, Theta Identity, Sigma_delta Identity, Sigma_upsilon 10*Identity for the first half of the features"
# Sigma_w = np.eye(d)
# Sigma_delta = np.eye(d)
# Sigma_upsilon = np.eye(d)
# Sigma_upsilon[0:int(d/2),0:int(d/2)] = 10*np.eye(int(d/2))

######### Attacking the second half of the features ##########

# data_model_type = DataModelType.KFeaturesModel
# feature_ratios = np.array([0.5,0.5])
# features_x = np.array([10,1])
# features_theta = np.array([1,1])
# data_model_name = f"KFeaturesModel_TwoFeatures_ProtectingIdentity_AttackingSecondHalf_{feature_ratios}_{features_x}_{features_theta}"
# data_model_description = "2 Features, Theta Identity, Sigma_delta Identity, Sigma_upsilon 10*Identity for the second half of the features"
# Sigma_w = np.eye(d)
# Sigma_delta = np.eye(d)
# Sigma_upsilon = np.eye(d)
# Sigma_upsilon[int(d/2):d,int(d/2):d] = 10*np.eye(int(d/2))

######### Attacking and protecting the second half of the features ##########

# data_model_type = DataModelType.KFeaturesModel
# feature_ratios = np.array([0.5,0.5])
# features_x = np.array([10,1])
# features_theta = np.array([1,1])
# data_model_name = f"KFeaturesModel_TwoFeatures_ProtectingSecondHalf_AttackingSecondHalf_{feature_ratios}_{features_x}_{features_theta}_SD_1_10_SU_1_10"
# data_model_description = "2 Features, Theta Identity, Sigma_delta Identity, Sigma_upsilon 10*Identity for the second half of the features, Sigma_delta 10*Identity for the second half of the features"
# Sigma_w = np.eye(d)
# Sigma_delta = np.eye(d)
# Sigma_delta[int(d/2):d,int(d/2):d] = 10*np.eye(int(d/2))
# Sigma_upsilon = np.eye(d)
# Sigma_upsilon[int(d/2):d,int(d/2):d] = 10*np.eye(int(d/2))


"""
------------------------------------------------------------------------------------------------------------------------
    Next, define your experiment.
------------------------------------------------------------------------------------------------------------------------
"""
experiment_filename = "sweep_experiment_11.json"


# # Create a SweepExperiment
def get_default_experiment():
    state_evolution_repetitions: int = 1
    erm_repetitions: int = 15
    alphas: np.ndarray = np.logspace(-0.9,2,18) #np.linspace(0.1,10,15) #
    epsilons: np.ndarray = np.array([0,0.2,0.4,0.6]) # np.linspace(0,0.6,2) # np.array([0.0,0.2]) # np.array([0,0.1,0.3,0.4,0.5]) 
    lambdas: np.ndarray = np.logspace(-4,2,15) # np.logspace(-1,2,1) #np.concatenate([-np.logspace(-4,-1,10),np.logspace(-6,-3,2)])  #np.array([-0.0001])
    taus: np.ndarray = np.array([0])
    ps: np.ndarray = None # np.array([0.6,0.75,0.9])
    dp: float = 0.01
    # round the lambdas, epsilons and alphas for 4 digits
    alphas = np.round(alphas,4)
    test_against_epsilons: np.ndarray = np.array([0.2,0.6])
    epsilons = np.round(epsilons,4)
    lambdas = np.round(lambdas,4)
    test_against_epsilons = np.round(test_against_epsilons,4)
    # experiment_type: ExperimentType = ExperimentType.OptimalLambdaAdversarialTestError
    experiment_type: ExperimentType = ExperimentType.SweepAtOptimalLambda
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