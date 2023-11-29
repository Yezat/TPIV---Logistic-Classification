import os
import inspect
import sys
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
from data_model import DataModelType
from experiment_information import NumpyEncoder, ExperimentInformation, ExperimentType, ProblemType, DataModelDefinition
import json
import numpy as np
from helpers import *

import logging
logger = logging.getLogger()
# Make the logger log to console
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)



feature_ratios = None
features_x = None
features_theta = None


delete_existing = False

"""
------------------------------------------------------------------------------------------------------------------------
    Start Here, and define your data_model_definitions
------------------------------------------------------------------------------------------------------------------------
"""
d = 1000

data_model_type = DataModelType.VanillaGaussian
data_model_name = "VanillaGaussian"
data_model_description = "A Data-Model with Identity Gaussians for all the covariances."
Sigma_w = np.eye(d)
Sigma_delta = np.eye(d)
Sigma_upsilon = np.eye(d)

vanilla_gaussian_data_model_definition = DataModelDefinition(delete_existing, Sigma_w, Sigma_delta, Sigma_upsilon, data_model_name, data_model_description, data_model_type)


########## Increasing the variance for all features ##########

## times 1000
# data_model_type = DataModelType.KFeaturesModel
# data_model_name = "VanillaGaussianTimes1000"
# data_model_description = "A Data-Model with Identity Gaussians times 1000 for all the covariances."
# feature_ratios = np.array([1.0])
# features_x = np.array([1000])
# features_theta = np.array([1,1])
# Sigma_w = np.eye(d)
# Sigma_delta = np.eye(d)
# Sigma_upsilon = np.eye(d)
# vanilla_gaussian_times_1000_data_model_definition

# ## times 100
# data_model_type = DataModelType.KFeaturesModel
# data_model_name = "VanillaGaussianTimes100"
# data_model_description = "A Data-Model with Identity Gaussians times 100 for all the covariances."
# feature_ratios = np.array([1.0])
# features_x = np.array([100])
# features_theta = np.array([1,1])
# Sigma_w = np.eye(d)
# Sigma_delta = np.eye(d)
# Sigma_upsilon = np.eye(d)

## times 10
data_model_type = DataModelType.KFeaturesModel
data_model_name = "VanillaGaussianTimes10"
data_model_description = "A Data-Model with Identity Gaussians times 10 for all the covariances."
feature_ratios = np.array([1.0])
features_x = np.array([10])
features_theta = np.array([1,1])
Sigma_w = np.eye(d)
Sigma_delta = np.eye(d)
Sigma_upsilon = np.eye(d)
vanilla_gaussian_times_10_data_model_definition = DataModelDefinition(delete_existing, Sigma_w, Sigma_delta, Sigma_upsilon, data_model_name, data_model_description, data_model_type, feature_ratios, features_x, features_theta)


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


data_model_type = DataModelType.KFeaturesModel
feature_ratios = np.array([0.5,0.5])
features_x = np.array([2,1])
features_theta = np.array([1,1])
data_model_name = f"KFeaturesModel_TwoFeatures_ProtectingIdentity_AttackingIdentity_{feature_ratios}_{features_x}_{features_theta}"
data_model_description = "2 Features, Theta Identity, Sigma_delta Identity"
Sigma_w = np.eye(d)
Sigma_delta = np.eye(d)
Sigma_upsilon = np.eye(d)
k_features_attacking_identity_protecting_identity_definition = DataModelDefinition(delete_existing, Sigma_w, Sigma_delta, Sigma_upsilon, data_model_name, data_model_description, data_model_type, feature_ratios, features_x, features_theta)

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


"""
Protecting second half
"""

######### Attacking and protecting the second half of the features ##########

# data_model_type = DataModelType.KFeaturesModel
# feature_ratios = np.array([0.5,0.5])
# features_x = np.array([10,1])
# features_theta = np.array([1,1])
# data_model_name = f"KFeaturesModel_TwoFeatures_ProtectingSecondHalf_AttackingSecondHalf_{feature_ratios}_{features_x}_{features_theta}_SD_1_10_SU_1_10"
# data_model_description = "2 Features, Theta Identity, Sigma_upsilon 10*Identity for the second half of the features, Sigma_delta 10*Identity for the second half of the features"
# Sigma_w = np.eye(d)
# Sigma_delta = np.eye(d)
# Sigma_delta[int(d/2):d,int(d/2):d] = 10*np.eye(int(d/2))
# Sigma_upsilon = np.eye(d)
# Sigma_upsilon[int(d/2):d,int(d/2):d] = 10*np.eye(int(d/2))

######### Protecting the second half of the features and attacking Identity ##########

# data_model_type = DataModelType.KFeaturesModel
# feature_ratios = np.array([0.5,0.5])
# features_x = np.array([10,1])
# features_theta = np.array([1,1])
# data_model_name = f"KFeaturesModel_TwoFeatures_ProtectingSecondHalf_AttackingIdentity_{feature_ratios}_{features_x}_{features_theta}_SD_1_10_SU_1_10"
# data_model_description = "2 Features, Theta Identity, Sigma_upsilon Identity, Sigma_delta 10*Identity for the second half of the features"
# Sigma_w = np.eye(d)
# Sigma_delta = np.eye(d)
# Sigma_delta[int(d/2):d,int(d/2):d] = 10*np.eye(int(d/2))
# Sigma_upsilon = np.eye(d)


# ######### Protecting the second half of the features and attacking the first half ##########

# data_model_type = DataModelType.KFeaturesModel
# feature_ratios = np.array([0.5,0.5])
# features_x = np.array([10,1])
# features_theta = np.array([1,1])
# data_model_name = f"KFeaturesModel_TwoFeatures_ProtectingSecondHalf_AttackingFirstHalf_{feature_ratios}_{features_x}_{features_theta}_SD_1_10_SU_1_10"
# data_model_description = "2 Features, Theta Identity, Sigma_upsilon 10*Identity for the first half of the features, Sigma_delta 10*Identity for the first half of the features"
# Sigma_w = np.eye(d)
# Sigma_delta = np.eye(d)
# Sigma_delta[int(d/2):d,int(d/2):d] = 10*np.eye(int(d/2))
# Sigma_upsilon = np.eye(d)
# Sigma_upsilon[0:int(d/2),0:int(d/2)] = 10*np.eye(int(d/2))


"""
Protecting first half
"""

######### Protecting the first half of the features and attacking the second half ##########

# data_model_type = DataModelType.KFeaturesModel
# feature_ratios = np.array([0.5,0.5])
# features_x = np.array([10,1])
# features_theta = np.array([1,1])
# data_model_name = f"KFeaturesModel_TwoFeatures_ProtectingFirstHalf_AttackingSecondHalf_{feature_ratios}_{features_x}_{features_theta}_SD_1_10_SU_1_10"
# data_model_description = "2 Features, Theta Identity, Sigma_upsilon 10*Identity for the second half of the features, Sigma_delta 10*Identity for the first half of the features"
# Sigma_w = np.eye(d)
# Sigma_delta = np.eye(d)
# Sigma_delta[0:int(d/2),0:int(d/2)] = 10*np.eye(int(d/2))
# Sigma_upsilon = np.eye(d)
# Sigma_upsilon[int(d/2):d,int(d/2):d] = 10*np.eye(int(d/2))

######### Protecting the first half of the features and attacking Identity ##########

# data_model_type = DataModelType.KFeaturesModel
# feature_ratios = np.array([0.5,0.5])
# features_x = np.array([10,1])
# features_theta = np.array([1,1])
# data_model_name = f"KFeaturesModel_TwoFeatures_ProtectingFirstHalf_AttackingIdentity_{feature_ratios}_{features_x}_{features_theta}_SD_1_10_SU_1_10"
# data_model_description = "2 Features, Theta Identity, Sigma_upsilon Identity, Sigma_delta 10*Identity for the first half of the features"
# Sigma_w = np.eye(d)
# Sigma_delta = np.eye(d)
# Sigma_delta[0:int(d/2),0:int(d/2)] = 10*np.eye(int(d/2))
# Sigma_upsilon = np.eye(d)


# ######### Protecting the first half of the features and attacking the first half ##########

# data_model_type = DataModelType.KFeaturesModel
# feature_ratios = np.array([0.5,0.5])
# features_x = np.array([10,1])
# features_theta = np.array([1,1])
# data_model_name = f"KFeaturesModel_TwoFeatures_ProtectingFirstHalf_AttackingFirstHalf_{feature_ratios}_{features_x}_{features_theta}_SD_1_10_SU_1_10"
# data_model_description = "2 Features, Theta Identity, Sigma_upsilon 10*Identity for the first half of the features, Sigma_delta 10*Identity for the first half of the features"
# Sigma_w = np.eye(d)
# Sigma_delta = np.eye(d)
# Sigma_delta[0:int(d/2),0:int(d/2)] = 10*np.eye(int(d/2))
# Sigma_upsilon = np.eye(d)
# Sigma_upsilon[0:int(d/2),0:int(d/2)] = 10*np.eye(int(d/2))


"""
------------------------------------------------------------------------------------------------------------------------
    Organize your data_model_definitions in a list.
------------------------------------------------------------------------------------------------------------------------
"""

data_model_definitions = [vanilla_gaussian_data_model_definition, k_features_attacking_identity_protecting_identity_definition]



data_models_description = "Vanilla vs KFeatures Identity vs Identity"

"""
------------------------------------------------------------------------------------------------------------------------
    Next, define your experiment.
------------------------------------------------------------------------------------------------------------------------
"""
# experiment_filename = "HighAlphaSweep/sweep_experiment_9.json"
# experiment_filename = f"AlphaSweepsLam10eM3/{data_model_name}/sweep_experiment.json"
experiment_filename = "sweep_experiment.json"


# extract data_model_names, data_model_descriptions and data_model_types from the data_model_definitions
data_model_names = [data_model_definition.name for data_model_definition in data_model_definitions]
data_model_descriptions = [data_model_definition.description for data_model_definition in data_model_definitions]
data_model_types = [data_model_definition.data_model_type for data_model_definition in data_model_definitions]

# # Create a SweepExperiment

# repetitions
state_evolution_repetitions: int = 1
erm_repetitions: int = 2

# sweeps
alphas: np.ndarray = np.logspace(-0.8,0,2) #np.linspace(0.1,10,15) #
epsilons: np.ndarray = np.array([0,0.4]) # np.linspace(0,0.6,2) # np.array([0.0,0.2]) # np.array([0,0.1,0.3,0.4,0.5]) 
lambdas: np.ndarray = np.array([1e-3]) #np.logspace(-4,2,15) # np.logspace(-1,2,1) #np.concatenate([-np.logspace(-4,-1,10),np.logspace(-6,-3,2)])  #np.array([-0.0001])
test_against_epsilons: np.ndarray = np.array([0.0,0.4])
taus: np.ndarray = np.array([0,1])

# round the lambdas, epsilons and alphas for 4 digits
alphas = np.round(alphas,4)    
epsilons = np.round(epsilons,4)
lambdas = np.round(lambdas,4)
taus = np.round(taus,4)
test_against_epsilons = np.round(test_against_epsilons,4)



# calibration
ps: np.ndarray = None # np.array([0.6,0.75,0.9])
dp: float = 0.01

# Type, name and problem
# experiment_type: ExperimentType = ExperimentType.OptimalLambdaAdversarialTestError
experiment_type: ExperimentType = ExperimentType.Sweep
# experiment_type: ExperimentType = ExperimentType.SweepAtOptimalLambdaAdversarialTestError
experiment_name: str = f"Sweep Alpha - {data_models_description}"
problem_types: list[ProblemType] = [ProblemType.Logistic]

# Fair error
gamma_fair_error: float = 0.01



experiment = ExperimentInformation(state_evolution_repetitions,erm_repetitions,alphas,epsilons,lambdas,taus,d,experiment_type,ps,dp, data_model_types,data_model_names, data_model_descriptions, test_against_epsilons,problem_types,gamma_fair_error, experiment_name)



# Log the experiment object
logger.info(f"Experiment: {experiment}")

"""
------------------------------------------------------------------------------------------------------------------------
    Now let the experiment create the data_model and save the experiment to a json file. 
------------------------------------------------------------------------------------------------------------------------
"""

experiment.store_data_model_definitions(data_model_definitions)


# make sure that the experiment directory exists
if os.path.dirname(experiment_filename) != "":
    os.makedirs(os.path.dirname(experiment_filename), exist_ok=True)

# use json dump to save the experiment parameters
with open(experiment_filename,"w") as f:
    # use the NumpyEncoder to encode numpy arrays
    # Let's produce some json
    json_string = json.dumps(experiment,cls= NumpyEncoder)
    # now write it
    f.write(json_string)

logger.info(f"Succesfully saved experiment to {experiment_filename}. You can run it even if we did not recreate the data_model!")

if os.path.dirname(experiment_filename) != "":
        
    # Now let's create a batch file to run the experiment
    content = f"""#!/bin/bash
    #SBATCH --chdir=/home/ktanner/TPIV---Logistic-Classification/experiments
    #SBATCH --job-name=TPIV-Adversarial
    #SBATCH --nodes=1
    #SBATCH --ntasks=20
    #SBATCH --cpus-per-task=1
    #SBATCH --mem=100G
    #SBATCH --output='{os.path.dirname(experiment_filename)}/out.txt'
    #SBATCH --error='{os.path.dirname(experiment_filename)}/error.txt'
    #SBATCH --time=24:00:00

    module purge
    module load gcc openmpi python/3.10.4
    source /home/ktanner/venvs/adv/bin/activate

    srun --mpi pmi2 python3 ./create_data_model.py '{experiment_filename}'
    srun --mpi=pmi2 python3 ./sweep.py '{experiment_filename}'

    deactivate
    """

    # save the batch file in the same directory as the experiment
    with open(f"{os.path.dirname(experiment_filename)}/run.sh","w") as f:
        f.write(content)

# # Start the MPI
# import subprocess
# # Run this command
# # mpiexec -n 2 python sweep.py sweep_experiment.json
# subprocess.run(["mpiexec","-n","4","python","sweep.py","sweep_experiment.json"])