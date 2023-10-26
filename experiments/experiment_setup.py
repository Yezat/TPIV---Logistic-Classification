import os
import inspect
import sys
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
from data_model import DataModelType
from experiment_information import NumpyEncoder, ExperimentInformation, ExperimentType
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

"""
------------------------------------------------------------------------------------------------------------------------
    Start Here, and define your data_model
------------------------------------------------------------------------------------------------------------------------
"""
d = 1000

data_model_type = DataModelType.VanillaGaussian
data_model_name = "VanillaGaussian"
data_model_description = "A Data-Model with Identity Gaussians for all the covariances."

# Sigma_w = power_law_diagonal_matrix(d, 1.4)
Sigma_w = np.eye(d)
# Sigma_delta = power_law_diagonal_matrix(d, 1.2)
Sigma_delta = np.eye(d)

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
    alphas: np.ndarray = np.array([3.5]) #np.linspace(0.1,6,3)
    epsilons: np.ndarray = np.array([0.35]) # np.array([0,0.1,0.3,0.4,0.5]) # np.linspace(0,1,5)
    lambdas: np.ndarray = np.array([-0.00001,-0.000008]) #np.concatenate([-np.logspace(-4,-1,10),np.logspace(-6,-3,2)])  #np.array([-0.0001])
    taus: np.ndarray = np.array([0])
    ps: np.ndarray = np.array([0.75]) 
    dp: float = 0.01
    experiment_type: ExperimentType = ExperimentType.Sweep
    experiment_name: str = "Vanilla Strong Weak Trials"
    compute_hessian: bool = False
    experiment = ExperimentInformation(state_evolution_repetitions,erm_repetitions,alphas,epsilons,lambdas,taus,d,experiment_type,ps,dp, data_model_type,data_model_name, data_model_description, experiment_name,compute_hessian)
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
    experiment.get_data_model(logger,source_pickle_path="../",delete_existing=True, Sigma_w=Sigma_w, Sigma_delta=Sigma_delta, name=data_model_name, description=data_model_description)
except Exception as e:
    # if you overwrite an existing data_model, you will get an exception. Still overwrite an experiment definition to ensure that you can run the experiment with the existing data_model.
    logger.info(f"Exception {e} occured.")

# use json dump to save the experiment parameters
with open(experiment_filename,"w") as f:
    # use the NumpyEncoder to encode numpy arrays
    # Let's produce some json
    json_string = json.dumps(experiment,cls= NumpyEncoder)
    # now write it
    f.write(json_string)

logger.info(f"Succesfully saved experiment to {experiment_filename}. You can run it even though we did not recreate the data_model!")

# # Start the MPI
# import subprocess
# # Run this command
# # mpiexec -n 2 python sweep.py sweep_experiment.json
# subprocess.run(["mpiexec","-n","4","python","sweep.py","sweep_experiment.json"])