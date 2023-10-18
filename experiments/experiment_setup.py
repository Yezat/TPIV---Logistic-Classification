import os
import inspect
import sys
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
from data_model import DataModelType
from experiment_information import NumpyEncoder, ExperimentInformation
import json
import numpy as np
# # Create a SweepExperiment
def get_default_experiment():
    state_evolution_repetitions: int = 1
    erm_repetitions: int = 2
    alphas: np.ndarray = np.linspace(0.1,8,3)
    epsilons: np.ndarray = np.array([0,0.1,0.5]) # np.linspace(0,1,5)
    lambdas: np.ndarray = np.array([100])
    taus: np.ndarray = np.array([0])
    ps: np.ndarray = np.array([0.75]) 
    dp: float = 0.01
    d: int = 1000
    p: int = 1000
    erm_methods: list = ["sklearn"] #"optimal_lambda"
    experiment_name: str = "Vanilla Strong Weak Trials"
    data_model_type: DataModelType = DataModelType.VanillaGaussian
    experiment = ExperimentInformation(state_evolution_repetitions,erm_repetitions,alphas,epsilons,lambdas,taus,d,erm_methods,ps,dp, data_model_type,p, experiment_name)
    return experiment
experiment = get_default_experiment()

import logging
logger = logging.getLogger()
# Make the logger log to console
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)

# Force a creation of a new data_model
experiment.get_data_model(logger,source_pickle_path="../",delete_existing=True)
# # use json dump to save the experiment parameters
with open("sweep_experiment.json","w") as f:
    # use the NumpyEncoder to encode numpy arrays
    # Let's produce some json
    json_string = json.dumps(experiment.__dict__,cls= NumpyEncoder)
    # now write it
    f.write(json_string)