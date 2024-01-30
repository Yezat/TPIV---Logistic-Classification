# usage: mpiexec -n 5 python create_data_model.py sweep_experiment.json

from mpi4py import MPI
from tqdm import tqdm
import time
import os
import inspect
import sys
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
from _version import __version__
from experiment_information import *
from data_model import *
from helpers import Task
import logging


# Define a function to process a data_model_definition
def process_data_model_definition(data_model_definition, experiment, logger):
    
    try:
        # Create the data model, just instantiating them produces the pickles...

        if data_model_definition.data_model_type == DataModelType.VanillaGaussian:
            data_model = VanillaGaussianDataModel(experiment.d,logger,source_pickle_path="../",delete_existing=data_model_definition.delete_existing,normalize_matrices=data_model_definition.normalize_matrices, Sigma_w_content=data_model_definition.Sigma_w_content, Sigma_delta_content=data_model_definition.Sigma_delta_content, Sigma_upsilon_content=data_model_definition.Sigma_upsilon_content, name=data_model_definition.name, description=data_model_definition.description)
        elif data_model_definition.data_model_type == DataModelType.SourceCapacity:
            data_model = SourceCapacityDataModel(experiment.d, logger, source_pickle_path="../", delete_existing=data_model_definition.delete_existing,normalize_matrices=data_model_definition.normalize_matrices, Sigma_w_content=data_model_definition.Sigma_w_content, Sigma_delta_content=data_model_definition.Sigma_delta_content, Sigma_upsilon_content=data_model_definition.Sigma_upsilon_content, name=data_model_definition.name, description=data_model_definition.description)
        elif data_model_definition.data_model_type == DataModelType.MarginGaussian:
            data_model = MarginGaussianDataModel(experiment.d,logger, source_pickle_path="../", delete_existing=data_model_definition.delete_existing,normalize_matrices=data_model_definition.normalize_matrices, Sigma_w_content=data_model_definition.Sigma_w_content, Sigma_delta_content=data_model_definition.Sigma_delta_content, Sigma_upsilon_content=data_model_definition.Sigma_upsilon_content, name=data_model_definition.name, description=data_model_definition.description)
        elif data_model_definition.data_model_type == DataModelType.KFeaturesModel:
            data_model = KFeaturesModel(experiment.d,logger, source_pickle_path="../", delete_existing=data_model_definition.delete_existing,normalize_matrices=data_model_definition.normalize_matrices, attack_equal_defense=data_model_definition.attack_equal_defense, Sigma_w_content=data_model_definition.Sigma_w_content, Sigma_delta_content=data_model_definition.Sigma_delta_content, Sigma_upsilon_content=data_model_definition.Sigma_upsilon_content, name=data_model_definition.name, description=data_model_definition.description, feature_ratios =data_model_definition.feature_ratios, features_x =data_model_definition.features_x, features_theta =data_model_definition.features_theta, process_sigma_type=data_model_definition.sigma_delta_process_type)
        else:
            raise ValueError("Unknown data model type")

    except Exception as e:
        # log the exception
        logger.exception(e)
        # set the result to the exception
        data_model_definition.result = e

    return data_model_definition

# Define the worker function
def worker(logger, experiment):
    # get the rank
    rank = MPI.COMM_WORLD.Get_rank()   

    while True:
        try:
            # Receive a task from the master
            task = MPI.COMM_WORLD.recv(source=0, tag=MPI.ANY_TAG)
            if task is None:
                # Signal to exit if there are no more tasks
                logger.info(f"Received exit signal - my rank is {rank}")
                break
            # Process the task
            result = process_data_model_definition(task, experiment, logger)
            # Send the result to the master
            MPI.COMM_WORLD.send(result, dest=0)
        except Exception as e:
            # log the exception
            logger.exception(e)
            MPI.COMM_WORLD.send(e,dest=0,tag=0)
            return
    logger.info(f"Worker exiting - my rank is {rank}")



def load_experiment(filename, logger):
    # Get the experiment information from this file.
    if filename is None:
        filename = "sweep_experiment.json"

    # load the experiment parameters from the json file
    try:
        with open(filename) as f:
            experiment_dict = json.load(f, cls=NumpyDecoder)
            experiment = ExperimentInformation.fromdict(experiment_dict) # Fromdict calls cls and thus creates a new experiment_id
            logger.info("Loaded experiment from file %s", filename)
            # log the dataModelType, name and description
            # logger.info(f"DataModelTypes:")
            # for data_model_type in experiment.data_model_types:
            #     logger.info(f"\t{data_model_type.name}")
            # logger.info(f"DataModelNames:")
            # for name in experiment.data_model_names:
            #     logger.info(f"\t{name}")
            # logger.info(f"DataModelDescriptions:")
            # for description in experiment.data_model_descriptions:
            #     logger.info(f"\t{description}")

            return experiment
    except FileNotFoundError:
        logger.error("Could not find file %s. Using the standard elements instead", filename)

    

# Define the master function
def master(num_processes, logger, experiment):

    logger.info("Starting Data-Model Creating with id ")

    # note starttime
    start = time.time()

    
    data_model_definitions = experiment.load_data_model_definitions(base_path="../")
    


    # Initialize the progress bar
    pbar = tqdm(total=len(data_model_definitions))

    # start the processes
    logger.info("Starting all processes")
    # Send the tasks to the workers
    task_idx = 0
    received_tasks = 0
    for i in range(num_processes):
        if task_idx >= len(data_model_definitions):
            break
        task = data_model_definitions[task_idx]
        logger.info(f"Sending task {task_idx} to {i+1}")
        MPI.COMM_WORLD.send(task, dest=i+1)
        task_idx += 1

    logger.info("All processes started - receiving results and sending new tasks")
    # Receive and store the results from the workers
    while received_tasks < len(data_model_definitions):
        status = MPI.Status()
        # log status information
        logger.info(f"Received the {received_tasks}th task") 
        
        task = MPI.COMM_WORLD.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
        received_tasks += 1

        logger.info(f"Received task {task} from {status.source}")

        
        # test if the result is an exception
        if not isinstance(task, Exception):
            logger.info(f"Created {task.name}")
        else:
            logger.error(f"Error {task.name}")

        # Update the progress bar
        pbar.update(1)
        logger.info("")
        # Send the next task to the worker that just finished
        if task_idx < len(data_model_definitions):
            task = data_model_definitions[task_idx]
            MPI.COMM_WORLD.send(task, dest=status.source, tag=task_idx)
            task_idx += 1


    logger.info("All tasks sent and received")

    # mark the experiment as finished
    logger.info(f"Created all Data-Models for experiment")
    end = time.time()
    duration = end - start
    logger.info("Creation took %d seconds", duration)

    # Close the progress bar
    pbar.close()

    logger.info("All done - signaling exit")
    # signal all workers to stop
    for i in range(num_processes):
        MPI.COMM_WORLD.send(None, dest=i+1, tag=0) 


if __name__ == "__main__":
    # create the MPI communicator
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # read the filename from the command line
    filename = sys.argv[1]
    

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(f'%(asctime)s - %(levelname)s - rank {rank} - %(message)s')
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    logger.info("The size is %d", size)

    logger.info("This process has rank %d", rank)

    experiment = load_experiment(filename, logger)

    if rank == 0:
        # run the master
        master(size-1, logger, experiment)
        
    else:
        # run the worker
        worker(logger, experiment)

    MPI.Finalize()
    

    