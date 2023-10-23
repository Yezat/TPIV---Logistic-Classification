# usage: mpiexec -n 5 python sweep.py sweep_experiment.json

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
from state_evolution import overlap_calibration, fixed_point_finder
from gradient_descent import sklearn_optimize, compute_experimental_teacher_calibration, run_optimizer
from data_model import *
from helpers import Task
import logging



def run_erm(logger, task, data_model):
    """
    Generate Data, run ERM and save the results to the database
    """
    logger.info(f"Starting ERM {task}")
    start = time.time()

    data = data_model.generate_data(int(task.alpha * task.d), task.tau)

    weights_erm = run_optimizer(task, data_model, data, logger)
    
    erm_information = ERMExperimentInformation(task, data_model, data, weights_erm)

    end = time.time()
    erm_information.duration = end - start
    
    logger.info(f"Finished ERM {task}")

    return erm_information

def run_state_evolution(logger,task, data_model):
    """
    Starts the state evolution and saves the results to the database
    """

    logger.info(f"Starting State Evolution {task}")
    start = time.time()

    overlaps = fixed_point_finder(logger, data_model, task)

    st_exp_info = StateEvolutionExperimentInformation(task,overlaps, data_model)


    end = time.time()
    experiment_duration = end-start
    st_exp_info.duration = experiment_duration

    
    logger.info(f"Finished State Evolution {task}")   
    overlaps.log_overlaps(logger)

    return st_exp_info


# Define a function to process a task
def process_task(task, logger, data_model):
    
    try:
        
        logger.info(f"Starting task {task.id}")

        if task.method == "state_evolution":
            task.result = run_state_evolution(logger,task, data_model)
        else:
            task.result = run_erm(logger,task, data_model)
    except Exception as e:
        # log the exception
        logger.exception(e)
        # set the result to the exception
        task.result = e

    return task

# Define the worker function
def worker(logger, data_model):
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
            result = process_task(task, logger, data_model)
            # Send the result to the master
            MPI.COMM_WORLD.send(result, dest=0, tag=task.id)
        except Exception as e:
            # log the exception
            logger.exception(e)
            MPI.COMM_WORLD.send(e,dest=0,tag=0)
            return
    logger.info(f"Worker exiting - my rank is {rank}")


def get_default_experiment():
    state_evolution_repetitions: int = 1
    erm_repetitions: int = 2
    # alphas: np.ndarray = np.array([0.2,0.8,1.3,1.7,2.,2.5])
    # alphas: np.ndarray = np.array([3,4,5,6,7])
    alphas: np.ndarray = np.linspace(1,3,3)
    # epsilons: np.ndarray = np.array([0,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09])
    # epsilons: np.ndarray = np.array([0,0.02,0.05,0.07,0.09,0.12])
    epsilons: np.ndarray = np.array([0.0,0.7])
    lambdas: np.ndarray = np.array([100])
    taus: np.ndarray = np.array([1.5])
    ps: np.ndarray = np.array([0.75])
    dp: float = 0.01
    d: int = 1000
    p: int = 1000
    erm_methods: list = ["sklearn"]
    experiment_name: str = "Default Experiment"
    dataModelType: DataModelType = DataModelType.VanillaGaussian
    experiment = ExperimentInformation(state_evolution_repetitions,erm_repetitions,alphas,epsilons,lambdas,taus,d,erm_methods,ps,dp,dataModelType,p,experiment_name)
    return experiment

def load_experiment(filename, logger):
    # Get the experiment information from this file.
    if filename is None:
        filename = "sweep_experiment.json"

    # create an experiment object and first define default values
    experiment = get_default_experiment()


    # load the experiment parameters from the json file
    try:
        with open(filename) as f:
            experiment.__dict__ = json.load(f, cls=NumpyDecoder)
            # create a new experiment id
            experiment.experiment_id = str(uuid.uuid4())
            # overwrite the code version
            experiment.code_version = __version__
            experiment.date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            logger.info("Loaded experiment from file %s", filename)
            # log the dataModelType
            logger.info(f"DataModelType: {experiment.data_model_type.name}")
    except FileNotFoundError:
        logger.error("Could not find file %s. Using the standard elements instead", filename)

    return experiment

# Define the master function
def master(num_processes, logger, experiment):

    with DatabaseHandler(logger) as db:
        db.insert_experiment(experiment)

    experiment_id = experiment.experiment_id
    logger.info("Starting Experiment with id %s", experiment_id)

    # note starttime
    start = time.time()

    # tasks = [{"id": i, "data": "Task " + str(i)} for i in range(1, 100)]
    tasks = []

    # iterate all the parameters and create process objects for each parameter
    idx = 1
    for alpha in experiment.alphas:
        for epsilon in experiment.epsilons:

            for tau in experiment.taus:
                
                # if method is "optimal_lambda":
                # compute first the optimal lambda
                if "optimal_lambda" in experiment.erm_methods:
                    import optimal_choice
                    logger.info(f"Computing optimal lambda for {alpha}, {epsilon}, {tau}")
                    lam = optimal_choice.get_optimal_lambda(alpha, epsilon, tau, logger)
                    # TODO make this sweep more efficient, i.e. parallelize it...

                    for _ in range(experiment.state_evolution_repetitions):
                        tasks.append(Task(idx,experiment_id,"state_evolution",alpha,epsilon,lam,tau,experiment.d,experiment.ps,None,experiment.data_model_type))

                    for _ in range(experiment.erm_repetitions):
                        method = "sklearn"
                        tasks.append(Task(idx,experiment_id,method,alpha,epsilon,lam,tau,experiment.d,experiment.ps,experiment.dp, experiment.data_model_type))
                else:

                # if method is not "optimal_lambda":
                    for lam in experiment.lambdas:
                    

                        for _ in range(experiment.state_evolution_repetitions):
                            tasks.append(Task(idx,experiment_id,"state_evolution",alpha,epsilon,lam,tau,experiment.d,experiment.ps,None, experiment.data_model_type))

                        for _ in range(experiment.erm_repetitions):
                            for method in experiment.erm_methods:
                                tasks.append(Task(idx,experiment_id,method,alpha,epsilon,lam,tau,experiment.d,experiment.ps,experiment.dp, experiment.data_model_type))

    # Initialize the progress bar
    pbar = tqdm(total=len(tasks))

    # start the processes
    logger.info("Starting all processes")
    # Send the tasks to the workers
    task_idx = 0
    received_tasks = 0
    for i in range(num_processes):
        if task_idx >= len(tasks):
            break
        task = tasks[task_idx]
        MPI.COMM_WORLD.send(task, dest=i+1, tag=task.id)
        task_idx += 1

    logger.info("All processes started - receiving results and sending new tasks")
    # Receive and store the results from the workers
    while received_tasks < len(tasks):
        status = MPI.Status()
        # log status information
        logger.info(f"Received task {received_tasks} from {status.source}") 
        
        task = MPI.COMM_WORLD.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
        received_tasks += 1

        # result
        result = task.result
        
        # test if the result is an exception
        if not isinstance(result, Exception):
            if task.method == "state_evolution":
                with DatabaseHandler(logger) as db:
                    db.insert_state_evolution(result)
            else:
                with DatabaseHandler(logger) as db:
                    db.insert_erm(result)

            logger.info(f"Saved {task}")
        else:
            logger.error(f"Error {task}")

        # Update the progress bar
        pbar.update(1)
        logger.info("")
        # Send the next task to the worker that just finished
        if task_idx < len(tasks):
            task = tasks[task_idx]
            MPI.COMM_WORLD.send(task, dest=status.source, tag=task.id)
            task_idx += 1


    logger.info("All tasks sent and received")

    # mark the experiment as finished
    logger.info(f"Marking experiment {experiment_id} as finished")
    end = time.time()
    duration = end - start
    logger.info("Experiment took %d seconds", duration)
    with DatabaseHandler(logger) as db:
        db.complete_experiment(experiment_id, duration)
    logger.info("Done")

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
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    logger.info("This process has rank %d", rank)

    experiment = load_experiment(filename, logger)

    if rank == 0:
        # run the master
        master(size-1, logger, experiment)
        
    else:
        # run the worker
        worker(logger, experiment.get_data_model(logger,delete_existing=False))

    MPI.Finalize()
    

    