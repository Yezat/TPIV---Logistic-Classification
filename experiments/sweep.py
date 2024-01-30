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
from ERM import compute_experimental_teacher_calibration, run_optimizer
from data_model import *
from helpers import Task
from scipy.optimize import minimize_scalar
import logging



def run_erm(logger, task, data_model, df_sigma):
    """
    Generate Data, run ERM and save the results to the database
    """
    logger.info(f"Starting ERM {task}")
    start = time.time()

    data = data_model.generate_data(int(task.alpha * task.d), task.tau)

    weights_erm, problem_instance = run_optimizer(task, data_model, data, logger, df_sigma)
    
    erm_information = ERMExperimentInformation(task, data_model, data, weights_erm, problem_instance, logger)

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

    st_exp_info = StateEvolutionExperimentInformation(task,overlaps, data_model,logger)


    end = time.time()
    experiment_duration = end-start
    st_exp_info.duration = experiment_duration

    
    logger.info(f"Finished State Evolution {task}")   
    overlaps.log_overlaps(logger)

    return st_exp_info

def minimizer_lambda(logger, task, data_model, lam):
    """
    Run the state evolution and return the generalization error
    """
    task.lam = lam
    overlaps = fixed_point_finder(logger, data_model, task, log=False)

    if task.method == "optimal_lambda":
        gen_error =  generalization_error(data_model.rho,overlaps.m,overlaps.q,task.tau)
    elif task.method == "optimal_adversarial_lambda":
        test_against_epsilon = task.test_against_epsilons[0]
        if len(task.test_against_epsilons) > 1:
            logger.error(f"Optimizing over Adversarial Test Error can only be done with one test against epsilon, for this round we picked the first one which is {test_against_epsilon}")
        gen_error = adversarial_generalization_error_overlaps(overlaps,task,data_model,test_against_epsilon)

    logger.info(f"Generalization error for lambda {task.lam} is {gen_error}")
    return gen_error

def get_optimal_lambda(logger,task, data_model):
    """
    Computes the optimal lambda given a task and a data_model, returns the lambda
    """
    logger.info(f"Starting optimal lambda {task}")
    start = time.time()

    res = minimize_scalar(lambda l : minimizer_lambda(logger, task, data_model, l),method="bounded", bounds=[-0.000001,1e3],options={'xatol': 1e-8,'maxiter':200})
    # res = minimize_scalar(lambda l : minimizer_lambda(logger, task, data_model, l),method="brent", bracket=(-1e5,0,1e5),options={'xtol': 1e-8,'maxiter':100})
    logger.info(f"Minimized success: {res.success}; Message: {res.message}")
    if not res.success:
        raise Exception("Optimization of lambda failed: " + str(res.message))

    end = time.time()
    experiment_duration = end-start

    logger.info(f"Finished optimal lambda {task} in {experiment_duration} seconds - optimal lambda is {res.x}")
    if task.method == "optimal_lambda":
        result = OptimalLambdaResult(task.alpha, task.epsilon, task.tau, res.x,data_model.model_type, data_model.name,task.problem_type)
    elif task.method == "optimal_adversarial_lambda":
        result = OptimalAdversarialLambdaResult(task.alpha, task.epsilon, task.test_against_epsilons[0], task.tau, res.x,data_model.model_type, data_model.name,task.problem_type)
    return result


def minimize_epsilon(logger, task, data_model):

    state_evolution_info = run_state_evolution(logger, task, data_model)

    total_calibration = np.abs(np.array(state_evolution_info.calibrations.calibrations)).sum()

    logger.info(f"Absolute integrated calibration for epsilon {task.epsilon} is {total_calibration}")
    return total_calibration

def get_optimal_epsilon(logger, task, data_model):
    
    ps = np.linspace(0.01,0.99,1000)
    task.ps = ps

    def minimize(e):
        task.epsilon = e
        return minimize_epsilon(logger,task,data_model)

    res = minimize_scalar(lambda e : minimize(e),method="bounded", bounds=[0,10.0],options={'xatol': 1e-4,'maxiter':200})
    logger.info(f"Minimized success: {res.success}; Message: {res.message}")
    if not res.success:
        raise Exception("Optimization of epsilon failed " + str(res.message))
    result = OptimalEpsilonResult(task.alpha, res.x, task.tau, task.lam, data_model.model_type, data_model.name)
    return result


# Define a function to process a task
def process_task(task, logger, data_model, df_sigma):
    
    try:
        
        logger.info(f"Starting task {task.id}")

        if task.method == "state_evolution":
            task.result = run_state_evolution(logger,task, data_model)
        elif task.method == "optimal_lambda" or task.method == "optimal_adversarial_lambda":
            task.result = get_optimal_lambda(logger,task, data_model)
        elif task.method == "optimal_epsilon":
            task.result = get_optimal_epsilon(logger, task, data_model)
        else:
            task.result = run_erm(logger,task, data_model, df_sigma)
    except Exception as e:
        # log the exception
        logger.exception(e)
        # set the result to the exception
        task.result = e

    return task

# Define the worker function
def worker(logger, experiment, df_sigma):
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

            # get the data model
            data_model = experiment._load_data_model(logger, task.data_model_name, task.data_model_type, source_pickle_path = "../")

            # Process the task
            result = process_task(task, logger, data_model, df_sigma)
            # Send the result to the master
            MPI.COMM_WORLD.send(result, dest=0, tag=task.id)
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
            logger.info(f"DataModelTypes:")
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

    with DatabaseHandler(logger) as db:
        db.insert_experiment(experiment)

    experiment_id = experiment.experiment_id 
    logger.info("Starting Experiment with id %s", experiment_id)

    # note starttime
    start = time.time()

    # tasks = [{"id": i, "data": "Task " + str(i)} for i in range(1, 100)]
    tasks = []


    dummy_optimal_result = OptimalLambdaResult(0,0,0,0,None,None,None)
    # load the optimal lambdas from the csv file
    optimal_lambdas = load_csv_to_object_dictionary(dummy_optimal_result)
    
    dummy_optimal_result = OptimalAdversarialLambdaResult(0,0,0,0,0,None,None,None)
    optimal_adversarial_lambdas = load_csv_to_object_dictionary(dummy_optimal_result)

    # iterate all the parameters and create process objects for each parameter
    idx = 1
    for data_model_type, data_model_name in zip(experiment.data_model_types, experiment.data_model_names):
        for problem in experiment.problem_types:
            for alpha in experiment.alphas:  

                if ExperimentType.OptimalEpsilon == experiment.experiment_type:
                    # if that is the case, for each tau and lambda, compute the optimal epsilon
                    for tau in experiment.taus:
                        for lam in experiment.lambdas:
                            tasks.append(Task(idx,experiment_id,"optimal_epsilon",problem,alpha,0,0,lam,tau,experiment.d,None,None, data_model_type, data_model_name,experiment.gamma_fair_error))
                            idx += 1
                else:
                    for epsilon in experiment.epsilons:
                        for tau in experiment.taus:
                            
                            
                            if ExperimentType.OptimalLambda == experiment.experiment_type:
                                
                                
                                optimal_result = OptimalLambdaResult(alpha,epsilon,tau,0,data_model_type, data_model_name, problem)

                                initial_lambda = 1

                                if not optimal_result.get_key() in optimal_lambdas.keys():
                                    tasks.append(Task(idx,experiment_id,"optimal_lambda",problem,alpha,epsilon, experiment.test_against_epsilons,initial_lambda,tau,experiment.d,experiment.ps,experiment.dp, data_model_type, data_model_name,experiment.gamma_fair_error))
                                    idx += 1

                            elif ExperimentType.OptimalLambdaAdversarialTestError == experiment.experiment_type:
                                
                                optimal_result = OptimalAdversarialLambdaResult(alpha,epsilon,experiment.test_against_epsilons[0],tau,0,data_model_type,data_model_name, problem)

                                initial_lambda = 1

                                if not optimal_result.get_key() in optimal_adversarial_lambdas.keys():
                                    tasks.append(Task(idx,experiment_id,"optimal_adversarial_lambda",problem,alpha,epsilon, experiment.test_against_epsilons,initial_lambda,tau,experiment.d,experiment.ps,experiment.dp, data_model_type, data_model_name,experiment.gamma_fair_error))
                                    idx += 1

                                
                            else: 
                                lambdas = experiment.lambdas
                                if ExperimentType.SweepAtOptimalLambda == experiment.experiment_type:

                                    optimal_result = OptimalLambdaResult(alpha,epsilon,tau,0,data_model_type, data_model_name, problem)

                                    if not optimal_result.get_key() in optimal_lambdas.keys():
                                        logger.info(f"The key is '{optimal_result.get_key()}'")
                                        logger.error("Optimal lambda not found in csv file. Run first a sweep to compute the optimal lambda. ")
                                        # lambdas = [0.001]
                                        lambdas = None
                                    else:
                                        lambdas = [optimal_lambdas[optimal_result.get_key()]]

                                elif ExperimentType.SweepAtOptimalLambdaAdversarialTestError == experiment.experiment_type:
                                
                                    optimal_result = OptimalAdversarialLambdaResult(alpha,epsilon,experiment.test_against_epsilons[0],tau,0,data_model_type,data_model_name, problem)

                                    if not optimal_result.get_key() in optimal_adversarial_lambdas.keys():
                                        logger.info(f"The key is '{optimal_result.get_key()}'")
                                        logger.error("Optimal lambda not found in csv file. Run first a sweep to compute the optimal lambda.")
                                        lambdas = None
                                    else:
                                        lambdas = [optimal_adversarial_lambdas[optimal_result.get_key()]]


                                if lambdas is not None:
                                    for lam in lambdas:                    

                                        for _ in range(experiment.state_evolution_repetitions):
                                            tasks.append(Task(idx,experiment_id,"state_evolution",problem,alpha,epsilon,experiment.test_against_epsilons,lam,tau,experiment.d,experiment.ps,None, data_model_type, data_model_name,experiment.gamma_fair_error))
                                            idx += 1

                                        for _ in range(experiment.erm_repetitions):
                                            tasks.append(Task(idx,experiment_id,"sklearn",problem,alpha,epsilon,experiment.test_against_epsilons,lam,tau,experiment.d,experiment.ps,experiment.dp, data_model_type, data_model_name,experiment.gamma_fair_error))
                                            idx += 1

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
        logger.info(f"Sending task {task_idx} to {i+1}")
        MPI.COMM_WORLD.send(task, dest=i+1, tag=task.id)
        task_idx += 1

    logger.info("All processes started - receiving results and sending new tasks")
    # Receive and store the results from the workers
    while received_tasks < len(tasks):
        status = MPI.Status()
        # log status information
        logger.info(f"Received the {received_tasks}th task") 
        
        task = MPI.COMM_WORLD.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
        received_tasks += 1

        logger.info(f"Received task {task.id} from {status.source}")

        # result
        result = task.result
        
        # test if the result is an exception
        if not isinstance(result, Exception):
            if task.method == "state_evolution":
                with DatabaseHandler(logger) as db:
                    db.insert_state_evolution(result)
            elif task.method == "optimal_lambda" or task.method == "optimal_adversarial_lambda":
                append_object_to_csv(result)
            elif task.method == "optimal_epsilon":
                append_object_to_csv(result)
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
    if experiment.experiment_type == ExperimentType.Sweep or experiment.experiment_type == ExperimentType.SweepAtOptimalLambda or experiment.experiment_type == ExperimentType.SweepAtOptimalLambdaAdversarialTestError:
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
    formatter = logging.Formatter(f'%(asctime)s - %(levelname)s - rank {rank} - %(message)s')
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    logger.info("The size is %d", size)

    logger.info("This process has rank %d", rank)

    experiment = load_experiment(filename, logger)

    # load the sigma state evolution dataframe if it exists
    if os.path.exists("../sigma_state_evolution.pkl"):
        df_sigma = pd.read_pickle("../sigma_state_evolution.pkl")
    else:
        df_sigma = None

    if os.path.exists("./empirical_values.csv"):
        try:
            df_sigma = pd.read_csv("./empirical_values.csv")
        except:
            logger.error("Could not load empirical_values.csv")
            df_sigma = None
    else:
        df_sigma = None

    if rank == 0:
        # run the master
        master(size-1, logger, experiment)
        
    else:

        # run the worker
        worker(logger, experiment, df_sigma)

    MPI.Finalize()
    

    