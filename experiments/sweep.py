# usage: mpiexec -n 5 python sweep.py

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
from state_evolution import fixed_point_finder, INITIAL_CONDITION, MIN_ITER_FPE, MAX_ITER_FPE, TOL_FPE, BLEND_FPE, INT_LIMS
from gradient_descent import sklearn_optimize

class Task:
    def __init__(self, id, experiment_id, method, alpha, epsilon, lam, tau,d):
        self.id = id
        self.experiment_id = experiment_id
        self.method = method
        self.alpha = alpha
        self.epsilon = epsilon
        self.lam = lam
        self.tau = tau
        self.d = d
        self.result = None

def run_erm(logger, experiment_id, method, alpha, epsilon, lam, tau, d):
    """
    Generate Data, run ERM and save the results to the database
    """
    logger.info(f"Starting ERM with alpha={alpha}, epsilon={epsilon}, lambda={lam}, tau={tau}, d={d}, method={method}")
    start = time.time()

    # generate ground truth
    w = sample_weights(d)

    # generate data
    Xtrain, y = sample_training_data(w,d,int(alpha * d),tau)
    n_test = 100000
    Xtest,ytest = sample_training_data(w,d,n_test,tau)

    w_gd = np.empty(w.shape,dtype=w.dtype)

    if method == "sklearn":
        w_gd = sklearn_optimize(sample_weights(d),Xtrain,y,lam,epsilon)
    else:
        raise Exception(f"Method {method} not implemented")


    end = time.time()
    duration = end - start
    erm_information = ERMExperimentInformation(experiment_id,duration,Xtest,w_gd,tau,y,Xtrain,w,ytest,d,method,epsilon,lam)


    logger.info(f"Finished ERM with alpha={alpha}, epsilon={epsilon}, lambda={lam}, tau={tau}, d={d}, method={method} in {end-start} seconds")

    return erm_information

def run_state_evolution(logger,experiment_id, alpha, epsilon, lam, tau, d):
    """
    Starts the state evolution and saves the results to the database
    """
    logger.info(f"Starting State Evolution with alpha={alpha}, epsilon={epsilon}, lambda={lam}, tau={tau}, d={d}")
    start = time.time()
    m,q,sigma = fixed_point_finder(logger,rho_w_star=1,alpha=alpha,epsilon=epsilon,tau=tau,lam=lam,abs_tol=TOL_FPE,min_iter=MIN_ITER_FPE,max_iter=MAX_ITER_FPE,blend_fpe=BLEND_FPE,int_lims=INT_LIMS,initial_condition=INITIAL_CONDITION)

    end = time.time()
    experiment_duration = end-start

    st_exp_info = StateEvolutionExperimentInformation(experiment_id,experiment_duration,sigma,q,m,INITIAL_CONDITION,alpha,epsilon,tau,lam,TOL_FPE,MIN_ITER_FPE,MAX_ITER_FPE,BLEND_FPE,INT_LIMS)

    logger.info(f"Finished State Evolution with alpha={alpha}, epsilon={epsilon}, lambda={lam}, tau={tau}, d={d}")
    return st_exp_info


# Define a function to process a task
def process_task(task, logger):
    
    try:

        if task.method == "state_evolution":
            task.result = run_state_evolution(logger,task.experiment_id,task.alpha,task.epsilon,task.lam,task.tau,task.d)
        else:
            task.result = run_erm(logger,task.experiment_id,task.method,task.alpha,task.epsilon,task.lam,task.tau,task.d)
    except Exception as e:
        # log the exception
        logger.exception(e)
        # set the result to the exception
        task.result = e

    return task

# Define the worker function
def worker(logger):
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
            result = process_task(task, logger)
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
    alphas: np.ndarray = np.linspace(1,3,2)
    # epsilons: np.ndarray = np.array([0,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09])
    # epsilons: np.ndarray = np.array([0,0.02,0.05,0.07,0.09,0.12])
    epsilons: np.ndarray = np.array([0.0,0.7])
    lambdas: np.ndarray = np.array([1])
    taus: np.ndarray = np.array([0])
    d: int = 1000
    erm_methods: list = ["sklearn"]
    experiment_name: str = "Default Experiment"
    experiment = ExperimentInformation(state_evolution_repetitions,erm_repetitions,alphas,epsilons,lambdas,taus,d,erm_methods,experiment_name)
    return experiment



# Define the master function
def master(num_processes, logger):

    # Get the experiment information from this file.
    filename = "sweep_experiment.json"

    # create an experiment object and first define default values
    experiment = get_default_experiment()



    # load the experiment parameters from the json file
    try:
        with open(filename) as f:
            experiment.__dict__ = json.load(f)
            # create a new experiment id
            experiment.experiment_id = str(uuid.uuid4())
            # overwrite the code version
            experiment.code_version = __version__
            experiment.date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    except FileNotFoundError:
        logger.error("Could not find file %s. Using the standard elements instead", filename)

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
            for lam in experiment.lambdas:
                for tau in experiment.taus:

                    for _ in range(experiment.state_evolution_repetitions):
                        tasks.append(Task(idx,experiment_id,"state_evolution",alpha,epsilon,lam,tau,experiment.d))

                    for _ in range(experiment.erm_repetitions):
                        for method in experiment.erm_methods:
                            tasks.append(Task(idx,experiment_id,method,alpha,epsilon,lam,tau,experiment.d))

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

            logger.info(f"Saved {task.method} to db with alpha={task.alpha}, epsilon={task.epsilon}, lambda={task.lam}, tau={task.tau}, d={task.d}")
        else:
            logger.error(f"Error in {task.method} with alpha={task.alpha}, epsilon={task.epsilon}, lambda={task.lam}, tau={task.tau}, d={task.d}")

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

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if rank == 0:
        # run the master
        master(size-1, logger)
        
    else:
        # run the worker
        worker(logger)

    MPI.Finalize()
    

    