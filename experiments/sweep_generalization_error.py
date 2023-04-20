import logging
import time
import numpy as np
from multiprocessing import Process
import multiprocessing
from tqdm.auto import tqdm
import os
import queue
import sys
import inspect
import json
import uuid
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
from _version import __version__
from experiment_information import *
from state_evolution import fixed_point_finder, INITIAL_CONDITION, MIN_ITER_FPE, MAX_ITER_FPE, TOL_FPE, BLEND_FPE, INT_LIMS
from gradient_descent import gd, lbfgs, sklearn_optimize
from erm import get_logistic_regressor

def run_erm(lock,logger, experiment_id, method, alpha, epsilon, lam, tau, d):
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
    if method == "gd":
        w_gd = gd(Xtrain,y,lam,epsilon,logger)
    elif method == "L-BFGS-B":
        w_gd = lbfgs(Xtrain,y,lam,epsilon,logger)
    elif method == "logistic":
        clf = get_logistic_regressor(Xtrain,y,lam)
        # reshape to 1d array
        w_gd = clf.coef_.reshape(-1)
    elif method == "sklearn":
        w_gd = sklearn_optimize(sample_weights(d),Xtrain,y,lam,epsilon)
    else:
        raise Exception(f"Method {method} not implemented")


    end = time.time()
    duration = end - start
    erm_information = ERMExperimentInformation(experiment_id,duration,Xtest,w_gd,tau,y,Xtrain,w,ytest,d,method,epsilon,lam)

    logger.info(f"Finished ERM with alpha={alpha}, epsilon={epsilon}, lambda={lam}, tau={tau}, d={d}, method={method} in {end-start} seconds")

    # save results to database
    with lock:
        with DatabaseHandler(logger) as db:
            db.insert_erm(erm_information)
            logger.info(f"Saved ERM with alpha={alpha}, epsilon={epsilon}, lambda={lam}, tau={tau}, d={d}, method={method} to database")

def run_state_evolution(lock,logger, experiment_id, alpha, epsilon, lam, tau, d):
    """
    Starts the state evolution and saves the results to the database
    """

    logger.info(f"Starting State Evolution with alpha={alpha}, epsilon={epsilon}, lambda={lam}, tau={tau}, d={d}")
    start = time.time()
    m,q,sigma = fixed_point_finder(logger,rho_w_star=1,alpha=alpha,epsilon=epsilon,tau=tau,lam=lam,abs_tol=TOL_FPE,min_iter=MIN_ITER_FPE,max_iter=MAX_ITER_FPE,blend_fpe=BLEND_FPE,int_lims=INT_LIMS,initial_condition=INITIAL_CONDITION)
    gen_error = generalization_error(1,m,q, tau)

    logger.info(f"Finished State Evolution with alpha={alpha}, epsilon={epsilon}, lambda={lam}, tau={tau}, d={d}, gen_error={gen_error}") # TODO: print also the training error
    end = time.time()
    experiment_duration = end-start
    logger.debug(f"State Evolution took {end-start} seconds")

    # save the results to the database
    with lock:
        start = time.time()
        with DatabaseHandler(logger) as db:
            st_exp_info = StateEvolutionExperimentInformation(experiment_id,experiment_duration,sigma,q,m,INITIAL_CONDITION,alpha,epsilon,tau,lam,TOL_FPE,MIN_ITER_FPE,MAX_ITER_FPE,BLEND_FPE,INT_LIMS)
            db.insert_state_evolution(st_exp_info)
            logger.info(f"Saved State Evolution to db with alpha={alpha}, epsilon={epsilon}, lambda={lam}, tau={tau}, d={d}, gen_error={gen_error}")
        end = time.time()
        logger.debug(f"Saving to db took {end-start} seconds")


started_procs = []
def start_work(procs, number_of_workers):
    c = number_of_workers
    number_of_processes = procs.qsize()
    completed_processes = 0
    with tqdm(total=number_of_processes) as pbar:
        while procs.qsize() > 0:
            if c > 0:
                proc = procs.get()

                proc.start()
                c -= 1
                started_procs.append(proc)
            for proc in started_procs:

                if not proc.is_alive():

                    c += 1
                    completed_processes += 1
                    logging.info("Completed %d of %d processes, %d %%", completed_processes, number_of_processes, completed_processes/number_of_processes*100)
                    started_procs.remove(proc)
                    pbar.update(1)
    logging.info("Done with starting work")

def get_default_experiment():
    state_evolution_repetitions: int = 1
    erm_repetitions: int = 8
    # alphas: np.ndarray = np.array([0.2,0.8,1.3,1.7,2.,2.5])
    # alphas: np.ndarray = np.array([3,4,5,6,7])
    alphas: np.ndarray = np.linspace(1,3,5)
    # epsilons: np.ndarray = np.array([0,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09])
    # epsilons: np.ndarray = np.array([0,0.02,0.05,0.07,0.09,0.12])
    epsilons: np.ndarray = np.array([0.2,0.4,0.6,0.7])
    lambdas: np.ndarray = np.array([1e-2,1])
    taus: np.ndarray = np.array([2])
    d: int = 1000
    erm_methods: list = ["sklearn"]
    experiment_name: str = "Default Experiment"
    experiment = ExperimentInformation(state_evolution_repetitions,erm_repetitions,alphas,epsilons,lambdas,taus,d,erm_methods,experiment_name)
    return experiment

if __name__ == "__main__":
    logger = multiprocessing.log_to_stderr()
    logger.setLevel(level=logging.INFO)

    # see if a filename has been past as an argument
    if len(sys.argv) > 1:
        filename = sys.argv[1]
    else:
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
    except FileNotFoundError:
        logging.error("Could not find file %s. Using the standard elements instead", filename)


    with DatabaseHandler(logger) as db:
        db.insert_experiment(experiment)

    experiment_id = experiment.experiment_id
    logging.info("Starting Experiment with id %s", experiment_id)


    # define the number of worker threads
    number_of_workers = 5

    # create a lock for db access
    lock = multiprocessing.Lock()

    # create a queue for the processes
    procs = queue.Queue()

    # note starttime
    start = time.time()

    # iterate all the parameters and create process objects for each parameter
    for alpha in experiment.alphas:
        for epsilon in experiment.epsilons:
            for lam in experiment.lambdas:
                for tau in experiment.taus:

                    for _ in range(experiment.state_evolution_repetitions):
                        proc = Process(target=run_state_evolution, args=(lock,logger,experiment_id,alpha,epsilon,lam,tau,experiment.d))
                        procs.put(proc)

                    for _ in range(experiment.erm_repetitions):
                        for method in experiment.erm_methods:
                            proc = Process(target=run_erm, args=(lock,logger,experiment_id,method,alpha,epsilon,lam,tau,experiment.d))
                            procs.put(proc)

    # start the processes
    logger.info("Starting all processes")
    start_work(procs, number_of_workers)

    # wait till all processes are done
    logger.info("Waiting for all processes to finish")
    for proc in started_procs:
        proc.join()

    # mark the experiment as finished
    logger.info(f"Marking experiment {experiment_id} as finished")
    end = time.time()
    duration = end - start
    logger.info("Experiment took %d seconds", duration)
    with DatabaseHandler(logger) as db:
        db.complete_experiment(experiment_id, duration)
    logger.info("Done")






