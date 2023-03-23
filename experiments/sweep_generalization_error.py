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
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 
from experiment_information import *
from state_evolution import fixed_point_finder, MAX_ITER_FPE, MIN_ITER_FPE, TOL_FPE, BLEND_FPE, INT_LIMS, INITIAL_CONDITION
from gradient_descent import gd, lbfgs

def run_erm(lock,logger, experiment_id: str, alpha: float, epsilon: float,lam: float,tau: float,d: int, method: str):
    """
    Generate Data, run ERM and save the results to the database
    """
    logger.info(f"Starting ERM with alpha={alpha}, epsilon={epsilon}, lambda={lam}, tau={tau}, d={d}")
    start = time.time()

    # generate ground truth
    w = sample_weights(d)

    # generate data
    Xtrain, y = sample_training_data(w,d,int(alpha * d),tau)
    n_test = 20000
    Xtest,ytest = sample_training_data(w,d,n_test,tau)

    w_gd = np.empty(w.shape,dtype=w.dtype)
    if method == "gd":
        w_gd = gd(Xtrain,y,lam,epsilon)
    elif method == "L-BFGS-B":
        w_gd = lbfgs(Xtrain,y,lam,epsilon)
    else:
        raise Exception(f"Method {method} not implemented")
    
    erm_information = ERMExperimentInformation(experiment_id,Xtest,w_gd,tau,y,Xtrain,w,ytest,d,method,epsilon,lam)

    end = time.time()
    logger.info(f"Finished ERM with alpha={alpha}, epsilon={epsilon}, lambda={lam}, tau={tau}, d={d} in {end-start} seconds")

    # save results to database
    with lock:
        with DatabaseHandler(logger) as db:
            db.insert_erm(erm_information)
            logger.info(f"Saved ERM with alpha={alpha}, epsilon={epsilon}, lambda={lam}, tau={tau}, d={d} to database")
    

def dummy_fixed_point_finder(INITIAL_CONDITION,rho_w_star=1,alpha=1,epsilon=1,tau=1,lam=1,abs_tol=1,min_iter=1,max_iter=1,blend_fpe=1,int_lims=1):
    # return three random numbers between 0 and 1 as a tuple
    return np.random.rand(3)

def run_state_evolution(lock,logger, experiment_id: str, alpha: float,epsilon: float,lam: float,tau: float,d: int):
       
    logger.info(f"Starting State Evolution with alpha={alpha}, epsilon={epsilon}, lambda={lam}, tau={tau}, d={d}")
    start = time.time()
    m,q,sigma = fixed_point_finder(logger, INITIAL_CONDITION,rho_w_star=1,alpha=alpha,epsilon=epsilon,tau=tau,lam=lam,abs_tol=TOL_FPE,min_iter=MIN_ITER_FPE,max_iter=MAX_ITER_FPE,blend_fpe=BLEND_FPE,int_lims=INT_LIMS)
    gen_error = generalization_error(1,m,sigma+q)
    
    logger.info(f"Finished State Evolution with alpha={alpha}, epsilon={epsilon}, lambda={lam}, tau={tau}, d={d}, gen_error={gen_error}")
    end = time.time()
    logger.debug(f"State Evolution took {end-start} seconds")

    # save the results to the database
    with lock:
        start = time.time()
        with DatabaseHandler(logger) as db:
            st_exp_info = StateEvolutionExperimentInformation(experiment_id,sigma,q,m,INITIAL_CONDITION,alpha,epsilon,tau,lam,TOL_FPE,MIN_ITER_FPE,MAX_ITER_FPE,BLEND_FPE,INT_LIMS)
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



if __name__ == "__main__":
    logger = multiprocessing.log_to_stderr()
    logger.setLevel(level=logging.INFO)
    
    # get an experiment id
    experiment_id = create_experiment_id()

    logging.info("Starting Experiment with id %s", experiment_id)

    # define the experiment
    alphas = np.linspace(0.1,5,10)
    epsilons = np.array([0,0.02])
    lambdas = np.array([1e-3]) # TODO try negative regularization
    tau = 2
    d = 300
    state_evolution_repetitions = 3
    erm_repetitions = 5
    erm_method = "gd"
    experiment = ExperimentInformation(experiment_id,state_evolution_repetitions,erm_repetitions,alphas,epsilons,lambdas,tau,d)
    with DatabaseHandler(logger) as db:
        db.insert_experiment(experiment)

    # define the number of worker threads
    number_of_workers = 8

    # create a lock for db access
    lock = multiprocessing.Lock()

    # create a queue for the processes
    procs = queue.Queue()

    # iterate all the parameters and create process objects for each parameter
    for alpha in alphas:
        for epsilon in epsilons:
            for lam in lambdas:
                
                for _ in range(state_evolution_repetitions):
                    proc = Process(target=run_state_evolution, args=(lock,logger,experiment_id,alpha,epsilon,lam,tau,d))
                    procs.put(proc)
                    
                for _ in range(erm_repetitions):
                    proc = Process(target=run_erm, args=(lock,logger,experiment_id,alpha,epsilon,lam,tau,d))
                    procs.put(proc)

    # start the processes
    logger.info("Starting all processes")
    start_work(procs, number_of_workers)

    # wait till all processes are done
    logger.info("Waiting for all processes to finish")
    for proc in started_procs:
        proc.join()

    # mark the experiment as finished
    logger.info("Marking experiment as finished")
    with DatabaseHandler(logger) as db:
        db.set_experiment_completed(experiment_id)
    logger.info("Done")
    

                



