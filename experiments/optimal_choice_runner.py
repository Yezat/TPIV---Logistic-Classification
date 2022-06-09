import os
import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 
from experiments.optimal_choice import ex
ex.run(config_updates={'tau':0.5,'epsilon': 0.04,'d':1000,'number_of_runs':20,'min_alpha':0.3,'max_alpha':5,'number_of_repeated_measurements':10,'repetitions_in_minimize':3,'test_size_factor':2,'method':"gd",'tol':1e-3})