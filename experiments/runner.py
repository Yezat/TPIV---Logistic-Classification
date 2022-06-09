import os
import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

from experiments.calibration_vs_parameters import ex
import numpy as np
lams = np.linspace(1e-5,1,40) 
# lams = np.logspace(-5,2,50)
lams = [1e-5]
ps = np.linspace(0,1,20)
ps = [0.75] 
epsilons = [0.0,0.01,0.02]
ex.run(config_updates={'lams':lams,'taus': [0.5],'ps':ps,'epsilons': epsilons, 'd':50,'n_test':20000,'min_alpha':0.3,'max_alpha':580,'number_of_repeated_measurements':10,'number_of_runs':50,'methods':["gd"],'debug':False,'filename':"calibration_vs_alpha_580_convergence"})
# methods = ["gd","L-BFGS-B"]    