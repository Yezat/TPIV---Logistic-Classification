This is a repository containing code that was written during a semester project in the Information, Learning and Physics, IdePHICS Lab at EPFL.

It contains code for computing the calibration and optimal l2-regularization strengths for a logit classification model on probit iid gaussian data.
For this, I adapted code from sklearn to make use and adapt numerical tricks to the adversarial setting.

# Usage

Essentially, all we can do is run sweeps. The recommended way to do this is to define an experiment in playground.ipynb (bottom of the file).
Then you can run a sweep using mpi as described at the top of the sweep.py file in the experiments folder, or alternatively run it on a cluster using the run.sh file.

All plotting can be done within the playground.ipynb file.

## Optimal l2-regularization strength and optimal epsilon in terms of calibration
The file optimal_choice can be used to compute the optimal l2-regularization strength in terms of generalization error.
The file optimal_epsilon can be used to compute the optimal epsilon in terms of calibration.

"# TPIV---Logistic-Classification" 
