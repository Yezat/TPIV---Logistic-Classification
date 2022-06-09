This is a repository containing code that was written during a semester project in the Information, Learning and Physics, IdePHICS Lab at EPFL.

It contains code for computing the calibration and optimal l2-regularization strengths for a logit classification model on probit iid gaussian data.
The optimization of the ERM problem can be done using gradient descent and scipy minimize L-BFGS-B

# Usage

## Calibration 

The file runner.py in the experiments folder shows how to run an experiment on various parameters.
The results of the experiments are stored in a json file and can be plotted using plot_information file.


## Choice of optimal l2-regularization strength

The optimal_choice_runner.py file can be used to optimize the l2-regularization strength for a given set of parameters.
The data is again output as a json file and plots can be generated using the plot_optimal_choices.py file.

### Calibration, test loss and test error at optimal l2-regularization strength

To obtain the calibration, test loss and test error at optimal l2-regularization strengths, it is recommended to use the script calibration_test_loss_at_optimal_lambdas.py
Again, the resulting json-file containing the data can be plotted using plot_calibration_test_loss_at_optimal_lambdas.py


"# TPIV---Logistic-Classification" 
