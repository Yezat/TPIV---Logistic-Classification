This is a repository containing code for the paper "A High Dimensional Model for Adversarial Training: Geometry and Trade-Offs"

To reproduce the figures from the paper, please use `experiment_setup.py` in the experiments folder.

All experiments have a commented definition for the data-model and problem types considered. The choice of the sweep parameters has to be customised.

Once the experiment has been defined, and a json file containing the data-model and sweep definition (usually called `sweep_experiment.json`) been created, the data-models have to be created using the `create_data_model.py` script.
Then, the experiment can be exectued using `sweep.py`.
Running these scripts requires a working MPI installation. The command sequence is
```
python experiment_setup.py
mpiexec -n 5 python create_data_model.py sweep_experiment.json
mpiexec -n 5 python sweep.py sweep_experiment.json
```

The `sweep.py` script stores all results in a sqlite database. We provide scripts to easily extract the data, as an example, consider the `playground.ipynb` notebook.

Alternatively, the report folder contains the scripts used to create the figures and pickles containing already preprocessed data. It is sufficient to run the imports, and only the code past the loading of the pickle.

The experiments on real data have been performed in the `pca_experiments.ipynb` notebook.
