# A High Dimensional Model for Adversarial Training: Geometry and Trade-Offs

<div width=auto>
    <img src="Figures/feature_combinations_alpha_sweep.pdf" width=100%>
</div>

<object data="Figures/feature_combinations_alpha_sweep.pdf" type="application/pdf" width="700px" height="700px">
    <embed src="Figures/feature_combinations_alpha_sweep.pdf">
        <p>This browser does not support PDFs. Please download the PDF to view it: <a href="Figures/feature_combinations_alpha_sweep.pdf">Download PDF</a>.</p>
    </embed>
</object>

<div style="text-align: center; margin: auto">
    <p><i>
    We construct combinations of non-robust/non-useful features and observe the (adversarial) generalisation error together with the boundary error and the class preserving error as a function of the sample complexity. There is good agreement between the theory (lines) and finite size simulations (points).
    </i><p>
</div>

To reproduce the figures from the paper, please use `define_experiment.ipynb` in the experiments folder.

All experiments have a definition for the data-model and problem types considered. The choice of the sweep parameters has to be customised.

Once the experiment has been defined, and a json file containing the data-model and sweep definition (usually called `sweep_experiment.json`) been created, the data-models have to be created using the `create_data_model.py` script.
Then, the experiment can be exectued using `sweep.py`.
Running these scripts requires a working MPI installation. The command sequence is
```
mpiexec -n 5 python create_data_model.py sweep_experiment.json
mpiexec -n 5 python sweep.py sweep_experiment.json
```
Alternatively, in a cluster environment, it is possible to use the generated `run.sh` file.

The `sweep.py` script stores all results in a sqlite database. We provide scripts to easily extract the data in the `Evaluate` folder.

The experiments on real data have been performed in the `pca_experiments.ipynb` notebook.
