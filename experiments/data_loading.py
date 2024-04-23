
import os
import sys
sys.path.insert(0,"../")

import json
import numpy as np
import pandas as pd
from experiment_information import *
from _version import __version__


def obtain_dataframes(logger, version_choice = None):

    df_experiments = None
    df_state_evolution = None
    df_erm = None

    
    with DatabaseHandler(logger,"../experiments/experiments.db") as dbHandler:

        df_experiments = dbHandler.get_experiments()
        df_state_evolution = dbHandler.get_state_evolutions()
        df_state_evolution["calibrations"] = df_state_evolution["calibrations"].apply(lambda x: json.loads(x))
        df_erm = dbHandler.get_erms()
        df_erm["analytical_calibrations"] = df_erm["analytical_calibrations"].apply(lambda x: json.loads(x))
        df_erm["erm_calibrations"] = df_erm["erm_calibrations"].apply(lambda x: json.loads(x))


    def explode_measures(df, new_columns, columns):
        for column in columns:
            def transform(column):
                # replace NaN in the column string by 0
                column = column.replace("NaN","0")
                # replace null in the column string by 0
                column = column.replace("null","0")
                # replace Infinity in the column string by np.inf
                column = column.replace("Infinity","np.inf")
                return eval(column)
            df[column] = df[column].apply(transform)

        exploded = df.explode(columns).reset_index(drop=True)

        for new_column, column in zip(new_columns, columns):
            if len(exploded[column].tolist()) > 0:
                exploded[["attack_epsilon",new_column]] = pd.DataFrame(exploded[column].tolist(), index=exploded.index)
            else:
                exploded[new_column] = np.nan
                # set attack_epsilon
                exploded["attack_epsilon"] = np.nan

        exploded = exploded.drop(columns=columns)
        return exploded


    def explode_erm_measures(df):
        columns = ["adversarial_generalization_errors","adversarial_generalization_errors_teacher","adversarial_generalization_errors_overlap","fair_adversarial_errors","test_losses","boundary_loss_test_es"]
        new_columns = ["adversarial_generalization_error","adversarial_generalization_error_teacher","adversarial_generalization_error_overlap","fair_adversarial_error","test_loss","boundary_loss_test"]
        return explode_measures(df, new_columns, columns)

    def explode_state_evolution_measures(df):
        columns = ["adversarial_generalization_errors","adversarial_generalization_errors_teacher","fair_adversarial_errors","first_term_fair_errors","second_term_fair_errors","third_term_fair_errors","test_losses","data_model_adversarial_generalization_errors","gamma_robustness_es","boundary_loss_test_es"]
        new_columns = ["adversarial_generalization_error","adversarial_generalization_error_teacher","fair_adversarial_error","first_term_fair_error","second_term_fair_error","third_term_fair_error","test_loss","data_model_adversarial_generalization_error","gamma_robustness","boundary_loss_test"] #
        return explode_measures(df, new_columns, columns)
        

    df_erm = explode_erm_measures(df_erm)

    df_state_evolution = explode_state_evolution_measures(df_state_evolution)
        

    # use the logger instead
    logger.info(f"Current code version, {__version__}")

    def extract_first_eigenvalue(row, column):
        array = row[column]

        # load json string to array
        array = json.loads(array)
        return np.array([float(array[0])])
    def extract_second_eigenvalue(row, column):
        array = row[column]

        # load json string to array
        array = json.loads(array)
        return np.array([float(array[-1])])
    def extract_trace(row, column):
        array = row[column]

        # load json string to array
        array = json.loads(array)
        return np.array([float(sum(array))])

    df2 = df_state_evolution.reset_index()


    df2["sigmax_first_ev"] = df2.apply(lambda row: extract_first_eigenvalue(row, "sigmax_eigenvalues"), axis=1)
    df2["sigmax_second_ev"] = df2.apply(lambda row: extract_second_eigenvalue(row, "sigmax_eigenvalues"), axis=1)
    df2["sigmax_trace"] = df2.apply(lambda row: extract_trace(row, "sigmax_eigenvalues"), axis=1)

    df2["sigmatheta_first_ev"] = df2.apply(lambda row: extract_first_eigenvalue(row, "sigmatheta_eigenvalues"), axis=1)
    df2["sigmatheta_second_ev"] = df2.apply(lambda row: extract_second_eigenvalue(row, "sigmatheta_eigenvalues"), axis=1)
    df2["sigmatheta_trace"] = df2.apply(lambda row: extract_trace(row, "sigmatheta_eigenvalues"), axis=1)

    df2["xtheta_first_ev"] = df2.apply(lambda row: extract_first_eigenvalue(row, "xtheta_eigenvalues"), axis=1)
    df2["xtheta_second_ev"] = df2.apply(lambda row: extract_second_eigenvalue(row, "xtheta_eigenvalues"), axis=1)
    df2["xtheta_trace"] = df2.apply(lambda row: extract_trace(row, "xtheta_eigenvalues"), axis=1)

    # drop the original eigenvalues columns
    df2 = df2.drop(columns=["sigmax_eigenvalues","sigmatheta_eigenvalues","xtheta_eigenvalues"])

    df_state_evolution = df2

    if version_choice is None:
        version_choice = __version__

    df_experiments = df_experiments[(df_experiments["completed"]==True) & (df_experiments["code_version"]==version_choice)]
    df_experiments = df_experiments.sort_values(by="date",ascending=False)

    df_state_evolution["estimation_error"] = 1 + df_state_evolution["q"] - 2 * df_state_evolution["m"]

    return df_experiments, df_state_evolution, df_erm