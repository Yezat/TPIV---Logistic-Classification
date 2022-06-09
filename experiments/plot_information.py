"""
Plots debug information
"""
from cProfile import label
import os
import sys
import inspect

from sympy import false
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 
from calibration import *

# plot imports
from matplotlib import collections
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as mpimg
import matplotlib.colors as colors
from matplotlib import rcParams
import json
from mpl_toolkits.mplot3d import Axes3D
rcParams.update({'figure.autolayout': True})

from gradient_descent import *


def load_data(filename):
    with open(f"../data/{filename}.json",'r') as f:
        s = f.read()
        return json.loads(s)

def start_plot():
    fig,ax = plt.subplots(1,1,figsize=(16,9))
    return ax

def plot_results(ax, results, stds, xaxis, comment = "",y_log_scale = False,x_log_scale=False):
    if y_log_scale:
        plt.yscale("log")
    if x_log_scale:
        plt.xscale("log")
    plt_teacher = ax.errorbar(xaxis,results,yerr=stds,marker="x", fmt='', label=f"{comment}")
    return plt_teacher

def finalize_plot(ax,handles,key,comment,filename, show = False,xlabel="",ylabel = "",title = "",y_log_scale=False,x_log_scale=False):
    ax.legend(handles=handles)
    
    if ylabel == "":
        ylabel = key

    if xlabel == "":
        xlabel = "$\\alpha$"

    if title=="":
        title = f"{key} {comment}"

    font_size = 25
    font = {'size':font_size}

    if y_log_scale:
        plt.yscale("symlog")
    if x_log_scale:
        plt.xscale("log")

    plt.title(f"{title}",fontdict=font)
    plt.xlabel(xlabel,fontdict=font,labelpad=font_size)
    plt.ylabel(ylabel,fontdict=font,labelpad=font_size)
    plt.setp(ax.get_xticklabels(),fontsize=font_size)
    plt.setp(ax.get_yticklabels(),fontsize=font_size)
    plt.legend(prop={'size': font_size})
    plt.savefig(fname=f"../assets/{filename}.pdf")
    

    if show:
        plt.show()
    else:
        plt.close()

def aggregate(data, alphas,taus,lams,epsilons,ps,methods,number_of_repeated_measurements, key):
    """
    Aggregate the data for each alpha, tau, lam, epsilon, p, method
    """
    print("aggregating over:",key,"alphas",alphas,"taus",taus,"lams",lams,"epsilons",epsilons,"ps",ps,"methods",methods)
    aggregate = []
    aggregate_std = []
    for alpha in alphas:
        for tau in taus:
            for lam in lams:
                for epsilon in epsilons:
                    for p in ps:
                        for method in methods:
                            values = []
                            for i in range(number_of_repeated_measurements):
                                inf = data[f"alpha_{alpha}"][f"tau_{tau}"][f"lam_{lam}"][f"epsilon_{epsilon}"][f"p_{p}"][method][f"run_{i}"]
                                values.append(inf[key])
                            aggregate.append(np.mean(values))
                            aggregate_std.append(np.std(values))
    return np.array(aggregate), np.array(aggregate_std)


def plot_aggregate_epsilon_for_key(data,taus,epsilon,lams,ps,methods,alphas,number_of_repeated_measurements,key,filename,d,show=False):
    mean, std = aggregate(data,alphas,[taus[0]],[lams[0]],[epsilon],[ps[0]],[methods[0]],number_of_repeated_measurements,key)
    ax = start_plot()
    handle1 = plot_results(ax,mean,std,alphas,comment=f"{key} $\epsilon={epsilon}$")
    handles = [handle1]
    finalize_plot(ax,handles,key="",comment=f"{key} - epsilon {epsilon} - tau {taus[0]} - lam {lams[0]} - p {ps[0]} - method {methods[0]} - d {d}",filename=f"{filename}_{epsilon}_epsilon_{key}",show=show)

def plot_aggregate_epsilon_vs_lambda_for_key(data,taus,epsilon,lams,ps,methods,alphas,key,filename,d,show=False):
    mean,std = aggregate(data,[alphas[0]],[taus[0]],lams,[epsilon],[ps[0]],[methods[0]],number_of_repeated_measurements,key)
    ax = start_plot()
    handle1 = plot_results(ax,mean,std,lams,comment=f"{key} $\epsilon={epsilon}$",x_log_scale=True)
    handles = [handle1]
    finalize_plot(ax,handles,key="",comment=f"{key} - epsilon {epsilon} - tau {taus[0]} - lam {lams[0]} - p {ps[0]} - method {methods[0]} - d {d}",filename=f"{filename}_{epsilon}_epsilon_{key}_method_{methods[0]}",show=show,xlabel="$\lambda$",ylabel=key)

def plot_aggregate_epsilons_vs_lambda_for_key(data,taus,epsilon,lams,ps,methods,alphas,key,filename,d,show=False,y_log_scale=False):
    
    ax = start_plot()
    handles = []
    for epsilon in epsilons:
        mean,std = aggregate(data,[alphas[0]],[taus[0]],lams,[epsilon],[ps[0]],[methods[0]],number_of_repeated_measurements,key)
        handle1 = plot_results(ax,mean,std,lams,comment=f"$\\varepsilon={epsilon}$",x_log_scale=True,y_log_scale=y_log_scale)
        handles.append(handle1)
    ylabel = key
    if key =="calibration":
        ylabel = "$\Delta p$"
    finalize_plot(ax,handles,key="",comment=f"{key} - $\\tau$ {taus[0]} - p {ps[0]} - {methods[0]} - d {d}",filename=f"{filename}_{key}_method_{methods[0]}",show=show,xlabel="$\lambda$",ylabel=ylabel)

def convergence_plot(filename,data,key,d,taus,epsilons,lams,ps,methods,alphas,number_of_repeated_measurements,ylabel="$\log{\Delta p}$",show_debug=False):
    """
    for each epsilon, plot log(f(x) - f(infty)) = c + -alpha*log(x)
    """    
    if ylabel == "":
        ylabel = key
    ax = start_plot()
    handles = []
    for epsilon in epsilons:
        mean, std = aggregate(data,alphas,[taus[0]],[lams[0]],[epsilon],[ps[0]],[methods[0]],number_of_repeated_measurements,key)
        mean = np.log(mean - mean[-1])
        h = plot_results(ax,mean[1:],std[1:],alphas[1:],comment=f"$\\varepsilon={epsilon}$")
        handles.append(h)
    finalize_plot(ax,handles,key="",comment=f"{key} vs alpha - $\\tau$ {taus[0]} - $\lambda$ {lams[0]} - p {ps[0]} - {methods[0]} - d {d}",filename=f"{filename}_{key}_vs_alpha_convergence",show=show_debug,ylabel=ylabel,y_log_scale=False,x_log_scale=True)


def plot_combined_key_for_multiple_epsilons(filename,data,key,d,taus,epsilons,lams,ps,methods,alphas,number_of_repeated_measurements,ylabel="$\Delta p$",show_debug=False,y_log_scale=False,x_log_scale=False):
    """
    for each epsilon, plot the combined results for the key
    """
    if ylabel == "":
        ylabel = key
    ax = start_plot()
    handles = []
    for epsilon in epsilons:
        mean, std = aggregate(data,alphas,[taus[0]],[lams[0]],[epsilon],[ps[0]],[methods[0]],number_of_repeated_measurements,key)
        h = plot_results(ax,mean,std,alphas,comment=f"$\\varepsilon={epsilon}$")
        handles.append(h)
    finalize_plot(ax,handles,key="",comment=f"{key} vs alpha - $\\tau$ {taus[0]} - $\lambda$ {lams[0]} - p {ps[0]} - {methods[0]} - d {d}",filename=f"{filename}_{key}_vs_alpha",show=show_debug,ylabel=ylabel,y_log_scale=y_log_scale,x_log_scale=x_log_scale)



def produce_vs_alpha_debug_plots(filename,data,d,taus,epsilon,lams,ps,methods,alphas,number_of_repeated_measurements,show_debug=False):
    plot_aggregate_epsilon_for_key(data,taus,epsilon,lams,ps,methods,alphas,number_of_repeated_measurements,"generalization_error",filename,d,show=show_debug)
    plot_aggregate_epsilon_for_key(data,taus,epsilon,lams,ps,methods,alphas,number_of_repeated_measurements,"training_error",filename,d,show=show_debug)
    plot_aggregate_epsilon_for_key(data,taus,epsilon,lams,ps,methods,alphas,number_of_repeated_measurements,"q",filename,d,show=show_debug)
    plot_aggregate_epsilon_for_key(data,taus,epsilon,lams,ps,methods,alphas,number_of_repeated_measurements,"m",filename,d,show=show_debug)
    plot_aggregate_epsilon_for_key(data,taus,epsilon,lams,ps,methods,alphas,number_of_repeated_measurements,"rho",filename,d,show=show_debug)
    plot_aggregate_epsilon_for_key(data,taus,epsilon,lams,ps,methods,alphas,number_of_repeated_measurements,"cosb",filename,d,show=show_debug)
    plot_aggregate_epsilon_for_key(data,taus,epsilon,lams,ps,methods,alphas,number_of_repeated_measurements,"norm_w",filename,d,show=show_debug)
    plot_aggregate_epsilon_for_key(data,taus,epsilon,lams,ps,methods,alphas,number_of_repeated_measurements,"norm_w_gd",filename,d,show=show_debug)
   
def produce_vs_lamba_debug_plots(filename,data,d,taus,epsilon,lams,ps,methods,alphas,show_debug=False):
    plot_aggregate_epsilon_vs_lambda_for_key(data,taus,epsilon,lams,ps,methods,alphas,"generalization_error",filename,d,show=show_debug)
    plot_aggregate_epsilon_vs_lambda_for_key(data,taus,epsilon,lams,ps,methods,alphas,"calibration",filename,d,show=show_debug)
    plot_aggregate_epsilon_vs_lambda_for_key(data,taus,epsilon,lams,ps,methods,alphas,"test_loss",filename,d,show=show_debug)
    

def plot_calibration_vs_alpha_for_epsilons(filename,data,d,taus,epsilons,lams,ps,methods,alphas,number_of_repeated_measurements, show_debug=False,show=False):
    """
    Assumes that taus, lams, ps and methods are ennumarable with one element.
    """
    # for each epsilon, produce all the debug plots...
    # for epsilon in epsilons:
    #     produce_vs_alpha_debug_plots(filename,data,d,taus,epsilon,lams,ps,methods,alphas,number_of_repeated_measurements,show_debug=show_debug)

    plot_combined_key_for_multiple_epsilons(filename,data,"calibration",d,taus,epsilons,lams,ps,methods,alphas,number_of_repeated_measurements,show_debug=False,y_log_scale=False,x_log_scale=False)
    convergence_plot(filename,data,"calibration",d,taus,epsilons,lams,ps,methods,alphas,number_of_repeated_measurements,show_debug=True)
    plot_combined_key_for_multiple_epsilons(filename,data,"training_error",d,taus,epsilons,lams,ps,methods,alphas,number_of_repeated_measurements,ylabel="",show_debug=show)
    plot_combined_key_for_multiple_epsilons(filename,data,"generalization_error",d,taus,epsilons,lams,ps,methods,alphas,number_of_repeated_measurements,ylabel="",show_debug=show)
    plot_combined_key_for_multiple_epsilons(filename,data,"q",d,taus,epsilons,lams,ps,methods,alphas,number_of_repeated_measurements,ylabel="",show_debug=show)
    plot_combined_key_for_multiple_epsilons(filename,data,"m",d,taus,epsilons,lams,ps,methods,alphas,number_of_repeated_measurements,ylabel="",show_debug=show)
    plot_combined_key_for_multiple_epsilons(filename,data,"rho",d,taus,epsilons,lams,ps,methods,alphas,number_of_repeated_measurements,ylabel="$\\rho$",show_debug=show)
    plot_combined_key_for_multiple_epsilons(filename,data,"cosb",d,taus,epsilons,lams,ps,methods,alphas,number_of_repeated_measurements,ylabel="$\cos{\\beta}$",show_debug=show)
    plot_combined_key_for_multiple_epsilons(filename,data,"norm_w",d,taus,epsilons,lams,ps,methods,alphas,number_of_repeated_measurements,ylabel="",show_debug=show)
    plot_combined_key_for_multiple_epsilons(filename,data,"norm_w_gd",d,taus,epsilons,lams,ps,methods,alphas,number_of_repeated_measurements,ylabel="",show_debug=show)
    print("Done plotting calibration vs alpha for epsilons")

def plot_calibration_vs_lambda_for_epsilons(filename,data,d,taus,epsilons,lams,ps,methods,alphas,show_debug=False):
    """
    Assumes that taus, alphas, ps and methods are ennumarable with one element.
    """
    # for epsilon in epsilons:
    #     produce_vs_lamba_debug_plots(filename,data,d,taus,epsilon,lams,ps,methods,alphas,show_debug=show_debug)
    plot_aggregate_epsilons_vs_lambda_for_key(data,taus,epsilons,lams,ps,methods,alphas,"generalization_error",filename,d,show=show_debug)
    plot_aggregate_epsilons_vs_lambda_for_key(data,taus,epsilons,lams,ps,methods,alphas,"calibration",filename,d,show=True)
    plot_aggregate_epsilons_vs_lambda_for_key(data,taus,epsilons,lams,ps,methods,alphas,"test_loss",filename,d,show=True,y_log_scale=True)
    print("Done plotting calibration vs lambda for epsilons")

def plot_calibration_vs_alpha(filename,show=False,show_debug=False):
    """
    Plot the calibration vs alpha 
    """
    data = load_data(filename)
    config = data["config"]
    print(config)
    d = config["d"]
    taus = config["taus"]
    epsilons = config["epsilons"]
    lams = config["lams"]
    ps = config["ps"]
    methods = config["methods"]
    alphas = np.array(data["alphas"])
    number_of_repeated_measurements = config["number_of_repeated_measurements"]
    plot_calibration_vs_alpha_for_epsilons(filename,data,d,[taus[0]],epsilons,[lams[0]],[ps[0]],[methods[0]],alphas,number_of_repeated_measurements,show_debug=show_debug,show=show)

if __name__ == "__main__":
    # Calibration vs alpha
    #filename="calibration_vs_parameters_lams_[1e-05]_taus_[2]_ps_[0.75]_epsilons_[0.0, 0.01, 0.02, 0.03, 0.04, 0.05]_d_300_ntest_20000_number_of_runs_40_max_a"
    
    # Calibration vs alpha up to alpha = 20
    # filename = "calib_vs_param_lams_2022-06-07_183441_1e-05_taus_0.5_ps_0.75_epsilons_0.01_d_300.0_ntest_20000.0_number_of_runs_40.0min_alpha0.3_max_alpha_20.0_number_of"

    # Calibration vs alpha up to alpha = 60
    # filename = "calib_vs_param_lams_2022-06-08_101124_1e-05_taus_0.5_ps_0.75_epsilons_0.01_d_300.0_ntest_20000.0_number_of_runs_40.0min_alpha0.3_max_alpha_60.0_number_of"


    #Calibration vs alpha up to alpha = 120
    # filename = "calib_vs_param_lams_2022-06-08_130037_1e-05_taus_0.5_ps_0.75_epsilons_0.01_d_300.0_ntest_20000.0_number_of_runs_50.0min_alpha0.3_max_alpha_120.0_number_of"

    #Calibration vs alpha up to alpha = 240
    filename = "calib_vs_param_lams_2022-06-09_092808_1e-05_taus_0.5_ps_0.75_epsilons_0.01_d_100.0_ntest_20000.0_number_of_runs_50.0min_alpha0.3_max_alpha_240.0_number_of"

    # Calibration vs lambda
    # filename="calib_vs_param_lams_2022-06-06_222437_7.134847074980522_taus_0.5_ps_0.6_epsilons_0.02_d_100.0_ntest_100000.0_number_of_runs_1.0min_alpha5_max_alpha_5.0_nu"

    #Calibration vs p
    #filename = "calib_vs_param_lams_2022-06-08_111444_1e-05_taus_0.5_ps_0.5_epsilons_0.01_d_300.0_ntest_20000.0_number_of_runs_5.0min_alpha0.3_max_alpha_6.0_number_of_rep"
    data = load_data(filename)
    config = data["config"]
    print(config)
    d = config["d"]
    taus = config["taus"]
    epsilons = config["epsilons"]
    lams = config["lams"]
    ps = config["ps"]
    methods = config["methods"]
    alphas = np.array(data["alphas"])
    number_of_repeated_measurements = config["number_of_repeated_measurements"]
    
    # for calibration and debug information vs lambda
    plot_calibration_vs_alpha_for_epsilons(filename,data,d,[taus[0]],[0.0,0.01,0.02],[lams[0]],[ps[0]],[methods[0]],alphas,number_of_repeated_measurements,show_debug=False)
    
    # for calibration, loss, generalization error vs lambda
    # plot_calibration_vs_lambda_for_epsilons(filename,data,d,[taus[0]],epsilons,lams,[ps[0]],[methods[0]],[alphas[0]],show_debug=False)
    # plot_calibration_vs_lambda_for_epsilons(filename,data,d,[taus[0]],epsilons,lams,[ps[0]],[methods[1]],[alphas[0]],show_debug=False)

    # TODO: for calibration vs p
    # plot_calibration_vs_p_for_epsilons(filename,data,d,[taus[0]],epsilons,[lams[0]],[ps[0]],[methods[0]],alphas,show_debug=False)
