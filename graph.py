import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from os import listdir
from os.path import isfile, join



data_path = "./results_euler/"


backbone_files = [f for f in listdir(data_path) if (isfile(join(data_path, f)) and "backbone" in f)]
homophilic_files = [f for f in listdir(data_path) if (isfile(join(data_path, f)) and "homophilic" in f)]

outcome_to_color = {"moderate convergence":'b',
                    "single":'g',
                    "double":'r',
                    "intermediate between single and double":'y'}


def get_backbone_file_params(file : str) -> dict:
    parsed_file = file.split("_")
    file_params ={
        "type": parsed_file[0],
        "N_agents": int(parsed_file[1][1:]),
        "mu": float(parsed_file[2][1:]),
        "theta": float(parsed_file[3][1:]),
        "n_edges": int(float(parsed_file[4][1:])),
    }
    return file_params

def get_homophilic_file_params(file : str) -> dict:
    parsed_file = file.split("_")
    file_params ={
        "type": parsed_file[0],
        "N_agents": int(parsed_file[1][1:]),
        "mu": float(parsed_file[2][1:]),
        "theta": float(parsed_file[3][1:]),
        "n_edges": int(float(parsed_file[4][1:])),
        "prob_friend": float(parsed_file[5][2:]),
        "prob_friend_of_friend": float(parsed_file[6][3:])
    }
    return file_params


def analyse_state(state : pd.DataFrame) -> dict:
    res = {}
    n_agents = state.shape
    X = state.abs().mean()
    res["X"] = X
    attractors_normalized_size = state.value_counts()/n_agents
    n = 1/(attractors_normalized_size*attractors_normalized_size).sum()
    res["n"] = n
    return res
    
def analyse_experiment(experiment : pd.DataFrame) -> dict:
    n_step = experiment.shape[0]
    res = {"X_series":[],
           "n_series":[]}
    for i in range(n_step)[-1:]:
        state_res = analyse_state(experiment.iloc[i,:])
        res["n_series"].append(state_res["n"])
        res["X_series"].append(state_res["X"])

    n_final = res["n_series"][-1]
    if n_final < 1.25:
        res["outcome"] = "single"
    elif n_final < 1.66:
        res["outcome"] = "intermediate between single and double"
    elif n_final < 2.33:
        res["outcome"] = "double"
    else:
        res["outcome"] = "moderate convergence"
    return res

def get_main_outcome(outcomes : list) -> str:
    temp = {}
    for outcome in outcomes:
        if outcome in temp:
            temp[outcome]+=1
        else:
            temp[outcome]=1
    max_occur=0
    max_outcome=""
    for outcome in temp.keys():
        if temp[outcome]>max_occur:
            max_occur = temp[outcome]
            max_outcome = outcome
    return max_outcome



scatter_plot_analysis = {}
for s1 in ["backbone", "homophilic_0.8_0.1", "homophilic_0.9_0.09", "homophilic_0.95_0.04"]:
    scatter_plot_analysis[s1] = {}
    for s2 in ["7", "13", "20"]:
        scatter_plot_analysis[s1][s2]={}
        for s3 in ["x", "y", "c", "outcome", "last_X", "last_n"]:
            scatter_plot_analysis[s1][s2][s3]=[]

#analyse backbone models results

file_idx = 0
while file_idx < len(backbone_files):
    experiments = []
    ref_file = backbone_files[file_idx]
    experiment_params = get_backbone_file_params(ref_file)
    while file_idx < len(backbone_files) and experiment_params == get_backbone_file_params(backbone_files[file_idx]):
        experiments.append(pd.read_csv(data_path + backbone_files[file_idx]).iloc[:,1:].round(2))
        file_idx+=1
    
    experiments_analysis = list(map(analyse_experiment, experiments))

    scatter_plot_analysis["backbone"][str(experiment_params["n_edges"])]["x"].append(experiment_params["mu"])
    scatter_plot_analysis["backbone"][str(experiment_params["n_edges"])]["y"].append(experiment_params["theta"])

    last_X = [experiment_analysis["X_series"][-1] for experiment_analysis in experiments_analysis]
    last_n = [experiment_analysis["n_series"][-1] for experiment_analysis in experiments_analysis]
    scatter_plot_analysis["backbone"][str(experiment_params["n_edges"])]["last_X"].append(np.mean(last_X))
    scatter_plot_analysis["backbone"][str(experiment_params["n_edges"])]["last_n"].append(np.mean(last_n))

    outcomes = [experiment_analysis["outcome"] for experiment_analysis in experiments_analysis]
    main_outcome = get_main_outcome(outcomes)
    scatter_plot_analysis["backbone"][str(experiment_params["n_edges"])]["c"].append(outcome_to_color[main_outcome])
    scatter_plot_analysis["backbone"][str(experiment_params["n_edges"])]["outcome"].append(main_outcome)


#analyse homophilic models results
file_idx = 0
while file_idx < len(homophilic_files):
    experiments = []
    ref_file = homophilic_files[file_idx]
    experiment_params = get_homophilic_file_params(ref_file)
    while file_idx < len(homophilic_files) and experiment_params == get_homophilic_file_params(homophilic_files[file_idx]):
        experiments.append(pd.read_csv(data_path + homophilic_files[file_idx]).iloc[:,1:].round(2))
        file_idx+=1
    
    experiments_analysis = list(map(analyse_experiment, experiments))
    
    exp_str = "homophilic_" + str(experiment_params["prob_friend"]) + "_" + str(experiment_params["prob_friend_of_friend"])
    scatter_plot_analysis[exp_str][str(experiment_params["n_edges"])]["x"].append(experiment_params["mu"])
    scatter_plot_analysis[exp_str][str(experiment_params["n_edges"])]["y"].append(experiment_params["theta"])

    last_X = [experiment_analysis["X_series"][-1] for experiment_analysis in experiments_analysis]
    last_n = [experiment_analysis["n_series"][-1] for experiment_analysis in experiments_analysis]
    scatter_plot_analysis[exp_str][str(experiment_params["n_edges"])]["last_X"].append(np.mean(last_X))
    scatter_plot_analysis[exp_str][str(experiment_params["n_edges"])]["last_n"].append(np.mean(last_n))

    outcomes = [experiment_analysis["outcome"] for experiment_analysis in experiments_analysis]
    main_outcome = get_main_outcome(outcomes)
    scatter_plot_analysis[exp_str][str(experiment_params["n_edges"])]["c"].append(outcome_to_color[main_outcome])
    scatter_plot_analysis[exp_str][str(experiment_params["n_edges"])]["outcome"].append(main_outcome)



for s1 in scatter_plot_analysis.keys():
    for s2 in scatter_plot_analysis[s1].keys():
        fig, ax = plt.subplots()

        scatter = ax.scatter(scatter_plot_analysis[s1][s2]["x"], scatter_plot_analysis[s1][s2]["y"], 
                   c=scatter_plot_analysis[s1][s2]["c"], s=100)

        handles = []
        for key in outcome_to_color.keys():
            handles.append(mpatches.Circle((0, 0), 1, color=outcome_to_color[key]))
        ax.grid(True)
        plt.legend(handles, outcome_to_color.keys())
        plt.title(s1+"_"+s2)
        plt.xlabel("mu")
        plt.ylabel("theta")
        plt.savefig("./scatter_plot/"+s1+"_"+s2+".png")



    