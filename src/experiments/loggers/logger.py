from pathlib import Path
import os

import pandas as pd

from ..experiments import ComparisonExperimentRecord

# ==================== Initialize path variables START ====================

# cfp := "current file path"
# Using `resolve` as per the official documentation's recommendation
# https://docs.python.org/3/library/pathlib.html#pathlib.PurePath.parent
__cfp = Path(__file__).resolve()
# Q: Why this weird syntax?
# A: This syntax raise an exception in case that the RHS does not contain
#   only *one* element
[__root_dir] = [parent_path.parent for parent_path in __cfp.parents if parent_path.name == 'src']

# ==================== Initialize path variables FINISH ====================

def write_social_system_experiments(
        comparison_experiment_record: ComparisonExperimentRecord,
        experiments_config: dict,
        name_prefix: str = '',
        name_postfix: str = ''
):
    """
    Write all comparative experiments conducted in CSV files

    Parameters
    ----------
    comparison_experiment_record : ComparisonExperimentRecord
        Holds the experiments conducted with a set of fixed configuration parameters 
    experiments_config : dict
        The fixed configurations parameters with which all experiments were initialized.
        Needs to include the following fields:
        1. graph
        2. interaction_intensity
        3. opinion_tolerance
        4. pr_edge_creation
        5. pr_friend_default
        6. pr_friend_of_friend_default

    name_prefix : str, optional
        A substring which will be written at the beginning of all output files. ; by default ''
    name_postfix : str, optional
        A substring which will be written at the end of all output files. ; by default ''
    """
    # Isolate experiment parameters.  
    n_agents = experiments_config['graph']
    mu = experiments_config['interaction_intensity']
    theta = experiments_config['opinion_tolerance']
    mean_edge_per_agent = experiments_config['pr_edge_creation']*n_agents
    pr_meeting_friend = experiments_config['pr_friend_default']
    pr_meeting_friend_of_friend = experiments_config['pr_friend_of_friend_default']

    # Create path to store the results.
    # In case that the folder does not exist, it is created. 

    # path_to_store = Path(__root_dir / 'results')  #for computing on personnal laptop
    path_to_store = Path(os.environ['SCRATCH'] + '/abm_results')    #for computing on euler
    
    path_to_store.mkdir(exist_ok=True)
    
    for expmt_idx, experiment in  enumerate(comparison_experiment_record.experiments):
            
        # Split the data into the backbone part and the homophilic part
        backbone_data, homophilic_data = experiment


        shared_config = f'N{n_agents}_μ{mu}_θ{theta}_E{mean_edge_per_agent}'
        homophily_config = f'pf{pr_meeting_friend}_pff{pr_meeting_friend_of_friend}'

        backbone_subpath = f'{name_prefix}' + 'backbone_' + shared_config + f'_{expmt_idx}.csv'
        homophilic_subpath = f'{name_prefix}' + 'homophilic_' + shared_config + '_' + homophily_config + f'_{expmt_idx}.csv'

        # Write the files.
        backbone_data.to_csv(Path(path_to_store / backbone_subpath), float_format="%.3f")
        homophilic_data.to_csv(Path(path_to_store / homophilic_subpath), float_format="%.3f")