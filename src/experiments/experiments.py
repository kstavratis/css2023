from pathlib import Path; import os
from copy import deepcopy
from typing import List, Tuple
from dataclasses import dataclass

import numpy as np
import pandas as pd

import networkx as nx

from src.social_system.backbone_social_system import BackboneSocialSystem
from src.social_system.homophilic_social_system import HomophilicSocialSystem

from ..social_system.utils.graph_helper_functions import reduce_to_graph_with_edges_in_range

@dataclass
class ComparisonExperimentRecord:
    """
    A data structure which holds information about experiments
    conducted comparatively with respect to the social systems'
    mechanisms.

    Attributes
    ----------
    `initial_backbone_social_system`: BackboneSocialSystem
        The initial `BackboneSocialSystem` configuration shared by all experiments.
        It may be thought as the "starting point" of all experiments.

    `initial_homophilic_social_system`: HomophilicSocialSystem
        The initial `HomophilicSocialSystem` configuration shared by all experiments.
        It may be thought as the "starting point" of all experiments.

    `experiments`: List[Tuple[pd.DataFrame, pd.DataFrame]]
        Possible evolutions of experiments that `initial_backbone_social_system` and
        `initial_homophilic_social_system` as their starting points.
        The difference in evolution (results per time step/iteration) is due to
        the inherent randmoness of the experiments
        (e.g. random graph creation, random matching etc.) 

    
    Methods
    -------

    `experiment(id: int) -> Tuple[pd.DataFrame, pd.DataFrame]`
    Returns a pair of experiment evolutions with the same starting configurations.

    `backbone_opinions_in_experiment(id: int) -> pd.DataFrame`
    Returns the evolution of the `id`-th `BackboneSocialSystem` experiment

    `homophilic_opinions_in_experiment(id: int) -> pd.DataFrame`
    Returns the evolution of the `id`-th `HomophilicSocialSystem` experiment 
    """

    initial_backbone_social_system: BackboneSocialSystem
    initial_homophilic_social_system: HomophilicSocialSystem
    experiments: List[Tuple[pd.DataFrame, pd.DataFrame]]

    # Defining convenience functions for more readability.

    def experiment(self, id: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
        return self.experiments[id]
    
    def backbone_opinions_in_experiment(self, id: int) -> pd.DataFrame:
        return self.experiments[id][0]
    
    def homophilic_opinions_in_experiment(self, id: int) -> pd.DataFrame:
        return self.experiments[id][1]

from .loggers.logger import _root_dir


def run_comparison_experiments(
        nr_experiments: int,
        time_steps: int,
        system_kwargs: dict
    ) -> ComparisonExperimentRecord:
    """
    Conducts experiments with `BackboneSocialSystem`s and `HomophilicSocialSystem`s
    which start from the same initial configuration
    (e.g. opinion values, interaction intensities μ, tolerances θ, connections, etc. )
    
    If it makes it clearer to understand,
    a *new* pair of initial social system configurations
    is generated at *each experiment*.
    Each social system is then run for the provided number of time steps. 

    Parameters
    ----------
    nr_experiments : int
        The number of different comparison experiment that will be conducted.
    time_steps : int
        The number of time steps each experiment will be executed for. 
    system_kwargs : dict
        Keyword arguments/Named parameters required by the different constructors
        of the Social Systems classes.
        The classes currently being considered for experimentation are
        `BackboneSocialSystem` and `HomophilicSocialSystem`.


    Returns
    -------
    Tuple[BackboneSocialSystem, HomophilicSocialSystem, List[Tuple[pd.DataFrame, pd.DataFrame]]]
    
    : BackboneSocialSystem
        The initial configuration of the social system with which all experiments were conducted.

        NOTE: All edge connections have a link strength associated with them;
        this is not part of the definition of a `BackboneSocialSystem`.
        The link strengths of this system are identical to the (initially) positive
        link strengths of the `HomophilicSocialSystem` output.

    : HomophilicSocialSystem:
        The initial configuration of the homophilic social system
        with which all experiments were conducted.

    : List[Tuple[pd.DataFrame, pd.DataFrame]]
        A list of pandas DataFrame pairs.

        Each pair constitutes a comparison experiment conducted between
        a BackboneSocialSystem and a HomophilicSocialSystem with identical initial configurations.
        
        The content of each pair (b: pd.DataFrame, h: pd.DataFrame)
        is the opinion values of the agents of the BackboneSocialSystem (b) and
        the HomophilicSocialSystem (h) at each time step

        third_output[e][0][t] := the evolution of opinion values held by agents
            of the *BackboneSocialSystem* at experiment #e.
        third_output[e][1][t] := the evolution of opinion values held by agents
            of the *HomophilicSocialSystem* at experiment #e.

    """
   

    # Two social systems creations START
    homophilic_system = HomophilicSocialSystem(**system_kwargs)
    
    baseline_system = BackboneSocialSystem(**system_kwargs)
    # Overwrite info.of one graph to the other.
    # We do this, because we wish for both social systems to have identical initial configurations.
    # Admittedly, this is a very ugly way to do it.
    # TODO IMPLEMENT
    # Implement a "copying" function, where the important info from one system is transferrred to the other.
    baseline_graph = deepcopy(homophilic_system.graph); reduce_to_graph_with_edges_in_range(baseline_graph, 0)
    baseline_system.graph = baseline_graph
    baseline_system.opinions = homophilic_system.opinions.copy()
    baseline_system.tolerances = homophilic_system.tolerances.copy()
    baseline_system.interaction_intensity = homophilic_system.interaction_intensity
    # Two social systems creations FINISH

    output = []

    # ========== Store the initial graphs in files START ==========
    n_agents = system_kwargs['graph']
    mu = system_kwargs['interaction_intensity']
    theta = system_kwargs['opinion_tolerance']
    mean_edge_per_agent = system_kwargs['pr_edge_creation']*n_agents
    pr_meeting_friend = system_kwargs['pr_friend_default']
    pr_meeting_friend_of_friend = system_kwargs['pr_friend_of_friend_default']
    
    shared_config = f'N{n_agents}_μ{mu:.3g}_θ{theta:.3g}_E{mean_edge_per_agent}'
    homophily_config = f'pf{pr_meeting_friend}_pff{pr_meeting_friend_of_friend}'

    baseline_subpath = Path('backbone_' + shared_config + '_' + homophily_config + '_init_graph.gml')
    homophilic_subpath = Path('homophilic_' + shared_config + '_' + homophily_config + '_init_graph.gml')

    # file_storage_path = _root_dir / 'results' # Run locally
    file_storage_path = Path(os.environ['SCRATCH'] + '/abm_results')# Run on Euler
    nx.write_gml(baseline_system.graph, file_storage_path / baseline_subpath, str)
    nx.write_gml(homophilic_system.graph, file_storage_path / homophilic_subpath, str)
    # TODO IMPLEMENT:
    # Apparently, `np.int` types, which is the type of edges' `weight`
    # are not considered instances of `int`. i.e. `isinstance(np.int) == int` is false.
    # Therefore they are rejected by the internal parser of `gml.py`.
    # Consequently, the weights should be converted to int again when read.
    
    # ========== Store the initial graphs in files FINISH ==========

    # Run different experiments with identical parameterizations.
    # Differences will be a consequence of randomness!
    for i in range(nr_experiments):
        base_exp_df = run_experiment(deepcopy(baseline_system), time_steps, i, system_kwargs)
        homophilic_exp_df = run_experiment(deepcopy(homophilic_system), time_steps, i, system_kwargs)
        output.append((base_exp_df, homophilic_exp_df))

    return ComparisonExperimentRecord(baseline_system, homophilic_system, output)

    

def run_experiment(
        social_system: BackboneSocialSystem, time_steps: int,
        experiment_id: int = None, social_system_kwargs: dict = None
    ) -> pd.DataFrame:
    # Initialize log data structures.
    opinions_list = [social_system.opinions.copy(),]

    # Conduct experiment START
    for _ in range(time_steps):
        social_system.step()
        # Log information
        opinions_list.append(social_system.opinions.copy())
    # Conduct experiment FINISH
        
    # ========== Store the final graphs in files START ==========
    if experiment_id is not None:
        n_agents = social_system_kwargs['graph']
        mu = social_system_kwargs['interaction_intensity']
        theta = social_system_kwargs['opinion_tolerance']
        mean_edge_per_agent = social_system_kwargs['pr_edge_creation']*n_agents
        pr_meeting_friend = social_system_kwargs['pr_friend_default']
        pr_meeting_friend_of_friend = social_system_kwargs['pr_friend_of_friend_default']
        
        basic_config = f'N{n_agents}_μ{mu:.3g}_θ{theta:.3g}_E{mean_edge_per_agent}'
        additional_config = f'pf{pr_meeting_friend}_pff{pr_meeting_friend_of_friend}'

        social_system_pathname = Path(
            f'{"homophilic" if type(social_system) == HomophilicSocialSystem else "backbone"}' +\
                '_' + basic_config + additional_config + '_final_graph_' + f'{experiment_id}' + '.gml'
        )

        # file_storage_path = _root_dir / 'results' # Run locally
        file_storage_path = Path(os.environ['SCRATCH'] + '/abm_results')# Run on Euler
        nx.write_gml(social_system.graph, file_storage_path / social_system_pathname, str)
        # TODO IMPLEMENT:
        # Apparently, `np.int` types, which is the type of edges' `weight`
        # are not considered instances of `int`. i.e. `isinstance(np.int) == int` is false.
        # Therefore they are rejected by the internal parser of `gml.py`.
        # Consequently, the weights should be converted to int again when read.
    # ========== Store the final graphs in files FINISH ==========

    

    # +1 to account for the initial configuration which is also recorded.
    opinions_array = np.stack(opinions_list, 0).reshape(time_steps+1, social_system.nr_agents)

    # The dataframe is of the format:
    # rows: 0...time_steps-1
    # columns:
    #   agent{i} where i \in {0, ..., social_system.nr_agents - 1}. The opinion value of agent{i} at each time step has been recorded.
    #   polarization_degree. The value of `social_system.polarization_degree()` for each time step.
    df = pd.DataFrame(
        opinions_array,
        columns=([*[f'agent{i}' for i in range(social_system.nr_agents)]])
    )

    return df

