from copy import deepcopy
from typing import List, Tuple
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.social_system.backbone_social_system import BackboneSocialSystem
from src.social_system.homophilic_social_system import HomophilicSocialSystem

from ..social_system.utils.graph_helper_functions import reduce_to_graph_with_edges_in_range

@dataclass
class __ComparisonExperimentRecord:

    initial_backbone_social_system: BackboneSocialSystem
    initial_homophilic_social_system: HomophilicSocialSystem
    experiments: Tuple[BackboneSocialSystem, HomophilicSocialSystem, List[Tuple[pd.DataFrame, pd.DataFrame]]]

    # Defining convenience functions for more readability.

    def experiment(self, id: int):
        return self.experiments[id]
    
    def backbone_opinions_in_experiment(self, id: int) -> pd.DataFrame:
        return self.experiments[id][0]
    
    def homophilic_opinions_in_experiment(self, id: int) -> pd.DataFrame:
        return self.experiments[id][1]



def run_comparison_experiments(
        nr_experiments: int,
        time_steps: int,
        system_kwargs: dict
    ) -> __ComparisonExperimentRecord:
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
    # Run different experiments with identical parameterizations.
    # Differences will be a consequence of randomness!
    for _ in range(nr_experiments):
        base_exp_df = run_experiment(deepcopy(baseline_system), time_steps)
        homophilic_exp_df = run_experiment(deepcopy(homophilic_system), time_steps)
        output.append((base_exp_df, homophilic_exp_df))

    return __ComparisonExperimentRecord(baseline_system, homophilic_system, output)

    

def run_experiment(social_system: BackboneSocialSystem, time_steps: int) -> pd.DataFrame:
    # Initialize log data structures.
    opinions_list = [social_system.opinions.copy(),]
    polarization_degrees_list = [social_system.polarization_degree(),]

    # Conduct experiment START
    for _ in range(time_steps):
        social_system.step()
        # Log information
        opinions_list.append(social_system.opinions.copy())
        polarization_degrees_list.append(social_system.polarization_degree())
    # Conduct experiment FINISH

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

