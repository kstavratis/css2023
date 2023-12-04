import numpy as np
import pandas as pd

from src.social_system.backbone_social_system import BackboneSocialSystem
from src.social_system.homophilic_social_system import HomophilicSocialSystem

from .graph_helper_functions import get_subgraph_with_edges_in_range

def run_comparison_experiment(kwargs: dict):
    time_steps = kwargs['time_steps']

    # Two social systems creations START
    homophilic_system = HomophilicSocialSystem(**kwargs)
    
    baseline_system = BackboneSocialSystem(**kwargs)
    # Overwrite info.of one graph to the other.
    # We do this, because we wish for both social systems to have identical initial configurations.
    # Admittedly, this is a very ugly way to do it.
    # TODO IMPLEMENT
    # Implement a "copying" function, where the important info from one system is transferrred to the other.
    baseline_system.graph = get_subgraph_with_edges_in_range(homophilic_system.graph, 0)
    baseline_system.opinions = homophilic_system.opinions.copy()
    baseline_system.tolerances = homophilic_system.tolerances.copy()
    baseline_system.interaction_intensity = homophilic_system.interaction_intensity
    # Two social systems creations FINISH

    base_exp_df = run_experiment(baseline_system, time_steps)
    homophilic_exp_df = run_experiment(homophilic_system, time_steps)

    return base_exp_df, homophilic_exp_df

    

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
    polarization_degrees_array = np.array(polarization_degrees_list)

    # print(opinions_array)
    # The dataframe is of the format:
    # rows: 0...time_steps-1
    # columns:
    #   agent{i} where i \in {0, ..., social_system.nr_agents - 1}. The opinion value of agent{i} at each time step has been recorded.
    #   polarization_degree. The value of `social_system.polarization_degree()` for each time step.
    df = pd.DataFrame(
        np.column_stack( (opinions_array, polarization_degrees_array) ),
        columns=([*[f'agent{i}' for i in range(social_system.nr_agents)], 'polarization_degree'])
    )

    return df