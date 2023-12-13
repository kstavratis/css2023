from typing import List, Dict

from src.social_system.utils.help_constant_structures import Guillible, Adamant, Tolerant, Impressionable

from .. import nr_agents

def prepare_experiment_configurations(nr_agents: int = nr_agents) -> List[Dict]:

    kwargs_list = []

    # TODO IMPLEMENT:
    # Create and finalize configurations for:
    # - `expected_edges_per_agent`
    # - `(pr_friend, pr_friend_of_friend)`
    # and name them.
    # See `help_constant_structures.py` 
    # for persuasiveness_params in (Guillible, Adamant, Tolerant, Impressionable):
    for expected_edges_per_agent in [3]:#, 10, 20):
        for pr_friend, pr_friend_of_friend in ((0.8, 0.1), (0.9, 0.09), (0.95, 0.04)):
            for mu in [0.1+0.2*i for i in range(5)]:
                for theta in [0.1+0.2*i for i in range(5)]:
                    kwargs = {
                        'graph' : nr_agents,
                        'interaction_intensity': mu,
                        'opinion_tolerance': theta,

                        'pr_edge_creation': expected_edges_per_agent / nr_agents,
                        'pr_positive': 0.0, # In the current implementation, this parameter is ignored.
                        'pr_friend_default': pr_friend,
                        'pr_friend_of_friend_default': pr_friend_of_friend
                    }
                    kwargs_list.append(kwargs)

    return kwargs_list