import warnings
# warnings.filterwarnings('error')

import numpy as np
from numpy import typing as npt

import networkx as nx

# Import package constants
from . import opinion_value_lb, opinion_value_ub
from . import interaction_intensity
from . import opinion_tolerance

from .utils.kernel_functions import kernel_functions_dict

class BackboneSocialSystem:

    # TODO IMPLEMENT:
    # WRITE A LINK STRENGTH INITIALIZATION SCHEME!!!
    # Something utilizing the Poission distribution sounds logical...
    def __init__(self,
                graph: nx.Graph or int,
                pr_edge_creation: float = None,
                interaction_intensity: float = interaction_intensity,
                opinion_tolerance: float = opinion_tolerance,
                **kwargs
                ):
        
        # `**kwargs` are inserted in case the function is overriden by a different class
        # and requires more arguments (contravariant parameters).

        if isinstance(graph, nx.Graph):
            self.graph = graph
        elif isinstance(graph, int):
            self.graph = self.weighted_erdos_renyi_graph_generator(graph, pr_edge_creation, **kwargs)
        else:
            raise TypeError(f'Argument `graph` was expected to be of type {nx.Graph} or {int}.\n'
                      f'{type(graph)} was provided instead.')
        
        # TODO IMPLEMENT:
        # Write an opinions initialization scheme.
        # Currently, opinions are generated uniformly.

        rng = np.random.default_rng()

        # unif[a, b) = (b-a) * unif[0,1) + a
        self.opinions = (opinion_value_ub - opinion_value_lb)\
            * rng.uniform(size=self.nr_agents)\
                + opinion_value_lb
        
        # TODO IMPLEMENT: Decide on a tolerance initialization scheme.
        # Basic solution: have initial opinion tolerance
        # be the same in all agents of the system.
        # θ, C
        self.tolerances = opinion_tolerance * np.ones_like(self.opinions)
        
        # μ
        self.interaction_intensity = interaction_intensity

        # Define initially empty helper variables
        self._matchings: npt.NDArray[np.bool_] = None

    
    @property
    def nr_agents(self):
        return self.graph.number_of_nodes()
    
    def polarization_degree(self):
        return np.abs(self.opinions).mean()
    
    # TODO IMPLEMENT
    # Implement the generalized clusters metric.
    # See: https://jasss.soc.surrey.ac.uk/9/3/8.html  3.4, 3.5 & 3.6
    # Search for different metrics or come up with our own metrics.

        
    def step(self):

        # Processing required before the time step takes place
        self._preprocess()

        # Create pairings of agents.
        self._matchings = self._match()
        # Have interactions take place.
        self._interact(self._matchings)


        # Processing required after the interactions take place.
        self._postprocess()

    def _preprocess(self):
        pass


    def _match(
            self,
            sample_size: int = 1
    ) -> npt.NDArray[np.bool_]:
        
        # Random neighbours are chosen uniformly be each agent.
        
        output = []
        rng = np.random.default_rng()

        for source in self.graph.nodes():

            # Variable declaration and initialization
            agent_choice = -np.ones(sample_size) # `-1` represents "no choice".
            # TODO: Handle case where 0 < #neighbours < sample_size
            # Priority: Low, as we will mostly be dealing with the case of
            # `sample_size` = 1.

            # Isolate the subgraph which contains only one "hop",
            # i.e. the graph which contains the source
            # and its immediate neighbours.
            source_subgraph: nx.DiGraph = nx.ego_graph(self.graph, source, 1)
            # Uniformly pick one of the neighbours.
            neighbours = list(source_subgraph.successors(source))
            if neighbours:
                agent_choice = rng.choice(neighbours, sample_size, replace=False)

            output.append(agent_choice)

        return np.vstack(output).astype(np.int_)



    def _interact(self, matchings: npt.NDArray[np.bool_], kf_name: str = 'bc') -> None :
        # Dynamic sanity check(s)
        matchings.ndim == 2

        # NOTE: If we wish for the code to be more general,
        # the kernel computation should be done inside the
        # `_change_opinions` and `_change_tolerances`
        # specific implementations, where they might require them.
        # It ends up requiring more computations, but
        # provides larger generalizability.
        # This is pointed out, because there are schemes
        # which do not require the kernel values at all.

        # Consider only the agents that have been matched.
        rows = np.repeat(
            np.arange(matchings.shape[0])[:, None],
            repeats=matchings.shape[1]
        ).flatten()
        cols = matchings.flatten()

        # Compute kernel values among all agents.
        kf_temp = kernel_functions_dict[kf_name](self.opinions, self.tolerances)

        # Declare and initialize variable.
        kf_result = np.zeros_like(kf_temp)
        kf_result[rows, cols] = kf_temp[rows, cols]

        

        self._change_opinions(kf_result)
        self._change_tolerances(kf_result)


    def _postprocess(self):
        pass



    def _change_opinions(self, kernel_values):

        # opinion_directions[i, j] := xj - xi, i.e. i->j
        opinion_directions = np.tile(self.opinions, reps=(self.opinions.size, 1))
        opinion_directions = opinion_directions - opinion_directions.T

        matchings_nhe = self.__n_hot_encoding(self._matchings)

        # Our assumption is that the final change of opinion of an agent
        # is the mean of total interactions they had in a single time step.
        # TODO DECIDE:
        # It should be decided whether a weighted average
        # will be computed instead.
        # The weights could be dependent on the link strength.
        # Personally (Constantine), I don't think it is required,
        # as the matching scheme which will be implemented is already
        # dependent on the link strength
        # (think of what this means for the expected value).
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            changes_of_opinion = (kernel_values * opinion_directions).mean(
                axis=1,
                where=matchings_nhe & (kernel_values != 0)
            )
        # Handle cases where the agent does not interact with anyone,
        # thus their mean value would result in nan
        np.nan_to_num(changes_of_opinion, copy=False)
        # changes_of_opinion[i] =
        # μ * (sum_{j \in matchings}  k(o_i, o_j, t_i, t_j) * (o_j - o_i)) / |matchings|

        self.opinions += self.interaction_intensity * changes_of_opinion 

    def _change_tolerances(self, kernel_values):

        # NOTE: This is but one possible generation scheme.
        # Other schemes, which do not require `kernel values`
        # may be defined.
        

        # tolerance_directions[i, j] := tj - ti, i.e. i->j
        tolerance_directions = np.tile(self.tolerances, reps=(self.tolerances.size, 1))
        tolerance_directions = tolerance_directions - tolerance_directions.T

        # Consider only agents which have interacted.
        matchings_nhe = self.__n_hot_encoding(self._matchings)

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            changes_of_tolerance = (kernel_values * tolerance_directions).mean(
                axis=1,
                where=matchings_nhe & (kernel_values != 0)
            )
        # Handle cases where the agent does not interact with anyone,
        # thus their mean value would result in nan
        np.nan_to_num(changes_of_tolerance, copy=False)
        # changes_of_opinion[i] = 
        # μ * (sum_{j \in matchings}  k(o_i, o_j, t_i, t_j) * (t_j - t_i)) / |matchings|

        self.tolerances += changes_of_tolerance


    @staticmethod
    def weighted_erdos_renyi_graph_generator(
        nr_vertices: int,
        pr_edge_creation: float,
        is_dense: bool = False,
        **kwargs
    ) -> nx.DiGraph:
        
        # Dynamic sanity checks.
        if not(0 <= pr_edge_creation <= 1):
            raise ValueError(
                f'`pr_edge_creation` expresses a probability.\
                Instead, it has a value of {pr_edge_creation}.'
            )

        if not isinstance(is_dense, bool):
            raise TypeError(
                f'Expected argument `is_dense` to be of type {bool}. '\
                    f'Received {type(is_dense)} instead.'
        )

        output_graph: nx.DiGraph = None # Variable declaration.
        if is_dense:
            output_graph = nx.gnp_random_graph(nr_vertices, pr_edge_creation, directed=True)
        else:
            output_graph: nx.DiGraph = nx.fast_gnp_random_graph(nr_vertices, pr_edge_creation, directed=True)

        return output_graph
    
    @staticmethod
    def __n_hot_encoding(matchings: npt.NDArray[np.int_]):

        nr_agents = matchings.shape[0]

        # The value `-1` refers to agents that have not matched with any other agent.
        to_consider_mask = matchings > -1
        
        # "nhe" := N-Hot Encoding
        matchings_nhe = np.zeros(shape=(nr_agents, nr_agents)).astype(bool)
        # `self._matchings.shape[1]`
        # corresponds to the sample size that was selected.
        rows = np.repeat(
            np.arange(nr_agents)[:, None],
            repeats=matchings.shape[1],
            axis=1
        )[to_consider_mask].astype(np.int_)
        cols = matchings[to_consider_mask]

        matchings_nhe[rows, cols] = True

        return matchings_nhe