import numpy as np
from numpy import typing as npt
from scipy.sparse import csr_array
import networkx as nx

from typing import Tuple

# Import package constants
from . import opinion_value_lb, opinion_value_ub
from . import link_strength_lb_init, link_strength_ub_init
from . import interaction_intensity
from . import opinion_tolerance
from . import pr_friend, pr_friend_of_friend, pr_random_agent

from .utils.kernel_functions import kernel_functions_dict

class BackboneSocialSystem:

    # TODO IMPLEMENT:
    # WRITE A LINK STRENGTH INITIALIZATION SCHEME!!!
    # Something utilizing the Poission distribution sounds logical...
    def __init__(self,
                nr_agents: int,
                interaction_intensity: float = interaction_intensity,
                opinion_tolerance: float = opinion_tolerance
                ):
        
        self._nr_agents = nr_agents

        self.__rng = np.random.default_rng()

        # TODO DECIDE:
        # What data structure should be used for adjacency?
        # Thinking of a realistic network, a sparse adjacency matrix
        # (a.k.a an adjacency matrix represented by a sparse matrix data structure)
        # seems reasonable.
        # NOTE:
        # The current (naive) implementation uses a dense square matrix.
        # self.adjacency_matrix = self.__rng.integers(low=link_strength_lb_init,
        #                                             high=link_strength_ub_init,
        #                                             size=(nr_agents, nr_agents)
        #                                             )
        
        self.__graph = self.__erdos_renyi_adjacency_generator(nr_agents, 2/nr_agents, 0.7)

        # TODO IMPLEMENT:
        # Write an opinions initialization scheme.
        # Currently, opinions are generated uniformly.

        # unif[a, b) = (b-a) * unif[0,1) + a
        self.opinions = (opinion_value_ub - opinion_value_lb)\
            * self.__rng.uniform(size=(nr_agents))\
                + opinion_value_lb
        
        # TODO IMPLEMENT: Decide on a tolerance initialization scheme.
        # Basic solution: have initial opinion tolerance
        # be the same in all agents of the system.
        self.tolerances = opinion_tolerance * np.ones_like(self.opinions)
        
        self.interaction_intensity = interaction_intensity

        # Define initially empty helper variables
        self._matchings: npt.NDArray[np.bool_] = None

        
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
            sample_size: int = 1,
            pr_friend: float = pr_friend,
            pr_friend_of_friend: float = pr_friend_of_friend,
            pr_random_agent: float = pr_random_agent
    ) -> npt.NDArray[np.bool_]:
        
        output = []

        # TODO IMPLEMENT:
        # Create a function which can be dynamically dictates
        # the PDF generation scheme. Currently, it is hardcoded.

        # Preprocessing for "friend of friend" and "random" choices.
        # Find all cycles that are created in the graph with an edge traversal
        # step of 2 (i.e. the "friend of a friend" case).
        # In this context, "cycles" refers to a friend of mine pointing me
        # which should be ignored (I will not match with myself).
        cycle_pairs_generator = nx.simple_cycles(self.__graph, 2)

        cycles_with_dict = {}
        for i, j in cycle_pairs_generator:
            cycles_with_dict.setdefault(i, []).append(j)
            cycles_with_dict.setdefault(j, []).append(i)

        for source in self.__graph.nodes():

            agent_choice = None # Variable declaration

            # Isolate the subgraph which contains only two "hops",
            # i.e. the graph which contains the source, as well
            # as connections related to the "friend of a friend".
            source_subgraph: nx.DiGraph = nx.ego_graph(self.__graph, source, 2)

            # Remove edges which create circles
            # (i.e. friends suggesting myself to me).
            if source in cycles_with_dict.keys():
                cycle_edges = np.array(cycles_with_dict[source])
                source_array = source * np.ones_like(cycle_edges) # I'm abusing the fact that `source` is both a numerical value and and ID in this multiplication.
                source_subgraph.remove_edges_from(
                    np.column_stack((cycle_edges, source_array))
                )

            pr_friend, pr_friend_of_friend, pr_random_agent =\
                self.__adjust_matching_pdfs(
                    source_subgraph,
                    source,
                    pr_friend,
                    pr_friend_of_friend,
                    pr_random_agent
                )

            # PDF generation scheme is hard-coded to be
            # a linear combination of the cases:
            # 0. choose an immediate friend
            # 1. choose the friend of a friend
            # 2. choose a random agent of the graph
            match_scheme = self.__rng.choice(3, 1, p=[pr_friend, pr_friend_of_friend, pr_random_agent]).item()
            
            candidates_pr = None # Variable declaration

            # TODO FIX
            # The current implementation does not handle cases
            # where the number of friends or firends of friends are insufficient.
            # e.g. an agent can have only one friend, but the `sample_size` is two.

            if match_scheme == 0: # Choose immediate neighbour
                
                positive_edges = []
                positive_friends = []
                for _, friend, ls in source_subgraph.edges(source, data=True):
                    link_strength = ls['weight']
                    if link_strength > 0:
                        positive_friends.append(friend)
                        positive_edges.append(link_strength)

                # "ls" = "link strength"
                candidates_ls = np.zeros(self._nr_agents) # Initialization
                candidates_ls[np.array(positive_friends)] = np.array(positive_edges)

                candidates_pr = np.zeros_like(candidates_ls) # Initialization

                # TODO DECIDE:
                # What normalization scheme should be used?
                # The current implementation considers proportional normalization,
                # but softmax could also be considered (although, it doesn't make
                # as much sense for me).
                normalize_factor = candidates_ls.sum()
                if normalize_factor > 0: # Normalize
                    candidates_pr = candidates_ls / normalize_factor

            elif match_scheme == 1: # Choose neighbour of neighbour
                
                graph_array: csr_array = nx.to_scipy_sparse_array(source_subgraph)
                # Default values return a CSR matrix, with values being 'weight'.

                friends_csr: csr_array = graph_array.maximum(0)
                #friend_array.eliminate_zeros() # We have to see if this takes more time than needed.

                # NOTE: Problematic behaviour of pure matrix multiplication.
                # The pure matrix multiplication presupposes that the agents are
                # aware of the link strengths of agents that are connected
                # with their "enemy" (i.e. negative link strengths).
                # However, in a more realistic scenario, it would make
                # sense that agents are aware of the opinions of their
                # immediate friends (positive link strength) for
                # other agents (either positive or negative),
                # but not of the opinions of "enemies"
                # This is the reason why `immediate_friends_adjacency`
                # is computed.
                candidates_scores = (friends_csr @ graph_array).getrow(source).todense()
                # Retain only the positive scores,
                # i.e. consider only the overall good friend recommendations. 
                candidates_scores = np.maximum(candidates_scores, 0)

                candidates_pr = np.zeros_like(candidates_scores)
                # TODO DECIDE:
                # What normalization scheme should be used?
                # The current implementation considers proportional normalization,
                # but softmax could also be considered (although, it doesn't make
                # as much sense for me).
                normalize_factor = candidates_scores.sum()
                if normalize_factor > 0: # Normalize
                    candidates_pr /= normalize_factor
                

            elif match_scheme == 2: # Choose random agent in the graph

                # NOTE: Friend of a friend may also be selected, which is consistent.
                random_agents = np.array(nx.non_neighbors(self.__graph, source))
                candidates_pr = np.zeros(self._nr_agents)
                candidates_pr[random_agents] = 1 # Uniform distribution
                normalize_factor = candidates_pr.sum() # Unless it's a small graph, very likely that this will be greater than zero.
                if normalize_factor > 0: # Normalize 
                    candidates_pr /= normalize_factor


                # # Friend of a friend case
                # immediate_friends_adjacency = np.maximum(self.adjacency_matrix , 0)
                # neighbour_of_neighbour_adjacency = immediate_friends_adjacency @ self.adjacency_matrix
                # # NOTE: Have the sparse matrices be of CSR format for fast numerical computations.

                # # Avoid all agents of the previous cases
                # # (i.e. "friend" and "friend of a friend")
                # potential_agents_mask = ~((self.adjacency_matrix > 0) |\
                #                         (neighbour_of_neighbour_adjacency > 0))
                
                # # Choice of random agents follows a uniform distribution.
                # normalize_factors = potential_agents_mask.sum(1)[:, np.newaxis]
                # potential_agents_mask = potential_agents_mask.astype(int, copy=False) # Normalization  of bool doesn't make sense.
                # candidates_pr = potential_agents_mask / normalize_factors

            
            agent_choice = self.__rng.choice(self._nr_agents, sample_size, p=candidates_pr)
            output.append(agent_choice)

        return np.vstack(output)


        # NOTE: This presupposes that agent IDs are
        # have a monotonically incrementing increasing
        # integer encoding. 
        agent_ids = np.arange(self.opinions.size)

        # TODO: The case where `sample_size` is greater
        # than the number of non-zero probabilities
        # must be considered for this code to be correct.
        # No need to worry for that in the case of
        # `sample_size = 1`.

        return np.apply_along_axis(
            lambda pdf: self.__rng.choice(agent_ids, size=sample_size, replace=False, p=pdf),
            axis=1, arr=candidates_pr
        )


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

        # Compute kernel values among all agents.
        kf_result = kernel_functions_dict[kf_name](self.opinions, self.tolerances)
        # Consider only the agents that have been matched.
        kf_result = np.where(matchings, kf_result, 0)

        self._change_opinions(kf_result)
        self._change_tolerances(kf_result)


    def _postprocess(self):
        pass



    def _change_opinions(self, kernel_values):

        # opinion_directions[i, j] := xj - xi, i.e. i->j
        opinion_directions = np.tile(self.opinions, reps=(self.opinions.size, 1))
        opinion_directions = opinion_directions - opinion_directions.T

        # TODO FIX:
        # Is the N-Hot Encoding I've written correct and
        # generalizable for `sample_size` > 1?
        # "nhe" := "N-Hot Encoding"
        nr_agents = self._matchings.shape[0]
        
        matchings_nhe = np.zeros(shape=(nr_agents, nr_agents)).astype(bool)
        # `self._matchings.shape[1]`
        # corresponds to the sample size that was selected.`
        rows = np.repeat(
            np.arange(nr_agents)[:, None],
            repeats=self._matchings.shape[1],
            axis=1
        )
        cols = self._matchings

        matchings_nhe[rows, cols] = True

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
        changes_of_opinion = (kernel_values * opinion_directions).mean(
            axis=1,
            where=matchings_nhe
        )
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

        # "ohe" := "One Hot Encoding"
        matchings_ohe = np.identity(
            self._matchings.shape[0]
            )[self._matchings.reshape(-1)].astype(bool)

        changes_of_tolerance = (kernel_values * tolerance_directions).mean(
            axis=1,
            where=matchings_ohe
        )
        # changes_of_opinion[i] = 
        # μ * (sum_{j \in matchings}  k(o_i, o_j, t_i, t_j) * (t_j - t_i)) / |matchings|

        self.tolerances += changes_of_tolerance

    @staticmethod
    def __erdos_renyi_adjacency_generator(
        nr_vertices: int,
        pr_edge_creation: float,
        pr_positive: float,
        is_dense: bool = False
    ) -> nx.DiGraph:
        
        assert 0 <= pr_edge_creation <= 1,\
            f'`pr_edge_creation` expresses a probability.\
                Instead, it has a value of {pr_edge_creation}.'
        assert 0 <= pr_positive <= 1,\
            f'`pr_positive` expresses a probability.\
                Instead, it has a value of {pr_positive}.'

        adjacency_list: nx.DiGraph = None # Variable declaration.
        if is_dense:
            adjacency_list = nx.gnp_random_graph(nr_vertices, pr_edge_creation, directed=True)
        else:
            adjacency_list: nx.DiGraph = nx.fast_gnp_random_graph(nr_vertices, pr_edge_creation, directed=True)
        
        # TODO DECIDE & IMPLEMENT:
        # Think up of a link strength initialization scheme.
        # adjacency_list.add_weighted_edges_from()

        return adjacency_list
    

    @staticmethod
    def __adjust_matching_pdfs(
        graph: nx.DiGraph,
        source,
        pr_friend: float = pr_friend,
        pr_friend_of_friend: float = pr_friend_of_friend,
        pr_random_agent: float = pr_random_agent
    ) -> Tuple[float, float, float]:
        
        # Consider three cases.

        # CASE 1: There is no outgoing edge with positive link strength.
        # Make the probability for choosing a random person 100% (and all else 0).
        # Analogy;
        # Q: What happens if an agent has no friends?
        # A: They can only meet new agents.
        has_friend = False
        for _, _, data in graph.edges(source, data=True):
            if data['weight'] > 0:
                has_friend = True
                break

        if not has_friend:
            return 0, 0, 1


        # CASE 2: There are no positive suggestions from friends.
        # Make the probability of choosing friend of a friend 0%
        # and adjust the remaining probabilities.
        # Analogy;
        # Q: What happens if their social circle does not suggest any other agent
        # A: All the agents belonging in the clique can only either meet their friends or meet new agents.
        
        graph_array: csr_array = nx.to_scipy_sparse_array(graph)
        # Default values return a CSR matrix, with values being 'weight'.

        friends_array: csr_array = graph_array.maximum(0)
        #friend_array.eliminate_zeros() # We have to see if this takes more time than needed.
        friend_suggestions = (friends_array @ graph_array).getrow(source)
        exists_friend_suggestion = friend_suggestions.maximum(0).count_nonzero() > 0

        # TODO DECIDE:
        # How do we distribute the arising void in the PDF,
        # since now `pr_friend_of_friend` must be equal to zero?
        # With the current implementation, all of the probability
        # is transferred to the `pr_friend`, since it could be
        # assumed that the person is shy, thus not going out of their
        # way to converse with random people in the street.
        if not exists_friend_suggestion:
            return pr_friend + pr_friend_of_friend, 0, pr_random_agent


        # CASE 3: None of the two above cases.
        # Proceed with the provided probabilities.
        return pr_friend, pr_friend_of_friend, pr_random_agent