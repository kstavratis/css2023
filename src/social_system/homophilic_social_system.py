import numpy as np
from numpy import typing as npt
from scipy.sparse import csr_array

import networkx as nx

from collections import defaultdict 

from typing import Tuple

from .utils.graph_helper_functions import increase_weights_of_edges

# Import SUPERCLASS(ES)
from .backbone_social_system import BackboneSocialSystem

# Import CONSTANT(S)
from . import link_strength_lb_init, link_strength_ub_init

class HomophilicSocialSystem(BackboneSocialSystem):

    def __init__(
        self,
        pr_friend_default: float,
        pr_friend_of_friend_default: float,
        **kwargs
    ):
        # Initialize the parent `BackboneSocialSystem`
        super().__init__(**kwargs)

        # ==================== Dynamic sanity checks of input START ====================
        if not(0 <= pr_friend_default <= 1):
            raise ValueError(
                'Argument `pr_friend_default` expresses part of a PDF.'\
                    'Consequently, Its values are bounded in the margin [0, 1].\n'\
                        f'Instead, a value of {pr_friend_default} was provided.'
            )
        if not(0 <= pr_friend_of_friend_default <= 1):
            raise ValueError(
                'Argument `pr_friend_of_friend_default` expresses part of a PDF.'\
                    'Consequently, Its values are bounded in the margin [0, 1].\n'\
                        f'Instead, a value of {pr_friend_of_friend_default} was provided.'
            )
        

        if pr_friend_default + pr_friend_of_friend_default > 1:
            raise ValueError(
                'Arguments `pr_friend_default` and `pr_friend_of_friend_default` express parts of a PDF.'\
                    'Consequently, Their sum is bounded in the margin [0, 1].\n'\
                        f'Instead, they have a sum of {pr_friend_default + pr_friend_of_friend_default}.'
            )
        
        # ==================== Dynamic sanity checks of input FINISH ====================


        # Store probabilities which will be used
        # as the default values of the probabilities
        # in the `_match` function.
        self.pr_friend_default = pr_friend_default
        self.pr_friend_of_friend_default = pr_friend_of_friend_default

    @property
    def pr_random_agent_default(self):
        return 1 - (self.pr_friend_default + self.pr_friend_of_friend_default)


    def _match(
            self,
            sample_size: int = 1,
            pr_friend: float = None,
            pr_friend_of_friend: float = None,
    ) -> npt.NDArray[np.bool_]:

        # Automatically filling in default named parameter values.
        if pr_friend is None:
            pr_friend = self.pr_friend_default
        if pr_friend_of_friend is None:
            pr_friend_of_friend = self.pr_friend_of_friend_default

        
        output = []
        rng = np.random.default_rng()

        # TODO IMPLEMENT:
        # Create a function which can be dynamically dictates
        # the PDF generation scheme. Currently, it is hardcoded.

        # Preprocessing for "friend of friend" and "random" choices.
        # Find all cycles that are created in the graph with an edge traversal
        # step of 2 (i.e. the "friend of a friend" case).
        # In this context, "cycles" refers to a friend of mine pointing me
        # which should be ignored (I will not match with myself).
        cycle_pairs_generator = nx.simple_cycles(self.graph, 2)

        # cycles_with_dict[nodeI] -> list[nodesJ]
        # `nodeI` forms a cycle with all nodes `nodesJ`.
        cycles_with_dict = defaultdict(list)
        for i, j in cycle_pairs_generator:
            cycles_with_dict[i].append(j)
            cycles_with_dict[j].append(i)

        for source in self.graph.nodes():

            agent_choice = None # Variable declaration

            # Isolate the subgraph which contains only two "hops",
            # i.e. the graph which contains the source, as well
            # as connections related to the "friend of a friend".
            source_subgraph: nx.DiGraph = nx.ego_graph(self.graph, source, 2)

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
                    1 - (pr_friend + pr_friend_of_friend)
                )

            # PDF generation scheme is hard-coded to be
            # a linear combination of the cases:
            # 0. choose an immediate friend
            # 1. choose the friend of a friend
            # 2. choose a random agent of the graph
            match_scheme = rng.choice(3, 1, p=[pr_friend, pr_friend_of_friend, pr_random_agent]).item()
            
            candidates_pr = None # Variable declaration

            # TODO FIX
            # The current implementation does not handle cases
            # where the number of friends or firends of friends are insufficient.
            # e.g. an agent can have only one friend, but the `sample_size` is two.
            # This is given low priority, since the most usual case is `sample_size=1`

            if match_scheme == 0: # Choose immediate neighbour
                
                positive_edges = []
                positive_friends = []
                for _, friend, ls in source_subgraph.edges(source, data=True):
                    link_strength = ls['weight']
                    if link_strength > 0:
                        positive_friends.append(friend)
                        positive_edges.append(link_strength)

                # "ls" = "link strength"
                candidates_ls = np.zeros(self.nr_agents) # Initialization
                candidates_ls[np.array(positive_friends)] = np.array(positive_edges)

                candidates_pr = np.zeros_like(candidates_ls) # Initialization

                # Normalize using proportional normalization.
                normalize_factor = candidates_ls.sum()
                if normalize_factor > 0: # Normalize
                    candidates_pr = candidates_ls / normalize_factor

            elif match_scheme == 1: # Choose neighbour of neighbour
                
                graph_array: csr_array = nx.to_scipy_sparse_array(source_subgraph)
                # Default values return a CSR matrix, with values being 'weight'.

                friends_csr: csr_array = graph_array.maximum(0)
                #friend_array.eliminate_zeros() # We have to see if this takes more time than needed.
                
                # Declare and initialize variable.
                candidates_scores = np.zeros(self.nr_agents)
                candidate_ids = list(source_subgraph.nodes)
                # NOTE: Problematic behaviour of pure matrix multiplication.
                # The pure matrix multiplication presupposes that the agents are
                # aware of the link strengths of agents that are connected
                # with their "enemy" (i.e. negative link strengths).
                # However, in a more realistic scenario, it would make
                # sense that agents are aware of the opinions of their
                # immediate friends (positive link strength) for
                # other agents (either positive or negative),
                # but not of the opinions of "enemies"
                # This is the reason why `friends_csr` was computed.
                candidates_scores[candidate_ids] = (friends_csr @ graph_array)[[candidate_ids.index(source)], :].todense().ravel()
                # Retain only the positive scores,
                # i.e. consider only the overall good friend recommendations. 
                candidates_scores = np.maximum(candidates_scores, 0)

                # Variable initialization
                candidates_pr = np.zeros_like(candidates_scores)
                
                # Normalize using proportional normalization.
                normalize_factor = candidates_scores.sum()
                if normalize_factor > 0: # Normalize
                    candidates_pr = candidates_scores / normalize_factor
                

            elif match_scheme == 2: # Choose random agent in the graph

                # NOTE: Friend of a friend may also be selected, which is consistent.
                random_agents = list(nx.non_neighbors(self.graph, source)) # Nodes the agent is not connected with
                # The agent might still meet people they have a negative opinion of at random.
                random_agents.extend([
                    ra for ra in source_subgraph.neighbors(source)
                    if source_subgraph.has_edge(source, ra) and
                    source_subgraph.edges[source, ra]['weight'] <= 0
                ])
                random_agents = np.array(random_agents)
                candidates_pr = np.zeros(self.nr_agents)
                candidates_pr[random_agents] = 1 # Uniform distribution

                # Normalize using proportional normalization.
                normalize_factor = candidates_pr.sum() # Unless it's a small graph, very likely that this will be greater than zero.
                if normalize_factor > 0: # Normalize 
                    candidates_pr /= normalize_factor

            # TODO FIX
            # This command can break in case where there are not enough
            # options fo the agents to choose from
            # when `sample_size` is too large
            # (for very sparse graphs, this can also happen for `sample_size=2`)
            # One solution would be to call the matching scheme one pairing at a time
            # i.e. do one matching `sample_size` times, excluding the agents
            # chosen in the previous round.
            # This is too expensive to do for the current simulation.
            # Moreover currently we are focusing on `sample_size=1`.
            agent_choice = rng.choice(self.nr_agents, sample_size, replace=False, p=candidates_pr)
            output.append(agent_choice)

        return np.vstack(output)
    
    def _interact(self, matchings: npt.NDArray[np.bool_], kf_name: str = 'bc'):

        previous_opinions = self.opinions.copy()
        # Have the agents interact as they would
        # (currently they change their opinion and tolerances).
        super()._interact(matchings, kf_name)

        # TODO IMPLEMENT
        # The `threshold` must be either an input/parameter or
        # associated with the kernel function used.
        threshold = 0

        link_incredecrements = \
            self.__compute_link_strength_incredecrements(self.opinions, previous_opinions, matchings, threshold)
        
        increase_weights_of_edges(self.graph, link_incredecrements)


    @staticmethod
    def __compute_link_strength_incredecrements(
        current_opinions,
        previous_opinions,
        matchings,
        threshold: float = 0
    ) -> npt.NDArray[np.int_]:

        was_impacted = np.absolute(current_opinions - previous_opinions) > threshold

        was_impacted_indices = np.where(was_impacted)[0]
        increment_links = np.vstack((
            np.repeat(was_impacted_indices, matchings.shape[1]),
            matchings[was_impacted_indices].ravel(),
            np.ones(was_impacted_indices.size * matchings.shape[1])
        )).T.astype(np.int_)



        not_impacted = ~was_impacted

        not_impacted_indices = np.where(not_impacted)[0]
        decrement_links = np.vstack((
            np.repeat(not_impacted_indices, matchings.shape[1]),
            matchings[not_impacted_indices].ravel(),
            -np.ones(not_impacted_indices.size * matchings.shape[1])
        )).T.astype(np.int_)

        return np.vstack((increment_links, decrement_links))


    
    @staticmethod
    def __adjust_matching_pdfs(
        graph: nx.DiGraph,
        source,
        pr_friend: float,
        pr_friend_of_friend: float,
        pr_random_agent: float
    ) -> Tuple[float, float, float]:
        
        # Consider four (4) cases.

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
        
        
        # CASE 2: The agent is fully networked.
        # The agent may be connected with all other agents
        # of the network with a direct link.
        # In this case, the "friend of a friend" and "random_agent" options should be disallowed.
        # We assume that the agent's friends do not influence their opinion of
        # someone they've already met
        # (thus increasing/decreasing the corresponding probability).
        is_fully_networked = next(nx.non_neighbors(graph, source), None) is None
        

        if is_fully_networked:
            # The agent has already formed an opinion about *all* agents
            # He/She will therefore choose to interact with one of his friends in the end,
            # as there are no strangers.

            # We assume that the agent's friends do not influence their opinion of
            # someone they've already met
            # (thus increasing/decreasing the corresponding probability).
            pr_friend += pr_friend_of_friend + pr_random_agent
            pr_friend_of_friend, pr_random_agent = 0, 0
        
        else:

            # CASE 3: There are no positive suggestions from friends.
            # Make the probability of choosing friend of a friend 0%
            # and adjust the remaining probabilities.
            # Analogy;
            # Q: What happens if their social circle does not suggest any other agent
            # A: All the agents belonging in the clique can only either meet their friends or meet new agents.
            
            graph_array: csr_array = nx.to_scipy_sparse_array(graph)
            # Default values return a CSR matrix, with values being 'weight'.

            friends_array: csr_array = graph_array.maximum(0)
            #friend_array.eliminate_zeros() # We have to see if this takes more time than needed.
            friend_suggestions = (friends_array @ graph_array)[[list(graph.nodes).index(source)], :]
            exists_friend_suggestion = friend_suggestions.maximum(0).count_nonzero() > 0

            if not exists_friend_suggestion:
                # The agent has friends and strangers, but no recommendations.

                # Split the probability of meeting *a friend of a friend* by half
                # and distribute it equally between the other two options.
                pr_friend += pr_friend_of_friend/2
                pr_random_agent += pr_friend_of_friend/2
                pr_friend_of_friend = 0


            # else: # has_friend and not is_fully_networked and exists_friend_suggestion
            #     pass
                # All three options:
                # - pick friend
                # - pick friend of a friend
                # - pick random agent
                # are available to the agent.
                # Do not(!) change the probabilities.


        # CASE 4: None of the above cases
        # i.e.
        # 1. Has at least one friend
        # 2. Has at least one recommendation
        # 3. Has at least one stranger
        # Proceed with the provided probabilities.
        return pr_friend, pr_friend_of_friend, pr_random_agent
    

    @staticmethod
    def weighted_erdos_renyi_graph_generator(
        nr_vertices: int,
        pr_edge_creation: float,
        pr_positive: float,
        is_dense: bool = False,
        **kwargs
    ) -> nx.DiGraph:
        
        output_graph = BackboneSocialSystem.weighted_erdos_renyi_graph_generator(nr_vertices, pr_edge_creation, is_dense)

        assert 0 <= pr_positive <= 1,\
            f'`pr_positive` expresses a probability.\
                Instead, it has a value of {pr_positive}.'
        
        # TODO DECIDE & IMPLEMENT:
        # Think up of a link strength initialization scheme.
        # AD-HOC uniformly create link strengths.

        rng: np.random.Generator = np.random.default_rng()
        # Uniformly distributed link strengths.
        rand_weights = rng.integers(link_strength_lb_init, link_strength_ub_init, len(output_graph.edges), endpoint=True)

        output_graph.add_weighted_edges_from(list((*edge, rand_weights[i]) for i, edge in enumerate(output_graph.edges)))
            
        return output_graph