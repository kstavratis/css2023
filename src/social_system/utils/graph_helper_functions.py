from copy import deepcopy

from numpy import inf, NINF

import networkx as nx

def get_subgraph_with_edges_in_range(graph: nx.Graph, low: int = NINF, high: int = inf) -> nx.Graph:
        """
        Returns the subgraph induced by edges with a `weight` within the provided range (inclusive).
        The induced subgraph contains each edge in edges and all nodes of the input graph.
        This may result in nodes which were initially connected to be isolated.

        NOTE: Edges which do not have a `weight` attribute are automatically included.

        Parameters
        ----------
        low : int, optional
            Lower boundary in which an edge is valid for the subgraph.
            In the case where argument `high` is left as `None`,
            this argument acts as a `high` instead and `low` has the value
            of 0 ; by default 1
        high : int, optional
            Higher boundary in which an edge is valid for the subgraph; by default 1

        Returns
        -------
        nx.DiGraph
            An edge-induced subgraph of this graph with the same edge attributes.

        """

        output_graph = deepcopy(graph)

        edges_to_remove = []
        for u, v, weight in graph.edges(data='weight'):
            if not (weight is None or low <= weight <= high):
                edges_to_remove.append((u, v))

        graph.remove_edges_from(edges_to_remove)
        return graph