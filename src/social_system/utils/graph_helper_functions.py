from copy import deepcopy

from numpy import inf, NINF
from numpy import typing as npt

import networkx as nx

def reduce_to_graph_with_edges_in_range(graph: nx.Graph, low: int = NINF, high: int = inf):
    """
    Returns a graph induced by edges with a `weight` within the provided range (inclusive).
    The induced subgraph contains each edge in edges and all nodes of the input graph.
    This may result in nodes which were initially connected to be isolated.

    NOTE: Edges which do not have a `weight` attribute are automatically included.

    NOTE: Changes are *inplace*

    Parameters
    ----------
    low : int, optional
        Lower boundary in which an edge is valid for the subgraph.
        In the case where argument `high` is left as `None`,
        this argument acts as a `high` instead and `low` has the value
        of 0 ; by default 1
    high : int, optional
        Higher boundary in which an edge is valid for the subgraph; by default 1

    Returns (inplace)
    -------
    nx.DiGraph
        The edge-induced subgraph of input graph with the same edge attributes.

    """

    edges_to_remove = []
    for u, v, weight in graph.edges(data='weight'):
        if not (weight is None or low <= weight <= high):
            edges_to_remove.append((u, v))

    graph.remove_edges_from(edges_to_remove)


def increase_weights_of_edges(graph: nx.Graph, link_increments: npt.NDArray):
    """
    Increases the `weight` attribute of the edges of the graph
    by the amount given in the `link_incredecrements` argument.

    NOTE: In case that an edge corresponding to an entry of `link_incredecements`
    does not exist, that edge is created.

    NOTE: The changes to the graph are done *inplace*! 

    Parameters
    ----------
    graph : nx.Graph
        Graph whose links' `weight` attribute will be changed.
    link_incredecrements : npt.NDArray
        shape: (?, 3);
        Each row is [s, d, v] where
        - s: source vertex,
        - d: destination vertex and
        - v: value by which the edge will be increased/decreased
        respectively
    """

    for row in link_increments:
        s, d, v = row[:3]

        if graph.has_edge(s, d):
            graph.edges[s, d]['weight'] += v    
        else: 
            graph.add_edge(s, d, weight=v)