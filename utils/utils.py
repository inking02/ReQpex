"""
File containing utility functions used in the repository
"""

import numpy as np
from numpy.typing import NDArray
import networkx as nx


def euclid_dist(pos1, pos2):
    """
    Calculates the euclidian distance between to points in a 2D plane.

    Parameters:
    - pos1 (NDArray[np.float_]): The coordinates of the first point in the 2D plane.
    - pos2 (NDArray[np.float_]): The coordinates of the second point in the 2D plane.

    Returns:
    float: The euclidian distance between the points
    """
    return ((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2) ** 0.5


def create_node_dictionnary(points: NDArray[np.float_]) -> dict:
    """
    Creates a dictionnary with the index of the position and its position on the map.

    Parameters:
    - points (NDArray[np.float_]): The array of the points to transform into the dictionary.

    Returns:
    dict: The dictionary of the positions.
    """
    res = dict()
    longitudes = points[:, 0]
    latitudes = points[:, 1]
    for i in range(np.shape(points)[0]):
        res[str(i)] = (longitudes[i], latitudes[i])
    return res


def disc_graph_to_connected(positions: NDArray[np.float_], radius: float) -> nx.Graph:
    """
    From a array of position of the vertices of the graph, create a connected networkx graph

    Parameters:
    - points (NDArray[np.float_]): The array of the points of the verticies of the graph.
    - radius (float): The radius of which if two circles generated by the points touch, they have an edge.

    Returns:
    netwrokx.graph: The networkx graph represntating the positions and its radius.
    """
    G = nx.Graph()
    nodes_dict = create_node_dictionnary(positions)
    for label, coord in nodes_dict.items():
        G.add_node(label, pos=coord)

    # Condition to add edges
    for i in G.nodes():
        for j in G.nodes():
            if i != j and euclid_dist(G.nodes[i]["pos"], G.nodes[j]["pos"]) <= radius:
                G.add_edge(i, j)
    return G
