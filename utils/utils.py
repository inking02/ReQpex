import numpy as np
from numpy.typing import NDArray
import networkx as nx


def create_node_dictionnary(points):
    res = dict()
    longitudes = points[:, 0]
    latitudes = points[:, 1]
    for i in range(np.shape(points)[0]):
        res[str(i)] = (longitudes[i], latitudes[i])
    return res


def disc_graph_to_connected(positions: NDArray[np.float_], radius) -> nx.Graph:
    G = nx.Graph()
    nodes_dict = create_node_dictionnary(positions)
    for label, coord in nodes_dict.items():
        G.add_node(label, pos=coord)

    # Distance between two positions
    def euclid_dist(pos1, pos2):
        return ((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2) ** 0.5

    # Condition to add edges
    for i in G.nodes():
        for j in G.nodes():
            if (
                i != j
                and euclid_dist(G.nodes[i]["pos"], G.nodes[j]["pos"]) <= 2 * radius
            ):
                G.add_edge(i, j)
    return G
