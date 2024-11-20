import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from Big_QMIS import BIG_QMIS
from utils.utils import create_node_dictionnary
from utils.generate_maps import generate_map, generate_town_graph_connected
from numpy.typing import NDArray
from QMIS_code.QMIS_utils import Pulse_constructor

data = pd.read_csv("/Users/lf/Documents/GitHub/ReQpex/datasets/cloches.csv", sep=";")

radius_km = 0.5

radius_lng_lat = (
    radius_km / 111.1
)  # https://www.sco.wisc.edu/2022/01/21/how-big-is-a-degree/#:~:text=Therefore%20we%20can%20easily%20compute,69.4%20miles%20(111.1%20km).
points = data[["Longitude", "Latitude", "Volume"]].to_numpy(dtype=float, copy=True)


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
                and (positions[int(i), 2] < 27754 or positions[int(j), 2] < 27754)
            ):
                G.add_edge(i, j)
    return G


"""
G = nx.Graph()  # Création d'un graphe sans sommet et sans arête
nodes = [0, 1, 2, 3, 4]
nodes = [str(i) for i in nodes]
edges = [(0, 1), (0, 2), (0, 4), (1, 2), (1, 4), (1, 3), (2, 4), (2, 3)]
edges = [tuple([str(j), str(k)]) for (j, k) in edges]
G.add_nodes_from(nodes)  # On ajoute 3 sommets
G.add_edges_from(edges)
nx.draw(G, with_labels=True)
"""
G = disc_graph_to_connected(positions=points, radius=radius_lng_lat)
generate_map(
    points,
    path="/Users/lf/Documents/GitHub/ReQpex/",
    file_name="figures/Ville_originale",
)
plt.clf()
generate_town_graph_connected(
    points,
    radius=radius_lng_lat,
    path="/Users/lf/Documents/GitHub/ReQpex/",
    file_name="figures/Ville_originale_connected",
)
plt.clf()


pulse = Pulse_constructor(4000, "Rise_fall")


solver = BIG_QMIS(G, num_atoms=6)
new_sommets = solver.run(pulse, print_progression=True)
print(new_sommets)


def create_sub_graph(G, nodes):
    # Trouver pourquoi la fonction subgraph marche pas
    subgraph = nx.Graph()
    subgraph.add_nodes_from(nodes)
    subgraph.add_edges_from(
        tuple([u, v]) for (u, v) in G.edges(nodes) if u in nodes and v in nodes
    )
    return subgraph


new_sommets_int = [int(i) for i in new_sommets]
new_points = points[new_sommets_int, :]

generate_map(
    new_points,
    path="/Users/lf/Documents/GitHub/ReQpex/",
    file_name="figures/Ville_simple",
)
plt.clf()
generate_town_graph_connected(
    new_points,
    radius=radius_lng_lat,
    path="/Users/lf/Documents/GitHub/ReQpex/",
    file_name="figures/Ville_simple_connected",
)

original_size = np.shape(points)[0]
new_size = np.shape(new_points)[0]
print()
print("Sizes")
print("OG size: ", original_size)
print("New size: ", new_size)
print("Bins removed: ", original_size - new_size)
