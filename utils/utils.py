import numpy as np
from numpy.typing import NDArray
import networkx as nx
import pandas as pd


def create_node_dictionnary(points: NDArray):
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


def simplify_points_map(
    data: pd.DataFrame, radius_km: float, seed: int = 545
) -> pd.DataFrame:

    radius_lng_lat = (
        radius_km / 111.1
    )  # https://www.sco.wisc.edu/2022/01/21/how-big-is-a-degree/#:~:text=Therefore%20we%20can%20easily%20compute,69.4%20miles%20(111.1%20km).
    points = data[["Longitude", "Latitude"]].to_numpy(dtype=float, copy=True)

    G = disc_graph_to_connected(points, radius_lng_lat)
    I = nx.maximal_independent_set(G, seed=seed)
    print(f"Maximum independent set of G: {I}")

    indexes = np.array(I, dtype=int)
    new_points = (data.iloc[indexes])[["Longitude", "Latitude"]].to_numpy(
        dtype=float, copy=True
    )
    return new_points
