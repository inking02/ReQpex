import numpy as np
import pandas as pd
import generate_maps
import networkx as nx
from utils import create_node_dictionnary

data = pd.read_csv(
    "/Users/lf/Desktop/Université/Session 3/BSQ201/Projet 2/ReQpex/datasets/liste_occupants_simple.csv"
)


def simplify_points(data: pd.DataFrame, radius_km: float, seed: int = 545):

    radius_lng_lat = (
        radius_km / 111.1
    )  # https://www.sco.wisc.edu/2022/01/21/how-big-is-a-degree/#:~:text=Therefore%20we%20can%20easily%20compute,69.4%20miles%20(111.1%20km).
    points = data[["Longitude", "Latitude"]].to_numpy(dtype=float, copy=True)
    nodes = create_node_dictionnary(points)
    G = nx.Graph()

    for label, coord in nodes.items():
        G.add_node(label, pos=coord)

    # Distance entre les positions
    def euclid_dist(pos1, pos2):
        return ((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2) ** 0.5

    # On met une condition pour ajouter des arêtes
    for i in G.nodes():
        for j in G.nodes():
            if (
                i != j
                and euclid_dist(G.nodes[i]["pos"], G.nodes[j]["pos"])
                <= 2 * radius_lng_lat
            ):
                G.add_edge(i, j)
    I = nx.maximal_independent_set(G, seed=seed)
    print(f"Maximum independent set of G: {I}")

    indexes = np.array(I, dtype=int)
    new_points = (data.iloc[indexes])[["Longitude", "Latitude"]].to_numpy(
        dtype=float, copy=True
    )
    generate_maps.generate_map(
        new_points,
        title="Carte simplifiée",
        path="/Users/lf/Desktop/Université/Session 3/BSQ201/Projet 2/ReQpex/",
        file_name="maps/carte_simplifiee.png",
    )
    return indexes


res = simplify_points(
    data,
    0.5,
)
print(len(res))
