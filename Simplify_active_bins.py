import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from Big_QMIS import BIG_QMIS
from utils.utils import create_node_dictionnary
from utils.generate_maps import generate_map, generate_town_graph_connected
from numpy.typing import NDArray
from QMIS_code.QMIS_utils import Pulse_constructor


def simplify_bins(
    radius_km, pulse=Pulse_constructor(4000, "Rise_fall"), generate_graphs: bool = False
):
    data = pd.read_csv(
        "/Users/lf/Documents/GitHub/ReQpex/datasets/cloches.csv", sep=";"
    )

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

    if generate_graphs:
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

    solver = BIG_QMIS(G, num_atoms=6)
    new_sommets = solver.run(pulse, print_progression=True)
    print(new_sommets)

    new_sommets_int = [int(i) for i in new_sommets]
    new_points = points[new_sommets_int, :]

    if generate_graphs:
        plt.clf()
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

    new_dataframe = data.iloc[new_sommets_int]
    new_dataframe.to_csv(
        "/Users/lf/Documents/GitHub/ReQpex/datasets/cloches_utiles.csv", index=True
    )
