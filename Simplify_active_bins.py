import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from Big_QMIS import BIG_QMIS
from utils.utils import create_node_dictionnary
from utils.generate_maps import generate_map, generate_town_graph_connected
from numpy.typing import NDArray
from QMIS_code.QMIS_utils import Pulse_constructor
import sys

sys.setrecursionlimit(2000)


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

    G = disc_graph_to_connected(positions=points, radius=radius_lng_lat)
    if generate_graphs:
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
        "/Users/lf/Documents/GitHub/ReQpex/datasets/cloches_utiles.csv", index=False
    )


def remove_possibles_new_locations(radius_km, generate_graphs: bool = False):
    bins = pd.read_csv(
        "/Users/lf/Documents/GitHub/ReQpex/datasets/cloches_utiles.csv", sep=","
    )
    bins_numpy = bins[["Longitude", "Latitude"]].to_numpy(dtype=float, copy=True)

    new_locations = pd.read_csv(
        "/Users/lf/Documents/GitHub/ReQpex/datasets/liste_occupants_simple.csv", sep=","
    )
    new_locations_numpy = new_locations[["Longitude", "Latitude"]].to_numpy(
        dtype=float, copy=True
    )

    if generate_graphs:
        plt.clf()
        generate_map(
            new_locations_numpy,
            path="/Users/lf/Documents/GitHub/ReQpex/",
            file_name="figures/endroits_possibles",
        )

    radius_lng_lat = (
        radius_km / 111.1
    )  # https://www.sco.wisc.edu/2022/01/21/how-big-is-a-degree/#:~:text=Therefore%20we%20can%20easily%20compute,69.4%20miles%20(111.1%20km).

    list_of_indexes = []

    def euclid_dist(pos1, pos2):
        return ((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2) ** 0.5

    for i, location in enumerate(new_locations_numpy):
        to_add = True
        for bin in bins_numpy:
            if euclid_dist(location, bin) < radius_lng_lat:
                to_add = False
                break
        if to_add:
            list_of_indexes.append(i)

    useful_locations = new_locations.iloc[list_of_indexes]
    valid_locations = new_locations_numpy[list_of_indexes, :]

    useful_locations.to_csv(
        "/Users/lf/Documents/GitHub/ReQpex/datasets/useful_locations.csv", index=False
    )

    if generate_graphs:
        plt.clf()
        generate_map(
            valid_locations,
            path="/Users/lf/Documents/GitHub/ReQpex/",
            file_name="figures/nouveaux_endroits_possibles",
        )
    original_size = np.shape(new_locations_numpy)[0]
    new_size = np.shape(valid_locations)[0]
    print()
    print("Sizes")
    print("OG size: ", original_size)
    print("New size: ", new_size)
    print("Locations removed: ", original_size - new_size)


def place_new_bins(
    radius_km: float,
    pulse=Pulse_constructor(4000, "Rise_fall"),
    generate_graphs: bool = False,
):
    bins = pd.read_csv(
        "/Users/lf/Documents/GitHub/ReQpex/datasets/cloches_utiles.csv", sep=","
    )
    bins_numpy = bins[["Longitude", "Latitude"]].to_numpy(dtype=float, copy=True)

    locations = pd.read_csv(
        "/Users/lf/Documents/GitHub/ReQpex/datasets/useful_locations.csv", sep=","
    )
    locations_numpy = locations[["Longitude", "Latitude"]].to_numpy(
        dtype=float, copy=True
    )

    radius_lng_lat = (
        radius_km / 111.1
    )  # https://www.sco.wisc.edu/2022/01/21/how-big-is-a-degree/#:~:text=Therefore%20we%20can%20easily%20compute,69.4%20miles%20(111.1%20km).

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

    G = disc_graph_to_connected(positions=locations_numpy, radius=radius_lng_lat)

    solver = BIG_QMIS(G, num_atoms=4)
    new_sommets = solver.run(pulse, print_progression=True)

    new_sommets_int = [int(i) for i in new_sommets]
    new_locations_numpy = locations_numpy[new_sommets_int, :]
    new_bins_numpy = np.empty(
        (
            np.shape(bins_numpy)[0] + np.shape(new_locations_numpy)[0],
            np.shape(bins_numpy)[1],
        )
    )
    new_bins_numpy[: np.shape(bins_numpy)[0], :] = bins_numpy
    new_bins_numpy[np.shape(bins_numpy)[0] :, :] = new_locations_numpy

    if generate_graphs:
        plt.clf()
        generate_map(
            new_bins_numpy,
            path="/Users/lf/Documents/GitHub/ReQpex/",
            file_name="figures/new_disposition",
        )

    new_number_of_bins = np.shape(new_bins_numpy)[0]
    print()
    print("New number of bins: ", new_number_of_bins)

    new_bins_location = pd.DataFrame(
        new_bins_numpy,
        columns=["Longitude", "Latitude"],
    )

    new_bins_location.to_csv(
        "/Users/lf/Documents/GitHub/ReQpex/datasets/nouvelles_cloches.csv", index=False
    )


simplify_bins(0.5, generate_graphs=True)
print("Bins simplified")
print("******************************************")

remove_possibles_new_locations(1.5, generate_graphs=True)
print("Possible locations simplified")
print("******************************************")

place_new_bins(1, generate_graphs=True)
print("New distribution calculated")
print("******************************************")
