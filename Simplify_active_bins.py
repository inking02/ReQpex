import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from Big_QMIS import BIG_QMIS
from utils.utils import create_node_dictionnary
from utils.generate_maps import (
    interactive_map,
)
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

    def max_with_volume(res_dict, nodes, other_info):
        max_numb_ones = 0
        best_volume = 0
        for key in res_dict.keys():
            max_numb_ones = max(max_numb_ones, key.count("1"))
            best_bitstring = key
        for i, key in enumerate(res_dict.keys()):
            if key.count("1") == max_numb_ones:
                volume = 0
                for k, j in enumerate(key):
                    if j == "1":
                        volume += other_info[int(nodes[k])]
                if best_volume < volume:
                    best_bitstring = key
                    best_volume = volume
        return best_bitstring

    G = disc_graph_to_connected(positions=points, radius=radius_lng_lat)
    if generate_graphs:
        interactive_map(data)

    solver = BIG_QMIS(G, num_atoms=6)
    new_sommets = solver.run(
        pulse,
        best_bitstring_getter=max_with_volume,
        other_info=points[:, 2],
        print_progression=True,
    )

    new_sommets_int = [int(i) for i in new_sommets]
    new_points = points[new_sommets_int, :]

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

    if generate_graphs:
        interactive_map(new_dataframe)


def remove_possibles_new_locations(radius_km, generate_graphs: bool = False):
    bins = pd.read_csv(
        "/Users/lf/Documents/GitHub/ReQpex/datasets/cloches_utiles.csv", sep=","
    )
    bins_numpy = bins[["Longitude", "Latitude"]].to_numpy(dtype=float, copy=True)

    new_locations = pd.read_csv(
        "/Users/lf/Documents/GitHub/ReQpex/datasets/liste_occupants_simple.csv", sep=";"
    )
    new_locations_numpy = new_locations[["Longitude", "Latitude"]].to_numpy(
        dtype=float, copy=True
    )

    estrie_aide = pd.read_csv(
        "/Users/lf/Documents/GitHub/ReQpex/datasets/estrieaide.csv", sep=","
    )

    estrie_aide_numpy = estrie_aide[["Longitude", "Latitude"]].to_numpy(
        dtype=float, copy=True
    )

    if generate_graphs:
        interactive_map(new_locations)

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
            for estrie in estrie_aide_numpy:
                if euclid_dist(location, estrie) < radius_lng_lat:
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
        interactive_map(useful_locations)
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
    bins_names = bins[["Nom de la borne", "Addresse", "Rue"]].to_numpy(
        dtype=str, copy=True
    )

    locations = pd.read_csv(
        "/Users/lf/Documents/GitHub/ReQpex/datasets/useful_locations.csv", sep=","
    )
    locations_numpy = locations[["Longitude", "Latitude"]].to_numpy(
        dtype=float, copy=True
    )
    locations_names = locations[["Nom de la borne", "Addresse", "Rue"]].to_numpy(
        dtype=str, copy=True
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

    new_locations_numpy = locations_numpy[new_sommets_int]
    new_bins_numpy = np.empty(
        (
            np.shape(bins_numpy)[0] + np.shape(new_locations_numpy)[0],
            np.shape(bins_numpy)[1],
        )
    )
    new_bins_numpy[: np.shape(bins_numpy)[0], :] = bins_numpy
    new_bins_numpy[np.shape(bins_numpy)[0] :, :] = new_locations_numpy

    new_number_of_bins = np.shape(new_bins_numpy)[0]
    print()
    print("New number of bins: ", new_number_of_bins)

    new_bins_location = pd.DataFrame(
        new_bins_numpy,
        columns=["Longitude", "Latitude"],
    )
    new_bins_location["Nom de la borne"] = np.append(
        bins_names[:, 0], locations_names[new_sommets_int, 0]
    )

    new_bins_location["Addresse"] = np.append(
        bins_names[:, 1], locations_names[new_sommets_int, 1]
    )
    new_bins_location["Rue"] = np.append(
        bins_names[:, 2], locations_names[new_sommets_int, 2]
    )

    new_bins_location.to_csv(
        "/Users/lf/Documents/GitHub/ReQpex/datasets/nouvelles_cloches.csv", index=False
    )
    if generate_graphs:
        interactive_map(new_bins_location)


simplify_bins(0.75, generate_graphs=False)
print("Bins simplified")
print("******************************************")

remove_possibles_new_locations(1.5, generate_graphs=False)
print("Possible locations simplified")
print("******************************************")

place_new_bins(1.4, generate_graphs=True)
print("New distribution calculated")
print("******************************************")
