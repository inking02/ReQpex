import numpy as np
import pandas as pd
import networkx as nx
from Big_QMIS import BIG_QMIS
from utils.utils import create_node_dictionary, disc_graph_to_connected, euclid_dist
from utils.generate_maps import interactive_map
from QMIS_code.pulse_utils import pulse_constructor
from Classical_MIS import MIS_Solver
from numpy.typing import NDArray
from typing import Callable, List


def simplify_bins(
    radius_km: float,
    pulse: Callable = pulse_constructor(4000, "Rise_fall"),
    show_map: bool = False,
    save_map: bool = False,
    path: str = "",
    bin_image: bool = False,
    use_quantum: bool = True,
    num_atoms: int = 6,
) -> None:
    """
    Function to remove some of the bins in their original bins with a MIS. If two locations are closer than the radius given,
    an edge between them is present except if both of their production is greater than 27754. It saves the simplified bins location
    in the dataset folder under the useful_bins.csv file.

    Parameters:
    - radius_km (float): The distance in km of which two locations must be connected if they do not produce enough volume.
    - pulse (Callable = Pulse_constructor(4000, "Rise_fall")): The pulse to be applied on the atoms in the QMIS algorithm.
    - show_map (bool = False): Whether or not to show the maps of the original bins locations and their simplified distribution.
    - save_map (bool = False): Whether or not to save the maps of the original bins locations and their simplified distribution.
    - path (str = ""):The local path to the ReQpex repository.
    - bin_image (bool = False): Whether or not to use Recupex' bins as pings on the map.
    - use_quantum (bool=True): Whether or not to use the  QAA method to find the mis instead of the classic MIS solution.
    - num_atoms (int=6): The maximum number of nodes per graph sent to the QMIS.

    Returns:
    - None
    """
    # Functions that will be used inside this one

    # Using the same function to create the graph but changing the condition of ans edge to check the volumes produced.
    def disc_graph_to_connected_volume(
        positions: NDArray[np.float_], radius
    ) -> nx.Graph:
        """
        From a array of position of the vertices of the graph, create a connected networkx graph. It will give an edge
        between two positions if they are too close without producing too much volume of clothes each.

        Parameters:
        - positions (NDArray[np.float_]): The array of the points of the vertices of the graph.
        - radius (float): The radius of which if two circles generated by the points touch, they have an edge.

        Returns:
        networkx.graph: The networkx graph representing the positions and its radius.
        """
        G = nx.Graph()
        nodes_dict = create_node_dictionary(positions)
        for label, coord in nodes_dict.items():
            G.add_node(label, pos=coord)

        # Condition to add edges
        for i in G.nodes():
            for j in G.nodes():
                if (
                    i != j
                    and euclid_dist(G.nodes[i]["pos"], G.nodes[j]["pos"]) <= radius
                ):
                    if positions[int(i), 2] < 27754 or positions[int(j), 2] < 27754:
                        G.add_edge(i, j)
        return G

    def max_with_volume(
        res_dict: dict, nodes: NDArray[np.int_], other_info: List[float]
    ) -> str:
        """
        From the result dictionary of the QMIS, return the bitstring with the maximal amount of nodes. If there is a tie,
        return the bitstring with the nodes producing the most volume of clothes.

        Parameters:
        - res_dict (dict): The counts dictionary of the result if the QMIS algorithm.
        - index_positions (NDArray[np.int_]): The order of the nodes implied in the bitstring.
        - other_info (List[float]): The volume of clothes produced by each bin.

        Returns:
        - str: The best bitstring with the constrained defined in the description of the function.
        """
        max_numb_ones = 0
        best_volume = 0
        for key in res_dict.keys():
            max_numb_ones = max(max_numb_ones, key.count("1"))
            best_bitstring = key
        for key in res_dict.keys():
            if key.count("1") == max_numb_ones:
                volume = 0
                for k, j in enumerate(key):
                    if j == "1":
                        volume += other_info[int(nodes[k])]
                if best_volume < volume:
                    best_bitstring = key
                    best_volume = volume
        return best_bitstring

    # Start of the main function
    # Loading the file
    bins = pd.read_csv(path + "datasets/bins.csv", sep=";")
    bins_numpy = bins[["Longitude", "Latitude", "Volume"]].to_numpy(
        dtype=float, copy=True
    )

    # Load or save the map
    if show_map or save_map:
        interactive_map(
            bins,
            bin_image=bin_image,
            path=path,
            show_map=show_map,
            save_map=save_map,
            file_name="original_bins",
        )

    # Building the graph
    radius_lng_lat = radius_km / 111.1
    G = disc_graph_to_connected_volume(positions=bins_numpy, radius=radius_lng_lat)

    if use_quantum:
        # Running the QMIS
        solver = BIG_QMIS(G, num_atoms=num_atoms)
        new_vertices = solver.run(
            pulse,
            best_bitstring_getter=max_with_volume,
            other_info=bins_numpy[:, 2],
            print_progression=True,
        )

    else:
        solver = MIS_Solver(G)
        res_dict = solver.run(shots=100)
        nodes = [int(i) for i in G.nodes()]
        best_bitstring = max_with_volume(res_dict, nodes, bins_numpy[:, 2])
        new_vertices = []
        for i, val in enumerate(best_bitstring):
            if val == "1":
                new_vertices.append(str(i))

    # Writing the result on a new file
    new_vertices_int = [int(i) for i in new_vertices]
    new_points = bins_numpy[new_vertices_int, :]

    # Printing the function results
    original_size = np.shape(bins_numpy)[0]
    new_size = np.shape(new_points)[0]
    print()
    print("Sizes")
    print("OG size: ", original_size)
    print("New size: ", new_size)
    print("Bins removed: ", original_size - new_size)

    new_dataframe = bins.iloc[new_vertices_int]
    new_dataframe.to_csv(path + "datasets/useful_bins.csv", index=False)

    # Show or save the map
    if show_map or save_map:
        interactive_map(
            new_dataframe,
            bin_image=bin_image,
            path=path,
            show_map=show_map,
            save_map=save_map,
            file_name="useful_original_bins",
        )


def remove_possibles_new_locations(
    radius_km: float,
    show_map: bool = False,
    save_map: bool = False,
    path: str = "",
    bin_image: bool = False,
) -> None:
    """
    Function that removes the possible locations that are too close from Estrie-Aide and Recupex simplified bins. The constrained
    new locations are saved under the useful_locations.csv file in the dataset directory.

    Parameters:
    - radius_km (float): The distance of which if a possible location is within that range of an bin or an Estrie-Aide bin,
                         it must be removed form the possibilities.
    - show_map (bool = False): Whether or not to show the maps of the simplified possible locations.
    - save_map (bool = False): Whether or not to save the maps of the simplified possible locations.
    - path (str = ""):The local path to the ReQpex repository.
    - bin_image (bool = False): Whether or not to use Recupex' bins as pings on the map.

    Returns:
    - None
    """
    # Loading the files
    bins = pd.read_csv(path + "datasets/useful_bins.csv", sep=",")
    bins_numpy = bins[["Longitude", "Latitude"]].to_numpy(dtype=float, copy=True)

    new_locations = pd.read_csv(path + "datasets/possible_locations.csv", sep=";")
    new_locations_numpy = new_locations[["Longitude", "Latitude"]].to_numpy(
        dtype=float, copy=True
    )

    estrie_aide = pd.read_csv(path + "datasets/estrieaide.csv", sep=",")
    estrie_aide_numpy = estrie_aide[["Longitude", "Latitude"]].to_numpy(
        dtype=float, copy=True
    )

    radius_lng_lat = radius_km / 111.1
    # Simplify the locations
    list_of_indexes = []

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

    # Writing a new file with the results
    useful_locations = new_locations.iloc[list_of_indexes]
    useful_locations_numpy = new_locations_numpy[list_of_indexes, :]

    useful_locations.to_csv(path + "datasets/useful_locations.csv", index=False)

    # Show or save the map
    if show_map or save_map:
        interactive_map(
            useful_locations,
            bin_image=bin_image,
            path=path,
            show_map=show_map,
            save_map=save_map,
            file_name="simplified_possible_locations",
        )
    # Printing the results of the simplification
    original_size = np.shape(new_locations_numpy)[0]
    new_size = np.shape(useful_locations_numpy)[0]
    print()
    print("Sizes")
    print("OG size: ", original_size)
    print("New size: ", new_size)
    print("Locations removed: ", original_size - new_size)


def place_new_bins(
    radius_km: float,
    pulse=pulse_constructor(4000, "Rise_fall"),
    show_map: bool = False,
    save_map: bool = False,
    path: str = "",
    bin_image: bool = False,
    use_quantum: bool = True,
    num_atoms: int = 4,
    test: bool = True,
) -> None:
    """
    Function to place new bins so that the bins have an optimal distribution on the map. The resulting bins will be saved in
    the new_bins.csv in the results folder.

    Parameters:
    - radius_km (float): The distance in km of which two locations must be connected if they do not produce enough volume.
    - pulse (Callable = Pulse_constructor(4000, "Rise_fall")): The pulse to be applied on the atoms in the QMIS algorithm.
    - show_map (bool = False): Whether or not to show the map of the new optimal distribution of the Recupex bins.
    - save_map (bool = False): Whether or not to save the map of the new optimal distribution of the Recupex bins.
    - path (str = ""):The local path to the ReQpex repository.
    - bin_image (bool = False): Whether or not to use Recupex' bins as pings on the map.
    - use_quantum (bool=True): Whether or not to use the  QAA mthod to find the mis instead of the classic MIS solution.
    - num_atoms (int=4): The maximum number of nodes per graph sent to the QMIS.

    Returns:
    - None
    """
    # LOad the files
    bins = pd.read_csv(path + "datasets/useful_bins.csv", sep=",")
    bins_numpy = bins[["Longitude", "Latitude"]].to_numpy(dtype=float, copy=True)
    bins_names = bins[["Nom de la borne", "Addresse", "Rue"]].to_numpy(
        dtype=str, copy=True
    )

    locations = pd.read_csv(path + "datasets/useful_locations.csv", sep=",")
    locations_numpy = locations[["Longitude", "Latitude"]].to_numpy(
        dtype=float, copy=True
    )
    locations_names = locations[["Nom de la borne", "Addresse", "Rue"]].to_numpy(
        dtype=str, copy=True
    )

    # Building the graph
    radius_lng_lat = (
        radius_km / 111.1
    )  # https://www.sco.wisc.edu/2022/01/21/how-big-is-a-degree/#:~:text=Therefore%20we%20can%20easily%20compute,69.4%20miles%20(111.1%20km).

    G = disc_graph_to_connected(positions=locations_numpy, radius=radius_lng_lat)

    if use_quantum:
        # Running the QMIS algorithm
        solver = BIG_QMIS(G, num_atoms=num_atoms)
        new_vertices = solver.run(pulse, print_progression=True)

    else:
        solver = MIS_Solver(G)
        res_dict = solver.run(shots=100)
        best_bitstring = max(zip(res_dict.values(), res_dict.keys()))[1]
        new_vertices = []
        for i, val in enumerate(best_bitstring):
            if val == "1":
                new_vertices.append(str(i))

    # Formatting the results into the new file
    new_vertices_int = [int(i) for i in new_vertices]

    new_locations_numpy = locations_numpy[new_vertices_int]
    new_bins_numpy = np.empty(
        (
            np.shape(bins_numpy)[0] + np.shape(new_locations_numpy)[0],
            np.shape(bins_numpy)[1],
        )
    )
    new_bins_numpy[: np.shape(bins_numpy)[0], :] = bins_numpy
    new_bins_numpy[np.shape(bins_numpy)[0] :, :] = new_locations_numpy

    new_bins_location = pd.DataFrame(
        new_bins_numpy,
        columns=["Longitude", "Latitude"],
    )
    new_bins_location["Nom de la borne"] = np.append(
        bins_names[:, 0], locations_names[new_vertices_int, 0]
    )

    new_bins_location["Addresse"] = np.append(
        bins_names[:, 1], locations_names[new_vertices_int, 1]
    )
    new_bins_location["Rue"] = np.append(
        bins_names[:, 2], locations_names[new_vertices_int, 2]
    )

    new_bins_location.to_csv(path + "results/new_bins.csv", index=False)

    # Show or save the map
    if show_map or save_map:
        interactive_map(
            new_bins_location,
            bin_image=bin_image,
            path=path,
            show_map=show_map,
            save_map=save_map,
            file_name="new_distribution_of_bins",
        )
    # Printing the results of the function
    new_number_of_bins = np.shape(new_bins_numpy)[0]
    print()
    print("New number of bins: ", new_number_of_bins)
