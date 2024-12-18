"""
File containing the class of the quantum analog computing MIS finder utilities functions. They are all listed in the QAA in the QAA.py file.
"""

from scipy.spatial import distance_matrix
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from itertools import product
from numpy.typing import NDArray
from typing import Tuple, List, Callable
from scipy.optimize import minimize


def scale_coordinates(
    radius: float,
    coordinates: NDArray[np.float_],
    min_distance: float,
    max_distance: float,
) -> Tuple[NDArray[np.float_], float]:
    """
    Function that scale the coordinates of a networkx graph that was layed-out to transform them into coordinates
    that can be used by a pulser's register.

    Parameters:
    - radius (float): The radius that determines the connection between the points.
    - coordinates (NDArray[np.float_]): The coordinates of the vertices of the graph that was layed-out.
    - min_distance (float): The minimum distance that must be between the points.
    - max_distance (float): The maximum distance that must be between the points.

    Returns:
    - Tuple[NDArray[np.float_], float]:   - The scaled coordinates of the vertices.
                                        - The scaled radius.
    """
    # Calculate the distances between the points to guess the scale.
    dist_matrix = distance_matrix(coordinates, coordinates)
    np.fill_diagonal(dist_matrix, np.inf)
    min_dist = dist_matrix.min()  # Minimal distance between the original points

    # Calculation of the scale factor to make sure that the smallest distance is at least `min_distance`
    scale_factor = min_distance / min_dist

    # Apply the scale factor
    scaled_coords = coordinates * scale_factor
    scaled_radius = radius * scale_factor

    # Center the coordinates so that they are close to the origin
    center_x = np.mean(scaled_coords[:, 0])
    center_y = np.mean(scaled_coords[:, 1])
    scaled_coords -= np.array([center_x, center_y])

    # Adjust the coordinates again of the maximum distance exceeds `max_distance`
    max_dist_from_center = np.max(np.linalg.norm(scaled_coords, axis=1))
    if max_dist_from_center > max_distance:
        scale_factor = max_distance / max_dist_from_center
        scaled_coords *= scale_factor
        scaled_radius *= scale_factor

    return scaled_coords, scaled_radius


def find_minimal_radius(G: nx.Graph, pos: NDArray[np.float_]) -> float:
    """
    Finds the minimal distance between two connected vertices of a layed-out graph.

    Parameters:
    - G (networkx.Graph): A networkx graph.
    - pos (NDArray[np.float_]): The coordinates of the vertices of the graph that was layed-out.

    Returns:
    - float: The minimal distance between two connected vertices.
    """
    max_distance = 0

    for u, v in G.edges():
        coord_u = np.array(pos[u])
        coord_v = np.array(pos[v])

        distance = euclid_dist(coord_u, coord_v)

        if distance > max_distance:
            max_distance = distance

    return max_distance


def plot_histogram(count_dict: dict, shots: int, file_name: str = "") -> None:
    """
    Saves and prints the histogram of the result of the runs of the algorithm.

    Parameters:
    - count_dict (dict): The counts dictionary of the results of the QMIS algorithm.
    - shots (int): The number of shots used in the algorithm.
    - file_name (str=""): The name to save the figure onto. It must include its path and the png extension. If the value is "", the figure will not be saved.

    Returns:
    - None
    """
    most_freq = {k: v for k, v in count_dict.items() if v > 0.02 * shots}
    plt.bar(list(most_freq.keys()), list(most_freq.values()))
    plt.xticks(rotation="vertical")
    plt.ylabel("counts")
    plt.xlabel("bitstrings")
    if file_name != "":
        plt.savefig(file_name)
    plt.show()


def euclid_dist(pos1: NDArray[np.float_], pos2: NDArray[np.float_]) -> float:
    """
    Calculates the euclidean distance between to points in a 2D plane.

    Parameters:
    - pos1 (NDArray[np.float_]): The coordinates of the first point in the 2D plane.
    - pos2 (NDArray[np.float_]): The coordinates of the second point in the 2D plane.

    Returns:
    - float: The euclidean distance between the points
    """
    return ((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2) ** 0.5


def create_sub_graph(G, nodes: List[str]) -> nx.Graph:
    """
    Method to create a subgraph of the class' main graph with the networkx.Graph architecture.

    Parameters:
    - nodes (List[str]): The list of the nodes that need to be included in the subgraph

    Returns:
    - networkx.Graph: The sub graph create with the nodes specified.
    """
    subgraph = nx.Graph()
    subgraph.add_nodes_from(nodes)
    subgraph.add_edges_from(
        tuple([u, v]) for (u, v) in G.edges(nodes) if u in nodes and v in nodes
    )
    return subgraph


def fusion_counts(counts, positions):
    total_counts = {}

    # creating all bitstring combinations from the given dictionaries
    bitstring_combinations = product(*[d.items() for d in counts])
    max_length = 0
    for dictionary in counts:
        # final bitstring length
        max_length += len(next(iter(dictionary)))
    for combination in bitstring_combinations:

        # the final bitstring is init with the value 0 everywhere
        final_bitstring = ["0"] * max_length

        combined_value = 0
        # place each bit in its given position with the combin   ed value
        for position, (bitstring, value) in zip(positions, combination):
            for bit, pos in zip(bitstring, position):
                final_bitstring[pos] = bit
            # calculate the combined value of the combined bitstrings
            combined_value += value

        # convert to the final bitstring
        combined_key = "".join(final_bitstring)
        total_counts[combined_key] = combined_value

    return total_counts


def base_minimizer(
    cost_function: Callable, params: NDArray[np.float_]
) -> NDArray[np.float_]:
    """
    Basic COBYLA optimizer that can be used in the QAOA class. It returns the optimized parameters.

     Parameters:
     - cost_function (Callable): The function to optimize.
     - params (NDArray[np.float_]): The original value of parameters of the function.

     Returns:
     - NDArray[np.float_]:  The optimized parameters that minimizes the cost function.
    """
    return minimize(cost_function, params, method="COBYLA").x
