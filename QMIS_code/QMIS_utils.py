"""
File containing the class of the quantum analog computing MIS finder utilities functions. They are all ised in the Quantum_MIS in the Quantum_MIS.py file.
"""

from scipy.spatial import distance_matrix
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from pulser.waveforms import (
    InterpolatedWaveform,
    RampWaveform,
    ConstantWaveform,
    CompositeWaveform,
    BlackmanWaveform,
)
from pulser import Pulse
from numpy.typing import NDArray
from typing import Tuple


def scale_coordinates(
    radius: float,
    coordinates: NDArray[np.float_],
    min_distance: float,
    max_distance: float,
) -> Tuple[NDArray[np.float_], float]:
    """
    Function that scale the coordinates of a netwrokx graph that was layed-out to transfom them into coordinates
    that can be used by a pulser's register.

    Parameters:
    - radius (float): The radius that determines the connection between the points.
    - coordinates (NDArray[np.float_]): The coordinates of the verticies of the graph that was layed-out.
    - min_distance (float): The minimum distance that must be between the points.
    - max_distance (float): The maximum distance that must be between the points.

    Returns:
    Tuple[NDArray[np.float_], float]:   - The scaled cooridnates of the verticies.
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

    # Center the cooridnates so that they are close to the origin
    center_x = np.mean(scaled_coords[:, 0])
    center_y = np.mean(scaled_coords[:, 1])
    scaled_coords -= np.array([center_x, center_y])

    # Ajust the cooridnates again of the maximum distance exceeds `max_distance`
    max_dist_from_center = np.max(np.linalg.norm(scaled_coords, axis=1))
    if max_dist_from_center > max_distance:
        scale_factor = max_distance / max_dist_from_center
        scaled_coords *= scale_factor
        scaled_radius *= scale_factor

    return scaled_coords, scaled_radius


def find_minimal_radius(G: nx.Graph, pos: NDArray[np.float_]) -> float:
    """
    Finds the minimal distance between two connected verticies of a layed-out graph.

    Parameters:
    - G (netwokx.Graph): A networkx graph.
    - pos (NDArray[np.float_]): The coordinates of the verticies of the graph that was layed-out.

    Returns:
    float: The minimal distance between two connected verticies.
    """
    max_distance = 0

    for u, v in G.edges():
        coord_u = np.array(pos[u])
        coord_v = np.array(pos[v])

        distance = euclid_dist(coord_u, coord_v)

        if distance > max_distance:
            max_distance = distance

    return max_distance


def plot_histogram(count_dict: dict, shots: int, file_name: str = "histo.png") -> None:
    """
    Saves and prints the histogram of the result of the runs of the algorithm.

    Parameters:
    - count_dict (dict): The counts dictionnary of the results of the QMIS algorithm.
    - shots (int): The number of shots used in the algorithm.
    - file_name (str="histo.png"): The name to save the figure onto. It must include its path and the png extension

    Returns:
    None
    """
    most_freq = {k: v for k, v in count_dict.items() if v > 0.02 * shots}
    plt.bar(list(most_freq.keys()), list(most_freq.values()))
    plt.xticks(rotation="vertical")
    plt.ylabel("counts")
    plt.xlabel("bitstrings")
    plt.savefig(file_name)
    plt.show()


def euclid_dist(pos1: NDArray[np.float_], pos2: NDArray[np.float_]) -> float:
    """
    Calculates the euclidian distance between to points in a 2D plane.

    Parameters:
    - pos1 (NDArray[np.float_]): The coordinates of the first point in the 2D plane.
    - pos2 (NDArray[np.float_]): The coordinates of the second point in the 2D plane.

    Returns:
    float: The euclidian distance between the points
    """
    return ((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2) ** 0.5


def Waveform_Pulse(Omega: float, T: float, delta_0: float, delta_f: float) -> Pulse:
    """
    Creates a waveform pulse object.

    Parameters:
    - Omega (float):
    - T (float):
    - delta_0 (float):
    - delta_f (float):

    Returns:
    Pulse: A pulser pusle object.
    """
    adiabatic_pulse = Pulse(
        InterpolatedWaveform(T, [1e-9, Omega, 1e-9]),
        InterpolatedWaveform(T, [delta_0, 0, delta_f]),
        0,
    )
    return adiabatic_pulse


def Rise_Fall_Waveform(Omega: float, T: float, delta_0: float, delta_f: float):
    """
    Creates a waveform pulse object.

    Parameters:
    - Omega (float):
    - T (float):
    - delta_0 (float):
    - delta_f (float):

    Returns:
    Pulse: A pulser pusle object.
    """
    up = RampWaveform(T / 2, 0, Omega)
    down = RampWaveform(T / 2, Omega, 0)
    d_up = RampWaveform(T / 2, delta_0, 0)
    d_down = RampWaveform(T / 2, 0, delta_f)

    rise_fall_Pulse = Pulse(
        CompositeWaveform(up, down), CompositeWaveform(d_up, d_down), 0
    )

    return rise_fall_Pulse


def Blackman_Waveform_Pulse(Omega: float, T: float, delta_0: float, delta_f: float):
    """
    Creates a waveform pulse object.

    Parameters:
    - Omega (float):
    - T (float):
    - delta_0 (float):
    - delta_f (float):

    Returns:
    Pulse: A pulser pusle object.
    """
    Blackman_Pulse = Pulse(
        BlackmanWaveform(T, Omega), InterpolatedWaveform(T, [delta_0, 0, delta_f]), 0
    )

    return Blackman_Pulse


def Constant_pulse_pyramide(
    Omega: float,
    T: float,
    T_pyramide: float,
    delta_0: float,
    delta_f: float,
    delta: float,
):
    """
    Creates a waveform pulse object.

    Parameters:
    - Omega (float):
    - T (float):
    - T_pyramide (float):
    - delta_0 (float):
    - delta_f (float):

    Returns:
    Pulse: A pulser pusle object.
    """

    Constant_1 = ConstantWaveform((T - T_pyramide) / 2, Omega - delta)
    up = RampWaveform(T_pyramide / 2, Omega - delta, Omega)
    down = RampWaveform(T_pyramide / 2, Omega, Omega - delta)
    Constant_2 = ConstantWaveform((T - T_pyramide) / 2, Omega - delta)

    r_Pulse = Pulse(
        CompositeWaveform(Constant_1, up, down, Constant_2),
        InterpolatedWaveform(T, [delta_0, 0, delta_f]),
        0,
    )
    return r_Pulse


def Pulse_constructor(
    T: float,
    Pulse_type: str,
    T_pyramide: float = 0,
    delta: float = 0,
    delta_0: float = -5,
    delta_f: float = 5,
):
    """
    Creates a waveform pulse object.

    Parameters:
    - T (float):
    - Pulse_type (str):
    - T_pyramide (float):
    - delta_0 (float):
    - delta_f (float):

    Returns:
    Pulse: A pulser pusle object.
    """
    if Pulse_type == "Waveform":
        return lambda Omega: Waveform_Pulse(Omega, T, delta_0, delta_f)

    if Pulse_type == "Rise_fall":
        return lambda Omega: Rise_Fall_Waveform(Omega, T, delta_0, delta_f)

    if Pulse_type == "Blackman":
        return lambda Omega: Blackman_Waveform_Pulse(Omega, T, delta_0, delta_f)

    if Pulse_type == "Pyramide":
        return lambda Omega: Constant_pulse_pyramide(
            Omega, T, T_pyramide, delta_0, delta_f, delta
        )
