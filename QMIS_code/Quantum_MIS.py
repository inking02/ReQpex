"""
File containing the class of the quantum analog computing MIS finder method. The class of this MIS finder and its method are in this class. Some of the useful 
fonction of the class are in the QMIS_utils.py file. The function that runs the main algorithm is the .run method.
"""

import numpy as np
import networkx as nx
from pulser import Register, Sequence
from pulser_simulation import QutipEmulator
from pulser.devices import AnalogDevice
from QMIS_code.QMIS_utils import scale_coordinates, find_minimal_radius, plot_histogram
from typing import Callable


class Quantum_MIS:
    def __init__(self, graph: nx.Graph) -> None:
        """
        Object that can run the quantum analog computing MIS algorithm. To create the object, networkx's graph architecture must be used.
        A graph with more than 15 atom will not give good results.

        Parameters:
        - graph (networkx.Graph): The graph to find an MIS on.

        Returns:
        None
        """
        self.G = graph
        self.pos = nx.spring_layout(
            self.G, seed=42
        )  # Mettre aléatoire et prendre best?
        self.coords = np.array(list(self.pos.values()))
        self.radius = find_minimal_radius(self.G, self.pos)
        self.reg = self.build_reg()

    def build_reg(self) -> Register:
        """
        Function that creates the pulser resgister for a given graph. It is optimal when the number of atoms is less than eleven.

        Parameters:
        - None

        Returns:
        Register: The pulser register of the atoms representating the graph.
        """
        MAX_D = 35
        MIN_D = 5
        scaled_coords, self.radius = scale_coordinates(
            self.radius, self.coords, MIN_D, MAX_D
        )
        reg = Register.from_coordinates(scaled_coords)
        return reg

    def print_reg(self) -> None:
        """
        Function that draws the positionnement and radius of the atoms of the quantum architecture.

        Parameters:
        - None

        Returns:
        None
        """
        self.reg.draw(
            blockade_radius=self.radius, draw_graph=True, draw_half_radius=True
        )

    def run(
        self,
        Pulse: Callable,
        shots: int = 1000,
        generate_histogram: bool = False,
        file_name: str = "QMIS_histo.pdf",
        progress_bar: bool = True,
    ) -> dict:
        """
        Method to run the quantum analog computing MIS algorithm. By using a given pulse, it will find the graph given to the object.

        Parameters:
        - Pulse (Callable): A callable of a function returning a Pulse class objcet from Pulser's library. It is the pulse given to the set of
                            the atoms to run the algorithm.
        - shots (int=1000): The number of times the algotihm must be runned. By default, it is set at 1000.
        - generate_histogram (bool = False): Generate the result histogram of the runs of the algorithms.
        - file_name (str = "QMIS_histo.pdf"): The file name that the histogram must be saved as. The filename must also include its path and use the extension png.
        - progress_bar (bool = True): Whether or not to print the evolution on the run on pulser's architecture.

        Returns:
        dict: The counts dictionnary of the results from the shots of the algorithms.
        """

        Omega_r_b = AnalogDevice.rabi_from_blockade(self.radius)
        Omega_pulse_max = AnalogDevice.channels["rydberg_global"].max_amp
        Omega = min(Omega_pulse_max, Omega_r_b)

        # creating pulse sequence
        seq = Sequence(self.reg, AnalogDevice)
        seq.declare_channel("ising", "rydberg_global")
        seq.add(Pulse(Omega), "ising")

        simul = QutipEmulator.from_sequence(seq)
        results = simul.run(progress_bar=progress_bar)

        count_dict = results.sample_final_state(N_samples=shots)

        if generate_histogram:
            plot_histogram(count_dict, shots, file_name)

        return count_dict
