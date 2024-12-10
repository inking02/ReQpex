"""
File containing the class of the quantum analog computing MIS finder method. The class of this MIS finder and its method are in this class. Some of the useful 
function of the class are in the QAA_utils.py file. The function that runs the main algorithm is the .run method.
"""

import numpy as np
import networkx as nx
from pulser import Register, Sequence
from pulser_simulation import QutipEmulator
from pulser.devices import DigitalAnalogDevice
from QMIS_code.QAA_utils import (
    scale_coordinates,
    find_minimal_radius,
    plot_histogram,
    create_sub_graph,
    fusion_counts,
)
from typing import Callable


class Quantum_MIS:
    def __init__(self, graph: nx.Graph, device=DigitalAnalogDevice) -> None:
        """
        Object that can run the quantum analog computing MIS algorithm. To create the object, networkx's graph architecture must be used.
        A graph with more than 15 atom will not give good results.

        Parameters:
        - graph (networkx.Graph): The graph to find an MIS on.

        Returns:
        None
        """
        self.device = device
        
        # we separate the graph in all its connected components
        self.sub_graphes = []
        self.nodes_positions = []
        for nodes in nx.connected_components(graph):
            sub_graph = create_sub_graph(graph, nodes)
            self.sub_graphes.append(sub_graph)
            nodes_to_add = [int(node) for node in nodes]
            self.nodes_positions.append(nodes_to_add)

        # finding coordinates that helps building a good register using spring_layout
        self.pos = [
            nx.spring_layout(sub_graph, k=0.1, seed=42)
            for sub_graph in self.sub_graphes
        ]
        self.coords = [np.array(list(position.values())) for position in self.pos]
        
        # defining the minimal blockade for each sub_graph
        self.R_blockades = [
            find_minimal_radius(sub_graph, position)
            for sub_graph, position in zip(self.sub_graphes, self.pos)
        ]

        # building each registers
        self.regs = [
            self.__build_reg__(coord, i) for i, coord in enumerate(self.coords)
        ]

    def __build_reg__(self, coord, i) -> Register:
        """
        Function that creates the pulser register for a given graph. It is optimal when the number of atoms is less than eleven.

        Parameters:
        - None

        Returns:
        Register: The pulser register of the atoms representing the graph.
        """
        min_dist = self.device.min_atom_distance
        max_dist = self.device.max_radial_distance
        coord, self.R_blockades[i] = scale_coordinates(
            self.R_blockades[i], coord, min_dist, max_dist
        )
        qubits = dict(enumerate(coord))
        reg = Register(qubits)

        return reg

    def print_regs(self) -> None:
        """
        Function that draws the positions and radius of the atoms of the quantum architecture.

        Parameters:
        - None

        Returns:
        None
        """
        for i, (reg, R_blockade) in enumerate(zip(self.regs, self.R_blockades)):
            if len(self.nodes_positions[i]) > 1:
                reg.draw(
                    blockade_radius=R_blockade, draw_graph=True, draw_half_radius=True
                )
            else:
                reg.draw()

    def run(
        self,
        Pulse: Callable,
        shots: int = 1000,
        generate_histogram: bool = False,
        file_name: str = "",
        progress_bar: bool = True,
    ) -> dict:
        """
        Method to run the quantum analog computing MIS algorithm. By using a given pulse, it will find the graph given to the object.

        Parameters:
        - Pulse (Callable): A callable of a function returning a Pulse class object from Pulser's library. It is the pulse given to the set of
                            the atoms to run the algorithm.
        - shots (int=1000): The number of times the algorithm must be run. By default, it is set at 1000.
        - generate_histogram (bool = False): Generate the result histogram of the runs of the algorithms.
        - file_name (str = ""): The file name that the histogram must be saved as. The filename must also include its path and use the extension png.
        - progress_bar (bool = True): Whether or not to print the evolution on the run on pulser's architecture.

        Returns:
        dict: The counts dictionary of the results from the shots of the algorithms.
        """
        # defining the omega for each pulse
        Omega_pulse_max = self.device.channels["rydberg_global"].max_amp
        Omegas = [min(Omega_pulse_max, R_blockade) for R_blockade in self.R_blockades]

        # creating pulse sequence
        seqs = [Sequence(reg, self.device) for reg in self.regs]
        count_dicts = []
        for i, (seq, Omega) in enumerate(zip(seqs, Omegas)):
            
            if len(self.nodes_positions[i]) > 1:
                seq.declare_channel(
                    "ising", "rydberg_global"
                )  # the pulse is applied to all the register globally
                seq.add(Pulse(Omega), "ising")
                
                # simulating the results
                simul = QutipEmulator.from_sequence(seq)
                results = simul.run(progress_bar=progress_bar)

                # extracting the count_dict for each register
                count_dict = results.sample_final_state(N_samples=shots)
                count_dicts.append(count_dict)

            else:
                count_dicts.append({"1": shots})

        # combining the registers
        count_total = fusion_counts(count_dicts, self.nodes_positions)

        if generate_histogram:
            plot_histogram(count_total, shots, file_name)
        return count_total
