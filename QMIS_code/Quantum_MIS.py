import numpy as np
import networkx as nx
from pulser import Register, Sequence
from pulser_simulation import QutipEmulator
from QMIS_utils import plot_histogram, find_minimal_radius, scale_coordinates, create_sub_graph, fusion_counts
from typing import Callable


class Quantum_MIS:
    def __init__(self, graph:nx.graph, device)-> None:
        self.device = device
        
        #we separate the graph in all its connected components
        self.sub_graphes = []
        self.nodes_positions = []
        for nodes in nx.connected_components(graph):
            self.sub_graphes.append(create_sub_graph(graph, nodes))
            self.nodes_positions.append(list(nodes))

        
        #finding coordinates that helps building a good register using spring_layout
        self.pos = [nx.spring_layout(sub_graph, k = 0.1, seed = 42) for sub_graph in self.sub_graphes]
        self.coords = [np.array(list(position.values())) for position in self.pos]

        #defining the minimal blockade for each sub_graph
        self.R_blockades = [find_minimal_radius(sub_graph, position) for sub_graph, position in zip(self.sub_graphes, self.pos)]

        #building each registers
        self.regs = [self.__build_reg__(coord, i) for i, coord in enumerate(self.coords)]

    def __build_reg__(self, coord, i)-> Register:
        min_dist = self.device.min_atom_distance
        max_dist = self.device.max_radial_distance
        coord, self.R_blockades[i] = scale_coordinates(self.R_blockades[i], coord, min_dist, max_dist)
        reg = Register.from_coordinates(coord)
        return reg



    def print_regs(self)-> None:
        for reg, R_blockade in zip(self.regs, self.R_blockades):
            reg.draw(blockade_radius=R_blockade, 
            draw_graph=True,
            draw_half_radius=True)
        

    def run(self, Pulse: Callable, shots: int = 1000, generate_histogram: bool = False, file_name: str = "QMIS_histo.pdf", progress_bar: bool = True):
        
        #defining the omega for each pulse
        Omega_pulse_max = self.device.channels['rydberg_global'].max_amp
        Omegas = [min(Omega_pulse_max, R_blockade) for R_blockade in self.R_blockades]

        #creating pulse sequence
        seqs = [Sequence(reg, self.device) for reg in self.regs]
        count_dicts = []
        for seq, Omega in zip(seqs, Omegas):
            seq.declare_channel("ising", "rydberg_global") #the pulse is applied to all the register globally 
            seq.add(Pulse(Omega), "ising")

            #simulating the results
            simul = QutipEmulator.from_sequence(seq)
            results = simul.run(progress_bar= progress_bar)

            #extracting the count_dict for each register
            count_dict = results.sample_final_state(N_samples=shots)
            count_dicts.append(count_dict)
            # if generate_histogram:
            #     plot_histogram(count_dict, shots, file_name)

        #combining the registers
        count_total = fusion_counts(count_dicts, self.nodes_positions)
        print(count_total)
        if generate_histogram:
            plot_histogram(count_total, shots, file_name)

        return count_total

