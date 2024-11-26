import numpy as np
import networkx as nx
from pulser import Register, Sequence
from pulser_simulation import QutipEmulator
from pulser.devices import DigitalAnalogDevice
from QMIS_utils import plot_histogram
from typing import  Callable
from scipy.spatial.distance import pdist, squareform


class Quantum_MIS:
    def __init__(self, graph:nx.graph)-> None:
        pos = nx.spring_layout(graph, k = 0.1, seed = 42)
        self.coords = np.array(list(pos.values()))
        self.reg = self.__build_reg__()

        int_matrix = squareform(DigitalAnalogDevice.interaction_coeff / pdist(self.coords) ** 6)
        Omega_max = np.median(int_matrix[int_matrix > 0].flatten())
        self.R_blockade = DigitalAnalogDevice.rydberg_blockade_radius(Omega_max)

    def __build_reg__(self)-> Register:
        val = np.min(pdist(self.coords))
        self.coords *= 5/val
        reg = Register.from_coordinates(self.coords)
        return reg


    def print_reg(self)-> None:
        self.reg.draw(blockade_radius=self.R_blockade, 
        draw_graph=True,
        draw_half_radius=True)
        

    def run(self, Pulse: Callable, shots: int = 1000, generate_histogram: bool = False, file_name: str = "QMIS_histo.pdf", progress_bar: bool = True):
        
        Omega_pulse_max = DigitalAnalogDevice.channels['rydberg_global'].max_amp
        Omega = min(Omega_pulse_max, self.R_blockade)

        #creating pulse sequence
        seq = Sequence(self.reg, DigitalAnalogDevice) #La séquence doit contenir un registre ainsi que la machine choisie 
        seq.declare_channel("ising", "rydberg_global") #On choisit un pulse global. 
        seq.add(Pulse(Omega), "ising")
        
        simul = QutipEmulator.from_sequence(seq)
        results = simul.run(progress_bar= progress_bar)

        count_dict = results.sample_final_state(N_samples=shots)

        if generate_histogram:
            plot_histogram(count_dict, shots, file_name)

        return count_dict


