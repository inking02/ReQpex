import numpy as np
import networkx as nx
from pulser import Register, Sequence
from pulser_simulation import QutipEmulator
from pulser.devices import DigitalAnalogDevice
from pulser.waveforms import InterpolatedWaveform
from utilss import plot_histogram, Pulse_constructor
from typing import  Callable
from scipy.spatial.distance import pdist, squareform


class Quantum_QAOA:
    def __init__(self, graph:nx.graph, layers: int=3)-> None:
        pos = nx.spring_layout(graph, k = 0.1, seed = 42)
        self.coords = np.array(list(pos.values()))
        self.reg = self.__build_reg__()
        self.layers = layers

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


    def create_qaoa_sequence(self):
        seq = Sequence(self.reg, DigitalAnalogDevice)
        seq.declare_channel("rydberg", "rydberg_global")

        t_list = seq.declare_variable("t_list", size=self.layers)
        s_list = seq.declare_variable("s_list", size=self.layers)

        Omega_r_b = DigitalAnalogDevice.rabi_from_blockade(self.R_blockade)
        Omega_pulse_max = DigitalAnalogDevice.channels['rydberg_global'].max_amp
        Omega = min(Omega_r_b, Omega_pulse_max)

        #delta_0, delta_f = -5, 5
        generate_pulse = Pulse_constructor(T=None, Pulse_type="waveform", delta_0=-5, delta_f=5)

        for t, s in zip(t_list, s_list):
            T_mixer = np.ceil(t * 1000 / 4) * 4
            T_cost = np.ceil(s * 1000 / 4) * 4
            
            mixer_pulse = generate_pulse(T_mixer, Omega)
                #InterpolatedWaveform(T_mixer, [1e-9, Omega, 1e-9]),
                #InterpolatedWaveform(T_mixer, [delta_0, 0, delta_f]),
                #0)
            
            cost_pulse = generate_pulse(T_cost, Omega)
                #InterpolatedWaveform(T_cost, [1e-9, Omega, 1e-9]),
                #InterpolatedWaveform(T_cost, [delta_0, 0, delta_f]),
                #0)
            
            seq.add(mixer_pulse, "rydberg")
            seq.add(cost_pulse, "rydberg")

        seq.measure("ground-rydberg")
        return seq
    
    def quantum_loop(self, parameters):
        t_params, s_params = np.reshape(parameters, (2, self.layers))
        seq = self.create_qaoa_sequence()
        assigned_seq = seq.build(t_list=t_params, s_list=s_params)
        simul = QutipEmulator.from_sequence(assigned_seq, sampling_rate=0.01)
        results = simul.run()
        count_dict = results.sample_final_state()
        return count_dict
    
    def run(self, Pulse: Callable, shots: int = 1000, generate_histogram: bool = False, file_name: str = "QAOA_histo.pdf"):
        np.random.seed(123)
        guess_t = np.random.uniform(8, 10, self.layers)
        guess_s = np.random.uniform(1, 3, self.layers)
        params = np.r_[guess_t, guess_s]
        result = self.quantum_loop(params)
        if generate_histogram:
            plot_histogram(result, shots, file_name)
        return result
