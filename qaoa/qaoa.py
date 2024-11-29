import numpy as np
import networkx as nx
from pulser import Register, Sequence, Pulse
from pulser_simulation import QutipEmulator
from pulser.devices import DigitalAnalogDevice
from utilss import plot_histogram
from typing import  Callable
from scipy.spatial.distance import pdist, squareform
from pulser.waveforms import InterpolatedWaveform, RampWaveform, ConstantWaveform, CompositeWaveform, BlackmanWaveform


class Quantum_QAOA:
    def __init__(self, graph:nx.graph, layers: int=2)-> None:
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
        seq.declare_channel("ising", "rydberg_global")

        t_list = seq.declare_variable("t_list", size=self.layers)
        s_list = seq.declare_variable("s_list", size=self.layers)

        Omega_r_b = DigitalAnalogDevice.rabi_from_blockade(self.R_blockade)
        Omega_pulse_max = DigitalAnalogDevice.channels['rydberg_global'].max_amp
        Omega = min(Omega_r_b, Omega_pulse_max)

        
        def rise_sweep_fall(Omega, T):
            rise = RampWaveform(T/4, 0, Omega)
            sweep = ConstantWaveform(T/2, Omega)
            fall = RampWaveform(T/4, Omega, 0)
            Omega_Wave = CompositeWaveform(rise, sweep, fall)
            constant1_d = ConstantWaveform(T/4, -Omega)
            rise_d = RampWaveform(T/2, -Omega, Omega)
            constant2_d = ConstantWaveform(T/4, Omega)
            detuning = CompositeWaveform(constant1_d, rise_d, constant2_d)
            return Pulse(Omega_Wave, detuning, 0)

        for t, s in zip(t_list, s_list):

            T_mixer = np.ceil(t * 1000 / 4) * 4
            T_cost = np.ceil(s * 1000 / 4) * 4

            ###Pulse_mixer = Pulse.ConstantPulse(T_mixer, Omega, 0.0, 0)
            ###Pulse_cost = Pulse.ConstantPulse(T_cost, 0.0, Omega, 0)

            Pulse_mixer = rise_sweep_fall(Omega, T_mixer)
            Pulse_cost = rise_sweep_fall(Omega, T_cost)
            
            
            seq.add(Pulse_mixer, "ising")
            seq.add(Pulse_cost, "ising")

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
    
    def run(self, shots: int = 1000, generate_histogram: bool = False, file_name: str = "QAOA_histo.pdf"):
        np.random.seed(123)
        guess_t = np.random.uniform(8, 10, self.layers)
        guess_s = np.random.uniform(1, 3, self.layers)
        params = np.r_[guess_t, guess_s]
        result = self.quantum_loop(params)
        if generate_histogram:
            plot_histogram(result, shots, file_name)
        return result
