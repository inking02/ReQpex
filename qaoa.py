import numpy as np
from pulser import Pulse, Sequence
from pulser.devices import DigitalAnalogDevice
from pulser.waveforms import InterpolatedWaveform
from pulser_simulation import QutipEmulator
from pulser.register import Register
from utilss import scale_coordinates, find_unit_disk_radius, plot_histogram

class Quantum_QAOA:
    def __init__(self, coords: np.ndarray, edges: list, layers: int, radius: float = None):
        self.coords = coords
        self.layers = layers
        self.radius = radius if radius else find_unit_disk_radius(self.coords)
        self.reg = self.build_register()



    def build_register(self):
        scaled_coords, self.radius = scale_coordinates(self.radius, self.coords, 5, 35)
        reg = Register.from_coordinates(scaled_coords, prefix='N')
        return reg

    def print_reg(self):
        self.reg.draw(blockade_radius=self.radius, 
        draw_graph=True,
        draw_half_radius=True)


    def create_qaoa_sequence(self):
        seq = Sequence(self.reg, DigitalAnalogDevice)
        seq.declare_channel("rydberg", "rydberg_global")
        
        t_list = seq.declare_variable("t_list", size=self.layers)
        s_list = seq.declare_variable("s_list", size=self.layers)
        
        #Omega = DigitalAnalogDevice.rabi_from_blockade(self.radius)
        Omega_r_b = DigitalAnalogDevice.rabi_from_blockade(self.radius)
        Omega_pulse_max = DigitalAnalogDevice.channels['rydberg_global'].max_amp
        Omega = min(Omega_r_b, Omega_pulse_max)

        delta_0 = -5
        delta_f = 5
        #T = 4000
        
        for t, s in zip(t_list, s_list):
            T_mixer = np.ceil(t * 1000 / 4) * 4  # Utiliser t pour paramétrer la durée du pulse 
            T_cost = np.ceil(s * 1000 / 4) * 4
            mixer_pulse = Pulse(
                InterpolatedWaveform(T_mixer, [1e-9, Omega, 1e-9]),
                InterpolatedWaveform(T_mixer, [delta_0, 0, delta_f]),
                0)
            
            cost_pulse = Pulse(
                InterpolatedWaveform(T_cost, [1e-9, Omega, 1e-9]),
                InterpolatedWaveform(T_cost, [delta_0, 0, delta_f]),
                0)
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

    def run(self, shots: int = 1000, generate_histogram: bool = False, file_name: str = "QAOA_histo.pdf"):
        np.random.seed(123)
        guess_t = np.random.uniform(8, 10, self.layers)
        guess_s = np.random.uniform(1, 3, self.layers)
        params = np.r_[guess_t, guess_s]
        result = self.quantum_loop(params)
        if generate_histogram:
            plot_histogram(result, shots, file_name)

        return result