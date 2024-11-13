from numpy.typing import NDArray
import numpy as np
from pulser import Register, Pulse, Sequence
from pulser_simulation import QutipEmulator
from pulser.devices import AnalogDevice
from pulser.waveforms import InterpolatedWaveform
from QMIS_utils import scale_coordinates, find_unit_disk_radius, plot_histogram


class Quantum_MIS:
    def __init__(self, coords: NDArray, radius: float = None)-> None:
        
        self.coords = coords
        if radius == None:
            self.radius = find_unit_disk_radius(self.coords)
        else:
            self.radius = radius
        self.reg = self.build_reg()

    
    def build_reg(self):
        MAX_D = 35  
        MIN_D = 5 
        
        # Scale coordinates based on min and max distances
        scaled_coords, self.radius = scale_coordinates(self.radius, self.coords, MIN_D, MAX_D)
        reg = Register.from_coordinates(scaled_coords, prefix='N')
      
        return reg

    def print_reg(self):
        
        self.reg.draw(blockade_radius=self.radius, 
        draw_graph=True,
        draw_half_radius=True)
        

        
    def run(self, shots: int = 1000, generate_histogram: bool = False, file_name: str = "QMIS_histo.pdf"):
        
        Omega_r_b = AnalogDevice.rabi_from_blockade(self.radius)
        Omega_pulse_max = AnalogDevice.channels['rydberg_global'].max_amp

        Omega = min(Omega_pulse_max, Omega_r_b)

        delta_0 = -5 #Valeur de désaccord souhaité au début du pulse
        delta_f = 5 #Valeur de désaccord souhaité à la fin du pulse
        T = 4_000 #temps en nanosecondes 

        #Forme du pulse
        adiabatic_pulse = Pulse(
            InterpolatedWaveform(T, [1e-9, Omega, 1e-9]),
            InterpolatedWaveform(T, [delta_0, 0, delta_f]),
            0,
        )

        #creating pulse sequence
        seq = Sequence(self.reg, AnalogDevice) #La séquence doit contenir un registre ainsi que la machine choisie 
        seq.declare_channel("ising", "rydberg_global") #On choisit un pulse global. 
        seq.add(adiabatic_pulse, "ising")
        
        simul = QutipEmulator.from_sequence(seq)
        results = simul.run(progress_bar= True)

        count_dict = results.sample_final_state(N_samples=shots)

        if generate_histogram:
            plot_histogram(count_dict, shots, file_name)

        return count_dict


