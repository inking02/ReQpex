import numpy as np
import networkx as nx
from pulser import Register, Sequence, Pulse
from pulser_simulation import QutipEmulator
from pulser.devices import DigitalAnalogDevice
from typing import Callable
from scipy.spatial.distance import pdist, squareform
from pulser.waveforms import RampWaveform, CompositeWaveform
import matplotlib.pyplot as plt


class Quantum_QAOA:
    """
    A class to implement the Quantum Approximate Optimization Algorithm (QAOA) 
    using the Pulser framework for solving problems encoded as Ising Hamiltonians.
    """

    def __init__(self, graph: nx.graph, layers: int = 2) -> None:
        """
        Initialize the QAOA class with the given graph and number of layers.
        
        Parameters:
        - graph: The input graph representing the problem.
        - layers: The number of QAOA layers to use.
        """
        # Generate spring-layout coordinates for the graph
        pos = nx.spring_layout(graph, k=0.1, seed=42)
        self.coords = np.array(list(pos.values()))  # Extract node positions
        self.reg = self.__build_reg__()  # Create a Pulser register
        self.layers = layers  # Set the number of QAOA layers

        # Compute interaction matrix and determine blockade radius
        int_matrix = squareform(DigitalAnalogDevice.interaction_coeff / pdist(self.coords) ** 6)
        Omega_max = np.median(int_matrix[int_matrix > 0].flatten())  # Median of interaction coefficients
        self.R_blockade = DigitalAnalogDevice.rydberg_blockade_radius(Omega_max)  # Blockade radius

        
    def __build_reg__(self) -> Register:
        """
        Builds a register of qubits for Pulser using the graph's node positions.
        
        Returns:
        - A Pulser Register object with qubits coordinates.
        """
        # Normalize coordinates to ensure qubits are appropriately spaced
        val = np.min(pdist(self.coords))  # Find the minimum distance between nodes
        self.coords *= 5 / val  # Scale coordinates
        reg = Register.from_coordinates(self.coords)  # Create a Pulser register
        return reg
    
    
    def print_reg(self) -> None:
        """
        Visualize the qubit register along with the blockade radius and graph structure.
        """
        self.reg.draw(blockade_radius=self.R_blockade, 
                      draw_graph=True,
                      draw_half_radius=True)
        

    def create_qaoa_sequence(self):
        """
        Create a QAOA sequence for Pulser, including the necessary pulses and measurements.

        Returns:
        - A Pulser Sequence object representing the QAOA process.
        """
        # Initialize a Pulser sequence
        seq = Sequence(self.reg, DigitalAnalogDevice)
        seq.declare_channel("ising", "rydberg_global")  # Declare a global Rydberg channel

        # Declare parameters for mixer and cost Hamiltonians
        t_list = seq.declare_variable("t_list", size=self.layers)  # Mixer times
        s_list = seq.declare_variable("s_list", size=self.layers)  # Cost times

        # Compute Rabi frequencies and detuning limits
        Omega_r_b = DigitalAnalogDevice.rabi_from_blockade(self.R_blockade)
        Omega_pulse_max = DigitalAnalogDevice.channels['rydberg_global'].max_amp
        Omega = min(Omega_r_b, Omega_pulse_max)  # Use the minimum safe Rabi frequency

        # Define a helper function for rise-fall waveforms
        def Rise_Fall_Waveform(Omega, T, delta_0, delta_f):
            """
            Create a pulse with a rise-and-fall waveform and detuning.

            Parameters:
            - Omega: Maximum Rabi frequency.
            - T: Duration of the pulse.
            - delta_0: Initial detuning.
            - delta_f: Final detuning.

            Returns:
            - A Pulser Pulse object.
            """
            up = RampWaveform(T / 2, 0, Omega)  # Ramp up
            down = RampWaveform(T / 2, Omega, 0)  # Ramp down
            d_up = RampWaveform(T / 2, delta_0, 0)  # Detuning ramp up
            d_down = RampWaveform(T / 2, 0, delta_f)  # Detuning ramp down
            rise_fall_Pulse = Pulse(
                CompositeWaveform(up, down),  # Composite Rabi waveform
                CompositeWaveform(d_up, d_down),  # Composite detuning waveform
                0  # Phase
            )
            return rise_fall_Pulse

        # Add mixer and cost Hamiltonian pulses for each layer
        for t, s in zip(t_list, s_list):
            T_mixer = np.ceil(t * 1500 / 4) * 4  # Mixer pulse duration
            T_cost = np.ceil(s * 1500 / 4) * 4  # Cost pulse duration

            # Create mixer and cost pulses
            Pulse_mixer = Rise_Fall_Waveform(Omega, T_mixer, -5, 5)
            Pulse_cost = Rise_Fall_Waveform(Omega, T_cost, -5, 5)

            # Add pulses to the sequence
            seq.add(Pulse_mixer, "ising")
            seq.add(Pulse_cost, "ising")

        # Measure the final state
        seq.measure("ground-rydberg")
        return seq


    def quantum_loop(self, parameters):
        """
        Execute a quantum loop of QAOA with the given parameters.

        Parameters:
        - parameters: A flattened array of QAOA parameters (t_list and s_list).

        Returns:
        - A dictionary of bitstring counts from the simulation.
        """
        # Reshape parameters into t_list and s_list
        t_params, s_params = np.reshape(parameters, (2, self.layers))
        
        # Build the QAOA sequence with the parameters
        seq = self.create_qaoa_sequence()
        assigned_seq = seq.build(t_list=t_params, s_list=s_params)

        # Simulate the sequence using QutipEmulator
        simul = QutipEmulator.from_sequence(assigned_seq, sampling_rate=0.01)
        results = simul.run()

        # Get the bitstring counts
        count_dict = results.sample_final_state()
        return count_dict
    

    def plot_histogram(self, count_dict, shots: int, file_name: str):
        """
        Plot a histogram of the bitstring counts from the simulation results.
    
        Parameters:
        - count_dict: A dictionary where keys are bitstrings and values are their respective counts.
        - shots: The total number of measurement shots performed in the simulation.
        - file_name: The name of the file where the histogram will be saved.
        """
        
        # Define target bitstrings to highlight in the histogram
        target_bitstrings = ["010101", "100101"]

        # Filter bitstrings with counts greater than 2% of the total shots
        # This ensures that only significant results are included in the histogram
        most_freq = {k: v for k, v in count_dict.items() if v > 0.02 * shots}
        
        # Assign colors: red for target bitstrings, blue for all other bitstrings
        colors = ['red' if bitstring in target_bitstrings else 'blue' for bitstring in most_freq.keys()]

        # Plot the histogram using a bar chart
        plt.bar(list(most_freq.keys()), list(most_freq.values()), color=colors)

        # Rotate the x-axis labels vertically for better readability
        plt.xticks(rotation="vertical")

        # Label the axes
        plt.ylabel('Counts')  # Y-axis shows the number of times a bitstring appears
        plt.xlabel('Bitstrings')  # X-axis shows the bitstrings measured

        # Save the plot to the specified file
        plt.savefig(file_name)

        # Display the plot
        plt.show()


    def run(self, shots: int = 1000, generate_histogram: bool = False, file_name: str = "QAOA_histo.pdf"):
        """
        Run the QAOA algorithm and optionally generate a histogram of results.

        Parameters:
        - shots: Number of measurement shots.
        - generate_histogram: Whether to generate a histogram of the results.
        - file_name: The name of the file to save the histogram.

        Returns:
        - A dictionary of bitstring counts from the simulation.
        """
        # Generate initial random guesses for t and s parameters
        np.random.seed(123)
        guess_t = np.random.uniform(8, 10, self.layers)
        guess_s = np.random.uniform(1, 3, self.layers)
        params = np.r_[guess_t, guess_s]  # Concatenate guesses into one array

        # Run the quantum loop with the guessed parameters
        result = self.quantum_loop(params)

        # Generate and save the histogram
        if generate_histogram:
            self.plot_histogram(result, shots, file_name)

        return result
