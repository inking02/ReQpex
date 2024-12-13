import numpy as np
import networkx as nx
from pulser import Register, Sequence
from pulser_simulation import QutipEmulator
from pulser.devices import DigitalAnalogDevice
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from QMIS_code.pulse_utils import Rise_Fall_Waveform
from numpy.typing import NDArray
from QMIS_code.pulse_utils import Rise_Fall_Waveform
from numpy.typing import NDArray
from QMIS_code.QMIS_utils import plot_histogram, base_minimizer
from typing import Callable


class Quantum_QAOA:
    """
    A class to implement the Quantum Approximate Optimization Algorithm (QAOA)
    using the Pulser framework for solving problems encoded as Ising Hamiltonian.
    """

    def __init__(self, graph: nx.Graph, layers: int = 2) -> None:
        """
        Initialize the QAOA class with the given graph and number of layers.

        Parameters:
        - self: The Quantum_QAOA object to create.
        - graph (networkx.Graph): The input graph representing the problem.
        - layers (int = 2): The number of QAOA layers to use.

        Returns:
        - None
        """
        # Generate spring-layout coordinates for the graph
        self.graph = nx.convert_node_labels_to_integers(graph, first_label=0)
        pos = nx.spring_layout(graph, k=0.1, seed=42)
        self.coords = np.array(list(pos.values()))  # Extract node positions
        self.reg = self.__build_reg__()  # Create a Pulser register
        self.layers = layers  # Set the number of QAOA layers

        # Compute interaction matrix and determine blockade radius
        int_matrix = squareform(
            DigitalAnalogDevice.interaction_coeff / pdist(self.coords) ** 6
        )
        Omega_max = np.median(
            int_matrix[int_matrix > 0].flatten()
        )  # Median of interaction coefficients
        self.R_blockade = DigitalAnalogDevice.rydberg_blockade_radius(
            Omega_max
        )  # Blockade radius

    def __build_reg__(self) -> Register:
        """
        Builds a register of qubits for Pulser using the graph's node positions.

        Parameters:
        - self: The Quantum_QAOA object to use.

        Returns:
        - Register: A Pulser Register object with qubit coordinates.
        """
        # Normalize coordinates to ensure qubits are appropriately spaced
        val = np.min(pdist(self.coords))  # Find the minimum distance between nodes
        self.coords *= 5 / val  # Scale coordinates
        reg = Register.from_coordinates(self.coords)  # Create a Pulser register
        return reg

    def print_reg(self) -> None:
        """
        Visualize the qubit register along with the blockade radius and graph structure.

        Parameters:
        - self: The Quantum_QAOA object to use.

        Returns:
        - None
        """
        self.reg.draw(
            blockade_radius=self.R_blockade, draw_graph=True, draw_half_radius=True
        )

    def create_qaoa_sequence(self) -> Sequence:
        """
        Create a QAOA sequence for Pulser, including the necessary pulses and measurements.

        Parameters:
        - self: The Quantum_QAOA object to use.

        Returns:
        - Sequence: A Pulser Sequence object representing the QAOA process.
        """
        # Initialize a Pulser sequence
        seq = Sequence(self.reg, DigitalAnalogDevice)
        seq.declare_channel(
            "ising", "rydberg_global"
        )  # Declare a global Rydberg channel

        # Declare parameters for mixer and cost Hamiltonians
        t_list = seq.declare_variable("t_list", size=self.layers)  # Mixer times
        s_list = seq.declare_variable("s_list", size=self.layers)  # Cost times

        # Compute Rabi frequencies and detuning limits
        Omega_r_b = DigitalAnalogDevice.rabi_from_blockade(self.R_blockade)
        Omega_pulse_max = DigitalAnalogDevice.channels["rydberg_global"].max_amp
        Omega = min(Omega_r_b, Omega_pulse_max)  # Use the minimum safe Rabi frequency

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

    def quantum_loop(self, parameters: NDArray[np.float_]) -> dict:
        """
        Execute a quantum loop of QAOA with the given parameters.

        Parameters:
        - self: The Quantum_QAOA object to use.
        - parameters: A flattened array of QAOA parameters (t_list and s_list).

        Returns:
        - dict: A dictionary of bitstring counts from the simulation.
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

    def compute_hamiltonian(self, graph: nx.Graph) -> NDArray[np.float_]:
        """
            Compute the Hamiltonian matrix for the given graph based on the Ising model ( for the Max-Cut problem ).

        For Max-Cut:
        - The goal is to partition the graph's nodes into two sets such that
          the number of edges between the two sets (the "cut size") is maximized.
        - Each edge contributes to the Hamiltonian based on whether its endpoints
          are in different partitions.

        This method constructs a diagonal Hamiltonian matrix where each diagonal
        element corresponds to the energy (cost) of a specific bitstring (state).

            Parameters:
            - self: The Quantum_QAOA object to use.
            - graph: A NetworkX graph object representing the problem.

            Returns:
            - NDArray[np.float_]: A diagonal Hamiltonian matrix where each state corresponds to a specific energy level.
        """
        num_nodes = graph.number_of_nodes()  # Number of nodes in the graph
        hamiltonian = np.zeros(
            (2**num_nodes, 2**num_nodes)
        )  # Initialize a 2^n x 2^n matrix

        # Loop through each edge in the graph to compute the contributions to the Hamiltonian
        for i, j in graph.edges:
            for state in range(
                2**num_nodes
            ):  # Iterate over all possible states (bitstrings)
                z_i = 1 if (state >> i) & 1 == 0 else -1
                z_j = 1 if (state >> j) & 1 == 0 else -1
                # Add the contribution of the edge (i, j) to the Hamiltonian
                hamiltonian[state, state] += (1 - z_i * z_j) / 2

        return hamiltonian  # Return the computed Hamiltonian

    def evaluate_hamiltonian(
        self, graph: nx.Graph, parameters: NDArray[np.float_]
    ) -> float:
        """
        Evaluate the expectation value of the Hamiltonian for the current QAOA parameters.

        Parameters:
        - self: The Quantum_QAOA object to use.
        - graph: A NetworkX graph object representing the problem.
        - parameters: The QAOA parameters (angles).

        Returns:
        - float: The expectation value of the Hamiltonian.
        """
        hamiltonian = self.compute_hamiltonian(
            graph
        )  # Compute the Hamiltonian for the graph
        count_dict = self.quantum_loop(
            parameters
        )  # Get the bitstring probabilities from QAOA

        # Calculate the expectation value of the Hamiltonian
        expectation_value = 0  # Initialize the total expectation value
        total_counts = sum(count_dict.values())  # Count the total number of samples

        for bitstring, count in count_dict.items():
            state_index = int(
                bitstring, 2
            )  # Convert the bitstring to an integer index corresponding to the state
            expectation_value += (
                hamiltonian[state_index, state_index] * count
            )  # Add the Hamiltonian value for this state

        return expectation_value / total_counts

    def optimize_parameters(
        self, graph, initial_params, minimizer: Callable
    ) -> NDArray[np.float_]:
        """
        Optimize the QAOA parameters to minimize the expectation value of the Hamiltonian.

        Parameters:
        - self: The Quantum_QAOA object to use.
        - graph: A NetworkX graph object representing the problem.
        - initial_params: Initial guesses for the QAOA parameters.

        Returns:
        - NDArray[np.float_]: The optimized QAOA parameters.
        """

        # # Define the cost function to minimize
        def cost_function(params):
            return -self.evaluate_hamiltonian(
                graph, params
            )  # This is negative because we minimize

        # Use scipy.optimize.minimize with the COBYLA method for optimization
        result = minimizer(cost_function, initial_params, method="COBYLA")

        return result

    def run(
        self,
        shots=1000,
        generate_histogram=False,
        file_name="QAOA_histo_optimized.pdf",
        minimizer: Callable = base_minimizer,
    ) -> dict:
        """
        Execute the QAOA algorithm with parameter optimization.

        Parameters:
        - NDArray[np.float_]:
        - graph: A NetworkX graph object representing the problem
        - shots: Number of measurements to perform in the quantum simulation.
        - generate_histogram: Whether to generate a histogram of the results.
        - file_name: Name of the file to save the histogram (if generated).

        Returns:
        - dict: A dictionary of bitstring counts after running QAOA with optimized parameters.
        """
        # Generate random guesses for the parameters t and s
        np.random.seed(123)  # Set a random seed
        guess_t = np.random.uniform(8, 10, self.layers)  # Random initial values for t
        guess_s = np.random.uniform(1, 3, self.layers)  # Random initial values for s
        initial_params = np.r_[
            guess_t, guess_s
        ]  # Combine t and s into a single parameter array

        # Optimize the parameters to minimize the Hamiltonian's expectation value
        optimized_params = self.optimize_parameters(
            self.graph, initial_params, minimizer
        )

        # Run the QAOA simulation with the optimized parameters
        result = self.quantum_loop(optimized_params)

        # If requested, generate and save a histogram of the results
        if generate_histogram:
            plot_histogram(result, shots, file_name)

        return result  # Return the final bitstring counts
