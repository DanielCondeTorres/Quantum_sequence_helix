import numpy as np
import pennylane as qml
from pennylane import numpy as qnp
from typing import List, Tuple, Dict, Any, Optional
import itertools
import matplotlib.pyplot as plt  # Added for plotting

class QAOASolver:
    """
    Manages the QAOA circuit and optimization process with PennyLane.
    Improved with multi-layer support, warm-start initialization, and native PennyLane optimizer.
    """
    def __init__(self, cost_hamiltonian, n_qubits, pauli_terms, amino_acids, L, bits_per_pos, layers=1, warm_start=False, shots: int = 1000):
        self.cost_hamiltonian = cost_hamiltonian
        self.n_qubits = n_qubits
        self.pauli_terms = pauli_terms
        self.amino_acids = amino_acids
        self.L = L
        self.bits_per_pos = bits_per_pos
        self.n_aa = len(amino_acids)
        self.dev = qml.device('lightning.qubit', wires=self.n_qubits, shots=shots)  # Added shots parameter
        self.layers = layers  # Number of QAOA layers (p)
        self.warm_start = warm_start  # Option for warm-start initialization

    def _qaoa_circuit(self, params):
        """
        Defines the QAOA circuit with p layers.
        """
        gammas = params[:self.layers]
        betas = params[self.layers:]
        
        # Initial state: uniform superposition
        for i in range(self.n_qubits):
            qml.Hadamard(wires=i)
        
        # QAOA layers
        for p in range(self.layers):
            # Cost layer
            qml.templates.ApproximateTimeEvolution(self.cost_hamiltonian, gammas[p], n=1)
            # Mixer layer
            for i in range(self.n_qubits):
                qml.RX(2 * betas[p], wires=i)

    def _cost_function(self, params):
        @qml.qnode(self.dev)
        def circuit():
            self._qaoa_circuit(params)
            return qml.expval(self.cost_hamiltonian)
        return circuit()

    def _initialize_params(self):
        """
        Initialize QAOA parameters, with optional warm-start.
        """
        if self.warm_start:
            # Simple warm-start: small values for gammas, pi/4 for betas
            gammas = qnp.random.uniform(0.01, 0.1, self.layers)
            betas = qnp.full(self.layers, np.pi / 4)
        else:
            gammas = qnp.random.uniform(0, np.pi, self.layers)
            betas = qnp.random.uniform(0, np.pi / 2, self.layers)
        return qnp.concatenate([gammas, betas])

    def solve(self, steps=100, learning_rate=0.1) -> Dict[str, Any]:
        """
        Optimizes the QAOA parameters using PennyLane's Adam optimizer.
        """
        print("\n🚀 Solving with Improved QAOA...")
        params = self._initialize_params()
        optimizer = qml.AdamOptimizer(stepsize=learning_rate)

        min_energy = float('inf')
        best_params = params
        costs = []

        for i in range(steps):
            params, cost = optimizer.step_and_cost(self._cost_function, params)
            costs.append(cost)
            if cost < min_energy:
                min_energy = cost
                best_params = params
            if i % 10 == 0:
                print(f"Step {i}: Energy = {cost:.6f}")

        # Sample from the optimized circuit to get probabilities
        @qml.qnode(self.dev)
        def prob_circuit():
            self._qaoa_circuit(best_params)
            return qml.probs(wires=range(self.n_qubits))

        probs = prob_circuit()
        print(f"Number of shots used: {self.dev.shots}")  # Debug shot count
        print("Probabilities:", probs)  # Debug print
        print("Number of probabilities:", len(probs))  # Debug print
        if len(probs) == 0:
            print("Warning: No probabilities computed. Check Hamiltonian or circuit.")
            return {
                'bitstring': '',
                'sequence': '',
                'energy': float('inf'),
                'costs': costs
            }

        repaired_bitstring = self._repair_with_marginals(probs)
        best_sequence = self.decode_solution(repaired_bitstring)
        best_energy = self.compute_energy_from_bitstring(repaired_bitstring)

        print(f"✅ QAOA optimization completed!")
        print(f"➡️ Best sequence: {best_sequence} | Energy: {best_energy:.6f}")

        # Plot the probability distribution with amino acid sequences
        self.plot_prob_with_sequences(probs)

        return {
            'bitstring': repaired_bitstring,
            'sequence': best_sequence,
            'energy': best_energy,
            'costs': costs  # Return costs for plotting
        }

    def plot_prob_with_sequences(self, probs: np.ndarray, top_k: int = 20):
        """
        Plots a bar chart of the top_k amino acid sequences and their probabilities.
        """
        # Sort the probabilities and get the top_k indices
        sorted_indices = np.argsort(probs)[::-1][:top_k]
        sorted_probs = probs[sorted_indices]
        sequences = [self.decode_solution(format(idx, f'0{self.n_qubits}b')) for idx in sorted_indices]

        plt.figure(figsize=(12, 6))
        plt.bar(range(len(sequences)), sorted_probs, color='blue')
        plt.xlabel('Amino Acid Sequences (Top ' + str(top_k) + ')')
        plt.ylabel('Probability')
        plt.title('Top Probability Distribution of Amino Acid Sequences from QAOA')
        plt.xticks(range(len(sequences)), sequences, rotation=90)
        plt.tight_layout()
        plt.savefig('qaoa_probability_plot.png')  # Save to file instead of show
        plt.close()  # Close the figure to avoid memory issues
        print("Plot saved as qaoa_probability_plot.png")

    def _get_marginals(self, probs: np.ndarray) -> np.ndarray:
        """
        Calculates the marginal probabilities for each amino acid at each position.
        """
        marginals = np.zeros((self.L, self.n_aa))
        for bitstring_int in range(len(probs)):
            bitstring = format(bitstring_int, f'0{self.n_qubits}b')
            sequence_code = []
            for i in range(self.L):
                pos_code_str = bitstring[i*self.bits_per_pos:(i+1)*self.bits_per_pos]
                pos_code_int = int(pos_code_str, 2)
                sequence_code.append(pos_code_int)
            for i in range(self.L):
                if sequence_code[i] < self.n_aa:
                    marginals[i, sequence_code[i]] += probs[bitstring_int]
        return marginals

    def _repair_with_marginals(self, probs: np.ndarray) -> str:
        """
        Repairs a QAOA solution by choosing the most probable amino acid at each position.
        """
        marginals = self._get_marginals(probs)
        repaired_sequence_code = np.argmax(marginals, axis=1)
        repaired_bitstring = ''.join(format(c, f'0{self.bits_per_pos}b') for c in repaired_sequence_code)
        return repaired_bitstring

    def decode_solution(self, bitstring: str) -> str:
        """
        Decodes a binary string back into a protein sequence.
        """
        decoded_sequence = ""
        for i in range(self.L):
            pos_code_str = bitstring[i*self.bits_per_pos:(i+1)*self.bits_per_pos]
            pos_code_int = int(pos_code_str, 2)
            if pos_code_int < self.n_aa:
                decoded_sequence += self.amino_acids[pos_code_int]
            else:
                decoded_sequence += 'X'  # Violation or invalid code
        return decoded_sequence

    def compute_energy_from_bitstring(self, bitstring: str) -> float:
        """
        Calculates the energy of a bitstring solution using the classical Hamiltonian.
        """
        energy = 0.0
        for coeff, pauli_string in self.pauli_terms:
            pauli_prod = 1.0
            for i, pauli in enumerate(pauli_string):
                if pauli == 'Z':
                    qubit_val = int(bitstring[i])
                    z_val = 1 if qubit_val == 0 else -1
                    pauli_prod *= z_val
            energy += coeff * pauli_prod
        return energy

class VQESolver:
    """
    Manages the VQE circuit and optimization process with PennyLane.
    Uses a more expressive ansatz for potentially better results.
    """
    def __init__(self, cost_hamiltonian, n_qubits, pauli_terms, amino_acids, L, bits_per_pos, layers=2, shots: int = 1000):
        self.cost_hamiltonian = cost_hamiltonian
        self.n_qubits = n_qubits
        self.pauli_terms = pauli_terms
        self.amino_acids = amino_acids
        self.L = L
        self.bits_per_pos = bits_per_pos
        self.n_aa = len(amino_acids)
        self.dev = qml.device('lightning.qubit', wires=self.n_qubits, shots=shots)  # Added shots parameter
        self.layers = layers  # Number of entangling layers

    def _vqe_ansatz(self, params):
        """
        Defines the VQE ansatz using StronglyEntanglingLayers for expressivity.
        """
        qml.templates.StronglyEntanglingLayers(params, wires=range(self.n_qubits))

    def _cost_function(self, params):
        @qml.qnode(self.dev)
        def circuit():
            self._vqe_ansatz(params)
            return qml.expval(self.cost_hamiltonian)
        return circuit()

    def _initialize_params(self):
        """
        Initialize VQE parameters.
        """
        shape = qml.templates.StronglyEntanglingLayers.shape(n_layers=self.layers, n_wires=self.n_qubits)
        return qnp.random.uniform(0, 2 * np.pi, shape)

    def solve(self, steps=200, learning_rate=0.1) -> Dict[str, Any]:
        """
        Optimizes the VQE parameters using PennyLane's Adam optimizer.
        """
        print("\n🔬 Solving with VQE...")
        params = self._initialize_params()
        optimizer = qml.AdamOptimizer(stepsize=learning_rate)

        min_energy = float('inf')
        best_params = params
        costs = []

        for i in range(steps):
            params, cost = optimizer.step_and_cost(self._cost_function, params)
            costs.append(cost)
            if cost < min_energy:
                min_energy = cost
                best_params = params
            if i % 20 == 0:
                print(f"Step {i}: Energy = {cost:.6f}")

        # Sample from the optimized circuit to get probabilities
        @qml.qnode(self.dev)
        def prob_circuit():
            self._vqe_ansatz(best_params)
            return qml.probs(wires=range(self.n_qubits))

        probs = prob_circuit()
        print(f"Number of shots used: {self.dev.shots}")  # Debug shot count
        print("Probabilities:", probs)  # Debug print
        print("Number of probabilities:", len(probs))  # Debug print
        if len(probs) == 0:
            print("Warning: No probabilities computed. Check Hamiltonian or circuit.")
            return {
                'bitstring': '',
                'sequence': '',
                'energy': float('inf'),
                'costs': costs
            }

        repaired_bitstring = self._repair_with_marginals(probs)
        best_sequence = self.decode_solution(repaired_bitstring)
        best_energy = self.compute_energy_from_bitstring(repaired_bitstring)

        print(f"✅ VQE optimization completed!")
        print(f"➡️ Best sequence: {best_sequence} | Energy: {best_energy:.6f}")

        # Plot the probability distribution with amino acid sequences
        self.plot_prob_with_sequences(probs)

        return {
            'bitstring': repaired_bitstring,
            'sequence': best_sequence,
            'energy': best_energy,
            'costs': costs  # Return costs for plotting
        }

    def plot_prob_with_sequences(self, probs: np.ndarray, top_k: int = 20):
        """
        Plots a bar chart of the top_k amino acid sequences and their probabilities.
        """
        # Sort the probabilities and get the top_k indices
        sorted_indices = np.argsort(probs)[::-1][:top_k]
        sorted_probs = probs[sorted_indices]
        sequences = [self.decode_solution(format(idx, f'0{self.n_qubits}b')) for idx in sorted_indices]

        plt.figure(figsize=(12, 6))
        plt.bar(range(len(sequences)), sorted_probs, color='green')
        plt.xlabel('Amino Acid Sequences (Top ' + str(top_k) + ')')
        plt.ylabel('Probability')
        plt.title('Top Probability Distribution of Amino Acid Sequences from VQE')
        plt.xticks(range(len(sequences)), sequences, rotation=90)
        plt.tight_layout()
        plt.savefig('vqe_probability_plot.png')  # Save to file instead of show
        plt.close()  # Close the figure to avoid memory issues
        print("Plot saved as vqe_probability_plot.png")

    def _get_marginals(self, probs: np.ndarray) -> np.ndarray:
        marginals = np.zeros((self.L, self.n_aa))
        for bitstring_int in range(len(probs)):
            bitstring = format(bitstring_int, f'0{self.n_qubits}b')
            sequence_code = []
            for i in range(self.L):
                pos_code_str = bitstring[i*self.bits_per_pos:(i+1)*self.bits_per_pos]
                pos_code_int = int(pos_code_str, 2)
                sequence_code.append(pos_code_int)
            for i in range(self.L):
                if sequence_code[i] < self.n_aa:
                    marginals[i, sequence_code[i]] += probs[bitstring_int]
        return marginals

    def _repair_with_marginals(self, probs: np.ndarray) -> str:
        marginals = self._get_marginals(probs)
        repaired_sequence_code = np.argmax(marginals, axis=1)
        repaired_bitstring = ''.join(format(c, f'0{self.bits_per_pos}b') for c in repaired_sequence_code)
        return repaired_bitstring

    def decode_solution(self, bitstring: str) -> str:
        """
        Decodes a binary string back into a protein sequence.
        """
        decoded_sequence = ""
        for i in range(self.L):
            pos_code_str = bitstring[i*self.bits_per_pos:(i+1)*self.bits_per_pos]
            pos_code_int = int(pos_code_str, 2)
            if pos_code_int < self.n_aa:
                decoded_sequence += self.amino_acids[pos_code_int]
            else:
                decoded_sequence += 'X'
        return decoded_sequence

    def compute_energy_from_bitstring(self, bitstring: str) -> float:
        """
        Calculates the energy of a bitstring solution using the classical Hamiltonian.
        """
        energy = 0.0
        for coeff, pauli_string in self.pauli_terms:
            pauli_prod = 1.0
            for i, pauli in enumerate(pauli_string):
                if pauli == 'Z':
                    qubit_val = int(bitstring[i])
                    z_val = 1 if qubit_val == 0 else -1
                    pauli_prod *= z_val
            energy += coeff * pauli_prod
        return energy

class ClassicalSolver:
    """
    Solves the QUBO classically by brute force enumeration.
    """
    def __init__(self, L: int, n_aa: int, bits_per_pos: int, pauli_terms: List, amino_acids: List[str]):
        self.L = L
        self.n_aa = n_aa
        self.bits_per_pos = bits_per_pos
        self.pauli_terms = pauli_terms
        self.amino_acids = amino_acids
    
    def decode_solution(self, bitstring: str) -> str:
        decoded_sequence = ""
        for i in range(self.L):
            pos_code_str = bitstring[i*self.bits_per_pos:(i+1)*self.bits_per_pos]
            pos_code_int = int(pos_code_str, 2)
            if pos_code_int < self.n_aa:
                decoded_sequence += self.amino_acids[pos_code_int]
            else:
                decoded_sequence += 'X'
        return decoded_sequence

    def compute_energy_from_bitstring(self, bitstring: str) -> float:
        energy = 0.0
        for coeff, pauli_string in self.pauli_terms:
            pauli_prod = 1.0
            for i, pauli in enumerate(pauli_string):
                if pauli == 'Z':
                    qubit_val = int(bitstring[i])
                    z_val = 1 if qubit_val == 0 else -1
                    pauli_prod *= z_val
            energy += coeff * pauli_prod
        return energy

    def solve(self) -> Dict[str, Any]:
        print("\n🏆 Solving with Classical Brute-Force...")
        best_bitstring = None
        best_energy = float('inf')
        top_sequences = []  # Store top sequences for debugging
        
        possible_codes = list(range(self.n_aa))
        all_combinations = list(itertools.product(possible_codes, repeat=self.L))
        
        for sequence_code in all_combinations:
            bitstring = ''.join(format(c, f'0{self.bits_per_pos}b') for c in sequence_code)
            current_energy = self.compute_energy_from_bitstring(bitstring)
            sequence = self.decode_solution(bitstring)
            top_sequences.append((sequence, bitstring, current_energy))
            if current_energy < best_energy:
                best_energy = current_energy
                best_bitstring = bitstring
        
        best_sequence = self.decode_solution(best_bitstring)
        
        # Debug: Print top 5 sequences and their energies
        top_sequences.sort(key=lambda x: x[2])  # Sort by energy
        print("\nTop 5 sequences by energy:")
        for seq, bits, energy in top_sequences[:5]:
            print(f"Sequence: {seq}, Energy: {energy:.6f}, Bitstring: {bits}")
        
        print(f"\n✅ Classical brute-force completed!")
        print(f"➡️ Best sequence: {best_sequence} | Energy: {best_energy:.6f}")

        return {
            'bitstring': best_bitstring,
            'sequence': best_sequence,
            'energy': best_energy,
            'costs': []  # Empty for classical
        }
        
        
        
