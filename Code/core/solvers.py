import numpy as np
import pennylane as qml
from pennylane import numpy as qnp
from typing import List, Tuple, Dict, Any, Optional
import itertools

class QAOASolver:
    """
    Manages the QAOA circuit and optimization process with PennyLane.
    """
    def __init__(self, cost_hamiltonian, n_qubits, pauli_terms, amino_acids, L, bits_per_pos):
        self.cost_hamiltonian = cost_hamiltonian
        self.n_qubits = n_qubits
        self.pauli_terms = pauli_terms
        self.amino_acids = amino_acids
        self.L = L
        self.bits_per_pos = bits_per_pos
        self.n_aa = len(amino_acids) # <--- Esta línea faltaba
        self.dev = qml.device('lightning.qubit', wires=self.n_qubits)

    def _get_marginals(self, probs: np.ndarray) -> np.ndarray:
        """
        Calculates the marginal probabilities for each amino acid at each position.
        """
        marginals = np.zeros((self.L, self.n_aa))
        n_aa = self.n_aa
        for bitstring_int in range(len(probs)):
            bitstring = format(bitstring_int, f'0{self.n_qubits}b')
            sequence_code = []
            for i in range(self.L):
                pos_code_str = bitstring[i*self.bits_per_pos:(i+1)*self.bits_per_pos]
                pos_code_int = int(pos_code_str, 2)
                sequence_code.append(pos_code_int)
            for i in range(self.L):
                if sequence_code[i] < n_aa:
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
                decoded_sequence += 'X' # Violation or invalid code
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
        
        possible_codes = list(range(self.n_aa))
        all_combinations = itertools.product(possible_codes, repeat=self.L)
        
        for sequence_code in all_combinations:
            bitstring = ''.join(format(c, f'0{self.bits_per_pos}b') for c in sequence_code)
            current_energy = self.compute_energy_from_bitstring(bitstring)
            if current_energy < best_energy:
                best_energy = current_energy
                best_bitstring = bitstring
        
        best_sequence = self.decode_solution(best_bitstring)
        
        print(f"✅ Classical brute-force completed!")
        print(f"➡️ Best sequence: {best_sequence} | Energy: {best_energy:.6f}")
        
        return {
            'bitstring': best_bitstring,
            'sequence': best_sequence,
            'energy': best_energy
        }