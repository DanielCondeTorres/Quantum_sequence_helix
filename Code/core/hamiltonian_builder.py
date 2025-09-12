# core/hamiltonian_builder.py
import numpy as np
import pennylane as qml
from typing import List, Tuple, Optional
from utils.general_utils import get_qubit_index
from data_loaders.energy_matrix_loader import _load_energy_matrix_file, _load_helix_pairs_matrix_file

class HamiltonianBuilder:
    """
    Builds the quantum Hamiltonian (QUBO) for the protein design problem.
    """
    def __init__(self, L: int, amino_acids: List[str], bits_per_pos: int,
                 n_qubits: int, **kwargs):
        self.L = L
        self.amino_acids = amino_acids
        self.n_aa = len(amino_acids)
        self.bits_per_pos = bits_per_pos
        self.n_qubits = n_qubits
        self.kwargs = kwargs
        self.pauli_terms = []
        self._init_amino_acid_properties()
        # Debugging prints
        print("\nDEBUG: Propiedades de los aminoácidos en uso:")
        for idx, aa in enumerate(self.amino_acids):
            print(f"  - {aa}: hidrofobicidad={self.hydrophobic[idx]:.2f}")
    
    def _init_amino_acid_properties(self):
        # Simplified properties for quantum demo
        properties = {
            'A': {'helix': 1.42, 'hydrophobic': 1.80, 'charge': 0},
            'R': {'helix': 0.98, 'hydrophobic': -4.50, 'charge': 1},
            'N': {'helix': 0.67, 'hydrophobic': -3.50, 'charge': 0},
            'D': {'helix': 1.01, 'hydrophobic': -3.50, 'charge': -1},
            'C': {'helix': 0.70, 'hydrophobic': 2.50, 'charge': 0},
            'E': {'helix': 1.51, 'hydrophobic': -3.50, 'charge': -1},
            'Q': {'helix': 1.11, 'hydrophobic': -3.50, 'charge': 0},
            'G': {'helix': 0.57, 'hydrophobic': -0.40, 'charge': 0},
            'H': {'helix': 1.00, 'hydrophobic': -3.20, 'charge': 0},
            'I': {'helix': 1.08, 'hydrophobic': 4.50, 'charge': 0},
            'L': {'helix': 1.21, 'hydrophobic': 3.80, 'charge': 0},
            'K': {'helix': 1.16, 'hydrophobic': -3.90, 'charge': 1},
            'M': {'helix': 1.45, 'hydrophobic': 1.90, 'charge': 0},
            'F': {'helix': 1.13, 'hydrophobic': 2.80, 'charge': 0},
            'P': {'helix': 0.57, 'hydrophobic': -1.60, 'charge': 0},
            'S': {'helix': 0.77, 'hydrophobic': -0.80, 'charge': 0},
            'T': {'helix': 0.83, 'hydrophobic': -0.70, 'charge': 0},
            'W': {'helix': 1.08, 'hydrophobic': -0.90, 'charge': 0},
            'Y': {'helix': 0.69, 'hydrophobic': -1.30, 'charge': 0},
            'V': {'helix': 1.06, 'hydrophobic': 4.20, 'charge': 0},
        }
        for aa in self.amino_acids:
            if aa not in properties:
                print(f"Warning: Unknown amino acid '{aa}'. Using neutral defaults.")
                properties[aa] = {'helix': 1.0, 'hydrophobic': 0.0, 'charge': 0}
        
        self.helix_prop = np.array([properties[aa]['helix'] for aa in self.amino_acids])
        self.hydrophobic = np.array([properties[aa]['hydrophobic'] for aa in self.amino_acids])
        self.charges = np.array([properties[aa]['charge'] for aa in self.amino_acids])
        self.h_alpha = self.helix_prop
    
    def _projector_terms_for_code(self, position: int, code: int, base_coeff: float):
        b = self.bits_per_pos
        s = []
        for k in range(b):
            v_k = (code >> k) & 1
            s.append(1.0 if v_k == 0 else -1.0)
        num_subsets = 1 << b
        for mask in range(num_subsets):
            coeff = base_coeff * (1.0 / (2 ** b))
            pauli = ['I'] * self.n_qubits
            for k in range(b):
                if (mask >> k) & 1:
                    coeff *= s[k]
                    w = get_qubit_index(position, k, self.bits_per_pos)
                    pauli[w] = 'Z'
            self.pauli_terms.append((coeff, ''.join(pauli)))
    
    def _add_local_terms(self, weight: float):
        for i in range(self.L):
            for α in range(self.n_aa):
                base = -weight * self.h_alpha[α]
                self._projector_terms_for_code(i, α, base)
    
    
    def _add_pairwise_terms(self, weight: float):
        if self.bits_per_pos > 3: return
        for i in range(self.L):
            for j in range(i+1, self.L):
                if abs(i-j) <= 3:
                    for α in range(self.n_aa):
                        for β in range(self.n_aa):
                            interaction = 0.0
                            if self.hydrophobic[α] > 0.5 and self.hydrophobic[β] > 0.5:
                                interaction = -0.1
                            elif self.charges[α] != 0 and self.charges[β] != 0:
                                if self.charges[α] * self.charges[β] > 0:
                                    interaction = 0.2
                                else:
                                    interaction = -0.1
                            if interaction != 0:
                                base = weight * interaction
                                b = self.bits_per_pos
                                s_i = [1.0 if ((α >> k) & 1) == 0 else -1.0 for k in range(b)]
                                s_j = [1.0 if ((β >> k) & 1) == 0 else -1.0 for k in range(b)]
                                for mask_i in range(1 << b):
                                    for mask_j in range(1 << b):
                                        coeff = base * (1.0 / (2 ** (2*b)))
                                        pauli = ['I'] * self.n_qubits
                                        for k in range(b):
                                            if (mask_i >> k) & 1:
                                                coeff *= s_i[k]
                                                pauli[get_qubit_index(i, k, self.bits_per_pos)] = 'Z'
                                            if (mask_j >> k) & 1:
                                                coeff *= s_j[k]
                                                pauli[get_qubit_index(j, k, self.bits_per_pos)] = 'Z'
                                        self.pauli_terms.append((coeff, ''.join(pauli)))
    
    def _add_miyazawa_jernigan_terms(self, weight: float, max_dist: int):
        """Adds Miyazawa-Jernigan interaction terms to the Hamiltonian."""
        mj_interaction, list_aa = _load_energy_matrix_file()
        aa_to_idx = {aa: i for i, aa in enumerate(list_aa)}

        for i in range(self.L):
            for j in range(i + 1, self.L):
                if abs(i - j) <= max_dist:
                    for α in range(self.n_aa):
                        for β in range(self.n_aa):
                            aa_i = self.amino_acids[α]
                            aa_j = self.amino_acids[β]
                            
                            mj_idx_i = aa_to_idx[aa_i]
                            mj_idx_j = aa_to_idx[aa_j]
                            interaction_energy = mj_interaction[min(mj_idx_i, mj_idx_j), max(mj_idx_i, mj_idx_j)]

                            if not np.isclose(interaction_energy, 0.0):
                                base = weight * interaction_energy
                                b = self.bits_per_pos
                                
                                s_i = [1.0 if ((α >> k) & 1) == 0 else -1.0 for k in range(b)]
                                s_j = [1.0 if ((β >> k) & 1) == 0 else -1.0 for k in range(b)]
                                for mask_i in range(1 << b):
                                    for mask_j in range(1 << b):
                                        coeff = base * (1.0 / (2 ** (2*b)))
                                        pauli = ['I'] * self.n_qubits
                                        for k in range(b):
                                            if (mask_i >> k) & 1:
                                                coeff *= s_i[k]
                                                pauli[get_qubit_index(i, k, self.bits_per_pos)] = 'Z'
                                            if (mask_j >> k) & 1:
                                                coeff *= s_j[k]
                                                pauli[get_qubit_index(j, k, self.bits_per_pos)] = 'Z'
                                        self.pauli_terms.append((coeff, ''.join(pauli)))
    
    def _add_helix_pairs_terms(self, weight: float, max_dist: int):
        """Adds helix pair propensity interaction terms to the Hamiltonian."""
        helix_matrix, list_aa = _load_helix_pairs_matrix_file()
        aa_to_idx = {aa: i for i, aa in enumerate(list_aa)}

        for i in range(self.L):
            for j in range(i + 1, self.L):
                if abs(i - j) <= max_dist:
                    for α in range(self.n_aa):
                        for β in range(self.n_aa):
                            aa_i = self.amino_acids[α]
                            aa_j = self.amino_acids[β]

                            idx_i = aa_to_idx[aa_i]
                            idx_j = aa_to_idx[aa_j]

                            # ahora usamos la matriz completa (no triangular)
                            interaction_prop = helix_matrix[idx_i, idx_j]

                            if not np.isclose(interaction_prop, 0.0):
                                base = weight * interaction_prop
                                b = self.bits_per_pos

                                s_i = [1.0 if ((α >> k) & 1) == 0 else -1.0 for k in range(b)]
                                s_j = [1.0 if ((β >> k) & 1) == 0 else -1.0 for k in range(b)]
                                for mask_i in range(1 << b):
                                    for mask_j in range(1 << b):
                                        coeff = base * (1.0 / (2 ** (2 * b)))
                                        pauli = ['I'] * self.n_qubits
                                        for k in range(b):
                                            if (mask_i >> k) & 1:
                                                coeff *= s_i[k]
                                                pauli[get_qubit_index(i, k, self.bits_per_pos)] = 'Z'
                                            if (mask_j >> k) & 1:
                                                coeff *= s_j[k]
                                                pauli[get_qubit_index(j, k, self.bits_per_pos)] = 'Z'
                                        self.pauli_terms.append((coeff, ''.join(pauli)))
    
    
    
    def _pos_in_membrane(self, pos: int) -> bool:
        mode = self.kwargs.get('membrane_mode', 'span')
        if mode == 'set':
            return pos in self.kwargs.get('membrane_positions', set())
        if mode == 'span':
            membrane_span = self.kwargs.get('membrane_span', None)
            if membrane_span is None: return False
            start, end = membrane_span
            return start <= pos <= end
        if mode == 'wheel':
            angle = (pos * 100.0 + self.kwargs.get('wheel_phase_deg', 0.0)) % 360.0
            if angle > 180.0: angle -= 360.0
            return abs(angle) <= self.kwargs.get('wheel_halfwidth_deg', 40.0)
        return False
    
    def _add_environment_terms(self, weight: float):
        print("\nDEBUG: Término de entorno activado.")
        for i in range(self.L):
            in_mem = self._pos_in_membrane(i)
            print(f"  - Posición {i}: {'En membrana' if in_mem else 'En agua'}")
            env_pref = 1.0 if in_mem else -1.0
            for α in range(self.n_aa):
                base = -weight * env_pref * self.hydrophobic[α]
                self._projector_terms_for_code(i, α, base)
    
    def _add_membrane_charge_term(self, weight: float):
        membrane_charge = self.kwargs.get('membrane_charge', 'neu')
        charge_sign = -1.0 if membrane_charge == 'neg' else (1.0 if membrane_charge == 'pos' else 0.0)
        for i in range(self.L):
            if not self._pos_in_membrane(i): continue
            for α in range(self.n_aa):
                base = weight * charge_sign * self.charges[α]
                self._projector_terms_for_code(i, α, base)
    
    def _add_hydrophobic_moment_terms(self, weight: float):
        if self.bits_per_pos > 3: return
        phi = np.deg2rad(100.0)
        for i in range(self.L):
            for j in range(i, self.L):
                cos_fac = np.cos(phi * (j - i))
                if np.isclose(cos_fac, 0.0): continue
                for α in range(self.n_aa):
                    for β in range(self.n_aa):
                        hij = self.hydrophobic[α] * self.hydrophobic[β]
                        if hij == 0: continue
                        base = -weight * hij * cos_fac
                        b = self.bits_per_pos
                        s_i = [1.0 if ((α >> k) & 1) == 0 else -1.0 for k in range(b)]
                        s_j = [1.0 if ((β >> k) & 1) == 0 else -1.0 for k in range(b)]
                        for mask_i in range(1 << b):
                            for mask_j in range(1 << b):
                                coeff = base * (1.0 / (2 ** (2*b)))
                                pauli = ['I'] * self.n_qubits
                                for k in range(b):
                                    if (mask_i >> k) & 1:
                                        coeff *= s_i[k]
                                        pauli[get_qubit_index(i, k, self.bits_per_pos)] = 'Z'
                                    if (mask_j >> k) & 1:
                                        coeff *= s_j[k]
                                        pauli[get_qubit_index(j, k, self.bits_per_pos)] = 'Z'
                                self.pauli_terms.append((coeff, ''.join(pauli)))
    
    def _add_invalid_code_penalties(self, weight: float):
        max_code = (1 << self.bits_per_pos) - 1
        if self.n_aa - 1 == max_code: return
        for i in range(self.L):
            for code in range(self.n_aa, max_code + 1):
                self._projector_terms_for_code(i, code, weight)
    
    def build_hamiltonian(self, backend: str):
        print("Building quantum Hamiltonian...")
        # Local amino acid preferences
        self._add_local_terms(weight=self.kwargs.get('lambda_local', 1.0))
        
        # Miyazawa-Jernigan pairwise interactions (always on)
        print("Adding Miyazawa-Jernigan terms...")
        self._add_miyazawa_jernigan_terms(weight=self.kwargs.get('lambda_pairwise', 1.0),
                                         max_dist=self.kwargs.get('max_interaction_dist', 3))

        # New: Helix pair propensities
        if self.kwargs.get('lambda_helix_pairs', 0.0) != 0.0:
            print("Adding Helix Pair Propensity terms...")
            self._add_helix_pairs_terms(weight=self.kwargs.get('lambda_helix_pairs', 0.0),
                                        max_dist=self.kwargs.get('max_interaction_dist', 3))

        # Environment preference
        if self.kwargs.get('lambda_env', 0.0) != 0.0:
            self._add_environment_terms(self.kwargs.get('lambda_env', 0.0))

        # Membrane charge interaction term
        if self.kwargs.get('lambda_charge', 0.0) != 0.0:
            self._add_membrane_charge_term(self.kwargs.get('lambda_charge', 0.0))

        # Hydrophobic moment encouragement
        if self.kwargs.get('lambda_mu', 0.0) != 0.0:
            self._add_hydrophobic_moment_terms(self.kwargs.get('lambda_mu', 0.0))

        # Penalize invalid codes
        self._add_invalid_code_penalties(weight=20.0)
        
        print(f"\nHamiltonian built with {len(self.pauli_terms)} Pauli terms")
        
        if backend == 'pennylane':
            coeffs = [term[0] for term in self.pauli_terms]
            observables = []
            for coeff, pauli_string in self.pauli_terms:
                obs_list = []
                for i, pauli in enumerate(pauli_string):
                    if pauli == 'Z': obs_list.append(qml.PauliZ(i))
                if obs_list:
                    if len(obs_list) == 1: observables.append(obs_list[0])
                    else: observables.append(qml.prod(*obs_list))
                else: observables.append(qml.Identity(0))
            hamiltonian = qml.Hamiltonian(coeffs, observables)
            print(f"PennyLane Hamiltonian created with {len(coeffs)} terms")
            return self.pauli_terms, hamiltonian
        return self.pauli_terms, None