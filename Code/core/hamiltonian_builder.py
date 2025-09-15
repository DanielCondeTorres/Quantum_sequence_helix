import numpy as np
import pennylane as qml
from typing import List, Tuple, Optional
from utils.general_utils import get_qubit_index
from data_loaders.energy_matrix_loader import _load_energy_matrix_file, _load_helix_pairs_matrix_file

class HamiltonianBuilder:
    """
    Builds the quantum Hamiltonian (QUBO) for the protein design problem.
    Enhanced for amphipathic helix formation in membranes with balanced terms.
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
            print(f"  - {aa}: hidrofobicidad={self.hydrophobic[idx]:.2f}, polar={self.is_polar[idx]}")

    def _init_amino_acid_properties(self):
        # Enhanced properties for amphipathic helix formation
        properties = {
            'A': {'helix': 1.42, 'hydrophobic': 1.80, 'charge': 0, 'polar': False, 'volume': 88.6},
            'R': {'helix': 0.98, 'hydrophobic': -4.50, 'charge': 1, 'polar': True, 'volume': 173.4},
            'N': {'helix': 0.67, 'hydrophobic': -3.50, 'charge': 0, 'polar': True, 'volume': 114.1},
            'D': {'helix': 1.01, 'hydrophobic': -3.50, 'charge': -1, 'polar': True, 'volume': 111.1},
            'C': {'helix': 0.70, 'hydrophobic': 2.50, 'charge': 0, 'polar': False, 'volume': 108.5},
            'E': {'helix': 1.51, 'hydrophobic': -3.50, 'charge': -1, 'polar': True, 'volume': 138.4},
            'Q': {'helix': 1.11, 'hydrophobic': -3.50, 'charge': 0, 'polar': True, 'volume': 143.8},
            'G': {'helix': 0.57, 'hydrophobic': -0.40, 'charge': 0, 'polar': False, 'volume': 60.1},
            'H': {'helix': 1.00, 'hydrophobic': -3.20, 'charge': 0, 'polar': True, 'volume': 153.2},
            'I': {'helix': 1.08, 'hydrophobic': 4.50, 'charge': 0, 'polar': False, 'volume': 166.7},
            'L': {'helix': 1.21, 'hydrophobic': 3.80, 'charge': 0, 'polar': False, 'volume': 166.7},
            'K': {'helix': 1.16, 'hydrophobic': -3.90, 'charge': 1, 'polar': True, 'volume': 168.6},
            'M': {'helix': 1.45, 'hydrophobic': 1.90, 'charge': 0, 'polar': False, 'volume': 162.9},
            'F': {'helix': 1.13, 'hydrophobic': 2.80, 'charge': 0, 'polar': False, 'volume': 189.9},
            'P': {'helix': 0.57, 'hydrophobic': -1.60, 'charge': 0, 'polar': False, 'volume': 112.7},
            'S': {'helix': 0.77, 'hydrophobic': -0.80, 'charge': 0, 'polar': True, 'volume': 89.0},
            'T': {'helix': 0.83, 'hydrophobic': -0.70, 'charge': 0, 'polar': True, 'volume': 116.1},
            'W': {'helix': 1.08, 'hydrophobic': -0.90, 'charge': 0, 'polar': False, 'volume': 227.8},
            'Y': {'helix': 0.69, 'hydrophobic': -1.30, 'charge': 0, 'polar': True, 'volume': 193.6},
            'V': {'helix': 1.06, 'hydrophobic': 4.20, 'charge': 0, 'polar': False, 'volume': 140.0},
        }
        
        for aa in self.amino_acids:
            if aa not in properties:
                print(f"Warning: Unknown amino acid '{aa}'. Using neutral defaults.")
                properties[aa] = {'helix': 1.0, 'hydrophobic': 0.0, 'charge': 0, 'polar': False, 'volume': 120.0}
        
        self.helix_prop = np.array([properties[aa]['helix'] for aa in self.amino_acids])
        self.hydrophobic = np.array([properties[aa]['hydrophobic'] for aa in self.amino_acids])
        self.charges = np.array([properties[aa]['charge'] for aa in self.amino_acids])
        self.is_polar = np.array([properties[aa]['polar'] for aa in self.amino_acids])
        self.volumes = np.array([properties[aa]['volume'] for aa in self.amino_acids])
        self.h_alpha = self.helix_prop / np.max(self.helix_prop)  # Normalize helix propensity

    def _projector_terms_for_code(self, position: int, code: int, base_coeff: float):
        b = self.bits_per_pos
        s = []
        for k in range(b):
            v_k = (code >> k) & 1
            s.append(1.0 if v_k == 0 else -1.0)
        num_subsets = 1 << b
        for mask in range(num_subsets):
            coeff = base_coeff / (2 ** b)  # Simplified normalization
            pauli = ['I'] * self.n_qubits
            for k in range(b):
                if (mask >> k) & 1:
                    coeff *= s[k]
                    w = get_qubit_index(position, k, self.bits_per_pos)
                    pauli[w] = 'Z'
            if abs(coeff) > 1e-10:  # Avoid adding negligible terms
                self.pauli_terms.append((coeff, ''.join(pauli)))

    def _add_local_terms(self, weight: float):
        """Enhanced local terms that favor helix-forming residues with normalized contribution"""
        for i in range(self.L):
            for α in range(self.n_aa):
                base = -weight * self.h_alpha[α]  # Use normalized helix propensity
                self._projector_terms_for_code(i, α, base)

    def _add_enhanced_environment_terms(self, weight: float):
        """
        Enhanced environment terms for amphipathic helix formation with balanced penalties.
        Prioritizes hydrophobic residues in membrane, polar/charged in water.
        """
        print("\nDEBUG: Término de entorno mejorado activado.")
        print(f"DEBUG: wheel_halfwidth_deg = {self.kwargs.get('wheel_halfwidth_deg', 90.0)}")
        
        for i in range(self.L):
            membrane_angle = (i * 100.0 + self.kwargs.get('wheel_phase_deg', 0.0)) % 360.0
            if membrane_angle > 180.0: 
                membrane_angle -= 360.0
            
            faces_membrane = abs(membrane_angle) <= self.kwargs.get('wheel_halfwidth_deg', 90.0)
            environment = "membrane" if faces_membrane else "water"
            
            print(f"  - Pos {i}: ángulo={membrane_angle:.1f}°, entorno={environment}")
            
            for α in range(self.n_aa):
                aa = self.amino_acids[α]
                hydrophobic_score = self.hydrophobic[α]
                is_polar = self.is_polar[α]
                
                env_bonus = 0.0
                
                if environment == "membrane":
                    # Reward hydrophobic residues, penalize polar/charged
                    if hydrophobic_score > 0:
                        env_bonus = -weight * hydrophobic_score * 0.7  # Reduced multiplier for balance
                    if is_polar or self.charges[α] != 0:
                        env_bonus += weight * abs(hydrophobic_score) * 1.0  # Reduced multiplier
                else:  # water
                    # Reward polar/charged, penalize hydrophobic
                    if is_polar or self.charges[α] != 0:
                        env_bonus = -weight * abs(hydrophobic_score) * 0.7
                    if hydrophobic_score > 0:
                        env_bonus += weight * hydrophobic_score * 1.0
                
                if abs(env_bonus) > 1e-6:
                    self._projector_terms_for_code(i, α, env_bonus)
                    print(f"    -> Added env_bonus = {env_bonus:.3f} for {aa}")

    def _add_amphipathic_segregation_terms(self, weight: float):
        """
        Add terms to encourage segregation of hydrophobic and polar residues.
        """
        if weight == 0.0:
            return
            
        print("Adding amphipathic segregation terms...")
        
        membrane_positions = []
        water_positions = []
        for i in range(self.L):
            membrane_angle = (i * 100.0 + self.kwargs.get('wheel_phase_deg', 0.0)) % 360.0
            if membrane_angle > 180.0: 
                membrane_angle -= 360.0
            faces_membrane = abs(membrane_angle) <= self.kwargs.get('wheel_halfwidth_deg', 90.0)
            if faces_membrane:
                membrane_positions.append(i)
            else:
                water_positions.append(i)
        
        for i in membrane_positions:
            for j in water_positions:
                for α in range(self.n_aa):
                    for β in range(self.n_aa):
                        if self.hydrophobic[α] > 1.0 and (self.is_polar[β] or self.charges[β] != 0):
                            coupling = -weight * 0.5  # Reduced coupling strength
                            self._add_pairwise_coupling_term(i, j, α, β, coupling)
                        elif (self.is_polar[α] or self.charges[α] != 0) and self.hydrophobic[β] > 1.0:
                            coupling = weight * 0.5
                            self._add_pairwise_coupling_term(i, j, α, β, coupling)

    def _add_pairwise_coupling_term(self, i: int, j: int, α: int, β: int, base_coeff: float):
        """Simplified pairwise coupling term to reduce term explosion"""
        b = self.bits_per_pos
        pauli = ['I'] * self.n_qubits
        for k in range(b):
            w_i = get_qubit_index(i, k, self.bits_per_pos)
            w_j = get_qubit_index(j, k, self.bits_per_pos)
            v_α = (α >> k) & 1
            v_β = (β >> k) & 1
            if v_α == 0 and v_β == 0:
                pauli[w_i] = 'Z'
                pauli[w_j] = 'Z'
            elif v_α == 1 and v_β == 1:
                pauli[w_i] = 'Z'
                pauli[w_j] = 'Z'
                base_coeff = -base_coeff  # Invert sign for consistency
        if abs(base_coeff) > 1e-10:
            self.pauli_terms.append((base_coeff, ''.join(pauli)))

    def _add_miyazawa_jernigan_terms(self, weight: float, max_dist: int):
        """Adds Miyazawa-Jernigan interaction terms with normalized weights."""
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
                                self._add_pairwise_coupling_term(i, j, α, β, weight * interaction_energy / 10.0)  # Normalize

    def _add_helix_pairs_terms(self, weight: float, max_dist: int):
        """Adds helix pair propensity interaction terms with normalized weights."""
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
                            interaction_prop = helix_matrix[idx_i, idx_j]
                            if not np.isclose(interaction_prop, 0.0):
                                self._add_pairwise_coupling_term(i, j, α, β, weight * interaction_prop / 10.0)  # Normalize

    def _add_membrane_charge_term(self, weight: float):
        """Enhanced membrane charge interaction with balanced penalty"""
        membrane_charge = self.kwargs.get('membrane_charge', 'neu')
        charge_sign = -1.0 if membrane_charge == 'neg' else (1.0 if membrane_charge == 'pos' else 0.0)
        
        for i in range(self.L):
            membrane_angle = (i * 100.0 + self.kwargs.get('wheel_phase_deg', 0.0)) % 360.0
            if membrane_angle > 180.0: 
                membrane_angle -= 360.0
            faces_membrane = abs(membrane_angle) <= self.kwargs.get('wheel_halfwidth_deg', 90.0)
            
            if faces_membrane:
                for α in range(self.n_aa):
                    penalty = weight * charge_sign * self.charges[α]
                    if abs(penalty) > 1e-6:
                        self._projector_terms_for_code(i, α, penalty)

    def _add_hydrophobic_moment_terms(self, weight: float):
        """Adds terms for the hydrophobic moment with simplified coupling."""
        if self.bits_per_pos > 3: return
        phi = np.deg2rad(100.0)
        
        for i in range(self.L):
            for j in range(self.L):
                if i == j: continue
                if abs(i - j) > 7: continue

                cos_term = np.cos(phi * (j - i))
                for α in range(self.n_aa):
                    for β in range(self.n_aa):
                        h_i = self.hydrophobic[α]
                        h_j = self.hydrophobic[β]
                        coupling = -weight * (h_i * h_j) * cos_term / 10.0  # Normalize
                        self._add_pairwise_coupling_term(i, j, α, β, coupling)

    def _add_invalid_code_penalties(self, weight: float):
        max_code = (1 << self.bits_per_pos) - 1
        if self.n_aa - 1 == max_code: return
        for i in range(self.L):
            for code in range(self.n_aa, max_code + 1):
                self._projector_terms_for_code(i, code, weight)

    def build_hamiltonian(self, backend: str):
        print("Building enhanced quantum Hamiltonian for amphipathic helix formation...")
        
        # Add terms with balanced contributions
        self._add_local_terms(weight=self.kwargs.get('lambda_local', 1.0))
        self._add_miyazawa_jernigan_terms(weight=self.kwargs.get('lambda_pairwise', 1.0),
                                          max_dist=self.kwargs.get('max_interaction_dist', 3))
        self._add_helix_pairs_terms(weight=self.kwargs.get('lambda_helix_pairs', 0.0),
                                    max_dist=self.kwargs.get('max_interaction_dist', 3))

        if self.kwargs.get('lambda_env', 0.0) > 0.0:
            self._add_enhanced_environment_terms(self.kwargs.get('lambda_env', 0.0))
        if self.kwargs.get('lambda_segregation', 0.0) > 0.0:
            self._add_amphipathic_segregation_terms(self.kwargs.get('lambda_segregation', 0.0))
        if self.kwargs.get('lambda_charge', 0.0) > 0.0:
            self._add_membrane_charge_term(self.kwargs.get('lambda_charge', 0.0))
        if self.kwargs.get('lambda_mu', 0.0) > 0.0:
            print("Adding enhanced hydrophobic moment terms...")
            self._add_hydrophobic_moment_terms(self.kwargs.get('lambda_mu', 0.0))
        
        self._add_invalid_code_penalties(weight=5.0)  # Reduced from 20.0
        
        print(f"\nEnhanced Hamiltonian built with {len(self.pauli_terms)} Pauli terms")
        coeffs = [term[0] for term in self.pauli_terms]
        print("DEBUG: Total coefficients:", coeffs)
        
        if backend == 'pennylane':
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