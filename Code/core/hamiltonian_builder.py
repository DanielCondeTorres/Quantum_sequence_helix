import numpy as np
import pennylane as qml
from typing import List, Tuple, Optional
from utils.general_utils import get_qubit_index
from data_loaders.energy_matrix_loader import _load_first_neighbors_matrix_file, _load_third_neighbors_matrix_file, _load_fourth_neighbors_matrix_file, _load_energy_matrix_file

class HamiltonianBuilder:
    """
    Builds the quantum Hamiltonian (QUBO) for the protein design problem.
    Enhanced for amphipathic helix formation in membranes with balanced terms.
    Based on Senes et al. statistical potentials and helix folding principles.
    All numerical values are z-score normalized.
    """
    def __init__(self, L: int, amino_acids: List[str], bits_per_pos: int, n_qubits: int, **kwargs):
        self.L = L
        self.amino_acids = amino_acids
        self.n_aa = len(amino_acids)
        self.bits_per_pos = bits_per_pos
        self.n_qubits = n_qubits
        self.kwargs = kwargs
        self.pauli_terms = []
        self._init_amino_acid_properties()
        
        # Debugging prints for original and normalized properties
        print("\nDEBUG: Propiedades de los aminoácidos en uso (original y normalizado):")
        for idx, aa in enumerate(self.amino_acids):
            print(f"  - {aa}:")
            print(f"    hidrofobicidad={self.hydrophobic[idx]:.2f}, normalizado={self.normalized_hydrophobic[idx]:.2f}")
            print(f"    helix_prop={self.helix_prop[idx]:.2f}, normalizado={self.normalized_helix_prop[idx]:.2f}")
            print(f"    charge={self.charges[idx]:.2f}, normalizado={self.normalized_charges[idx]:.2f}")
            print(f"    polar={self.is_polar[idx]}, normalizado={self.normalized_is_polar[idx]:.2f}")
            print(f"    volume={self.volumes[idx]:.2f}, normalizado={self.normalized_volumes[idx]:.2f}")
            print(f"    ez={self.ez_values[idx]:.2f}, normalizado={self.normalized_ez_values[idx]:.2f}")
            print(f"    interface_pref={self.interface_pref[idx]:.2f}, normalizado={self.normalized_interface_pref[idx]:.2f}")

    def _init_amino_acid_properties(self):
        """Enhanced properties for amphipathic helix formation with z-score normalization"""
        properties = {
            'A': {'helix': 1.42, 'hydrophobic': 1.80, 'charge': 0, 'polar': False, 'volume': 88.6, 'ez': 0.17},
            'R': {'helix': 0.98, 'hydrophobic': -4.50, 'charge': 1, 'polar': True, 'volume': 173.4, 'ez': 1.81},
            'N': {'helix': 0.67, 'hydrophobic': -3.50, 'charge': 0, 'polar': True, 'volume': 114.1, 'ez': 2.05},
            'D': {'helix': 1.01, 'hydrophobic': -3.50, 'charge': -1, 'polar': True, 'volume': 111.1, 'ez': 2.06},
            'C': {'helix': 0.70, 'hydrophobic': 2.50, 'charge': 0, 'polar': False, 'volume': 108.5, 'ez': 0.24},
            'E': {'helix': 1.51, 'hydrophobic': -3.50, 'charge': -1, 'polar': True, 'volume': 138.4, 'ez': 2.68},
            'Q': {'helix': 1.11, 'hydrophobic': -3.50, 'charge': 0, 'polar': True, 'volume': 143.8, 'ez': 0.77},
            'G': {'helix': 0.57, 'hydrophobic': -0.40, 'charge': 0, 'polar': False, 'volume': 60.1, 'ez': 0.01},
            'H': {'helix': 1.00, 'hydrophobic': -3.20, 'charge': 0, 'polar': True, 'volume': 153.2, 'ez': 0.96},
            'I': {'helix': 1.08, 'hydrophobic': 4.50, 'charge': 0, 'polar': False, 'volume': 166.7, 'ez': -1.12},
            'L': {'helix': 1.21, 'hydrophobic': 3.80, 'charge': 0, 'polar': False, 'volume': 166.7, 'ez': -1.25},
            'K': {'helix': 1.16, 'hydrophobic': -3.90, 'charge': 1, 'polar': True, 'volume': 168.6, 'ez': 2.80},
            'M': {'helix': 1.45, 'hydrophobic': 1.90, 'charge': 0, 'polar': False, 'volume': 162.9, 'ez': -0.23},
            'F': {'helix': 1.13, 'hydrophobic': 2.80, 'charge': 0, 'polar': False, 'volume': 189.9, 'ez': -1.85},
            'P': {'helix': 0.57, 'hydrophobic': -1.60, 'charge': 0, 'polar': False, 'volume': 112.7, 'ez': 0.45},
            'S': {'helix': 0.77, 'hydrophobic': -0.80, 'charge': 0, 'polar': True, 'volume': 89.0, 'ez': 1.13},
            'T': {'helix': 0.83, 'hydrophobic': -0.70, 'charge': 0, 'polar': True, 'volume': 116.1, 'ez': 0.14},
            'W': {'helix': 1.08, 'hydrophobic': -0.90, 'charge': 0, 'polar': False, 'volume': 227.8, 'ez': -1.85},
            'Y': {'helix': 0.69, 'hydrophobic': -1.30, 'charge': 0, 'polar': True, 'volume': 193.6, 'ez': -0.94},
            'V': {'helix': 1.06, 'hydrophobic': 4.20, 'charge': 0, 'polar': False, 'volume': 140.0, 'ez': -0.46},
        }
        
        interface_preference = {
            'A': -1.80, 'R': 4.5, 'N': 3.5, 'D': 3.5, 'C': -2.5,
            'E': 3.5, 'Q': 3.50, 'G': 0.4, 'H': 3.2, 'I': -4.5,
            'L': -3.8, 'K': 3.9, 'M': -1.9, 'F': -2.8, 'P': 1.6,
            'S': 0.8, 'T': 0.7, 'W': 0.9, 'Y': 1.3, 'V': -4.2
        }
        
        for aa in self.amino_acids:
            if aa not in properties:
                print(f"Warning: Unknown amino acid '{aa}'. Using neutral defaults.")
                properties[aa] = {'helix': 1.0, 'hydrophobic': 0.0, 'charge': 0, 'polar': False, 'volume': 120.0, 'ez': 0.0}
        
        # Initialize original arrays
        self.helix_prop = np.array([properties[aa]['helix'] for aa in self.amino_acids])
        self.hydrophobic = np.array([properties[aa]['hydrophobic'] for aa in self.amino_acids])
        self.charges = np.array([properties[aa]['charge'] for aa in self.amino_acids])
        self.is_polar = np.array([properties[aa]['polar'] for aa in self.amino_acids], dtype=float)
        self.volumes = np.array([properties[aa]['volume'] for aa in self.amino_acids])
        self.ez_values = np.array([properties[aa]['ez'] for aa in self.amino_acids])
        self.interface_pref = np.array([interface_preference.get(aa, 0.0) for aa in self.amino_acids])
        
        # Z-score normalization for all properties
        def z_score_normalize(array):
            mean = np.mean(array)
            std = np.std(array, ddof=1) if len(array) > 1 else 1.0
            if std == 0:
                std = 1.0
            return (array - mean) / std, mean, std
        
        self.normalized_helix_prop, self.mean_helix_prop, self.std_helix_prop = z_score_normalize(self.helix_prop)
        self.normalized_hydrophobic, self.mean_hydrophobic, self.std_hydrophobic = z_score_normalize(self.hydrophobic)
        self.normalized_charges, self.mean_charges, self.std_charges = z_score_normalize(self.charges)
        self.normalized_is_polar, self.mean_is_polar, self.std_is_polar = z_score_normalize(self.is_polar)
        self.normalized_volumes, self.mean_volumes, self.std_volumes = z_score_normalize(self.volumes)
        self.normalized_ez_values, self.mean_ez_values, self.std_ez_values = z_score_normalize(self.ez_values)
        self.normalized_interface_pref, self.mean_interface_pref, self.std_interface_pref = z_score_normalize(self.interface_pref)
        
        self.h_alpha = self.normalized_helix_prop

    def _projector_terms_for_code(self, position: int, code: int, base_coeff: float):
        """Projects onto specific amino acid codes using Pauli operators"""
        b = self.bits_per_pos
        s = []
        for k in range(b):
            v_k = (code >> k) & 1
            s.append(1.0 if v_k == 0 else -1.0)
        num_subsets = 1 << b
        for mask in range(num_subsets):
            coeff = base_coeff / (2 ** b)
            pauli = ['I'] * self.n_qubits
            for k in range(b):
                if (mask >> k) & 1:
                    coeff *= s[k]
                    w = get_qubit_index(position, k, self.bits_per_pos)
                    pauli[w] = 'Z'
            if abs(coeff) > 1e-10:
                self.pauli_terms.append((coeff, ''.join(pauli)))

    def _normalize_pauli_terms(self):
        """Normalize the coefficients in pauli_terms using z-score"""
        if not self.pauli_terms:
            return
        coeffs = np.array([term[0] for term in self.pauli_terms if abs(term[0]) > 1e-10])
        if len(coeffs) == 0:
            return
        mean_coeff = np.mean(coeffs)
        std_coeff = np.std(coeffs, ddof=1) if len(coeffs) > 1 else 1.0
        if std_coeff == 0:
            std_coeff = 1.0
        normalized_coeffs = (coeffs - mean_coeff) / std_coeff
        idx = 0
        new_pauli_terms = []
        for coeff, pauli_string in self.pauli_terms:
            if abs(coeff) > 1e-10:
                new_pauli_terms.append((normalized_coeffs[idx], pauli_string))
                idx += 1
            else:
                new_pauli_terms.append((coeff, pauli_string))
        self.pauli_terms = new_pauli_terms
        self.mean_coeff = mean_coeff
        self.std_coeff = std_coeff

    def _get_membrane_environment(self, position: int):
        """Determines if position faces membrane or water"""
        phase_deg = self.kwargs.get('wheel_phase_deg', 0.0)
        halfwidth_deg = self.kwargs.get('wheel_halfwidth_deg', 90.0)
        membrane_angle = (position * 100.0 - phase_deg) % 360.0  # Changed: - phase_deg
        if membrane_angle > 180.0: 
            membrane_angle -= 360.0
        faces_membrane = abs(membrane_angle) <= halfwidth_deg
        return "membrane" if faces_membrane else "water"
    
    def _is_interface_region(self, position: int):
        """Determines if position is in membrane-water interface"""
        membrane_angle = (position * 100.0 + self.kwargs.get('wheel_phase_deg', 0.0)) % 360.0
        if membrane_angle > 180.0: 
            membrane_angle -= 360.0
        interface_width = 30.0
        membrane_width = self.kwargs.get('wheel_halfwidth_deg', 90.0)
        return abs(abs(membrane_angle) - membrane_width) <= interface_width

    def _add_local_terms(self, weight: float):
        """Enhanced local terms using normalized helix propensity"""
        for i in range(self.L):
            for α in range(self.n_aa):
                base = -weight * self.normalized_helix_prop[α]
                self._projector_terms_for_code(i, α, base)
        self._normalize_pauli_terms()

    def _add_depth_dependent_terms(self, weight: float):
        """Adds normalized Ez potential terms"""
        print("\nDEBUG: Adding depth-dependent Ez terms (normalized)...")
        for i in range(self.L):
            faces_membrane = self._get_membrane_environment(i) == "membrane"
            if faces_membrane:
                for α in range(self.n_aa):
                    aa = self.amino_acids[α]
                    ez_penalty = weight * self.normalized_ez_values[α]
                    if abs(ez_penalty) > 1e-6:
                        self._projector_terms_for_code(i, α, ez_penalty)
                        print(f"  - Pos {i} ({aa}): Normalized Ez penalty = {ez_penalty:.3f}")
        self._normalize_pauli_terms()

    def _add_enhanced_environment_terms(self, weight: float):
        """Enhanced environment terms using normalized properties"""
        print("\nDEBUG: Término de entorno mejorado activado (normalizado).")
        print(f"DEBUG: wheel_halfwidth_deg = {self.kwargs.get('wheel_halfwidth_deg', 90.0)}")
        for i in range(self.L):
            environment = self._get_membrane_environment(i)
            print(f"  - Pos {i}: entorno={environment}")
            for α in range(self.n_aa):
                aa = self.amino_acids[α]
                hydrophobic_score = self.normalized_hydrophobic[α]
                is_polar = self.normalized_is_polar[α]
                is_charged = self.normalized_charges[α]
                env_bonus = 0.0
                if environment == "membrane":
                    if is_polar < 0 and is_charged == 0 and hydrophobic_score > 0:
                        env_bonus = -weight * hydrophobic_score * 100.0
                    if is_polar > 0 or is_charged != 0:
                        env_bonus += weight * abs(hydrophobic_score) * 100.0
                else:
                    if is_polar > 0 or is_charged != 0:
                        env_bonus = -weight * abs(hydrophobic_score) * 100.0
                    if hydrophobic_score > 0 and is_polar < 0 and is_charged == 0:
                        env_bonus += weight * hydrophobic_score * 100.0
                if abs(env_bonus) > 1e-6:
                    self._projector_terms_for_code(i, α, env_bonus)
                    print(f"    -> Added normalized env_bonus = {env_bonus:.3f} for {aa}")
        self._normalize_pauli_terms()

    def _add_membrane_interface_terms(self, weight: float):
        """Special terms for membrane-water interface using normalized properties"""
        print("\nDEBUG: Adding membrane interface terms (normalized)...")
        for i in range(self.L):
            is_interface = self._is_interface_region(i)
            if is_interface:
                for α in range(self.n_aa):
                    aa = self.amino_acids[α]
                    if abs(self.normalized_interface_pref[α]) > 1e-6:
                        bonus = weight * self.normalized_interface_pref[α]
                        self._projector_terms_for_code(i, α, bonus)
                        print(f"  - Pos {i} ({aa}): normalized interface bonus = {bonus:.3f}")
        self._normalize_pauli_terms()

    def _add_helical_periodicity_terms(self, weight: float):
        """Enhanced helical periodicity using normalized properties"""
        print("\nDEBUG: Adding enhanced helical periodicity terms (normalized)...")
        for i in range(self.L):
            for offset in [3, 4]:
                j = i + offset
                if j >= self.L:
                    continue
                phase = (offset * 100.0) % 360.0
                same_face = (offset == 3)
                for α in range(self.n_aa):
                    for β in range(self.n_aa):
                        if same_face:
                            if self.normalized_hydrophobic[α] > 0 and self.normalized_hydrophobic[β] > 0:
                                coupling = -weight * 0.3
                            elif (self.normalized_is_polar[α] > 0 and self.normalized_is_polar[β] > 0) or \
                                 (self.normalized_charges[α] != 0 and self.normalized_charges[β] != 0):
                                coupling = -weight * 0.2
                            else:
                                coupling = weight * 0.1
                        else:
                            coupling = -weight * 0.1 * abs(self.normalized_hydrophobic[α] * self.normalized_hydrophobic[β]) / 10.0
                        if abs(coupling) > 1e-6:
                            self._add_pairwise_coupling_term(i, j, α, β, coupling)
        self._normalize_pauli_terms()

    def _add_electrostatic_screening_terms(self, weight: float):
        """Electrostatic interactions using normalized charges"""
        print("\nDEBUG: Adding electrostatic screening terms (normalized)...")
        for i in range(self.L):
            for j in range(i + 1, self.L):
                if abs(i - j) > 8:
                    continue
                env_i = self._get_membrane_environment(i)
                env_j = self._get_membrane_environment(j)
                if env_i == "membrane" and env_j == "membrane":
                    dielectric_factor = 0.1
                elif env_i == "water" and env_j == "water":
                    dielectric_factor = 1.0
                else:
                    dielectric_factor = 0.3
                distance_decay = 1.0 / (1.0 + abs(i - j))
                for α in range(self.n_aa):
                    for β in range(self.n_aa):
                        q_i = self.normalized_charges[α]
                        q_j = self.normalized_charges[β]
                        if q_i != 0 and q_j != 0:
                            coupling = weight * q_i * q_j * distance_decay / dielectric_factor
                            if abs(coupling) > 1e-6:
                                self._add_pairwise_coupling_term(i, j, α, β, coupling)
        self._normalize_pauli_terms()

    def _add_enhanced_hydrophobic_moment(self, weight: float):
        """Enhanced hydrophobic moment using normalized properties"""
        print("\nDEBUG: Adding enhanced hydrophobic moment terms (normalized)...")
        phi = np.deg2rad(100.0)
        phase_deg = self.kwargs.get('wheel_phase_deg', 0.0)
        phase_rad = np.deg2rad(phase_deg)
        for i in range(self.L):
            for j in range(self.L):
                if i == j or abs(i - j) > 7:
                    continue
                angle_i = phi * i
                angle_j = phi * j
                alignment = np.cos(angle_i - angle_j)
                # Clustering term (existing, rotation-invariant)
                for α in range(self.n_aa):
                    for β in range(self.n_aa):
                        h_i = max(0, self.normalized_hydrophobic[α])
                        h_j = max(0, self.normalized_hydrophobic[β])
                        coupling_cluster = -weight * h_i * h_j * alignment / (self.L * 2.0)
                        if abs(coupling_cluster) > 1e-6:
                            self._add_pairwise_coupling_term(i, j, α, β, coupling_cluster)
                # Orientation term: biases cluster toward membrane direction (phase)
                # cos(angle_i + angle_j - 2*phase) encourages average angle ~ phase
                alignment_ori = np.cos(angle_i + angle_j - 2 * phase_rad)
                for α in range(self.n_aa):
                    for β in range(self.n_aa):
                        h_i = max(0, self.normalized_hydrophobic[α])
                        h_j = max(0, self.normalized_hydrophobic[β])
                        coupling_ori = -weight * h_i * h_j * alignment_ori / (self.L * 2.0)  # /2 from expansion, but tune if needed
                        if abs(coupling_ori) > 1e-6:
                            self._add_pairwise_coupling_term(i, j, α, β, coupling_ori)
        self._normalize_pauli_terms()

    def _add_amphipathic_segregation_terms(self, weight: float):
        """Amphipathic segregation using normalized properties"""
        if weight == 0.0:
            return
        print("Adding amphipathic segregation terms (normalized)...")
        membrane_positions = [i for i in range(self.L) if self._get_membrane_environment(i) == "membrane"]
        water_positions = [i for i in range(self.L) if self._get_membrane_environment(i) != "membrane"]
        for i in membrane_positions:
            for j in water_positions:
                for α in range(self.n_aa):
                    for β in range(self.n_aa):
                        if self.normalized_hydrophobic[α] > 0 and \
                           (self.normalized_is_polar[β] > 0 or self.normalized_charges[β] != 0):
                            coupling = -weight * 0.5
                            self._add_pairwise_coupling_term(i, j, α, β, coupling)
                        elif (self.normalized_is_polar[α] > 0 or self.normalized_charges[α] != 0) and \
                             self.normalized_hydrophobic[β] > 0:
                            coupling = weight * 0.5
                            self._add_pairwise_coupling_term(i, j, α, β, coupling)
        self._normalize_pauli_terms()

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
                base_coeff = -base_coeff
        if abs(base_coeff) > 1e-10:
            self.pauli_terms.append((base_coeff, ''.join(pauli)))

    def _add_miyazawa_jernigan_terms(self, weight: float, max_dist: int):
        """Miyazawa-Jernigan terms with normalized interaction energies"""
        mj_interaction, list_aa = _load_energy_matrix_file()
        aa_to_idx = {aa: i for i, aa in enumerate(list_aa)}
        mj_interaction_flat = mj_interaction[np.triu_indices_from(mj_interaction, k=1)]
        mean_mj = np.mean(mj_interaction_flat)
        std_mj = np.std(mj_interaction_flat, ddof=1) if len(mj_interaction_flat) > 1 else 1.0
        if std_mj == 0:
            std_mj = 1.0
        normalized_mj = (mj_interaction - mean_mj) / std_mj
        for i in range(self.L):
            for j in range(i + 1, self.L):
                if abs(i - j) <= max_dist:
                    for α in range(self.n_aa):
                        for β in range(self.n_aa):
                            aa_i = self.amino_acids[α]
                            aa_j = self.amino_acids[β]
                            mj_idx_i = aa_to_idx[aa_i]
                            mj_idx_j = aa_to_idx[aa_j]
                            interaction_energy = normalized_mj[min(mj_idx_i, mj_idx_j), max(mj_idx_i, mj_idx_j)]
                            if not np.isclose(interaction_energy, 0.0):
                                self._add_pairwise_coupling_term(i, j, α, β, weight * interaction_energy)
        self._normalize_pauli_terms()

    def _add_helix_first_neighbors_terms(self, weight: float):
        """Helix first neighbors terms with normalized propensities"""
        first_matrix, list_aa = _load_first_neighbors_matrix_file()
        aa_to_idx = {aa: i for i, aa in enumerate(list_aa)}
        mean_first = np.mean(first_matrix)
        std_first = np.std(first_matrix, ddof=1) if first_matrix.size > 1 else 1.0
        if std_first == 0:
            std_first = 1.0
        normalized_first = (first_matrix - mean_first) / std_first
        print("\nDEBUG: Adding helix first neighbors terms (normalized)...")
        for i in range(self.L - 1):
            for α in range(self.n_aa):
                for β in range(self.n_aa):
                    aa_i = self.amino_acids[α]
                    aa_j = self.amino_acids[β]
                    idx_i = aa_to_idx[aa_i]
                    idx_j = aa_to_idx[aa_j]
                    interaction_prop = normalized_first[idx_i, idx_j]
                    if not np.isclose(interaction_prop, 0.0):
                        coupling = weight * interaction_prop
                        self._add_pairwise_coupling_term(i, i + 1, α, β, coupling)
                        print(f"  - Pos {i} & {i+1} ({aa_i}, {aa_j}): normalized coupling = {coupling:.3f}")
        self._normalize_pauli_terms()

    def _add_helix_third_neighbors_terms(self, weight: float):
        """Helix third neighbors terms with normalized propensities"""
        third_matrix, list_aa = _load_third_neighbors_matrix_file()
        aa_to_idx = {aa: i for i, aa in enumerate(list_aa)}
        mean_third = np.mean(third_matrix)
        std_third = np.std(third_matrix, ddof=1) if third_matrix.size > 1 else 1.0
        if std_third == 0:
            std_third = 1.0
        normalized_third = (third_matrix - mean_third) / std_third
        print("\nDEBUG: Adding helix third neighbors terms (normalized)...")
        for i in range(self.L - 3):
            for α in range(self.n_aa):
                for β in range(self.n_aa):
                    aa_i = self.amino_acids[α]
                    aa_j = self.amino_acids[β]
                    idx_i = aa_to_idx[aa_i]
                    idx_j = aa_to_idx[aa_j]
                    interaction_prop = normalized_third[idx_i, idx_j]
                    if not np.isclose(interaction_prop, 0.0):
                        coupling = weight * interaction_prop
                        self._add_pairwise_coupling_term(i, i + 3, α, β, coupling)
                        print(f"  - Pos {i} & {i+3} ({aa_i}, {aa_j}): normalized coupling = {coupling:.3f}")
        self._normalize_pauli_terms()

    def _add_helix_fourth_neighbors_terms(self, weight: float):
        """Helix fourth neighbors terms with normalized propensities"""
        fourth_matrix, list_aa = _load_fourth_neighbors_matrix_file()
        aa_to_idx = {aa: i for i, aa in enumerate(list_aa)}
        mean_fourth = np.mean(fourth_matrix)
        std_fourth = np.std(fourth_matrix, ddof=1) if fourth_matrix.size > 1 else 1.0
        if std_fourth == 0:
            std_fourth = 1.0
        normalized_fourth = (fourth_matrix - mean_fourth) / std_fourth
        print("\nDEBUG: Adding helix fourth neighbors terms (normalized)...")
        for i in range(self.L - 4):
            for α in range(self.n_aa):
                for β in range(self.n_aa):
                    aa_i = self.amino_acids[α]
                    aa_j = self.amino_acids[β]
                    idx_i = aa_to_idx[aa_i]
                    idx_j = aa_to_idx[aa_j]
                    interaction_prop = normalized_fourth[idx_i, idx_j]
                    if not np.isclose(interaction_prop, 0.0):
                        coupling = weight * interaction_prop
                        self._add_pairwise_coupling_term(i, i + 4, α, β, coupling)
                        print(f"  - Pos {i} & {i+4} ({aa_i}, {aa_j}): normalized coupling = {coupling:.3f}")
        self._normalize_pauli_terms()

    def _add_membrane_charge_term(self, weight: float):
        """Membrane charge interaction with normalized charges"""
        membrane_charge = self.kwargs.get('membrane_charge', 'neu')
        charge_sign = -1.0 if membrane_charge == 'neg' else (1.0 if membrane_charge == 'pos' else 0.0)
        for i in range(self.L):
            faces_membrane = self._get_membrane_environment(i) == "membrane"
            if faces_membrane:
                for α in range(self.n_aa):
                    penalty = weight * charge_sign * self.normalized_charges[α]
                    if abs(penalty) > 1e-6:
                        self._projector_terms_for_code(i, α, penalty)
        self._normalize_pauli_terms()

    def _add_hydrophobic_moment_terms(self, weight: float):
        """Hydrophobic moment terms with normalized hydrophobicity"""
        if self.bits_per_pos > 3:
            return
        phi = np.deg2rad(100.0)
        for i in range(self.L):
            for j in range(self.L):
                if i == j or abs(i - j) > 7:
                    continue
                cos_term = np.cos(phi * (j - i))
                for α in range(self.n_aa):
                    for β in range(self.n_aa):
                        h_i = self.normalized_hydrophobic[α]
                        h_j = self.normalized_hydrophobic[β]
                        coupling = -weight * (h_i * h_j) * cos_term
                        self._add_pairwise_coupling_term(i, j, α, β, coupling)
        self._normalize_pauli_terms()

    def _add_invalid_code_penalties(self, weight: float):
        """Penalize invalid amino acid codes"""
        max_code = (1 << self.bits_per_pos) - 1
        if self.n_aa - 1 == max_code:
            return
        for i in range(self.L):
            for code in range(self.n_aa, max_code + 1):
                self._projector_terms_for_code(i, code, weight)
        self._normalize_pauli_terms()

    def build_hamiltonian(self, backend: str):
        """Build the complete enhanced Hamiltonian with normalized terms"""
        print("Building enhanced quantum Hamiltonian for amphipathic helix formation (normalized)...")
        
        self._add_local_terms(weight=self.kwargs.get('lambda_local', 1.0))
        self._add_miyazawa_jernigan_terms(
            weight=self.kwargs.get('lambda_pairwise', 1.0),
            max_dist=self.kwargs.get('max_interaction_dist', 3)
        )
        if self.kwargs.get('lambda_env', 0.0) > 0.0:
            self._add_enhanced_environment_terms(self.kwargs.get('lambda_env', 0.0))
        if self.kwargs.get('lambda_ez', 0.0) > 0.0:
            self._add_depth_dependent_terms(self.kwargs.get('lambda_ez', 0.0))
        if self.kwargs.get('lambda_interface', 0.0) > 0.0:
            self._add_membrane_interface_terms(self.kwargs.get('lambda_interface', 0.0))
        if self.kwargs.get('lambda_segregation', 0.0) > 0.0:
            self._add_amphipathic_segregation_terms(self.kwargs.get('lambda_segregation', 0.0))
        if self.kwargs.get('lambda_charge', 0.0) > 0.0:
            self._add_membrane_charge_term(self.kwargs.get('lambda_charge', 0.0))
        if self.kwargs.get('lambda_periodicity', 0.0) > 0.0:
            self._add_helical_periodicity_terms(self.kwargs.get('lambda_periodicity', 0.0))
        if self.kwargs.get('lambda_electrostatic', 0.0) > 0.0:
            self._add_electrostatic_screening_terms(self.kwargs.get('lambda_electrostatic', 0.0))
        if self.kwargs.get('lambda_mu', 0.0) > 0.0:
            if self.kwargs.get('use_enhanced_mu', True):
                print("Adding enhanced hydrophobic moment terms (normalized)...")
                self._add_enhanced_hydrophobic_moment(self.kwargs.get('lambda_mu', 0.0))
            else:
                print("Adding standard hydrophobic moment terms (normalized)...")
                self._add_hydrophobic_moment_terms(self.kwargs.get('lambda_mu', 0.0))
        if self.kwargs.get('lambda_helix_first', 0.0) > 0.0:
            self._add_helix_first_neighbors_terms(self.kwargs.get('lambda_helix_first', 0.0))
        if self.kwargs.get('lambda_helix_third', 0.0) > 0.0:
            self._add_helix_third_neighbors_terms(self.kwargs.get('lambda_helix_third', 0.0))
        if self.kwargs.get('lambda_helix_fourth', 0.0) > 0.0:
            self._add_helix_fourth_neighbors_terms(self.kwargs.get('lambda_helix_fourth', 0.0))
        self._add_invalid_code_penalties(weight=5.0)
        
        self._normalize_pauli_terms()
        
        print(f"\nEnhanced Hamiltonian built with {len(self.pauli_terms)} Pauli terms")
        coeffs = [term[0] for term in self.pauli_terms]
        print(f"DEBUG: Total terms: {len(coeffs)}")
        print(f"DEBUG: Normalized coefficient range: [{min(coeffs):.3f}, {max(coeffs):.3f}]")
        
        if backend == 'pennylane':
            observables = []
            for coeff, pauli_string in self.pauli_terms:
                obs_list = []
                for i, pauli in enumerate(pauli_string):
                    if pauli == 'Z':
                        obs_list.append(qml.PauliZ(i))
                if obs_list:
                    if len(obs_list) == 1:
                        observables.append(obs_list[0])
                    else:
                        observables.append(qml.prod(*obs_list))
                else:
                    observables.append(qml.Identity(0))
            hamiltonian = qml.Hamiltonian(coeffs, observables)
            print(f"PennyLane Hamiltonian created with {len(coeffs)} terms")
            return self.pauli_terms, hamiltonian
        
        return self.pauli_terms, None

    def get_energy_summary(self):
        """Returns a summary of energy contributions with normalized values"""
        coeffs = np.array([term[0] for term in self.pauli_terms if abs(term[0]) > 1e-10])
        if len(coeffs) > 0:
            mean_coeff = getattr(self, 'mean_coeff', np.mean(coeffs))
            std_coeff = getattr(self, 'std_coeff', np.std(coeffs, ddof=1) if len(coeffs) > 1 else 1.0)
            if std_coeff == 0:
                std_coeff = 1.0
            normalized_coeffs = (coeffs - mean_coeff) / std_coeff
            coeff_range = [np.min(coeffs), np.max(coeffs)]
            normalized_coeff_range = [np.min(normalized_coeffs), np.max(normalized_coeffs)]
        else:
            mean_coeff = 0.0
            std_coeff = 1.0
            normalized_coeffs = np.array([])
            coeff_range = [0.0, 0.0]
            normalized_coeff_range = [0.0, 0.0]
        
        active_terms = {}
        if self.kwargs.get('lambda_local', 0.0) > 0:
            active_terms['local'] = self.kwargs.get('lambda_local', 0.0)
        if self.kwargs.get('lambda_pairwise', 0.0) > 0:
            active_terms['miyazawa_jernigan'] = self.kwargs.get('lambda_pairwise', 0.0)
        if self.kwargs.get('lambda_env', 0.0) > 0:
            active_terms['environment'] = self.kwargs.get('lambda_env', 0.0)
        if self.kwargs.get('lambda_ez', 0.0) > 0:
            active_terms['depth_dependent'] = self.kwargs.get('lambda_ez', 0.0)
        if self.kwargs.get('lambda_interface', 0.0) > 0:
            active_terms['membrane_interface'] = self.kwargs.get('lambda_interface', 0.0)
        if self.kwargs.get('lambda_segregation', 0.0) > 0:
            active_terms['amphipathic_segregation'] = self.kwargs.get('lambda_segregation', 0.0)
        if self.kwargs.get('lambda_charge', 0.0) > 0:
            active_terms['membrane_charge'] = self.kwargs.get('lambda_charge', 0.0)
        if self.kwargs.get('lambda_periodicity', 0.0) > 0:
            active_terms['helical_periodicity'] = self.kwargs.get('lambda_periodicity', 0.0)
        if self.kwargs.get('lambda_electrostatic', 0.0) > 0:
            active_terms['electrostatic_screening'] = self.kwargs.get('lambda_electrostatic', 0.0)
        if self.kwargs.get('lambda_mu', 0.0) > 0:
            active_terms['hydrophobic_moment'] = self.kwargs.get('lambda_mu', 0.0)
        if self.kwargs.get('lambda_helix_first', 0.0) > 0:
            active_terms['helix_first_neighbors'] = self.kwargs.get('lambda_helix_first', 0.0)
        if self.kwargs.get('lambda_helix_third', 0.0) > 0:
            active_terms['helix_third_neighbors'] = self.kwargs.get('lambda_helix_third', 0.0)
        if self.kwargs.get('lambda_helix_fourth', 0.0) > 0:
            active_terms['helix_fourth_neighbors'] = self.kwargs.get('lambda_helix_fourth', 0.0)
        
        if active_terms:
            weights = np.array(list(active_terms.values()))
            mean_weights = np.mean(weights)
            std_weights = np.std(weights, ddof=1) if len(weights) > 1 else 1.0
            if std_weights == 0:
                std_weights = 1.0
            normalized_active_terms = {key: (value - mean_weights) / std_weights for key, value in active_terms.items()}
        else:
            normalized_active_terms = {}
        
        summary = {
            'total_terms': len(self.pauli_terms),
            'coefficient_range': coeff_range,
            'normalized_coefficient_range': normalized_coeff_range.tolist(),
            'mean_coefficient': mean_coeff,
            'std_coefficient': std_coeff,
            'active_terms': active_terms,
            'normalized_active_terms': normalized_active_terms,
            'amino_acid_properties': {
                'helix_prop': {'original': self.helix_prop.tolist(), 'normalized': self.normalized_helix_prop.tolist(), 'mean': self.mean_helix_prop, 'std': self.std_helix_prop},
                'hydrophobic': {'original': self.hydrophobic.tolist(), 'normalized': self.normalized_hydrophobic.tolist(), 'mean': self.mean_hydrophobic, 'std': self.std_hydrophobic},
                'charges': {'original': self.charges.tolist(), 'normalized': self.normalized_charges.tolist(), 'mean': self.mean_charges, 'std': self.std_charges},
                'is_polar': {'original': self.is_polar.tolist(), 'normalized': self.normalized_is_polar.tolist(), 'mean': self.mean_is_polar, 'std': self.std_is_polar},
                'volumes': {'original': self.volumes.tolist(), 'normalized': self.normalized_volumes.tolist(), 'mean': self.mean_volumes, 'std': self.std_volumes},
                'ez_values': {'original': self.ez_values.tolist(), 'normalized': self.normalized_ez_values.tolist(), 'mean': self.mean_ez_values, 'std': self.std_ez_values},
                'interface_pref': {'original': self.interface_pref.tolist(), 'normalized': self.normalized_interface_pref.tolist(), 'mean': self.mean_interface_pref, 'std': self.std_interface_pref}
            }
        }
        
        return summary