# core/hamiltonian_builder.py
import numpy as np
import pennylane as qml
from typing import List, Tuple, Optional
from utils.general_utils import get_qubit_index
from data_loaders.energy_matrix_loader import _load_energy_matrix_file, _load_helix_pairs_matrix_file

class HamiltonianBuilder:
    """
    Builds the quantum Hamiltonian (QUBO) for the protein design problem.
    Enhanced for amphipathic helix formation in membranes.
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
        """Enhanced local terms that favor helix-forming residues"""
        for i in range(self.L):
            for α in range(self.n_aa):
                # Enhanced helix propensity bonus
                base = -weight * self.h_alpha[α] * 1.5  # Increased weight for helix formation
                self._projector_terms_for_code(i, α, base)
    
    def _get_helix_face(self, position: int) -> str:
        """
        Determine which face of the helix a position is on.
        Assumes ideal alpha-helix geometry (100° per residue).
        """
        angle = (position * 100.0) % 360.0
        
        # Define faces based on helix wheel
        # Hydrophobic face: ~0-120° and ~240-360°
        # Polar face: ~120-240°
        if (angle <= 120.0) or (angle >= 240.0):
            return "hydrophobic"
        else:
            return "polar"
    
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
    
    def _add_enhanced_environment_terms(self, weight: float):
        """
        Enhanced environment terms that properly favor amphipathic helix formation.
        In wheel mode: positions "in membrane" get hydrophobic preference only if on hydrophobic face.
        """
        print("\nDEBUG: Término de entorno mejorado activado.")
        print(f"DEBUG: wheel_halfwidth_deg = {self.kwargs.get('wheel_halfwidth_deg', 90.0)}")
        
        for i in range(self.L):
            # Calculate position angle for membrane determination
            membrane_angle = (i * 100.0 + self.kwargs.get('wheel_phase_deg', 0.0)) % 360.0
            if membrane_angle > 180.0: 
                membrane_angle -= 360.0
            
            # Calculate helix face
            helix_face = self._get_helix_face(i)
            
            # Determine environment preference
            # In wheel mode, "membrane" means the hydrophobic face should contact lipids
            membrane_contact = abs(membrane_angle) <= self.kwargs.get('wheel_halfwidth_deg', 90.0)
            
            print(f"  - Pos {i}: ángulo={membrane_angle:.1f}°, cara={helix_face}, contacto_membrana={membrane_contact}")
            
            for α in range(self.n_aa):
                aa = self.amino_acids[α]
                hydrophobic_score = self.hydrophobic[α]
                is_polar = self.is_polar[α]
                is_charged = self.charges[α] != 0
                
                env_bonus = 0.0
                
                print(f"    - AA {aa}: hydrophobic={hydrophobic_score:.2f}, polar={is_polar}, charged={is_charged}")
                
                if membrane_contact:
                    # This side of helix contacts the membrane
                    if helix_face == "hydrophobic":
                        # Hydrophobic face contacting membrane - STRONGLY favor hydrophobic residues
                        if hydrophobic_score > 2.0:  # Very hydrophobic (V=4.2, I=4.5, L=3.8)
                            env_bonus = -weight * 3.0  # Strong bonus
                            print(f"      -> STRONG bonus for very hydrophobic {aa} on hydrophobic face in membrane")
                        elif hydrophobic_score > 0.5:  # Moderately hydrophobic (A=1.8, C=2.5, M=1.9, F=2.8)
                            env_bonus = -weight * 2.0  # Good bonus
                            print(f"      -> Good bonus for hydrophobic {aa} on hydrophobic face in membrane")
                        elif is_charged:  # Charged residues in membrane contact
                            env_bonus = weight * 4.0  # VERY strong penalty
                            print(f"      -> VERY STRONG penalty for charged {aa} on hydrophobic face in membrane")
                        elif is_polar:  # Polar but uncharged (S=-0.8, N=-3.5, Q=-3.5)
                            env_bonus = weight * 2.5  # Strong penalty
                            print(f"      -> Strong penalty for polar {aa} on hydrophobic face in membrane")
                    else:
                        # Polar face contacting membrane - moderate penalties
                        if is_charged:
                            env_bonus = weight * 2.0  # Charged in membrane contact
                            print(f"      -> Penalty for charged {aa} on polar face in membrane")
                        elif is_polar:
                            env_bonus = weight * 1.0  # Polar in membrane contact
                            print(f"      -> Small penalty for polar {aa} on polar face in membrane")
                        elif hydrophobic_score > 1.0:
                            env_bonus = weight * 0.5  # Hydrophobic on wrong face
                            print(f"      -> Small penalty for hydrophobic {aa} on polar face in membrane")
                else:
                    # This side of helix faces water/cytoplasm
                    if helix_face == "polar":
                        # Polar face facing water - STRONGLY favor polar/charged residues
                        if is_charged:
                            env_bonus = -weight * 3.0  # Strong bonus for charged
                            print(f"      -> STRONG bonus for charged {aa} on polar face in water")
                        elif is_polar:
                            env_bonus = -weight * 2.5  # Strong bonus for polar
                            print(f"      -> Strong bonus for polar {aa} on polar face in water")
                        elif hydrophobic_score > 2.0:
                            env_bonus = weight * 3.0  # Strong penalty for very hydrophobic in water
                            print(f"      -> STRONG penalty for very hydrophobic {aa} on polar face in water")
                        elif hydrophobic_score > 0.0:
                            env_bonus = weight * 1.5  # Penalty for hydrophobic in water
                            print(f"      -> Penalty for hydrophobic {aa} on polar face in water")
                    else:
                        # Hydrophobic face facing water - penalize hydrophobic residues
                        if hydrophobic_score > 2.0:
                            env_bonus = weight * 2.5  # Strong penalty
                            print(f"      -> Strong penalty for very hydrophobic {aa} on hydrophobic face in water")
                        elif hydrophobic_score > 0.5:
                            env_bonus = weight * 1.8  # Moderate penalty
                            print(f"      -> Penalty for hydrophobic {aa} on hydrophobic face in water")
                        elif is_charged:
                            env_bonus = -weight * 1.0  # Bonus for charged
                            print(f"      -> Bonus for charged {aa} on hydrophobic face in water")
                        elif is_polar:
                            env_bonus = -weight * 0.8  # Bonus for polar
                            print(f"      -> Bonus for polar {aa} on hydrophobic face in water")
                
                if abs(env_bonus) > 1e-6:
                    self._projector_terms_for_code(i, α, env_bonus)
                    print(f"      -> Added env_bonus = {env_bonus:.3f}")
    
    def _add_amphipathic_coupling_terms(self, weight: float):
        """
        Add terms that couple adjacent positions to encourage proper amphipathic arrangement.
        """
        if self.bits_per_pos > 3 or weight == 0.0:
            return
            
        print("Adding amphipathic coupling terms...")
        
        for i in range(self.L - 1):
            face_i = self._get_helix_face(i)
            face_j = self._get_helix_face(i + 1)
            
            # If adjacent positions are on same face, encourage similar properties
            if face_i == face_j:
                for α in range(self.n_aa):
                    for β in range(self.n_aa):
                        if face_i == "hydrophobic":
                            # Both should be hydrophobic
                            if self.hydrophobic[α] > 0.5 and self.hydrophobic[β] > 0.5:
                                coupling = -weight * 0.2  # Bonus
                            elif self.is_polar[α] and self.is_polar[β]:
                                coupling = weight * 0.3  # Penalty for both polar
                            else:
                                coupling = 0.0
                        else:  # polar face
                            # Both should be polar/charged
                            if (self.is_polar[α] or self.charges[α] != 0) and \
                               (self.is_polar[β] or self.charges[β] != 0):
                                coupling = -weight * 0.2  # Bonus
                            elif self.hydrophobic[α] > 0.5 and self.hydrophobic[β] > 0.5:
                                coupling = weight * 0.3  # Penalty for both hydrophobic
                            else:
                                coupling = 0.0
                        
                        if abs(coupling) > 1e-6:
                            self._add_pairwise_coupling_term(i, i + 1, α, β, coupling)
    
    def _add_pairwise_coupling_term(self, i: int, j: int, α: int, β: int, base_coeff: float):
        """Helper to add a pairwise coupling term between positions i and j for amino acids α and β"""
        b = self.bits_per_pos
        s_i = [1.0 if ((α >> k) & 1) == 0 else -1.0 for k in range(b)]
        s_j = [1.0 if ((β >> k) & 1) == 0 else -1.0 for k in range(b)]
        
        for mask_i in range(1 << b):
            for mask_j in range(1 << b):
                coeff = base_coeff * (1.0 / (2 ** (2*b)))
                pauli = ['I'] * self.n_qubits
                for k in range(b):
                    if (mask_i >> k) & 1:
                        coeff *= s_i[k]
                        pauli[get_qubit_index(i, k, self.bits_per_pos)] = 'Z'
                    if (mask_j >> k) & 1:
                        coeff *= s_j[k]
                        pauli[get_qubit_index(j, k, self.bits_per_pos)] = 'Z'
                self.pauli_terms.append((coeff, ''.join(pauli)))
    
    def _add_pairwise_terms(self, weight: float):
        """Enhanced pairwise interactions"""
        if self.bits_per_pos > 3: return
        for i in range(self.L):
            for j in range(i+1, self.L):
                if abs(i-j) <= 3:
                    for α in range(self.n_aa):
                        for β in range(self.n_aa):
                            interaction = 0.0
                            
                            # Enhanced interaction rules
                            if self.hydrophobic[α] > 1.0 and self.hydrophobic[β] > 1.0:
                                interaction = -0.15  # Stronger hydrophobic clustering
                            elif self.charges[α] != 0 and self.charges[β] != 0:
                                if self.charges[α] * self.charges[β] > 0:
                                    interaction = 0.25  # Electrostatic repulsion
                                else:
                                    interaction = -0.15  # Salt bridge attraction
                            elif (self.is_polar[α] and self.is_polar[β]) and \
                                 (self.charges[α] == 0 and self.charges[β] == 0):
                                interaction = -0.05  # Weak polar-polar attraction
                            
                            if interaction != 0:
                                self._add_pairwise_coupling_term(i, j, α, β, weight * interaction)
    
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
                                self._add_pairwise_coupling_term(i, j, α, β, weight * interaction_energy)
    
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

                            interaction_prop = helix_matrix[idx_i, idx_j]

                            if not np.isclose(interaction_prop, 0.0):
                                self._add_pairwise_coupling_term(i, j, α, β, weight * interaction_prop)
    
    def _add_membrane_charge_term(self, weight: float):
        """Enhanced membrane charge interaction"""
        membrane_charge = self.kwargs.get('membrane_charge', 'neu')
        charge_sign = -1.0 if membrane_charge == 'neg' else (1.0 if membrane_charge == 'pos' else 0.0)
        
        for i in range(self.L):
            if not self._pos_in_membrane(i): 
                continue
                
            helix_face = self._get_helix_face(i)
            
            for α in range(self.n_aa):
                # Only apply to positions that would be buried in membrane
                if helix_face == "hydrophobic":
                    penalty = weight * charge_sign * self.charges[α] * 2.0  # Stronger penalty
                else:
                    penalty = weight * charge_sign * self.charges[α] * 0.5  # Weaker on polar face
                    
                if abs(penalty) > 1e-6:
                    self._projector_terms_for_code(i, α, penalty)
    
    def _add_hydrophobic_moment_terms(self, weight: float):
        """Enhanced hydrophobic moment terms for amphipathic structures"""
        if self.bits_per_pos > 3: return
        phi = np.deg2rad(100.0)  # Ideal alpha-helix angle
        
        for i in range(self.L):
            for j in range(i, min(i + 7, self.L)):  # Consider local helical turn (~2 turns)
                cos_fac = np.cos(phi * (j - i))
                if np.isclose(cos_fac, 0.0): continue
                
                for α in range(self.n_aa):
                    for β in range(self.n_aa):
                        hij = self.hydrophobic[α] * self.hydrophobic[β]
                        if hij == 0: continue
                        
                        # Enhanced moment calculation
                        distance_factor = np.exp(-0.1 * abs(j - i))  # Decay with distance
                        base = -weight * hij * cos_fac * distance_factor
                        
                        self._add_pairwise_coupling_term(i, j, α, β, base)
    
    def _add_invalid_code_penalties(self, weight: float):
        max_code = (1 << self.bits_per_pos) - 1
        if self.n_aa - 1 == max_code: return
        for i in range(self.L):
            for code in range(self.n_aa, max_code + 1):
                self._projector_terms_for_code(i, code, weight)
    
    def build_hamiltonian(self, backend: str):
        print("Building enhanced quantum Hamiltonian for amphipathic helix formation...")
        
        # Enhanced local amino acid preferences
        self._add_local_terms(weight=self.kwargs.get('lambda_local', 1.0))
        
        # Miyazawa-Jernigan pairwise interactions
        print("Adding Miyazawa-Jernigan terms...")
        self._add_miyazawa_jernigan_terms(weight=self.kwargs.get('lambda_pairwise', 1.0),
                                         max_dist=self.kwargs.get('max_interaction_dist', 3))

        # Helix pair propensities
        if self.kwargs.get('lambda_helix_pairs', 0.0) != 0.0:
            print("Adding Helix Pair Propensity terms...")
            self._add_helix_pairs_terms(weight=self.kwargs.get('lambda_helix_pairs', 0.0),
                                        max_dist=self.kwargs.get('max_interaction_dist', 3))

        # Enhanced environment preference for amphipathic structures
        if self.kwargs.get('lambda_env', 0.0) != 0.0:
            self._add_enhanced_environment_terms(self.kwargs.get('lambda_env', 0.0))

        # New: Amphipathic coupling terms
        if self.kwargs.get('lambda_amphipathic', 0.0) != 0.0:
            print("Adding amphipathic coupling terms...")
            self._add_amphipathic_coupling_terms(self.kwargs.get('lambda_amphipathic', 0.0))

        # Enhanced membrane charge interaction
        if self.kwargs.get('lambda_charge', 0.0) != 0.0:
            self._add_membrane_charge_term(self.kwargs.get('lambda_charge', 0.0))

        # Enhanced hydrophobic moment
        if self.kwargs.get('lambda_mu', 0.0) != 0.0:
            print("Adding enhanced hydrophobic moment terms...")
            self._add_hydrophobic_moment_terms(self.kwargs.get('lambda_mu', 0.0))

        # Penalize invalid codes
        self._add_invalid_code_penalties(weight=20.0)
        
        print(f"\nEnhanced Hamiltonian built with {len(self.pauli_terms)} Pauli terms")
        
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