import numpy as np
import pennylane as qml
from typing import List, Tuple, Optional, Dict, Any
from utils.general_utils import get_qubit_index
from data_loaders.energy_matrix_loader import _load_first_neighbors_matrix_file, _load_third_neighbors_matrix_file, _load_fourth_neighbors_matrix_file, _load_energy_matrix_file
import gc

class HamiltonianBuilder:
    """
    Builds the quantum Hamiltonian (QUBO) for the protein design problem.
    FIXED VERSION: Properly rewards hydrophobic in membrane and polar in water.
    Supports both PennyLane and Qiskit backends.
    """
    def __init__(self, L: int, amino_acids: List[str], bits_per_pos: int, n_qubits: int, **kwargs):
        self.L = L
        self.amino_acids = amino_acids
        self.n_aa = len(amino_acids)
        self.bits_per_pos = bits_per_pos
        self.n_qubits = n_qubits
        self.kwargs = kwargs
        self.pauli_terms = []
        self.scale_factor = kwargs.get('scale_factor', 0.01)  # Added for numerical stability
        self._init_amino_acid_properties()
        self._initialize_mj_matrix()
        self._initialize_helix_pairs()
        
        print("\n" + "="*70)
        print("🧬 AMINO ACID PROPERTIES (RAW VALUES - NO NORMALIZATION)")
        print("="*70)
        for idx, aa in enumerate(self.amino_acids):
            print(f"{aa}: hydro={self.hydrophobic[idx]:6.2f} | "
                  f"polar={'YES' if self.is_polar[idx] else 'NO ':3s} | "
                  f"charge={self.charges[idx]:+2.0f} | "
                  f"helix={self.helix_prop[idx]:.2f}")
        print("="*70 + "\n")

    def _init_amino_acid_properties(self):
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
        
        self.helix_prop = np.array([properties[aa]['helix'] for aa in self.amino_acids])
        self.hydrophobic = np.array([properties[aa]['hydrophobic'] for aa in self.amino_acids])
        self.charges = np.array([properties[aa]['charge'] for aa in self.amino_acids])
        self.is_polar = np.array([properties[aa]['polar'] for aa in self.amino_acids], dtype=float)
        self.volumes = np.array([properties[aa]['volume'] for aa in self.amino_acids])
        self.ez_values = np.array([properties[aa]['ez'] for aa in self.amino_acids])

    def _initialize_mj_matrix(self):
        """Initialize simplified Miyazawa-Jernigan interaction matrix."""
        self.mj_matrix = np.zeros((self.n_aa, self.n_aa))
        hydrophobic = {'A', 'C', 'I', 'L', 'M', 'F', 'V'}
        positive = {'R', 'K'}
        negative = {'D', 'E'}
        for i, aa1 in enumerate(self.amino_acids):
            for j, aa2 in enumerate(self.amino_acids):
                if aa1 in hydrophobic and aa2 in hydrophobic:
                    self.mj_matrix[i, j] = -1.0 * self.scale_factor
                elif (aa1 in positive and aa2 in negative) or (aa1 in negative and aa2 in positive):
                    self.mj_matrix[i, j] = -0.5 * self.scale_factor
                elif aa1 == aa2 and aa1 in {'C', 'H'}:
                    self.mj_matrix[i, j] = -0.8 * self.scale_factor
                else:
                    self.mj_matrix[i, j] = 0.0

    def _initialize_helix_pairs(self):
        """Initialize helix pair propensities (simplified)."""
        self.helix_pairs = np.zeros((self.n_aa, self.n_aa))
        hydrophobic = {'A', 'C', 'I', 'L', 'M', 'F', 'V'}
        for i, aa1 in enumerate(self.amino_acids):
            for j, aa2 in enumerate(self.amino_acids):
                if aa1 in hydrophobic and aa2 in hydrophobic:
                    self.helix_pairs[i, j] = -0.5 * self.scale_factor
                elif aa1 == aa2 and aa1 in {'A', 'L', 'E', 'K'}:
                    self.helix_pairs[i, j] = -0.3 * self.scale_factor
                else:
                    self.helix_pairs[i, j] = 0.0

    def _projector_terms_for_code(self, position: int, code: int, base_coeff: float):
        b = self.bits_per_pos
        s = []
        for k in range(b):
            v_k = (code >> k) & 1
            s.append(1.0 if v_k == 0 else -1.0)
        
        num_subsets = 1 << b
        terms = []
        for mask in range(num_subsets):
            coeff = base_coeff / (2 ** b)
            pauli = ['I'] * self.n_qubits
            for k in range(b):
                if (mask >> k) & 1:
                    coeff *= s[k]
                    w = get_qubit_index(position, k, self.bits_per_pos)
                    pauli[w] = 'Z'
            if abs(coeff) > 1e-10:
                terms.append((coeff, ''.join(pauli)))
        return terms

    def _get_membrane_environment(self, position: int):
        phase_deg = self.kwargs.get('wheel_phase_deg', 0.0)
        halfwidth_deg = self.kwargs.get('wheel_halfwidth_deg', 80.0)
        membrane_angle = (position * 100.0 + phase_deg) % 360.0
        if membrane_angle > 180.0: 
            membrane_angle -= 360.0
        faces_membrane = abs(membrane_angle) <= halfwidth_deg
        return "membrane" if faces_membrane else "water"

    def _add_simple_environment_terms(self, weight: float):
        """
        SIMPLIFIED: Direct reward/penalty based on raw hydrophobicity.
        NO normalization, NO conflicting logic.
        """
        if weight == 0:
            return
        
        print("\n" + "="*70)
        print("🎯 ADDING SIMPLE ENVIRONMENT TERMS (HYDROPHOBIC PRIORITY)")
        print("="*70)
        
        batch_terms = []
        
        for i in range(self.L):
            environment = self._get_membrane_environment(i)
            
            for α in range(self.n_aa):
                aa = self.amino_acids[α]
                hydro = self.hydrophobic[α]
                polar = self.is_polar[α]
                charge = abs(self.charges[α])
                
                bonus = 0.0
                
                if environment == "membrane":
                    if hydro > 0:
                        bonus = -weight * hydro * 2.0 * self.scale_factor
                        print(f"✅ Pos {i} MEMBRANE: {aa} (hydro={hydro:+.2f}) → REWARD bonus={bonus:.3f}")
                    if polar or charge > 0:
                        penalty_strength = 3.0 if charge > 0 else 2.0
                        bonus += weight * penalty_strength * self.scale_factor
                        print(f"❌ Pos {i} MEMBRANE: {aa} (polar={polar}, charge={charge}) → PENALTY bonus={bonus:.3f}")
                
                else:  # water
                    if polar or charge > 0:
                        reward_strength = 2.5 if charge > 0 else 2.0
                        bonus = -weight * reward_strength * self.scale_factor
                        print(f"✅ Pos {i} WATER: {aa} (polar={polar}, charge={charge}) → REWARD bonus={bonus:.3f}")
                    if hydro > 0:
                        bonus += weight * hydro * 2.0 * self.scale_factor
                        print(f"❌ Pos {i} WATER: {aa} (hydro={hydro:+.2f}) → PENALTY bonus={bonus:.3f}")
                
                if abs(bonus) > 1e-6:
                    terms = self._projector_terms_for_code(i, α, bonus)
                    batch_terms.extend(terms)
        
        self.pauli_terms.extend(batch_terms)
        print(f"\n📊 Added {len(batch_terms)} environment terms")
        print("="*70 + "\n")
        gc.collect()

    def _add_charge_terms(self, weight: float, membrane_charge: str):
        if weight == 0:
            return
        
        print("\n" + "="*70)
        print("⚡️ ADDING CHARGE TERMS")
        print("="*70)
        
        batch_terms = []
        membrane_charge_val = {'neu': 0.0, 'pos': 1.0, 'neg': -1.0}.get(membrane_charge, 0.0)
        
        for i in range(self.L):
            environment = self._get_membrane_environment(i)
            if environment == "membrane" and abs(membrane_charge_val) > 0:
                for α in range(self.n_aa):
                    aa = self.amino_acids[α]
                    charge = self.charges[α]
                    energy = weight * charge * membrane_charge_val * self.scale_factor
                    if abs(energy) > 1e-6:
                        terms = self._projector_terms_for_code(i, α, energy)
                        batch_terms.extend(terms)
                        print(f"⚡️ Pos {i} MEMBRANE: {aa} (charge={charge:+.0f}) → energy={energy:.3f}")
        
        self.pauli_terms.extend(batch_terms)
        print(f"\n📊 Added {len(batch_terms)} charge terms")
        print("="*70 + "\n")
        gc.collect()

    def _add_hydrophobic_moment_terms(self, weight: float):
        if weight == 0 or self.kwargs.get('membrane_mode') != 'wheel':
            return
        
        print("\n" + "="*70)
        print("🌊 ADDING HYDROPHOBIC MOMENT TERMS")
        print("="*70)
        
        batch_terms = []
        phase = np.deg2rad(self.kwargs.get('wheel_phase_deg', 0.0))
        
        for i in range(self.L):
            angle = (i * np.deg2rad(100.0) + phase) % (2 * np.pi)
            for α in range(self.n_aa):
                aa = self.amino_acids[α]
                hydrophobicity = self.hydrophobic[α]
                energy = weight * hydrophobicity * np.cos(angle) * self.scale_factor
                if abs(energy) > 1e-6:
                    terms = self._projector_terms_for_code(i, α, energy)
                    batch_terms.extend(terms)
                    print(f"🌊 Pos {i} (angle={np.rad2deg(angle):.1f}°): {aa} (hydro={hydrophobicity:+.2f}) → energy={energy:.3f}")
        
        self.pauli_terms.extend(batch_terms)
        print(f"\n📊 Added {len(batch_terms)} hydrophobic moment terms")
        print("="*70 + "\n")
        gc.collect()

    def _add_local_preference_terms(self, weight: float):
        if weight == 0:
            return
        
        print("\n" + "="*70)
        print("🧬 ADDING LOCAL PREFERENCE TERMS (HELIX PROPENSITY)")
        print("="*70)
        
        batch_terms = []
        helix_formers = {'A', 'L', 'E', 'K'}
        
        for i in range(self.L):
            for α in range(self.n_aa):
                aa = self.amino_acids[α]
                if aa in helix_formers:
                    energy = -weight * self.scale_factor
                    terms = self._projector_terms_for_code(i, α, energy)
                    batch_terms.extend(terms)
                    print(f"🧬 Pos {i}: {aa} (helix={self.helix_prop[α]:.2f}) → REWARD energy={energy:.3f}")
        
        self.pauli_terms.extend(batch_terms)
        print(f"\n📊 Added {len(batch_terms)} local preference terms")
        print("="*70 + "\n")
        gc.collect()

    def _add_pairwise_interaction_terms(self, weight: float, max_dist: int):
        if weight == 0:
            return
        
        print("\n" + "="*70)
        print("🤝 ADDING PAIRWISE INTERACTION TERMS")
        print("="*70)
        
        batch_terms = []
        
        for i in range(self.L):
            for j in range(i + 1, self.L):
                if abs(i - j) <= max_dist:
                    for α in range(self.n_aa):
                        for β in range(self.n_aa):
                            energy = weight * self.mj_matrix[α, β]
                            if abs(energy) > 1e-6:
                                aa1, aa2 = self.amino_acids[α], self.amino_acids[β]
                                terms1 = self._projector_terms_for_code(i, α, 1.0)
                                terms2 = self._projector_terms_for_code(j, β, energy)
                                for c1, p1 in terms1:
                                    for c2, p2 in terms2:
                                        coeff = c1 * c2
                                        pauli = ['I'] * self.n_qubits
                                        for k in range(self.n_qubits):
                                            if p1[k] == 'Z' or p2[k] == 'Z':
                                                pauli[k] = 'Z'
                                        if abs(coeff) > 1e-10:
                                            batch_terms.append((coeff, ''.join(pauli)))
                                            print(f"🤝 Pos {i}-{j}: {aa1}-{aa2} (MJ={self.mj_matrix[α, β]:.2f}) → energy={coeff:.3f}")
        
        self.pauli_terms.extend(batch_terms)
        print(f"\n📊 Added {len(batch_terms)} pairwise interaction terms")
        print("="*70 + "\n")
        gc.collect()

    def _add_helix_pair_terms(self, weight: float):
        if weight == 0:
            return
        
        print("\n" + "="*70)
        print("🌀 ADDING HELIX PAIR PROPENSITY TERMS")
        print("="*70)
        
        batch_terms = []
        
        for i in range(self.L - 4):
            for j in range(i + 4, self.L):
                for α in range(self.n_aa):
                    for β in range(self.n_aa):
                        energy = weight * self.helix_pairs[α, β]
                        if abs(energy) > 1e-6:
                            aa1, aa2 = self.amino_acids[α], self.amino_acids[β]
                            terms1 = self._projector_terms_for_code(i, α, 1.0)
                            terms2 = self._projector_terms_for_code(j, β, energy)
                            for c1, p1 in terms1:
                                for c2, p2 in terms2:
                                    coeff = c1 * c2
                                    pauli = ['I'] * self.n_qubits
                                    for k in range(self.n_qubits):
                                        if p1[k] == 'Z' or p2[k] == 'Z':
                                            pauli[k] = 'Z'
                                    if abs(coeff) > 1e-10:
                                        batch_terms.append((coeff, ''.join(pauli)))
                                        print(f"🌀 Pos {i}-{j}: {aa1}-{aa2} (helix_pair={self.helix_pairs[α, β]:.2f}) → energy={coeff:.3f}")
        
        self.pauli_terms.extend(batch_terms)
        print(f"\n📊 Added {len(batch_terms)} helix pair terms")
        print("="*70 + "\n")
        gc.collect()

    def _add_amphipathic_segregation_terms(self, weight: float):
        if weight == 0 or self.kwargs.get('membrane_mode') != 'wheel':
            return
        
        print("\n" + "="*70)
        print("🌗 ADDING AMPHIPATHIC SEGREGATION TERMS")
        print("="*70)
        
        batch_terms = []
        phase = np.deg2rad(self.kwargs.get('wheel_phase_deg', 0.0))
        halfwidth = np.deg2rad(self.kwargs.get('wheel_halfwidth_deg', 80.0))
        hydrophobic = {'A', 'C', 'I', 'L', 'M', 'F', 'V'}
        
        for i in range(self.L):
            for j in range(i + 1, self.L):
                angle_i = (i * np.deg2rad(100.0) + phase) % (2 * np.pi)
                angle_j = (j * np.deg2rad(100.0) + phase) % (2 * np.pi)
                if angle_i > np.pi: angle_i -= 2 * np.pi
                if angle_j > np.pi: angle_j -= 2 * np.pi
                same_side = (abs(angle_i) <= halfwidth and abs(angle_j) <= halfwidth) or \
                           (abs(angle_i) > halfwidth and abs(angle_j) > halfwidth)
                
                for α in range(self.n_aa):
                    for β in range(self.n_aa):
                        aa1, aa2 = self.amino_acids[α], self.amino_acids[β]
                        is_hydrophobic1 = aa1 in hydrophobic
                        is_hydrophobic2 = aa2 in hydrophobic
                        if same_side and ((is_hydrophobic1 and is_hydrophobic2) or 
                                        (not is_hydrophobic1 and not is_hydrophobic2)):
                            energy = -weight * self.scale_factor
                        elif not same_side and is_hydrophobic1 != is_hydrophobic2:
                            energy = -weight * self.scale_factor
                        else:
                            energy = weight * self.scale_factor
                        if abs(energy) > 1e-6:
                            terms1 = self._projector_terms_for_code(i, α, 1.0)
                            terms2 = self._projector_terms_for_code(j, β, energy)
                            for c1, p1 in terms1:
                                for c2, p2 in terms2:
                                    coeff = c1 * c2
                                    pauli = ['I'] * self.n_qubits
                                    for k in range(self.n_qubits):
                                        if p1[k] == 'Z' or p2[k] == 'Z':
                                            pauli[k] = 'Z'
                                    if abs(coeff) > 1e-10:
                                        batch_terms.append((coeff, ''.join(pauli)))
                                        print(f"🌗 Pos {i}-{j}: {aa1}-{aa2} (same_side={same_side}) → energy={coeff:.3f}")
        
        self.pauli_terms.extend(batch_terms)
        print(f"\n📊 Added {len(batch_terms)} amphipathic segregation terms")
        print("="*70 + "\n")
        gc.collect()

    def _add_electrostatic_interaction_terms(self, weight: float):
        if weight == 0:
            return
        
        print("\n" + "="*70)
        print("⚡️ ADDING ELECTROSTATIC INTERACTION TERMS")
        print("="*70)
        
        batch_terms = []
        
        for i in range(self.L):
            for j in range(i + 1, self.L):
                for α in range(self.n_aa):
                    for β in range(self.n_aa):
                        aa1, aa2 = self.amino_acids[α], self.amino_acids[β]
                        charge1, charge2 = self.charges[α], self.charges[β]
                        energy = weight * charge1 * charge2 * self.scale_factor
                        if abs(energy) > 1e-6:
                            terms1 = self._projector_terms_for_code(i, α, 1.0)
                            terms2 = self._projector_terms_for_code(j, β, energy)
                            for c1, p1 in terms1:
                                for c2, p2 in terms2:
                                    coeff = c1 * c2
                                    pauli = ['I'] * self.n_qubits
                                    for k in range(self.n_qubits):
                                        if p1[k] == 'Z' or p2[k] == 'Z':
                                            pauli[k] = 'Z'
                                    if abs(coeff) > 1e-10:
                                        batch_terms.append((coeff, ''.join(pauli)))
                                        print(f"⚡️ Pos {i}-{j}: {aa1}-{aa2} (charge={charge1:+.0f},{charge2:+.0f}) → energy={coeff:.3f}")
        
        self.pauli_terms.extend(batch_terms)
        print(f"\n📊 Added {len(batch_terms)} electrostatic interaction terms")
        print("="*70 + "\n")
        gc.collect()

    def _add_invalid_code_penalties(self, weight: float):
        print("\n" + "="*70)
        print("🚫 ADDING INVALID CODE PENALTY TERMS")
        print("="*70)
        
        max_code = (1 << self.bits_per_pos) - 1
        if self.n_aa - 1 == max_code:
            return
        
        batch_terms = []
        for i in range(self.L):
            for code in range(self.n_aa, max_code + 1):
                terms = self._projector_terms_for_code(i, code, weight * self.scale_factor)
                batch_terms.extend(terms)
                print(f"🚫 Pos {i}: Invalid code {code} → penalty={weight * self.scale_factor:.3f}")
        
        self.pauli_terms.extend(batch_terms)
        print(f"\n📊 Added {len(batch_terms)} invalid code penalty terms")
        print("="*70 + "\n")
        gc.collect()

    def _combine_pauli_terms(self):
        """Combine identical Pauli terms to reduce Hamiltonian size."""
        term_dict = {}
        for coeff, pauli_string in self.pauli_terms:
            if pauli_string in term_dict:
                term_dict[pauli_string] += coeff
            else:
                term_dict[pauli_string] = coeff
        
        self.pauli_terms = [(coeff, pauli) for pauli, coeff in term_dict.items() if abs(coeff) > 1e-10]
        
        identity_term = 'I' * self.n_qubits
        identity_coeff = term_dict.get(identity_term, 0.0)
        if abs(identity_coeff) > 1e-10:
            print(f"⚠️ Identity term found with coefficient {identity_coeff:.3f}. This is a constant offset.")

    def build_hamiltonian(self, backend: str = 'pennylane') -> Tuple[List[Tuple[float, str]], Any]:
        print("\n" + "="*70)
        print("🏗️  BUILDING FULL HAMILTONIAN (ALL TERMS)")
        print("="*70)
        
        self.pauli_terms = []
        
        self._add_simple_environment_terms(self.kwargs.get('lambda_env', 1.0))
        self._add_charge_terms(self.kwargs.get('lambda_charge', 0.5), self.kwargs.get('membrane_charge', 'neu'))
        self._add_hydrophobic_moment_terms(self.kwargs.get('lambda_mu', 1.0))
        self._add_local_preference_terms(self.kwargs.get('lambda_local', 0.5))
        self._add_pairwise_interaction_terms(self.kwargs.get('lambda_pairwise', 0.5), self.kwargs.get('max_interaction_dist', 1))
        self._add_helix_pair_terms(self.kwargs.get('lambda_helix_pairs', 0.5))
        self._add_amphipathic_segregation_terms(self.kwargs.get('lambda_segregation', 1.0))
        self._add_electrostatic_interaction_terms(self.kwargs.get('lambda_electrostatic', 0.5))
        self._add_invalid_code_penalties(weight=10.0)
        
        # Combine identical Pauli terms
        self._combine_pauli_terms()
        
        if not self.pauli_terms:
            print("Error: No Pauli terms were constructed!")
            return [], None
        
        print("\n" + "="*70)
        print(f"✅ Hamiltonian built with {len(self.pauli_terms)} Pauli terms")
        coeffs = [term[0] for term in self.pauli_terms]
        if coeffs:
            print(f"📊 Coefficient range: [{min(coeffs):.3f}, {max(coeffs):.3f}]")
            print(f"📊 Mean: {np.mean(coeffs):.3f} | Std: {np.std(coeffs):.3f}")
        else:
            print("Error: Coefficient list is empty!")
            return [], None
        
        try:
            hamiltonian = qml.Hamiltonian(
                [term[0] for term in self.pauli_terms],
                [qml.pauli.string_to_pauli_word(term[1]) for term in self.pauli_terms]
            )
            print(f"✅ PennyLane Hamiltonian constructed with {len(self.pauli_terms)} terms")
        except Exception as e:
            print(f"Error constructing qml.Hamiltonian: {e}")
            return self.pauli_terms, None
        
        if backend == 'qiskit':
            print("Returning Pauli terms for Qiskit backend")
            return self.pauli_terms, hamiltonian
        else:
            print("Returning PennyLane Hamiltonian")
            return self.pauli_terms, hamiltonian

    def get_energy_summary(self):
        coeffs = [term[0] for term in self.pauli_terms]
        return {
            'coefficient_range': (min(coeffs), max(coeffs)) if coeffs else (0, 0),
            'mean_coefficient': np.mean(coeffs) if coeffs else 0,
            'std_coefficient': np.std(coeffs) if coeffs else 0,
            'total_terms': len(coeffs)
        }