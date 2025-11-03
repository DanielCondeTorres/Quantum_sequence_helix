import numpy as np
import pennylane as qml
from typing import List, Tuple, Optional, Dict, Any
from utils.general_utils import get_qubit_index
from data_loaders.energy_matrix_loader import (
    _load_first_neighbors_matrix_file, 
    _load_third_neighbors_matrix_file, 
    _load_fourth_neighbors_matrix_file, 
    _load_energy_matrix_file
)
import gc
from collections import defaultdict

class HamiltonianBuilder:
    """
    Builds the quantum Hamiltonian (QUBO) for the protein design problem.
    FIXED VERSION: Ensures polar amino acids go to water, hydrophobic to membrane.
    Supports both PennyLane and Qiskit backends.
    NOW LOADS MJ MATRIX AND HELIX PROPENSITY FROM FILES.
    """
    def __init__(self, L: int, amino_acids: List[str], bits_per_pos: int, n_qubits: int, **kwargs):
        self.L = L
        self.amino_acids = amino_acids
        self.n_aa = len(amino_acids)
        self.bits_per_pos = bits_per_pos
        self.n_qubits = n_qubits
        self.kwargs = kwargs
        self.pauli_terms = []
        self.scale_factor = kwargs.get('scale_factor', 0.1)
        self.kwargs['wheel_phase_deg'] = kwargs.get('wheel_phase_deg', 0.0)
        self.kwargs['wheel_halfwidth_deg'] = kwargs.get('wheel_halfwidth_deg', 90.0)
        
        # Initialize amino acid properties (basic)
        self._init_amino_acid_properties()
        
        # Load matrices from files (REQUIRED - NO FALLBACK)
        self._load_matrices_from_files()
        
        self.terms_by_type = defaultdict(list)
        
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
        """Initialize basic amino acid properties (hydrophobicity, charge, polarity)."""
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
            'W': {'helix': 1.08, 'hydrophobic': 0.90, 'charge': 0, 'polar': False, 'volume': 227.8, 'ez': -1.85}, # 'hydrophobic': 0.90
            'Y': {'helix': 0.69, 'hydrophobic': -0.30, 'charge': 0, 'polar': True, 'volume': 193.6, 'ez': -0.94},
            'V': {'helix': 1.06, 'hydrophobic': 4.20, 'charge': 0, 'polar': False, 'volume': 140.0, 'ez': -0.46},
        }
        
        self.helix_prop = np.array([properties[aa]['helix'] for aa in self.amino_acids])
        self.hydrophobic = np.array([properties[aa]['hydrophobic'] for aa in self.amino_acids])
        self.charges = np.array([properties[aa]['charge'] for aa in self.amino_acids])
        self.is_polar = np.array([properties[aa]['polar'] for aa in self.amino_acids], dtype=float)
        self.volumes = np.array([properties[aa]['volume'] for aa in self.amino_acids])
        self.ez_values = np.array([properties[aa]['ez'] for aa in self.amino_acids])

    def _load_matrices_from_files(self):
        """Load MJ matrix and helix propensity matrix from files."""
        print("\n" + "="*70)
        print("📁 LOADING MATRICES FROM FILES")
        print("="*70)
        
        # Load MJ matrix (REQUIRED - NO FALLBACK)
        mj_matrix_full, mj_symbols = _load_energy_matrix_file()
        print('BLOODY MERY: ',mj_matrix_full)
        print(f"✅ Loaded MJ matrix with symbols: {mj_symbols}")
        print(f"   Matrix shape: {mj_matrix_full.shape}")
        print(f"   Sample values (C-C, C-M, M-M): {mj_matrix_full[0,0]:.2f}, {mj_matrix_full[0,1]:.2f}, {mj_matrix_full[1,1]:.2f}")
        
        # Map amino acids to indices in the loaded matrix
        aa_to_mj_idx = {aa: idx for idx, aa in enumerate(mj_symbols)}
        
        # Create matrix for current amino acid set
        self.mj_matrix = np.zeros((self.n_aa, self.n_aa))
        for i, aa1 in enumerate(self.amino_acids):
            for j, aa2 in enumerate(self.amino_acids):
                if aa1 not in aa_to_mj_idx or aa2 not in aa_to_mj_idx:
                    raise ValueError(f"❌ Amino acid {aa1} or {aa2} not found in MJ matrix file!")
                idx1 = aa_to_mj_idx[aa1]
                idx2 = aa_to_mj_idx[aa2]
                self.mj_matrix[i, j] = mj_matrix_full[idx1, idx2]
        
        print(f"✅ MJ matrix mapped to current amino acids")
        print(f"   Matrix shape: {self.mj_matrix.shape}")
        print(f"   Matrix range: [{self.mj_matrix.min():.3f}, {self.mj_matrix.max():.3f}]")
        print(f"   Non-zero entries: {np.count_nonzero(self.mj_matrix)}/{self.mj_matrix.size}")
        
        # Load helix propensity matrix (REQUIRED - NO FALLBACK)
        helix_matrix_full, helix_symbols = _load_first_neighbors_matrix_file()
        print(f"✅ Loaded Helix matrix with symbols: {helix_symbols}")
        
        # Map amino acids to indices in the loaded matrix
        aa_to_helix_idx = {aa: idx for idx, aa in enumerate(helix_symbols)}
        
        # Create matrix for current amino acid set
        self.helix_pairs = np.zeros((self.n_aa, self.n_aa))
        
        for i, aa1 in enumerate(self.amino_acids):
            for j, aa2 in enumerate(self.amino_acids):
                if aa1 not in aa_to_helix_idx or aa2 not in aa_to_helix_idx:
                    raise ValueError(f"❌ Amino acid {aa1} or {aa2} not found in helix propensity matrix file!")
                idx1 = aa_to_helix_idx[aa1]
                idx2 = aa_to_helix_idx[aa2]
                self.helix_pairs[i, j] = helix_matrix_full[idx1, idx2]
        print('MATRIZ HELICES: ',self.helix_pairs)
        
        print(f"✅ Helix propensity matrix mapped to current amino acids")
        print(f"   Matrix shape: {self.helix_pairs.shape}")
        print(f"   Matrix range: [{self.helix_pairs.min():.3f}, {self.helix_pairs.max():.3f}]")
        
        print("="*70 + "\n")

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
        halfwidth_deg = self.kwargs.get('wheel_halfwidth_deg', 90.0)
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
                    # In membrane: reward hydrophobic, penalize polar/charged
                    if hydro > 0:
                        bonus = -weight * hydro * 2.0 * self.scale_factor  # Negative = reward
                        print(f"✅ Pos {i} MEMBRANE: {aa} (hydro={hydro:+.2f}) → REWARD bonus={bonus:.3f}")
                    elif polar or charge > 0:
                        penalty_strength = 3.0 if charge > 0 else 2.0
                        bonus = weight * penalty_strength * self.scale_factor  # Positive = penalty
                        print(f"❌ Pos {i} MEMBRANE: {aa} (polar={polar}, charge={charge}) → PENALTY bonus={bonus:.3f}")
                
                else:  # water
                    # In water: reward polar/charged, penalize hydrophobic
                    if polar or charge > 0:
                        reward_strength = 2.5 if charge > 0 else 2.0
                        bonus = -weight * reward_strength * self.scale_factor  # Negative = reward
                        print(f"✅ Pos {i} WATER: {aa} (polar={polar}, charge={charge}) → REWARD bonus={bonus:.3f}")
                    elif hydro > 0:
                        bonus = weight * hydro * 2.0 * self.scale_factor  # Positive = penalty
                        print(f"❌ Pos {i} WATER: {aa} (hydro={hydro:+.2f}) → PENALTY bonus={bonus:.3f}")
                
                if abs(bonus) > 1e-6:
                    terms = self._projector_terms_for_code(i, α, bonus)
                    batch_terms.extend(terms)
        
        self.terms_by_type['Environment Terms'].extend(batch_terms)
        print(f"\n📊 Added {len(batch_terms)} environment terms to 'Environment Terms'")
        print("="*70 + "\n")
        gc.collect()

    def _add_charge_terms(self, weight: float, membrane_charge: str):
        if weight == 0:
            print("⚠️ Charge terms skipped: weight is 0")
            return
        
        print("\n" + "="*70)
        print(f"⚡️ ADDING CHARGE TERMS (weight={weight}, membrane_charge={membrane_charge})")
        print("="*70)
        
        batch_terms = []
        membrane_charge_val = {'neu': 0.0, 'pos': 1.0, 'neg': -1.0}.get(membrane_charge.lower(), 0.0)
        print(f"Membrane charge value: {membrane_charge_val}")
        
        membrane_positions = []
        for i in range(self.L):
            environment = self._get_membrane_environment(i)
            if environment == "membrane" and abs(membrane_charge_val) > 0:
                membrane_positions.append(i)
                for α in range(self.n_aa):
                    aa = self.amino_acids[α]
                    charge = self.charges[α]
                    energy = weight * charge * membrane_charge_val * self.scale_factor
                    if abs(energy) > 1e-6:
                        terms = self._projector_terms_for_code(i, α, energy)
                        batch_terms.extend(terms)
                        print(f"⚡️ Pos {i} MEMBRANE: {aa} (charge={charge:+.0f}) → energy={energy:.3f}")
        
        self.terms_by_type['Charge Terms'].extend(batch_terms)
        print(f"\n📊 Added {len(batch_terms)} charge terms to 'Charge Terms'")
        print(f"Membrane positions: {membrane_positions}")
        if len(batch_terms) == 0:
            print(f"⚠️ No charge terms added. Reasons: No membrane positions ({len(membrane_positions)} found) or membrane_charge_val={membrane_charge_val}")
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
        
        self.terms_by_type['Hydrophobic Moment Terms'].extend(batch_terms)
        print(f"\n📊 Added {len(batch_terms)} hydrophobic moment terms to 'Hydrophobic Moment Terms'")
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
        
        self.terms_by_type['Local Preference Terms'].extend(batch_terms)
        print(f"\n📊 Added {len(batch_terms)} local preference terms to 'Local Preference Terms'")
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
                                            if i < 2 and j < 3:  # Only print first few
                                                print(f"🤝 Pos {i}-{j}: {aa1}-{aa2} (MJ={self.mj_matrix[α, β]:.2f}) → energy={coeff:.3f}")
        
        self.terms_by_type['Pairwise Interaction Terms'].extend(batch_terms)
        print(f"\n📊 Added {len(batch_terms)} pairwise interaction terms to 'Pairwise Interaction Terms'")
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
            for j in range(i + 3, self.L):
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
                                        if i < 2 and j < 6:  # Only print first few
                                            print(f"🌀 Pos {i}-{j}: {aa1}-{aa2} (helix_pair={self.helix_pairs[α, β]:.2f}) → energy={coeff:.3f}")
        
        self.terms_by_type['Helix Pair Terms'].extend(batch_terms)
        print(f"\n📊 Added {len(batch_terms)} helix pair terms to 'Helix Pair Terms'")
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
        halfwidth = np.deg2rad(self.kwargs.get('wheel_halfwidth_deg', 90.0))
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
        
        self.terms_by_type['Amphipathic Segregation Terms'].extend(batch_terms)
        print(f"\n📊 Added {len(batch_terms)} amphipathic segregation terms to 'Amphipathic Segregation Terms'")
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
        
        self.terms_by_type['Electrostatic Interaction Terms'].extend(batch_terms)
        print(f"\n📊 Added {len(batch_terms)} electrostatic interaction terms to 'Electrostatic Interaction Terms'")
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
        
        self.terms_by_type['Invalid Code Penalties'].extend(batch_terms)
        print(f"\n📊 Added {len(batch_terms)} invalid code penalty terms to 'Invalid Code Penalties'")
        print("="*70 + "\n")
        gc.collect()

    def _combine_pauli_terms(self):
            """
            Combina términos de Pauli idénticos y aísla el término de Identidad
            (offset constante) para una clasificación de energía correcta.
            """
            term_dict = {}
            identity_term = 'I' * self.n_qubits
            
            # Inicializar la nueva categoría de offset si aún no existe
            if 'Constant Offset' not in self.terms_by_type:
                self.terms_by_type['Constant Offset'] = []
                
            # 1. Procesar términos, separar identidad y combinar dentro de cada categoría.
            # Usamos list(keys()) para que el diccionario no cambie durante la iteración.
            for category in list(self.terms_by_type.keys()):
                # No procesamos la categoría 'Constant Offset' que estamos llenando
                if category == 'Constant Offset':
                    continue
                    
                cat_dict = {}
                terms_to_move = []
                
                for coeff, pauli in self.terms_by_type[category]:
                    if pauli == identity_term:
                        # **MOVER TÉRMINOS DE IDENTIDAD (OFFSET) A SU PROPIA CATEGORÍA**
                        terms_to_move.append((coeff, pauli))
                        continue # No agregar este término a la categoría original
                    
                    # Combinar términos no-Identidad
                    if pauli in cat_dict:
                        cat_dict[pauli] += coeff
                    else:
                        cat_dict[pauli] = coeff
                
                # Asignar términos combinados no-Identidad de vuelta a la categoría original
                self.terms_by_type[category] = [(coeff, pauli) for pauli, coeff in cat_dict.items() if abs(coeff) > 1e-10]
                
                # Mover los términos de identidad acumulados
                self.terms_by_type['Constant Offset'].extend(terms_to_move)

                # Recolectar para el diccionario total (incluyendo el offset que se sumará luego)
                for coeff, pauli in self.terms_by_type[category]:
                    if pauli in term_dict:
                        term_dict[pauli] += coeff
                    else:
                        term_dict[pauli] = coeff

            # 2. Manejar el término de identidad (Constant Offset) por separado.
            identity_coeff_total = 0.0
            
            # Sumar todos los coeficientes de identidad agrupados de todas las categorías
            identity_coeff_total = sum(coeff for coeff, pauli in self.terms_by_type['Constant Offset'])
            
            # Reemplazar la lista de términos por un único término sumado para 'Constant Offset'
            if abs(identity_coeff_total) > 1e-10:
                self.terms_by_type['Constant Offset'] = [(identity_coeff_total, identity_term)]
                # Añadir al diccionario total de todos los términos del Hamiltoniano
                term_dict[identity_term] = identity_coeff_total 
            elif 'Constant Offset' in self.terms_by_type:
                # Eliminar si el coeficiente total es cero
                del self.terms_by_type['Constant Offset']

            # 3. Finalizar la lista de términos totales.
            self.pauli_terms = [(coeff, pauli) for pauli, coeff in term_dict.items() if abs(coeff) > 1e-10]
            
            if abs(identity_coeff_total) > 1e-10:
                print(f"⚠️ Identity term (Constant Offset) found with coefficient {identity_coeff_total:.3f}.")

    def build_hamiltonian(self, backend: str = 'pennylane') -> Tuple[List[Tuple[float, str]], Any]:
        print("\n" + "="*70)
        print("🏗️  BUILDING FULL HAMILTONIAN (ALL TERMS)")
        print("="*70)
        
        self.pauli_terms = []
        
        self._add_simple_environment_terms(self.kwargs.get('lambda_env', 2.0))
        self._add_charge_terms(self.kwargs.get('lambda_charge', 0.0), self.kwargs.get('membrane_charge', 'neg'))
        self._add_hydrophobic_moment_terms(self.kwargs.get('lambda_mu', 2.0))
        self._add_local_preference_terms(self.kwargs.get('lambda_local', 0.0))
        self._add_pairwise_interaction_terms(self.kwargs.get('lambda_pairwise', 0.0), self.kwargs.get('max_interaction_dist', 4))
        self._add_helix_pair_terms(self.kwargs.get('lambda_helix_pairs', 0.0))
        self._add_amphipathic_segregation_terms(self.kwargs.get('lambda_segregation', 1.0))
        self._add_electrostatic_interaction_terms(self.kwargs.get('lambda_electrostatic', 0.0))
        self._add_invalid_code_penalties(weight=10.0)
        
        # Recolecta todos los términos para el Hamiltoniano total
        self.pauli_terms = []
        for terms in self.terms_by_type.values():
            self.pauli_terms.extend(terms)
        
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
