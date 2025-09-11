import numpy as np
from pennylane import numpy as qnp
import pennylane as qml
from pennylane import qaoa
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import Dict, List, Tuple, Optional
import itertools
import argparse
import sys
# Clasico y helice a mas vecinos
# Calculo estadistico de combinaciones posibles intentant hacer 4 qubits por posicion,  intentan juntar helices mas parecidos juntarlos
# Also support Qiskit
try:
    from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
    from qiskit.algorithms import QAOA, VQE
    from qiskit.algorithms.optimizers import COBYLA, SPSA
    from qiskit.opflow import PauliSumOp
    from qiskit.providers.aer import AerSimulator
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False
    print("Qiskit not available. Using PennyLane only.")

class QuantumProteinDesign:
    """
    Quantum implementation of protein sequence design QUBO
    Supports both PennyLane and Qiskit backends
    """
    
    def __init__(self, sequence_length: int, amino_acids: List[str] = None, 
                 quantum_backend: str = 'pennylane',
                 membrane_span: Optional[Tuple[int, int]] = None,
                 membrane_positions: Optional[List[int]] = None,
                 membrane_mode: str = 'span',  # 'span' | 'set' | 'wheel'
                 wheel_phase_deg: float = 0.0,
                 wheel_halfwidth_deg: float = 40.0,
                 membrane_charge: str = 'neu',
                 lambda_charge: float = 0.0,
                 lambda_env: float = 0.0,
                 lambda_mu: float = 0.0):
        
        self.L = sequence_length
        if amino_acids is None:
            # Use subset of amino acids for quantum feasibility
            self.amino_acids = ['A', 'L', 'E']  # 4 amino acids for demo
        else:
            self.amino_acids = amino_acids
            
        self.n_aa = len(self.amino_acids)
        # Binary encoding: bits per position
        self.bits_per_pos = max(1, int(np.ceil(np.log2(self.n_aa))))
        self.n_qubits = self.L * self.bits_per_pos
        
        print(f"🧬 QUANTUM PROTEIN DESIGN SETUP 🧬")
        print(f"Sequence length: {self.L}")
        print(f"Amino acids: {self.amino_acids}")
        print(f"Required qubits: {self.n_qubits}")
        print(f"Quantum backend: {quantum_backend}")
        
        if self.n_qubits > 20:
            print("⚠️  Warning: >20 qubits may be slow on simulators")
        
        self.backend = quantum_backend
        # Membrane/environment settings
        self.membrane_span = membrane_span  # (start_idx, end_idx) 0-based inclusive or None
        self.membrane_positions = set(membrane_positions) if membrane_positions else set()
        self.membrane_mode = membrane_mode
        self.wheel_phase_deg = float(wheel_phase_deg)
        self.wheel_halfwidth_deg = float(wheel_halfwidth_deg)
        self.membrane_charge = membrane_charge  # 'neg' | 'pos' | 'neu'
        self.lambda_charge = float(lambda_charge)
        self.lambda_env = float(lambda_env)
        self.lambda_mu = float(lambda_mu)
        self._init_amino_acid_properties()
        self._build_hamiltonian()
        
        # Initialize quantum devices
        if quantum_backend == 'pennylane':
            self._setup_pennylane()
        elif quantum_backend == 'qiskit' and QISKIT_AVAILABLE:
            self._setup_qiskit()
    
    def _init_amino_acid_properties(self):
        """Initialize properties for selected amino acids"""
        
        # Simplified properties for quantum demo
        properties = {
            # helix: rough Chou-Fasman tendency (relative), hydrophobic: Kyte-Doolittle, charge: at pH ~7
            'A': {'helix': 1.42, 'hydrophobic': 1.80, 'charge': 0},   # Alanine
            'R': {'helix': 0.98, 'hydrophobic': -4.50, 'charge': 1},  # Arginine
            'N': {'helix': 0.67, 'hydrophobic': -3.50, 'charge': 0},  # Asparagine
            'D': {'helix': 1.01, 'hydrophobic': -3.50, 'charge': -1}, # Aspartate
            'C': {'helix': 0.70, 'hydrophobic': 2.50, 'charge': 0},   # Cysteine
            'E': {'helix': 1.51, 'hydrophobic': -3.50, 'charge': -1}, # Glutamate
            'Q': {'helix': 1.11, 'hydrophobic': -3.50, 'charge': 0},  # Glutamine
            'G': {'helix': 0.57, 'hydrophobic': -0.40, 'charge': 0},  # Glycine
            'H': {'helix': 1.00, 'hydrophobic': -3.20, 'charge': 0},  # Histidine (can be +1; approx 0)
            'I': {'helix': 1.08, 'hydrophobic': 4.50, 'charge': 0},   # Isoleucine
            'L': {'helix': 1.21, 'hydrophobic': 3.80, 'charge': 0},   # Leucine
            'K': {'helix': 1.16, 'hydrophobic': -3.90, 'charge': 1},  # Lysine
            'M': {'helix': 1.45, 'hydrophobic': 1.90, 'charge': 0},   # Methionine
            'F': {'helix': 1.13, 'hydrophobic': 2.80, 'charge': 0},   # Phenylalanine
            'P': {'helix': 0.57, 'hydrophobic': -1.60, 'charge': 0},  # Proline
            'S': {'helix': 0.77, 'hydrophobic': -0.80, 'charge': 0},  # Serine
            'T': {'helix': 0.83, 'hydrophobic': -0.70, 'charge': 0},  # Threonine
            'W': {'helix': 1.08, 'hydrophobic': -0.90, 'charge': 0},  # Tryptophan
            'Y': {'helix': 0.69, 'hydrophobic': -1.30, 'charge': 0},  # Tyrosine
            'V': {'helix': 1.06, 'hydrophobic': 4.20, 'charge': 0},   # Valine
        }
        # If a residue is not in the table, add neutral defaults instead of failing
        for aa in self.amino_acids:
            if aa not in properties:
                print(f"Warning: Unknown amino acid '{aa}'. Using neutral defaults.")
                properties[aa] = {'helix': 1.0, 'hydrophobic': 0.0, 'charge': 0}
        
        # Extract properties for selected amino acids
        self.helix_prop = [properties[aa]['helix'] for aa in self.amino_acids]
        self.hydrophobic = [properties[aa]['hydrophobic'] for aa in self.amino_acids] 
        self.charges = [properties[aa]['charge'] for aa in self.amino_acids]
        
        # Normalize for QUBO
        self.h_alpha = np.array(self.helix_prop) + np.array(self.hydrophobic)
        
    def get_qubit_index(self, position: int, bit_idx: int) -> int:
        """Convert (position, bit_index) to qubit index under binary encoding"""
        return position * self.bits_per_pos + bit_idx
    
    def _build_hamiltonian(self):
        """Build the quantum Hamiltonian as sum of Pauli operators"""
        
        print("Building quantum Hamiltonian...")
        
        # Store Hamiltonian as list of (coefficient, pauli_string) pairs
        self.pauli_terms = []
        
        # 1. Local amino acid preferences (H1 terms) via binary projectors
        h1_weight = 1.0
        self._add_local_terms(h1_weight)
        
        # 2. Simple pairwise interactions (H2 terms) 
        h2_weight = 0.5
        self._add_pairwise_terms(h2_weight)

        # 3. Environment preference (polar in solvent, hydrophobic in membrane)
        if self.lambda_env != 0.0:
            self._add_environment_terms(self.lambda_env)

        # 4. Membrane charge interaction term
        if self.lambda_charge != 0.0 and self.membrane_charge in ['neg', 'pos'] and self.membrane_span is not None:
            self._add_membrane_charge_term(self.lambda_charge, self.membrane_charge)

        # 5. Hydrophobic moment encouragement (amphipathic helix)
        if self.lambda_mu != 0.0:
            self._add_hydrophobic_moment_terms(self.lambda_mu)

        # 6. Penalize invalid codes (>= n_aa) under binary encoding
        self._add_invalid_code_penalties(weight=20.0)
        
        print(f"Hamiltonian built with {len(self.pauli_terms)} Pauli terms")
    
    def _projector_terms_for_code(self, position: int, code: int, base_coeff: float):
        """Expand base_coeff * 1_{code}(bits at position) into Z products.
        Using P(bit=v) = (I + (-1)^v Z)/2 per bit and product over bits.
        """
        b = self.bits_per_pos
        # Precompute s_k = (-1)^v_k where v_k is k-th bit (LSB at k=0)
        s = []
        for k in range(b):
            v_k = (code >> k) & 1
            s.append(1.0 if v_k == 0 else -1.0)
        # Iterate over all subsets of bits to place Zs
        num_subsets = 1 << b
        for mask in range(num_subsets):
            coeff = base_coeff * (1.0 / (2 ** b))
            pauli = ['I'] * self.n_qubits
            ok = True
            for k in range(b):
                if (mask >> k) & 1:
                    coeff *= s[k]
                    w = self.get_qubit_index(position, k)
                    pauli[w] = 'Z'
            if ok:
                self.pauli_terms.append((coeff, ''.join(pauli)))
    
    def _add_local_terms(self, weight: float):
        """Add local amino acid preference terms"""
        
        for i in range(self.L):
            for α in range(self.n_aa):
                # Favorability mapped via projector onto amino acid α
                base = -weight * self.h_alpha[α]
                self._projector_terms_for_code(i, α, base)
    
    def _add_pairwise_terms(self, weight: float):
        """Add simple pairwise interaction terms"""
        
        if self.bits_per_pos > 3:
            # Avoid combinatorial blowup for large alphabets
            return
        # Simple rule: hydrophobic-hydrophobic attraction, charge-charge repulsion
        for i in range(self.L):
            for j in range(i+1, self.L):
                if abs(i-j) <= 3:  # Only nearby residues interact
                    for α in range(self.n_aa):
                        for β in range(self.n_aa):
                            # Hydrophobic-hydrophobic attraction
                            if self.hydrophobic[α] > 0.5 and self.hydrophobic[β] > 0.5:
                                interaction = -0.1  # Attractive
                            # Charge-charge interactions
                            elif self.charges[α] != 0 and self.charges[β] != 0:
                                if self.charges[α] * self.charges[β] > 0:
                                    interaction = 0.2   # Repulsive (same charge)
                                else:
                                    interaction = -0.1  # Attractive (opposite charge)
                            else:
                                interaction = 0.0
                            
                            if interaction != 0:
                                base = weight * interaction
                                # Add projector for α at i and β at j (product)
                                b = self.bits_per_pos
                                # Precompute s-signs for both codes
                                s_i = [1.0 if ((α >> k) & 1) == 0 else -1.0 for k in range(b)]
                                s_j = [1.0 if ((β >> k) & 1) == 0 else -1.0 for k in range(b)]
                                for mask_i in range(1 << b):
                                    for mask_j in range(1 << b):
                                        coeff = base * (1.0 / (2 ** (2*b)))
                                        pauli = ['I'] * self.n_qubits
                                        for k in range(b):
                                            if (mask_i >> k) & 1:
                                                coeff *= s_i[k]
                                                pauli[self.get_qubit_index(i, k)] = 'Z'
                                            if (mask_j >> k) & 1:
                                                coeff *= s_j[k]
                                                pauli[self.get_qubit_index(j, k)] = 'Z'
                                        self.pauli_terms.append((coeff, ''.join(pauli)))

    def _pos_in_membrane(self, pos: int) -> bool:
        mode = self.membrane_mode
        if mode == 'set':
            return pos in self.membrane_positions
        if mode == 'span':
            if self.membrane_span is None:
                return False
            start, end = self.membrane_span
            return start <= pos <= end
        if mode == 'wheel':
            # Helical wheel selector: residues whose angle near membrane-facing direction
            angle = (pos * 100.0 + self.wheel_phase_deg) % 360.0
            # Map to [-180, 180]
            if angle > 180.0:
                angle -= 360.0
            return abs(angle) <= self.wheel_halfwidth_deg
        return False

    def _add_environment_terms(self, weight: float):
        """Linear terms that favor polar residues in solvent and hydrophobic residues in membrane."""
        for i in range(self.L):
            in_mem = self._pos_in_membrane(i)
            env_pref = 1.0 if in_mem else -1.0  # membrane: favor hydrophobic (+), solvent: favor polar (-)
            for α in range(self.n_aa):
                base = weight * env_pref * self.hydrophobic[α]
                self._projector_terms_for_code(i, α, base)

    def _add_membrane_charge_term(self, weight: float, membrane_charge: str):
        """Charge-based linear term active only in membrane span."""
        charge_sign = -1.0 if membrane_charge == 'neg' else (1.0 if membrane_charge == 'pos' else 0.0)
        for i in range(self.L):
            if not self._pos_in_membrane(i):
                continue
            for α in range(self.n_aa):
                base = weight * charge_sign * self.charges[α]
                self._projector_terms_for_code(i, α, base)

    def _add_hydrophobic_moment_terms(self, weight: float):
        """Quadratic terms to encourage amphipathic helix via hydrophobic moment alignment.
        μ_H^2 ≈ Σ_{i,j} h_i h_j cos(φ (i-j)) x_i x_j. We add terms for all positions and AAs.
        """
        if self.bits_per_pos > 3:
            return
        phi = np.deg2rad(100.0)
        for i in range(self.L):
            for j in range(i, self.L):
                cos_fac = np.cos(phi * (j - i))
                if np.isclose(cos_fac, 0.0):
                    continue
                for α in range(self.n_aa):
                    for β in range(self.n_aa):
                        hij = self.hydrophobic[α] * self.hydrophobic[β]
                        if hij == 0:
                            continue
                        base = -weight * hij * cos_fac
                        # Use projector product expansion
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
                                        pauli[self.get_qubit_index(i, k)] = 'Z'
                                    if (mask_j >> k) & 1:
                                        coeff *= s_j[k]
                                        pauli[self.get_qubit_index(j, k)] = 'Z'
                                self.pauli_terms.append((coeff, ''.join(pauli)))

    def _add_invalid_code_penalties(self, weight: float):
        """Penalize binary codes that map outside available amino acids."""
        max_code = (1 << self.bits_per_pos) - 1
        if self.n_aa - 1 == max_code:
            return
        for i in range(self.L):
            for code in range(self.n_aa, max_code + 1):
                self._projector_terms_for_code(i, code, weight)
    
    def _setup_pennylane(self):
        """Setup PennyLane quantum device and circuit"""
        
        # Choose device - use simulator for demo
        self.dev = qml.device('default.qubit', wires=self.n_qubits)
        
        # Convert Pauli terms to PennyLane Hamiltonian
        coeffs = [term[0] for term in self.pauli_terms]
        observables = []
        
        for coeff, pauli_string in self.pauli_terms:
            # Convert Pauli string to PennyLane observable
            obs_list = []
            for i, pauli in enumerate(pauli_string):
                if pauli == 'Z':
                    obs_list.append(qml.PauliZ(i))
                elif pauli == 'X':
                    obs_list.append(qml.PauliX(i))
                elif pauli == 'Y':
                    obs_list.append(qml.PauliY(i))
            
            if obs_list:
                if len(obs_list) == 1:
                    observables.append(obs_list[0])
                else:
                    observables.append(qml.prod(*obs_list))
            else:
                # Identity term (all 'I' in pauli_string)
                observables.append(qml.Identity(0))
        
        self.hamiltonian = qml.Hamiltonian(coeffs, observables)
        print(f"PennyLane Hamiltonian created with {len(coeffs)} terms")

    def compute_energy_from_bitstring(self, bitstring: str) -> float:
        """Compute classical energy of a computational-basis bitstring under Z-only Hamiltonian."""
        # Map bit -> Z eigenvalue (+1 for |0>, -1 for |1>)
        z_vals = np.array([1 if b == '0' else -1 for b in bitstring])
        energy = 0.0
        for coeff, pauli_string in self.pauli_terms:
            prod = 1.0
            for i, p in enumerate(pauli_string):
                if p == 'Z':
                    prod *= z_vals[i]
            energy += coeff * prod
        return float(energy)

    def _codes_to_bitstring(self, codes: List[int]) -> str:
        bits = ['0'] * self.n_qubits
        for i, code in enumerate(codes):
            code = max(0, min(self.n_aa - 1, int(code)))
            for k in range(self.bits_per_pos):
                q = self.get_qubit_index(i, k)
                bits[q] = '1' if ((code >> k) & 1) else '0'
        return ''.join(bits)

    def _random_valid_codes(self) -> List[int]:
        rng = np.random.default_rng()
        return [int(rng.integers(low=0, high=self.n_aa)) for _ in range(self.L)]

    def solve_classical_qubo(self, max_iters: int = 20000, restarts: int = 5,
                             temperature: float = 1.0, cooling: float = 0.995,
                             exhaustive_limit: int = 1_000_000):
        """Classical heuristic solver for the binary-encoded QUBO.
        If n_aa^L <= exhaustive_limit, perform exhaustive search over valid codes.
        Else, run simulated annealing with restarts, flipping random position codes.
        """
        total_states = (self.n_aa ** self.L)
        best_energy = np.inf
        best_codes = None
        if total_states <= exhaustive_limit:
            # Exhaustive over codes per position
            for codes in itertools.product(range(self.n_aa), repeat=self.L):
                bitstring = self._codes_to_bitstring(list(codes))
                e = self.compute_energy_from_bitstring(bitstring)
                if e < best_energy:
                    best_energy = e
                    best_codes = list(codes)
        else:
            rng = np.random.default_rng()
            for _ in range(restarts):
                codes = self._random_valid_codes()
                bitstring = self._codes_to_bitstring(codes)
                energy = self.compute_energy_from_bitstring(bitstring)
                T = temperature
                for _ in range(max_iters):
                    # propose: change code at random position (to a different aa)
                    pos = int(rng.integers(low=0, high=self.L))
                    new_code = int(rng.integers(low=0, high=self.n_aa))
                    if new_code == codes[pos]:
                        continue
                    old_code = codes[pos]
                    codes[pos] = new_code
                    new_bitstring = self._codes_to_bitstring(codes)
                    new_energy = self.compute_energy_from_bitstring(new_bitstring)
                    dE = new_energy - energy
                    if dE <= 0 or rng.random() < np.exp(-dE / max(1e-9, T)):
                        # accept
                        bitstring = new_bitstring
                        energy = new_energy
                        if energy < best_energy:
                            best_energy = energy
                            best_codes = codes.copy()
                    else:
                        # revert
                        codes[pos] = old_code
                    T *= cooling
        # Prepare result
        best_bitstring = self._codes_to_bitstring(best_codes)
        sequence = self.decode_solution(best_bitstring)
        return {
            'solution': best_bitstring,
            'repaired_solution': best_bitstring,
            'repaired_sequence': sequence,
            'repaired_cost': float(best_energy),
            'cost': float(best_energy),
            'costs': []
        }

    def repair_to_one_hot(self, bitstring: str) -> str:
        """Force exactly one '1' per position block by a simple deterministic rule."""
        bits = list(bitstring)
        for i in range(self.L):
            start = i * self.n_aa
            end = start + self.n_aa
            block = bits[start:end]
            ones = [j for j, b in enumerate(block) if b == '1']
            if len(ones) == 1:
                continue
            elif len(ones) == 0:
                # set first aa as chosen
                for j in range(self.n_aa):
                    block[j] = '1' if j == 0 else '0'
            else:
                # keep only the first '1'
                first = ones[0]
                for j in range(self.n_aa):
                    block[j] = '1' if j == first else '0'
            bits[start:end] = block
        return ''.join(bits)

    def repair_with_marginals(self, probs: np.ndarray) -> str:
        """Binary repair using per-qubit marginals p(z=1) from the full probability vector.
        For each position, set each bit to 1 if marginal>0.5, else 0. If the resulting code
        is >= n_aa, clamp to nearest valid (n_aa-1)."""
        num_states = probs.shape[0]
        num_qubits = self.n_qubits
        # Precompute marginals: p(qubit i = 1)
        marginals = np.zeros(num_qubits)
        for state in range(num_states):
            p = probs[state]
            bits = format(state, f'0{num_qubits}b')
            for i, b in enumerate(bits):
                if b == '1':
                    marginals[i] += p
        # Build repaired bits per position
        repaired = ['0'] * num_qubits
        for pos in range(self.L):
            code = 0
            for k in range(self.bits_per_pos):
                q = self.get_qubit_index(pos, k)
                bit = 1 if marginals[q] > 0.5 else 0
                if bit:
                    code |= (1 << k)
            if code >= self.n_aa:
                code = self.n_aa - 1
            # write bits back
            for k in range(self.bits_per_pos):
                q = self.get_qubit_index(pos, k)
                repaired[q] = '1' if ((code >> k) & 1) else '0'
        return ''.join(repaired)
    
    def _setup_qiskit(self):
        """Setup Qiskit quantum circuit and Hamiltonian"""
        if not QISKIT_AVAILABLE:
            raise ImportError("Qiskit not available")
        
        # Convert to Qiskit Pauli operators
        pauli_list = []
        for coeff, pauli_string in self.pauli_terms:
            pauli_list.append((pauli_string, coeff))
        
        self.qiskit_hamiltonian = PauliSumOp.from_list(pauli_list)
        self.qiskit_backend = AerSimulator()
        print(f"Qiskit Hamiltonian created with {len(pauli_list)} Pauli terms")
    
    def solve_qaoa_pennylane(self, p_layers: int = 2, max_iterations: int = 200, n_starts: int = 4, init_strategy: str = 'linear', warm_start: bool = True):
        """Solve using QAOA with PennyLane"""
        
        print(f"\n🔥 Solving with PennyLane QAOA (p={p_layers})...")
        
        def make_cost(p_local: int):
            @qml.qnode(self.dev)
            def _cost(params):
                for i in range(self.n_qubits):
                    qml.Hadamard(wires=i)
                for layer in range(p_local):
                    qml.ApproxTimeEvolution(self.hamiltonian, params[0][layer], 1)
                    beta = params[1][layer]
                    for w in range(self.n_qubits):
                        qml.RX(2 * beta, wires=w)
                return qml.expval(self.hamiltonian)
            return _cost
        
        def init_params(seed_offset=0):
            rng = np.random.default_rng(1234 + seed_offset)
            if init_strategy == 'linear':
                # Linear-ramp initialization per-layer
                layers = np.arange(1, p_layers + 1)
                betas0 = 0.8 * np.pi * (layers / (p_layers + 1))
                gammas0 = 0.8 * np.pi * (1 - (layers - 0.5) / p_layers)
            else:
                betas0 = rng.uniform(0.0, np.pi, size=p_layers)
                gammas0 = rng.uniform(0.0, 2 * np.pi, size=p_layers)
            return (
                qnp.array(gammas0, requires_grad=True),
                qnp.array(betas0, requires_grad=True),
            )

        best_params = None
        best_cost = np.inf
        best_costs_trace = []
        # Warm-start schedule: p=1 -> p_layers
        prev_params = None
        for p_stage in range(1, p_layers + 1):
            cost_function = make_cost(p_stage)
            stage_best_cost = np.inf
            stage_best_params = None
            # prepare starts
            starts = []
            if warm_start and prev_params is not None:
                # interpolate previous params (length p_stage-1) to p_stage by repeating last value
                prev_g, prev_b = prev_params
                g_ext = qnp.concatenate([prev_g, qnp.array([prev_g[-1]], requires_grad=True)])
                b_ext = qnp.concatenate([prev_b, qnp.array([prev_b[-1]], requires_grad=True)])
                starts.append((g_ext, b_ext))
            # random/linear starts
            for s in range(len(starts), n_starts):
                starts.append(init_params(s + p_stage * 100))
            # optimize each start
            for idx, params in enumerate(starts):
                optimizer = qml.AdamOptimizer(stepsize=0.1)
                costs = []
                for i in range(max_iterations // 2):
                    params, cost = optimizer.step_and_cost(cost_function, params)
                    costs.append(cost)
                optimizer_fine = qml.AdamOptimizer(stepsize=0.02)
                for i in range(max_iterations // 2, max_iterations):
                    params, cost = optimizer_fine.step_and_cost(cost_function, params)
                    costs.append(cost)
                print(f"p={p_stage} start {idx+1}/{len(starts)}: final cost {costs[-1]:.6f}")
                if costs[-1] < stage_best_cost:
                    stage_best_cost = costs[-1]
                    stage_best_params = params
                if p_stage == p_layers and (len(best_costs_trace) == 0 or costs[-1] < best_cost):
                    best_costs_trace = costs
            prev_params = stage_best_params
            best_params = stage_best_params
            best_cost = stage_best_cost
        
        # Get final measurement
        @qml.qnode(self.dev)
        def get_probabilities(params):
            for i in range(self.n_qubits):
                qml.Hadamard(wires=i)
            for p in range(p_layers):
                qml.ApproxTimeEvolution(self.hamiltonian, params[0][p], 1)
                beta = params[1][p]
                for w in range(self.n_qubits):
                    qml.RX(2 * beta, wires=w)
            return qml.probs(wires=range(self.n_qubits))
        
        probs = get_probabilities(best_params)
        best_bitstring = np.argmax(probs)
        best_solution = format(best_bitstring, f'0{self.n_qubits}b')
        # Repair with marginals and compute classical energy
        repaired_solution = self.repair_with_marginals(probs)
        repaired_sequence = self.decode_solution(repaired_solution)
        repaired_energy = self.compute_energy_from_bitstring(repaired_solution)
        
        print(f"✅ QAOA completed! Final cost: {best_cost:.6f}")
        print(f"Best solution probability: {max(probs):.4f}")
        print(f"➡️  Repaired sequence: {repaired_sequence} | Energy (classical): {repaired_energy:.6f}")
        
        # Draw the final QAOA circuit (text and matplotlib)
        try:
            circuit_text = qml.draw(cost_function)(best_params)
            print("\n📐 QAOA Circuit (text):\n")
            print(circuit_text)
        except Exception as e:
            print(f"Could not render text circuit: {e}")
        
        try:
            fig, ax = qml.draw_mpl(cost_function)(best_params)
            fig.suptitle("QAOA Circuit")
            plt.show()
        except Exception as e:
            print(f"Could not render matplotlib circuit: {e}")
        
        return {
            'solution': best_solution,
            'cost': best_cost,
            'costs': best_costs_trace,
            'probabilities': probs,
            'params': best_params,
            'repaired_solution': repaired_solution,
            'repaired_sequence': repaired_sequence,
            'repaired_cost': repaired_energy
        }
    
    def solve_vqe_pennylane(self, max_iterations: int = 100):
        """Solve using VQE with PennyLane"""
        
        print(f"\n⚡ Solving with PennyLane VQE...")
        
        @qml.qnode(self.dev)
        def cost_function(params):
            # Parameterized ansatz (RY rotations + entangling gates)
            for i in range(self.n_qubits):
                qml.RY(params[i], wires=i)
            
            # Entangling layers
            for i in range(0, self.n_qubits-1, 2):
                qml.CNOT(wires=[i, i+1])
            for i in range(1, self.n_qubits-1, 2):
                qml.CNOT(wires=[i, i+1])
            
            # Second layer of rotations
            for i in range(self.n_qubits):
                qml.RY(params[i + self.n_qubits], wires=i)
            
            return qml.expval(self.hamiltonian)
        
        # Random initialization
        params = qnp.array(np.random.uniform(0, 2*np.pi, 2 * self.n_qubits), requires_grad=True)
        
        # Optimize
        optimizer = qml.AdamOptimizer(stepsize=0.1)
        costs = []
        
        for i in range(max_iterations):
            params, cost = optimizer.step_and_cost(cost_function, params)
            costs.append(cost)
            if i % 20 == 0:
                print(f"Iteration {i}: Cost = {cost:.6f}")
        
        # Get measurement
        @qml.qnode(self.dev)
        def measure_circuit(params):
            for i in range(self.n_qubits):
                qml.RY(params[i], wires=i)
            for i in range(0, self.n_qubits-1, 2):
                qml.CNOT(wires=[i, i+1])
            for i in range(1, self.n_qubits-1, 2):
                qml.CNOT(wires=[i, i+1])
            for i in range(self.n_qubits):
                qml.RY(params[i + self.n_qubits], wires=i)
            return qml.probs(wires=range(self.n_qubits))
        
        probs = measure_circuit(params)
        best_bitstring = np.argmax(probs)
        best_solution = format(best_bitstring, f'0{self.n_qubits}b')
        repaired_solution = self.repair_to_one_hot(best_solution)
        repaired_sequence = self.decode_solution(repaired_solution)
        repaired_energy = self.compute_energy_from_bitstring(repaired_solution)
        
        print(f"✅ VQE completed! Final cost: {costs[-1]:.6f}")
        
        return {
            'solution': best_solution,
            'cost': costs[-1],
            'costs': costs,
            'probabilities': probs,
            'params': params,
            'repaired_solution': repaired_solution,
            'repaired_sequence': repaired_sequence,
            'repaired_cost': repaired_energy
        }
    
    def decode_solution(self, bitstring: str) -> str:
        """Convert binary-encoded solution to amino acid sequence."""
        sequence = []
        for i in range(self.L):
            code = 0
            for k in range(self.bits_per_pos):
                q = self.get_qubit_index(i, k)
                if bitstring[q] == '1':
                    code |= (1 << k)
            if code < self.n_aa:
                sequence.append(self.amino_acids[code])
            else:
                sequence.append('X')  # invalid code
        return ''.join(sequence)
    
    def analyze_solution(self, result: dict):
        """Analyze the quantum solution"""
        
        # Prefer repaired one-hot solution if available
        solution = result.get('repaired_solution', result['solution'])
        sequence = result.get('repaired_sequence', self.decode_solution(solution))
        
        print(f"\n🧬 QUANTUM SOLUTION ANALYSIS 🧬")
        print(f"Binary solution: {solution}")
        print(f"Decoded sequence: {sequence}")
        print(f"Final energy: {result['cost']:.6f}")
        
        # Check constraint satisfaction for binary encoding (invalid code)
        violations = sequence.count('X')
        
        print(f"Constraint violations: {violations}/{self.L}")
        
        # Analyze sequence properties
        if violations == 0:
            hydrophobic_residues = sum(1 for aa in sequence if aa in ['A', 'L', 'F', 'W'])
            charged_residues = sum(1 for aa in sequence if aa in ['E', 'K', 'R', 'D'])
            
            print(f"Hydrophobic residues: {hydrophobic_residues}/{len(sequence)}")
            print(f"Charged residues: {charged_residues}/{len(sequence)}")
        
        return sequence, violations
    
    def plot_optimization(self, costs: List[float]):
        """Plot optimization convergence"""
        
        plt.figure(figsize=(10, 6))
        plt.plot(costs, 'b-', linewidth=2)
        plt.xlabel('Iteration')
        plt.ylabel('Energy')
        plt.title('Quantum Optimization Convergence')
        plt.grid(True, alpha=0.3)
        plt.show()

    def plot_alpha_helix_wheel(self, sequence: str):
        """Plot an alpha-helix wheel of the given amino acid sequence with color coding.
        Colors:
            - Polar (uncharged): green
            - Nonpolar (hydrophobic): brown
            - Negatively charged: red
            - Positively charged: blue
        """
        # Simplified classification for included amino acids
        polar = set(['S','T','N','Q','Y','C','G'])
        nonpolar = set(['A','V','L','I','M','F','W','P'])
        negative = set(['D','E'])
        positive = set(['K','R','H'])

        color_map = {}
        for aa in sequence:
            if aa in negative:
                color_map[aa] = 'red'
            elif aa in positive:
                color_map[aa] = 'blue'
            elif aa in nonpolar:
                color_map[aa] = '#8B4513'  # brown
            elif aa in polar:
                color_map[aa] = 'green'
            else:
                color_map[aa] = 'gray'

        # Helical wheel: residues separated by ~100 degrees
        angle_increment = np.deg2rad(100.0)
        radius = 1.0
        angles = [i * angle_increment for i in range(len(sequence))]
        xs = [radius * np.cos(a) for a in angles]
        ys = [radius * np.sin(a) for a in angles]

        plt.figure(figsize=(7, 7))
        for i, aa in enumerate(sequence):
            plt.scatter(xs[i], ys[i], s=600, color=color_map[aa], edgecolors='k', zorder=3)
            # Amino acid letter
            plt.text(xs[i], ys[i], aa, ha='center', va='center', fontsize=12, weight='bold', color='white', zorder=4)
            # Residue index (1-based), slightly offset radially outward
            r_idx = radius + 0.12
            ang_i = angles[i]
            xi = r_idx * np.cos(ang_i)
            yi = r_idx * np.sin(ang_i)
            plt.text(xi, yi, f"{i+1}", ha='center', va='center', fontsize=9, color='black', zorder=5)

        # Connect residues in sequence order to show the helical path
        for i in range(len(sequence) - 1):
            plt.plot([xs[i], xs[i+1]], [ys[i], ys[i+1]], color='k', alpha=0.35, linewidth=1.5, zorder=2)
        # Draw circle
        circle = plt.Circle((0, 0), radius, color='k', fill=False, alpha=0.3)
        ax = plt.gca()
        ax.add_artist(circle)
        
        # Draw membrane orientation if wheel mode is active (or membrane params provided)
        try:
            if getattr(self, 'membrane_mode', 'span') == 'wheel':
                # Membrane-facing direction is angle 0 after phase; draw band ±halfwidth
                phase = np.deg2rad(getattr(self, 'wheel_phase_deg', 0.0))
                halfw = np.deg2rad(getattr(self, 'wheel_halfwidth_deg', 40.0))
                # Two boundary lines at +halfw and -halfw
                for sign in [+1, -1]:
                    ang = sign * halfw
                    x = radius * np.cos(ang)
                    y = radius * np.sin(ang)
                    # rotate by phase
                    xr = x * np.cos(phase) - y * np.sin(phase)
                    yr = x * np.sin(phase) + y * np.cos(phase)
                    ax.plot([0, xr], [0, yr], color='gray', alpha=0.6, linestyle='--', linewidth=2, zorder=1)
                # Shade membrane-facing wedge
                wedge = mpatches.Wedge(center=(0,0), r=radius, theta1=np.rad2deg(-halfw)+np.rad2deg(phase),
                                       theta2=np.rad2deg(halfw)+np.rad2deg(phase), facecolor='#FFE4B5', alpha=0.3)
                ax.add_patch(wedge)
                # Annotate regions
                # Membrane side (lipids)
                mid_ang = phase
                xm = 1.15 * radius * np.cos(mid_ang)
                ym = 1.15 * radius * np.sin(mid_ang)
                ax.text(xm, ym, 'Membrane (lipids)', ha='center', va='center', fontsize=10, color='#8B4513', weight='bold')
                # Water side (opposite)
                xa = 1.15 * radius * np.cos(mid_ang + np.pi)
                ya = 1.15 * radius * np.sin(mid_ang + np.pi)
                ax.text(xa, ya, 'Water', ha='center', va='center', fontsize=10, color='teal', weight='bold')
        except Exception:
            pass
        ax.set_aspect('equal')
        ax.set_xlim(-1.3, 1.3)
        ax.set_ylim(-1.3, 1.3)
        ax.axis('off')
        plt.title('Alpha-Helix Wheel')
        plt.show()


def describe_qaoa(n_qubits: int, p_layers: int):
    print("🔥 QAOA CIRCUIT STRUCTURE:")
    print(f"   • Qubits: {n_qubits}")
    print(f"   • p-layers: {p_layers}")
    print(f"   • Gate sequence:")
    print(f"     1. Hadamard on all qubits → superposition")
    print(f"     2. For each p-layer:")
    print(f"        - Cost Hamiltonian evolution (problem-specific)")
    print(f"        - Mixer Hamiltonian evolution (RX rotations)")

def run_quantum_protein_design(sequence_length: int = 3, amino_acids: Optional[List[str]] = None,
                               membrane_span: Optional[Tuple[int,int]] = None,
                               membrane_charge: str = 'neu',
                               lambda_charge: float = 0.0,
                               lambda_env: float = 0.0,
                               lambda_mu: float = 0.0,
                               membrane_positions: Optional[List[int]] = None,
                               membrane_mode: str = 'span',
                               wheel_phase_deg: float = 0.0,
                               wheel_halfwidth_deg: float = 40.0,
                               classical: bool = False):
    """Complete example of quantum protein design"""
    
    print("🚀 QUANTUM PROTEIN SEQUENCE DESIGN 🚀")
    print("="*50)
    
    # Create small example (4 amino acids, 3 positions = 12 qubits)
    designer = QuantumProteinDesign(
        sequence_length=sequence_length,
        amino_acids=amino_acids if amino_acids is not None else ['A', 'L', 'E', 'K', 'W'],
        quantum_backend='pennylane',
        membrane_span=membrane_span,
        membrane_charge=membrane_charge,
        lambda_charge=lambda_charge,
        lambda_env=lambda_env,
        lambda_mu=lambda_mu,
        membrane_positions=membrane_positions,
        membrane_mode=membrane_mode,
        wheel_phase_deg=wheel_phase_deg,
        wheel_halfwidth_deg=wheel_halfwidth_deg,
    )
    
    print(f"\nTotal qubits required: {designer.n_qubits}")
    print(f"Sequence length (L): {designer.L} | Amino acids (|Σ|): {designer.n_aa} | bits/pos: {designer.bits_per_pos}")
    print(f"QUBO matrix would be: {designer.n_qubits} × {designer.n_qubits}")
    
    # Print qubit mapping for binary encoding (qubit -> position:bit)
    print("\nQubit mapping (qubit -> position:bit):")
    for i in range(designer.L):
        row = []
        for b in range(designer.bits_per_pos):
            q = designer.get_qubit_index(i, b)
            row.append(f"{q}->{i}:b{b}")
        print("  " + "  ".join(row))
    
    # Describe QAOA circuit dynamically
    describe_qaoa(n_qubits=designer.n_qubits, p_layers=2)
    
    if classical:
        print("\n🧮 Solving classically (QUBO heuristic)...")
        qaoa_result = designer.solve_classical_qubo()
        sequence_qaoa, violations_qaoa = designer.analyze_solution(qaoa_result)
    else:
        # Solve with QAOA
        qaoa_result = designer.solve_qaoa_pennylane(p_layers=2, max_iterations=200)
        sequence_qaoa, violations_qaoa = designer.analyze_solution(qaoa_result)
        # Plot helix for repaired QAOA sequence
        designer.plot_alpha_helix_wheel(qaoa_result['repaired_sequence'])
    
    # Report results
    print(f"\n📊 QAOA RESULT 📊")
    print(f"QAOA solution: {sequence_qaoa} (violations: {violations_qaoa})")
    print(f"QAOA repaired: {qaoa_result['repaired_sequence']} | E(classical): {qaoa_result['repaired_cost']:.6f}")
    print(f"QAOA energy (expval): {qaoa_result['cost']:.6f}")
    
    # Plot convergence (only if QAOA)
    if not classical and qaoa_result.get('costs'):
        designer.plot_optimization(qaoa_result['costs'])
    
    return designer, qaoa_result


def demonstrate_scaling():
    """Demonstrate how qubit requirements scale"""
    
    print("\n📏 QUBIT SCALING ANALYSIS 📏")
    print("="*40)
    
    lengths = [2, 3, 4, 5, 10]
    n_amino_acids = [4, 8, 20]  # Different amino acid sets
    
    print("Sequence | AA Types | Total Qubits | Feasible?")
    print("-" * 45)
    
    for L in lengths:
        for n_aa in n_amino_acids:
            total_qubits = L * n_aa
            feasible = "✅" if total_qubits <= 20 else "❌" if total_qubits <= 50 else "🚫"
            print(f"{L:8d} | {n_aa:8d} | {total_qubits:11d} | {feasible}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quantum peptide design with QAOA")
    parser.add_argument("--length", "-L", type=int, default=6, help="Peptide length (number of residues)")
    parser.add_argument("--residues", "-R", type=str, default='VLT', help="Allowed residues, e.g. 'A,L,E,K' or 'ALEK'")
    parser.add_argument("--membrane", "-M", type=str, default=None, help="Membrane span as start:end (0-based, inclusive), e.g. '1:4'")
    parser.add_argument("--membrane_positions", type=str, default=None, help="Explicit membrane-facing positions, e.g. '0,2,5' (0-based)")
    parser.add_argument("--membrane_mode", type=str, default='span', choices=['span','set','wheel'], help="Membrane selection mode")
    parser.add_argument("--wheel_phase_deg", type=float, default=0.0, help="Helical wheel phase in degrees (membrane-facing direction)")
    parser.add_argument("--wheel_halfwidth_deg", type=float, default=40.0, help="Half-width (deg) around membrane-facing direction")
    parser.add_argument("--membrane_charge", type=str, default='neu', choices=['neg','pos','neu'], help="Membrane charge: neg/pos/neu")
    parser.add_argument("--lambda_charge", type=float, default=0.0, help="Weight for membrane charge term")
    parser.add_argument("--lambda_env", type=float, default=0.0, help="Weight for environment hydrophobicity term")
    parser.add_argument("--lambda_mu", type=float, default=0.0, help="Weight for hydrophobic moment alignment term")
    parser.add_argument("--classical", action='store_true', help="Solve classically (no QAOA)")
    args = parser.parse_args()

    if args.residues is None:
        aa_list = None
    else:
        s = args.residues.strip().upper()
        if "," in s:
            aa_list = [t.strip() for t in s.split(",") if t.strip()]
        else:
            aa_list = [c for c in s if c.strip()]

    mem_span = None
    if args.membrane:
        try:
            a, b = args.membrane.split(":")
            mem_span = (int(a), int(b))
        except Exception:
            print("Invalid --membrane format. Use start:end, e.g. 1:4")
            sys.exit(1)

    mem_positions = None
    if args.membrane_positions:
        try:
            mem_positions = [int(t) for t in args.membrane_positions.split(',') if t.strip()]
        except Exception:
            print("Invalid --membrane_positions. Use comma-separated indices, e.g. 0,2,5")
            sys.exit(1)

    # Run the quantum protein design
    designer, qaoa_result = run_quantum_protein_design(
        sequence_length=args.length,
        amino_acids=aa_list,
        membrane_span=mem_span,
        membrane_charge=args.membrane_charge,
        lambda_charge=args.lambda_charge,
        lambda_env=args.lambda_env,
        lambda_mu=args.lambda_mu,
        membrane_positions=mem_positions,
        membrane_mode=args.membrane_mode,
        wheel_phase_deg=args.wheel_phase_deg,
        wheel_halfwidth_deg=args.wheel_halfwidth_deg,
        classical=args.classical,
    )
    
    # Show scaling analysis
    demonstrate_scaling()
    

# python cursor_peptide_seq.py -L 6 -R VQ  --membrane_mode wheel --wheel_phase_deg 0 --wheel_halfwidth_deg 30  --lambda_env 0.6 --lambda_mu 0.4 --lambda_charge 0.3 --membrane_charge neg
