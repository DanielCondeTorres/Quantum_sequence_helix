# core/solvers.py
import numpy as np
from pennylane import numpy as qnp
import pennylane as qml
from typing import List, Dict, Any, Optional
import itertools
from utils.general_utils import decode_solution, compute_energy_from_bitstring, get_qubit_index

class QAOASolver:
    def __init__(self, cost_hamiltonian, n_qubits, pauli_terms, amino_acids, L, bits_per_pos):
        self.cost_hamiltonian = cost_hamiltonian
        self.n_qubits = n_qubits
        self.pauli_terms = pauli_terms
        self.amino_acids = amino_acids
        self.L = L
        self.bits_per_pos = bits_per_pos
        self.dev = qml.device('lightning.qubit', wires=self.n_qubits)

    def _make_cost(self, p_local: int):
        @qml.qnode(self.dev)
        def _cost(params):
            for i in range(self.n_qubits):
                qml.Hadamard(wires=i)
            for layer in range(p_local):
                qml.ApproxTimeEvolution(self.cost_hamiltonian, params[0][layer], 1)
                beta = params[1][layer]
                for w in range(self.n_qubits):
                    qml.RX(2 * beta, wires=w)
            return qml.expval(self.cost_hamiltonian)
        return _cost

    def _init_params(self, p_layers: int, init_strategy: str, seed_offset=0):
        rng = np.random.default_rng(1234 + seed_offset)
        if init_strategy == 'linear':
            layers = np.arange(1, p_layers + 1)
            betas0 = 0.8 * np.pi * (layers / (p_layers + 1))
            gammas0 = 0.8 * np.pi * (1 - (layers - 0.5) / p_layers)
        else:
            betas0 = rng.uniform(0.0, np.pi, size=p_layers)
            gammas0 = rng.uniform(0.0, 2 * np.pi, size=p_layers)
        return (qnp.array(gammas0, requires_grad=True), qnp.array(betas0, requires_grad=True))

    def solve(self, p_layers: int = 2, max_iterations: int = 200, n_starts: int = 4, init_strategy: str = 'linear', warm_start: bool = True):
        print(f"\n🔥 Solving with PennyLane QAOA (p={p_layers})...")
        best_params = None
        best_cost = np.inf
        best_costs_trace = []
        prev_params = None
        
        for p_stage in range(1, p_layers + 1):
            cost_function = self._make_cost(p_stage)
            stage_best_cost = np.inf
            stage_best_params = None
            starts = []
            if warm_start and prev_params is not None:
                prev_g, prev_b = prev_params
                g_ext = qnp.concatenate([prev_g, qnp.array([prev_g[-1]], requires_grad=True)])
                b_ext = qnp.concatenate([prev_b, qnp.array([prev_b[-1]], requires_grad=True)])
                starts.append((g_ext, b_ext))
            for s in range(len(starts), n_starts):
                starts.append(self._init_params(p_stage, init_strategy, s + p_stage * 100))
            
            optimizer = qml.AdamOptimizer(stepsize=0.1)
            optimizer_fine = qml.AdamOptimizer(stepsize=0.02)
            for idx, params in enumerate(starts):
                costs = []
                for i in range(max_iterations // 2):
                    params, cost = optimizer.step_and_cost(cost_function, params)
                    costs.append(cost)
                
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
        
        @qml.qnode(self.dev)
        def get_probabilities(params):
            for i in range(self.n_qubits): qml.Hadamard(wires=i)
            for p in range(p_layers):
                qml.ApproxTimeEvolution(self.cost_hamiltonian, params[0][p], 1)
                beta = params[1][p]
                for w in range(self.n_qubits): qml.RX(2 * beta, wires=w)
            return qml.probs(wires=range(self.n_qubits))
        
        probs = get_probabilities(best_params)
        best_bitstring_int = np.argmax(probs)
        best_bitstring_str = format(best_bitstring_int, f'0{self.n_qubits}b')
        repaired_solution = self._repair_with_marginals(probs)
        repaired_sequence = decode_solution(repaired_solution, self.L, self.bits_per_pos, self.amino_acids)
        repaired_energy = compute_energy_from_bitstring(repaired_solution, self.pauli_terms)
        
        print(f"✅ QAOA completed! Final cost: {best_cost:.6f}")
        print(f"Best solution bitstring: {best_bitstring_str}")
        
        return {
            'bitstring': best_bitstring_str,
            'energy': best_cost,
            'costs': best_costs_trace,
            'repaired_sequence': repaired_sequence,
            'repaired_cost': repaired_energy
        }

    def _repair_with_marginals(self, probs: np.ndarray) -> str:
        """Binary repair using per-qubit marginals p(z=1) from the full probability vector."""
        num_states = probs.shape[0]
        num_qubits = self.n_qubits
        
        marginals = np.zeros(num_qubits)
        for state in range(num_states):
            p = probs[state]
            bits = format(state, f'0{num_qubits}b')
            for i, b in enumerate(bits):
                if b == '1':
                    marginals[i] += p
        
        repaired = ['0'] * num_qubits
        for pos in range(self.L):
            code = 0
            for k in range(self.bits_per_pos):
                q = get_qubit_index(pos, k, self.bits_per_pos)
                if q < len(marginals) and marginals[q] > 0.5:
                    repaired[q] = '1'
                    code |= (1 << k)
            
            if code >= len(self.amino_acids):
                code = len(self.amino_acids) - 1
            
            for k in range(self.bits_per_pos):
                q = get_qubit_index(pos, k, self.bits_per_pos)
                repaired[q] = '1' if ((code >> k) & 1) else '0'

        return ''.join(repaired)
        
class ClassicalSolver:
    def __init__(self, L: int, n_aa: int, bits_per_pos: int, pauli_terms: List[Dict[str, Any]], amino_acids: List[str]):
        self.L = L
        self.n_aa = n_aa
        self.bits_per_pos = bits_per_pos
        self.pauli_terms = pauli_terms
        self.amino_acids = amino_acids

    def solve(self):
        print("\n🧮 Solving classically...")
        best_bitstring = None
        best_energy = float('inf')
        
        num_qubits = self.L * self.bits_per_pos
        all_bitstrings = list(itertools.product('01', repeat=num_qubits))
        
        for bitstring_tuple in all_bitstrings:
            bitstring = "".join(bitstring_tuple)
            energy = compute_energy_from_bitstring(bitstring, self.pauli_terms)
            
            if energy < best_energy:
                best_energy = energy
                best_bitstring = bitstring
        
        repaired_sequence = decode_solution(best_bitstring, self.L, self.bits_per_pos, self.amino_acids)
        
        print(f"✅ Classical solver completed! Final energy: {best_energy:.6f}")
        print(f"Best solution bitstring: {best_bitstring}")
        
        return {
            'bitstring': best_bitstring,
            'energy': best_energy,
            'costs': [],
            'repaired_sequence': repaired_sequence,
            'repaired_cost': best_energy
        }