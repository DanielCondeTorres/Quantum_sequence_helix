import os
os.environ['MPLBACKEND'] = 'Agg'  # Forzar backend sin GUI
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pennylane import numpy as qnp
import pennylane as qml
from typing import Dict, List, Tuple, Optional, Any
import itertools
import argparse
import sys
import time
from visualization.plot_utils import ProteinPlotter
from core.hamiltonian_builder import HamiltonianBuilder
from core.solvers import QAOASolver, ClassicalSolver
from core.solvers_qiskit import CustomQAOA, CustomVQE, HybridMultiStartOptimizer
from collections import defaultdict
# Qiskit imports
try:
    from qiskit import QuantumCircuit
    from qiskit_algorithms import QAOA, VQE
    from qiskit_algorithms.optimizers import SPSA, COBYLA
    from qiskit.quantum_info import SparsePauliOp, Statevector
    try:
        from qiskit_aer.primitives import Estimator, Sampler
        QISKIT_PRIMITIVES_SOURCE = 'qiskit_aer.primitives'
    except ImportError:
        try:
            from qiskit.primitives import Estimator, Sampler
            QISKIT_PRIMITIVES_SOURCE = 'qiskit.primitives'
        except ImportError:
            QISKIT_PRIMITIVES_SOURCE = None
    QISKIT_AVAILABLE = True
    print(f"✅ Qiskit available. Primitives source: {QISKIT_PRIMITIVES_SOURCE}")
except ImportError as e:
    QISKIT_AVAILABLE = False
    QISKIT_PRIMITIVES_SOURCE = None
    print(f"❌ Qiskit not available: {e}")



class QuantumProteinDesign:
    def __init__(self, sequence_length: int, amino_acids: List[str] = None, 
                 quantum_backend: str = 'pennylane', shots: int = 5000, 
                 use_statevector: bool = False, **kwargs):
        self.L = sequence_length
        self.amino_acids = amino_acids
        self.n_aa = len(amino_acids)
        self.bits_per_pos = max(1, int(np.ceil(np.log2(self.n_aa))))
        self.n_qubits = self.L * self.bits_per_pos
        self.backend = quantum_backend
        self.shots = shots
        self.use_statevector = use_statevector
        self.kwargs = kwargs
        
        output_dir = kwargs.get('output_dir', 'output')
        self.plotter = ProteinPlotter(output_dir=output_dir)
        os.makedirs(output_dir, exist_ok=True)
        print(f"Output directory: {output_dir}")
        print(f"Absolute output path: {os.path.abspath(output_dir)}")
        
        print("🧬 QUANTUM PROTEIN DESIGN SETUP 🧬")
        print(f"Sequence length: {self.L}")
        print(f"Amino acids: {self.amino_acids}")
        print(f"Bits per position: {self.bits_per_pos}")
        print(f"Required qubits: {self.n_qubits}")
        print(f"Quantum backend: {self.backend}")
        print(f"Number of shots: {self.shots}")
        print(f"Use statevector: {self.use_statevector}")
        print("="*50)

        self.hamiltonian_builder = HamiltonianBuilder(
            L=self.L,
            amino_acids=self.amino_acids,
            bits_per_pos=self.bits_per_pos,
            n_qubits=self.n_qubits,
            **self.kwargs
        )
        self.pauli_terms, self.cost_hamiltonian = self.hamiltonian_builder.build_hamiltonian(self.backend)
        print("Hamiltonian type:", type(self.cost_hamiltonian))
        print("Sample Pauli terms:", self.pauli_terms[:5])
        if self.cost_hamiltonian is None:
            print("Error: Hamiltonian is None. Check hamiltonian_builder.py logs.")
            raise ValueError("Cannot proceed with None Hamiltonian")
        if not self.pauli_terms:
            print("Error: No Pauli terms constructed. Check hamiltonian_builder.py logs.")
            raise ValueError("Cannot proceed with empty Pauli terms")
        
        for coeff, pauli in self.pauli_terms:
            if abs(coeff.imag) > 1e-10:
                print(f"Warning: Complex coefficient detected: {coeff} for Pauli {pauli}")
        self.pauli_terms = [(float(coeff.real), pauli) for coeff, pauli in self.pauli_terms]
        
        if self.backend == 'pennylane':
            print("Hamiltonian coefficients:", [float(coeff) for coeff in self.cost_hamiltonian.coeffs])
            print("Hamiltonian operators:", [str(op) for op in self.cost_hamiltonian.ops])
            print("Number of Pauli terms:", len(self.cost_hamiltonian.coeffs))
            if any(np.iscomplex(coeff) for coeff in self.cost_hamiltonian.coeffs):
                print("Warning: Hamiltonian contains complex coefficients!")
                self.cost_hamiltonian = qml.Hamiltonian(
                    [float(coeff.real) for coeff in self.cost_hamiltonian.coeffs],
                    self.cost_hamiltonian.ops
                )
                print("Forced coefficients to real values.")
        
        self.qaoa_solver = QAOASolver(
            cost_hamiltonian=self.cost_hamiltonian,
            n_qubits=self.n_qubits,
            pauli_terms=self.pauli_terms,
            amino_acids=self.amino_acids,
            L=self.L,
            bits_per_pos=self.bits_per_pos,
            shots=self.shots
        )
        self.classical_solver = ClassicalSolver(
            L=self.L,
            n_aa=self.n_aa,
            bits_per_pos=self.bits_per_pos,
            pauli_terms=self.pauli_terms,
            amino_acids=self.amino_acids
        )

    def decode_solution(self, bitstring: str) -> str:
        """Decodes a binary string back into a protein sequence."""
        print(f"Decoding bitstring: {bitstring}")
        decoded_sequence = ""
        for i in range(self.L):
            pos_code_str = bitstring[i*self.bits_per_pos:(i+1)*self.bits_per_pos]
            try:
                pos_code_int = int(pos_code_str, 2)
                if pos_code_int < self.n_aa:
                    decoded_sequence += self.amino_acids[pos_code_int]
                else:
                    decoded_sequence += 'X'
            except ValueError:
                print(f"Invalid bitstring segment at position {i}: {pos_code_str}")
                decoded_sequence += 'X'
        print(f"Decoded sequence: {decoded_sequence}")
        return decoded_sequence

    def compute_energy_from_bitstring(self, bitstring: str) -> float:
        """Compute classical energy of a computational-basis bitstring under Z-only Hamiltonian."""
        z_vals = np.array([1 if b == '0' else -1 for b in bitstring])
        energy = 0.0
        for coeff, pauli_string in self.pauli_terms:
            prod = 1.0
            for i, p in enumerate(pauli_string):
                if p == 'Z':
                    prod *= z_vals[i]
            energy += coeff * prod
        return float(energy)

    def _mask_invalid_probabilities(self, probs: np.ndarray) -> np.ndarray:
        """Zero out probabilities of states that decode to invalid amino-acid codes (>= n_aa)."""
        if probs.size == 0:
            return probs
        masked = probs.copy()
        n_bits = self.n_qubits
        for idx in range(masked.size):
            bitstr = format(idx, f'0{n_bits}b')
            is_valid = True
            for pos in range(self.L):
                start = pos * self.bits_per_pos
                end = start + self.bits_per_pos
                code = int(bitstr[start:end], 2)
                if code >= self.n_aa:
                    is_valid = False
                    break
            if not is_valid:
                masked[idx] = 0.0
        return masked

    def solve_qaoa_pennylane(self, p_layers: int = 2, max_iterations: int = 1000, n_starts: int = 2, init_strategy: str = 'linear', warm_start: bool = True) -> Dict[str, Any]:
        print(f"\n🔥 Solving with PennyLane QAOA (p={p_layers})...")
        # Implementation remains unchanged
        def make_cost(p_local: int):
            @qml.qnode(self.qaoa_solver.dev)
            def _cost(params):
                gammas, betas = params
                for i in range(self.n_qubits):
                    qml.Hadamard(wires=i)
                for p in range(p_local):
                    qml.templates.ApproxTimeEvolution(self.qaoa_solver.cost_hamiltonian, gammas[p], n=1)
                    for w in range(self.n_qubits):
                        qml.RX(2 * betas[p], wires=w)
                return qml.expval(self.qaoa_solver.cost_hamiltonian)
            return _cost
        
        def init_params(p_local: int, seed_offset=0):
            rng = np.random.default_rng(1234 + seed_offset)
            if init_strategy == 'linear':
                layers = np.arange(1, p_local + 1)
                betas0 = 0.8 * np.pi * (layers / (p_local + 1))
                gammas0 = 0.8 * np.pi * (1 - (layers - 0.5) / p_local)
            else:
                betas0 = rng.uniform(0.0, np.pi, size=p_local)
                gammas0 = rng.uniform(0.0, 2 * np.pi, size=p_local)
            return (qnp.array(gammas0, requires_grad=True), qnp.array(betas0, requires_grad=True))
        
        best_params = None
        best_cost = np.inf
        best_costs_trace = []
        prev_params = None
        
        for p_stage in range(1, p_layers + 1):
            cost_function = make_cost(p_stage)
            stage_best_cost = np.inf
            stage_best_params = None
            starts = []
            if warm_start and prev_params is not None:
                prev_g, prev_b = prev_params
                g_ext = qnp.concatenate([prev_g, qnp.array([prev_g[-1]], requires_grad=True)])
                b_ext = qnp.concatenate([prev_b, qnp.array([prev_b[-1]], requires_grad=True)])
                starts.append((g_ext, b_ext))
            for s in range(len(starts), n_starts):
                starts.append(init_params(p_stage, s + p_stage * 100))
            
            for idx, params in enumerate(starts):
                gammas, betas = params
                params_flat = qnp.hstack([gammas, betas])
                optimizer = qml.AdamOptimizer(stepsize=0.1)
                costs = []
                for i in range(max_iterations // 2):
                    n_gammas = p_stage
                    gammas_new = params_flat[:n_gammas]
                    betas_new = params_flat[n_gammas:]
                    params_new = (gammas_new, betas_new)
                    params_flat, cost = optimizer.step_and_cost(lambda p: cost_function((p[:n_gammas], p[n_gammas:])), params_flat)
                    costs.append(float(cost))
                optimizer_fine = qml.AdamOptimizer(stepsize=0.02)
                for i in range(max_iterations // 2, max_iterations):
                    n_gammas = p_stage
                    gammas_new = params_flat[:n_gammas]
                    betas_new = params_flat[n_gammas:]
                    params_new = (gammas_new, betas_new)
                    params_flat, cost = optimizer_fine.step_and_cost(lambda p: cost_function((p[:n_gammas], p[n_gammas:])), params_flat)
                    costs.append(float(cost))
                print(f"p={p_stage} start {idx+1}/{len(starts)}: final cost {costs[-1]:.6f}")
                if costs[-1] < stage_best_cost:
                    stage_best_cost = costs[-1]
                    stage_best_params = (gammas_new, betas_new)
                if p_stage == p_layers and (len(best_costs_trace) == 0 or costs[-1] < best_cost):
                    best_costs_trace = costs
            prev_params = stage_best_params
            best_params = stage_best_params
            best_cost = stage_best_cost
        
        self.plotter.save_pennylane_circuit(make_cost(p_layers), best_params, 'qaoa_circuit_pennylane.png')
            
        @qml.qnode(self.qaoa_solver.dev)
        def get_probabilities(params):
            gammas, betas = params
            for i in range(self.n_qubits):
                qml.Hadamard(wires=i)
            for p in range(p_layers):
                qml.templates.ApproxTimeEvolution(self.qaoa_solver.cost_hamiltonian, gammas[p], n=1)
                for w in range(self.n_qubits):
                    qml.RX(2 * betas[p], wires=w)
            return qml.probs(wires=range(self.n_qubits))
        
        probs = get_probabilities(best_params)
        print(f"Number of shots used: {self.qaoa_solver.dev.shots}")
        print("Probabilities:", probs)
        print("Number of probabilities:", len(probs))
        if len(probs) == 0:
            print("Warning: No probabilities computed. Check Hamiltonian or circuit.")
            return {
                'bitstring': '',
                'energy': float('inf'),
                'costs': best_costs_trace,
                'repaired_sequence': ''
            }

        best_bitstring_int = np.argmax(probs)
        best_bitstring = format(best_bitstring_int, f'0{self.n_qubits}b')
        repaired_solution = self.qaoa_solver._repair_with_marginals(probs)
        repaired_sequence = self.decode_solution(repaired_solution)
        repaired_energy = self.qaoa_solver.compute_energy_from_bitstring(repaired_solution)
        
        print(f"✅ QAOA completed! Final cost: {best_cost:.6f}")
        print(f"Best solution probability: {max(probs):.4f}")
        print(f"➡️ Repaired sequence: {repaired_sequence} | Energy (classical): {repaired_energy:.6f}")
        
        self.plotter.plot_prob_with_sequences(
                    probs=probs,
                    decoder_fn=self.decode_solution,
                    n_qubits=self.n_qubits,
                    solver_name="QAOA",
                    top_k=20
                    )
        return {
            'bitstring': best_bitstring,
            'energy': best_cost,
            'costs': best_costs_trace,
            'repaired_sequence': repaired_sequence,
            'repaired_cost': repaired_energy
        }

    def solve_vqe_pennylane(self, layers: int = 6, max_iterations: int = 1000, n_starts: int = 8) -> Dict[str, Any]:
        print(f"\n🔥 Solving with PennyLane VQE (layers={layers})...")
        # Implementation remains unchanged
        dev = qml.device('default.qubit', wires=self.n_qubits, shots=None)
        
        def ansatz(params, wires):
            for l in range(layers):
                for i in range(len(wires)):
                    qml.RX(params[l * len(wires) + i], wires=i)
                for i in range(len(wires) - 1):
                    qml.CNOT(wires=[i, i + 1])
        
        @qml.qnode(dev, diff_method='parameter-shift')
        def cost_function(params):
            ansatz(params, wires=range(self.n_qubits))
            return qml.expval(self.cost_hamiltonian)
        
        shape = (layers * self.n_qubits,)
        best_cost = np.inf
        best_params = None
        best_costs_trace = []
        
        for start in range(n_starts):
            params = qnp.array(np.random.uniform(0, 2 * np.pi, shape), requires_grad=True)
            optimizer = qml.AdamOptimizer(stepsize=0.1)
            costs = []
            for i in range(max_iterations // 2):
                params, cost = optimizer.step_and_cost(cost_function, params)
                costs.append(float(cost))
                if i % 10 == 0:
                    print(f"Start {start+1}/{n_starts}, Iteration {i}: Cost = {cost:.6f}")
            optimizer_fine = qml.AdamOptimizer(stepsize=0.02)
            for i in range(max_iterations // 2, max_iterations):
                params, cost = optimizer_fine.step_and_cost(cost_function, params)
                costs.append(float(cost))
                if i % 10 == 0:
                    print(f"Start {start+1}/{n_starts}, Iteration {i}: Cost = {cost:.6f}")
            print(f"Start {start+1}/{n_starts}: final cost {costs[-1]:.6f}")
            if costs[-1] < best_cost:
                best_cost = costs[-1]
                best_params = params
                best_costs_trace = costs
        
        if best_params is None:
            print("Warning: No valid optimization runs completed.")
            return {
                'bitstring': '',
                'energy': float('inf'),
                'costs': best_costs_trace,
                'repaired_sequence': '',
                'repaired_cost': float('inf')
            }
        
        def _pl_vqe_circuit():
            ansatz(best_params, wires=range(self.n_qubits))
            return
        self.plotter.save_pennylane_circuit(_pl_vqe_circuit, None, 'vqe_circuit_pennylane.png')

        dev_sampling = qml.device('default.qubit', wires=self.n_qubits, shots=self.shots)
        
        @qml.qnode(dev_sampling)
        def get_probabilities(params):
            ansatz(params, wires=range(self.n_qubits))
            return qml.probs(wires=range(self.n_qubits))
        
        probs = get_probabilities(best_params)
        print(f"Number of shots used: {dev_sampling.shots}")
        print("Probabilities:", probs)
        print("Number of probabilities:", len(probs))
        if len(probs) == 0:
            print("Warning: No probabilities computed. Check Hamiltonian or circuit.")
            return {
                'bitstring': '',
                'energy': float('inf'),
                'costs': best_costs_trace,
                'repaired_sequence': '',
                'repaired_cost': float('inf')
            }

        probs_np = np.array(probs)
        probs_np = self._mask_invalid_probabilities(probs_np)
        best_idx = int(np.argmax(probs_np))
        repaired_solution = format(best_idx, f'0{self.n_qubits}b')
        repaired_sequence = self.decode_solution(repaired_solution)
        repaired_energy = self.qaoa_solver.compute_energy_from_bitstring(repaired_solution)
        
        print(f"✅ VQE completed! Final cost: {best_cost:.6f}")
        print(f"Best solution probability: {max(probs):.4f}")
        print(f"➡️ Repaired sequence: {repaired_sequence} | Energy (classical): {repaired_energy:.6f}")
        
        self.plotter.plot_prob_with_sequences(probs, "VQE")

        return {
            'bitstring': repaired_solution,
            'energy': best_cost,
            'costs': best_costs_trace,
            'repaired_sequence': repaired_sequence,
            'repaired_cost': repaired_energy
        }

    def solve_qaoa_qiskit(self, p_layers: int = 8, max_iterations: int = 2000) -> Dict[str, Any]:
        if not QISKIT_AVAILABLE:
            raise ImportError("Qiskit is not installed.")
        # Implementation remains unchanged
        print(f"\n🔥 Solving with Qiskit QAOA (p={p_layers})...")
        print(f"   Using primitives from: {QISKIT_PRIMITIVES_SOURCE}")
        print(f"   Use statevector: {self.use_statevector}")
        print(f"   Shots: {self.shots}")
        
        pauli_list = [(pauli, float(coeff.real)) for coeff, pauli in self.pauli_terms]
        print(f"Sample Pauli terms: {pauli_list[:5]}")
        
        try:
            hamiltonian = SparsePauliOp.from_list(pauli_list)
            print(f"Qiskit Hamiltonian constructed with {len(pauli_list)} terms")
        except Exception as e:
            print(f"❌ Error constructing SparsePauliOp: {e}")
            return self._empty_qaoa_result()
        
        sampler = Sampler() if QISKIT_PRIMITIVES_SOURCE == 'qiskit.primitives' else Sampler(backend_options={'device': 'CPU'})
        optimizer = HybridMultiStartOptimizer(max_iterations=max_iterations, n_starts=20)
        costs = []
        
        def callback(nfev, parameters, energy, *args):
            costs.append(float(energy))
            if nfev % 10 == 0:
                print(f"  Iteration {nfev}: Energy = {energy:.6f}")
        
        qaoa = CustomQAOA(
            sampler=sampler,
            optimizer=optimizer,
            reps=p_layers,
            initial_point=np.random.uniform(0, np.pi, 2 * p_layers),
            callback=callback
        )
        
        try:
            result = qaoa.compute_minimum_eigenvalue(operator=hamiltonian)
            optimal_params = result.optimal_parameters
            optimal_value = float(result.optimal_value)
            print(f"✅ QAOA optimization completed! Best energy: {optimal_value:.6f}")
        except Exception as e:
            print(f"❌ QAOA optimization failed: {e}")
            return self._empty_qaoa_result()
        
        if self.use_statevector:
            print("   🧮 Computing probabilities via Statevector...")
            optimal_circuit = qaoa.construct_circuit(hamiltonian, optimal_params)
            statevector = Statevector(optimal_circuit)
            probs = np.abs(statevector.data) ** 2
        else:
            optimal_circuit = qaoa.construct_circuit(hamiltonian, optimal_params)
            optimal_circuit.measure_all()
            self.plotter.save_qiskit_circuit(optimal_circuit, 'qaoa_circuit_qiskit.png')
            shots = max(self.shots, 5000)
            job = sampler.run([optimal_circuit], shots=shots)
            result_sampler = job.result()
            quasi_dist = result_sampler.quasi_dists[0]
            probs = np.zeros(2**self.n_qubits)
            for key, prob in quasi_dist.items():
                idx = int(key) if isinstance(key, int) else int(key, 2)
                if 0 <= idx < len(probs):
                    probs[idx] = float(prob)
        
        probs = self._mask_invalid_probabilities(probs)
        best_idx = int(np.argmax(probs))
        repaired_solution = format(best_idx, f'0{self.n_qubits}b')
        repaired_sequence = self.decode_solution(repaired_solution)
        repaired_energy = self.compute_energy_from_bitstring(repaired_solution)
        
        print(f"✅ QAOA completed! Repaired sequence: {repaired_sequence} (energy: {repaired_energy:.6f})")
        self.plotter.plot_prob_with_sequences(probs, self.decode_solution, self.n_qubits, "QAOA", top_k=20)
        if costs:
            self.plotter.plot_optimization(costs, solver_name="QAOA")
        
        return {
            'bitstring': repaired_solution,
            'energy': optimal_value,
            'costs': costs,
            'repaired_sequence': repaired_sequence,
            'repaired_cost': repaired_energy,
            'probs': probs
        }

    def compute_classical_minimum(self):
        """Compute exact global minimum by enumerating all states for small systems"""
        print("🧮 Computing exact classical minimum...")
        min_energy = float('inf')
        min_bitstring = ""
        min_sequence = ""
        for idx in range(2**self.n_qubits):
            bitstring = format(idx, f'0{self.n_qubits}b')
            energy = self.compute_energy_from_bitstring(bitstring)
            if energy < min_energy:
                min_energy = energy
                min_bitstring = bitstring
                min_sequence = self.decode_solution(bitstring)
        print(f"✅ Classical minimum: Sequence={min_sequence}, Energy={min_energy:.6f}, Bitstring={min_bitstring}")
        return min_bitstring, min_energy, min_sequence

    def solve_vqe_qiskit(self, layers: int = 8, max_iterations: int = 2500) -> Dict[str, Any]:
        if not QISKIT_AVAILABLE:
            raise ImportError("Qiskit is not installed.")
        # Implementation remains unchanged
        print(f"\n🔥 Solving with Qiskit VQE (layers={layers})...")
        print(f"   Using primitives from: {QISKIT_PRIMITIVES_SOURCE}")
        print(f"   Use statevector: {self.use_statevector}")
        print(f"   Shots: {self.shots}")
        
        pauli_list = [(pauli, float(coeff.real)) for coeff, pauli in self.pauli_terms]
        print(f"Sample Pauli terms: {pauli_list[:5]}")
        
        try:
            hamiltonian = SparsePauliOp.from_list(pauli_list)
            print(f"Qiskit Hamiltonian constructed with {len(pauli_list)} terms")
        except Exception as e:
            print(f"❌ Error constructing SparsePauliOp: {e}")
            return self._empty_vqe_result()
        
        if self.n_qubits <= 2:
            print("Small system detected, computing classical minimum for verification...")
            classical_bitstring, classical_energy, classical_sequence = self.compute_classical_minimum()
        
        from qiskit.circuit.library import TwoLocal
        ansatz = TwoLocal(self.n_qubits, rotation_blocks=['ry'], entanglement_blocks='cz', reps=layers, entanglement='full')
        if self.use_statevector:
            print("Using statevector calculation (no primitives needed)")
            estimator = None
            sampler = None
        else:
            try:
                from qiskit.primitives import Estimator as BaseEstimator, Sampler as BaseSampler
                estimator = BaseEstimator()
                sampler = BaseSampler()
                print("Using qiskit.primitives Estimator/Sampler for VQE")
            except Exception:
                try:
                    estimator = Estimator() if QISKIT_PRIMITIVES_SOURCE == 'qiskit.primitives' else Estimator(backend_options={'device': 'GPU'})
                    sampler = Sampler() if QISKIT_PRIMITIVES_SOURCE == 'qiskit.primitives' else Sampler(backend_options={'device': 'GPU'})
                except Exception:
                    estimator = Estimator() if QISKIT_PRIMITIVES_SOURCE == 'qiskit.primitives' else Estimator(backend_options={'device': 'CPU'})
                    sampler = Sampler() if QISKIT_PRIMITIVES_SOURCE == 'qiskit.primitives' else Sampler(backend_options={'device': 'CPU'})
        optimizer = HybridMultiStartOptimizer(max_iterations=max_iterations, n_starts=20, patience=100, restart_threshold=1e-6)
        costs = []
        
        def callback(nfev, parameters, energy, *args):
            if np.isfinite(energy):
                costs.append(float(energy))
                if nfev % 10 == 0:
                    print(f"  Iteration {nfev}: Energy = {energy:.6f}")
        
        initial_point = np.random.uniform(0, 2 * np.pi, ansatz.num_parameters)
        bounds = [(0, 2 * np.pi) for _ in range(ansatz.num_parameters)]
        vqe = CustomVQE(
            estimator=estimator,
            ansatz=ansatz,
            optimizer=optimizer,
            initial_point=initial_point,
            callback=callback
        )
        
        try:
            vqe.optimizer.external_callback = callback
        except Exception:
            pass
        
        optimal_params = initial_point
        optimal_value = float('inf')
        try:
            result = vqe.compute_minimum_eigenvalue(operator=hamiltonian)
            optimal_params = result.optimal_parameters
            optimal_value = float(result.optimal_value)
            print(f"✅ VQE optimization completed! Best energy: {optimal_value:.6f}")
            if hasattr(result, 'all_energies') and result.all_energies:
                costs = result.all_energies
                print(f"📊 Collected {len(costs)} energy evaluations for convergence plot")
            elif hasattr(result, 'history') and result.history:
                costs = result.history
                print(f"📊 Using optimizer history ({len(costs)} points) for convergence plot")
        except Exception as e:
            print(f"❌ VQE optimization failed: {e}")
            if self.n_qubits <= 2:
                print("Falling back to classical minimum...")
                optimal_value = classical_energy
                probs = np.zeros(2**self.n_qubits)
                probs[int(classical_bitstring, 2)] = 1.0
            else:
                print("⚠️ Using initial parameters for probability computation")
                probs = None
        
        if not costs:
            print("⚠️ No convergence data collected, using fallback data")
            costs = [optimal_value] if np.isfinite(optimal_value) else [1e6]
        
        if self.use_statevector:
            print("   🧮 Computing probabilities via Statevector...")
            try:
                optimal_circuit = ansatz.assign_parameters(optimal_params)
                statevector = Statevector(optimal_circuit)
                probs = np.abs(statevector.data) ** 2
            except Exception as e:
                print(f"⚠️ Statevector computation failed: {e}")
                probs = None
        else:
            optimal_circuit = ansatz.assign_parameters(optimal_params)
            optimal_circuit.measure_all()
            try:
                print("Attempting to save Qiskit circuit: vqe_circuit_qiskit.png")
                self.plotter.save_qiskit_circuit(optimal_circuit, 'vqe_circuit_qiskit.png')
                print(f"Qiskit circuit saved: ../vqe_5/vqe_circuit_qiskit.png")
                shots = max(self.shots, 5000)
                job = sampler.run([optimal_circuit], shots=shots)
                result_sampler = job.result()
                quasi_dist = result_sampler.quasi_dists[0]
                probs = np.zeros(2**self.n_qubits)
                for key, prob in quasi_dist.items():
                    idx = int(key) if isinstance(key, int) else int(key, 2)
                    if 0 <= idx < len(probs):
                        probs[idx] = float(prob)
            except Exception as e:
                print(f"⚠️ Sampler failed: {e}")
                probs = None
        
        if probs is None or np.sum(probs) < 1e-10:
            print("⚠️ No valid probabilities, falling back to uniform distribution or classical minimum.")
            if self.n_qubits <= 2:
                probs = np.zeros(2**self.n_qubits)
                probs[int(classical_bitstring, 2)] = 1.0
            else:
                probs = np.ones(2**self.n_qubits) / (2**self.n_qubits)
        
        probs = self._mask_invalid_probabilities(probs)
        best_idx = int(np.argmax(probs))
        repaired_solution = format(best_idx, f'0{self.n_qubits}b')
        repaired_sequence = self.decode_solution(repaired_solution)
        repaired_energy = self.compute_energy_from_bitstring(repaired_solution)
        
        print(f"✅ VQE completed! Repaired sequence: {repaired_sequence} (energy: {repaired_energy:.6f})")
        if self.n_qubits <= 2:
            print(f"Classical verification: Sequence={classical_sequence}, Energy={classical_energy:.6f}")
            if abs(repaired_energy - classical_energy) > 1e-3:
                print(f"⚠️ VQE energy ({repaired_energy:.6f}) deviates from classical minimum ({classical_energy:.6f})")
        
        self.plotter.plot_prob_with_sequences(probs, self.decode_solution, self.n_qubits, "VQE", top_k=20)
        if costs and len(costs) > 1:
            print("Generating convergence plot...")
            self.plotter.plot_optimization(costs, solver_name="VQE")
        else:
            print("⚠️ No convergence data available for plotting")
        
        return {
            'bitstring': repaired_solution,
            'energy': repaired_energy if optimal_value == float('inf') else optimal_value,
            'costs': costs,
            'repaired_sequence': repaired_sequence,
            'repaired_cost': repaired_energy,
            'probs': probs
        }

    def _empty_qaoa_result(self):
        """Helper method for empty QAOA results"""
        return {
            'bitstring': '',
            'energy': float('inf'),
            'costs': [],
            'repaired_sequence': '',
            'repaired_cost': float('inf')
        }

    def _empty_vqe_result(self):
        """Helper method for empty VQE results"""
        return {
            'bitstring': '',
            'energy': float('inf'),
            'costs': [],
            'repaired_sequence': '',
            'repaired_cost': float('inf')
        }

    def solve_classical_qubo(self) -> Dict[str, Any]:
        result = self.classical_solver.solve()
        return result

    def get_top_sequences_from_probs(self, probs, top_k=1000):
        """Get top sequences from probability distribution"""
        top_k = min(top_k, len(probs))
        
        if top_k < len(probs):
            sorted_indices = np.argpartition(probs, -top_k)[-top_k:]
            sorted_indices = sorted_indices[np.argsort(-probs[sorted_indices])]
        else:
            sorted_indices = np.argsort(-probs)
        
        results = []
        for idx in sorted_indices:
            bitstring = format(idx, f'0{self.n_qubits}b')
            sequence = self.decode_solution(bitstring)
            probability = probs[idx]
            energy = self.compute_energy_from_bitstring(bitstring)
            results.append((bitstring, sequence, probability, energy))
        
        return results

    def get_top_sequences_classical(self, top_k=1000):
        """Get top sequences using classical exhaustive search"""
        import heapq
        
        total_combinations = self.n_aa ** self.L
        top_heap = []
        
        sample_size = min(total_combinations, max(top_k * 100, 10000))
        step = max(1, total_combinations // sample_size)
        
        for idx in range(0, total_combinations, step):
            sequence_codes = []
            temp_idx = idx
            for _ in range(self.L):
                aa_code = temp_idx % self.n_aa
                sequence_codes.append(aa_code)
                temp_idx //= self.n_aa
            sequence_codes.reverse()
            
            bitstring = ''.join(format(c, f'0{self.bits_per_pos}b') for c in sequence_codes)
            energy = self.compute_energy_from_bitstring(bitstring)
            sequence = self.decode_solution(bitstring)
            
            if len(top_heap) < top_k:
                heapq.heappush(top_heap, (-energy, sequence, energy, bitstring))
            elif energy < -top_heap[0][0]:
                heapq.heapreplace(top_heap, (-energy, sequence, energy, bitstring))
        
        results = []
        for _, seq, energy, bitstring in sorted(top_heap, key=lambda x: x[2]):
            results.append((bitstring, seq, 0.0, energy))
        
        return results

    '''def compute_energy_breakdown(self, bitstring):
            """Compute energy breakdown by Hamiltonian term types"""
            z_vals = np.array([1 if b == '0' else -1 for b in bitstring])
            
            breakdown = {
                'Local Terms': 0.0,
                'Pairwise Terms': 0.0,
                'Environment Terms': 0.0,
                'Charge Terms': 0.0,
                'Hydrophobic Moment Terms': 0.0,
                'Other Terms': 0.0
            }
            
            for coeff, pauli_string in self.pauli_terms:
                prod = 1.0
                for i, p in enumerate(pauli_string):
                    if p == 'Z':
                        prod *= z_vals[i]
                
                term_energy = coeff * prod
                z_count = pauli_string.count('Z')
                print('TERMINOSTERMINOSTERMINOSTERMINOSTERMINOSTERMINOSTERMINOSTERMINOSTERMINOSTERMINOSTERMINOS',term_energy,'ZCOUNT',z_count,'terminos',str(coeff).lower())
                if z_count == 1:
                    breakdown['Local Terms'] += term_energy
                elif z_count == 2:
                    breakdown['Pairwise Terms'] += term_energy
                elif 'environment' in str(coeff).lower() or 'hydrophobic' in str(coeff).lower():
                    print('ENVIRONMENTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT',str(coeff).lower(),term_energy)
                    breakdown['Environment Terms'] += term_energy
                elif 'charge' in str(coeff).lower():
                    breakdown['Charge Terms'] += term_energy
                elif 'moment' in str(coeff).lower():
                    breakdown['Hydrophobic Moment Terms'] += term_energy
                else:
                    breakdown['Other Terms'] += term_energy
            
            return breakdown'''


    def compute_energy_breakdown(self, bitstring):
        """Compute energy breakdown by Hamiltonian term types"""
        z_vals = np.array([1 if b == '0' else -1 for b in bitstring])
        
        breakdown = {}
        total_energy = 0.0
        
        # CAMBIO: Acceder a terms_by_type desde hamiltonian_builder
        for category, terms in self.hamiltonian_builder.terms_by_type.items():
            category_energy = 0.0
            for coeff, pauli_string in terms:
                prod = 1.0
                for i, p in enumerate(pauli_string):
                    if p == 'Z':
                        prod *= z_vals[i]
                term_energy = coeff * prod
                category_energy += term_energy
            
            breakdown[category] = category_energy
            total_energy += category_energy
        
        breakdown['Total Energy'] = total_energy
        
        return breakdown

    def analyze_solution(self, result: Dict[str, Any]):
        print("\n🧬 QUANTUM SOLUTION ANALYSIS 🧬")
        bitstring = result['bitstring']
        sequence = result.get('repaired_sequence', result.get('sequence', ''))
        energy = result.get('repaired_cost', result.get('energy', float('inf')))
        
        print(f"Binary solution: {bitstring}")
        print(f"Decoded sequence: {sequence}")
        print(f"Final energy: {energy:.6f}")
        
        violation_count = sequence.count('X')
        print(f"Constraint violations: {violation_count}/{self.L}")
        
        if violation_count == 0:
            hydrophobic_residues = sum(1 for aa in sequence if aa in ['A', 'L', 'I', 'M', 'F', 'W', 'V'])
            charged_residues = sum(1 for aa in sequence if aa in ['E', 'K', 'R', 'D'])
            print(f"Hydrophobic residues: {hydrophobic_residues}/{self.L}")
            print(f"Charged residues: {charged_residues}/{self.L}")
        
        return sequence, violation_count

def save_energy_results(designer, result, solver_name, output_dir, lowest_energy_sequence, lowest_energy, top_k=1000):
    """Save top sequences with energies and Hamiltonian terms to energy.txt"""
    os.makedirs(output_dir, exist_ok=True)
    energy_file = os.path.join(output_dir, "energy.txt")
    
    with open(energy_file, 'w') as f:
        f.write(f"=== ENERGY RESULTS - {solver_name.upper()} SOLVER ===\n")
        f.write(f"Solver: {solver_name}\n")
        f.write(f"Sequence Length: {designer.L}\n")
        f.write(f"Amino Acids: {designer.amino_acids}\n")
        f.write(f"Total Qubits: {designer.n_qubits}\n")
        f.write(f"Total Hamiltonian Terms: {len(designer.pauli_terms)}\n")
        f.write(f"Lowest Energy Sequence: {lowest_energy_sequence} (Energy: {lowest_energy:.6f})\n")
        f.write(f"Alpha Helix Wheel Plot: {os.path.join(output_dir, f'alpha_helix_wheel_{solver_name}.png')}\n")
        f.write("="*60 + "\n\n")
        
        top_sequences = designer.get_top_sequences_from_probs(result['probs'], top_k) if 'probs' in result and len(result['probs']) > 0 and np.sum(result['probs']) > 1e-10 else designer.get_top_sequences_classical(top_k)
        
        f.write(f"TOP {min(top_k, len(top_sequences))} SEQUENCES BY ENERGY:\n")
        f.write("-" * 60 + "\n")
        
        for rank, (bitstring, sequence, probability, energy) in enumerate(top_sequences, 1):
            f.write(f"Rank {rank:2d}: {sequence} | Energy: {energy:.6f}")
            if probability > 0:
                f.write(f" | Probability: {probability:.6f}")
            f.write(f" | Bitstring: {bitstring}\n")
            
            energy_breakdown = designer.compute_energy_breakdown(bitstring)
            f.write(f"         Energy Breakdown:\n")
            for term_name, term_energy in energy_breakdown.items():
                f.write(f"           {term_name}: {term_energy:.6f}\n")
            f.write("\n")
        
        f.write("="*60 + "\n")
        f.write("HAMILTONIAN TERMS SUMMARY:\n")
        f.write("-" * 60 + "\n")
        
        term_groups = {}
        for coeff, pauli_string in designer.pauli_terms:
            z_count = pauli_string.count('Z')
            term_type = "Identity" if z_count == 0 else "Local" if z_count == 1 else "Pairwise" if z_count == 2 else f"Higher-order ({z_count}-body)"
            if term_type not in term_groups:
                term_groups[term_type] = []
            term_groups[term_type].append((coeff, pauli_string))
        
        for term_type, terms in term_groups.items():
            f.write(f"{term_type} Terms: {len(terms)}\n")
            total_coeff = sum(coeff for coeff, _ in terms)
            f.write(f"  Total coefficient: {total_coeff:.6f}\n")
            f.write(f"  Coefficient range: [{min(coeff for coeff, _ in terms):.6f}, {max(coeff for coeff, _ in terms):.6f}]\n")
    
    print(f"✅ Energy results saved to: {energy_file}")

def run_quantum_protein_design(sequence_length, amino_acids, quantum_backend='pennylane', 
                               shots: int = 5000, **kwargs):
    L = sequence_length
    phase = kwargs.get('wheel_phase_deg', 0.0)  # Updated to match command-line argument
    halfwidth = kwargs.get('wheel_halfwidth_deg', 60.0)  # Updated to match command-line argument
    output_dir = kwargs.get('output_dir', '../classical_2_res_logica_rebeca_F')  # Updated to match command-line argument
    
    # Debug environment setup
    print(f"\nDEBUG ENV (phase={phase}°, halfwidth={halfwidth}°):")
    print(f"Output directory: {output_dir}")
    print(f"Absolute output path: {os.path.abspath(output_dir)}")
    
    # Determine membrane and water positions for helical wheel
    mem_pos, water_pos = [], []
    for i in range(L):
        angle = (i * 100.0 + phase) % 360.0
        if angle > 180: angle -= 360
        env = "membrane" if abs(angle) <= halfwidth else "water"
        if env == "membrane":
            mem_pos.append(i)
        else:
            water_pos.append(i)
        print(f"Pos {i}: angle={angle:.1f}° → {env} (raw angle={i * 100.0 + phase:.1f}°)")
    print(f"Membrane positions: {mem_pos}")
    print(f"Water positions: {water_pos}")
    
    # Initialize the QuantumProteinDesign object
    designer = QuantumProteinDesign(
        sequence_length=sequence_length,
        amino_acids=amino_acids,
        quantum_backend=quantum_backend,
        shots=shots,
        **kwargs
    )
    
    # Run the solver
    solver = kwargs.get('solver', 'classical')  # Default to classical to match command-line argument
    if solver == 'classical':
        result = designer.classical_solver.solve()  # Fixed to use ClassicalSolver.solve()
    elif solver == 'vqe':
        result = designer.solve_vqe_pennylane() if quantum_backend == 'pennylane' else designer.solve_vqe_qiskit()
    else:
        result = designer.solve_qaoa_pennylane() if quantum_backend == 'pennylane' else designer.solve_qaoa_qiskit()
    
    # Analyze the solution
    decoded_sequence, violations = designer.analyze_solution(result)
    
    if violations > 0:
        print(f"⚠️ Warning: Solution contains {violations} constraint violations ('X' residues).")
    
    # Plot optimization convergence if available
    if 'costs' in result and result['costs']:
        solver_name = 'VQE' if solver == 'vqe' else ('QAOA' if solver == 'qaoa' else 'CLASSICAL')
        designer.plotter.plot_optimization(result['costs'], solver_name=solver_name)
    
    # Get the top sequences and find the one with the lowest energy
    top_k = 1000
    top_sequences = designer.get_top_sequences_from_probs(result['probs'], top_k) if 'probs' in result and len(result['probs']) > 0 and np.sum(result['probs']) > 1e-10 else designer.get_top_sequences_classical(top_k)
    
    # Find the sequence with the lowest energy
    if top_sequences:
        lowest_energy_sequence = min(top_sequences, key=lambda x: x[3])[1]  # x[3] is energy
        lowest_energy = min(top_sequences, key=lambda x: x[3])[3]
        print(f"Lowest energy sequence: {lowest_energy_sequence} (Energy: {lowest_energy:.6f})")
    else:
        lowest_energy_sequence = result.get('repaired_sequence', result.get('sequence', ''))
        lowest_energy = result.get('repaired_cost', result.get('energy', float('inf')))
        print(f"Falling back to solver result sequence: {lowest_energy_sequence} (Energy: {lowest_energy:.6f})")
    
    # Generate and save the alpha helix wheel plot for the lowest-energy sequence
    if lowest_energy_sequence and lowest_energy_sequence.count('X') < len(lowest_energy_sequence):
        try:
            helix_plot_path = os.path.join(output_dir, f"alpha_helix_wheel_{solver}.png")
            print(f"Generating alpha helix wheel plot for lowest-energy sequence: {lowest_energy_sequence}")
            designer.plotter.plot_alpha_helix_wheel(
                sequence=lowest_energy_sequence,
                membrane_mode=kwargs.get('membrane_mode', 'wheel'),  # Use wheel mode
                wheel_phase_deg=phase,
                wheel_halfwidth_deg=halfwidth
            )
            print(f"✅ Alpha helix wheel plot saved to: {helix_plot_path}")
        except Exception as e:
            print(f"⚠️ Failed to generate alpha helix wheel plot: {e}")
    else:
        print(f"⚠️ Cannot generate alpha helix wheel plot: Invalid sequence {lowest_energy_sequence}")
    
    # Save energy results
    save_energy_results(designer, result, solver, output_dir, lowest_energy_sequence, lowest_energy, top_k=400000)
    
    return designer, result

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Quantum protein sequence design.')
    parser.add_argument('-L', '--length', type=int, default=4, help='Sequence length.')
    parser.add_argument('-R', '--residues', type=str, default="V,Q,L,R", help='Amino acids to use, comma-separated.')
    parser.add_argument('-b', '--backend', type=str, default='pennylane', choices=['pennylane', 'qiskit'], help='Quantum backend to use.')
    parser.add_argument('--solver', type=str, default='qaoa', choices=['qaoa', 'vqe', 'classical'], help='Solver to use.')
    parser.add_argument('--shots', type=int, default=5000, help='Number of shots for quantum simulation.')
    parser.add_argument('--membrane', type=str, help='Membrane span (e.g., 1:4)')
    parser.add_argument('--membrane_positions', type=str, help='Membrane positions (e.g., 0,2,5)')
    parser.add_argument('--membrane_mode', type=str, default='wheel', choices=['span', 'set', 'wheel'], help='Mode for defining membrane positions.')
    parser.add_argument('--wheel_phase_deg', type=float, default=-50.0, help='Phase angle for helical wheel in degrees.')
    parser.add_argument('--wheel_halfwidth_deg', type=float, default=120.0, help='Half-width of the membrane sector in degrees for helical wheel.')
    parser.add_argument('--lambda_env', type=float, default=1., help='Weight of the environment preference term.')
    parser.add_argument('--lambda_charge', type=float, default=1.0, help='Weight of the membrane charge term.')
    parser.add_argument('--lambda_mu', type=float, default=0.5, help='Weight of the hydrophobic moment term.')
    parser.add_argument('--lambda_local', type=float, default=1.0, help='Weight of the local preference terms.')
    parser.add_argument('--lambda_pairwise', type=float, default=0.05, help='Weight of the pairwise interaction term.') #en el q funciona vale 1
    parser.add_argument('--lambda_helix_pairs', type=float, default=0.05, help='Weight of the helix pair propensity term.')
    parser.add_argument('--lambda_segregation', type=float, default=0.5, help='Weight of the amphipathic segregation term.')
    parser.add_argument('--lambda_electrostatic', type=float, default=0.5, help='Weight of the electrostatics term.')
    parser.add_argument('--max_interaction_dist', type=int, default=4, help='Maximum sequence distance for pairwise interactions.')
    parser.add_argument('--membrane_charge', type=str, default='neg', choices=['neu', 'neg', 'pos'], help='Charge of the membrane.')
    parser.add_argument('--output_dir', type=str, default='output', help='Directory to save output files (plots and logs).')
    parser.add_argument('--use_statevector', action='store_true', default=False, help='Use statevector instead of shots for Qiskit (default: use shots)')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    aa_list = ["A", "R", "N", "D", "C", "E", "Q", "G", "H", "I", "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V"]
    if args.residues:
        s = args.residues.upper().strip()
        if len(s) > 1 and ',' in s:
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

    start_time = time.time()
    designer, result = run_quantum_protein_design(
        sequence_length=args.length,
        amino_acids=aa_list,
        quantum_backend=args.backend,
        shots=args.shots,
        membrane_span=mem_span,
        membrane_charge=args.membrane_charge,
        lambda_charge=args.lambda_charge,
        lambda_env=args.lambda_env,
        lambda_mu=args.lambda_mu,
        lambda_local=args.lambda_local,
        lambda_pairwise=args.lambda_pairwise,
        lambda_helix_pairs=args.lambda_helix_pairs,
        lambda_segregation=args.lambda_segregation,
        lambda_electrostatic=args.lambda_electrostatic,
        max_interaction_dist=args.max_interaction_dist,
        membrane_positions=mem_positions,
        membrane_mode=args.membrane_mode,
        wheel_phase_deg=args.wheel_phase_deg,
        wheel_halfwidth_deg=args.wheel_halfwidth_deg,
        solver=args.solver,
        output_dir=args.output_dir,
        use_statevector=args.use_statevector
    )
    end_time = time.time()
    execution_time = end_time - start_time
    
    print("Optimization complete!")
    
    if args.solver == 'classical':
        print("\n🏆 Solución Clásica:")
        print(f"Secuencia: {result['sequence']}")
        print(f"Energía: {result['energy']:.6f}")
    elif args.solver == 'vqe':
        print("\n⚛️ Solución Cuántica (VQE):")
        print(f"Secuencia Reparada: {result['repaired_sequence']}")
        print(f"Energía Final: {result['repaired_cost']:.6f}")
    else:
        print("\n⚛️ Solución Cuántica (QAOA):")
        print(f"Secuencia Reparada: {result['repaired_sequence']}")
        print(f"Energía Final: {result['repaired_cost']:.6f}")

    log_entry = f"Solver: {args.solver} | Execution Time: {execution_time:.4f} seconds | Phase: {args.wheel_phase_deg}° | Halfwidth: {args.wheel_halfwidth_deg}° | Sequence: {result.get('repaired_sequence', result.get('sequence', ''))}\n"
    log_path = os.path.join(args.output_dir, "execution_log.txt")
    with open(log_path, "a") as log_file:
        log_file.write(log_entry)
    print(f"\nExecution time logged to {log_path}")
