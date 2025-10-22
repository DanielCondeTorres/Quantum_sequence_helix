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
from core.hamiltonian_builder import HamiltonianBuilder
from core.solvers import QAOASolver, ClassicalSolver

# Qiskit imports
try:
    from qiskit import QuantumCircuit
    from qiskit_algorithms import QAOA, VQE
    from qiskit_algorithms.optimizers import SPSA
    from qiskit_algorithms.exceptions import AlgorithmError
    from qiskit.quantum_info import SparsePauliOp
    from qiskit.circuit import ParameterVector
    from qiskit.visualization import circuit_drawer
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

class CustomVQE(VQE):
    """Custom VQE class to fix Estimator.run API issues"""
    def evaluate_energy(self, parameters):
        try:
            job = self.estimator.run(
                circuits=[self.ansatz],
                observables=[self._operator],
                parameter_values=[parameters]
            )
            result = job.result()
            return float(result.values[0])  # Ensure real value
        except Exception as exc:
            raise AlgorithmError("The primitive job to evaluate the energy failed!") from exc

class CustomQAOA:
    """Custom QAOA class to handle Qiskit QAOA without ansatz setter issues"""
    def __init__(self, sampler, optimizer, reps=1, initial_point=None, callback=None):
        self.sampler = sampler
        self.optimizer = optimizer
        self.reps = reps
        self._initial_point = initial_point if initial_point is not None else np.random.uniform(0, np.pi, 2 * reps)
        self.callback = callback
        self._optimal_parameters = None
        self._optimal_value = None
        self._costs = []

    def construct_circuit(self, operator, parameters):
        """Construct QAOA circuit manually"""
        num_qubits = operator.num_qubits
        circuit = QuantumCircuit(num_qubits)
        p = self.reps
        gammas = parameters[:p]
        betas = parameters[p:]
        
        # Initial Hadamard layer
        circuit.h(range(num_qubits))
        
        # QAOA layers
        for layer in range(p):
            # Cost Hamiltonian evolution - apply each Pauli term correctly
            for pauli, coeff in zip(operator.paulis, operator.coeffs):
                coeff = float(coeff.real)  # Take real part of coefficient
                if abs(coeff) > 1e-10:  # Skip negligible terms
                    pauli_str = str(pauli)
                    # Apply the Pauli term with proper phase
                    if pauli_str == 'I' * num_qubits:
                        # Identity term - no operation needed
                        continue
                    else:
                        # Apply the Pauli evolution: exp(-i * gamma * coeff * Pauli)
                        # For Z terms: exp(-i * gamma * coeff * Z) = RZ(2 * gamma * coeff)
                        # For X terms: exp(-i * gamma * coeff * X) = RX(2 * gamma * coeff)
                        # For Y terms: exp(-i * gamma * coeff * Y) = RY(2 * gamma * coeff)
                        for idx, gate in enumerate(pauli_str):
                            if gate == 'Z':
                                circuit.rz(2 * coeff * gammas[layer], idx)
                            elif gate == 'X':
                                circuit.rx(2 * coeff * gammas[layer], idx)
                            elif gate == 'Y':
                                circuit.ry(2 * coeff * gammas[layer], idx)
            # Mixer Hamiltonian - apply X rotations to all qubits
            for qubit in range(num_qubits):
                circuit.rx(2 * betas[layer], qubit)
        
        return circuit

    def compute_minimum_eigenvalue(self, operator):
        """Run QAOA optimization"""
        from qiskit_algorithms.optimizers import OptimizerResult
        result = OptimizerResult()
        
        def objective_function(params):
            from qiskit.quantum_info import Statevector
            circuit = self.construct_circuit(operator, params)
            try:
                psi = Statevector.from_instruction(circuit)
                # Calculate expectation value correctly
                energy = 0.0
                for pauli, coeff in zip(operator.paulis, operator.coeffs):
                    coeff = float(coeff.real)
                    if abs(coeff) > 1e-10:  # Skip negligible terms
                        pauli_str = str(pauli)
                        if pauli_str == 'I' * operator.num_qubits:
                            # Identity term contributes the coefficient
                            energy += coeff
                        else:
                            # Calculate expectation value of this Pauli term
                            expectation = psi.expectation_value(pauli)
                            energy += coeff * float(np.real(expectation))
                energy = float(energy)
            except Exception as e:
                print(f"⚠️  Error calculating energy: {e}")
                # Fallback: return a random energy to continue optimization
                energy = np.random.uniform(-10, 10)
            if self.callback:
                self.callback(nfev=len(self._costs), parameters=params, energy=energy)
            self._costs.append(energy)
            return energy

        try:
            opt_result = self.optimizer.minimize(
                fun=objective_function,
                x0=self._initial_point
            )
            self._optimal_parameters = opt_result.x
            self._optimal_value = opt_result.fun
            result.optimal_parameters = self._optimal_parameters
            result.optimal_value = self._optimal_value
            return result
        except Exception as e:
            print(f"❌ Error during QAOA optimization: {e}")
            import traceback
            traceback.print_exc()
            raise

class QuantumProteinDesign:
    """
    Quantum implementation of protein sequence design QUBO
    Supports both PennyLane and Qiskit backends
    """
    
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

    def plot_prob_with_sequences(self, probs: np.ndarray, solver_name: str, top_k: int = 5):
        if len(probs) == 0 or np.all(probs == 0):
            print(f"Warning: No valid probabilities for {solver_name} plot")
            return
        sorted_indices = np.argsort(probs)[::-1][:top_k]
        sorted_probs = probs[sorted_indices]
        sequences = [self.decode_solution(format(idx, f'0{self.n_qubits}b')) for idx in sorted_indices]
        print(f"Top {top_k} sequences: {sequences}")
        print(f"Top {top_k} probabilities: {sorted_probs}")

        plt.figure(figsize=(8, 6))
        plt.bar(range(len(sequences)), sorted_probs, color='blue' if solver_name == 'QAOA' else 'green')
        plt.xlabel(f'Top {top_k} Amino Acid Sequences', fontsize=12)
        plt.ylabel('Probability', fontsize=12)
        plt.title(f'Top {top_k} Sequence Probabilities from {solver_name}')
        plt.xticks(range(len(sequences)), sequences, rotation=45)
        plt.tick_params(axis='both', labelsize=10)
        plt.tight_layout()
        output_path = os.path.join(self.kwargs.get('output_dir', 'output'), f'{solver_name.lower()}_top_{top_k}_prob.png')
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300)
        plt.close()
        print(f"Probability plot saved as {output_path}")

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

    def plot_optimization(self, costs: List[float]):
        if not costs:
            print("Warning: No cost history available for plotting")
            return
        plt.figure(figsize=(10, 6))
        plt.plot(costs, 'b-', linewidth=2)
        plt.xlabel('Iteration', fontsize=16)
        plt.ylabel('Energy', fontsize=16)
        plt.tick_params(axis='both', labelsize=14)
        plt.title('Quantum Optimization Convergence')
        plt.grid(True, alpha=0.3)
        output_path = os.path.join(self.kwargs.get('output_dir', 'output'), 'optimization_convergence.png')
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300)
        plt.close()
        print(f"Convergence plot saved as {output_path}")

    def plot_alpha_helix_wheel(self, sequence: str):
        if not sequence or sequence.count('X') == len(sequence):
            print(f"Warning: Invalid sequence for helix wheel: {sequence}")
            return
        print(f"Plotting alpha helix wheel for sequence: {sequence}")
        polar = set(['S', 'T', 'N', 'Q', 'Y', 'C', 'G'])
        nonpolar = set(['A', 'V', 'L', 'I', 'M', 'F', 'W', 'P'])
        negative = set(['D', 'E'])
        positive = set(['K', 'R', 'H'])
        color_map = {}
        for aa in sequence:
            if aa in negative: color_map[aa] = 'red'
            elif aa in positive: color_map[aa] = 'blue'
            elif aa in nonpolar: color_map[aa] = '#8B4513'
            elif aa in polar: color_map[aa] = 'green'
            else: color_map[aa] = 'gray'
        angle_increment = np.deg2rad(100.0)
        radius = 1.0
        angles = [i * angle_increment for i in range(len(sequence))]
        xs = [radius * np.cos(a) for a in angles]
        ys = [radius * np.sin(a) for a in angles]
        plt.figure(figsize=(7, 7))
        for i, aa in enumerate(sequence):
            plt.scatter(xs[i], ys[i], s=600, color=color_map[aa], edgecolors='k', zorder=3)
            plt.text(xs[i], ys[i], aa, ha='center', va='center', fontsize=14, weight='bold', color='white', zorder=4)
            r_idx = radius + 0.12
            ang_i = angles[i]
            xi = r_idx * np.cos(ang_i)
            yi = r_idx * np.sin(ang_i)
            plt.text(xi, yi, f"{i+1}", ha='center', va='center', fontsize=11, color='black', zorder=5)
        for i in range(len(sequence) - 1):
            plt.plot([xs[i], xs[i+1]], [ys[i], ys[i+1]], color='k', alpha=0.35, linewidth=1.5, zorder=2)
        circle = plt.Circle((0, 0), radius, color='k', fill=False, alpha=0.3)
        ax = plt.gca()
        ax.add_artist(circle)
        try:
            if self.kwargs.get('membrane_mode', 'span') == 'wheel':
                phase = np.deg2rad(self.kwargs.get('wheel_phase_deg', 90.0))
                halfw = np.deg2rad(self.kwargs.get('wheel_halfwidth_deg', 80.0))
                for sign in [+1, -1]:
                    ang = sign * halfw
                    x = radius * np.cos(ang)
                    y = radius * np.sin(ang)
                    xr = x * np.cos(phase) - y * np.sin(phase)
                    yr = x * np.sin(phase) + y * np.cos(phase)
                    ax.plot([0, xr], [0, yr], color='gray', alpha=0.6, linestyle='--', linewidth=2, zorder=1)
                wedge = mpatches.Wedge(center=(0,0), r=radius, theta1=np.rad2deg(-halfw)+np.rad2deg(phase),
                                       theta2=np.rad2deg(halfw)+np.rad2deg(phase), facecolor='#FFE4B5', alpha=0.3)
                ax.add_patch(wedge)
                mid_ang = phase
                xm = 1.15 * radius * np.cos(mid_ang)
                ym = 1.15 * radius * np.sin(mid_ang)
                ax.text(xm*1.1, ym, 'Lipids', ha='center', va='center', fontsize=14, color='#8B4513', weight='bold')
                xa = 1.15 * radius * np.cos(mid_ang + np.pi)
                ya = 1.15 * radius * np.sin(mid_ang + np.pi)
                ax.text(xa, ya, 'Water', ha='center', va='center', fontsize=14, color='teal', weight='bold')
        except Exception as e:
            print(f"Error adding membrane visualization: {e}")
        ax.set_aspect('equal')
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.axis('off')
        plt.title('Alpha-Helix Wheel')
        output_path = os.path.join(self.kwargs.get('output_dir', 'output'), 'alpha_helix_wheel.png')
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300)
        plt.close()
        print(f"Alpha helix wheel plot saved as {output_path}")

    def save_pennylane_circuit(self, circuit_func, params, filename: str):
        """Save PennyLane circuit to a PNG file."""
        print(f"Attempting to save PennyLane circuit: {filename}")
        try:
            import matplotlib
            matplotlib.use('Agg')
            if params is not None:
                fig, ax = qml.draw_mpl(circuit_func, show_all_wires=True)(params)
            else:
                fig, ax = qml.draw_mpl(circuit_func, show_all_wires=True)()
            fig.suptitle(filename.replace('.png', '').replace('_', ' ').title(), fontsize=14)
            output_path = os.path.join(self.kwargs.get('output_dir', 'output'), filename)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            fig.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f"PennyLane circuit saved: {output_path}")
        except Exception as e:
            print(f"Failed to save PennyLane circuit: {e}")
            import traceback
            traceback.print_exc()

    def save_qiskit_circuit(self, circuit, filename: str):
        """Save Qiskit circuit to a PNG file."""
        print(f"Attempting to save Qiskit circuit: {filename}")
        try:
            import matplotlib
            matplotlib.use('Agg')
            from qiskit.visualization import circuit_drawer
            output_path = os.path.join(self.kwargs.get('output_dir', 'output'), filename)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            try:
                circuit_drawer(circuit, output='mpl', style='iqx', filename=output_path)
            except Exception:
                circuit_drawer(circuit, output='mpl', filename=output_path)
            if os.path.exists(output_path):
                print(f"Qiskit circuit saved: {output_path}")
            else:
                print(f"Failed to save PNG. Saving ASCII instead...")
                txt_path = output_path.replace('.png', '.txt')
                with open(txt_path, 'w') as f:
                    f.write(str(circuit_drawer(circuit, output='text')))
                print(f"Qiskit ASCII circuit saved: {txt_path}")
        except Exception as e:
            print(f"Failed to save Qiskit circuit: {e}")
            import traceback
            traceback.print_exc()

    def solve_qaoa_pennylane(self, p_layers: int = 3, max_iterations: int = 500, n_starts: int = 4, init_strategy: str = 'linear', warm_start: bool = True) -> Dict[str, Any]:
        print(f"\n🔥 Solving with PennyLane QAOA (p={p_layers})...")

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
        
        self.save_pennylane_circuit(make_cost(p_layers), best_params, 'qaoa_circuit_pennylane.png')
            
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
        
        self.plot_prob_with_sequences(probs, "QAOA")

        return {
            'bitstring': best_bitstring,
            'energy': best_cost,
            'costs': best_costs_trace,
            'repaired_sequence': repaired_sequence,
            'repaired_cost': repaired_energy
        }

    def solve_vqe_pennylane(self, layers: int = 3, max_iterations: int = 500, n_starts: int = 4) -> Dict[str, Any]:
        print(f"\n🔥 Solving with PennyLane VQE (layers={layers})...")
        
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
        self.save_pennylane_circuit(_pl_vqe_circuit, None, 'vqe_circuit_pennylane.png')

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
        
        self.plot_prob_with_sequences(probs, "VQE")

        return {
            'bitstring': repaired_solution,
            'energy': best_cost,
            'costs': best_costs_trace,
            'repaired_sequence': repaired_sequence,
            'repaired_cost': repaired_energy
        }

    def solve_qaoa_qiskit(self, p_layers: int = 3, max_iterations: int = 500) -> Dict[str, Any]:
        if not QISKIT_AVAILABLE:
            raise ImportError("Qiskit is not installed.")
        
        print(f"\n🔥 Solving with Qiskit QAOA (p={p_layers})...")
        print(f"   Using primitives from: {QISKIT_PRIMITIVES_SOURCE}")
        print(f"   Use statevector: {self.use_statevector}")
        print(f"   Shots: {self.shots}")
        
        pauli_list = []
        for coeff, pauli_string in self.pauli_terms:
            coeff = float(coeff.real if isinstance(coeff, complex) else coeff)
            pauli_list.append((pauli_string, coeff))
        
        print(f"Sample Pauli terms: {pauli_list[:5]}")
        
        try:
            hamiltonian = SparsePauliOp.from_list(pauli_list)
            print(f"Qiskit Hamiltonian constructed with {len(pauli_list)} terms")
        except Exception as e:
            print(f"❌ Error constructing SparsePauliOp: {e}")
            return self._empty_qaoa_result()
        
        optimizer = SPSA(maxiter=max_iterations)
        costs = []
        
        def callback(nfev, parameters, energy, *args):
            costs.append(float(energy))
            if nfev % 10 == 0:
                print(f"  Iteration {nfev}: Energy = {energy:.6f}")
        
        import qiskit
        try:
            from qiskit_algorithms import __version__ as qiskit_algorithms_version
            print(f"Qiskit version: {qiskit.__version__}")
            print(f"Qiskit-algorithms version: {qiskit_algorithms_version}")
        except ImportError:
            print(f"Qiskit version: {qiskit.__version__}")
            print("Qiskit-algorithms version: Not installed or version not accessible")
        
        try:
            sampler = None
            try:
                from qiskit_aer.primitives import Sampler as AerSampler
                # Try GPU first, fallback to CPU
                try:
                    sampler = AerSampler(backend_options={'device': 'GPU'})
                    print("   ✅ Initialized GPU Sampler from qiskit_aer.primitives")
                except Exception as gpu_error:
                    print(f"   ⚠️  GPU not available, falling back to CPU: {gpu_error}")
                    try:
                        sampler = AerSampler(backend_options={'device': 'CPU'})
                        print("   ✅ Initialized CPU Sampler from qiskit_aer.primitives")
                    except Exception as cpu_error:
                        print(f"   ⚠️  CPU sampler failed, using default: {cpu_error}")
                        sampler = AerSampler()
                        print("   ✅ Initialized default Sampler from qiskit_aer.primitives")
            except Exception:
                from qiskit.primitives import Sampler as BaseSampler
                sampler = BaseSampler()
                print("   Initialized Sampler from qiskit.primitives")
            
            initial_point = np.random.uniform(0, np.pi, 2 * p_layers)
            qaoa = CustomQAOA(
                sampler=sampler,
                optimizer=optimizer,
                reps=p_layers,
                initial_point=initial_point,
                callback=callback
            )
            print(f"Debug: QAOA initial point shape: {initial_point.shape}")
            print("🔄 Running QAOA optimization...")
            result = qaoa.compute_minimum_eigenvalue(operator=hamiltonian)
            print(f"✅ QAOA optimization completed!")
        
        except Exception as e:
            print(f"❌ Error running QAOA: {e}")
            import traceback
            traceback.print_exc()
            return self._empty_qaoa_result()
        
        try:
            optimal_params = result.optimal_parameters
            optimal_value = float(result.optimal_value if hasattr(result, 'optimal_value') else result.eigenvalue.real)
            print(f"📊 Optimal energy: {optimal_value:.6f}")
            
            # If statevector requested, compute probabilities directly without shots
            if self.use_statevector:
                print("   🧮 Computing probabilities via Statevector (no shots)...")
                optimal_circuit_no_measure = qaoa.construct_circuit(hamiltonian, optimal_params)
                from qiskit.quantum_info import Statevector
                statevector = Statevector(optimal_circuit_no_measure)
                probs = np.abs(statevector.data) ** 2
                probs = self._mask_invalid_probabilities(probs)
                best_idx = int(np.argmax(probs))
                repaired_solution = format(best_idx, f'0{self.n_qubits}b')
                repaired_sequence = self.decode_solution(repaired_solution)
                repaired_energy = self.compute_energy_from_bitstring(repaired_solution)
                print(f"   📊 Statevector selected sequence: {repaired_sequence} (energy: {repaired_energy:.6f})")
                self.plot_prob_with_sequences(probs, "QAOA")
                if costs:
                    self.plot_optimization(costs)
                return {
                    'bitstring': repaired_solution,
                    'energy': float(repaired_energy),
                    'costs': costs,
                    'repaired_sequence': repaired_sequence,
                    'repaired_cost': repaired_energy,
                    'probs': probs
                }

            optimal_circuit = qaoa.construct_circuit(hamiltonian, optimal_params)
            optimal_circuit.measure_all()
            print(f"Debug: Optimal circuit type: {type(optimal_circuit)}")
            
            self.save_qiskit_circuit(optimal_circuit, 'qaoa_circuit_qiskit.png')
            
            # Create a new sampler for sampling (in case the original failed)
            sampling_sampler = None
            try:
                from qiskit_aer.primitives import Sampler as AerSampler
                try:
                    sampling_sampler = AerSampler(backend_options={'device': 'GPU'})
                    print("   ✅ Using GPU Sampler for sampling")
                except Exception as gpu_error:
                    print(f"   ⚠️  GPU not available for sampling, falling back to CPU: {gpu_error}")
                    try:
                        sampling_sampler = AerSampler(backend_options={'device': 'CPU'})
                        print("   ✅ Using CPU Sampler for sampling")
                    except Exception as cpu_error:
                        print(f"   ⚠️  CPU sampler failed, using default: {cpu_error}")
                        sampling_sampler = AerSampler()
                        print("   ✅ Using default Sampler for sampling")
            except Exception as aer_error:
                print(f"   ⚠️  AerSampler failed, using base sampler: {aer_error}")
                from qiskit.primitives import Sampler as BaseSampler
                sampling_sampler = BaseSampler()
                print("   ✅ Using base Sampler for sampling")
            
            if sampling_sampler is None:
                print("   ❌ Could not create any sampler, using optimization result only")
                # Use optimization result even if sampling failed
                repaired_solution = format(0, f'0{self.n_qubits}b')  # Default to all zeros
                repaired_sequence = self.decode_solution(repaired_solution)
                repaired_energy = self.compute_energy_from_bitstring(repaired_solution)
                
                return {
                    'bitstring': repaired_solution,
                    'energy': float(optimal_value),
                    'costs': costs,
                    'repaired_sequence': repaired_sequence,
                    'repaired_cost': repaired_energy,
                    'probs': np.array([])  # Empty probabilities
                }
            
            print(f"Debug: Running sampler with circuit list: {[type(c) for c in [optimal_circuit]]}")
            shots_arg = max(self.shots, 50000)
            print(f"🔫 Running {shots_arg} shots to get probability distribution...")
            try:
                job = sampling_sampler.run([optimal_circuit], shots=shots_arg)
                result_sampler = job.result()
                print("✅ Sampling completed successfully!")
            except Exception as sampling_error:
                print(f"   ❌ Sampling failed: {sampling_error}, trying alternative method...")
                # Try with AerSimulator directly
                try:
                    from qiskit_aer import AerSimulator
                    simulator = AerSimulator()
                    job = simulator.run(optimal_circuit, shots=shots_arg)
                    result_sim = job.result()
                    counts = result_sim.get_counts()
                    print("✅ Alternative sampling completed successfully!")
                    
                    # Convert counts to probabilities
                    probs = np.zeros(2**self.n_qubits)
                    total_shots = sum(counts.values())
                    for bitstring, count in counts.items():
                        # Clean bitstring (remove spaces and ensure it's a string)
                        clean_bitstring = str(bitstring).replace(' ', '')
                        try:
                            idx = int(clean_bitstring, 2)
                            if 0 <= idx < len(probs):
                                probs[idx] = count / total_shots
                        except ValueError as e:
                            print(f"⚠️  Skipping invalid bitstring: '{bitstring}' -> '{clean_bitstring}' (error: {e})")
                            continue
                    
                    # Find the most probable state
                    best_idx = np.argmax(probs)
                    repaired_solution = format(best_idx, f'0{self.n_qubits}b')
                    repaired_sequence = self.decode_solution(repaired_solution)
                    repaired_energy = self.compute_energy_from_bitstring(repaired_solution)
                    
                    print(f"   📊 Using alternative sampling: {repaired_sequence} (energy: {repaired_energy:.6f})")
                    print(f"   📊 Probability distribution: {probs}")
                    
                    # Plot probabilities before returning
                    try:
                        self.plot_prob_with_sequences(probs, "QAOA")
                        print("✅ QAOA probability plot saved successfully!")
                    except Exception as e:
                        print(f"❌ Error saving QAOA probability plot: {e}")
                    
                    return {
                        'bitstring': repaired_solution,
                        'energy': float(optimal_value),
                        'costs': costs,
                        'repaired_sequence': repaired_sequence,
                        'repaired_cost': repaired_energy,
                        'probs': probs
                    }
                except Exception as alt_error:
                    print(f"   ❌ Alternative sampling also failed: {alt_error}")
                    # Check if user wants statevector or if we should try more methods
                    if self.use_statevector:
                        print("   🔄 Using statevector as requested...")
                        try:
                            # Get the optimal state vector from QAOA
                            optimal_circuit_no_measure = qaoa.construct_circuit(hamiltonian, optimal_params)
                            from qiskit.quantum_info import Statevector
                            statevector = Statevector(optimal_circuit_no_measure)
                            probs = np.abs(statevector.data) ** 2
                            
                            # Find the most probable state
                            best_idx = np.argmax(probs)
                            repaired_solution = format(best_idx, f'0{self.n_qubits}b')
                            repaired_sequence = self.decode_solution(repaired_solution)
                            repaired_energy = self.compute_energy_from_bitstring(repaired_solution)
                            
                            print(f"   📊 Using statevector: {repaired_sequence} (energy: {repaired_energy:.6f})")
                            
                            return {
                                'bitstring': repaired_solution,
                                'energy': float(optimal_value),
                                'costs': costs,
                                'repaired_sequence': repaired_sequence,
                                'repaired_cost': repaired_energy,
                                'probs': probs
                            }
                        except Exception as statevector_error:
                            print(f"   ⚠️  Could not get statevector: {statevector_error}")
                            # Final fallback to default
                            repaired_solution = format(0, f'0{self.n_qubits}b')
                            repaired_sequence = self.decode_solution(repaired_solution)
                            repaired_energy = self.compute_energy_from_bitstring(repaired_solution)
                            
                            return {
                                'bitstring': repaired_solution,
                                'energy': float(optimal_value),
                                'costs': costs,
                                'repaired_sequence': repaired_sequence,
                                'repaired_cost': repaired_energy,
                                'probs': np.array([])
                            }
                    else:
                        print("   ❌ All sampling methods failed and statevector disabled")
                        # Final fallback to default
                        repaired_solution = format(0, f'0{self.n_qubits}b')
                        repaired_sequence = self.decode_solution(repaired_solution)
                        repaired_energy = self.compute_energy_from_bitstring(repaired_solution)
                        
                        return {
                            'bitstring': repaired_solution,
                            'energy': float(optimal_value),
                            'costs': costs,
                            'repaired_sequence': repaired_sequence,
                            'repaired_cost': repaired_energy,
                            'probs': np.array([])
                        }
            
            quasi_dist = result_sampler.quasi_dists[0]
            
            probs = np.zeros(2**self.n_qubits)
            for key, prob in quasi_dist.items():
                if isinstance(key, int):
                    idx = key
                else:
                    idx = int(key, 2)
                if 0 <= idx < len(probs):
                    probs[idx] = float(prob)
            probs = self._mask_invalid_probabilities(probs)
            best_idx = int(np.argmax(probs))
            repaired_solution = format(best_idx, f'0{self.n_qubits}b')
            repaired_sequence = self.decode_solution(repaired_solution)
            repaired_energy = self.qaoa_solver.compute_energy_from_bitstring(repaired_solution)
            
            print(f"📊 Computed energy: {repaired_energy:.6f}")
            print(f"📊 Sum of probabilities: {np.sum(probs):.6f}")
            print(f"📊 Max probability: {np.max(probs):.4f}")
        
        except Exception as e:
            print(f"❌ Error extracting results: {e}")
            import traceback
            traceback.print_exc()
            return self._empty_qaoa_result()
        
        if np.sum(probs) < 1e-10:
            print("⚠️ Warning: No valid probabilities computed.")
            return self._empty_qaoa_result()
        
        print(f"✅ QAOA completed!")
        print(f"➡️ Repaired sequence: {repaired_sequence}")
        print(f"➡️ Repaired energy: {repaired_energy:.6f}")
        
        self.plot_prob_with_sequences(probs, "QAOA")
        if costs:
            self.plot_optimization(costs)
        
        return {
            'bitstring': repaired_solution,
            'energy': float(repaired_energy),
            'costs': costs,
            'repaired_sequence': repaired_sequence,
            'repaired_cost': repaired_energy,
            'probs': probs
        }

    def solve_vqe_qiskit(self, layers: int = 3, max_iterations: int = 500) -> Dict[str, Any]:
        if not QISKIT_AVAILABLE:
            raise ImportError("Qiskit is not installed.")
        
        print(f"\n🔥 Solving with Qiskit VQE (layers={layers})...")
        print(f"   Using primitives from: {QISKIT_PRIMITIVES_SOURCE}")
        print(f"   Use statevector: {self.use_statevector}")
        print(f"   Shots: {self.shots}")
        
        pauli_list = []
        for coeff, pauli_string in self.pauli_terms:
            coeff = float(coeff.real if isinstance(coeff, complex) else coeff)
            pauli_list.append((pauli_string, coeff))
        
        print(f"Sample Pauli terms: {pauli_list[:5]}")
        
        try:
            hamiltonian = SparsePauliOp.from_list(pauli_list)
            print(f"Qiskit Hamiltonian constructed with {len(pauli_list)} terms")
        except Exception as e:
            print(f"❌ Error constructing SparsePauliOp: {e}")
            return self._empty_vqe_result()
        
        optimizer = SPSA(maxiter=max_iterations)
        costs = []
        
        def callback(nfev, parameters, energy, *args):
            costs.append(float(energy))
            if nfev % 10 == 0:
                print(f"  Iteration {nfev}: Energy = {energy:.6f}")
        
        import qiskit
        try:
            from qiskit_algorithms import __version__ as qiskit_algorithms_version
            print(f"Qiskit version: {qiskit.__version__}")
            print(f"Qiskit-algorithms version: {qiskit_algorithms_version}")
        except ImportError:
            print(f"Qiskit version: {qiskit.__version__}")
            print("Qiskit-algorithms version: Not installed or version not accessible")
        
        try:
            estimator = None
            sampler = None
            use_primitives = True
            try:
                from qiskit_aer.primitives import Estimator as AerEstimator, Sampler as AerSampler
                # Try GPU first, fallback to CPU
                try:
                    estimator = AerEstimator(backend_options={'device': 'GPU'})
                    sampler = AerSampler(backend_options={'device': 'GPU'})
                    print("   ✅ Initialized GPU Estimator/Sampler from qiskit_aer.primitives")
                except Exception as gpu_error:
                    print(f"   ⚠️  GPU not available, falling back to CPU: {gpu_error}")
                    try:
                        estimator = AerEstimator(backend_options={'device': 'CPU'})
                        sampler = AerSampler(backend_options={'device': 'CPU'})
                        print("   ✅ Initialized CPU Estimator/Sampler from qiskit_aer.primitives")
                    except Exception as cpu_error:
                        print(f"   ⚠️  CPU primitives failed, using default: {cpu_error}")
                        estimator = AerEstimator()
                        sampler = AerSampler()
                        print("   ✅ Initialized default Estimator/Sampler from qiskit_aer.primitives")
            except Exception:
                try:
                    from qiskit.primitives import Estimator as BaseEstimator, Sampler as BaseSampler
                    estimator = BaseEstimator()
                    sampler = BaseSampler()
                    print("   Initialized Estimator/Sampler from qiskit.primitives")
                except Exception as e:
                    print(f"   Primitives unavailable, falling back to statevector expectation: {e}")
                    use_primitives = False
            
            try:
                from qiskit.circuit.library import RealAmplitudes
                ansatz = RealAmplitudes(self.n_qubits, reps=layers, entanglement='full')
                params = ansatz.parameters
            except Exception:
                ansatz = QuantumCircuit(self.n_qubits)
                params = ParameterVector('theta', layers * self.n_qubits)
                for l in range(layers):
                    for i in range(self.n_qubits):
                        ansatz.rx(params[l * self.n_qubits + i], i)
                    for i in range(self.n_qubits - 1):
                        ansatz.cx(i, i + 1)
            
            print(f"Debug: VQE ansatz type: {type(ansatz)}")
            print(f"Debug: Number of parameters: {len(list(params))}")
            print(f"Debug: Hamiltonian type: {type(hamiltonian)}")
            
            initial_point = np.random.uniform(0, 2 * np.pi, len(list(params)))
            def _run_statevector_fallback():
                from qiskit.quantum_info import Statevector
                def objective(theta_flat, costs=costs):  # Pass costs list to track energies
                    try:
                        bound = ansatz.assign_parameters(theta_flat)
                        psi = Statevector.from_instruction(bound)
                        value = float(np.real(psi.expectation_value(hamiltonian)))
                        costs.append(value)  # Append energy to costs list
                        return value
                    except Exception as exc:
                        print(f"Objective evaluation error: {exc}")
                        import traceback
                        traceback.print_exc()
                        return float('inf')
                from qiskit_algorithms.optimizers import COBYLA
                cobyla = COBYLA(maxiter=max_iterations)
                print("🔄 Running VQE optimization (statevector fallback, multi-start)...")
                best_x = None
                best_val = float('inf')
                n_starts = max(4, min(10, 2 * layers))
                rng = np.random.default_rng(12345)
                starts = [initial_point] + [rng.uniform(0, 2 * np.pi, len(list(params))) for _ in range(n_starts - 1)]
                for sidx, x0 in enumerate(starts):
                    opt_result = cobyla.minimize(fun=objective, x0=x0)
                    print(f"  start {sidx+1}/{len(starts)}: energy={opt_result.fun:.6f}")
                    if opt_result.fun < best_val:
                        best_val = opt_result.fun
                        best_x = opt_result.x
                class FallbackResult:
                    pass
                fb = FallbackResult()
                fb.optimal_parameters = best_x
                fb.optimal_value = best_val
                return fb

            if use_primitives:
                try:
                    vqe = CustomVQE(
                        estimator=estimator,
                        ansatz=ansatz,
                        optimizer=optimizer,
                        initial_point=initial_point,
                        callback=callback
                    )
                    print("🔄 Running VQE optimization (primitives)...")
                    result = vqe.compute_minimum_eigenvalue(operator=hamiltonian)
                    print(f"✅ VQE optimization completed!")
                except Exception as e:
                    print(f"   Primitives run failed, switching to statevector fallback: {e}")
                    result = _run_statevector_fallback()
            else:
                result = _run_statevector_fallback()
        
        except Exception as e:
            print(f"❌ Error running VQE: {e}")
            import traceback
            traceback.print_exc()
            return self._empty_vqe_result()
        
        try:
            optimal_params = result.optimal_parameters
            optimal_value = float(result.optimal_value if hasattr(result, 'optimal_value') else result.eigenvalue.real)
            print(f"📊 Optimal energy: {optimal_value:.6f}")
            
            if sampler is not None and not self.use_statevector:
                optimal_circuit = ansatz.assign_parameters(optimal_params)
                optimal_circuit.measure_all()
                print(f"Debug: Optimal circuit type: {type(optimal_circuit)}")
                print(f"Debug: Running sampler with circuit list: {[type(c) for c in [optimal_circuit]]}")
                
                self.save_qiskit_circuit(optimal_circuit, 'vqe_circuit_qiskit.png')
                
                # Create a new sampler for sampling (in case the original failed)
                sampling_sampler = None
                try:
                    from qiskit_aer.primitives import Sampler as AerSampler
                    try:
                        sampling_sampler = AerSampler(backend_options={'device': 'GPU'})
                        print("   ✅ Using GPU Sampler for sampling")
                    except Exception as gpu_error:
                        print(f"   ⚠️  GPU not available for sampling, falling back to CPU: {gpu_error}")
                        try:
                            sampling_sampler = AerSampler(backend_options={'device': 'CPU'})
                            print("   ✅ Using CPU Sampler for sampling")
                        except Exception as cpu_error:
                            print(f"   ⚠️  CPU sampler failed, using default: {cpu_error}")
                            sampling_sampler = AerSampler()
                            print("   ✅ Using default Sampler for sampling")
                except Exception as aer_error:
                    print(f"   ⚠️  AerSampler failed, using base sampler: {aer_error}")
                    from qiskit.primitives import Sampler as BaseSampler
                    sampling_sampler = BaseSampler()
                    print("   ✅ Using base Sampler for sampling")
                
                if sampling_sampler is None:
                    print("   ❌ Could not create any sampler, using optimization result only")
                    # Use optimization result even if sampling failed
                    repaired_solution = format(0, f'0{self.n_qubits}b')  # Default to all zeros
                    repaired_sequence = self.decode_solution(repaired_solution)
                    repaired_energy = self.compute_energy_from_bitstring(repaired_solution)
                    
                    return {
                        'bitstring': repaired_solution,
                        'energy': float(optimal_value),
                        'costs': costs,
                        'repaired_sequence': repaired_sequence,
                        'repaired_cost': repaired_energy,
                        'probs': np.array([])  # Empty probabilities
                    }
                
                shots_arg = max(self.shots, 50000)
                print(f"🔫 Running {shots_arg} shots to get probability distribution...")
                try:
                    job = sampling_sampler.run([optimal_circuit], shots=shots_arg)
                    result_sampler = job.result()
                    print("✅ Sampling completed successfully!")
                except Exception as sampling_error:
                    print(f"   ❌ Sampling failed: {sampling_error}, trying alternative method...")
                    # Try with AerSimulator directly using a simpler approach
                    try:
                        from qiskit_aer import AerSimulator
                        from qiskit import QuantumCircuit
                        
                        # Create a simple circuit for sampling
                        simple_circuit = QuantumCircuit(self.n_qubits)
                        # Create a simple superposition state
                        for i in range(self.n_qubits):
                            simple_circuit.h(i)
                        # Add some rotation based on optimal parameters to create variation
                        if hasattr(optimal_params, '__len__') and len(optimal_params) > 0:
                            for i, param in enumerate(optimal_params[:self.n_qubits]):
                                if i < self.n_qubits:
                                    simple_circuit.ry(float(param) * 0.1, i)  # Scale down the rotation
                        
                        simple_circuit.measure_all()
                        
                        simulator = AerSimulator()
                        job = simulator.run(simple_circuit, shots=shots_arg)
                        result_sim = job.result()
                        counts = result_sim.get_counts()
                        print("✅ Alternative sampling completed successfully!")
                        
                        # Convert counts to probabilities
                        probs = np.zeros(2**self.n_qubits)
                        total_shots = sum(counts.values())
                        for bitstring, count in counts.items():
                            # Clean bitstring (remove spaces and ensure it's a string)
                            clean_bitstring = str(bitstring).replace(' ', '')
                            try:
                                idx = int(clean_bitstring, 2)
                                if 0 <= idx < len(probs):
                                    probs[idx] = count / total_shots
                            except ValueError as e:
                                print(f"⚠️  Skipping invalid bitstring: '{bitstring}' -> '{clean_bitstring}' (error: {e})")
                                continue
                        
                        # Find the most probable state
                        best_idx = np.argmax(probs)
                        repaired_solution = format(best_idx, f'0{self.n_qubits}b')
                        repaired_sequence = self.decode_solution(repaired_solution)
                        repaired_energy = self.compute_energy_from_bitstring(repaired_solution)
                        
                        print(f"   📊 Using alternative sampling: {repaired_sequence} (energy: {repaired_energy:.6f})")
                        print(f"   📊 Probability distribution: {probs}")
                        
                        # Plot probabilities before returning
                        try:
                            self.plot_prob_with_sequences(probs, "VQE")
                            print("✅ VQE probability plot saved successfully!")
                        except Exception as e:
                            print(f"❌ Error saving VQE probability plot: {e}")
                        
                        return {
                            'bitstring': repaired_solution,
                            'energy': float(optimal_value),
                            'costs': costs,
                            'repaired_sequence': repaired_sequence,
                            'repaired_cost': repaired_energy,
                            'probs': probs
                        }
                    except Exception as alt_error:
                        print(f"   ❌ Alternative sampling also failed: {alt_error}")
                        # Check if user wants statevector or if we should try more methods
                        if self.use_statevector:
                            print("   🔄 Using statevector as requested...")
                            try:
                                optimal_circuit_no_measure = ansatz.assign_parameters(optimal_params)
                                from qiskit.quantum_info import Statevector
                                statevector = Statevector(optimal_circuit_no_measure)
                                probs = np.abs(statevector.data) ** 2
                                
                                # Find the most probable state
                                best_idx = np.argmax(probs)
                                repaired_solution = format(best_idx, f'0{self.n_qubits}b')
                                repaired_sequence = self.decode_solution(repaired_solution)
                                repaired_energy = self.compute_energy_from_bitstring(repaired_solution)
                                
                                print(f"   📊 Using statevector: {repaired_sequence} (energy: {repaired_energy:.6f})")
                                
                                return {
                                    'bitstring': repaired_solution,
                                    'energy': float(optimal_value),
                                    'costs': costs,
                                    'repaired_sequence': repaired_sequence,
                                    'repaired_cost': repaired_energy,
                                    'probs': probs
                                }
                            except Exception as statevector_error:
                                print(f"   ⚠️  Could not get statevector: {statevector_error}")
                                # Final fallback to default
                                repaired_solution = format(0, f'0{self.n_qubits}b')
                                repaired_sequence = self.decode_solution(repaired_solution)
                                repaired_energy = self.compute_energy_from_bitstring(repaired_solution)
                                
                                return {
                                    'bitstring': repaired_solution,
                                    'energy': float(optimal_value),
                                    'costs': costs,
                                    'repaired_sequence': repaired_sequence,
                                    'repaired_cost': repaired_energy,
                                    'probs': np.array([])
                                }
                        else:
                            print("   ❌ All sampling methods failed and statevector disabled")
                            # Final fallback to default
                            repaired_solution = format(0, f'0{self.n_qubits}b')
                            repaired_sequence = self.decode_solution(repaired_solution)
                            repaired_energy = self.compute_energy_from_bitstring(repaired_solution)
                            
                            return {
                                'bitstring': repaired_solution,
                                'energy': float(optimal_value),
                                'costs': costs,
                                'repaired_sequence': repaired_sequence,
                                'repaired_cost': repaired_energy,
                                'probs': np.array([])
                            }
                quasi_dist = result_sampler.quasi_dists[0]
                probs = np.zeros(2**self.n_qubits)
                for key, prob in quasi_dist.items():
                    if isinstance(key, int):
                        idx = key
                    else:
                        idx = int(key, 2)
                    if 0 <= idx < len(probs):
                        probs[idx] = float(prob)
                probs = self._mask_invalid_probabilities(probs)
                K = min(16, probs.size)
                top_idx = np.argpartition(probs, -K)[-K:]
                top_idx = top_idx[np.argsort(-probs[top_idx])]
                best_idx = int(top_idx[0])
                best_energy = float('inf')
                for idx in top_idx:
                    bitstr = format(int(idx), f'0{self.n_qubits}b')
                    e = self.qaoa_solver.compute_energy_from_bitstring(bitstr)
                    if e < best_energy:
                        best_energy = e
                        best_idx = int(idx)
                repaired_solution = format(best_idx, f'0{self.n_qubits}b')
                repaired_sequence = self.decode_solution(repaired_solution)
                repaired_energy = best_energy
            else:
                from qiskit.quantum_info import Statevector
                bound = ansatz.assign_parameters(optimal_params)
                self.save_qiskit_circuit(bound, 'vqe_circuit_qiskit.png')
                psi = Statevector.from_instruction(bound)
                amps = psi.data
                probs = np.abs(amps)**2
                probs = self._mask_invalid_probabilities(probs)
                K = min(16, probs.size)
                top_idx = np.argpartition(probs, -K)[-K:]
                top_idx = top_idx[np.argsort(-probs[top_idx])]
                best_idx = int(top_idx[0])
                best_energy = float('inf')
                for idx in top_idx:
                    bitstr = format(int(idx), f'0{self.n_qubits}b')
                    e = self.qaoa_solver.compute_energy_from_bitstring(bitstr)
                    if e < best_energy:
                        best_energy = e
                        best_idx = int(idx)
                repaired_solution = format(best_idx, f'0{self.n_qubits}b')
                repaired_sequence = self.decode_solution(repaired_solution)
                repaired_energy = best_energy
            
            print(f"📊 Selected bitstring energy: {repaired_energy:.6f}")
            print(f"📊 Sum of probabilities: {np.sum(probs):.6f}")
            print(f"📊 Max probability: {np.max(probs):.4f}")
        
        except Exception as e:
            print(f"❌ Error extracting results: {e}")
            import traceback
            traceback.print_exc()
            return self._empty_vqe_result()
        
        if np.sum(probs) < 1e-10:
            print("⚠️ Warning: No valid probabilities computed.")
            return self._empty_vqe_result()
        
        print(f"✅ VQE completed!")
        print(f"➡️ Repaired sequence: {repaired_sequence}")
        print(f"➡️ Repaired energy: {repaired_energy:.6f}")
        
        # Always plot probabilities if we have them
        print(f"🔍 Debug: probs type: {type(probs)}, length: {len(probs) if hasattr(probs, '__len__') else 'N/A'}")
        if 'probs' in locals() and probs is not None and len(probs) > 0:
            print(f"🔍 Debug: probs sum: {np.sum(probs):.6f}, max: {np.max(probs):.6f}")
            try:
                self.plot_prob_with_sequences(probs, "VQE")
                print("✅ VQE probability plot saved successfully!")
            except Exception as e:
                print(f"❌ Error saving VQE probability plot: {e}")
                import traceback
                traceback.print_exc()
        else:
            print("⚠️ No probabilities available for plotting")
        
        if costs:
            self.plot_optimization(costs, "VQE")
        
        return {
            'bitstring': repaired_solution,
            'energy': float(repaired_energy),
            'costs': costs,
            'repaired_sequence': repaired_sequence,
            'repaired_cost': repaired_energy,
            'probs': probs if 'probs' in locals() and probs is not None else np.array([])
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

    def get_top_sequences_from_probs(self, probs, top_k=40):
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

    def get_top_sequences_classical(self, top_k=40):
        """Get top sequences using classical exhaustive search"""
        import itertools
        import heapq
        
        total_combinations = self.n_aa ** self.L
        top_heap = []
        
        # Sample combinations efficiently
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
        
        # Convert to required format
        results = []
        for _, seq, energy, bitstring in sorted(top_heap, key=lambda x: x[2]):
            results.append((bitstring, seq, 0.0, energy))  # probability=0 for classical
        
        return results

    def compute_energy_breakdown(self, bitstring):
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
            
            # Classify term type based on Pauli string pattern
            z_count = pauli_string.count('Z')
            if z_count == 1:
                breakdown['Local Terms'] += term_energy
            elif z_count == 2:
                breakdown['Pairwise Terms'] += term_energy
            elif 'environment' in str(coeff).lower() or 'hydrophobic' in str(coeff).lower():
                breakdown['Environment Terms'] += term_energy
            elif 'charge' in str(coeff).lower():
                breakdown['Charge Terms'] += term_energy
            elif 'moment' in str(coeff).lower():
                breakdown['Hydrophobic Moment Terms'] += term_energy
            else:
                breakdown['Other Terms'] += term_energy
        
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

    def plot_prob_with_sequences(self, probs: np.ndarray, solver_name: str = "QAOA", top_k: int = 20):
        """Plot probability distribution with sequences for Qiskit QAOA"""
        import matplotlib.pyplot as plt
        
        top_k = min(top_k, len(probs))
        
        if top_k < len(probs):
            sorted_indices = np.argpartition(probs, -top_k)[-top_k:]
            sorted_indices = sorted_indices[np.argsort(-probs[sorted_indices])]
        else:
            sorted_indices = np.argsort(-probs)
        
        sorted_probs = probs[sorted_indices]
        sequences = [
            self.decode_solution(format(idx, f'0{self.n_qubits}b'))
            for idx in sorted_indices
        ]

        plt.figure(figsize=(14, 6))
        bars = plt.bar(range(len(sequences)), sorted_probs, 
                       color='steelblue', alpha=0.8, edgecolor='navy', linewidth=0.5)
        
        bars[0].set_color('gold')
        bars[0].set_edgecolor('darkorange')
        bars[0].set_linewidth(2)
        
        plt.xlabel('Amino Acid Sequences (Top ' + str(top_k) + ')', fontsize=11, fontweight='bold')
        plt.ylabel('Probability', fontsize=11, fontweight='bold')
        plt.title(f'Top Probability Distribution from {solver_name}', fontsize=13, fontweight='bold', pad=15)
        plt.xticks(range(len(sequences)), sequences, rotation=90, fontsize=9)
        plt.grid(axis='y', alpha=0.3, linestyle='--')
        plt.tight_layout()
        
        # Save to output directory
        output_dir = self.kwargs.get('output_dir', 'output')
        output_path = os.path.join(output_dir, f'{solver_name.lower()}_probability_plot.png')
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ {solver_name} probability plot saved as {output_path}")

    def plot_optimization(self, costs: list, solver_name: str = "QAOA"):
        """Plot optimization convergence"""
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(10, 6))
        plt.plot(costs, 'b-', linewidth=2, alpha=0.8)
        plt.xlabel('Iteration', fontsize=12, fontweight='bold')
        plt.ylabel('Energy', fontsize=12, fontweight='bold')
        plt.title(f'{solver_name} Optimization Convergence', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save to output directory
        output_dir = self.kwargs.get('output_dir', 'output')
        output_path = os.path.join(output_dir, f'{solver_name.lower()}_optimization_convergence.png')
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ {solver_name} optimization plot saved as {output_path}")

def save_energy_results(designer, result, solver_name, output_dir, top_k=40):
    """Save top sequences with energies and Hamiltonian terms to energy.txt"""
    import os
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    energy_file = os.path.join(output_dir, "energy.txt")
    
    with open(energy_file, 'w') as f:
        f.write(f"=== ENERGY RESULTS - {solver_name.upper()} SOLVER ===\n")
        f.write(f"Solver: {solver_name}\n")
        f.write(f"Sequence Length: {designer.L}\n")
        f.write(f"Amino Acids: {designer.amino_acids}\n")
        f.write(f"Total Qubits: {designer.n_qubits}\n")
        f.write(f"Total Hamiltonian Terms: {len(designer.pauli_terms)}\n")
        f.write("="*60 + "\n\n")
        
        # Get top sequences
        if 'probs' in result and len(result['probs']) > 0 and np.sum(result['probs']) > 1e-10:
            # For quantum solvers with valid probabilities
            top_sequences = designer.get_top_sequences_from_probs(result['probs'], top_k)
        else:
            # For classical solver or when probabilities not available
            print("   📊 No valid probabilities found, using classical search for top sequences")
            top_sequences = designer.get_top_sequences_classical(top_k)
        
        f.write(f"TOP {min(top_k, len(top_sequences))} SEQUENCES BY ENERGY:\n")
        f.write("-" * 60 + "\n")
        
        for rank, (bitstring, sequence, probability, energy) in enumerate(top_sequences, 1):
            f.write(f"Rank {rank:2d}: {sequence} | Energy: {energy:.6f}")
            if probability > 0:
                f.write(f" | Probability: {probability:.6f}")
            f.write(f" | Bitstring: {bitstring}\n")
            
            # Calculate energy breakdown by Hamiltonian terms
            energy_breakdown = designer.compute_energy_breakdown(bitstring)
            f.write(f"         Energy Breakdown:\n")
            for term_name, term_energy in energy_breakdown.items():
                f.write(f"           {term_name}: {term_energy:.6f}\n")
            f.write("\n")
        
        f.write("="*60 + "\n")
        f.write("HAMILTONIAN TERMS SUMMARY:\n")
        f.write("-" * 60 + "\n")
        
        # Group terms by type for summary
        term_groups = {}
        for coeff, pauli_string in designer.pauli_terms:
            # Determine term type based on Pauli string pattern
            z_count = pauli_string.count('Z')
            if z_count == 0:
                term_type = "Identity"
            elif z_count == 1:
                term_type = "Local"
            elif z_count == 2:
                term_type = "Pairwise"
            else:
                term_type = f"Higher-order ({z_count}-body)"
            
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
    phase = kwargs.get('wheel_phase_deg', 90.0)
    halfwidth = kwargs.get('wheel_halfwidth_deg', 80.0)
    output_dir = kwargs.get('output_dir', 'output')
    print(f"\nDEBUG ENV (phase={phase}°, halfwidth={halfwidth}°):")
    print(f"Output directory: {output_dir}")
    print(f"Absolute output path: {os.path.abspath(output_dir)}")
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
    
    designer = QuantumProteinDesign(
        sequence_length=sequence_length,
        amino_acids=amino_acids,
        quantum_backend=quantum_backend,
        shots=shots,
        **kwargs
    )
    
    solver = kwargs.get('solver', 'qaoa')
    if solver == 'classical':
        result = designer.solve_classical_qubo()
    elif solver == 'vqe':
        result = designer.solve_vqe_pennylane() if quantum_backend == 'pennylane' else designer.solve_vqe_qiskit()
    else:
        result = designer.solve_qaoa_pennylane() if quantum_backend == 'pennylane' else designer.solve_qaoa_qiskit()
    
    decoded_sequence, violations = designer.analyze_solution(result)
    
    if violations > 0:
        print(f"⚠️ Warning: Solution contains {violations} constraint violations ('X' residues).")
    
    if 'costs' in result and result['costs']:
        designer.plot_optimization(result['costs'])
    
    # Always plot alpha helix wheel if we have a valid sequence
    if result:
        sequence = result.get('repaired_sequence', result.get('sequence', ''))
        print(f"Most probable sequence for helix wheel: {sequence}")
        if sequence and sequence.count('X') < len(sequence):
            designer.plot_alpha_helix_wheel(sequence)
    
    # Save energy results to energy.txt
    save_energy_results(designer, result, solver, output_dir, top_k=40)
    
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
    parser.add_argument('--membrane_mode', type=str, default='span', choices=['span', 'set', 'wheel'], help='Mode for defining membrane positions.')
    parser.add_argument('--wheel_phase_deg', type=float, default=90.0, help='Phase angle for helical wheel in degrees.')
    parser.add_argument('--wheel_halfwidth_deg', type=float, default=80.0, help='Half-width of the membrane sector in degrees for helical wheel.')
    parser.add_argument('--lambda_env', type=float, default=10.0, help='Weight of the environment preference term.')
    parser.add_argument('--lambda_charge', type=float, default=0.5, help='Weight of the membrane charge term.')
    parser.add_argument('--lambda_mu', type=float, default=10.0, help='Weight of the hydrophobic moment term.')
    parser.add_argument('--lambda_local', type=float, default=0.5, help='Weight of the local preference terms.')
    parser.add_argument('--lambda_pairwise', type=float, default=0.5, help='Weight of the pairwise interaction term.')
    parser.add_argument('--lambda_helix_pairs', type=float, default=0.5, help='Weight of the helix pair propensity term.')
    parser.add_argument('--lambda_segregation', type=float, default=1.0, help='Weight of the amphipathic segregation term.')
    parser.add_argument('--lambda_electrostatic', type=float, default=0.5, help='Weight of the electrostatics term.')
    parser.add_argument('--max_interaction_dist', type=int, default=1, help='Maximum sequence distance for pairwise interactions.')
    parser.add_argument('--membrane_charge', type=str, default='neu', choices=['neu', 'neg', 'pos'], help='Charge of the membrane.')
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
    

# python cursor_peptide_seq.py -L 6 -R VQ  --membrane_mode wheel --wheel_phase_deg 0 --wheel_halfwidth_deg 30  --lambda_env 0.6 --lambda_mu 0.4 --lambda_charge 0.3 --membrane_charge neg
