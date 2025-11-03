# core/protein_designer.py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import Dict, List, Any, Tuple, Optional
from core.hamiltonian_builder import HamiltonianBuilder
from core.solvers import QAOASolver, ClassicalSolver
from utils.general_utils import decode_solution, get_qubit_index, compute_energy_from_bitstring
import pennylane as qml
from pennylane import numpy as qnp

class QuantumProteinDesign:
    def __init__(self, sequence_length: int, amino_acids: List[str] = None,
                 quantum_backend: str = 'pennylane', **kwargs):
        self.L = sequence_length
        self.amino_acids = amino_acids
        self.n_aa = len(amino_acids)
        self.bits_per_pos = max(1, int(np.ceil(np.log2(self.n_aa))))
        self.n_qubits = self.L * self.bits_per_pos
        self.kwargs = kwargs
        self.backend = quantum_backend
        
        print("🧬 QUANTUM PROTEIN DESIGN SETUP 🧬")
        print(f"Sequence length: {self.L}")
        print(f"Amino acids: {self.amino_acids}")
        print(f"Bits per position: {self.bits_per_pos}")
        print(f"Required qubits: {self.n_qubits}")
        print(f"Quantum backend: {self.backend}")
        print("="*50)

        self.hamiltonian_builder = HamiltonianBuilder(
            L=self.L,
            amino_acids=self.amino_acids,
            bits_per_pos=self.bits_per_pos,
            n_qubits=self.n_qubits,
            **self.kwargs
        )
        self.pauli_terms, self.cost_hamiltonian = self.hamiltonian_builder.build_hamiltonian(self.backend)
        
        self.qaoa_solver = QAOASolver(
            cost_hamiltonian=self.cost_hamiltonian,
            n_qubits=self.n_qubits,
            pauli_terms=self.pauli_terms,
            amino_acids=self.amino_acids,
            L=self.L,
            bits_per_pos=self.bits_per_pos
        )
        self.classical_solver = ClassicalSolver(
            L=self.L,
            n_aa=self.n_aa,
            bits_per_pos=self.bits_per_pos,
            pauli_terms=self.pauli_terms,
            amino_acids=self.amino_acids
        )
    
    def plot_qaoa_circuit(self, p_layers: int = 1):
        """Draws the QAOA circuit for a given number of layers."""
        dev = qml.device('default.qubit', wires=self.n_qubits)
        @qml.qnode(dev)
        def circuit(params):
            for i in range(self.n_qubits):
                qml.Hadamard(wires=i)
            for layer in range(p_layers):
                qml.ApproxTimeEvolution(self.cost_hamiltonian, params[0][layer], 1)
                beta = params[1][layer]
                for w in range(self.n_qubits):
                    qml.RX(2 * beta, wires=w)
            return qml.expval(qml.PauliZ(0))
        
        print("\n🎨 Drawing QAOA Circuit...")
        dummy_params = (qnp.zeros(p_layers), qnp.zeros(p_layers))
        print(qml.draw(circuit)(dummy_params))
        
    def solve_qaoa_pennylane(self, p_layers: int = 2, max_iterations: int = 200, n_starts: int = 4, init_strategy: str = 'linear', warm_start: bool = True) -> Dict[str, Any]:
        """Solve using QAOA with PennyLane"""
        print(f"\n🔥 Solving with PennyLane QAOA (p={p_layers})...")
        def make_cost(p_local: int):
            @qml.qnode(self.qaoa_solver.dev)
            def _cost(params):
                for i in range(self.n_qubits):
                    qml.Hadamard(wires=i)
                for layer in range(p_local):
                    qml.ApproxTimeEvolution(self.qaoa_solver.cost_hamiltonian, params[0][layer], 1)
                    beta = params[1][layer]
                    for w in range(self.n_qubits):
                        qml.RX(2 * beta, wires=w)
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
        
        try:
            fig, ax = qml.draw_mpl(cost_function)(best_params)
            fig.suptitle("QAOA Circuit")
            plt.show()
        except Exception as e:
            print(f"Could not render matplotlib circuit: {e}")
            
        @qml.qnode(self.qaoa_solver.dev)
        def get_probabilities(params):
            for i in range(self.n_qubits):
                qml.Hadamard(wires=i)
            for p in range(p_layers):
                qml.ApproxTimeEvolution(self.qaoa_solver.cost_hamiltonian, params[0][p], 1)
                beta = params[1][p]
                for w in range(self.n_qubits):
                    qml.RX(2 * beta, wires=w)
            return qml.probs(wires=range(self.n_qubits))
        
        probs = get_probabilities(best_params)
        best_bitstring_int = np.argmax(probs)
        best_bitstring = format(best_bitstring_int, f'0{self.n_qubits}b')
        repaired_solution = self.qaoa_solver._repair_with_marginals(probs)
        repaired_sequence = decode_solution(repaired_solution, self.L, self.bits_per_pos, self.amino_acids)
        repaired_energy = compute_energy_from_bitstring(repaired_solution, self.pauli_terms)
        
        print(f"✅ QAOA completed! Final cost: {best_cost:.6f}")
        print(f"Best solution probability: {max(probs):.4f}")
        print(f"➡️ Repaired sequence: {repaired_sequence} | Energy (classical): {repaired_energy:.6f}")
        
        return {
            'bitstring': best_bitstring,
            'energy': best_cost,
            'costs': best_costs_trace,
            'repaired_sequence': repaired_sequence,
            'repaired_cost': repaired_energy
        }

    def solve_classical_qubo(self) -> Dict[str, Any]:
        result = self.classical_solver.solve()
        return result

    def analyze_solution(self, result: Dict[str, Any]):
        print("\n🧬 QUANTUM SOLUTION ANALYSIS 🧬")
        bitstring = result['bitstring']
        repaired_sequence = result['repaired_sequence']
        repaired_energy = result['repaired_cost']
        
        print(f"Binary solution: {bitstring}")
        print(f"Decoded sequence: {repaired_sequence}")
        print(f"Final energy: {repaired_energy:.6f}")
        
        violation_count = repaired_sequence.count('X')
        print(f"Constraint violations: {violation_count}/{self.L}")
        
        if violation_count == 0:
            hydrophobic_residues = sum(1 for aa in repaired_sequence if aa in ['A', 'L', 'I', 'M', 'F', 'W', 'V'])
            charged_residues = sum(1 for aa in repaired_sequence if aa in ['E', 'K', 'R', 'D'])
            print(f"Hydrophobic residues: {hydrophobic_residues}/{self.L}")
            print(f"Charged residues: {charged_residues}/{self.L}")
        
        return repaired_sequence, violation_count

    def plot_optimization(self, costs: List[float]):
        plt.figure(figsize=(10, 6))
        plt.plot(costs, 'b-', linewidth=2)
        plt.xlabel('Iteration')
        plt.ylabel('Energy')
        plt.title('Quantum Optimization Convergence')
        plt.grid(True, alpha=0.3)
        plt.show()

    def plot_alpha_helix_wheel(self, sequence: str):
        polar = set(['S','T','N','Q','Y','C','G'])
        nonpolar = set(['A','V','L','I','M','F','W','P'])
        negative = set(['D','E'])
        positive = set(['K','R','H'])
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
            plt.text(xs[i], ys[i], aa, ha='center', va='center', fontsize=12, weight='bold', color='white', zorder=4)
            r_idx = radius + 0.12
            ang_i = angles[i]
            xi = r_idx * np.cos(ang_i)
            yi = r_idx * np.sin(ang_i)
            plt.text(xi, yi, f"{i+1}", ha='center', va='center', fontsize=9, color='black', zorder=5)
        for i in range(len(sequence) - 1):
            plt.plot([xs[i], xs[i+1]], [ys[i], ys[i+1]], color='k', alpha=0.35, linewidth=1.5, zorder=2)
        circle = plt.Circle((0, 0), radius, color='k', fill=False, alpha=0.3)
        ax = plt.gca()
        ax.add_artist(circle)
        try:
            if self.kwargs.get('membrane_mode', 'span') == 'wheel':
                phase = np.deg2rad(self.kwargs.get('wheel_phase_deg', 0.0))
                halfw = np.deg2rad(self.kwargs.get('wheel_halfwidth_deg', 40.0))
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
                ax.text(xm, ym, 'Membrane (lipids)', ha='center', va='center', fontsize=10, color='#8B4513', weight='bold')
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