import numpy as np
from pennylane import numpy as qnp
import pennylane as qml
from pennylane import qaoa
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import Dict, List, Tuple, Optional, Any
import itertools
import argparse
import sys

# Clasico y helice a mas vecinos
# Calculo estadistico de combinaciones posibles intentant hacer 4 qubits por posicion, intentant juntar helices mas parecidos juntarlos
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
                 quantum_backend: str = 'pennylane', shots: int = 1000, **kwargs):
        self.L = sequence_length
        self.amino_acids = amino_acids
        self.n_aa = len(amino_acids)
        self.bits_per_pos = max(1, int(np.ceil(np.log2(self.n_aa))))
        self.n_qubits = self.L * self.bits_per_pos
        self.backend = quantum_backend
        self.shots = shots  # Default to 1000 shots
        self.kwargs = kwargs
        
        print("🧬 QUANTUM PROTEIN DESIGN SETUP 🧬")
        print(f"Sequence length: {self.L}")
        print(f"Amino acids: {self.amino_acids}")
        print(f"Bits per position: {self.bits_per_pos}")
        print(f"Required qubits: {self.n_qubits}")
        print(f"Quantum backend: {self.backend}")
        print(f"Number of shots: {self.shots}")
        print("="*50)

        from core.hamiltonian_builder import HamiltonianBuilder
        from core.solvers import QAOASolver, ClassicalSolver
        
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
            bits_per_pos=self.bits_per_pos,
            shots=self.shots  # Pass shots to solver
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
        decoded_sequence = ""
        for i in range(self.L):
            pos_code_str = bitstring[i*self.bits_per_pos:(i+1)*self.bits_per_pos]
            pos_code_int = int(pos_code_str, 2)
            if pos_code_int < self.n_aa:
                decoded_sequence += self.amino_acids[pos_code_int]
            else:
                decoded_sequence += 'X'  # Violation or invalid code
        return decoded_sequence

    def plot_prob_with_sequences(self, probs: np.ndarray, solver_name: str, top_k: int = 20):
        """
        Plots a bar chart of the top_k amino acid sequences and their probabilities.
        """
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt

        # Sort the probabilities and get the top_k indices
        sorted_indices = np.argsort(probs)[::-1][:top_k]
        sorted_probs = probs[sorted_indices]
        sequences = [self.decode_solution(format(idx, f'0{self.n_qubits}b')) for idx in sorted_indices]

        plt.figure(figsize=(12, 6))
        plt.bar(range(len(sequences)), sorted_probs, color='blue' if solver_name == 'QAOA' else 'green')
        plt.xlabel('Amino Acid Sequences (Top ' + str(top_k) + ')')
        plt.ylabel('Probability')
        plt.title(f'Top Probability Distribution of Amino Acid Sequences from {solver_name}')
        plt.xticks(range(len(sequences)), sequences, rotation=90)
        plt.tight_layout()
        plt.savefig(f'{solver_name.lower()}_probability_plot.png')  # Save to file
        plt.close()  # Close the figure to avoid memory issues
        print(f"Plot saved as {solver_name.lower()}_probability_plot.png")

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
            fig, ax = qml.draw_mpl(make_cost(p_layers), show_all_wires=True)(best_params)
            fig.suptitle("QAOA Optimized Circuit")
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
        print(f"Number of shots used: {self.qaoa_solver.dev.shots}")  # Debug shot count
        print("Probabilities:", probs)  # Debug print
        print("Number of probabilities:", len(probs))  # Debug print
        if len(probs) == 0:
            print("Warning: No probabilities computed. Check Hamiltonian or circuit.")
            return {
                'bitstring': '',
                'energy': float('inf'),
                'costs': best_costs_trace,
                'repaired_sequence': '',
                'repaired_cost': float('inf')
            }

        best_bitstring_int = np.argmax(probs)
        best_bitstring = format(best_bitstring_int, f'0{self.n_qubits}b')
        repaired_solution = self.qaoa_solver._repair_with_marginals(probs)
        repaired_sequence = self.decode_solution(repaired_solution)
        repaired_energy = self.qaoa_solver.compute_energy_from_bitstring(repaired_solution)
        
        print(f"✅ QAOA completed! Final cost: {best_cost:.6f}")
        print(f"Best solution probability: {max(probs):.4f}")
        print(f"➡️ Repaired sequence: {repaired_sequence} | Energy (classical): {repaired_energy:.6f}")
        
        # Plot probability distribution with sequences
        self.plot_prob_with_sequences(probs, "QAOA")

        return {
            'bitstring': best_bitstring,
            'energy': best_cost,
            'costs': best_costs_trace,
            'repaired_sequence': repaired_sequence,
            'repaired_cost': repaired_energy
        }

    def solve_vqe_pennylane(self, layers: int = 2, max_iterations: int = 200, n_starts: int = 4) -> Dict[str, Any]:
        """Solve using VQE with PennyLane"""
        print(f"\n🔥 Solving with PennyLane VQE (layers={layers})...")
        
        # Create a new device with the specified number of shots
        dev = qml.device('lightning.qubit', wires=self.n_qubits, shots=self.shots)
        
        def make_cost():
            @qml.qnode(dev)
            def _cost(params):
                qml.StronglyEntanglingLayers(params, wires=range(self.n_qubits))
                return qml.expval(self.cost_hamiltonian)
            return _cost
        
        cost_function = make_cost()
        
        shape = qml.StronglyEntanglingLayers.shape(n_layers=layers, n_wires=self.n_qubits)
        
        best_cost = np.inf
        best_params = None
        best_costs_trace = []
        
        for start in range(n_starts):
            params = qnp.random.uniform(0, 2 * np.pi, shape, requires_grad=True)
            optimizer = qml.AdamOptimizer(stepsize=0.1)
            costs = []
            for i in range(max_iterations // 2):
                params, cost = optimizer.step_and_cost(cost_function, params)
                costs.append(cost)
            optimizer_fine = qml.AdamOptimizer(stepsize=0.02)
            for i in range(max_iterations // 2, max_iterations):
                params, cost = optimizer_fine.step_and_cost(cost_function, params)
                costs.append(cost)
            
            print(f"Start {start+1}/{n_starts}: final cost {costs[-1]:.6f}")
            if costs[-1] < best_cost:
                best_cost = costs[-1]
                best_params = params
                best_costs_trace = costs
        
        try:
            fig, ax = qml.draw_mpl(cost_function, show_all_wires=True)(best_params)
            fig.suptitle("VQE Optimized Circuit")
            plt.show()
        except Exception as e:
            print(f"Could not render matplotlib circuit: {e}")
        
        @qml.qnode(dev)
        def get_probabilities(params):
            qml.StronglyEntanglingLayers(params, wires=range(self.n_qubits))
            return qml.probs(wires=range(self.n_qubits))
        
        probs = get_probabilities(best_params)
        print(f"Number of shots used: {dev.shots}")  # Debug shot count
        print("Probabilities:", probs)  # Debug print
        print("Number of probabilities:", len(probs))  # Debug print
        if len(probs) == 0:
            print("Warning: No probabilities computed. Check Hamiltonian or circuit.")
            return {
                'bitstring': '',
                'energy': float('inf'),
                'costs': best_costs_trace,
                'repaired_sequence': '',
                'repaired_cost': float('inf')
            }

        repaired_solution = self.qaoa_solver._repair_with_marginals(probs)
        repaired_sequence = self.decode_solution(repaired_solution)
        repaired_energy = self.qaoa_solver.compute_energy_from_bitstring(repaired_solution)
        
        print(f"✅ VQE completed! Final cost: {best_cost:.6f}")
        print(f"Best solution probability: {max(probs):.4f}")
        print(f"➡️ Repaired sequence: {repaired_sequence} | Energy (classical): {repaired_energy:.6f}")
        
        # Plot probability distribution with sequences
        self.plot_prob_with_sequences(probs, "VQE")

        return {
            'bitstring': repaired_solution,
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
        plt.savefig('optimization_convergence.png')  # Save to file
        plt.close()

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
        plt.savefig('alpha_helix_wheel.png')  # Save to file
        plt.close()

def run_quantum_protein_design(sequence_length, amino_acids, quantum_backend='pennylane', 
                               shots: int = 1000, **kwargs):
    
    designer = QuantumProteinDesign(
        sequence_length=sequence_length,
        amino_acids=amino_acids,
        quantum_backend=quantum_backend,
        shots=shots,  # Pass shots to constructor
        **kwargs
    )
    
    solver = kwargs.get('solver', 'qaoa')
    if solver == 'classical':
        result = designer.solve_classical_qubo()
    elif solver == 'vqe':
        result = designer.solve_vqe_pennylane()
    else:
        result = designer.solve_qaoa_pennylane()
    
    decoded_sequence, violations = designer.analyze_solution(result)
    
    if violations > 0:
        print(f"⚠️ Warning: Solution contains {violations} constraint violations ('X' residues).")
    
    if 'costs' in result:
        designer.plot_optimization(result['costs'])
    
    return designer, result

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Quantum protein sequence design.')
    parser.add_argument('-L', '--length', type=int, default=4, help='Sequence length.')
    parser.add_argument('-R', '--residues', type=str, default="V,Q,L,R", help='Amino acids to use, comma-separated.')
    parser.add_argument('-b', '--backend', type=str, default='pennylane', choices=['pennylane', 'qiskit'], help='Quantum backend to use.')
    parser.add_argument('--solver', type=str, default='qaoa', choices=['qaoa', 'vqe', 'classical'], help='Solver to use.')
    parser.add_argument('--shots', type=int, default=1000, help='Number of shots for quantum simulation.')
    parser.add_argument('--membrane', type=str, help='Membrane span (e.g., 1:4)')
    parser.add_argument('--membrane_positions', type=str, help='Membrane positions (e.g., 0,2,5)')
    parser.add_argument('--membrane_mode', type=str, default='span', choices=['span', 'set', 'wheel'], help='Mode for defining membrane positions.')
    parser.add_argument('--wheel_phase_deg', type=float, default=0.0, help='Phase angle for helical wheel in degrees.')
    parser.add_argument('--wheel_halfwidth_deg', type=float, default=90.0, help='Half-width of the membrane sector in degrees for helical wheel.')
    parser.add_argument('--lambda_env', type=float, default=0.0, help='Weight of the environment preference term.')
    parser.add_argument('--lambda_charge', type=float, default=0.0, help='Weight of the membrane charge term.')
    parser.add_argument('--lambda_mu', type=float, default=0.0, help='Weight of the hydrophobic moment term.')
    parser.add_argument('--lambda_local', type=float, default=1.0, help='Weight of the local preference terms.')
    parser.add_argument('--lambda_pairwise', type=float, default=1.0, help='Weight of the pairwise interaction term (Miyazawa-Jernigan).')
    parser.add_argument('--lambda_helix_pairs', type=float, default=0.0, help='Weight of the helix pair propensity term.')
    parser.add_argument('--max_interaction_dist', type=int, default=3, help='Maximum sequence distance for pairwise interactions.')
    parser.add_argument('--membrane_charge', type=str, default='neu', choices=['neu', 'neg', 'pos'], help='Charge of the membrane.')
    
    args = parser.parse_args()
    
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

    # Run the quantum protein design
    designer, qaoa_result = run_quantum_protein_design(
        sequence_length=args.length,
        amino_acids=aa_list,
        quantum_backend=args.backend,
        shots=args.shots,  # Pass shots from command line
        membrane_span=mem_span,
        membrane_charge=args.membrane_charge,
        lambda_charge=args.lambda_charge,
        lambda_env=args.lambda_env,
        lambda_mu=args.lambda_mu,
        lambda_local=args.lambda_local,
        lambda_pairwise=args.lambda_pairwise,
        lambda_helix_pairs=args.lambda_helix_pairs,
        max_interaction_dist=args.max_interaction_dist,
        membrane_positions=mem_positions,
        membrane_mode=args.membrane_mode,
        wheel_phase_deg=args.wheel_phase_deg,
        wheel_halfwidth_deg=args.wheel_halfwidth_deg,
        solver=args.solver,
    )
    
    # Show the results
    print("Optimization complete!")
    
    if args.solver == 'classical':
        print("\n🏆 Solución Clásica:")
        print(f"Secuencia: {qaoa_result['sequence']}")
        print(f"Energía: {qaoa_result['energy']:.6f}")
    elif args.solver == 'vqe':
        print("\n⚛️ Solución Cuántica (VQE):")
        print(f"Secuencia Reparada: {qaoa_result['repaired_sequence']}")
        print(f"Energía Final: {qaoa_result['repaired_cost']:.6f}")
    else:
        print("\n⚛️ Solución Cuántica (QAOA):")
        print(f"Secuencia Reparada: {qaoa_result['repaired_sequence']}")
        print(f"Energía Final: {qaoa_result['repaired_cost']:.6f}")

    if args.membrane_mode == 'wheel' and qaoa_result:
        sequence = qaoa_result.get('repaired_sequence', qaoa_result.get('sequence'))
        designer.plot_alpha_helix_wheel(sequence)

#  python main_final.py -L 6 -R V,Q,A,N --lambda_pairwise 1.0 --lambda_helix_pairs 1.5 --lambda_env 2.0 --lambda_charge 1.5 --lambda_mu 1.0 --membrane_mode wheel --wheel_phase_deg 0 --wheel_halfwidth_deg 90 --shots 1000
# python main_final.py -L 6 -R V,A,N,S --lambda_pairwise 1.0 --lambda_helix_pairs 1.5 --membrane_mode wheel --wheel_phase_deg -90 --wheel_halfwidth_deg 90 --shots 1000