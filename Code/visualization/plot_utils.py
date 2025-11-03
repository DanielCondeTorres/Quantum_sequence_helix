import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

class ProteinPlotter:
    """Clase para manejar todas las visualizaciones del diseño de proteínas"""
    
    def __init__(self, output_dir='output'):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
    
    def plot_prob_with_sequences(self, probs: np.ndarray, decoder_fn, n_qubits: int, 
                                solver_name: str = "QAOA", top_k: int = 20):
        """Plot probability distribution with sequences"""
        if len(probs) == 0 or np.all(probs == 0):
            print(f"Warning: No valid probabilities for {solver_name} plot")
            return
        
        top_k = min(top_k, len(probs))
        
        if top_k < len(probs):
            sorted_indices = np.argpartition(probs, -top_k)[-top_k:]
            sorted_indices = sorted_indices[np.argsort(-probs[sorted_indices])]
        else:
            sorted_indices = np.argsort(-probs)
        
        sorted_probs = probs[sorted_indices]
        sequences = [
            decoder_fn(format(idx, f'0{n_qubits}b'))
            for idx in sorted_indices
        ]
        
        print(f"Top {top_k} sequences: {sequences}")
        print(f"Top {top_k} probabilities: {sorted_probs}")

        plt.figure(figsize=(14, 6))
        bars = plt.bar(range(len(sequences)), sorted_probs, 
                       color='steelblue', alpha=0.8, edgecolor='navy', linewidth=0.5)
        
        if len(bars) > 0:
            bars[0].set_color('gold')
            bars[0].set_edgecolor('darkorange')
            bars[0].set_linewidth(2)
        
        plt.xlabel('Amino Acid Sequences (Top ' + str(top_k) + ')', fontsize=16, fontweight='bold')
        plt.ylabel('Probability', fontsize=16, fontweight='bold')
        #plt.title(f'Top Probability Distribution from {solver_name}', fontsize=13, fontweight='bold', pad=15)
        plt.xticks(range(len(sequences)), sequences, rotation=90, fontsize=14)
        plt.yticks(fontsize=14)
        plt.grid(axis='y', alpha=0.3, linestyle='--')
        plt.tight_layout()
        
        output_path = os.path.join(self.output_dir, f'{solver_name.lower()}_probability_plot.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ {solver_name} probability plot saved as {output_path}")
    
    def plot_alpha_helix_wheel(self, sequence: str, membrane_mode: str = 'span', 
                            wheel_phase_deg: float = -50.0,  # Adjusted to -50.0
                            wheel_halfwidth_deg: float = 120.0):  # Adjusted to 120.0
        """Plot alpha helix wheel visualization"""
        if not sequence or sequence.count('X') == len(sequence):
            print(f"Warning: Invalid sequence for helix wheel: {sequence}")
            return
        
        print(f"Plotting alpha helix wheel for sequence: {sequence}")
        
        # Color mapping for amino acids
        polar = set(['S', 'T', 'N', 'Q', 'Y', 'C', 'G'])
        nonpolar = set(['A', 'V', 'L', 'I', 'M', 'F', 'W', 'P'])
        negative = set(['D', 'E'])
        positive = set(['K', 'R', 'H'])
        
        color_map = {}
        for aa in sequence:
            if aa in negative:
                color_map[aa] = 'red'
            elif aa in positive:
                color_map[aa] = 'blue'
            elif aa in nonpolar:
                color_map[aa] = '#8B4513'
            elif aa in polar:
                color_map[aa] = 'green'
            else:
                color_map[aa] = 'gray'
        
        # Calculate rotated positions
        angle_increment = np.deg2rad(100.0)
        radius = 1.0
        phase_rad = np.deg2rad(wheel_phase_deg)
        angles = [(i * angle_increment + phase_rad) for i in range(len(sequence))]
        xs = [radius * np.cos(a) for a in angles]
        ys = [radius * np.sin(a) for a in angles]
        
        plt.figure(figsize=(7, 7))
        
        # Plot amino acids
        for i, aa in enumerate(sequence):
            plt.scatter(xs[i], ys[i], s=600, color=color_map[aa], 
                    edgecolors='k', zorder=3)
            plt.text(xs[i], ys[i], aa, ha='center', va='center', 
                    fontsize=14, weight='bold', color='white', zorder=4)
            
            # Position labels
            r_idx = radius + 0.2
            ang_i = angles[i]
            xi = r_idx * np.cos(ang_i)
            yi = r_idx * np.sin(ang_i)
            plt.text(xi, yi, f"{i+1}", ha='center', va='center', 
                    fontsize=14, color='black', zorder=5)
        
        # Connect residues
        for i in range(len(sequence) - 1):
            plt.plot([xs[i], xs[i+1]], [ys[i], ys[i+1]], 
                    color='k', alpha=0.35, linewidth=1.5, zorder=2)
        
        # Circle
        circle = plt.Circle((0, 0), radius, color='k', fill=False, alpha=0.3)
        ax = plt.gca()
        ax.add_artist(circle)
        
        # Membrane visualization for wheel mode
        if membrane_mode == 'wheel':
            halfw = np.deg2rad(wheel_halfwidth_deg)
            wedge = mpatches.Wedge(
                center=(0, 0), 
                r=radius, 
                theta1=np.rad2deg(-halfw),
                theta2=np.rad2deg(halfw), 
                facecolor='#FFE4B5', 
                alpha=0.3
            )
            ax.add_patch(wedge)
            
            mid_ang = 0
            xm = 1.25 * radius * np.cos(mid_ang)
            ym = 1.15 * radius * np.sin(mid_ang)
            ax.text(xm * 1.1, ym, 'Lipids', ha='center', va='center', 
                fontsize=14, color='#8B4513', weight='bold')
            
            xa = 1.35 * radius * np.cos(mid_ang + np.pi)
            ya = 1.15 * radius * np.sin(mid_ang + np.pi)
            ax.text(xa, ya, 'Water', ha='center', va='center', 
                fontsize=14, color='teal', weight='bold')
        
        ax.set_aspect('equal')
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.axis('off')
        plt.title('Alpha-Helix Wheel')
        
        output_path = os.path.join(self.output_dir, 'alpha_helix_wheel.png')
        plt.savefig(output_path, dpi=300)
        plt.close()
        print(f"Alpha helix wheel plot saved as {output_path}")
    
    def save_pennylane_circuit(self, circuit_func, params, filename: str):
        """Save PennyLane circuit to a PNG file."""
        print(f"Attempting to save PennyLane circuit: {filename}")
        try:
            import pennylane as qml
            matplotlib.use('Agg')
            
            if params is not None:
                fig, ax = qml.draw_mpl(circuit_func, show_all_wires=True)(params)
            else:
                fig, ax = qml.draw_mpl(circuit_func, show_all_wires=True)()
            
            fig.suptitle(filename.replace('.png', '').replace('_', ' ').title(), fontsize=14)
            output_path = os.path.join(self.output_dir, filename)
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
            matplotlib.use('Agg')
            from qiskit.visualization import circuit_drawer
            
            output_path = os.path.join(self.output_dir, filename)
            
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
            
            
    def plot_optimization(self, costs: list, solver_name: str, tick_size: int = 14):
        """
        Plots the optimization convergence (costs vs iterations).
        Args:
            costs: List of cost/energy values per iteration.
            solver_name: Solver name (e.g., 'QAOA', 'VQE').
        """
        if not costs or len(costs) < 2:
            print(f"⚠️ Not enough data to plot {solver_name} (len={len(costs)})")
            return
        
        iterations = np.arange(len(costs))
        plt.figure(figsize=(10, 6))
        
        # Main line
        plt.plot(iterations, costs, 'b-', linewidth=2, marker='o', markersize=4, alpha=0.8, label='Cost')
        
        # 🔥 CAMBIO: Highlight the BEST (mínimo) cost, NO el último
        best_idx = np.argmin(costs)  # Índice del mínimo
        best_cost = costs[best_idx]
        plt.scatter(best_idx, best_cost, color='red', s=250, zorder=5,
                    edgecolor='darkred', linewidth=3, label=f'Minimum energy')#{best_cost:.4f} (iter {best_idx})')
        
        plt.tick_params(axis='both', which='major', labelsize=tick_size)  # ← AQUÍ!
        plt.tick_params(axis='both', which='minor', labelsize=tick_size-2)
        
        # Styling
        plt.xlabel('Iterations', fontsize=15, fontweight='bold')
        plt.ylabel('Energy / Cost', fontsize=15, fontweight='bold')
        # plt.title(f'{solver_name} Convergence - {len(costs)} Iterations', fontsize=14, fontweight='bold', pad=15)
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.legend(loc='best')
        plt.tight_layout()
        
        # Save
        output_path = os.path.join(self.output_dir, f'{solver_name.lower()}_optimization_convergence.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ {solver_name} convergence plot saved: {output_path}")
        print(f"🔥 Best values: Start={costs[0]:.4f} → **BEST={best_cost:.4f}** (at iteration {best_idx})")

    '''   def plot_optimization(self, costs: list, solver_name: str, tick_size: int = 14):
        """
        Plots the optimization convergence (costs vs iterations).
        
        Args:
            costs: List of cost/energy values per iteration.
            solver_name: Solver name (e.g., 'QAOA', 'VQE').
        """
        if not costs or len(costs) < 2:
            print(f"⚠️ Not enough data to plot {solver_name} (len={len(costs)})")
            return
        
        iterations = np.arange(len(costs))
        
        plt.figure(figsize=(10, 6))
        
        # Main line
        plt.plot(iterations, costs, 'b-', linewidth=2, marker='o', markersize=4, alpha=0.8, label='Cost')
        
        # Highlight the best final cost
        final_cost = costs[-1]
        plt.scatter(len(costs)-1, final_cost, color='red', s=200, zorder=5, 
                    edgecolor='darkred', linewidth=2, label=f'Best: {final_cost:.4f}')
        plt.tick_params(axis='both', which='major', labelsize=tick_size)  # ← AQUÍ!
        plt.tick_params(axis='both', which='minor', labelsize=tick_size-2)
        # Styling
        plt.xlabel('Iterations', fontsize=15, fontweight='bold')
        plt.ylabel('Energy / Cost', fontsize=15, fontweight='bold')
        #plt.title(f'{solver_name} Convergence - {len(costs)} Iterations', fontsize=14, fontweight='bold', pad=15)
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.legend(loc='best')
        plt.tight_layout()
        
        # Save
        output_path = os.path.join(self.output_dir, f'{solver_name.lower()}_optimization_convergence.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ {solver_name} convergence plot saved: {output_path}")
        print(f"   Best values: Start={costs[0]:.4f} → Final={final_cost:.4f}")'''