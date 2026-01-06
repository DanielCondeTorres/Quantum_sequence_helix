import os
import matplotlib
# CRÍTICO: 'Agg' permite guardar gráficas sin pantalla (evita errores en clúster)
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

class ProteinPlotter:
    """Clase para manejar todas las visualizaciones del diseño de proteínas"""
    
    def __init__(self, output_dir='output'):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    # --- FUNCIÓN DE OPTIMIZACIÓN (Tu versión favorita con el punto rojo en el mínimo) ---
    def plot_optimization(self, costs: list, solver_name: str, tick_size: int = 14):
        """
        Plots the optimization convergence (costs vs iterations).
        Highlights the MINIMUM cost found.
        """
        if not costs:
            print(f"⚠️ Not enough data to plot {solver_name} (len=0)")
            return
        
        iterations = np.arange(len(costs))
        plt.figure(figsize=(10, 6))
        
        # Main line
        plt.plot(iterations, costs, 'b-', linewidth=2, marker='o', markersize=4, alpha=0.8, label='Cost')
        
        # Highlight the BEST (minimum) cost
        best_idx = np.argmin(costs)
        best_cost = costs[best_idx]
        plt.scatter(best_idx, best_cost, color='red', s=250, zorder=5,
                    edgecolor='darkred', linewidth=3, label=f'Minimum energy')
        
        plt.tick_params(axis='both', which='major', labelsize=tick_size)
        plt.tick_params(axis='both', which='minor', labelsize=tick_size-2)
        
        # Styling
        plt.xlabel('Iterations', fontsize=15, fontweight='bold')
        plt.ylabel('Energy / Cost', fontsize=15, fontweight='bold')
        
        title = f'{solver_name} Convergence'
        if solver_name.upper() == "QAOA":
            title = "QAOA Energy Convergence"
        plt.title(title, fontsize=16, fontweight='bold', pad=15)
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.legend(loc='best')
        plt.tight_layout()
        
        # Save
        filename = f'{solver_name.lower()}_optimization_convergence.png'
        output_path = os.path.join(self.output_dir, filename)
        try:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"✅ {solver_name} convergence plot saved: {output_path}")
        except Exception as e:
            print(f"⚠️ Error saving optimization plot: {e}")
        finally:
            plt.close()

    # Alias por si acaso quantum_engine llama a plot_convergence
    def plot_convergence(self, costs, title="QAOA"):
        self.plot_optimization(costs, solver_name=title)

    # --- FUNCIÓN DE HELICAL WHEEL (Restaurada con tus parámetros exactos) ---
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
            circle = plt.Circle((0, 0), radius, color='k', fill=False, alpha=1)
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
    

    # --- OTRAS FUNCIONES (Probabilidades, Circuitos) ---
    def plot_prob_with_sequences(self, probs: np.ndarray, decoder_fn, n_qubits: int, 
                                solver_name: str = "QAOA", top_k: int = 20):
        
        # Adaptador para cuando probs es un diccionario (nuestro caso MPS)
        if isinstance(probs, dict):
            # Convertir dict a formato compatible
            sorted_items = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:top_k]
            sequences = [decoder_fn(bs) for bs, _ in sorted_items]
            sorted_probs = [p for _, p in sorted_items]
        else:
            # Lógica original para arrays
            if len(probs) == 0 or np.all(probs == 0): return
            top_k = min(top_k, len(probs))
            sorted_indices = np.argsort(-probs)[:top_k]
            sorted_probs = probs[sorted_indices]
            sequences = [decoder_fn(format(idx, f'0{n_qubits}b')) for idx in sorted_indices]

        plt.figure(figsize=(14, 6))
        bars = plt.bar(range(len(sequences)), sorted_probs, color='steelblue', alpha=0.8, edgecolor='navy')
        if len(bars) > 0:
            bars[0].set_color('gold')
            bars[0].set_edgecolor('darkorange')
        
        plt.xlabel(f'Amino Acid Sequences (Top {len(sequences)})', fontsize=16, fontweight='bold')
        plt.ylabel('Probability', fontsize=16, fontweight='bold')
        plt.xticks(range(len(sequences)), sequences, rotation=90, fontsize=14)
        plt.tight_layout()
        
        output_path = os.path.join(self.output_dir, f'{solver_name.lower()}_probability_plot.png')
        try:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"✓ Probability plot saved as {output_path}")
        except: pass
        finally: plt.close()

    def save_qiskit_circuit(self, circuit, filename: str):
        """Save Qiskit circuit to a PNG/TXT file."""
        output_path = os.path.join(self.output_dir, filename)
        try:
            from qiskit.visualization import circuit_drawer
            try:
                circuit_drawer(circuit, output='mpl', filename=output_path)
            except:
                # Fallback a texto si falla mpl
                txt_path = output_path.replace('.png', '.txt')
                with open(txt_path, 'w') as f:
                    f.write(str(circuit_drawer(circuit, output='text')))
        except: pass
