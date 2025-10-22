import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import List

def plot_optimization(costs: List[float]):
    """Plot optimization convergence"""
    
    plt.figure(figsize=(10, 6))
    plt.plot(costs, 'b-', linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('Energy')
    plt.title('Quantum Optimization Convergence')
    plt.grid(True, alpha=0.3)
    plt.show()

def plot_alpha_helix_wheel(sequence: str, membrane_mode: str = 'span', wheel_phase_deg: float = 0.0, wheel_halfwidth_deg: float = 40.0):
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
        plt.text(xs[i], ys[i], aa, ha='center', va='center', fontsize=16, weight='bold', color='white', zorder=4)
        # Residue index (1-based), slightly offset radially outward
        r_idx = radius + 0.12
        ang_i = angles[i]
        xi = r_idx * np.cos(ang_i)
        yi = r_idx * np.sin(ang_i)
        plt.text(xi, yi, f"{i+1}", ha='center', va='center', fontsize=12, color='black', zorder=5)

    # Connect residues in sequence order to show the helical path
    for i in range(len(sequence) - 1):
        plt.plot([xs[i], xs[i+1]], [ys[i], ys[i+1]], color='k', alpha=0.35, linewidth=1.5, zorder=2)
    # Draw circle
    circle = plt.Circle((0, 0), radius, color='k', fill=False, alpha=0.3)
    ax = plt.gca()
    ax.add_artist(circle)
    
    # Draw membrane orientation if wheel mode is active
    if membrane_mode == 'wheel':
        phase = np.deg2rad(wheel_phase_deg)
        halfw = np.deg2rad(wheel_halfwidth_deg)
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
        ax.text(xm, ym, 'Membrane (lipids)', ha='center', va='center', fontsize=12, color='#8B4513', weight='bold')
        # Water side (opposite)
        xa = 1.15 * radius * np.cos(mid_ang + np.pi)
        ya = 1.15 * radius * np.sin(mid_ang + np.pi)
        ax.text(xa, ya, 'Water', ha='center', va='center', fontsize=12, color='teal', weight='bold')

    ax.set_aspect('equal')
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.3, 1.3)
    ax.axis('off')
    plt.title('Alpha-Helix Wheel')
    plt.show()