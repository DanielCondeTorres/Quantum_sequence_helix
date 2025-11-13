import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from typing import List, Tuple, Optional, Dict, Any
from collections import defaultdict
import gc
import os

# --- Project Dependencies ---
# We assume these modules are in your PYTHONPATH
try:
    from utils.general_utils import get_qubit_index
    from data_loaders.energy_matrix_loader import (
        _load_first_neighbors_matrix_file,
        _load_third_neighbors_matrix_file,
        _load_fourth_neighbors_matrix_file,
        _load_energy_matrix_file
    )
except ImportError:
    print("WARNING: Could not import 'utils' or 'data_loaders'.")
    print("This script will fail if they are not in the PYTHONPATH.")
    # Mock implementations so the script is self-contained for review.
    # YOU MUST REMOVE THIS if your data_loaders work.
    def _load_energy_matrix_file():
        print("Using MOCK _load_energy_matrix_file")
        symbols = list('ARNDCEQGHILKMFPSTWYV')
        matrix = np.random.rand(20, 20)
        return matrix, symbols
    def _load_first_neighbors_matrix_file():
        print("Using MOCK _load_first_neighbors_matrix_file")
        symbols = list('ARNDCEQGHILKMFPSTWYV')
        matrix = np.random.rand(20, 20)
        return matrix, symbols
# --- End of Dependencies ---


class EnergyCalculator:
    """
    Calculates the CLASSICAL energies of each Hamiltonian component
    for a sensitivity analysis.
    
    Based on HamiltonianBuilder, but without the quantum (Pauli/QUBO) part.
    """
    
    # We copy the initialization methods from HamiltonianBuilder
    def __init__(self, L: int, amino_acids: List[str], **kwargs):
        print("INITIALIZING CLASSICAL ENERGY CALCULATOR...")
        self.L = L
        self.amino_acids = amino_acids
        self.n_aa = len(amino_acids)
        self.kwargs = kwargs
        self.scale_factor = kwargs.get('scale_factor', 0.1)
        self.kwargs['wheel_phase_deg'] = kwargs.get('wheel_phase_deg', 0.0)
        self.kwargs['wheel_halfwidth_deg'] = kwargs.get('wheel_halfwidth_deg', 90.0)
        
        self._init_amino_acid_properties()
        self._load_matrices_from_files()
        
        # Map amino acids to indices for quick access
        self.aa_to_index = {aa: i for i, aa in enumerate(self.amino_acids)}
        
        # Component names (for the plots)
        # *** MODIFIED: Commented out terms per user request ***
        self.component_names = [
            'Environment',
            'Charge',
            'HydroMoment',
            'LocalPref',
            'Pairwise',
            'HelixPair',
            # 'AmphiSeg',      # <-- COMMENTED OUT
            # 'Electrostatic'  # <-- COMMENTED OUT
        ]
        print(f"Calculator ready for L={L}. Analyzing {len(self.component_names)} components.\n")

    def _init_amino_acid_properties(self):
        """Initialize basic amino acid properties (hydrophobicity, charge, polarity)."""
        # (Copied directly from your HamiltonianBuilder)
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
        # (This only needs to run once, but it's fine in init)
        print("\n" + "="*70)
        print("📁 LOADING MATRICES FROM FILES (for calculator)")
        print("="*70)
        
        mj_matrix_full, mj_symbols = _load_energy_matrix_file()
        aa_to_mj_idx = {aa: idx for idx, aa in enumerate(mj_symbols)}
        self.mj_matrix = np.zeros((self.n_aa, self.n_aa))
        for i, aa1 in enumerate(self.amino_acids):
            for j, aa2 in enumerate(self.amino_acids):
                if aa1 not in aa_to_mj_idx or aa2 not in aa_to_mj_idx:
                    raise ValueError(f"❌ Amino acid {aa1} or {aa2} not found in MJ matrix file!")
                idx1 = aa_to_mj_idx[aa1]
                idx2 = aa_to_mj_idx[aa2]
                self.mj_matrix[i, j] = mj_matrix_full[idx1, idx2]
        print("✅ MJ matrix mapped.")
        
        helix_matrix_full, helix_symbols = _load_first_neighbors_matrix_file()
        aa_to_helix_idx = {aa: idx for idx, aa in enumerate(helix_symbols)}
        self.helix_pairs = np.zeros((self.n_aa, self.n_aa))
        for i, aa1 in enumerate(self.amino_acids):
            for j, aa2 in enumerate(self.amino_acids):
                if aa1 not in aa_to_helix_idx or aa2 not in aa_to_helix_idx:
                    raise ValueError(f"❌ Amino acid {aa1} or {aa2} not found in helix propensity matrix file!")
                idx1 = aa_to_helix_idx[aa1]
                idx2 = aa_to_helix_idx[aa2]
                self.helix_pairs[i, j] = helix_matrix_full[idx1, idx2]
        print("✅ Helix propensity matrix mapped.")
        print("="*70 + "\n")

    def _get_membrane_environment(self, position: int):
        # (Copied directly from your HamiltonianBuilder)
        phase_deg = self.kwargs.get('wheel_phase_deg', 0.0)
        halfwidth_deg = self.kwargs.get('wheel_halfwidth_deg', 90.0)
        membrane_angle = (position * 100.0 + phase_deg) % 360.0
        if membrane_angle > 180.0: 
            membrane_angle -= 360.0
        faces_membrane = abs(membrane_angle) <= halfwidth_deg
        return "membrane" if faces_membrane else "water"

    # --- Classical Energy Functions (The H_i) ---
    # Note: The 'weight' (lambda) is assumed to be 1.0 to get the base energy.
    
    def _calculate_classical_env(self, seq_indices: List[int]) -> float:
        total_energy = 0.0
        weight = 1.0 # Assume lambda=1
        for i in range(self.L):
            environment = self._get_membrane_environment(i)
            α = seq_indices[i] # Index of the amino acid at pos i
            
            hydro = self.hydrophobic[α]
            polar = self.is_polar[α]
            charge = abs(self.charges[α])
            
            bonus = 0.0
            if environment == "membrane":
                if hydro > 0:
                    bonus = -weight * hydro * 2.0 * self.scale_factor
                elif polar or charge > 0:
                    penalty_strength = 3.0 if charge > 0 else 2.0
                    bonus = weight * penalty_strength * self.scale_factor
            else: # water
                if polar or charge > 0:
                    reward_strength = 2.5 if charge > 0 else 2.0
                    bonus = -weight * reward_strength * self.scale_factor
                elif hydro > 0:
                    bonus = weight * hydro * 2.0 * self.scale_factor
            
            total_energy += bonus
        return total_energy

    def _calculate_classical_charge(self, seq_indices: List[int]) -> float:
        total_energy = 0.0
        weight = 1.0 # Assume lambda=1
        membrane_charge = self.kwargs.get('membrane_charge', 'neg')
        membrane_charge_val = {'neu': 0.0, 'pos': 1.0, 'neg': -1.0}.get(membrane_charge.lower(), 0.0)
        if abs(membrane_charge_val) == 0:
            return 0.0

        for i in range(self.L):
            environment = self._get_membrane_environment(i)
            if environment == "membrane":
                α = seq_indices[i]
                charge = self.charges[α]
                energy = weight * charge * membrane_charge_val * self.scale_factor
                total_energy += energy
        return total_energy

    def _calculate_classical_mu(self, seq_indices: List[int]) -> float:
        if self.kwargs.get('membrane_mode') != 'wheel':
            return 0.0
        
        total_energy = 0.0
        weight = 1.0 # Assume lambda=1
        phase = np.deg2rad(self.kwargs.get('wheel_phase_deg', 0.0))
        
        for i in range(self.L):
            angle = (i * np.deg2rad(100.0) + phase) % (2 * np.pi)
            α = seq_indices[i]
            hydrophobicity = self.hydrophobic[α]
            energy = weight * hydrophobicity * np.cos(angle) * self.scale_factor
            total_energy += energy
        return total_energy

    def _calculate_classical_local(self, seq_indices: List[int]) -> float:
        total_energy = 0.0
        weight = 1.0 # Assume lambda=1
        helix_formers = {'A', 'L', 'E', 'K'}
        
        for i in range(self.L):
            α = seq_indices[i]
            aa = self.amino_acids[α]
            if aa in helix_formers:
                energy = -weight * self.scale_factor
                total_energy += energy
        return total_energy

    def _calculate_classical_pairwise(self, seq_indices: List[int]) -> float:
        total_energy = 0.0
        weight = 1.0 # Assume lambda=1
        max_dist = self.kwargs.get('max_interaction_dist', 4)
        
        for i in range(self.L):
            for j in range(i + 1, self.L):
                if abs(i - j) <= max_dist:
                    α = seq_indices[i]
                    β = seq_indices[j]
                    # NOTE: Your original code does not use scale_factor here
                    energy = weight * self.mj_matrix[α, β] 
                    total_energy += energy
        return total_energy

    def _calculate_classical_helix_pairs(self, seq_indices: List[int]) -> float:
        total_energy = 0.0
        weight = 1.0 # Assume lambda=1
        
        # This loop is safe: if L <= 4, range(self.L - 4) is empty
        for i in range(self.L - 4): 
            for j in range(i + 3, self.L):
                α = seq_indices[i]
                β = seq_indices[j]
                # NOTE: Your original code does not use scale_factor here
                energy = weight * self.helix_pairs[α, β]
                total_energy += energy
        return total_energy

    def _calculate_classical_segregation(self, seq_indices: List[int]) -> float:
        """
        *** THIS FUNCTION IS NOT CURRENTLY BEING CALLED (COMMENTED OUT) ***
        USER-REQUESTED COMMENT: This term encourages amphipathic segregation.
        ... (rest of comment) ...
        """
        if self.kwargs.get('membrane_mode') != 'wheel':
            return 0.0
        # ... (rest of implementation) ...
        return 0.0 # Bypassed

    def _calculate_classical_electrostatic(self, seq_indices: List[int]) -> float:
        """
        *** THIS FUNCTION IS NOT CURRENTLY BEING CALLED (COMMENTED OUT) ***
        USER-REQUESTED COMMENT: This term models simple electrostatic (Coulomb) interactions.
        ... (rest of comment) ...
        """
        total_energy = 0.0
        weight = 1.0 # Assume lambda=1
        # ... (rest of implementation) ...
        return 0.0 # Bypassed

    def get_all_classical_energies(self, seq_indices: List[int]) -> Dict[str, float]:
        """Calculates all base energies for a single sequence."""
        if len(seq_indices) != self.L:
            raise ValueError(f"Sequence has length {len(seq_indices)}, expected {self.L}")
            
        energies = {}
        energies['Environment'] = self._calculate_classical_env(seq_indices)
        energies['Charge'] = self._calculate_classical_charge(seq_indices)
        energies['HydroMoment'] = self._calculate_classical_mu(seq_indices)
        energies['LocalPref'] = self._calculate_classical_local(seq_indices)
        energies['Pairwise'] = self._calculate_classical_pairwise(seq_indices)
        energies['HelixPair'] = self._calculate_classical_helix_pairs(seq_indices)
        
        # --- Terms Commented Out Per User Request ---
        # if 'AmphiSeg' in self.component_names:
        #     energies['AmphiSeg'] = self._calculate_classical_segregation(seq_indices)
        # if 'Electrostatic' in self.component_names:
        #     energies['Electrostatic'] = self._calculate_classical_electrostatic(seq_indices)
        
        # Ensure the order matches self.component_names
        return {name: energies[name] for name in self.component_names}

    def generate_random_sequences(self, M_sequences: int) -> np.ndarray:
        """Generates M random sequences of amino acid indices."""
        print(f"Generating {M_sequences} random sequences of length {self.L}...")
        return np.random.randint(0, self.n_aa, size=(M_sequences, self.L))

    def build_energy_matrix(self, M_sequences: int) -> pd.DataFrame:
        """Builds the (M_sequences, N_components) Energy Matrix."""
        sequences = self.generate_random_sequences(M_sequences)
        
        energy_data = []
        print(f"Calculating energies for {M_sequences} sequences...")
        for i, seq in enumerate(sequences):
            if (i+1) % (M_sequences // 20 or 1) == 0: # Print more frequent updates
                print(f"  Processing sequence {i+1}/{M_sequences}...")
            energies = self.get_all_classical_energies(seq)
            energy_data.append(energies)
            
        df = pd.DataFrame(energy_data, columns=self.component_names)
        print("Energy matrix built.")
        return df

# --- Analysis and Plotting Functions ---

# Define EVEN LARGER font sizes per user request
TITLE_FONTSIZE = 24
LABEL_FONTSIZE = 20
TICK_FONTSIZE = 14
ANNOT_FONTSIZE = 12 # Font size for annotations inside heatmap

def run_variance_analysis(energy_df: pd.DataFrame, output_dir: str, length: int):
    """
    Technique 1: Analyze the Variance of each H_i component.
    Components with high variance are more 'sensitive' to sequence changes.
    """
    print("\n" + "="*70)
    print("📈 TECHNIQUE 1: VARIANCE ANALYSIS")
    print("="*70)
    
    variances = energy_df.var()
    variances_log = np.log10(variances.abs() + 1e-9) # Use log for better visualization
    
    print("Variance (linear scale):")
    print(variances.sort_values(ascending=False))
    
    plt.figure(figsize=(14, 9)) # Increased figure size for labels
    sns.barplot(x=variances_log.index, y=variances_log.values, palette='viridis')
    plt.title(f'Technique 1: Component Variance (L={length})', fontsize=TITLE_FONTSIZE)
    plt.ylabel('Log10(Energy Variance)', fontsize=LABEL_FONTSIZE)
    plt.xlabel('Hamiltonian Component ($H_i$)', fontsize=LABEL_FONTSIZE)
    plt.xticks(rotation=45, ha='right', fontsize=TICK_FONTSIZE)
    plt.yticks(fontsize=TICK_FONTSIZE)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    filename = os.path.join(output_dir, f'technique_1_variance_L{length}.png')
    plt.savefig(filename)
    print(f"✅ Variance plot saved to: {filename}")
    plt.close()

def run_pca_analysis(energy_df: pd.DataFrame, output_dir: str, length: int):
    """
    Technique 3: Principal Component Analysis (PCA).
    Finds the 'optimal combination' of H_i that maximizes total variance.
    """
    print("\n" + "="*70)
    print("🔬 TECHNIQUE 3: PCA ANALYSIS (Variance Maximization)")
    print("="*70)
    
    # 1. Standardizing data is CRUCIAL for PCA
    # Remove columns with zero variance *before* scaling
    non_zero_var_cols = energy_df.columns[energy_df.var().abs() > 1e-9]
    if len(non_zero_var_cols) < len(energy_df.columns):
        removed_cols = set(energy_df.columns) - set(non_zero_var_cols)
        print(f"Warning: Removing {len(removed_cols)} columns with zero variance: {removed_cols}")
    
    E_filtered = energy_df[non_zero_var_cols]
    
    # Handle case where no columns have variance (e.g., L=2 and only Pairwise terms)
    if E_filtered.empty:
        print("Error: No components with non-zero variance. Skipping PCA.")
        return

    scaler = StandardScaler()
    E_scaled = scaler.fit_transform(E_filtered)
    
    pca = PCA()
    E_pca = pca.fit_transform(E_scaled)
    
    # --- Plot 1: Explained Variance (Scree Plot) ---
    explained_variance_ratio = pca.explained_variance_ratio_
    
    plt.figure(figsize=(14, 9)) # Increased figure size
    sns.barplot(x=list(range(len(explained_variance_ratio))), 
                y=explained_variance_ratio, 
                palette='mako')
    plt.title(f'Technique 3: Explained Variance by PC (L={length})', fontsize=TITLE_FONTSIZE)
    plt.ylabel('Explained Variance Ratio', fontsize=LABEL_FONTSIZE)
    plt.xlabel('Principal Component (PC)', fontsize=LABEL_FONTSIZE)
    plt.xticks(fontsize=TICK_FONTSIZE)
    plt.yticks(fontsize=TICK_FONTSIZE)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    filename = os.path.join(output_dir, f'technique_3_pca_explained_variance_L{length}.png')
    plt.savefig(filename)
    print(f"✅ Explained Variance plot saved to: {filename}")
    plt.close()

    # --- Plot 2: PC1 Loadings (The most important) ---
    # pc1_loadings tells us the 'recipe' for maximum sensitivity
    pc1_loadings = pca.components_[0]
    pc1_df = pd.DataFrame({
        'Component': E_filtered.columns,
        'PC1 Loading (Weight)': pc1_loadings
    }).sort_values(by='PC1 Loading (Weight)', ascending=False)
    
    print("\nLoadings for Principal Component 1 (PC1):")
    print(pc1_df)
    
    plt.figure(figsize=(14, 9)) # Increased figure size
    sns.barplot(x='PC1 Loading (Weight)', y='Component', data=pc1_df, palette='coolwarm')
    plt.title(f'Technique 3: PC1 Loadings (Optimal Mix, L={length})', fontsize=TITLE_FONTSIZE)
    plt.xlabel('Weight in PC1 Combination', fontsize=LABEL_FONTSIZE)
    plt.ylabel('Hamiltonian Component ($H_i$)', fontsize=LABEL_FONTSIZE)
    plt.xticks(fontsize=TICK_FONTSIZE)
    plt.yticks(fontsize=TICK_FONTSIZE)
    plt.axvline(0, color='black', linewidth=0.8, linestyle='--')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    filename = os.path.join(output_dir, f'technique_3_pca_pc1_loadings_L{length}.png')
    plt.savefig(filename)
    print(f"✅ PC1 Loadings plot saved to: {filename}")
    plt.close()

    # --- Plot 3: Correlation Matrix (Covariance) ---
    corr_matrix = E_filtered.corr()
    
    plt.figure(figsize=(14, 12)) # Increased figure size
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0, 
                annot_kws={"size": ANNOT_FONTSIZE}) # Annot size
    plt.title(f'Correlation Matrix between $H_i$ (L={length})', fontsize=TITLE_FONTSIZE)
    plt.xticks(rotation=45, ha='right', fontsize=TICK_FONTSIZE)
    plt.yticks(rotation=0, fontsize=TICK_FONTSIZE)
    plt.tight_layout()
    
    filename = os.path.join(output_dir, f'technique_3_correlation_heatmap_L{length}.png')
    plt.savefig(filename)
    print(f"✅ Correlation Heatmap saved to: {filename}")
    plt.close()


# --- MAIN ENTRY POINT ---

if __name__ == "__main__":
    
    # --- Analysis Configuration ---
    # *** MODIFIED: Set list of lengths to test ***
    LENGTHS_TO_TEST = [2, 4, 6, 8, 10, 12] 
    AMINO_ACIDS = list('ARNDCEQGHILKMFPSTWYV') # The 20 amino acids
    M_SEQUENCES = 500000       # Number of random sequences to test (PER LENGTH)
    BASE_OUTPUT_DIR = "sensitivity_analysis_by_length"
    
    # Parameters (kwargs) for the Hamiltonian (you can adjust these)
    # These affect the H_i (e.g., which positions are 'membrane')
    KWARGS = {
        'scale_factor': 0.1,
        'membrane_mode': 'wheel',
        'wheel_phase_deg': 0.0,
        'wheel_halfwidth_deg': 90.0,
        'membrane_charge': 'neg',
        'max_interaction_dist': 4,
    }
    # --- End of Configuration ---
    
    # Create the main base directory
    if not os.path.exists(BASE_OUTPUT_DIR):
        os.makedirs(BASE_OUTPUT_DIR)
        
    print(f"Starting analysis for lengths: {LENGTHS_TO_TEST}")
    print(f"Sequences per length: {M_SEQUENCES}")
    print("============================================================\n")

    # *** MODIFIED: Loop over all requested lengths ***
    for length in LENGTHS_TO_TEST:
        
        print(f"\n" + "="*70)
        print(f"🚀 STARTING ANALYSIS FOR LENGTH L = {length}")
        print("="*70)
        
        # 1. Set up a unique output directory for this length
        current_output_dir = os.path.join(BASE_OUTPUT_DIR, f"analysis_L_{length}")
        if not os.path.exists(current_output_dir):
            os.makedirs(current_output_dir)
            
        # 2. Initialize the energy calculator FOR THIS LENGTH
        calculator = EnergyCalculator(L=length, amino_acids=AMINO_ACIDS, **KWARGS)
        
        # 3. Build the (M x N) energy matrix
        energy_df = calculator.build_energy_matrix(M_SEQUENCES)
        
        print(f"\n--- Preview of Energy Matrix (L={length}) ---")
        print(energy_df.head())
        print("-------------------------------------------------")
        
        # 4. Run Variance Analysis (Technique 1)
        run_variance_analysis(energy_df, current_output_dir, length)
        
        # 5. Run PCA Analysis (Technique 3)
        run_pca_analysis(energy_df, current_output_dir, length)
        
        print(f"\n🎉 ANALYSIS FOR L = {length} COMPLETE.")
        print(f"Results saved to folder: '{current_output_dir}'")
        print("="*70 + "\n")
        
        # Clean up memory before next loop
        del calculator, energy_df
        gc.collect()

    print("\n" + "="*70)
    print("🎉 ALL ANALYSES COMPLETE.")
    print(f"All results are in the base folder: '{BASE_OUTPUT_DIR}'")
    print("="*70)
