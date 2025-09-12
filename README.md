# Quantum Sequence Helix

🧬 **Quantum Protein Design with QAOA** 🧬  
This repository implements a **quantum approach to protein sequence design**, formulating it as a **Quadratic Unconstrained Binary Optimization (QUBO)** problem.  
It supports both **PennyLane** and **Qiskit** backends, allowing experimentation with **variational quantum algorithms** (QAOA, VQE) for peptide and helix sequence optimization.

---

## 🚀 Features

- **Quantum Protein Design**  
  - Encodes amino acid sequences into binary strings with `log2(N)` qubits per position  
  - Builds Hamiltonians including:
    - Local amino acid preferences  
    - Pairwise interaction terms (Miyazawa–Jernigan)  
    - Helix pair propensity terms  
    - Hydrophobic moment and environment contributions  
    - Membrane interaction terms  

- **Backends**  
  - [PennyLane](https://pennylane.ai) (default)  
  - [Qiskit](https://qiskit.org) (optional, if installed)  

- **Optimization**  
  - QAOA with warm-starts and layered parameter initialization  
  - Classical brute-force solver for validation  
  - Convergence tracking and energy analysis  

- **Visualization**  
  - Optimization convergence plots  
  - Alpha-helix wheel diagrams with membrane/water partition  

---

## 📦 Installation

Clone the repository and create the environment using **conda**:

```bash
git clone https://github.com/DanielCondeTorres/Quantum_sequence_helix.git
cd Quantum_sequence_helix
conda env create -f environment.yml
conda activate quantum_protein 


## Usage

```
python main.py -L 6 -R V,Q,N,S \
    --lambda_pairwise 0.5 \
    --lambda_helix_pairs 0.5 \
    --lambda_env 5.0 \
    --lambda_local 0.2 \
    --membrane_mode wheel \
    --wheel_phase_deg 0 \
    --wheel_halfwidth_deg 90
```