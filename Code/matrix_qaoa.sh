#!/bin/bash
#SBATCH -J matrix_gpu    # Nombre del trabajo
#SBATCH -n 1
#SBATCH -c 32               # 32 Cores de CPU (para pre-procesamiento rápido)
#SBATCH --ntasks-per-node=1
#SBATCH -t 100:00:00         # 100h es mucho, QAOA en GPU debería tardar menos
#SBATCH --mem=128G          # 128GB está perfecto para 32 qubits
#SBATCH --gres=gpu:a100:1   # Solicita 1 GPU A100

# --- Optimización de Entorno ---
# Asegura que Python aproveche los 32 cores para tareas clásicas
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_PROC_BIND=true
export OMP_PLACES=threads

# Cargar módulos
module purge
module load cesga/2022 miniconda3/22.11.1-1 gcc/system cuda/11.8.0
# Nota: Cargar CUDA es vital para que Qiskit Aer detecte la GPU

# Activar entorno
conda activate /mnt/netapp2/Store_uni/home/usc/cq/dct/.envi/quantum_protein_design2


# Instalar numba solo si no está (ahorra tiempo en re-lanzamientos)
python -c "import numba" 2>/dev/null || pip install numba

echo "🚀 Iniciando simulación de 32 Qubits en A100..."
echo "Backend configurado: GPU (Single Precision)"



python main_final.py -L 8 -R A,R,D,E,Q,G,L,M,F,P,S,T,W,Y,V,H --membrane_mode wheel --wheel_phase_deg 90 --wheel_halfwidth_deg 90   --output_dir ../qaoa_8_angel_GPU  --solver qaoa --backend qiskit




python main_final.py -L 10 -R A,R,D,E,Q,G,L,M,F,P,S,T,W,Y,V,H --membrane_mode wheel --wheel_phase_deg 90 --wheel_halfwidth_deg 90   --output_dir ../qaoa_10_angel_GPU  --solver qaoa --backend qiskit




python main_final.py -L 12 -R A,R,D,E,Q,G,L,M,F,P,S,T,W,Y,V,H --membrane_mode wheel --wheel_phase_deg 90 --wheel_halfwidth_deg 90   --output_dir ../qaoa_12_angel_GPU  --solver qaoa --backend qiskit
