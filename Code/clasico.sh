#!/bin/bash
#SBATCH -J clasico
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --ntasks-per-node=1
#SBATCH -t 100:00:00
#SBATCH --mem=128G
#SBATCH --partition=thinnode

# --- Optimización Entorno ---
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

# 1. Cargar Entorno Base
module purge
module load cesga/2022 miniconda3/22.11.1-1 gcc/system 
# Cargamos CUDA genérico solo para evitar errores de carga de librerías, aunque usemos CPU
module load cuda/12.0 2>/dev/null || module load cuda 2>/dev/null 

eval "$(conda shell.bash hook)"
conda activate protein_env_correct2

# ==========================================================
# 🛑 ZONA DE REPARACIÓN (Solo se ejecuta una vez)
# ==========================================================
echo "🛑 LIMPIANDO EL CAOS DE VERSIONES..."

# 1. Borramos TODO lo que pueda dar conflicto

# 2. Instalamos la "TRINIDAD ESTABLE" (Versiones 0.46)
# Estas versiones funcionan juntas perfectamente y no piden drivers raros.
echo "🔙 Instalando Qiskit 0.46 (Compatible con todo)..."

# Qiskit Core 0.46: La última versión que soporta el código antiguo Y el nuevo

# Qiskit Aer 0.13.3: Versión estable para CPU/GPU

# Qiskit Algorithms 0.3.0: Compatible con 0.46

# ==========================================================

echo "----------------------------------------"
echo "🔍 VERIFICACIÓN:"
# Esto debería imprimir versiones 0.46 / 0.13 / 0.3
python -c "import qiskit, qiskit_aer, qiskit_algorithms; print(f'Core: {qiskit.__version__} | Aer: {qiskit_aer.__version__}')" || exit 1
echo "✅ Entorno reparado."
echo "----------------------------------------"
python main_final.py \
 #   -L 2 \
 #   -R A,R,D,E,Q,G,L,M,F,P,S,T,W,Y,V,H \
 #   --membrane_mode wheel \
#    --wheel_phase_deg 90 \
#    --wheel_halfwidth_deg 90 \
#    --output_dir ../clasico_2 \
#    --solver classical \
# --- EJECUCIÓN DEL CÓDIGO ---
#python main_final.py \
 #   -L 4 \
  #  -R A,R,D,E,Q,G,L,M,F,P,S,T,W,Y,V,H \
  #  --membrane_mode wheel \
  #  --wheel_phase_deg 90 \
  #  --wheel_halfwidth_deg 90 \
  #  --output_dir ../clasico_4 \
#    --solver classical \

python main_final.py -L 6 -R A,R,D,E,Q,G,L,M,F,P,S,T,W,Y,V,H --membrane_mode wheel --wheel_phase_deg 90 --wheel_halfwidth_deg 90 --output_dir ../clasico_6 --solver classical 
echo "🚀 Iniciando L=8 en CPU (MPS)..."
#python main_final.py \
#    -L 8 \
#    -R A,R,D,E,Q,G,L,M,F,P,S,T,W,Y,V,H \
#    --membrane_mode wheel \
#    --wheel_phase_deg 90 \
#    --wheel_halfwidth_deg 90 \
#    --output_dir ../clasico_8 \
#    --solver classical \

