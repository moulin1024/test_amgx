#!/bin/bash -l
#
# Submit with:
#   sbatch submit_mpi.sh
# Override at submit time, for example:
#   sbatch --ntasks=4 --gres=gpu:a100:4 submit_mpi.sh
#
#SBATCH -J test_amgx_mpi
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --partition=gpudev
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=18
#SBATCH --mem=64G
#SBATCH --time=00:15:00
#SBATCH -o test_amgx_mpi.%j.out
#SBATCH -e test_amgx_mpi.%j.err

set -euo pipefail

module purge
module load gcc/13
module load nvhpcsdk/25
module load cuda/12.6-nvhpcsdk_25
module load openmpi/5.0
module load openmpi_gpu/5.0

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${BUILD_DIR:-${SCRIPT_DIR}/build}"
BIN="${BUILD_DIR}/test_amgx_mpi"
DATA_DIR="${DATA_DIR:-${SCRIPT_DIR}/circular}"
AMGX_CFG="${AMGX_CFG:-${SCRIPT_DIR}/amgx_config.json}"

if [[ ! -x "${BIN}" ]]; then
  echo "Executable not found: ${BIN}"
  echo "Build first, e.g.: ./build.sh"
  exit 1
fi

if [[ ! -f "${AMGX_CFG}" ]]; then
  echo "AMGX config not found: ${AMGX_CFG}"
  exit 1
fi

echo "Job ${SLURM_JOB_ID}: running test_amgx_mpi"
echo "  ntasks      = ${SLURM_NTASKS}"
echo "  cpus/task   = ${SLURM_CPUS_PER_TASK}"
echo "  gpus        = ${SLURM_GPUS:-unset}"
echo "  data dir    = ${DATA_DIR}"
echo "  amgx config = ${AMGX_CFG}"
echo "  executable  = ${BIN}"

# Rank/GPU visibility diagnostics
srun --gpu-bind=single:1 bash -lc '
  echo "[rank ${SLURM_PROCID}] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}";
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi -L || true
  fi
'

# Run mini-app (each rank starts from full input; solver handles distributed partitioning)
srun --gpu-bind=single:1 "${BIN}" "${DATA_DIR}" "${AMGX_CFG}"

