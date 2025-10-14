#!/bin/bash -l
# Job name
#SBATCH -J PAccX-utest
# number of nodes
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:10:00
#SBATCH --mem=64G
#SBATCH --partition=gpudev
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=18

module purge
module load gcc/13
module load nvhpcsdk/25
module load cuda/12.6-nvhpcsdk_25
module load openmpi/5.0
module load openmpi_gpu/5.0


# srun nsys profile -o amgx_jacobi.%q{SLURM_PROCID} -t cuda,mpi ./amgx_mpi_capi -m circular/matrix.mtx -c ./PBICGSTAB_CLASSICAL_JACOBI.json 
