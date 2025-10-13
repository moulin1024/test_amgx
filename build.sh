module purge
module load gcc/13
module load nvhpcsdk/25
module load cuda/12.9-nvhpcsdk_25
module load openmpi/5.0
module load openmpi_gpu/5.0
module load mkl/2024.0     # Intel Math Kernel Library for BLAS and LAPACK

rm -rf build
mkdir build
cd build

cmake .. -DAMGX_ROOT=/u/limo/amgx
# cmake --build build -j
make 
cd ..

./build/test_amgx SOLVER_CONFIG.json iter
# export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
# ./build/sanity_amgx