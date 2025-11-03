rm -rf build
mkdir build
cd build

cmake .. -DAMGX_ROOT=/home/moulin/Workspace/AMGX -DCMAKE_CUDA_ARCHITECTURES=80
make 
cd ..

./build/test_amgx circular