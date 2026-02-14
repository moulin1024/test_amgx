cmake_minimum_required(VERSION 3.20)
project(test_amgx LANGUAGES C CUDA CXX)

# Find AMGX via local find module
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
find_package(AMGX REQUIRED)
find_package(MPI REQUIRED COMPONENTS CXX)
find_package(CUDAToolkit REQUIRED)

add_executable(test_amgx test_amgx.cu)
target_link_libraries(test_amgx PRIVATE AMGX::AMGX CUDA::cudart)

add_executable(test_amgx_mpi test_amgx_mpi.cu)
target_link_libraries(test_amgx_mpi PRIVATE AMGX::AMGX MPI::MPI_CXX CUDA::cudart)

# Sanity-check target: compile AMGX MPI example code path as-is.
add_executable(amgx_mpi_capi_sanity amgx_mpi_capi_sanity.c)
target_link_libraries(amgx_mpi_capi_sanity PRIVATE AMGX::AMGX MPI::MPI_CXX CUDA::cudart)
