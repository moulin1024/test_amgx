cmake_minimum_required(VERSION 3.20)
project(test_amgx LANGUAGES CUDA CXX)

# Find AMGX via local find module
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
find_package(AMGX REQUIRED)
find_package(MPI REQUIRED COMPONENTS CXX)

add_executable(test_amgx test_amgx.cu)
target_link_libraries(test_amgx PRIVATE AMGX::AMGX cuda)

add_executable(test_amgx_mpi test_amgx_mpi.cu)
target_link_libraries(test_amgx_mpi PRIVATE AMGX::AMGX MPI::MPI_CXX cuda)
