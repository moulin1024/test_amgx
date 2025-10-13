cmake_minimum_required(VERSION 3.20)
project(test_amgx LANGUAGES CUDA CXX)

# Let user specify AMGX_ROOT manually
set(AMGX_ROOT "" CACHE PATH "Path to AMGX installation root (containing include/ and lib/)")

if(NOT AMGX_ROOT)
    message(FATAL_ERROR "Please specify AMGX_ROOT, e.g. cmake -DAMGX_ROOT=/opt/amgx ..")
endif()

set(AMGX_INCLUDE_DIR ${AMGX_ROOT}/include)
set(AMGX_LIBRARY ${AMGX_ROOT}/lib/libamgxsh.so)

add_executable(test_amgx test_amgx.cu)
target_include_directories(test_amgx PRIVATE ${AMGX_INCLUDE_DIR})
target_link_libraries(test_amgx PRIVATE ${AMGX_LIBRARY} cuda)

message(STATUS "AMGX include dir: ${AMGX_INCLUDE_DIR}")
message(STATUS "AMGX library: ${AMGX_LIBRARY}")
