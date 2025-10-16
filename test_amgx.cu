#include <iostream>
#include <fstream>
#include <stdexcept>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <string>
#include <amgx_c.h>
#include <cuda_runtime.h>
#include <chrono>

// ========================= Kernels =========================

// 1) Extract diagonal and its inverse: dinv[i] = 1/a_ii if |a_ii|>eps, else 1.0 (no scaling)
__global__ void extract_diagonal_inverse(
    int n,
    const int* __restrict__ row_ptr,
    const int* __restrict__ col_ind,
    const double* __restrict__ val,
    double* __restrict__ dinv,
    double eps /* e.g., 1e-30 to avoid 1/0 */
){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    double aii = 0.0;
    int start = row_ptr[i];
    int end   = row_ptr[i + 1];
    for (int jj = start; jj < end; ++jj) {
        if (col_ind[jj] == i) { aii = val[jj]; break; }
    }
    dinv[i] = (fabs(aii) > eps) ? (1.0 / aii) : 1.0; // 1.0 keeps row unscaled
}

// 2) Left-scale matrix rows: A <- D^{-1} A   (row i multiplied by dinv[i])
__global__ void scale_matrix_rows_by_dinv(
    int n,
    const int* __restrict__ row_ptr,
    double* __restrict__ val,
    const double* __restrict__ dinv
){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    double s = dinv[i];
    int start = row_ptr[i];
    int end   = row_ptr[i + 1];
    for (int jj = start; jj < end; ++jj) {
        val[jj] *= s;
    }
}

// 3) Left-scale rhs: b <- D^{-1} b
__global__ void scale_rhs_by_dinv(
    int n,
    double* __restrict__ rhs,
    const double* __restrict__ dinv
){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    rhs[i] *= dinv[i];
}

// ---------- small helpers ----------
template <typename T>
T* load_bin(const std::string& path, size_t& out_count) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open " + path);

    f.seekg(0, std::ios::end);
    std::streamsize bytes = f.tellg();
    f.seekg(0, std::ios::beg);

    if (bytes < 0 || (static_cast<size_t>(bytes) % sizeof(T)) != 0)
        throw std::runtime_error("File size of " + path + " is not a multiple of element size");

    out_count = static_cast<size_t>(bytes) / sizeof(T);
    T* data = new T[out_count];
    if (!f.read(reinterpret_cast<char*>(data), bytes)) {
        delete[] data;
        throw std::runtime_error("Failed reading " + path);
    }
    return data;
}

int main(int argc, char** argv) {
    const std::string data_dir = (argc > 1) ? argv[1] : "data";

    // ---- Init & callbacks ----
    AMGX_initialize();
    AMGX_install_signal_handler();

    // ---- Config / resources / handles ----
    AMGX_config_handle cfg = nullptr;
    AMGX_config_create_from_file(&cfg, "amgx_config.json");
    AMGX_config_add_parameters(&cfg, "exception_handling=1");

    AMGX_resources_handle rsrc = nullptr;
    AMGX_resources_create_simple(&rsrc, cfg);

    // Double values, 32-bit indices
    const AMGX_Mode mode = AMGX_mode_dDDI;

    AMGX_matrix_handle A = nullptr;
    AMGX_vector_handle b = nullptr, x = nullptr;
    AMGX_solver_handle solver = nullptr;

    AMGX_matrix_create(&A, rsrc, mode);
    AMGX_vector_create(&b, rsrc, mode);
    AMGX_vector_create(&x, rsrc, mode);
    AMGX_solver_create(&solver, rsrc, mode, cfg);

    // ---- Load RHS & CSR from binaries (host pointers) ----
    double*  h_rhs     = nullptr;
    int*     h_row_ptr = nullptr;
    int*     h_col_ind = nullptr;
    double*  h_val     = nullptr;

    size_t rhs_n = 0, row_ptr_n = 0, col_ind_n = 0, val_n = 0;

    h_rhs     = load_bin<double>(data_dir+"/rhs.bin",     rhs_n);
    h_row_ptr = load_bin<int>   (data_dir+"/row_ptr.bin", row_ptr_n);
    h_col_ind = load_bin<int>   (data_dir+"/col_ind.bin", col_ind_n);
    h_val     = load_bin<double>(data_dir+"/val.bin",     val_n);

    const int n   = static_cast<int>(row_ptr_n - 1);
    const int nnz = static_cast<int>(val_n);

    // ---- Create device mirrors of CSR (for the experiment) ----
    int    *d_row_ptr = nullptr, *d_col_ind = nullptr;
    double *d_val     = nullptr, *d_rhs = nullptr;
    double *d_val_copy     = nullptr, *d_rhs_copy = nullptr;
    double *d_dinv    = nullptr;

    cudaMalloc((void**)&d_row_ptr, row_ptr_n * sizeof(int));
    cudaMalloc((void**)&d_col_ind, col_ind_n * sizeof(int));
    cudaMalloc((void**)&d_val,     val_n     * sizeof(double));
    cudaMalloc((void**)&d_rhs,     rhs_n     * sizeof(double));
    cudaMalloc((void**)&d_dinv,    n         * sizeof(double));
    cudaMalloc((void**)&d_val_copy,     val_n     * sizeof(double));
    cudaMalloc((void**)&d_rhs_copy,     rhs_n     * sizeof(double));

    cudaMemcpy(d_row_ptr, h_row_ptr, row_ptr_n * sizeof(int),    cudaMemcpyHostToDevice);
    cudaMemcpy(d_col_ind, h_col_ind, col_ind_n * sizeof(int),    cudaMemcpyHostToDevice);
    cudaMemcpy(d_val,     h_val,     val_n    * sizeof(double),  cudaMemcpyHostToDevice);
    cudaMemcpy(d_rhs,     h_rhs,     rhs_n    * sizeof(double),  cudaMemcpyHostToDevice);
    cudaMemcpy(d_val_copy,h_val,     val_n    * sizeof(double),  cudaMemcpyHostToDevice);
    cudaMemcpy(d_rhs_copy,h_rhs,     rhs_n    * sizeof(double),  cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks  = (n + threads - 1) / threads;

    // ---- Jacobi left scaling via 3 kernels on (d_val, d_rhs) ----
    extract_diagonal_inverse<<<blocks, threads>>>(n, d_row_ptr, d_col_ind, d_val, d_dinv, 1e-30);
    cudaGetLastError();
    cudaDeviceSynchronize();

    scale_matrix_rows_by_dinv<<<blocks, threads>>>(n, d_row_ptr, d_val, d_dinv);
    cudaGetLastError();
    cudaDeviceSynchronize();

    scale_rhs_by_dinv<<<blocks, threads>>>(n, d_rhs, d_dinv);
    cudaGetLastError();
    cudaDeviceSynchronize();

    // ---- Upload matrix (with DEVICE pointers on purpose) ----
    AMGX_matrix_upload_all(
        A, n, nnz, 1, 1,
        d_row_ptr,     // device pointer on purpose
        d_col_ind,     // device pointer on purpose
        d_val,         // device pointer on purpose (already scaled)
        nullptr
    );

    // ---- Create & upload vectors ----
    AMGX_vector_bind(b, A);
    AMGX_vector_bind(x, A);
    AMGX_vector_upload(b, n, 1, d_rhs); // device pointer (scaled)
    AMGX_vector_set_zero(x, n, 1);

    // ---- Initial residual norm
    double* t_norm = new double[1];
    std::fill(t_norm, t_norm + 1, 0.0);
    AMGX_solver_calculate_residual_norm(solver, A, b, x, t_norm);

    // ---- Warmup Setup & Solve ----
    AMGX_solver_setup(solver, A);
    AMGX_solver_solve(solver, b, x);

    // ---- Status & iterations ----
    AMGX_SOLVE_STATUS status;
    AMGX_solver_get_status(solver, &status);

    AMGX_vector_set_zero(x, n, 1);

    cudaDeviceSynchronize();
    auto start_setup = std::chrono::high_resolution_clock::now();

    // Recompute scaling on the *copy* (original values) then replace coefficients & rhs
    extract_diagonal_inverse<<<blocks, threads>>>(n, d_row_ptr, d_col_ind, d_val_copy, d_dinv, 1e-30);
    cudaGetLastError();
    cudaDeviceSynchronize();

    scale_matrix_rows_by_dinv<<<blocks, threads>>>(n, d_row_ptr, d_val_copy, d_dinv);
    cudaGetLastError();
    cudaDeviceSynchronize();

    scale_rhs_by_dinv<<<blocks, threads>>>(n, d_rhs_copy, d_dinv);
    cudaGetLastError();
    cudaDeviceSynchronize();

    AMGX_matrix_replace_coefficients(A, n, nnz, d_val_copy, NULL);
    AMGX_vector_upload(b, n, 1, d_rhs_copy);
    AMGX_solver_resetup(solver, A);

    cudaDeviceSynchronize();
    auto end_setup = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_setup = end_setup - start_setup;

    auto start_solve = std::chrono::high_resolution_clock::now();
    AMGX_solver_solve(solver, b, x);
    cudaDeviceSynchronize();
    auto end_solve = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_solve = end_solve - start_solve;

    std::cout << "Setup time: " << elapsed_setup.count() << " s\n";
    std::cout << "Solve time: " << elapsed_solve.count() << " s\n";
    std::cout << "Total time: " << elapsed_solve.count() + elapsed_setup.count() << " s\n";

    // ---- Cleanup ----
    delete[] t_norm;
    delete[] h_rhs;
    delete[] h_row_ptr;
    delete[] h_col_ind;
    delete[] h_val;

    cudaFree(d_row_ptr);
    cudaFree(d_col_ind);
    cudaFree(d_val);
    cudaFree(d_val_copy);
    cudaFree(d_rhs);
    cudaFree(d_rhs_copy);
    cudaFree(d_dinv);

    AMGX_vector_destroy(x);
    AMGX_vector_destroy(b);
    AMGX_matrix_destroy(A);
    AMGX_solver_destroy(solver);
    AMGX_resources_destroy(rsrc);
    AMGX_config_destroy(cfg);
    AMGX_finalize();

    return (status == AMGX_SOLVE_SUCCESS) ? 0 : 1;
}
