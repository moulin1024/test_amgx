#include <iostream>
#include <fstream>
#include <stdexcept>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <string>
#include <amgx_c.h>
#include <cuda_runtime.h>

// ---------- CUDA helper ----------
#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t _e = (call);                                               \
        if (_e != cudaSuccess) {                                               \
            std::cerr << "CUDA error " << cudaGetErrorString(_e)               \
                      << " at " << __FILE__ << ":" << __LINE__ << "\n";        \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    } while (0)

static void amgx_check(AMGX_RC rc, const char* what) {
    if (rc != AMGX_RC_OK) {
        std::cerr << what << " failed with code " << rc << "\n";
        std::exit(EXIT_FAILURE);
    }
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

static inline std::string join_path(const std::string& dir, const std::string& file) {
#ifdef _WIN32
    const char sep = '\\';
#else
    const char sep = '/';
#endif
    if (dir.empty()) return file;
    if (dir.back() == '/' || dir.back() == '\\') return dir + file;
    return dir + sep + file;
}

int main(int argc, char** argv) {
    // argv[1] = config file (optional), argv[2] = data directory (optional)
    const std::string cfg_file = (argc > 1) ? argv[1] : "amgx_solver.json";
    const std::string data_dir = (argc > 2) ? argv[2] : "benchmark";

    // ---- AMGX init ----
    amgx_check(AMGX_initialize(), "AMGX_initialize");
    amgx_check(AMGX_install_signal_handler(), "AMGX_install_signal_handler");

    AMGX_config_handle cfg = nullptr;
    amgx_check(AMGX_config_create_from_file(&cfg, cfg_file.c_str()), "AMGX_config_create_from_file");
    amgx_check(AMGX_config_add_parameters(&cfg, "exception_handling=1"), "AMGX_config_add_parameters");

    AMGX_resources_handle rsrc = nullptr;
    amgx_check(AMGX_resources_create_simple(&rsrc, cfg), "AMGX_resources_create_simple");

    const AMGX_Mode mode = AMGX_mode_dDDI;
    AMGX_matrix_handle A = nullptr;
    AMGX_vector_handle b = nullptr, x = nullptr;
    AMGX_solver_handle solver = nullptr;

    amgx_check(AMGX_matrix_create(&A, rsrc, mode), "AMGX_matrix_create");
    amgx_check(AMGX_vector_create(&b, rsrc, mode), "AMGX_vector_create(b)");
    amgx_check(AMGX_vector_create(&x, rsrc, mode), "AMGX_vector_create(x)");
    amgx_check(AMGX_solver_create(&solver, rsrc, mode, cfg), "AMGX_solver_create");

    // ---- Load data to host ----
    double*  h_rhs     = nullptr;
    int*     h_row_ptr = nullptr;
    int*     h_col_ind = nullptr;
    double*  h_val     = nullptr;

    size_t rhs_n = 0, row_ptr_n = 0, col_ind_n = 0, val_n = 0;

    try {
        h_rhs     = load_bin<double>(join_path(data_dir, "rhs.bin"),     rhs_n);
        h_row_ptr = load_bin<int>   (join_path(data_dir, "row_ptr.bin"), row_ptr_n);
        h_col_ind = load_bin<int>   (join_path(data_dir, "col_ind.bin"), col_ind_n);
        h_val     = load_bin<double>(join_path(data_dir, "val.bin"),     val_n);
    } catch (const std::exception& e) {
        std::cerr << e.what() << "\n";
        delete[] h_rhs; delete[] h_row_ptr; delete[] h_col_ind; delete[] h_val;
        AMGX_finalize();
        return EXIT_FAILURE;
    }

    if (row_ptr_n == 0) {
        std::cerr << "row_ptr.bin is empty\n";
        delete[] h_rhs; delete[] h_row_ptr; delete[] h_col_ind; delete[] h_val;
        AMGX_finalize();
        return EXIT_FAILURE;
    }

    const int n  = static_cast<int>(row_ptr_n - 1);
    const int nz = static_cast<int>(val_n);

    if (nz != static_cast<int>(col_ind_n) ||
        h_row_ptr[row_ptr_n - 1] != nz ||
        static_cast<int>(rhs_n) != n) {
        std::cerr << "Input sizes inconsistent.\n";
        delete[] h_rhs; delete[] h_row_ptr; delete[] h_col_ind; delete[] h_val;
        AMGX_finalize();
        return EXIT_FAILURE;
    }

    // ---- Copy CSR & vectors to device ----
    int    *d_row_ptr = nullptr, *d_col_ind = nullptr;
    double *d_val     = nullptr, *d_rhs = nullptr, *d_x = nullptr;

    CUDA_CHECK(cudaMalloc((void**)&d_row_ptr, row_ptr_n * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_col_ind, col_ind_n * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_val,     val_n    * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&d_rhs,     rhs_n    * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&d_x,       static_cast<size_t>(n) * sizeof(double)));

    CUDA_CHECK(cudaMemcpy(d_row_ptr, h_row_ptr, row_ptr_n * sizeof(int),    cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_col_ind, h_col_ind, col_ind_n * sizeof(int),    cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_val,     h_val,     val_n    * sizeof(double),  cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_rhs,     h_rhs,     rhs_n    * sizeof(double),  cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_x, 0, static_cast<size_t>(n) * sizeof(double))); // initial guess = 0

    // ---- Upload matrix & vectors using DEVICE pointers (your build supports this) ----
    amgx_check(AMGX_matrix_upload_all(
        A, n, nz, 1, 1,
        d_row_ptr, d_col_ind, d_val, nullptr), "AMGX_matrix_upload_all(device)");

    amgx_check(AMGX_vector_bind(b, A), "AMGX_vector_bind(b,A)");
    amgx_check(AMGX_vector_bind(x, A), "AMGX_vector_bind(x,A)");

    amgx_check(AMGX_vector_upload(b, n, 1, d_rhs), "AMGX_vector_upload(b device)");
    amgx_check(AMGX_vector_upload(x, n, 1, d_x),   "AMGX_vector_upload(x device)");

    // ---- Solve ----
    amgx_check(AMGX_solver_setup(solver, A), "AMGX_solver_setup");
    AMGX_RC solve_rc = AMGX_solver_solve(solver, b, x);
    if (solve_rc != AMGX_RC_OK) {
        std::fprintf(stderr, "AMGX_solver_solve returned %d\n", (int)solve_rc);
    }

    // ---- Print result summary ----
    AMGX_SOLVE_STATUS status;
    amgx_check(AMGX_solver_get_status(solver, &status), "AMGX_solver_get_status");
    int iters = 0;
    amgx_check(AMGX_solver_get_iterations_number(solver, &iters), "AMGX_solver_get_iterations_number");
    std::printf("Solve status: %s, iterations: %d\n",
                (status == AMGX_SOLVE_SUCCESS ? "SUCCESS" : "NOT CONVERGED"), iters);

    // ---- Cleanup ----
    delete[] h_rhs;
    delete[] h_row_ptr;
    delete[] h_col_ind;
    delete[] h_val;

    CUDA_CHECK(cudaFree(d_row_ptr));
    CUDA_CHECK(cudaFree(d_col_ind));
    CUDA_CHECK(cudaFree(d_val));
    CUDA_CHECK(cudaFree(d_rhs));
    CUDA_CHECK(cudaFree(d_x));

    AMGX_vector_destroy(x);
    AMGX_vector_destroy(b);
    AMGX_matrix_destroy(A);
    AMGX_solver_destroy(solver);
    AMGX_resources_destroy(rsrc);
    amgx_check(AMGX_config_destroy(cfg), "AMGX_config_destroy");
    amgx_check(AMGX_finalize(), "AMGX_finalize");

    return (status == AMGX_SOLVE_SUCCESS) ? 0 : 1;
}
