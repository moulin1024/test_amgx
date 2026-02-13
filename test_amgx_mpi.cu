#include <amgx_c.h>
#include <cuda_runtime.h>
#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

template <typename T>
T* load_bin(const std::string& path, size_t& out_count)
{
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        throw std::runtime_error("Cannot open " + path);
    }

    f.seekg(0, std::ios::end);
    const std::streamsize bytes = f.tellg();
    f.seekg(0, std::ios::beg);

    if (bytes < 0 || (static_cast<size_t>(bytes) % sizeof(T)) != 0) {
        throw std::runtime_error("File size of " + path + " is not a multiple of element size");
    }

    out_count = static_cast<size_t>(bytes) / sizeof(T);
    T* data = new T[out_count];
    if (!f.read(reinterpret_cast<char*>(data), bytes)) {
        delete[] data;
        throw std::runtime_error("Failed reading " + path);
    }
    return data;
}

struct partition_range {
    int begin;
    int end;
    int size;
};

partition_range block_partition(const int global_n, const int rank, const int nranks)
{
    const int base = global_n / nranks;
    const int rem = global_n % nranks;
    partition_range pr{};
    pr.begin = rank * base + std::min(rank, rem);
    pr.size = base + ((rank < rem) ? 1 : 0);
    pr.end = pr.begin + pr.size;
    return pr;
}

int local_rank_from_env()
{
    const char* envs[] = {
        "OMPI_COMM_WORLD_LOCAL_RANK",
        "MV2_COMM_WORLD_LOCAL_RANK",
        "SLURM_LOCALID",
        "PMI_LOCAL_RANK"
    };
    for (const char* key : envs) {
        const char* raw = std::getenv(key);
        if (raw != nullptr && *raw != '\0') {
            return std::atoi(raw);
        }
    }
    return -1;
}

void scale_rows_and_rhs_in_place(
    const int global_row_begin,
    const int local_n,
    const int* row_ptr,
    const int* col_ind,
    double* val,
    double* rhs,
    const double eps)
{
    for (int lr = 0; lr < local_n; ++lr) {
        const int global_row = global_row_begin + lr;
        const int start = row_ptr[lr];
        const int end = row_ptr[lr + 1];

        double diag = 0.0;
        for (int jj = start; jj < end; ++jj) {
            if (col_ind[jj] == global_row) {
                diag = val[jj];
                break;
            }
        }

        const double dinv = (std::abs(diag) > eps) ? (1.0 / diag) : 1.0;
        for (int jj = start; jj < end; ++jj) {
            val[jj] *= dinv;
        }
        rhs[lr] *= dinv;
    }
}

} // namespace

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);

    int rank = 0;
    int nranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);

    const std::string data_dir = (argc > 1) ? argv[1] : "data";
    const char* cfg_path = (argc > 2) ? argv[2] : "amgx_config.json";

    // Bind rank to a visible GPU
    int gpu_count = 0;
    cudaGetDeviceCount(&gpu_count);
    if (gpu_count <= 0) {
        if (rank == 0) {
            std::cerr << "No CUDA devices found.\n";
        }
        MPI_Finalize();
        return 1;
    }
    int lrank = local_rank_from_env();
    if (lrank < 0) {
        lrank = rank;
    }
    const int device_id = lrank % gpu_count;
    cudaSetDevice(device_id);

    try {
        // Load global CSR + RHS (each rank starts from full data, same as main workflow).
        size_t rhs_n = 0, row_ptr_n = 0, col_ind_n = 0, val_n = 0;
        double* h_rhs_global = load_bin<double>(data_dir + "/rhs.bin", rhs_n);
        int* h_row_ptr_global = load_bin<int>(data_dir + "/row_ptr.bin", row_ptr_n);
        int* h_col_ind_global = load_bin<int>(data_dir + "/col_ind.bin", col_ind_n);
        double* h_val_global = load_bin<double>(data_dir + "/val.bin", val_n);

        const int global_n = static_cast<int>(rhs_n);
        if (row_ptr_n != rhs_n + 1 || static_cast<size_t>(h_row_ptr_global[global_n]) != val_n) {
            throw std::runtime_error("Inconsistent CSR dimensions in input binaries");
        }

        // Partition rows by contiguous blocks.
        const partition_range pr = block_partition(global_n, rank, nranks);
        const int local_n = pr.size;

        // Build local row block with global columns.
        std::vector<int> row_ptr_local(static_cast<size_t>(local_n + 1), 0);
        int local_nnz = 0;
        for (int lr = 0; lr < local_n; ++lr) {
            const int gr = pr.begin + lr;
            local_nnz += h_row_ptr_global[gr + 1] - h_row_ptr_global[gr];
        }
        std::vector<int> col_ind_local(static_cast<size_t>(local_nnz), 0);
        std::vector<double> val_local(static_cast<size_t>(local_nnz), 0.0);
        std::vector<double> rhs_local(static_cast<size_t>(local_n), 0.0);
        std::vector<double> x_local(static_cast<size_t>(local_n), 0.0);

        int cursor = 0;
        for (int lr = 0; lr < local_n; ++lr) {
            row_ptr_local[static_cast<size_t>(lr)] = cursor;
            const int gr = pr.begin + lr;
            rhs_local[static_cast<size_t>(lr)] = h_rhs_global[gr];
            for (int jj = h_row_ptr_global[gr]; jj < h_row_ptr_global[gr + 1]; ++jj) {
                col_ind_local[static_cast<size_t>(cursor)] = h_col_ind_global[jj];
                val_local[static_cast<size_t>(cursor)] = h_val_global[jj];
                ++cursor;
            }
        }
        row_ptr_local[static_cast<size_t>(local_n)] = cursor;

        // Build explicit partition vector (global row -> owning rank)
        std::vector<int> partition_vector(static_cast<size_t>(global_n), 0);
        for (int rr = 0; rr < nranks; ++rr) {
            const partition_range rr_pr = block_partition(global_n, rr, nranks);
            for (int i = rr_pr.begin; i < rr_pr.end; ++i) {
                partition_vector[static_cast<size_t>(i)] = rr;
            }
        }

        // Optional Jacobi row scaling, matching single-rank mini-app behavior.
        scale_rows_and_rhs_in_place(
            pr.begin,
            local_n,
            row_ptr_local.data(),
            col_ind_local.data(),
            val_local.data(),
            rhs_local.data(),
            1e-30);

        // AMGX setup
        AMGX_initialize();
        AMGX_install_signal_handler();

        AMGX_config_handle cfg = nullptr;
        AMGX_resources_handle rsrc = nullptr;
        AMGX_matrix_handle A = nullptr;
        AMGX_vector_handle b = nullptr;
        AMGX_vector_handle x = nullptr;
        AMGX_solver_handle solver = nullptr;

        AMGX_SAFE_CALL(AMGX_config_create_from_file(&cfg, cfg_path));
        AMGX_SAFE_CALL(AMGX_config_add_parameters(&cfg, "exception_handling=1"));

        MPI_Comm amgx_comm = MPI_COMM_WORLD;
        AMGX_SAFE_CALL(AMGX_resources_create(&rsrc, cfg, &amgx_comm, 1, &device_id));

        const AMGX_Mode mode = AMGX_mode_dDDI;
        AMGX_SAFE_CALL(AMGX_matrix_create(&A, rsrc, mode));
        AMGX_SAFE_CALL(AMGX_vector_create(&b, rsrc, mode));
        AMGX_SAFE_CALL(AMGX_vector_create(&x, rsrc, mode));
        AMGX_SAFE_CALL(AMGX_solver_create(&solver, rsrc, mode, cfg));

        AMGX_SAFE_CALL(AMGX_vector_bind(b, A));
        AMGX_SAFE_CALL(AMGX_vector_bind(x, A));

        // AMGX global upload in distributed mode.
        int nrings = 1;
        AMGX_config_get_default_number_of_rings(cfg, &nrings);
        AMGX_SAFE_CALL(AMGX_matrix_upload_all_global(
            A,
            global_n,
            local_n,
            local_nnz,
            1,
            1,
            row_ptr_local.data(),
            col_ind_local.data(),
            val_local.data(),
            nullptr,
            nrings,
            nrings,
            partition_vector.data()));

        AMGX_SAFE_CALL(AMGX_vector_upload(b, local_n, 1, rhs_local.data()));
        AMGX_SAFE_CALL(AMGX_vector_set_zero(x, local_n, 1));

        MPI_Barrier(MPI_COMM_WORLD);
        AMGX_SAFE_CALL(AMGX_solver_setup(solver, A));
        MPI_Barrier(MPI_COMM_WORLD);
        AMGX_SAFE_CALL(AMGX_solver_solve(solver, b, x));

        AMGX_SOLVE_STATUS status = AMGX_SOLVE_FAILED;
        AMGX_SAFE_CALL(AMGX_solver_get_status(solver, &status));
        int iters = -1;
        AMGX_SAFE_CALL(AMGX_solver_get_iterations_number(solver, &iters));

        AMGX_SAFE_CALL(AMGX_vector_download(x, x_local.data()));

        // Gather full solution on rank 0 for quick validation/debug.
        std::vector<int> recvcounts(static_cast<size_t>(nranks), 0);
        std::vector<int> displs(static_cast<size_t>(nranks), 0);
        for (int rr = 0; rr < nranks; ++rr) {
            const partition_range rr_pr = block_partition(global_n, rr, nranks);
            recvcounts[static_cast<size_t>(rr)] = rr_pr.size;
            displs[static_cast<size_t>(rr)] = rr_pr.begin;
        }
        std::vector<double> x_full;
        if (rank == 0) {
            x_full.resize(static_cast<size_t>(global_n), 0.0);
        }
        MPI_Gatherv(
            x_local.data(),
            local_n,
            MPI_DOUBLE,
            rank == 0 ? x_full.data() : nullptr,
            recvcounts.data(),
            displs.data(),
            MPI_DOUBLE,
            0,
            MPI_COMM_WORLD);

        if (rank == 0) {
            std::ofstream fsol(data_dir + "/sol_mpi.bin", std::ios::binary);
            fsol.write(reinterpret_cast<const char*>(x_full.data()), static_cast<std::streamsize>(x_full.size() * sizeof(double)));
            fsol.close();
            std::cout << "AMGX MPI solve status=" << static_cast<int>(status)
                      << ", iterations=" << iters
                      << ", nprocs=" << nranks
                      << ", global_n=" << global_n
                      << std::endl;
        }

        // Cleanup
        delete[] h_rhs_global;
        delete[] h_row_ptr_global;
        delete[] h_col_ind_global;
        delete[] h_val_global;

        AMGX_SAFE_CALL(AMGX_solver_destroy(solver));
        AMGX_SAFE_CALL(AMGX_vector_destroy(x));
        AMGX_SAFE_CALL(AMGX_vector_destroy(b));
        AMGX_SAFE_CALL(AMGX_matrix_destroy(A));
        AMGX_SAFE_CALL(AMGX_resources_destroy(rsrc));
        AMGX_SAFE_CALL(AMGX_config_destroy(cfg));
        AMGX_SAFE_CALL(AMGX_finalize());
    }
    catch (const std::exception& e) {
        std::cerr << "[rank " << rank << "] ERROR: " << e.what() << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    MPI_Finalize();
    return 0;
}

