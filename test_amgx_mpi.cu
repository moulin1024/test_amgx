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
#include <limits>

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

bool looks_like_one_based(
    const int global_n,
    const int* row_ptr,
    const int row_ptr_n,
    const int* col_ind,
    const int nnz)
{
    if (row_ptr_n <= 0 || nnz <= 0) {
        return false;
    }
    int col_min = col_ind[0];
    int col_max = col_ind[0];
    for (int i = 1; i < nnz; ++i) {
        col_min = std::min(col_min, col_ind[i]);
        col_max = std::max(col_max, col_ind[i]);
    }
    const bool rowptr_one_based = (row_ptr[0] == 1);
    const bool cols_one_based_range = (col_min >= 1 && col_max <= global_n);
    return rowptr_one_based || cols_one_based_range;
}

void shift_global_csr_to_zero_based(
    int* row_ptr,
    const int row_ptr_n,
    int* col_ind,
    const int nnz)
{
    for (int i = 0; i < row_ptr_n; ++i) {
        row_ptr[i] -= 1;
    }
    for (int i = 0; i < nnz; ++i) {
        col_ind[i] -= 1;
    }
}

bool amgx_check(const AMGX_RC rc, const char* what, const int rank)
{
    if (rc == AMGX_RC_OK) {
        return true;
    }
    char err[4096];
    err[0] = '\0';
    AMGX_get_error_string(rc, err, 4096);
    if (rank == 0) {
        std::cerr << "[rank " << rank << "] AMGX call failed: " << what
                  << " rc=" << static_cast<int>(rc)
                  << " msg=\"" << err << "\""
                  << std::endl;
    }
    return false;
}

void amgx_print_callback(const char* msg, int length)
{
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0) {
        std::cerr << "[AMGX rank " << rank << "] "
                  << std::string(msg, static_cast<size_t>(std::max(0, length)));
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
    const char* partvec_path = (argc > 3) ? argv[3] : nullptr;

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
        if (looks_like_one_based(
                global_n,
                h_row_ptr_global,
                static_cast<int>(row_ptr_n),
                h_col_ind_global,
                static_cast<int>(val_n))) {
            shift_global_csr_to_zero_based(
                h_row_ptr_global,
                static_cast<int>(row_ptr_n),
                h_col_ind_global,
                static_cast<int>(val_n));
        }

        // Partition vector (global row -> owning rank):
        // - if provided, use it
        // - otherwise fall back to contiguous blocks
        std::vector<int> partition_vector(static_cast<size_t>(global_n), 0);
        if (partvec_path != nullptr && std::strlen(partvec_path) > 0) {
            size_t part_n = 0;
            int* h_part = load_bin<int>(std::string(partvec_path), part_n);
            if (part_n != static_cast<size_t>(global_n)) {
                delete[] h_part;
                throw std::runtime_error("partition_vector.bin size does not match global_n");
            }
            for (int i = 0; i < global_n; ++i) {
                const int owner = h_part[i];
                if (owner < 0 || owner >= nranks) {
                    delete[] h_part;
                    throw std::runtime_error("partition_vector.bin contains invalid rank id");
                }
                partition_vector[static_cast<size_t>(i)] = owner;
            }
            delete[] h_part;
        } else {
            for (int rr = 0; rr < nranks; ++rr) {
                const partition_range rr_pr = block_partition(global_n, rr, nranks);
                for (int i = rr_pr.begin; i < rr_pr.end; ++i) {
                    partition_vector[static_cast<size_t>(i)] = rr;
                }
            }
        }

        // Build local owned row list from partition vector (supports non-contiguous ownership).
        std::vector<int> owned_rows;
        owned_rows.reserve(static_cast<size_t>(global_n / std::max(1, nranks) + 1));
        for (int gr = 0; gr < global_n; ++gr) {
            if (partition_vector[static_cast<size_t>(gr)] == rank) {
                owned_rows.push_back(gr);
            }
        }
        const int local_n = static_cast<int>(owned_rows.size());

        // Build local row block with global columns.
        std::vector<int> row_ptr_local(static_cast<size_t>(local_n + 1), 0);
        int local_nnz = 0;
        for (int lr = 0; lr < local_n; ++lr) {
            const int gr = owned_rows[static_cast<size_t>(lr)];
            local_nnz += h_row_ptr_global[gr + 1] - h_row_ptr_global[gr];
        }
        std::vector<int> col_ind_local(static_cast<size_t>(local_nnz), 0);
        std::vector<long> col_ind_local64(static_cast<size_t>(local_nnz), 0);
        std::vector<double> val_local(static_cast<size_t>(local_nnz), 0.0);
        std::vector<double> rhs_local(static_cast<size_t>(local_n), 0.0);
        std::vector<double> x_local(static_cast<size_t>(local_n), 0.0);

        int cursor = 0;
        for (int lr = 0; lr < local_n; ++lr) {
            row_ptr_local[static_cast<size_t>(lr)] = cursor;
            const int gr = owned_rows[static_cast<size_t>(lr)];
            rhs_local[static_cast<size_t>(lr)] = h_rhs_global[gr];
            for (int jj = h_row_ptr_global[gr]; jj < h_row_ptr_global[gr + 1]; ++jj) {
                col_ind_local[static_cast<size_t>(cursor)] = h_col_ind_global[jj];
                val_local[static_cast<size_t>(cursor)] = h_val_global[jj];
                ++cursor;
            }
        }
        row_ptr_local[static_cast<size_t>(local_n)] = cursor;
        for (int i = 0; i < local_nnz; ++i) {
            col_ind_local64[static_cast<size_t>(i)] = static_cast<long>(col_ind_local[static_cast<size_t>(i)]);
        }

        // Keep original operator/RHS from binaries (no extra scaling) to match AMGX MPI examples.

        // AMGX setup
        AMGX_initialize();
        AMGX_register_print_callback(&amgx_print_callback);
        AMGX_install_signal_handler();

        AMGX_config_handle cfg = nullptr;
        AMGX_resources_handle rsrc = nullptr;
        AMGX_matrix_handle A = nullptr;
        AMGX_vector_handle b = nullptr;
        AMGX_vector_handle x = nullptr;
        AMGX_solver_handle solver = nullptr;

        if (!amgx_check(AMGX_config_create_from_file(&cfg, cfg_path), "AMGX_config_create_from_file", rank)) {
            throw std::runtime_error("AMGX config create failed");
        }
        if (!amgx_check(AMGX_config_add_parameters(&cfg, "exception_handling=1"), "AMGX_config_add_parameters", rank)) {
            throw std::runtime_error("AMGX config add parameters failed");
        }

        MPI_Comm amgx_comm = MPI_COMM_WORLD;
        if (!amgx_check(AMGX_resources_create(&rsrc, cfg, &amgx_comm, 1, &device_id), "AMGX_resources_create", rank)) {
            throw std::runtime_error("AMGX resources create failed");
        }

        const AMGX_Mode mode = AMGX_mode_dDDI;
        if (!amgx_check(AMGX_matrix_create(&A, rsrc, mode), "AMGX_matrix_create", rank)
            || !amgx_check(AMGX_vector_create(&b, rsrc, mode), "AMGX_vector_create(b)", rank)
            || !amgx_check(AMGX_vector_create(&x, rsrc, mode), "AMGX_vector_create(x)", rank)
            || !amgx_check(AMGX_solver_create(&solver, rsrc, mode, cfg), "AMGX_solver_create", rank)) {
            throw std::runtime_error("AMGX handle creation failed");
        }

        // AMGX global upload in distributed mode.
        int nrings = 1;
        AMGX_config_get_default_number_of_rings(cfg, &nrings);
        int local_col_min = std::numeric_limits<int>::max();
        int local_col_max = std::numeric_limits<int>::min();
        int local_bad_col = 0;
        int local_unsorted_rows = 0;
        int local_duplicate_cols = 0;
        for (int i = 0; i < local_nnz; ++i) {
            local_col_min = std::min(local_col_min, col_ind_local[static_cast<size_t>(i)]);
            local_col_max = std::max(local_col_max, col_ind_local[static_cast<size_t>(i)]);
            if (col_ind_local[static_cast<size_t>(i)] < 0 || col_ind_local[static_cast<size_t>(i)] >= global_n) {
                local_bad_col += 1;
            }
        }
        for (int r = 0; r < local_n; ++r) {
            const int b = row_ptr_local[static_cast<size_t>(r)];
            const int e = row_ptr_local[static_cast<size_t>(r + 1)];
            for (int jj = b + 1; jj < e; ++jj) {
                const int prev = col_ind_local[static_cast<size_t>(jj - 1)];
                const int curr = col_ind_local[static_cast<size_t>(jj)];
                if (curr < prev) {
                    local_unsorted_rows += 1;
                    break;
                }
                if (curr == prev) {
                    local_duplicate_cols += 1;
                }
            }
        }
        int global_col_min = 0;
        int global_col_max = 0;
        int global_bad_col = 0;
        int global_unsorted_rows = 0;
        int global_duplicate_cols = 0;
        MPI_Allreduce(&local_col_min, &global_col_min, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        MPI_Allreduce(&local_col_max, &global_col_max, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
        MPI_Allreduce(&local_bad_col, &global_bad_col, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(&local_unsorted_rows, &global_unsorted_rows, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(&local_duplicate_cols, &global_duplicate_cols, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        if (rank == 0) {
            std::cerr << "[rank " << rank << "] upload: local_n=" << local_n
                      << " local_nnz=" << local_nnz
                      << " row_ptr0=" << row_ptr_local.front()
                      << " row_ptrN=" << row_ptr_local.back()
                      << " local_col_min=" << local_col_min
                      << " local_col_max=" << local_col_max
                      << " global_col_min=" << global_col_min
                      << " global_col_max=" << global_col_max
                      << " local_bad_col=" << local_bad_col
                      << " local_unsorted_rows=" << local_unsorted_rows
                      << " local_duplicate_cols=" << local_duplicate_cols
                      << " global_bad_col=" << global_bad_col
                      << " global_unsorted_rows=" << global_unsorted_rows
                      << " global_duplicate_cols=" << global_duplicate_cols
                      << " global_n=" << global_n
                      << std::endl;
        }
        if (!amgx_check(AMGX_matrix_upload_all_global(
            A,
            global_n,
            local_n,
            local_nnz,
            1,
            1,
            row_ptr_local.data(),
            col_ind_local64.data(),
            val_local.data(),
            nullptr,
            nrings,
            nrings,
            partition_vector.data()), "AMGX_matrix_upload_all_global", rank)) {
            throw std::runtime_error("AMGX matrix upload failed");
        }

        // Follow AMGX MPI examples: bind vectors after matrix upload establishes comm pattern.
        if (!amgx_check(AMGX_vector_bind(b, A), "AMGX_vector_bind(b,A)", rank)
            || !amgx_check(AMGX_vector_bind(x, A), "AMGX_vector_bind(x,A)", rank)) {
            throw std::runtime_error("AMGX vector bind failed");
        }

        if (!amgx_check(AMGX_vector_upload(b, local_n, 1, rhs_local.data()), "AMGX_vector_upload(b)", rank)
            || !amgx_check(AMGX_vector_upload(x, local_n, 1, x_local.data()), "AMGX_vector_upload(x)", rank)) {
            throw std::runtime_error("AMGX vector upload failed");
        }

        MPI_Barrier(MPI_COMM_WORLD);
        if (rank == 0) {
            std::cerr << "[rank " << rank << "] entering AMGX_solver_setup" << std::endl;
        }
        const AMGX_RC rc_setup = AMGX_solver_setup(solver, A);
        if (!amgx_check(rc_setup, "AMGX_solver_setup", rank)) {
            throw std::runtime_error("AMGX solver setup failed");
        }
        MPI_Barrier(MPI_COMM_WORLD);
        if (rank == 0) {
            std::cerr << "[rank " << rank << "] entering AMGX_solver_solve" << std::endl;
        }
        const AMGX_RC rc_solve = AMGX_solver_solve(solver, b, x);
        if (!amgx_check(rc_solve, "AMGX_solver_solve", rank)) {
            AMGX_SOLVE_STATUS failed_status = AMGX_SOLVE_FAILED;
            const AMGX_RC rc_status = AMGX_solver_get_status(solver, &failed_status);
            if (rc_status == AMGX_RC_OK) {
                if (rank == 0) {
                    std::cerr << "[rank " << rank << "] AMGX solver status after failed solve: "
                              << static_cast<int>(failed_status) << std::endl;
                }
            }
            throw std::runtime_error("AMGX solver solve failed");
        }

        AMGX_SOLVE_STATUS status = AMGX_SOLVE_FAILED;
        if (!amgx_check(AMGX_solver_get_status(solver, &status), "AMGX_solver_get_status", rank)) {
            throw std::runtime_error("AMGX solver status query failed");
        }
        int iters = -1;
        if (!amgx_check(AMGX_solver_get_iterations_number(solver, &iters), "AMGX_solver_get_iterations_number", rank)) {
            throw std::runtime_error("AMGX iteration query failed");
        }

        if (!amgx_check(AMGX_vector_download(x, x_local.data()), "AMGX_vector_download(x)", rank)) {
            throw std::runtime_error("AMGX vector download failed");
        }

        // Gather full solution on rank 0 for quick validation/debug.
        std::vector<double> x_rank_full(static_cast<size_t>(global_n), 0.0);
        for (int lr = 0; lr < local_n; ++lr) {
            const int gr = owned_rows[static_cast<size_t>(lr)];
            x_rank_full[static_cast<size_t>(gr)] = x_local[static_cast<size_t>(lr)];
        }
        std::vector<double> x_full(static_cast<size_t>(global_n), 0.0);
        MPI_Allreduce(
            x_rank_full.data(),
            x_full.data(),
            global_n,
            MPI_DOUBLE,
            MPI_SUM,
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

        amgx_check(AMGX_solver_destroy(solver), "AMGX_solver_destroy", rank);
        amgx_check(AMGX_vector_destroy(x), "AMGX_vector_destroy(x)", rank);
        amgx_check(AMGX_vector_destroy(b), "AMGX_vector_destroy(b)", rank);
        amgx_check(AMGX_matrix_destroy(A), "AMGX_matrix_destroy", rank);
        amgx_check(AMGX_resources_destroy(rsrc), "AMGX_resources_destroy", rank);
        amgx_check(AMGX_config_destroy(cfg), "AMGX_config_destroy", rank);
        amgx_check(AMGX_finalize(), "AMGX_finalize", rank);
    }
    catch (const std::exception& e) {
        if (rank == 0) {
            std::cerr << "[rank " << rank << "] ERROR: " << e.what() << std::endl;
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    MPI_Finalize();
    return 0;
}

