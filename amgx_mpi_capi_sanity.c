#include <amgx_c.h>
#include <mpi.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#define MAX_MSG_LEN 4096

static int g_rank = 0;

static void err_and_abort(const char *msg)
{
    fprintf(stderr, "%s\n", msg);
    fflush(stderr);
    MPI_Abort(MPI_COMM_WORLD, 1);
}

static int amgx_ok(AMGX_RC rc, const char *what)
{
    if (rc == AMGX_RC_OK) {
        return 1;
    }
    char err[MAX_MSG_LEN];
    err[0] = '\0';
    AMGX_get_error_string(rc, err, MAX_MSG_LEN);
    fprintf(stderr, "[rank %d] AMGX call failed: %s rc=%d msg=\"%s\"\n", g_rank, what, (int)rc, err);
    return 0;
}

static int cuda_ok(cudaError_t err, const char *what)
{
    if (err == cudaSuccess) {
        return 1;
    }
    fprintf(stderr, "[rank %d] CUDA call failed: %s err=\"%s\"\n", g_rank, what, cudaGetErrorString(err));
    return 0;
}

static void print_callback(const char *msg, int length)
{
    (void)length;
    if (g_rank == 0) {
        fputs(msg, stdout);
    }
}

static void print_usage_and_exit(void)
{
    char msg[MAX_MSG_LEN] =
        "Usage: mpirun [-n nranks] ./amgx_mpi_capi_sanity "
        "[-mode [dDDI|dDFI|dFFI]] -m <binary_data_dir> "
        "[-c config_file | -amg \"key=value,...\"] "
        "[-partvec partition_vector.bin] [-r N]\n";
    strcat(msg, "  Binary data dir must contain: row_ptr.bin col_ind.bin val.bin rhs.bin\n");
    print_callback(msg, MAX_MSG_LEN);
    MPI_Finalize();
    exit(0);
}

static int find_param_index(char **argv, int argc, const char *parm)
{
    int count = 0;
    int index = -1;
    int i;
    for (i = 0; i < argc; ++i) {
        if (strncmp(argv[i], parm, 100) == 0) {
            index = i;
            count++;
        }
    }
    if (count <= 1) {
        return index;
    }
    fprintf(stderr, "ERROR: parameter %s has been specified more than once\n", parm);
    exit(1);
}

static int local_rank_from_env(void)
{
    const char *keys[] = {
        "OMPI_COMM_WORLD_LOCAL_RANK",
        "MV2_COMM_WORLD_LOCAL_RANK",
        "SLURM_LOCALID",
        "PMI_LOCAL_RANK"
    };
    size_t i;
    for (i = 0; i < sizeof(keys) / sizeof(keys[0]); ++i) {
        const char *raw = getenv(keys[i]);
        if (raw != NULL && raw[0] != '\0') {
            return atoi(raw);
        }
    }
    return -1;
}

static int is_directory_path(const char *path)
{
    struct stat st;
    if (stat(path, &st) != 0) {
        return 0;
    }
    return S_ISDIR(st.st_mode) ? 1 : 0;
}

static void make_path(char *out, size_t out_size, const char *dir, const char *name)
{
    size_t dlen = strlen(dir);
    if (dlen > 0 && dir[dlen - 1] == '/') {
        snprintf(out, out_size, "%s%s", dir, name);
    } else {
        snprintf(out, out_size, "%s/%s", dir, name);
    }
}

static void *read_bin_file(const char *path, size_t elem_size, size_t *count_out)
{
    FILE *f = fopen(path, "rb");
    void *data = NULL;
    long bytes;
    size_t count;
    if (f == NULL) {
        return NULL;
    }
    if (fseek(f, 0, SEEK_END) != 0) {
        fclose(f);
        return NULL;
    }
    bytes = ftell(f);
    if (bytes < 0 || ((size_t)bytes % elem_size) != 0) {
        fclose(f);
        return NULL;
    }
    rewind(f);
    count = (size_t)bytes / elem_size;
    data = malloc(count * elem_size);
    if (data == NULL) {
        fclose(f);
        return NULL;
    }
    if (fread(data, elem_size, count, f) != count) {
        free(data);
        fclose(f);
        return NULL;
    }
    fclose(f);
    *count_out = count;
    return data;
}

static void normalize_to_zero_based(int *row_ptr, size_t row_ptr_n, int *col_ind, size_t nnz, int global_n)
{
    int one_based;
    size_t i;
    if (row_ptr_n == 0 || nnz == 0) {
        return;
    }
    one_based = (row_ptr[0] == 1);
    if (!one_based) {
        int cmin = col_ind[0];
        int cmax = col_ind[0];
        for (i = 1; i < nnz; ++i) {
            if (col_ind[i] < cmin) cmin = col_ind[i];
            if (col_ind[i] > cmax) cmax = col_ind[i];
        }
        if (cmin >= 1 && cmax <= global_n) {
            one_based = 1;
        }
    }
    if (one_based) {
        for (i = 0; i < row_ptr_n; ++i) row_ptr[i] -= 1;
        for (i = 0; i < nnz; ++i) col_ind[i] -= 1;
    }
}

static void block_partition(int n, int rank, int nranks, int *begin, int *size)
{
    int base = n / nranks;
    int rem = n % nranks;
    *begin = rank * base + ((rank < rem) ? rank : rem);
    *size = base + ((rank < rem) ? 1 : 0);
}

static int read_partition_vector_file(const char *path, int **vec_out, int *size_out)
{
    FILE *f = fopen(path, "rb");
    long bytes;
    int *vec = NULL;
    int n;
    if (f == NULL) {
        return 0;
    }
    if (fseek(f, 0, SEEK_END) != 0) {
        fclose(f);
        return 0;
    }
    bytes = ftell(f);
    if (bytes < 0 || (bytes % (long)sizeof(int)) != 0) {
        fclose(f);
        return 0;
    }
    rewind(f);
    n = (int)(bytes / (long)sizeof(int));
    vec = (int *)malloc((size_t)n * sizeof(int));
    if (vec == NULL) {
        fclose(f);
        return 0;
    }
    if ((int)fread((void *)vec, sizeof(int), (size_t)n, f) != n) {
        free(vec);
        fclose(f);
        return 0;
    }
    fclose(f);
    *vec_out = vec;
    *size_out = n;
    return 1;
}

static AMGX_RC amgx_read_system_distributed_from_bin(
    AMGX_matrix_handle mtx,
    AMGX_vector_handle rhs,
    AMGX_vector_handle sol,
    const char *data_dir,
    int allocated_halo_depth,
    int num_partitions,
    const int *partition_sizes,
    int partition_vector_size,
    const int *partition_vector)
{
    int rank = 0, nranks = 1;
    char row_ptr_path[4096], col_ind_path[4096], val_path[4096], rhs_path[4096];
    size_t row_ptr_n = 0, col_ind_n = 0, val_n = 0, rhs_n = 0;
    int *row_ptr = NULL, *col_ind = NULL;
    double *val = NULL, *rhs_full = NULL;
    int global_n;
    int local_begin = 0, local_n = 0;
    int local_nnz = 0;
    int *row_ptr_local = NULL;
    long *col_ind_local = NULL;
    double *val_local = NULL, *rhs_local = NULL, *sol_local = NULL;
    int *partvec_owned = NULL;
    const int *partvec_arg = partition_vector;
    AMGX_RC rc;
    int lr, cursor;

    (void)partition_sizes;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);

    if (!is_directory_path(data_dir)) {
        fprintf(stderr, "[rank %d] -m must point to directory containing row_ptr.bin/col_ind.bin/val.bin/rhs.bin\n", rank);
        return AMGX_RC_IO_ERROR;
    }

    make_path(row_ptr_path, sizeof(row_ptr_path), data_dir, "row_ptr.bin");
    make_path(col_ind_path, sizeof(col_ind_path), data_dir, "col_ind.bin");
    make_path(val_path, sizeof(val_path), data_dir, "val.bin");
    make_path(rhs_path, sizeof(rhs_path), data_dir, "rhs.bin");

    row_ptr = (int *)read_bin_file(row_ptr_path, sizeof(int), &row_ptr_n);
    col_ind = (int *)read_bin_file(col_ind_path, sizeof(int), &col_ind_n);
    val = (double *)read_bin_file(val_path, sizeof(double), &val_n);
    rhs_full = (double *)read_bin_file(rhs_path, sizeof(double), &rhs_n);
    if (!row_ptr || !col_ind || !val || !rhs_full) {
        fprintf(stderr, "[rank %d] failed reading binary inputs from %s\n", rank, data_dir);
        rc = AMGX_RC_IO_ERROR;
        goto cleanup;
    }

    global_n = (int)rhs_n;
    if (row_ptr_n != rhs_n + 1 || col_ind_n != val_n || (size_t)row_ptr[global_n] != val_n) {
        fprintf(stderr, "[rank %d] inconsistent CSR dimensions in binary input\n", rank);
        rc = AMGX_RC_BAD_PARAMETERS;
        goto cleanup;
    }
    normalize_to_zero_based(row_ptr, row_ptr_n, col_ind, val_n, global_n);

    if (partition_vector != NULL && partition_vector_size == global_n) {
        int first = -1, last = -1;
        int i;
        for (i = 0; i < global_n; ++i) {
            if (partition_vector[i] == rank) {
                if (first < 0) first = i;
                last = i;
            } else if (first >= 0 && last >= 0 && i > (last + 1)) {
                fprintf(stderr, "[rank %d] non-contiguous partition vector unsupported by this loader\n", rank);
                rc = AMGX_RC_BAD_PARAMETERS;
                goto cleanup;
            }
        }
        if (first < 0) {
            local_begin = 0;
            local_n = 0;
        } else {
            local_begin = first;
            local_n = last - first + 1;
        }
    } else {
        int parts = (num_partitions > 0) ? num_partitions : nranks;
        block_partition(global_n, rank, parts, &local_begin, &local_n);
    }

    for (lr = 0; lr < local_n; ++lr) {
        int gr = local_begin + lr;
        local_nnz += row_ptr[gr + 1] - row_ptr[gr];
    }

    row_ptr_local = (int *)malloc((size_t)(local_n + 1) * sizeof(int));
    col_ind_local = (long *)malloc((size_t)local_nnz * sizeof(long));
    val_local = (double *)malloc((size_t)local_nnz * sizeof(double));
    rhs_local = (double *)malloc((size_t)local_n * sizeof(double));
    sol_local = (double *)calloc((size_t)local_n, sizeof(double));
    if (!row_ptr_local || !col_ind_local || !val_local || !rhs_local || !sol_local) {
        rc = AMGX_RC_NO_MEMORY;
        goto cleanup;
    }

    cursor = 0;
    for (lr = 0; lr < local_n; ++lr) {
        int gr = local_begin + lr;
        int jj;
        row_ptr_local[lr] = cursor;
        rhs_local[lr] = rhs_full[gr];
        for (jj = row_ptr[gr]; jj < row_ptr[gr + 1]; ++jj) {
            col_ind_local[cursor] = (long)col_ind[jj];
            val_local[cursor] = val[jj];
            cursor++;
        }
    }
    row_ptr_local[local_n] = cursor;

    if (partvec_arg == NULL) {
        int parts = (num_partitions > 0) ? num_partitions : nranks;
        int rr;
        partvec_owned = (int *)malloc((size_t)global_n * sizeof(int));
        if (partvec_owned == NULL) {
            rc = AMGX_RC_NO_MEMORY;
            goto cleanup;
        }
        for (rr = 0; rr < parts; ++rr) {
            int b = 0, s = 0, i;
            block_partition(global_n, rr, parts, &b, &s);
            for (i = b; i < b + s; ++i) {
                partvec_owned[i] = rr;
            }
        }
        partvec_arg = partvec_owned;
    }

    rc = AMGX_matrix_upload_all_global(
        mtx,
        global_n,
        local_n,
        local_nnz,
        1,
        1,
        row_ptr_local,
        col_ind_local,
        val_local,
        NULL,
        allocated_halo_depth,
        allocated_halo_depth,
        partvec_arg);
    if (rc != AMGX_RC_OK) {
        goto cleanup;
    }

    rc = AMGX_vector_bind(rhs, mtx);
    if (rc != AMGX_RC_OK) goto cleanup;
    rc = AMGX_vector_bind(sol, mtx);
    if (rc != AMGX_RC_OK) goto cleanup;
    rc = AMGX_vector_upload(rhs, local_n, 1, rhs_local);
    if (rc != AMGX_RC_OK) goto cleanup;
    rc = AMGX_vector_upload(sol, local_n, 1, sol_local);
    if (rc != AMGX_RC_OK) goto cleanup;

cleanup:
    free(row_ptr);
    free(col_ind);
    free(val);
    free(rhs_full);
    free(row_ptr_local);
    free(col_ind_local);
    free(val_local);
    free(rhs_local);
    free(sol_local);
    free(partvec_owned);
    return rc;
}

int main(int argc, char **argv)
{
    int pidx = -1;
    int pidy = -1;
    int rank = 0;
    int nranks = 0;
    int lrank = -1;
    int gpu_count = 0;
    MPI_Comm amgx_mpi_comm = MPI_COMM_WORLD;
    AMGX_Mode mode = AMGX_mode_dDDI;
    AMGX_config_handle cfg = NULL;
    AMGX_resources_handle rsrc = NULL;
    AMGX_matrix_handle A = NULL;
    AMGX_vector_handle b = NULL;
    AMGX_vector_handle x = NULL;
    AMGX_solver_handle solver = NULL;
    AMGX_SOLVE_STATUS status = AMGX_SOLVE_FAILED;
    int *partition_vector = NULL;
    int partition_vector_size = 0;
    int *partition_sizes = NULL;
    int nrings = 1;
    int n = 0, block_dimx = 1, block_dimy = 1;
    int global_n = 0;
    int nrepeats = 1;
    size_t sizeof_v_val = sizeof(double);
    void *x_h = NULL;
    int r;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(amgx_mpi_comm, &nranks);
    MPI_Comm_rank(amgx_mpi_comm, &rank);
    g_rank = rank;

    if (!cuda_ok(cudaGetDeviceCount(&gpu_count), "cudaGetDeviceCount") || gpu_count <= 0) {
        err_and_abort("ERROR: no CUDA devices found");
    }
    lrank = local_rank_from_env();
    if (lrank < 0) {
        lrank = rank;
    }
    lrank = lrank % gpu_count;
    if (!cuda_ok(cudaSetDevice(lrank), "cudaSetDevice")) {
        err_and_abort("ERROR: failed to bind CUDA device");
    }
    printf("Process %d selecting device %d\n", rank, lrank);

    if (argc == 1) {
        print_usage_and_exit();
    }

    if (!amgx_ok(AMGX_initialize(), "AMGX_initialize")) err_and_abort("AMGX init failed");
    if (!amgx_ok(AMGX_register_print_callback(&print_callback), "AMGX_register_print_callback")) err_and_abort("AMGX callback registration failed");
    if (!amgx_ok(AMGX_install_signal_handler(), "AMGX_install_signal_handler")) err_and_abort("AMGX signal handler registration failed");

    if ((pidx = find_param_index(argv, argc, "-mode")) != -1) {
        if (strncmp(argv[pidx + 1], "dDDI", 100) == 0) {
            mode = AMGX_mode_dDDI;
        } else if (strncmp(argv[pidx + 1], "dDFI", 100) == 0) {
            mode = AMGX_mode_dDFI;
        } else if (strncmp(argv[pidx + 1], "dFFI", 100) == 0) {
            mode = AMGX_mode_dFFI;
        } else {
            err_and_abort("ERROR: invalid mode");
        }
    }

    pidx = find_param_index(argv, argc, "-amg");
    pidy = find_param_index(argv, argc, "-c");
    if (pidx != -1 && pidy != -1) {
        if (!amgx_ok(AMGX_config_create_from_file_and_string(&cfg, argv[pidy + 1], argv[pidx + 1]), "AMGX_config_create_from_file_and_string")) {
            err_and_abort("AMGX config creation failed");
        }
    } else if (pidy != -1) {
        if (!amgx_ok(AMGX_config_create_from_file(&cfg, argv[pidy + 1]), "AMGX_config_create_from_file")) {
            err_and_abort("AMGX config creation failed");
        }
    } else if (pidx != -1) {
        if (!amgx_ok(AMGX_config_create(&cfg, argv[pidx + 1]), "AMGX_config_create")) {
            err_and_abort("AMGX config creation failed");
        }
    } else {
        err_and_abort("ERROR: no config was specified (use -c or -amg)");
    }

    if (!amgx_ok(AMGX_config_add_parameters(&cfg, "exception_handling=1"), "AMGX_config_add_parameters")) err_and_abort("AMGX config parameter set failed");
    if (!amgx_ok(AMGX_resources_create(&rsrc, cfg, &amgx_mpi_comm, 1, &lrank), "AMGX_resources_create")) err_and_abort("AMGX resources create failed");
    if (!amgx_ok(AMGX_matrix_create(&A, rsrc, mode), "AMGX_matrix_create")) err_and_abort("AMGX matrix create failed");
    if (!amgx_ok(AMGX_vector_create(&x, rsrc, mode), "AMGX_vector_create(x)")) err_and_abort("AMGX vector create x failed");
    if (!amgx_ok(AMGX_vector_create(&b, rsrc, mode), "AMGX_vector_create(b)")) err_and_abort("AMGX vector create b failed");
    if (!amgx_ok(AMGX_solver_create(&solver, rsrc, mode, cfg), "AMGX_solver_create")) err_and_abort("AMGX solver create failed");

    if ((pidx = find_param_index(argv, argc, "-partvec")) != -1) {
        if (!read_partition_vector_file(argv[pidx + 1], &partition_vector, &partition_vector_size)) {
            err_and_abort("ERROR: reading partition vector file");
        }
        if (rank == 0) {
            printf("Read partition vector with %d entries\n", partition_vector_size);
        }
    }

    if (!amgx_ok(AMGX_config_get_default_number_of_rings(cfg, &nrings), "AMGX_config_get_default_number_of_rings")) err_and_abort("AMGX nrings query failed");
    pidx = find_param_index(argv, argc, "-m");
    if (pidx == -1) {
        err_and_abort("ERROR: no linear system was specified (use -m)");
    }
    if (!amgx_ok(
            amgx_read_system_distributed_from_bin(
                A, b, x,
                argv[pidx + 1],
                nrings, nranks,
                partition_sizes, partition_vector_size, partition_vector),
            "amgx_read_system_distributed_from_bin")) {
        err_and_abort("Binary system load/upload failed");
    }

    if (!amgx_ok(AMGX_matrix_get_size(A, &n, &block_dimx, &block_dimy), "AMGX_matrix_get_size")) err_and_abort("AMGX matrix size query failed");
    MPI_Reduce(&n, &global_n, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        if (block_dimx > 1 || block_dimy > 1) {
            printf("Matrix A has %d rows with %d x %d blocks\n", global_n, block_dimx, block_dimy);
        } else {
            printf("Matrix A is scalar and has %d rows\n", global_n);
        }
    }

    sizeof_v_val = (AMGX_GET_MODE_VAL(AMGX_VecPrecision, mode) == AMGX_vecDouble) ? sizeof(double) : sizeof(float);
    pidx = find_param_index(argv, argc, "-r");
    if (pidx != -1) {
        nrepeats = atoi(argv[pidx + 1]);
        x_h = malloc((size_t)n * (size_t)block_dimx * sizeof_v_val);
        if (x_h == NULL) {
            err_and_abort("ERROR: failed to allocate host buffer for repeats");
        }
        if (!amgx_ok(AMGX_vector_download(x, x_h), "AMGX_vector_download")) err_and_abort("AMGX initial x download failed");
        if (rank == 0) {
            printf("Running for %d repeats\n", nrepeats);
        }
    }

    for (r = 0; r < nrepeats; ++r) {
        if (r > 0) {
            if (!amgx_ok(AMGX_vector_upload(x, n, block_dimx, x_h), "AMGX_vector_upload(reset x)")) err_and_abort("AMGX x reset upload failed");
        }
        MPI_Barrier(amgx_mpi_comm);
        if (!amgx_ok(AMGX_solver_setup(solver, A), "AMGX_solver_setup")) err_and_abort("AMGX solver setup failed");
        MPI_Barrier(amgx_mpi_comm);
        if (!amgx_ok(AMGX_solver_solve(solver, b, x), "AMGX_solver_solve")) err_and_abort("AMGX solver solve failed");
        if (!amgx_ok(AMGX_solver_get_status(solver, &status), "AMGX_solver_get_status")) err_and_abort("AMGX solver status query failed");

        if (status == AMGX_SOLVE_DIVERGED) {
            print_callback("***Solver Diverged\n", 0);
        } else if (status == AMGX_SOLVE_NOT_CONVERGED) {
            print_callback("***Solver Did Not Converge\n", 0);
        } else if (status == AMGX_SOLVE_FAILED) {
            print_callback("***Solver Failed\n", 0);
        }
    }

    free(x_h);
    free(partition_vector);
    AMGX_solver_destroy(solver);
    AMGX_vector_destroy(x);
    AMGX_vector_destroy(b);
    AMGX_matrix_destroy(A);
    AMGX_resources_destroy(rsrc);
    AMGX_config_destroy(cfg);
    AMGX_finalize();
    MPI_Finalize();
    return 0;
}

