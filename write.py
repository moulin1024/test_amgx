import numpy as np
import netCDF4 as nc
from scipy.sparse import csr_matrix
import sys
# Read from NetCDF file
# Get data directory from input argument
datadir = sys.argv[1]

dataset0 = nc.Dataset(datadir + 'helmholtz_data_out.nc', 'r')
hcsr = dataset0.groups['hcsr']
dataset = nc.Dataset(datadir + 'helmholtz_data_out.nc', 'r')
print(dataset)
# Extract variables and convert masked arrays to regular numpy arrays
row_ptr = np.array(hcsr.variables['i'][:]) - 1
col_ind = np.array(hcsr.variables['j'][:]) - 1
values = np.array(hcsr.variables['val'][:])
rhs = np.array(dataset.variables['rhs'][:])
guess = np.array(dataset.variables['guess'][:])
sol = np.array(dataset.variables['sol'][:])

# Close the netCDF file
dataset.close()

# Create CSR matrix
n = len(row_ptr) - 1  # matrix size
print(n)
matrix = csr_matrix((values, col_ind, row_ptr), shape=(n, n))

# Extract diagonal directly using scipy's method
diagonal = matrix.diagonal()

# Compute inverse of diagonal elements
diagonal_inverse = 1.0 / diagonal

# Scale the matrix by the inverse diagonal (D^-1 * A)
values_scaled = values.copy()  # Create a copy to preserve original values
rhs_scaled = rhs.copy()
for i in range(n):
    values_scaled[row_ptr[i]:row_ptr[i+1]] *= diagonal_inverse[i]
    rhs_scaled[i] *= diagonal_inverse[i]

rhs_norm = np.linalg.norm(rhs_scaled)
values_scaled /= rhs_norm
rhs_scaled /= rhs_norm
# Write all arrays to binary files
row_ptr.astype(np.int32).tofile(datadir + 'row_ptr.bin')
col_ind.astype(np.int32).tofile(datadir + 'col_ind.bin')
values_scaled.astype(np.float64).tofile(datadir + 'val.bin')  # Write scaled values
diagonal_inverse.astype(np.float64).tofile(datadir + 'dinv.bin')
rhs_scaled.astype(np.float64).tofile(datadir + 'rhs.bin')
guess.astype(np.float64).tofile(datadir + 'guess.bin')
sol.astype(np.float64).tofile(datadir + 'sol.bin')

# # Write matrix in MTX format
# with open(datadir + 'matrix.mtx', 'w') as f:
#     # Write MTX header
#     f.write('%%MatrixMarket matrix coordinate real general\n')
#     f.write(f'{n} {n} {len(values)}\n')
    
#     # Convert CSR to COO format for MTX
#     row_indices = np.zeros_like(col_ind)
#     for i in range(n):
#         row_indices[row_ptr[i]:row_ptr[i+1]] = i
    
#     # Write entries (1-based indexing for MTX format)
#     for i, j, val in zip(row_indices, col_ind, values):
#         f.write(f'{i+1} {j+1} {val}\n')

# Print some information
print(f"Matrix size: {n} x {n}")
print(f"Number of non-zeros: {len(values)}")


