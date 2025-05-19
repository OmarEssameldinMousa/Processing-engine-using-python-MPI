from mpi4py import MPI
import numpy as np
import sys
import time

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    result = None

    try:
        # Read input parameters
        if rank == 0:
            if len(sys.argv) != 4:
                raise ValueError("Invalid arguments. Usage: python matrix_mul.py <A.csv> <B.csv> <output.csv>")
            
            mat_a = np.loadtxt(sys.argv[1], delimiter=',')
            mat_b = np.loadtxt(sys.argv[2], delimiter=',')
            
            if mat_a.shape[1] != mat_b.shape[0]:
                raise ValueError(f"Matrix dimension mismatch: {mat_a.shape} vs {mat_b.shape}")
            
            rows, cols = mat_a.shape[0], mat_b.shape[1]
            split_sizes = [mat_a.shape[0] // size + (1 if x < mat_a.shape[0] % size else 0) for x in range(size)]
            displacements = np.insert(np.cumsum(split_sizes[:-1]), 0)
        else:
            mat_a = mat_b = None
            split_sizes = displacements = None
            rows = cols = 0

        # Broadcast common data
        split_sizes = comm.bcast(split_sizes, root=0)
        displacements = comm.bcast(displacements, root=0)
        mat_b = comm.bcast(mat_b, root=0)
        rows = comm.bcast(rows, root=0)
        cols = comm.bcast(cols, root=0)

        # Create buffer for local matrix part
        local_a = np.zeros((split_sizes[rank], mat_a.shape[1])) if rank == 0 else np.zeros((split_sizes[rank], mat_b.shape[0]))
        local_result = np.zeros((split_sizes[rank], cols))

        # Scatter matrix A
        comm.Scatterv([mat_a, split_sizes, displacements, MPI.DOUBLE], local_a, root=0)

        # Local multiplication
        local_result = np.dot(local_a, mat_b)

        # Gather results
        if rank == 0:
            result = np.zeros((rows, cols))
        
        comm.Gatherv(local_result, [result, split_sizes, displacements, MPI.DOUBLE], root=0)

        # Save results
        if rank == 0:
            np.savetxt(sys.argv[3], result, delimiter=',', fmt='%.2f')
            print(f"{time.time() - start_time:.4f}")

    except Exception as e:
        if rank == 0:
            print(f"ERROR: {str(e)}")
        comm.Abort(1)

if __name__ == "__main__":
    start_time = time.time()
    main()