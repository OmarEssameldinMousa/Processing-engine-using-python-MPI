from mpi4py import MPI
import sys
import time
import numpy as np

def odd_even_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(i%2, n-1, 2):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
    return arr

if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    if rank == 0:
        with open(sys.argv[1], 'r') as f:
            data = list(map(int, f.read().split(',')))
    else:
        data = None
    
    data = comm.bcast(data, root=0)
    total_size = len(data)
    chunk_size = total_size // size
    start = rank * chunk_size
    end = start + chunk_size if rank != size-1 else total_size
    
    local_data = data[start:end]
    start_time = time.time()
    
    # Local sort
    local_sorted = odd_even_sort(local_data)
    
    # Global sort
    for _ in range(size):
        if rank % 2 == 0:
            if rank+1 < size:
                recv_data = comm.sendrecv(local_sorted, dest=rank+1, source=rank+1)
                merged = sorted(local_sorted + recv_data)
                local_sorted = merged[:len(local_sorted)]
        else:
            if rank-1 >= 0:
                recv_data = comm.sendrecv(local_sorted, dest=rank-1, source=rank-1)
                merged = sorted(local_sorted + recv_data)
                local_sorted = merged[len(recv_data):]
        comm.Barrier()
    
    # Gather results
    gathered = comm.gather(local_sorted, root=0)
    
    if rank == 0:
        final = [num for chunk in gathered for num in chunk]
        exec_time = time.time() - start_time
        print(','.join(map(str, final)))
        print(f"{exec_time:.4f}")