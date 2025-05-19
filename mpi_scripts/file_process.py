from mpi4py import MPI
import sys
import time
import numpy as np

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        with open(sys.argv[1], 'r') as f:
            text = f.read()
        words = text.split()
        chunks = np.array_split(words, size)
    else:
        chunks = None

    # Distribute chunks to all processes
    local_chunk = comm.scatter(chunks, root=0)

    # Start timing after distribution
    comm.Barrier()
    start_time = time.time()

    # Local processing
    local_word_count = len(local_chunk)
    local_unique = set(local_chunk)

    # Reduce total word count
    total_words = comm.reduce(local_word_count, op=MPI.SUM, root=0)

    # Gather unique words
    local_unique_list = list(local_unique)
    gathered_lists = comm.gather(local_unique_list, root=0)

    # Calculate execution time
    comm.Barrier()
    exec_time = time.time() - start_time

    if rank == 0:
        # Combine unique words
        all_unique = set()
        for lst in gathered_lists:
            all_unique.update(lst)
        total_unique = len(all_unique)
        
        # Output results
        print(total_words)
        print(total_unique)
        print(f"{exec_time:.4f}")

if __name__ == "__main__":
    main()