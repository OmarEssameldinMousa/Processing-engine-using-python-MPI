from mpi4py import MPI
import sys
import time

def find_keyword_positions(text, keyword):
    positions = []
    start = 0
    klen = len(keyword)
    while True:
        pos = text.find(keyword, start)
        if pos == -1:
            break
        positions.append(pos)
        start = pos + 1  # Move past current match
    return positions

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        # Read input file and keyword
        input_file = sys.argv[1]
        keyword = sys.argv[2]
        
        with open(input_file, 'r') as f:
            text = f.read()
        
        klen = len(keyword)
        text_len = len(text)
        
        # Split text into chunks with overlap
        chunk_size = (text_len + size - 1) // size  # Ceiling division
        overlap = klen - 1
        chunks = []
        starts = []
        
        for i in range(size):
            start_idx = max(0, i * chunk_size - overlap)
            end_idx = min((i + 1) * chunk_size, text_len)
            chunk = text[start_idx:end_idx]
            chunks.append(chunk)
            starts.append(start_idx)
        
        start_time = time.time()
    else:
        chunks = None
        starts = None
        keyword = None

    # Broadcast keyword and scatter chunks
    keyword = comm.bcast(keyword, root=0)
    local_chunk = comm.scatter(chunks, root=0)
    start_idx = comm.scatter(starts, root=0)

    # Find positions in local chunk
    local_positions = find_keyword_positions(local_chunk, keyword)
    global_positions = [pos + start_idx for pos in local_positions]

    # Gather all positions
    all_positions = comm.gather(global_positions, root=0)

    if rank == 0:
        # Deduplicate and sort
        seen = set()
        unique_positions = []
        for sublist in all_positions:
            for pos in sublist:
                if pos not in seen:
                    seen.add(pos)
                    unique_positions.append(pos)
        unique_positions.sort()
        
        total_time = time.time() - start_time
        
        # Output results
        print(len(unique_positions))
        print(','.join(map(str, unique_positions)))
        print(f"{total_time:.4f}")

if __name__ == "__main__":
    main()