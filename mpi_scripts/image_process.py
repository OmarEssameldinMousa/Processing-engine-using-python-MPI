from mpi4py import MPI
import sys
import cv2
import numpy as np
import time

def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def blur(img, kernel_size=5):
    return cv2.blur(img, (kernel_size, kernel_size))

def process_image(args):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        # Read image and prepare chunks
        img = cv2.imread(args['input_path'])
        filter_type = args['filter_type']
        kernel_size = args.get('kernel_size', 5)
        
        # Split image into horizontal chunks
        height = img.shape[0]
        chunk_height = height // size
        chunks = []
        for i in range(size):
            start = i * chunk_height
            end = (i+1) * chunk_height if i != size-1 else height
            chunks.append(img[start:end])
    else:
        chunks = None
        filter_type = None
        kernel_size = None

    # Broadcast processing parameters
    filter_type = comm.bcast(filter_type, root=0)
    kernel_size = comm.bcast(kernel_size, root=0)
    
    # Scatter image chunks
    local_chunk = comm.scatter(chunks, root=0)

    # Process local chunk
    start_time = time.time()
    if filter_type == 'grayscale':
        processed = grayscale(local_chunk)
    elif filter_type == 'blur':
        processed = blur(local_chunk, kernel_size)
    else:
        processed = local_chunk.copy()
    processing_time = time.time() - start_time

    # Gather processed chunks
    gathered_chunks = comm.gather(processed, root=0)

    if rank == 0:
        # Combine processed chunks
        final_image = np.vstack(gathered_chunks)
        total_time = time.time() - start_time
        
        # Save output
        cv2.imwrite(args['output_path'], final_image)
        return total_time, processing_time
    return None, None

if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    args = {
        'input_path': sys.argv[1],
        'output_path': sys.argv[2],
        'filter_type': sys.argv[3],
        'kernel_size': int(sys.argv[4]) if len(sys.argv) > 4 else 5
    }
    
    total_time, proc_time = process_image(args)
    if rank == 0:
        print(f"{total_time:.4f}")
        print(f"{proc_time:.4f}")