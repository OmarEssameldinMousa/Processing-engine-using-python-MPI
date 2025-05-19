import streamlit as st
import subprocess
import tempfile
import os
import time
import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO
import random
import numpy as np
import cv2

# Configure page
st.set_page_config(page_title="MPI Parallel Processing", layout="wide")

def main():
    st.title("MPI Parallel Processing Platform")
    
    # Sidebar controls
    st.sidebar.header("Configuration")
    num_procs = st.sidebar.number_input("Number of Processes", 1, 16, 4)
    task = st.sidebar.selectbox("Select Task", [
        "Sorting", 
        "File Processing", 
        "Image Processing",
        "Text Search",
        "Matrix Multiplication"
    ])

    # Task handling
    if task == "Sorting":
        handle_sorting(num_procs)
    elif task == "File Processing":
        handle_file_processing(num_procs)
    elif task == "Image Processing":
        handle_image_processing(num_procs)
    elif task == "Text Search":
        handle_text_search(num_procs)
    elif task == "Matrix Multiplication":
        handle_matrix_mul(num_procs)

def run_mpi_command(command):
    start_time = time.time()
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    exec_time = time.time() - start_time
    return result, exec_time

# Sorting Task
def handle_sorting(num_procs):
    st.header("Odd-Even Transposition Sort")
    
    col1, col2 = st.columns(2)
    input_path = None  # Initialize input_path
    
    with col1:
        input_type = st.radio("Input Type", ["Manual Entry", "File Upload", "Random Generation"])
        numbers = []
        
        if input_type == "Manual Entry":
            input_str = st.text_input("Enter numbers (comma-separated)")
            if input_str:
                numbers = list(map(int, input_str.split(',')))
        elif input_type == "Random Generation":
            num_count = st.number_input("How many numbers?", min_value=1, max_value=100000, value=100, step=1)
            min_val = st.number_input("Minimum Value", value=-100, step=1)
            max_val = st.number_input("Maximum Value", value=100, step=1)
            if 'rand_numbers' not in st.session_state:
                st.session_state.rand_numbers = []
            if st.button("Generate Random Numbers"):
                st.session_state.rand_numbers = [random.randint(min_val, max_val) for _ in range(num_count)]
            numbers = st.session_state.rand_numbers
            if numbers:
                st.subheader("Generated Numbers Preview")
                st.dataframe(pd.DataFrame(numbers[:20], columns=["Number"]))
                if len(numbers) > 20:
                    st.caption(f"Showing first 20 of {len(numbers)} numbers.")
        else:
            file = st.file_uploader("Upload numbers file", type=["txt", "csv"])
            if file:
                numbers = list(map(int, file.read().decode().split(',')))
    
    if numbers and st.button("Run Sorting"):
        try:
            # Save to temp file
            with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
                f.write(','.join(map(str, numbers)))
                input_path = f.name
            
            # Run MPI
            mpi_cmd = f"mpirun -n {num_procs} --oversubscribe python3 mpi_scripts/sort.py {input_path}"
            result, _ = run_mpi_command(mpi_cmd)
            
            # Display results
            with col2:
                if result.returncode == 0:
                    output = result.stdout.split('\n')
                    sorted_nums = list(map(int, output[0].split(',')))
                    mpi_time = float(output[1])
                    
                    st.subheader("Results")
                    st.write("Sorted Numbers (first 20):")
                    st.dataframe(pd.DataFrame(sorted_nums[:20], columns=["Sorted Number"]))
                    if len(sorted_nums) > 20:
                        st.caption(f"Showing first 20 of {len(sorted_nums)} numbers.")
                    
                    # Performance comparison
                    seq_start = time.time()
                    # Serial Odd-Even Transposition Sort
                    def odd_even_transposition_sort(arr):
                        n = len(arr)
                        arr = arr.copy()
                        for i in range(n):
                            for j in range(1 if i % 2 else 0, n - 1, 2):
                                if arr[j] > arr[j + 1]:
                                    arr[j], arr[j + 1] = arr[j + 1], arr[j]
                        return arr
                    odd_even_transposition_sort(numbers)
                    seq_time = time.time() - seq_start
                    
                    perf_data = pd.DataFrame({
                        'Metric': ['Sequential', 'Parallel'],
                        'Time (s)': [seq_time, mpi_time]
                    })
                    
                    st.bar_chart(perf_data.set_index('Metric'))
                    
                    st.metric("Speedup", f"{seq_time/mpi_time:.2f}x")
                    st.metric("Parallel Efficiency", f"{(seq_time/(num_procs*mpi_time))*100:.1f}%")
                else:
                    st.error(f"Error: {result.stderr}")
        finally:
            # Clean up temporary file
            if input_path and os.path.exists(input_path):
                os.remove(input_path)

# file processing task
def handle_file_processing(num_procs):
    st.header("File Processing: Word Count & Unique Words")
    
    col1, col2 = st.columns(2)
    input_path = None
    
    with col1:
        file = st.file_uploader("Upload text file", type=["txt"])
    
    if file and st.button("Process File"):
        # Save to temp file (keep reference until processing completes)
        with tempfile.NamedTemporaryFile(mode='wb', delete=False) as f:
            f.write(file.read())
            input_path = f.name
        
        try:
            # Run MPI command (don't delete input file yet)
            mpi_cmd = f"mpirun -n {num_procs} --oversubscribe python3 mpi_scripts/file_process.py {input_path}"
            result, _ = run_mpi_command(mpi_cmd)
            
            if result.returncode == 0:
                output = result.stdout.strip().split('\n')
                if len(output) >= 3:
                    total_words = int(output[0])
                    total_unique = int(output[1])
                    mpi_time = float(output[2])
                    
                    # Sequential processing (using original file content)
                    seq_start = time.time()
                    words = file.getvalue().decode().split()
                    seq_total = len(words)
                    seq_unique = len(set(words))
                    seq_time = time.time() - seq_start
                    
                    # Display results
                    with col2:
                        st.subheader("Results")
                        st.write(f"**Total Words:** {total_words}")
                        st.write(f"**Unique Words:** {total_unique}")
                        
                        # Performance comparison
                        st.subheader("Performance Comparison")
                        perf_data = pd.DataFrame({
                            'Method': ['Sequential', 'Parallel'],
                            'Time (s)': [seq_time, mpi_time]
                        })
                        
                        st.bar_chart(perf_data.set_index('Method'))
                        st.write(f"**Speedup:** {seq_time/mpi_time:.2f}x")
                        st.write(f"**Efficiency:** {(seq_time/(num_procs*mpi_time))*100:.1f}%")
                        
                        # Validation
                        st.subheader("Validation")
                        if seq_total == total_words and seq_unique == total_unique:
                            st.success("Results match sequential computation!")
                        else:
                            st.error("Discrepancy detected in results!")
                else:
                    st.error("Unexpected output format from MPI process")
            else:
                st.error(f"MPI Error: {result.stderr}")
        finally:
            # Clean up temporary file
            if input_path and os.path.exists(input_path):
                os.remove(input_path)

def handle_image_processing(num_procs):
    st.header("Image Processing")
    
    input_path = None
    output_path = None
    file = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])
    filter_type = st.selectbox("Select Filter", ["grayscale", "blur"])
    kernel_size = 5
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original Image")
        # display the select image 
        if file is not None:
            file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            if img is not None:
                st.image(img, channels="BGR", caption="Selected Image", use_container_width=True)
            file.seek(0)  # Reset file pointer for later use

        if filter_type == 'blur':
            kernel_size = st.slider("Blur Kernel Size", 3, 15, 5, step=2)
    
    if file and st.button("Process Image"):
        try:
            # Save input image
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as f_in:
                f_in.write(file.read())
                input_path = f_in.name
            
            # Create output temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as f_out:
                output_path = f_out.name
            
            # Build MPI command
            mpi_cmd = (
                f"mpirun -n {num_procs} python3 mpi_scripts/image_process.py "
                f"{input_path} {output_path} {filter_type} {kernel_size}"
            )
            
            result, _ = run_mpi_command(mpi_cmd)
            
            if result.returncode == 0:
                # Load and display results
                output = result.stdout.strip().split('\n')
                total_time = float(output[0])
                proc_time = float(output[1])
                
                with col2:
                    st.subheader("Processed Image")
                    processed_img = cv2.imread(output_path)
                    st.image(processed_img, channels="BGR", use_container_width=True)
                    
                    # Performance metrics
                    st.subheader("Performance")
                    st.write(f"Total Processing Time: {total_time:.4f}s")
                    st.write(f"Computation Time: {proc_time:.4f}s")
                    st.write(f"Communication Overhead: {total_time - proc_time:.4f}s")
                    
                    # Sequential comparison
                    seq_start = time.time()
                    img = cv2.imread(input_path)
                    if filter_type == 'grayscale':
                        cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    else:
                        cv2.blur(img, (kernel_size, kernel_size))
                    seq_time = time.time() - seq_start
                    
                    st.metric("Speedup", f"{seq_time/total_time:.2f}x")
                    st.metric("Efficiency", f"{(seq_time/(num_procs*total_time))*100:.1f}%")
            else:
                st.error(f"Processing failed: {result.stderr}")
        
        finally:
            # Cleanup temporary files
            for path in [input_path, output_path]:
                if path and os.path.exists(path):
                    os.remove(path)

def handle_text_search(num_procs):
    st.header("Parallel Text Search")
    
    col1, col2 = st.columns(2)
    input_path = None
    
    with col1:
        file = st.file_uploader("Upload text file", type=["txt"])
        keyword = st.text_input("Search Keyword")
    
    if file and keyword and st.button("Search"):
        if len(keyword.strip()) == 0:
            st.error("Please enter a valid search keyword")
            return
        
        try:
            # Save to temp file
            with tempfile.NamedTemporaryFile(mode='wb', delete=False) as f:
                f.write(file.read())
                input_path = f.name
            
            # Run MPI command
            mpi_cmd = f"mpirun -n {num_procs} --oversubscribe python3 mpi_scripts/text_search.py {input_path} '{keyword}'"
            result, _ = run_mpi_command(mpi_cmd)
            
            if result.returncode == 0:
                output = result.stdout.strip().split('\n')
                if len(output) >= 3:
                    total = int(output[0])
                    positions = list(map(int, output[1].split(','))) if output[1] else []
                    mpi_time = float(output[2])
                    
                    # Sequential processing
                    seq_start = time.time()
                    content = file.getvalue().decode()
                    seq_positions = []
                    start = 0
                    while True:
                        pos = content.find(keyword, start)
                        if pos == -1:
                            break
                        seq_positions.append(pos)
                        start = pos + 1
                    seq_time = time.time() - seq_start
                    
                    # Display results
                    with col2:
                        st.subheader("Results")
                        st.write(f"**Total occurrences:** {total}")
                        
                        if total > 0:
                            st.write("**Positions (first 20):**")
                            st.write(positions[:20])
                            if len(positions) > 20:
                                st.caption(f"Showing first 20 of {len(positions)} positions")
                        
                        # Performance comparison
                        st.subheader("Performance")
                        perf_data = pd.DataFrame({
                            'Method': ['Sequential', 'Parallel'],
                            'Time (s)': [seq_time, mpi_time]
                        })
                        st.bar_chart(perf_data.set_index('Method'))
                        
                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.metric("Speedup", f"{seq_time/mpi_time:.2f}x")
                        with col_b:
                            st.metric("Efficiency", f"{(seq_time/(num_procs*mpi_time))*100:.1f}%")
                        
                        # Validation
                        if total == len(seq_positions):
                            st.success("Results match sequential count!")
                        else:
                            st.error(f"Discrepancy: Parallel {total} vs Sequential {len(seq_positions)}")
                else:
                    st.error("Unexpected output format from MPI process")
            else:
                st.error(f"MPI Error: {result.stderr}")
        finally:
            if input_path and os.path.exists(input_path):
                os.remove(input_path)

def handle_matrix_mul(num_procs):
    st.header("Distributed Matrix Multiplication")
    
    col1, col2 = st.columns(2)
    input_method = col1.radio("Input Method", ["Generate Random", "Upload CSV"])
    
    # Session state initialization
    if 'matrices' not in st.session_state:
        st.session_state.matrices = None
    if 'mul_result' not in st.session_state:
        st.session_state.mul_result = None

    # Input handling
    if input_method == "Generate Random":
        r1 = col1.number_input("Matrix A rows", 1, 1000, 4)
        c1 = col1.number_input("Matrix A columns", 1, 1000, 4)
        r2 = col1.number_input("Matrix B rows", min_value=c1, max_value=1000, value=c1)
        c2 = col1.number_input("Matrix B columns", 1, 1000, 4)
        
        if col1.button("Generate Matrices"):
            matrix_a = np.random.randint(-10, 10, size=(r1, c1))
            matrix_b = np.random.randint(-10, 10, size=(r2, c2))
            st.session_state.matrices = (matrix_a, matrix_b)
            st.session_state.mul_result = None
            
    else:
        file_a = col1.file_uploader("Upload Matrix A (CSV)", type=["csv"])
        file_b = col1.file_uploader("Upload Matrix B (CSV)", type=["csv"])
        
        if file_a and file_b:
            try:
                matrix_a = np.loadtxt(file_a, delimiter=',')
                matrix_b = np.loadtxt(file_b, delimiter=',')
                st.session_state.matrices = (matrix_a, matrix_b)
                st.session_state.mul_result = None
            except Exception as e:
                st.error(f"CSV Error: {str(e)}")

    # Display preview
    if st.session_state.matrices:
        a, b = st.session_state.matrices
        with col1:
            st.write("Matrix A Preview (5x5):")
            st.dataframe(pd.DataFrame(a[:5, :5]))
            st.write("Matrix B Preview (5x5):")
            st.dataframe(pd.DataFrame(b[:5, :5]))

    if col1.button("Run Multiplication") and st.session_state.matrices:
        a, b = st.session_state.matrices
        
        try:
            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as fa, \
                 tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as fb, \
                 tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as fout:

                # Ensure files are closed before MPI access
                np.savetxt(fa.name, a, delimiter=',', fmt='%d')
                np.savetxt(fb.name, b, delimiter=',', fmt='%d')
                a_path, b_path, out_path = fa.name, fb.name, fout.name

            # Run MPI with progress
            with st.spinner(f"Multiplying matrices using {num_procs} processes..."):
                cmd = [
                    "mpirun", "-n", str(num_procs),
                    "python3", "mpi_scripts/matrix_mul.py",
                    a_path, b_path, out_path
                ]
                
                result = subprocess.run(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=120
                )

            if result.returncode == 0:
                # Load and display results
                result_matrix = np.loadtxt(out_path, delimiter=',')
                st.session_state.mul_result = result_matrix
                
                with col2:
                    st.subheader("Results")
                    st.write("Result Preview (5x5):")
                    st.dataframe(pd.DataFrame(result_matrix[:5, :5]))
                    
                    # Performance metrics
                    mpi_time = float(result.stdout.strip())
                    seq_time = timeit(lambda: np.dot(a, b), number=1)
                    
                    st.metric("Parallel Time", f"{mpi_time:.4f}s")
                    st.metric("Speedup", f"{seq_time/mpi_time:.2f}x")
                    
                    # Validation
                    if np.allclose(result_matrix, np.dot(a, b)):
                        st.success("Result validated successfully!")
                    else:
                        st.error("Result validation failed!")

            else:
                st.error(f"MPI Error:\n{result.stderr}")

        except subprocess.TimeoutExpired:
            st.error("Calculation timed out after 2 minutes!")
        except Exception as e:
            st.error(f"Error: {str(e)}")
        finally:
            # Cleanup files
            for path in [a_path, b_path, out_path]:
                if path and os.path.exists(path):
                    os.remove(path)

# Run the app
if __name__ == "__main__":
    main()