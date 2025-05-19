# Processing-engine-using-python-MPI
## Overview

This project is a parallel processing engine built with Python and MPI (Message Passing Interface), featuring a Streamlit web interface for interactive experimentation. It demonstrates distributed computing for common data processing tasks, leveraging the `mpi4py` library for parallel execution.

## Features

- **Sorting**: Parallel odd-even transposition sort for large integer arrays.
- **File Processing**: Distributed word count and unique word extraction from text files.
- **Image Processing**: Parallel grayscale and blur filters for images.
- **Text Search**: Fast, parallelized keyword search in large text files.
- **Matrix Multiplication**: Distributed multiplication of large matrices.

## Directory Structure

```
mpi_streamlit_app/
├── app.py                # Streamlit web interface
├── script.py             # Data generation utility (integers, words, matrices)
├── mpi_scripts/
│   ├── file_process.py   # Parallel file processing
│   ├── image_process.py  # Parallel image processing
│   ├── matrix_mul.py     # Parallel matrix multiplication
│   ├── sort.py           # Parallel sorting
│   └── text_search.py    # Parallel text search
```

## Getting Started

### Prerequisites

- Python 3.7+
- [mpi4py](https://mpi4py.readthedocs.io/)
- [Streamlit](https://streamlit.io/)
- OpenMPI or MPICH
- numpy, pandas, matplotlib, opencv-python, faker

Install dependencies:

```bash
pip install -r requirements.txt
```

### Running the App

Start the Streamlit interface:

```bash
streamlit run app.py
```

The web UI allows you to select tasks, upload/generate data, and visualize results and performance metrics.

### Running Scripts Manually

Each script in `mpi_scripts/` can be run directly with `mpirun`. For example:

```bash
mpirun -n 4 python3 mpi_scripts/sort.py input.txt
```

Use `script.py` to generate test data:

```bash
python3 script.py int --count 1000 --min 0 --max 100 --output numbers.txt
python3 script.py matrix --width 100 --height 100 --output matrix.csv
python3 script.py word --count 3000000 --output words.txt
```

## Example Tasks

- **Sorting**: Upload or generate a list of numbers, run parallel sort, and compare with sequential results.
- **Matrix Multiplication**: Upload or generate two matrices, run distributed multiplication, and validate results.
- **Image Processing**: Upload an image, select a filter, and process it in parallel.

## Notes

- All parallel tasks use MPI for distributing work across processes.
- The Streamlit app manages temporary files and validates results against sequential computation for correctness.
- Performance metrics (speedup, efficiency) are displayed for each task.

## Common Bugs
matrix mulitplication is under development