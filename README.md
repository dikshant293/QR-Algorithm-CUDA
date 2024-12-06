# GPU-Accelerated QR Eigenvalue Computation

## Overview
This project demonstrates a GPU-accelerated QR-based eigenvalue computation using NVIDIA's CUDA, cuBLAS, and cuSOLVER libraries. It performs repeated QR factorizations on a given matrix and approximates its eigenvalues. The code is written in C++ and uses `nvc++` (from the NVIDIA HPC SDK) as the compiler.

## Requirements

### Hardware
- An NVIDIA GPU with CUDA capability.

### Software
- **CUDA Toolkit:** Required for `cudart`, `cublas`, and `cusolver` libraries.
- **NVIDIA HPC SDK** (for `nvc++`):  
  [NVIDIA HPC SDK Download](https://developer.nvidia.com/hpc-sdk)
- **BLAS and LAPACK:**  
  Commonly available through system package managers.  
  For example, on Ubuntu/Debian:
  ```bash
  sudo apt-get update
  sudo apt-get install libblas-dev liblapack-dev
  ```
- **OpenBLAS:**  
  Also often available through the package manager:
  ```bash
  sudo apt-get install libopenblas-dev
  ```
- **OpenMP:**  
  Typically included by default in modern compilers. The NVIDIA HPC SDK supports OpenMP offloading if needed.

## Installation Steps

1. **Install NVIDIA HPC SDK:**
   Follow instructions at [NVIDIA HPC SDK Downloads](https://developer.nvidia.com/hpc-sdk) to install `nvc++`.

2. **Install CUDA Toolkit:**
   If not already available, follow instructions at [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-downloads).

3. **Install BLAS, LAPACK, and OpenBLAS:**
   Using your system's package manager:
   ```bash
   sudo apt-get install libblas-dev liblapack-dev libopenblas-dev
   ```

4. **Verify Paths:**
   Ensure that the CUDA and HPC SDK environments are set up. Typically, this involves adding paths to your `.bashrc` or shell environment. For example:
   ```bash
   export PATH=/path/to/hpc_sdk/Linux_x86_64/<version>/compilers/bin:$PATH
   export PATH=/usr/local/cuda/bin:$PATH
   export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
   ```

## Building the Code

The `Makefile` is configured to compile `main.cu` using `nvc++` and link against the required libraries.

- To build with the default matrix size (N=3):
  ```bash
  make
  ```

- To build and run for a custom matrix size, for example `N=50`:
  ```bash
  make run N=50
  ```

This will:
1. Compile the code with `-DPSIZE=50`.
2. Run the generated executable `eigenvalue_test`.

## Running the Code

- **Default Run (N=3):**
  ```bash
  make run
  ```
  This runs the code with `N=3`.

- **Custom Size:**
  ```bash
  make run N=<size>
  ```
  For instance, for a 50x50 matrix:
  ```bash
  make run N=50
  ```

### Example Output (50x50 Matrix)
```
$ make run N=50
nvc++ main.cu -lcudart -lcublas -lcusolver -llapacke -llapack -lblas -use_fast_math -Xcompiler -fopenmp -lopenblas -O3 --diag_suppress set_but_not_used -o eigenvalue_test -DPSIZE=50
./eigenvalue_test
size = 0.000019 GB

check = 62475.000000
computation took 0.13757 seconds
```

- **`check`**: A diagnostic value derived from the computed `Q` matrix (sum of its diagonal elements), used as a simple convergence check.
- **`computation took ... seconds`**: Time measured for the QR computation and eigenvalue iteration steps.

## Notes
- Adjust `PSIZE` to scale the problem size and observe performance differences.
- For large matrices, GPU acceleration yields more pronounced performance benefits.
- If you encounter linking errors, ensure all library paths (e.g., `-L` flags) and environment variables are correctly set.

### Python Implementation

---

To check performance of serial python code, run the `py-eigenval.ipynb` notebook. Adjust the `n` variable to vary the size of the matrix.