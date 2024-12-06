# GPU-Accelerated QR Algorithm for Eigenvalue Computation Using CUDA

**Abstract**  
Matrix eigenvalue computation is a critical operation in many scientific and engineering applications. Traditionally, this process can be time-consuming on conventional CPUs. The project explores the use of Graphics Processing Units (GPUs) to accelerate the QR algorithm, a classic method for extracting eigenvalues of a square matrix. By leveraging NVIDIA’s CUDA platform and associated libraries (cuBLAS and cuSOLVER), we implemented a GPU-accelerated QR factorization method. Although initial experiments were conducted on small matrices, the approach demonstrated notable speed improvements compared to naive CPU methods. This accelerated performance suggests that GPU-enabled QR algorithms can significantly reduce computational time, enabling faster simulations and analyses in fields such as computational physics, data science, and machine learning.

---

**Introduction**  
Eigenvalue computations lie at the heart of numerous computational problems, including stability analysis, vibration modes in mechanical structures, and principal component analysis in data science. Traditional CPU-based eigenvalue algorithms can become computational bottlenecks for large matrices. To address this challenge, we implemented a GPU-accelerated QR algorithm to approximate eigenvalues. By harnessing the parallel processing capabilities of modern GPUs, the goal was to decrease computation time and lay the groundwork for scalable, high-performance eigenvalue decomposition techniques.

**Background**  
The QR algorithm has been a cornerstone of numerical linear algebra for decades, providing a stable and widely used method for eigenvalue computation. Historically, researchers have optimized the algorithm’s efficiency on CPUs, but recent trends point towards GPU acceleration to handle the ever-increasing size and complexity of data. NVIDIA’s CUDA framework and its associated libraries (cuBLAS and cuSOLVER) have simplified GPU programming. State-of-the-art approaches now use these tools to offload linear algebra computations to the GPU, achieving significant speedups over CPU-only implementations.

**Methodology/Approach**  
1. **Matrix Setup:**  
   We initialized a test matrix (A) on the host (CPU) memory and transferred it to the device (GPU).

2. **QR Factorization Using cuSOLVER:**  
   The QR factorization (A = Q R) was performed entirely on the GPU using `cusolverDnDgeqrf` and `cusolverDnDorgqr`. This eliminated costly CPU-GPU data transfers during the decomposition step.

3. **Iterative Eigenvalue Approximation:**  
   After obtaining (Q) and (R), we updated the matrix as (A = RQ) and repeated the QR step. In theory, repeated iterations drive (A) toward an upper triangular form, whose diagonal elements approximate the eigenvalues. Although we fixed a small number of iterations for demonstration (e.g., 10), the method can be extended until convergence.

4. **Performance Measurement:**  
   We measured the execution time using C++’s `std::chrono` utilities. By comparing different matrix sizes and iteration counts, we observed the time impact of offloading computations to the GPU.

---

**Results**  
To assess performance, we benchmarked the GPU-accelerated QR eigenvalue computation against a Python 3.10 implementation for various matrix sizes (N) and recorded both runtime and approximate memory usage. The GPU implementation was compiled with `nvc++` (CUDA), while the CPU-only reference was implemented in Python. The results are shown in the table below:

| Dimension (N x N) | Memory (GB) | CUDA (nvc++) Time (s) | Python 3.10 Time (s) |
|-------------------|-------------|-----------------------|----------------------|
| 10                | 0.000001    | 0.1324                | 0.0037              |
| 50                | 0.000019    | 0.0720                | 0.0026              |
| 100               | 0.000075    | 0.0892                | 1.2164              |
| 500               | 0.001863    | 0.2542                | 5.4108              |
| 1000              | 0.007451    | 0.6621                | 20.1788             |
| 5000              | 0.186265    | 37.2650               | 328.1599            |

**Analysis of Results:**  
- **Small Matrices (N ≤ 50):** For very small matrices, Python’s simple implementation outperforms the GPU code. Python’s runtimes (0.0026–0.0037 s) are faster than CUDA’s (0.0720–0.1324 s) due to negligible parallelization overhead and very fast CPU operations at this scale.
  
- **Medium Matrices (N = 100 to 500):** As matrix sizes increase, the GPU implementation begins to show significant advantages. At (N = 100), CUDA completes the task in 0.0892 s, while Python takes 1.2164 s. The disparity widens as (N) grows further: at (N = 500), CUDA is approximately 20 times faster than Python.
  
- **Large Matrices (N = 1000 and Above):** For (N = 1000) and larger, the GPU acceleration becomes even more apparent. At (N = 1000), the GPU solution (0.6621 s) is about 30 times faster than Python (20.1788 s). At (N = 5000), CUDA completes the operation in about 37.2650 s, compared to Python’s 328.1599 s. While the absolute runtime for the GPU version does increase substantially at this scale, the relative speedup over Python is significant.

**Memory Usage:**  
As expected, memory usage scales approximately with (N^2). Even at (N = 5000), the allocated memory (0.186265 GB) remains manageable for most modern GPU systems, indicating feasibility for even larger scales with appropriate hardware resources.

---

**Discussion**  
The results confirm that GPU acceleration via CUDA provides substantial performance benefits over a Python-only implementation for medium to large problem sizes. Initially, for very small matrices, the overhead of GPU operations does not pay off, and Python outperforms the CUDA code. This is largely because the overhead of data transfer and kernel launches cannot be amortized over a sufficiently large computation. However, as matrix dimensions increase, these overheads become negligible compared to the raw computational workload, allowing the GPU’s parallelism to dominate.

For matrices of size (N = 100) and above, the GPU implementation rapidly outpaces Python, offering faster turnaround times that are critical in fields requiring large-scale eigenvalue computations. This aligns with state-of-the-art practices where large linear algebra problems are routinely offloaded to GPUs. The time savings are substantial, potentially enabling more frequent simulations, larger problem sizes, and faster iterative design loops in engineering, data science, and scientific research.

With additional resources or further optimization, the GPU code could be refined to reduce overhead at smaller scales (e.g., via improved data transfer strategies or batched operations). Moreover, exploring advanced methods in the cuSOLVER and cuBLAS libraries or applying mixed-precision computations may yield even greater speedups.

In conclusion, while Python offers simplicity and excellent performance at very small scales, GPUs begin to deliver significant advantages as matrix dimensions grow. This crossover in performance emphasizes the importance of problem size consideration when choosing computational strategies for eigenvalue problems.

**References**  
"QR Algorithm." Wikipedia, Wikimedia Foundation, 2 Dec. 2024, en.wikipedia.org/wiki/QR_algorithm. Accessed 6 Dec. 2024.  
“cuBLAS.” NVIDIA Developer, developer.nvidia.com/cublas. Accessed 06 Dec. 2024.  
“cuSOLVER.” NVIDIA Developer, developer.nvidia.com/cusolver. Accessed 06 Dec. 2024.  
“CUDA Toolkit Documentation 12.6 Update 3.” CUDA Toolkit Documentation 12.6 Update 3, 20 Nov. 2024, docs.nvidia.com/cuda/. Accessed 06 Dec. 2024.
