#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <cublas_v2.h>
#include <iostream>
#include <string>
#include <cstdlib>
#include <chrono>
// #define PSIZE 3
cusolverDnHandle_t cusolverH = NULL;
cublasHandle_t cublasH = NULL;

// global clock timer
auto clk = std::chrono::high_resolution_clock::now();

void start_timer(){
    clk = std::chrono::high_resolution_clock::now();
}

void end_timer(std::string func){
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - clk);
    std::cout<<func<<" took "<<1.0e-9 * duration.count()<<" seconds\n";
}

__global__ void extractR(double *d_A, double *d_R, int m, int n) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < m && col < n) {
        if (row <= col) {
            d_R[row + col * m] = d_A[row + col * m];
        } else {
            d_R[row + col * m] = 0.0;
        }
    }
}


void printMatrix(double *mat, int m, int n){
    for(int i=0;i<m;i++){
        for(int j=0;j<n;j++){
            printf("%lf ",mat[i*n+j]);
        }
        printf("\n");
    }
    printf("\n");
}

__global__ void printMatrixKernel(double* matrix, int width, int height) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int idy = threadIdx.y + blockIdx.y * blockDim.y;
    if (idx < width && idy < height) {
        printf("Element at [%d, %d]: %f\n", idy, idx, matrix[idy * width + idx]);
    }
}

void transposeMatrix(double* matrix, int m, int n) {
    for (int i = 0; i < m; ++i) {
        for (int j = i + 1; j < n; ++j) {
            std::swap(matrix[i * n + j], matrix[j * n + i]);
        }
    }
}

double* qr(double *d_A, double *d_R, double *d_B, int m, int n, int lda){

    // QR factorization
    double *d_Tau = NULL;
    cudaMalloc((void**)&d_Tau, sizeof(double) * min(m, n));

    int Lwork = 0;
    cusolverDnDgeqrf_bufferSize(cusolverH, m, n, d_A, lda, &Lwork);

    double *d_Work = NULL;
    cudaMalloc((void**)&d_Work, sizeof(double) * Lwork);

    int *devInfo = NULL;
    cudaMalloc((void**)&devInfo, sizeof(int));

    cusolverDnDgeqrf(cusolverH, m, n, d_A, lda, d_Tau, d_Work, Lwork, devInfo);

    int info_gpu = 0;
    cudaMemcpy(&info_gpu, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
    if (info_gpu != 0) {
        fprintf(stderr, "Error: QR factorization failed\n");
        return nullptr;
    }

    // Extract R
    dim3 blockSize(16, 16);
    dim3 gridSize((n + blockSize.x - 1) / blockSize.x,
                  (m + blockSize.y - 1) / blockSize.y);

    extractR<<<gridSize, blockSize>>>(d_A, d_R, m, n);
    cudaDeviceSynchronize();

    // Generate Q
    int Lwork_orgqr = 0;
    cusolverDnDorgqr_bufferSize(cusolverH, m, n, n, d_A, lda, d_Tau, &Lwork_orgqr);

    if (Lwork_orgqr > Lwork) {
        cudaFree(d_Work);
        Lwork = Lwork_orgqr;
        cudaMalloc((void**)&d_Work, sizeof(double) * Lwork);
    }

    cusolverDnDorgqr(cusolverH, m, n, n, d_A, lda, d_Tau, d_Work, Lwork, devInfo);

    cudaMemcpy(&info_gpu, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
    if (info_gpu != 0) {
        fprintf(stderr, "Error: Generation of Q failed\n");
        return nullptr;
    }

    double alpha = 1.0;
    double beta = 0.0;

    cublasDgemm(
        cublasH,
        CUBLAS_OP_N, CUBLAS_OP_N,
        m, n, n,
        &alpha,
        d_R, lda,
        d_A, lda,
        &beta,
        d_B, lda
    );
    cudaFree(d_Tau);
    cudaFree(d_Work);
    cudaFree(devInfo);

    return d_B;
}


void eig(double *d_A, double *d_R, double *d_B, int m, int n, int lda, double *h_Q, double *h_R){

    double diff  = 1.0;
    int iters = 10;
    d_A = qr(d_A,d_R,d_B,m,n,lda);
    double leig,temp;
    cudaMemcpy(&leig, d_A+(m*n-1), sizeof(double), cudaMemcpyDeviceToHost);
    while(iters--){
        d_A = qr(d_A,d_R,d_B,m,n,lda);
        cudaMemcpy(&temp, d_A+(m*n-1), sizeof(double), cudaMemcpyDeviceToHost);
        diff = abs(leig - temp);
        leig = temp;
    }


    cudaMemcpy(h_Q, d_A, sizeof(double) * m * n, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_R, d_R, sizeof(double) * m * n, cudaMemcpyDeviceToHost);

    transposeMatrix(h_Q,m,n);
    transposeMatrix(h_R,m,n);
    double check = 0.0;
    for(int i=0;i<m;i++){
        check+=h_Q[i*m+i];
    }
    printf("\ncheck = %lf\n",check);
}

int main(int argc, char** argv) {
    cusolverDnCreate(&cusolverH);
    cublasCreate(&cublasH);

    int m = PSIZE;
    int n = PSIZE;
    int lda = m;

    printf("size = %lf GB\n",(double)m*n*sizeof(double)/1024/1024/1024);
    // Host memory allocation
    double *h_A = (double *)malloc(m * n * sizeof(double));
    double *h_Q = (double *)malloc(m * n * sizeof(double));
    double *h_R = (double *)malloc(m * n * sizeof(double));


    


    // Initialize h_A with your matrix data
    for (int i = 0; i < m * n; ++i) {
        h_A[i] = i;
    }
    transposeMatrix(h_A,m,n);
    start_timer();
    // Device memory allocation
    double *d_A = NULL;
    cudaMalloc((void**)&d_A, sizeof(double) * lda * n);
    cudaMemcpy(d_A, h_A, sizeof(double) * lda * n, cudaMemcpyHostToDevice);
    double *d_R = NULL;
    cudaMalloc((void**)&d_R, sizeof(double) * m * n);
    double *d_B = NULL;
    cudaMalloc((void**)&d_B, sizeof(double) * m * n);

    eig(d_A,d_R,d_B,m,n,lda,h_Q,h_R);
    
    end_timer("computation");

    free(h_A);
    free(h_Q);
    free(h_R);

    cusolverDnDestroy(cusolverH);
    cublasDestroy(cublasH);
    cudaDeviceReset();

    return EXIT_SUCCESS;
}
