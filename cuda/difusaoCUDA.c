%%gpu

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define N 2000
#define T 500
#define D 0.1
#define DELTA_T 0.01
#define DELTA_X 1.0
#define BLOCK_SIZE 32

__global__ void diffusion_kernel(double *d_C, double *d_C_new, int n, double *d_diff_sum) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    int bid = blockIdx.x + blockIdx.y * gridDim.x;
    int idx_diff = bid * (blockDim.x * blockDim.y) + tid;

    __shared__ double diff_local[BLOCK_SIZE * BLOCK_SIZE];
    diff_local[tid] = 0.0;

    if (i > 0 && i < n-1 && j > 0 && j < n-1) {
        int idx = i * n + j;
        int idx_up = (i-1) * n + j;
        int idx_down = (i+1) * n + j;
        int idx_left = i * n + (j-1);
        int idx_right = i * n + (j+1);

        d_C_new[idx] = d_C[idx] + D * DELTA_T * (
            (d_C[idx_up] + d_C[idx_down] + d_C[idx_left] + d_C[idx_right] - 4 * d_C[idx])
            / (DELTA_X * DELTA_X)
        );

        diff_local[tid] = fabs(d_C_new[idx] - d_C[idx]);
    }

    __syncthreads();

    for (int s = blockDim.x * blockDim.y / 2; s > 0; s >>= 1) {
        if (tid < s) {
            diff_local[tid] += diff_local[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        d_diff_sum[bid] = diff_local[0];
    }
}

__global__ void copy_kernel(double *d_C, double *d_C_new, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i > 0 && i < n-1 && j > 0 && j < n-1) {
        int idx = i * n + j;
        d_C[idx] = d_C_new[idx];
    }
}

void diff_eq_cuda(double *h_C, double *h_C_new, int n) {
    double *d_C, *d_C_new, *d_diff_sum;
    size_t size = n * n * sizeof(double);

    cudaMalloc(&d_C, size);
    cudaMalloc(&d_C_new, size);
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((n + block.x - 1) / block.x, (n + block.y - 1) / block.y);
    int num_blocks = grid.x * grid.y;
    cudaMalloc(&d_diff_sum, num_blocks * sizeof(double));

    cudaMemcpy(d_C, h_C, size, cudaMemcpyHostToDevice);

    double *h_diff_sum = (double*)malloc(num_blocks * sizeof(double));

    for (int t = 0; t < T; t++) {
        diffusion_kernel<<<grid, block>>>(d_C, d_C_new, n, d_diff_sum);

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("CUDA error: %s\n", cudaGetErrorString(err));
            break;
        }

        cudaMemcpy(h_diff_sum, d_diff_sum, num_blocks * sizeof(double), cudaMemcpyDeviceToHost);

        double difmedio = 0.0;
        for (int i = 0; i < num_blocks; i++) {
            difmedio += h_diff_sum[i];
        }
        difmedio /= ((n-2) * (n-2));

        copy_kernel<<<grid, block>>>(d_C, d_C_new, n);

        if ((t % 100) == 0) {
            printf("CUDA - Iteracao %d - Diferenca: %g\n", t, difmedio);
        }
    }

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    free(h_diff_sum);
    cudaFree(d_C);
    cudaFree(d_C_new);
    cudaFree(d_diff_sum);
}

int main() {
    double seq_time_s = 13.048;

    double *h_C = (double*)calloc(N * N, sizeof(double));
    double *h_C_new = (double*)calloc(N * N, sizeof(double));

    h_C[N/2 * N + N/2] = 1.0;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    diff_eq_cuda(h_C, h_C_new, N);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    float cuda_time_ms = 0;
    cudaEventElapsedTime(&cuda_time_ms, start, stop);
    float cuda_time_s = cuda_time_ms / 1000.0;
    printf("\nTempo CUDA: %.3f segundos\n", cuda_time_s);

    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);
    int total_threads = grid.x * grid.y * block.x * block.y;

    double speedup = seq_time_s / cuda_time_s;
    printf("Speedup: %.2f\n", speedup);

    free(h_C);
    free(h_C_new);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}