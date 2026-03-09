#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(call)                                                       \
    do {                                                                        \
        cudaError_t err = call;                                                 \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,   \
                    cudaGetErrorString(err));                                    \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

// TODO: Implement the matmul kernel.
// Each thread computes one element C[row][col].
// Use blockIdx.y/x and threadIdx.y/x to compute row and col.
// Don't forget bounds check (row < n && col < n).
// C[row][col] = sum over k of A[row][k] * B[k][col]
// Row-major layout: element (i,j) is at index [i * n + j].
__global__ void matmul_kernel(const float *A, const float *B, float *C, int n) {
    // YOUR CODE HERE
}

int main(int argc, char *argv[]) {
    int n = 512;
    if (argc > 1) n = atoi(argv[1]);

    size_t size = n * n * sizeof(float);
    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_C = (float *)malloc(size);

    for (int i = 0; i < n * n; i++) {
        h_A[i] = (float)(i % 100) / 100.0f;
        h_B[i] = (float)((i * 7) % 100) / 100.0f;
    }

    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, size));
    CUDA_CHECK(cudaMalloc(&d_B, size));
    CUDA_CHECK(cudaMalloc(&d_C, size));

    CUDA_CHECK(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));

    // TODO: Set up 2D block and grid dimensions using dim3.
    // Use a block of (16, 16) threads. Compute grid size so it covers n x n.
    // dim3 block(?, ?);
    // dim3 grid(?, ?);
    // YOUR CODE HERE

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // TODO: Launch matmul_kernel with <<<grid, block>>>
    CUDA_CHECK(cudaEventRecord(start));
    // YOUR CODE HERE
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    CUDA_CHECK(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost));

    double checksum = 0.0;
    for (int i = 0; i < n * n; i++)
        checksum += h_C[i];

    printf("GPU matmul %dx%d: %.4f s (kernel only), checksum = %.2f\n",
           n, n, ms / 1000.0f, checksum);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
