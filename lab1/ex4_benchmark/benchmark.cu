#include <cstdio>
#include <cstdlib>
#include <ctime>

#define CUDA_CHECK(call)                                                       \
    do {                                                                        \
        cudaError_t err = call;                                                 \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,   \
                    cudaGetErrorString(err));                                    \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

void matmul_cpu(const float *A, const float *B, float *C, int n) {
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            float sum = 0.0f;
            for (int k = 0; k < n; k++)
                sum += A[i * n + k] * B[k * n + j];
            C[i * n + j] = sum;
        }
}

// TODO: Implement the GPU matmul kernel.
// Each thread should compute one element of C at position (row, col).
// Use blockIdx, blockDim, threadIdx to compute row and col.
// Don't forget the bounds check (row < n && col < n).
__global__ void matmul_gpu(const float *A, const float *B, float *C, int n) {
    // YOUR CODE HERE
}

void init_matrices(float *A, float *B, int n) {
    for (int i = 0; i < n * n; i++) {
        A[i] = (float)(i % 100) / 100.0f;
        B[i] = (float)((i * 7) % 100) / 100.0f;
    }
}

int main() {
    int sizes[] = {64, 128, 256, 512, 1024};
    int block_sizes[] = {8, 16, 32};
    int n_sizes = sizeof(sizes) / sizeof(sizes[0]);
    int n_blocks = sizeof(block_sizes) / sizeof(block_sizes[0]);

    // Warmup: launch a dummy kernel to initialize the CUDA context
    matmul_gpu<<<1, 1>>>(NULL, NULL, NULL, 0);
    CUDA_CHECK(cudaDeviceSynchronize());

    printf("N,block_size,cpu_ms,gpu_ms\n");

    for (int si = 0; si < n_sizes; si++) {
        int n = sizes[si];
        size_t size = n * n * sizeof(float);

        float *h_A = (float *)malloc(size);
        float *h_B = (float *)malloc(size);
        float *h_C = (float *)malloc(size);
        init_matrices(h_A, h_B, n);

        // CPU timing
        struct timespec t0, t1;
        clock_gettime(CLOCK_MONOTONIC, &t0);
        matmul_cpu(h_A, h_B, h_C, n);
        clock_gettime(CLOCK_MONOTONIC, &t1);
        double cpu_ms = ((t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9) * 1000.0;

        // TODO: Allocate GPU memory for d_A, d_B, d_C using cudaMalloc
        float *d_A, *d_B, *d_C;
        // YOUR CODE HERE

        // TODO: Copy h_A and h_B to GPU using cudaMemcpy
        // YOUR CODE HERE

        // TODO: Create CUDA events for timing
        cudaEvent_t start, stop;
        // YOUR CODE HERE

        for (int bi = 0; bi < n_blocks; bi++) {
            int bs = block_sizes[bi];

            // TODO: Set up dim3 block and grid dimensions
            // block should be (bs, bs), grid should cover the full n x n matrix
            // YOUR CODE HERE

            // TODO: Record start event, launch kernel, record stop event,
            // synchronize, and get elapsed time into gpu_ms
            float gpu_ms = 0.0f;
            // YOUR CODE HERE

            // TODO: Print CSV line: N,block_size,cpu_ms,gpu_ms
            // YOUR CODE HERE
        }

        // TODO: Clean up — destroy events, free GPU and CPU memory
        // YOUR CODE HERE
    }

    return 0;
}
