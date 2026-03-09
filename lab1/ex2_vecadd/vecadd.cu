#include <cstdio>
#include <cstdlib>
#include <cmath>

#define CUDA_CHECK(call)                                                       \
    do {                                                                        \
        cudaError_t err = call;                                                 \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,   \
                    cudaGetErrorString(err));                                    \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

#define N 1000000

__global__ void vecadd_kernel(const float *a, const float *b, float *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        // c[...] = a[...] ... b[...]; @TODO
}

int main() {
    size_t size = N * sizeof(float);

    float *h_a = (float *)malloc(size);
    float *h_b = //@TODO
    //@TODO

    for (int i = 0; i < N; i++) {
        h_a[i] = (float)i;
        h_b[i] = (float)(2 * i);
    }

    float *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc(&d_a, size));
    CUDA_CHECK(cudaMalloc(&d_b, size));
    // CUDA_CHECK(...); @TODO

    CUDA_CHECK(cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice));
    // CUDA_CHECK(...); @TODO

    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    // vecadd_kernel<<<blocks, threads>>>(d_a, ..., d_c, ...); @TODO

    // CUDA_CHECK(cudaMemcpy(..., ..., size, ...));

    int pass = 1;
    for (int i = 0; i < N; i++) {
        if (fabsf(h_c[i] - 3.0f * i) > 1e-5f) {
            pass = 0;
            printf("FAIL at index %d: expected %f, got %f\n", i, 3.0f * i, h_c[i]);
            break;
        }
    }
    printf("%s\n", pass ? "PASS" : "FAIL");

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    free(h_a);
    free(h_b);
    free(h_c);

    return 0;
}
