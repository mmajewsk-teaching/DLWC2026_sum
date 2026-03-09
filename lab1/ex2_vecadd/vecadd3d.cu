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

#define NX 128
#define NY 128
#define NZ 64

// TODO: Implement the 3D addition kernel.
// Use blockIdx.x/y/z and threadIdx.x/y/z to compute (x, y, z).
// Don't forget bounds check (x < nx && y < ny && z < nz).
// Convert 3D index to flat: idx = z * ny * nx + y * nx + x
__global__ void add3d_kernel(const float *a, const float *b, float *c,
                             int nx, int ny, int nz) {
    // YOUR CODE HERE
}

int main() {
    int total = NX * NY * NZ;
    size_t size = total * sizeof(float);

    float *h_a = (float *)malloc(size);
    float *h_b = (float *)malloc(size);
    float *h_c = (float *)malloc(size);

    for (int i = 0; i < total; i++) {
        h_a[i] = (float)(i % 1000);
        h_b[i] = (float)((i * 3) % 1000);
    }

    float *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc(&d_a, size));
    CUDA_CHECK(cudaMalloc(&d_b, size));
    CUDA_CHECK(cudaMalloc(&d_c, size));

    CUDA_CHECK(cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice));

    // TODO: Set up 3D block and grid dimensions using dim3.
    // Use block(8, 8, 8) and compute grid to cover NX x NY x NZ.
    // YOUR CODE HERE

    // TODO: Launch add3d_kernel with <<<grid, block>>>
    // YOUR CODE HERE

    CUDA_CHECK(cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost));

    int pass = 1;
    for (int i = 0; i < total; i++) {
        float expected = (float)(i % 1000) + (float)((i * 3) % 1000);
        if (fabsf(h_c[i] - expected) > 1e-5f) {
            pass = 0;
            printf("FAIL at index %d: expected %f, got %f\n", i, expected, h_c[i]);
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
