#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void matmul(const float *A, const float *B, float *C, int n) {
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            float sum = 0.0f;
            for (int k = 0; k < n; k++)
                sum += A[i * n + k] * B[k * n + j];
            C[i * n + j] = sum;
        }
}

int main(int argc, char *argv[]) {
    int n = 512;
    if (argc > 1) n = atoi(argv[1]);

    size_t size = n * n * sizeof(float);
    float *A = (float *)malloc(size);
    float *B = (float *)malloc(size);
    float *C = (float *)malloc(size);

    for (int i = 0; i < n * n; i++) {
        A[i] = (float)(i % 100) / 100.0f;
        B[i] = (float)((i * 7) % 100) / 100.0f;
    }

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    matmul(A, B, C, n);
    clock_gettime(CLOCK_MONOTONIC, &t1);

    double elapsed = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9;

    double checksum = 0.0;
    for (int i = 0; i < n * n; i++)
        checksum += C[i];

    printf("CPU matmul %dx%d: %.4f s, checksum = %.2f\n", n, n, elapsed, checksum);

    free(A);
    free(B);
    free(C);
    return 0;
}
