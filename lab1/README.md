# Lab 1: Introduction to CUDA Programming

## Setup

```bash
cd lab1
pixi install
pixi shell
```

This gives you `nvcc`, `gcc`, and Python with matplotlib/pandas.

---

## Exercise 1: Hello World

A minimal CUDA program that launches a kernel on the GPU.

**Concepts**: `__global__`, `<<<blocks, threads>>>`, `threadIdx`, `blockIdx`, `cudaDeviceSynchronize()`

```bash
cd ex1_hello
nvcc -o hello hello.cu
./hello
```

**Expected output**: 8 "Hello from block X, thread Y" lines (order may vary!) followed by "Hello from the host!"

**Try**:
- Change the launch config from `<<<2, 4>>>` to `<<<4, 2>>>` or `<<<1, 8>>>`. What changes?
- What happens if you remove `cudaDeviceSynchronize()`?

---

## Exercise 2: Vector Addition

### Part A: 1D Vector Addition

Adds two vectors of 1M floats on the GPU.

**Concepts**: `cudaMalloc`, `cudaMemcpy`, `cudaFree`, grid/block sizing, error checking

```bash
cd ex2_vecadd
nvcc -o vecadd vecadd.cu
./vecadd
```

**Expected output**: `PASS`

**Try**:
- Change `threads` from 256 to 64 or 1024. Does the result still pass?
- What happens if you remove the `if (i < n)` guard in the kernel?

### Part B: 3D Tensor Addition

Same idea but with a 3D array (128×128×64), using `dim3` for both block and grid dimensions. This introduces multi-dimensional thread indexing before Exercise 3.

**Concepts**: `dim3`, `blockIdx.x/y/z`, `threadIdx.x/y/z`

```bash
nvcc -o vecadd3d vecadd3d.cu
./vecadd3d
```

**Expected output**: `PASS`

**Key difference from Part A**: instead of a flat 1D index, each thread computes its `(x, y, z)` position using `blockIdx.x/y/z` and `threadIdx.x/y/z`, then converts to a flat index for memory access.

**Try**:
- Change `dim3 block(8, 8, 8)` to `dim3 block(16, 16, 4)`. Same total threads per block (512), different shape. Does it still work?

---

## Exercise 3: Matrix Multiplication
Compare a naive CPU matrix multiply against a GPU version.

**Concepts**: 2D grids (`dim3`), `threadIdx.x/y`, CUDA event timing

First, build and run the CPU version:
```bash
cd ex3_matmul
gcc -O2 -o matmul_cpu matmul_cpu.c -lm
./matmul_cpu 512
```

**Your task**: Open `matmul_gpu.cu` and fill in the TODO sections:
1. Implement the matmul kernel (each thread computes one element of C)
2. Set up `dim3` block and grid dimensions
3. Launch the kernel

```bash
nvcc -O2 -o matmul_gpu matmul_gpu.cu
./matmul_gpu 512
```

**Verification**: checksums should match between CPU and GPU for the same N. GPU becomes significantly faster at N >= 1024.

**Try**:
- Run with N = 128, 256, 512, 1024, 2048. At what point does the GPU become faster?
- Change block size from `(16, 16)` to `(32, 32)`. Does performance change?

---

## Exercise 4: Benchmarking & Visualization
Benchmark matrix multiplication across different matrix sizes and GPU block sizes, then visualize the results.

**Concepts**: performance measurement, CUDA events, CPU vs GPU comparison, parameter tuning

**Your task**: Open `benchmark.cu` and fill in the TODO sections:
1. Implement the GPU matmul kernel
2. Allocate GPU memory and copy data
3. Set up CUDA event timing
4. Configure `dim3` block/grid dimensions
5. Launch the kernel and measure time
6. Print CSV output

```bash
cd ex4_benchmark
nvcc -O2 -o benchmark benchmark.cu
./benchmark > results.csv
cat results.csv
python plot.py results.csv
```

**Expected CSV output format**:
```
N,block_size,cpu_ms,gpu_ms
64,8,0.3412,0.0531
64,16,0.3412,0.0487
...
```

**Expected plots** (saved to `results.png`):
1. CPU vs GPU time (log-log scale)
2. GPU speedup over CPU
3. Effect of block size on GPU performance

**Questions**:
- At what matrix size does the GPU become faster than the CPU?
- Which block size gives the best performance? Why?
- Why is the GPU slower for small matrices?

---

## Exercise 5: Barnsley Fern
Generate a [Barnsley fern](https://en.wikipedia.org/wiki/Barnsley_fern) fractal on the GPU using an Iterated Function System (IFS).

**Concepts**: `atomicAdd`, per-thread random number generation, GPU histogram, image output

**How it works**: Each CUDA thread runs an independent random walk through 4 affine transformations. Threads accumulate pixel hits into a shared 2D histogram using `atomicAdd`. The result is saved as a BMP image.

**Your task**: Open `fern.cu` and fill in the TODO sections:
1. Implement the 4 affine transformations (coefficients given in comments)
2. Implement the kernel: RNG init, iteration loop, coordinate mapping, atomic histogram update
3. Allocate GPU memory and launch the kernel

```bash
cd ex5_fern
nvcc -O2 -o fern fern.cu
./fern
```

**Expected output**: `fern.bmp` — a green Barnsley fern fractal image (800x1200), viewable directly in any image viewer.

**Try**:
- Change `ITERS` (iterations per thread). How does image quality change?
- Change `BLOCKS` and `THREADS`. What happens with more/fewer total threads?
- What happens if you remove `atomicAdd` and use a regular `+=` instead?

---

## Useful References

- [CUDA C Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [CUDA Runtime API](https://docs.nvidia.com/cuda/cuda-runtime-api/)
- `nvidia-smi` — check GPU status and utilization
- `nvcc --help` — compiler options
