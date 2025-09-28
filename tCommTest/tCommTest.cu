// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>
// Primary header is compatible with pre-C++11, collective algorithm headers require C++11
#include <cooperative_groups.h>
#include <iostream>
#include <stdio.h>
#include <cuda/barrier>

#define ITERS 1000000
#define THREADS_PER_BLOCK 32 // this is needed i think
#ifndef COMM_BYTES
#define COMM_BYTES 8
#endif

#define unit_t uint32_t
#define COMM_UNITS (COMM_BYTES/sizeof(unit_t))
#define CACHE_LINE_BYTES 128
#define UNITS_PER_CACHE_LINE (CACHE_LINE_BYTES/sizeof(COMM_UNITS))

namespace cg = cooperative_groups;

__global__ void tCommTest(unit_t *R, int tt) {
    int b = blockIdx.x;
    int m = gridDim.x / 2;
    cg::grid_group g = cg::this_grid();
    int i = 0;
    while (i < ITERS) {
        unit_t val[COMM_UNITS];
        for (int i = 0; i < COMM_UNITS; i++) {
            val[i] = R[(COMM_UNITS*b+i)*32+threadIdx.x];
        }
        g.sync();
        // Communication
        int bd = (b > m) ? b-m : b+m;
        for (int i = 0; i < COMM_UNITS; i++) {
            R[(COMM_UNITS*bd+i)*32+threadIdx.x] = val[i];
        }
        g.sync();
        i++;
    }
} 

static inline void launchKernel(void* kernelFunc, unit_t *d_R, int tb, int tpb) {
    dim3 gridDim(tb);
    dim3 blockDim(tpb);
    int tt = tb * tpb;
    void *args[] = {(void*)&d_R, (void*)&tt};
    cudaLaunchCooperativeKernel(kernelFunc, gridDim, blockDim, args);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to launch kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    cudaDeviceSynchronize();
}

int main(int argc, char **argv)
{
    assert((argc == 2) && "Need number of fibers and number of threads.");

    //int fpt = atoi(argv[1]);
    int tt = atoi(argv[1]);
    //std::cerr << "# fibers per thread: " << fpt <<
    std::cerr << "# threads: " << tt << std::endl;
    std::cerr << "# comm bytes: " << COMM_BYTES << std::endl;

    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;

    size_t size = tt * COMM_BYTES;

    // Allocate the device output vector C
    unit_t *d_R = NULL;
    err        = cudaMalloc((void **)&d_R, size);
    
    // Allocate the host input vector A
    //int *h_R = (int *)malloc(size);

    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device vector C (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    int tb = tt/THREADS_PER_BLOCK;
    int tpb = THREADS_PER_BLOCK;
    
    if (tb == 0) {
        tb = 1;
        tpb = tt;
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float tcomm_ms = 0;

    cudaEventRecord(start);
    launchKernel((void*)tCommTest, d_R, tb, tpb);
    //tCommTest<<<tb,tpb>>>(d_R,tt);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&tcomm_ms, start, stop);

    std::cout << tcomm_ms << std::endl;

    err = cudaFree(d_R);

    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to free device vector R (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    
    //free(h_R);
    return 0;
}