// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>
// Primary header is compatible with pre-C++11, collective algorithm headers require C++11
#include <cooperative_groups.h>
#include <iostream>
#include <stdio.h>
#include <cuda/barrier>

#define ITERS 1000000
#define THREADS_PER_BLOCK 1024
#ifndef COMM_BYTES
#define COMM_BYTES 8
#endif

struct commStruct {
    uint8_t data[COMM_BYTES];
};

namespace cg = cooperative_groups;

__global__ void tCommTest(commStruct *R, int tt) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    int m = tt/2;
    //cg::grid_group g = cg::this_grid();
    int i = 0;
    while (i < ITERS) {
        // (Empty) computation
        // Full communication costs will be realized here as the access is going to miss.
        commStruct val = R[t];
        //g.sync();
        __syncthreads();
        // Communication
        if (t > m) {
            R[t-m] = val;
        } else {
            R[t+m] = val; 
        }
        //g.sync();
        __syncthreads();
        i++;
    }
} 

static inline void launchKernel(void* kernelFunc, commStruct *d_R, int tb, int tpb) {
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

    size_t size = tt * sizeof(commStruct);

    // Allocate the device output vector C
    commStruct *d_R = NULL;
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
    //launchKernel((void*)tCommTest, d_R, tb, tpb);
    tCommTest<<<tb,tpb>>>(d_R,tt);
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