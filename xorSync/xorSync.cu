// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>
// Primary header is compatible with pre-C++11, collective algorithm headers require C++11
#include <cooperative_groups.h>
#include <iostream>
#include <stdio.h>
#include <cuda/barrier>

#define THREADS_PER_BLOCK 32

namespace cg = cooperative_groups;

__global__ void xorShift(int *R, int lg_fptf) {
    int x = threadIdx.x;
    int y = blockIdx.x;
    int z = threadIdx.x + 1;
    cg::grid_group g = cg::this_grid();
    volatile int w = blockDim.x - threadIdx.x;
    int iters = 1 << (16 - lg_fptf);
    for (int j = 0; j < iters; j++) {
        int tmp=(x^(x<<15));  
        int fpt=(((w^(w>>21))^(tmp^(tmp>>4))) & 31) << lg_fptf; 
        for (int i = 0; i < fpt; i++) {
            tmp=(x^(x<<15)); x=y; y=z; z=w;  
            w=(w^(w>>21))^(tmp^(tmp>>4)); 
        }
        g.sync();
    }
} 

__global__ void xorShiftNoSync(int *R, int lg_fptf) {
    int x = threadIdx.x;
    int y = blockIdx.x;
    int z = threadIdx.x + 1;
    volatile int w = blockDim.x - threadIdx.x;
    int iters = 1 << (16 - lg_fptf);
    for (int j = 0; j < iters; j++) {
        int tmp=(x^(x<<15));  
        int fpt=(((w^(w>>21))^(tmp^(tmp>>4))) & 31) << lg_fptf; 
        for (int i = 0; i < fpt; i++) {
            tmp=(x^(x<<15)); x=y; y=z; z=w;  
            w=(w^(w>>21))^(tmp^(tmp>>4)); 
        }
    }
} 

static inline void launchKernel(void* kernelFunc, int *d_R, int lg_fptf, int tb, int tpb) {
    dim3 gridDim(tb);
    dim3 blockDim(tpb);
    void *args[] = {(void*)&d_R, (void*)&lg_fptf};
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
    assert((argc == 3) && "Need log2(factor to multiply number of fibers) and number of threads.");

    int lg_fptf = atoi(argv[1]);
    int tt = atoi(argv[2]);
    std::cerr << "# max fibers per thread: " << 31*(1<<lg_fptf) << "\n# threads: " << tt << std::endl;

    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;

    size_t size = 1 * sizeof(int);

    // Allocate the device output vector C
    int *d_R = NULL;
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

    float sync_ms = 0;
    float nosync_ms = 0;

    cudaEventRecord(start);
    launchKernel((void*)xorShift, d_R, lg_fptf, tb, tpb);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&sync_ms, start, stop);
    
    cudaEventRecord(start);
    launchKernel((void*)xorShiftNoSync, d_R, lg_fptf, tb, tpb);
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&nosync_ms, start, stop);

    std::cout << nosync_ms << "," << sync_ms << std::endl;
    // print 'efficiency'
    //std::cout << (nosync_ms)/(sync_ms) * 100. << std::endl;

    err = cudaFree(d_R);

    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to free device vector R (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    
    //free(h_R);
    return 0;
}
