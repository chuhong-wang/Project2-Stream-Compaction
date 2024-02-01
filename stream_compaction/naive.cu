#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

#define threadPerBlock 128; 

namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
        // TODO: __global__
        __global__ void scanGPUkernel(int d, int N, int *buffer_d_0, int *buffer_d_1) {
            int index = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (index > N) {return; }
            int offset = 1<<(d-1); 
            if (index >= offset){
                buffer_d_1[index] += buffer_d_0[index - offset]; 
            }
            else {
                buffer_d_1[index] = buffer_d_0[index]; 
            }
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            timer().startGpuTimer();
            // TODO
            dim3 fullBlockPerGrid = (n + threadPerBlock - 1) / threadPerBlock; 
            auto nBytes = n * sizeof(int); 

            int *buffer_d_0, *buffer_d_1; 
            cudaMalloc((void**)&buffer_d_0, nBytes);
            checkCUDAError("cudaMalloc buffer_d_0 failed!");
            cudaMalloc((void**)&buffer_d_1, nBytes);
            checkCUDAError("cudaMalloc buffer_d_1 failed!");

            bool output_at_1 = true; 
            cudaMemcpy(buffer_d_0, idata, nBytes, cudaMemcpyDeviceToHost); 

            for (auto i = 1; i<ilog2ceil(n); ++i){
                scanGPUkernel<<<fullBlockPerGrid, threadPerBlock>>>(i, n, buffer_d_0, buffer_d_1);
                // swap input and output 
                auto tmp = buffer_d_0; 
                buffer_d_0 = buffer_d_1; 
                buffer_d_1 = tmp;  

                output_at_1 = !output_at_1; 
            }

            if (output_at_1) {
                cudaMemcpy(odata, buffer_d_1, nBytes, cudaMemcpyDeviceToHost); 
            }
            else {
                cudaMemcpy(odata, buffer_d_0, nBytes, cudaMemcpyDeviceToHost); 
            }

            cudaFree(buffer_d_0); cudaFree(buffer_d_1); 
            timer().endGpuTimer();
        }
    }
}
