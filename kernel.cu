
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <windows.h>
#include "C:/Program Files (x86)/Windows Kits/10/Include/10.0.22621.0/um/profileapi.h"

#include <stdio.h>
#include <cmath>
#include <stdlib.h>

#define TILE_WIDTH 8
#define THRESHOLD 1e-3

cudaError_t multiplyWithCudaNaive(float* c, float* a, float* b, int dima_1, int dima_2, int dimb_1, int dimb_2);
cudaError_t multiplyWithCudaTiled(float* c, float* a, float* b, int dima_1, int dima_2, int dimb_1, int dimb_2);
cudaError_t multiplyWithCudaTurned(float* c, float* a, float* b, int dima_1, int dima_2, int dimb_1, int dimb_2);
cudaError_t transposeWithCuda(float* a, float* a_t, int m, int n);

void randFillMatrix(float* a, int size, float X) {
    for (int k = 0; k < size; k++) {
        a[k] = static_cast <float> (rand()) / (static_cast <float> (RAND_MAX / X));
    }
}

void isSame(float* a, float* b, int size2, int size1, float threshold) {
    bool successful = true;
    float maxVal = 0.0f;
    for (int i = 0; i < size1; i++) {
        for (int j = 0; j < size2; j++) {
            if (fabs(b[i * size2 + j] - a[i * size2 + j]) > threshold) {
                successful = false;
            }
            maxVal = std::max(fabs(b[i * size2 + j] - a[i * size2 + j]), maxVal);
            printf("%f ", a[i * size2 + j]);
        }
        printf("\n");
    }

    if (successful) {
        printf("Passes Threshold of %.10e with max error of %.10e\n", threshold, maxVal);
    }
    else {
        printf("Fails to Pass Threshold of %.10e with max error of %.10e\n", threshold, maxVal);
    }
}

void CPUSerialKernel(float* outputc, float* inputa, float* inputb, int aRows, int aCols, int bRows, int bCols) {
    if (aCols == bRows) {
        for (int y = 0; y < aRows; y++) {
            for (int x = 0; x < bCols; x++) {
                float sum = 0.0f;
                for (int i = 0; i < aCols; i++) {
                    sum += inputa[y * aCols + i] * inputb[i * bCols + x];
                }
                outputc[y * bCols + x] = sum;
            }
        }
    }
}

__global__ void GPU_Naive_Kernel(float* outputc, float* inputa, float* inputb, int aRows, int aCols, int bRows, int bCols) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < bCols && y < aRows && aCols == bRows) {
        float sum = 0.0f;
        for (int i = 0; i < aCols; i++) {
            if (y * aCols + i < aCols * aRows && i * bCols + x < bCols * bRows) {
                sum += inputa[y * aCols + i] * inputb[i * bCols + x];
            }
            //printf("aIndex: %d, bIndex: %d \n", y * aCols + i, i * bCols + x);
        }
        outputc[y * bCols + x] = sum;
    }
}

__global__ void GPU_Tiled_Kernel(float* outputc, float* inputa, float* inputb, int aRows, int aCols, int bRows, int bCols) {
    __shared__ float ds_M[TILE_WIDTH][TILE_WIDTH];
    __shared__ float ds_N[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x; int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;

    int Row = by * blockDim.y + ty;
    int Col = bx * blockDim.x + tx;

    float Pvalue = 0.0f;

    // Loop over the M and N tiles required to compute the P element
    for (int p = 0; p < ceilf((float)(aCols + 1) / (float)TILE_WIDTH); p++) {
        // Collaborative loading of M and N tiles into shared memory
        if (Row < aRows && (p * TILE_WIDTH + tx) < aCols) {
            ds_M[ty][tx] = inputa[Row * aCols + p * TILE_WIDTH + tx];
        }
        else {
            ds_M[ty][tx] = 0.0f;
        }

        if ((p * TILE_WIDTH + ty) < bRows && Col < bCols) {
            ds_N[ty][tx] = inputb[(p * TILE_WIDTH + ty) * bCols + Col];
        }
        else {
            ds_N[ty][tx] = 0.0f;
        }
        __syncthreads();

        for (int i = 0; i < TILE_WIDTH; i++) {
            Pvalue += ds_M[ty][i] * ds_N[i][tx];
        }

        __syncthreads();

        //printf("%d %d\n", ty, tx);
    }

    if (Row < aRows && Col < bCols) {
        outputc[Row * bCols + Col] = Pvalue;
    }
}

__global__ void GPU_Corner_Kernel(float* outputc, float* inputa, float* inputb, int aRows, int aCols, int bRows, int bCols) {
    __shared__ float ds_M[TILE_WIDTH][TILE_WIDTH];
    __shared__ float ds_N[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x; int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;

    int Row = by * blockDim.y + ty;
    int Col = bx * blockDim.x + tx;

    float Pvalue = 0.0f;

    // Loop over the M and N tiles required to compute the P element
    for (int p = 0; p < ceilf((float)(aCols + 1) / (float)TILE_WIDTH); p++) {
        // Collaborative loading of M and N tiles into shared memory
        if (Row < aRows && (p * TILE_WIDTH + tx) < aCols) {
            ds_M[ty][tx] = inputa[Row * aCols + p * TILE_WIDTH + tx];
        }
        else {
            ds_M[ty][tx] = 0.0f;
        }

        if ((p * TILE_WIDTH + ty) < bRows && Col < bCols) {
            ds_N[tx][ty] = inputb[(bx * TILE_WIDTH + ty) * bCols + (p * TILE_WIDTH + tx)];
        }
        else {
            ds_N[tx][ty] = 0.0f;
        }
        __syncthreads();

        for (int i = 0; i < TILE_WIDTH; i++) {
            Pvalue += ds_M[ty][i] * ds_N[i][tx];
        }

        __syncthreads();

        //printf("%d %d\n", ty, tx);
    }

    if (Row < aRows && Col < bCols) {
        outputc[Row * bCols + Col] = Pvalue;
    }
}

// Extra Credit. 
__global__ void GPU_Transpose_Kernel(float* in, float* out, int inRow, int inCols) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int x = bx * blockDim.x + tx;
    int y = by * blockDim.y + ty;

    if (x < inCols && y < inRow && x >= 0 && y >= 0) {
        out[y * inRow + x] = in[x * inCols + y];
        //printf("in loc: %d, out loc: %d\n", x * inCols + y, y * inRow + x);
    }
}

int main()
{
    // Handling Time based Variables
    LARGE_INTEGER time1st;
    LARGE_INTEGER time1end;
    LARGE_INTEGER time2st;
    LARGE_INTEGER time2end;
    LARGE_INTEGER time3st;
    LARGE_INTEGER time3end;
    LARGE_INTEGER time4st;
    LARGE_INTEGER time4end;
    LARGE_INTEGER time5st;
    LARGE_INTEGER time5end;
    LARGE_INTEGER time5mid;

    //Basic Cuda Status
    cudaError_t cudaStatus;


    //Array Dimensions
    const int dima_1 = 256;
    const int dima_2 = 256;
    const int dimb_1 = 256;
    const int dimb_2 = 256;
    const int AarraySize2 = dima_1 * dima_2;
    const int BarraySize2 = dimb_1 * dimb_2;
    const int CarraySize2 = dima_1 * dimb_2;

    /*
    Testing Transposing 
    */

    /*float* test_a = (float*)malloc(AarraySize2 * sizeof(float));
    float* test_a_t = (float*)calloc(AarraySize2, sizeof(float));

    randFillMatrix(test_a, AarraySize2, 1.0f);

    cudaStatus = transposeWithCuda(test_a, test_a_t, dima_1, dima_2);

    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cuda Naive failed!");
        return 1;
    }*/
    /*for (int k = 0; k < dima_1; k++) {
        for (int l = 0; l < dima_2; l++) {
            printf("%f ", test_a[k * dima_2 + l]);
        }
        printf("\n");
    }

    printf("\n");

    for (int k = 0; k < dima_1; k++) {
        for (int l = 0; l < dima_2; l++) {
            printf("%f ", test_a_t[k * dima_2 + l]);
        }
        printf("\n");
    }*/


    QueryPerformanceCounter(&time1st);

    //Base matmul Array Mallocing
    float* a_input = (float*)malloc(AarraySize2 * sizeof(float));
    float* b_input = (float*)malloc(BarraySize2 * sizeof(float));
    float* b_input_t = (float*)calloc(BarraySize2, sizeof(float));
    randFillMatrix(a_input, AarraySize2, 1.0f);
    randFillMatrix(b_input, BarraySize2, 1.0f);
    float* cCPU = (float*)malloc(CarraySize2 * sizeof(float));
    float* cGPU_Naive = (float*)malloc(CarraySize2 * sizeof(float));
    float* cGPU_Tiled = (float*)malloc(CarraySize2 * sizeof(float));
    float* cGPU_Turned = (float*)malloc(CarraySize2 * sizeof(float));

    // Handling CPU Multiplication
    QueryPerformanceCounter(&time2st);
    CPUSerialKernel(cCPU, a_input, b_input, dima_1, dima_2, dimb_1, dimb_2);
    printf("CPU\n");
    for (int k = 0; k < dima_1; k++) {
        for (int l = 0; l < dimb_2; l++) {
            printf("%f ", cCPU[k * dimb_2 + l]);
        }
        printf("\n");
    }

    QueryPerformanceCounter(&time2end);
    printf("\n");
    // Handling Naive Multiplication
    QueryPerformanceCounter(&time3st);
    cudaStatus = multiplyWithCudaNaive(cGPU_Naive, a_input, b_input, dima_1, dima_2, dimb_1, dimb_2);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cuda Naive failed!");
        return 1;
    }

    /*for (int k = 0; k < dima_1; k++) {
        for (int l = 0; l < dimb_2; l++) {
            printf("%f ", cGPU_Naive[k * dimb_2 + l]);
        }
        printf("\n");
    }*/

    QueryPerformanceCounter(&time3end);
    //Handling Tiled Multiplication
    QueryPerformanceCounter(&time4st);
    cudaStatus = multiplyWithCudaTiled(cGPU_Tiled, a_input, b_input, dima_1, dima_2, dimb_1, dimb_2);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cuda Tiled failed!");
        return 1;
    }

    QueryPerformanceCounter(&time4end);
    /*for (int k = 0; k < dima_1; k++) {
        for (int l = 0; l < dimb_2; l++) {
            printf("%f ", cGPU_Tiled[k * dimb_2 + l]);
        }
        printf("\n");
    }*/

    QueryPerformanceCounter(&time5st);
    cudaStatus = transposeWithCuda(b_input, b_input_t, dima_1, dima_2);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cuda Transpose failed!");
        return 1;
    }

    QueryPerformanceCounter(&time5mid);
    cudaStatus = multiplyWithCudaTurned(cGPU_Turned, a_input, b_input_t, dima_1, dima_2, dimb_1, dimb_2);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cuda Corner turning failed!");
        return 1;
    }

    QueryPerformanceCounter(&time5end);
    QueryPerformanceCounter(&time1end);

    printf("Naive\n");
    isSame(cGPU_Naive, cCPU, dimb_2, dima_1, THRESHOLD);
    printf("Tiled\n");
    isSame(cGPU_Tiled, cCPU, dimb_2, dima_1, THRESHOLD);
    printf("Turned\n");
    isSame(cGPU_Turned, cCPU, dimb_2, dima_1, THRESHOLD);


    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    printf("\n\n");
    printf("OverallTime for all kernels: %d ms\n", time1end.QuadPart - time1st.QuadPart);
    printf("Time for CPU Kernel: %d ms\n", time2end.QuadPart - time2st.QuadPart);
    printf("Time for Naive Kernel: %d ms\n", time3end.QuadPart - time3st.QuadPart);
    printf("Time for Tiled Kernel: %d ms\n", time4end.QuadPart - time4st.QuadPart);
    printf("Time for Transpose: %d ms\n", time5mid.QuadPart - time5st.QuadPart);
    printf("Time for Turned Kernel: %d ms\n", time5end.QuadPart - time5st.QuadPart);

    free(a_input);
    free(b_input);
    /*free(test_a);
    free(test_a_t);*/
    free(b_input_t);

    free(cCPU);
    free(cGPU_Naive);
    free(cGPU_Tiled);
    free(cGPU_Turned);

    return 0;
}

// Helper function for using CUDA to add vectors in parallel.

cudaError_t multiplyWithCudaNaive(float* c, float* a, float* b, int dima_1, int dima_2, int dimb_1, int dimb_2) {
    float* dev_a = 0;
    float* dev_b = 0;
    float* dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, dima_1 * dimb_2 * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, dima_1 * dima_2 * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, dimb_1 * dimb_2 * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, dima_1 * dima_2 * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, dimb_1 * dimb_2 * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    const dim3 threadsPerBlock(16, 16);
    const dim3 blocksPerGrid(ceil((float)(dimb_2 + 1) / 16.0), ceil((float)(dima_1 + 1) / 16.0));

    // Launch a kernel on the GPU with one thread for each element.
    GPU_Naive_Kernel << <blocksPerGrid, threadsPerBlock >> > (dev_c, dev_a, dev_b, dima_1, dima_2, dimb_1, dimb_2);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, dima_1 * dimb_2 * sizeof(float), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);

    return cudaStatus;
}

cudaError_t multiplyWithCudaTiled(float* c, float* a, float* b, int dima_1, int dima_2, int dimb_1, int dimb_2) {
    float* dev_a = 0;
    float* dev_b = 0;
    float* dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, dima_1 * dimb_2 * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, dima_1 * dima_2 * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, dimb_1 * dimb_2 * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, dima_1 * dima_2 * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, dimb_1 * dimb_2 * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    const dim3 threadsPerBlock(TILE_WIDTH, TILE_WIDTH);
    const dim3 blocksPerGrid(ceil((float)(dimb_2 + 1) / (float)TILE_WIDTH), ceil((float)(dima_1 + 1) / (float)TILE_WIDTH));

    // Launch a kernel on the GPU with one thread for each element.
    GPU_Tiled_Kernel << <blocksPerGrid, threadsPerBlock >> > (dev_c, dev_a, dev_b, dima_1, dima_2, dimb_1, dimb_2);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, dima_1 * dimb_2 * sizeof(float), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);

    return cudaStatus;
}

cudaError_t multiplyWithCudaTurned(float* c, float* a, float* b, int dima_1, int dima_2, int dimb_1, int dimb_2) {
    float* dev_a = 0;
    float* dev_b = 0;
    float* dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, dima_1 * dimb_2 * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, dima_1 * dima_2 * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, dimb_1 * dimb_2 * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, dima_1 * dima_2 * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, dimb_1 * dimb_2 * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    const dim3 threadsPerBlock(TILE_WIDTH, TILE_WIDTH);
    const dim3 blocksPerGrid(ceil((float)(dimb_2 + 1) / (float)TILE_WIDTH), ceil((float)(dima_1 + 1) / (float)TILE_WIDTH));

    // Launch a kernel on the GPU with one thread for each element.
    GPU_Corner_Kernel << <blocksPerGrid, threadsPerBlock >> > (dev_c, dev_a, dev_b, dima_1, dima_2, dimb_1, dimb_2);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, dima_1 * dimb_2 * sizeof(float), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);

    return cudaStatus;
}

// where m is numRows and n is numCols
cudaError_t transposeWithCuda(float* a, float* a_t, int m, int n) {
    float* dev_a;
    float* dev_a_t;

    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_a, m * n * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a_t, n * m * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, n * m * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    const dim3 threadsPerBlock(16, 16);
    const dim3 blocksPerGrid(ceil((float)(m + 1) / 16.0), ceil((float)(n + 1) / 16.0));

    // Launch a kernel on the GPU with one thread for each element.
    GPU_Transpose_Kernel << <blocksPerGrid, threadsPerBlock >> > (dev_a, dev_a_t, m, n);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching transposeKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(a_t, dev_a_t, n * m * sizeof(float), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_a);

    return cudaStatus;
}