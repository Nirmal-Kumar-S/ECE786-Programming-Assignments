#include <stdio.h>
#include <fstream>
#include <iostream>
#include <vector>
#include <sys/time.h>
#include <cuda_runtime.h>

__global__ void matrix_multiply(const float *input, float *output, const float *Umatrix, int size, int qbit) 
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int mask = 1 << qbit;
  int index = i ^ mask;
  if (i <= (size - mask)) 
  {
    if((i/mask) % 2 != 1)
    {
     output[i] = Umatrix[0] * input[i] + Umatrix[1] * input[index];
     output[index] = Umatrix[2] * input[i] + Umatrix[3] * input[index];
    }
  }
}

int main(int argc, char *argv[])
{
    char *trace_file; // Variable that holds trace file name;
    trace_file = argv[1];
    
    // read input matrix and vector from file
    std::ifstream file(trace_file);

    // Read the 2x2 matrix
    float matrix[4];
    for (int i = 0; i < 4; i++) 
    {
        file >> matrix[i];
    }

    // Read the input vector
    std::vector<float> values;
    float value;
    while (file >> value) 
    {
      values.push_back(value);
    }
      
    // Read the qbit value
    int t;
    t = values.back();
    values.pop_back();

    // Size of input vector
    int n;
    n = values.size();

    // Compute size of input vector and matrix
    size_t size = n * sizeof(float);
    size_t size_m = 4 * sizeof(float);

    // Allocate the host input vector A
    float *h_A = (float *)malloc(size);

    // Allocate the host input matrix U
    float *h_U = (float *)malloc(size_m);

    // Allocate the host output vector B
    float *h_B = (float *)malloc(size);

    // Initialize the host input vector and matrix
    for (int i = 0; i < n; ++i)
    {
        h_A[i] = values[i];
    }

    for (int i = 0; i < 4; i++) 
    {
        h_U[i]= matrix[i];
    }

    // Allocate the device input vector A
    float *d_A = NULL;
    cudaMalloc((void **)&d_A, size);
    // Allocate the device input matrix U
    float *d_U = NULL;
    cudaMalloc((void **)&d_U, size_m);
    // Allocate the device output vector B
    float *d_B = NULL;
    cudaMalloc((void **)&d_B, size);

    // Copy the host input A and U in host memory to the device input vectors in device memory
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_U, h_U, size_m, cudaMemcpyHostToDevice);

    // Launch the Vector Add CUDA Kernel
    int threadsPerBlock = 256;
    int blocksPerGrid =(n + threadsPerBlock - 1) / threadsPerBlock;

    //Timing Report
    //struct timeval begin, end; 
    //gettimeofday (&begin, NULL);

    matrix_multiply<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_U, n, t);

    //gettimeofday (&end, NULL); 
    //int time_in_us = 1e6 * (end.tv_sec - begin.tv_sec) + (end.tv_usec - begin.tv_usec);

    // Copy the device result vector in device memory to the host result vector in host memory.
    cudaMemcpy(h_A, d_A, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_B, d_B, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_U, d_U, size_m, cudaMemcpyDeviceToHost);

    for (int i = 0; i < n; ++i)
    {
        std::cout<<h_B[i]<<std::endl;
    }

    //std::cout<<"Time in use = "<< time_in_us <<std::endl;
    
    // Free device global memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_U);

    // Free host memory
    free(h_A);
    free(h_B);
    free(h_U);

    // Reset the device and exit
    cudaDeviceReset();
    return 0;
}