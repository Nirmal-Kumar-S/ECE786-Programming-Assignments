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

    // Read the 1D array of float values
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
    int size;
    size = values.size();

    // Allocate shared memory for input and output vector and matrix
    float *a, *b, *U;
    cudaMallocManaged(&a, size * sizeof(float));
    cudaMallocManaged(&b, size * sizeof(float));
    cudaMallocManaged(&U, 4 * sizeof(float));

    // Initialize input vector and matrix
    for (int i = 0; i < 4; i++)
    {
        U[i]=matrix[i];
    }
    for (int i = 0; i < size; i++) 
    {
        a[i]=values[i];
    }

    // perform matrix multiplication on GPU
    int threadsPerBlock = 256;
    int blocksPerGrid =(size + threadsPerBlock - 1) / threadsPerBlock;

    //Timing Report
    //struct timeval begin, end; 
    //gettimeofday (&begin, NULL);

    matrix_multiply<<<blocksPerGrid, threadsPerBlock>>>(a, b, U, size, t);

    //gettimeofday (&end, NULL); 
    //int time_in_us = 1e6 * (end.tv_sec - begin.tv_sec) + (end.tv_usec - begin.tv_usec);

    cudaDeviceSynchronize();
    
    // print output vector
    for (int i = 0; i < size; i++)
    {
        std::cout << b[i] << std::endl;
    }

    //std::cout<<"Time in use = "<< time_in_us <<std::endl;
    
    //Free memory
    cudaFree(a);
    cudaFree(b);
    cudaFree(U);

    return 0;
}