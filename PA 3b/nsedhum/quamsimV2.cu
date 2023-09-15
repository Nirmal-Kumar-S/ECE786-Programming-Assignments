#include <stdio.h>
#include <fstream>
#include <iostream>
#include <vector>
#include <sys/time.h>
#include <cuda_runtime.h>
#include <algorithm>
using namespace std;

// Function to find positions of set bits in a binary number
void findSetBits(int num, int positions[], int& count) 
{
    int position = 0;
    while (num > 0) {
        if (num & 1) {  // Check if the least significant bit is set
            positions[count++] = position;  // Add position to array
        }
        num = num >> 1;  // Right shift the number by 1 to check the next bit
        position++;  // Increment the position
    }
}

// Function to set the bits at specified indices
void setBits(int& num, int index, int value) 
{
    int mask = 1 << index;
    if (value == 1) {
        num |= mask;
    } else {
        num &= ~mask;
    }
}

__global__ void matrix_multiply(float *d_A, float *d_B, float *d_U, int *d_indices, int *d_TB_indices, int *d_qbit_indices) 
{ 
    __shared__ float s[64];
    float temporary = 0;
    s[2*threadIdx.x] = d_A[d_indices[2*threadIdx.x] + d_TB_indices[blockIdx.x]];
    s[2*threadIdx.x+1] = d_A[d_indices[2*threadIdx.x+1] + d_TB_indices[blockIdx.x]];
    __syncthreads();
    
    for(int k = 0; k < 6; k++)
    {
        temporary = d_U[(k*4)] * s[2*threadIdx.x - threadIdx.x % d_qbit_indices[k]] + d_U[(k*4) + 1] * s[(2*threadIdx.x - threadIdx.x % d_qbit_indices[k]) + d_qbit_indices[k]];
        s[(2*threadIdx.x - threadIdx.x % d_qbit_indices[k]) + d_qbit_indices[k]] = d_U[(k*4) + 2] * s[2*threadIdx.x - threadIdx.x % d_qbit_indices[k]] + d_U[(k*4) + 3] * s[(2*threadIdx.x - threadIdx.x % d_qbit_indices[k]) + d_qbit_indices[k]];
        s[(2*threadIdx.x - threadIdx.x % d_qbit_indices[k])] = temporary;    
    }
    __syncthreads();
    
    d_B[d_indices[2*threadIdx.x] + d_TB_indices[blockIdx.x]] = s[2*threadIdx.x];
    d_B[d_indices[2*threadIdx.x+1] + d_TB_indices[blockIdx.x]] = s[2*threadIdx.x+1];
    
}

int main(int argc, char *argv[])
{
    char *trace_file; // Variable that holds trace file name;
    trace_file = argv[1];
    
    // read input matrix and vector from file
    ifstream file(trace_file);

    // Read the 2x2 matrix
    float matrix[24];
    for (int i = 0; i < 24; i++) 
    {
        file >> matrix[i];
    }

    // Read the input vector
    vector<float> values;
    float value;
    while (file >> value) 
    {
      values.push_back(value);
    }

    // Read the qubit values
    int qubit[6];
    qubit[5] = values.back();
    values.pop_back();
    qubit[4] = values.back();
    values.pop_back();
    qubit[3] = values.back();
    values.pop_back();
    qubit[2] = values.back();
    values.pop_back();
    qubit[1] = values.back();
    values.pop_back();
    qubit[0] = values.back();
    values.pop_back();
    
    sort(qubit,qubit+6);

    // Size of input vector
    float n;
    n = values.size();

    int length = log2(n);

    int non_qubit[length];
    int newSize = length;

    for(int i = 0; i < length; i++)
    {
        non_qubit[i]=i;
    }

    // Loop through the elements to remove
    for (int i = 0; i < length; ++i) {
        int* iter = find(non_qubit, non_qubit + newSize, qubit[i]);
        if (iter != non_qubit + newSize) {
            // Shift the elements to fill the gap
            for (int* j = iter; j < non_qubit + newSize - 1; ++j) {
                *j = *(j + 1);
            }
            // Decrease the size of the array
            --newSize;
        }
    }
    
    int indices[64];

    for (int i = 0; i < 64; ++i)
    {
        indices[i] = 0;
        if(i%2 == 0)
        {
            setBits(indices[i],qubit[0],0);
        }
        else
        {
            setBits(indices[i],qubit[0],1);
        }
    }
    for (int i = 0; i < 64; i += 4)
    {
        setBits(indices[i],qubit[1],0);
        setBits(indices[i+1],qubit[1],0);
        setBits(indices[i+2],qubit[1],1);
        setBits(indices[i+3],qubit[1],1);
    }
    for (int i = 0; i < 64; i += 8) 
    {
        for (int j = 0; j < 4; j++) 
        {
            setBits(indices[i + j],qubit[2],0);
            setBits(indices[i + j + 4],qubit[2],1);
        }
    }
    for (int i = 0; i < 64; i += 16) 
    {
        for (int j = 0; j < 8; j++) 
        {
            setBits(indices[i + j],qubit[3],0);
            setBits(indices[i + j + 8],qubit[3],1);
        }
    }
    for (int i = 0; i < 64; i += 32) 
    {
        for (int j = 0; j < 16; j++) 
        {
            setBits(indices[i + j],qubit[4],0);
            setBits(indices[i + j + 16],qubit[4],1);
        }
    }
    for (int i = 0; i < 64; i += 64) 
    {
        for (int j = 0; j < 32; j++) 
        {
            setBits(indices[i + j],qubit[5],0);
            setBits(indices[i + j + 32],qubit[5],1);
        }
    }
    
    int Blocks = n/64;
    int Threads = 32;

    int TB_indices[Blocks];
    TB_indices[0]=0;
    for (int i = 1; i < Blocks; i++) 
    {
        int positions[Blocks];
        int count = 0;
        int index = 0;
        findSetBits(i, positions, count);
        for(int j = 0; j < count; j++)
        {
            index += pow(2,non_qubit[positions[j]]); 
        }
        TB_indices[i]=index;
    }
    
    int qbit_indices[6]={1,2,4,8,16,32};

    //Print Check
    /*
    for(int i = 0 ; i < 6; i++)
    {
        cout<<"Qubit ["<<i<<"] = "<<qubit[i]<<endl;
    }
    for(int i = 0 ; i < newSize; i++)
    {
        cout<<"Non_Qubit ["<<i<<"] = "<<non_qubit[i]<<endl;
    }
    for(int i = 0 ; i < 64; i++)
    {
        cout<<"Indices ["<<i<<"] = "<<indices[i]<<endl;
    }
    for(int i = 0 ; i < Blocks; i++)
    {
        cout<<"TB_indices ["<<i<<"] = "<<TB_indices[i]<<endl;
    }
    */
    // Compute size of input vector and matrix
    size_t size = n * sizeof(float);
    size_t size_m = 24 * sizeof(float);

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

    for (int i = 0; i < 24; i++) 
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
    int *d_indices = NULL;
    cudaMalloc((void **)&d_indices, 64 * sizeof(int));
    int *d_TB_indices = NULL;
    cudaMalloc((void **)&d_TB_indices, Blocks * sizeof(int));
    int *d_qbit_indices = NULL;
    cudaMalloc((void **)&d_qbit_indices, 6 * sizeof(int));

    // Copy the host input A and U in host memory to the device input vectors in device memory
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_U, h_U, size_m, cudaMemcpyHostToDevice);
    cudaMemcpy(d_indices, indices, 64 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_TB_indices, TB_indices, Blocks * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_qbit_indices, qbit_indices, 6 * sizeof(int), cudaMemcpyHostToDevice);
    //Timing Report
    //struct timeval begin, end; 
    //gettimeofday (&begin, NULL);

    // Launch the Vector Add CUDA Kernel
    matrix_multiply<<<Blocks, Threads>>>(d_A, d_B, d_U, d_indices, d_TB_indices, d_qbit_indices);
    cudaDeviceSynchronize();
    
    //gettimeofday (&end, NULL); 
    //int time_in_us = 1e6 * (end.tv_sec - begin.tv_sec) + (end.tv_usec - begin.tv_usec);

    // Copy the device result vector in device memory to the host result vector in host memory.
    cudaMemcpy(h_A, d_A, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_B, d_B, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_U, d_U, size_m, cudaMemcpyDeviceToHost);
    
    for (int i = 0; i < n; ++i)
    {
        printf("%.3f\n", h_B[i]);
    }
    
    //cout<<"Time in use = "<< time_in_us <<endl;
    
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

