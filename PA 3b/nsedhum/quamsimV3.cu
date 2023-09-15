#include <stdio.h>
#include <fstream>
#include <iostream>
#include <vector>
#include <sys/time.h>
#include <cuda_runtime.h>
using namespace std;

__global__ void matrix_multiply(float *input, float *output, float *Umatrix, int size, int qbit)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int segment1 = i/qbit;
	int j= i+ (size/4);
	int segment2=j/qbit;

	i=i+segment1*qbit;
	j=j+segment2*qbit;
	
    output[i]=(Umatrix[0]*input[i]) + (Umatrix[1]*input[i+qbit]);                
	output[i+qbit]=(Umatrix[2]*input[i])+(Umatrix[3]*input[i+qbit]); 
	output[j]=(Umatrix[0]*input[j]) + (Umatrix[1]*input[j+qbit]);                 
	output[j+qbit]=(Umatrix[2]*input[j])+(Umatrix[3]*input[j+qbit]);  
}

int main(int argc, char *argv[])
{
    char *trace_file; // Variable that holds trace file name;
    trace_file = argv[1];
    
    // read input matrix and vector from file
    ifstream file(trace_file);

    // Read the 2x2 matrix
    float matrix_1[4];
    for (int i = 0; i < 4; i++) 
    {
        file >> matrix_1[i];
    }

    float matrix_2[4];
    for (int i = 0; i < 4; i++) 
    {
        file >> matrix_2[i];
    }

    float matrix_3[4];
    for (int i = 0; i < 4; i++) 
    {
        file >> matrix_3[i];
    }
    
    float matrix_4[4];
    for (int i = 0; i < 4; i++) 
    {
        file >> matrix_4[i];
    }
    
    float matrix_5[4];
    for (int i = 0; i < 4; i++) 
    {
        file >> matrix_5[i];
    }
    
    float matrix_6[4];
    for (int i = 0; i < 4; i++) 
    {
        file >> matrix_6[i];
    }

    // Read the input vector
    vector<float> values;
    float value;
    while (file >> value) 
    {
      values.push_back(value);
    }

    // Read the qbit values
    float q1,q2,q3,q4,q5,q6;
    q6 = values.back();
    values.pop_back();
    q5 = values.back();
    values.pop_back();
    q4 = values.back();
    values.pop_back();
    q3 = values.back();
    values.pop_back();
    q2 = values.back();
    values.pop_back();
    q1 = values.back();
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
        h_U[i]= matrix_1[i];
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
    int threadsPerBlock = 32;
    int blocksPerGrid = n/64;

    //Timing Report
    //struct timeval begin, end; 
    //gettimeofday (&begin, NULL);

    //Applying Qubit Gate 1
    matrix_multiply<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_U, n, pow(2,q1));
    cudaDeviceSynchronize();
    cudaMemcpy(h_A, d_B, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    for (int i = 0; i < 4; i++) 
    {
        h_U[i]= matrix_2[i];
    }
    cudaMemcpy(d_U, h_U, size_m, cudaMemcpyHostToDevice);
    //Applying Qubit Gate 2
    matrix_multiply<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_U, n, pow(2,q2));
    cudaDeviceSynchronize();
    cudaMemcpy(h_A, d_B, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    for (int i = 0; i < 4; i++) 
    {
        h_U[i]= matrix_3[i];
    }
    cudaMemcpy(d_U, h_U, size_m, cudaMemcpyHostToDevice);
    //Applying Qubit Gate 3
    matrix_multiply<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_U, n, pow(2,q3));
    cudaDeviceSynchronize();
    cudaMemcpy(h_A, d_B, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    for (int i = 0; i < 4; i++) 
    {
        h_U[i]= matrix_4[i];
    }
    cudaMemcpy(d_U, h_U, size_m, cudaMemcpyHostToDevice);
    //Applying Qubit Gate 4
    matrix_multiply<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_U, n, pow(2,q4));
    cudaDeviceSynchronize();
    cudaMemcpy(h_A, d_B, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    for (int i = 0; i < 4; i++) 
    {
        h_U[i]= matrix_5[i];
    }
    cudaMemcpy(d_U, h_U, size_m, cudaMemcpyHostToDevice);
    //Applying Qubit Gate 5
    matrix_multiply<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_U, n, pow(2,q5));
    cudaDeviceSynchronize();
    cudaMemcpy(h_A, d_B, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    for (int i = 0; i < 4; i++) 
    {
        h_U[i]= matrix_6[i];
    }
    cudaMemcpy(d_U, h_U, size_m, cudaMemcpyHostToDevice);
    //Applying Qubit Gate 6
    matrix_multiply<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_U, n, pow(2,q6));
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