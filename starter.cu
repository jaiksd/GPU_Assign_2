/**
*   CS6023: GPU Programming 
*   Assignment 2
*   
*   Please don't change any existing code in this file.
*
*   Please add necessary memory APIs for your implementation. Use cudaFree() 
*   to free up memory as soon as you're done with an allocation. 
*   This will ensure that you don't run out of memory while running
*   large test cases. Use the minimum required memory for your 
*   implementation. DO NOT change the kernel configuration parameters.
*/

#include <chrono>
#include <fstream>
#include <iostream>
#include <stdio.h>
#include <cuda.h>

using namespace std;

using std::cin;
using std::cout;

typedef long long ll;


//kernel for convolution
__global__ void dkernel(long int* a1, long int* b1, long int* ans, int m, int n, int k) {
    //creating shared memoery 
    extern __shared__ long int s[];

    unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned i = id / n;
    unsigned j = id % n;

    // Initialize shared memory for caching input matrices
    if (threadIdx.x == 0) {
        for (long int i = 0; i < k * k; i++) {
            s[i] = b1[i];
        }
    }
    __syncthreads();


       // Compute convolution using cached input matrices in shared memory
    long int sum = 0;
    for (long int a = -k / 2; a <= k / 2; a++) {
        if (i + a < 0 || i + a >= m)
            continue;

        for (long int b = -k / 2; b <= k / 2; b++) {
            if (j + b < 0 || j + b >= n)
                continue;

            sum += a1[(i + a) * n + (b + j)] * s[(a + k / 2) * k + (b + k / 2)];
        }
    }
    //appying coalesing here
    // Writing result to output matrix with coalesced memory access
    ans[id] = sum;
}



int main(int argc, char** argv) {

    int m,n,k;
    cin>>m>>n>>k;


    long int* h_mat = new long int[m * n];
    long int* h_filter = new long int[k * k];

    long int* h_ans = new long int[m * n];


    for (long int i = 0; i < m * n; i++) {
        cin>>h_mat[i];
    }

    for (long int i = 0; i < k * k; i++) {
        cin>>h_filter[i];
    }

    /**
     * 
     * DO NOT CHANGE ANYTHING ABOVE THIS LINE
     * 
    **/

    /****************************************************Start Here***********************************************************/
    
    long int* g_ans;
    long int* g_mat;
    long int* g_filter;
    cudaMalloc(&g_mat, sizeof(long int) * m * n);
    cudaMalloc(&g_filter, sizeof(long int) * k * k);
    cudaMalloc(&g_ans, sizeof(long int) * m * n);
    cudaMemcpy(g_mat, h_mat, sizeof(long int) * m * n, cudaMemcpyHostToDevice);
    cudaMemcpy(g_filter, h_filter, sizeof(long int) * k * k, cudaMemcpyHostToDevice);

    //kernel launch
    auto start = std::chrono::high_resolution_clock::now();//keep it just before the kernel launch
    dkernel << <m, n, sizeof(long int)* (k * k) >> > (a1, b1, an, m, n, k);
    auto end = std::chrono::high_resolution_clock::now();//keep it just after the kernel launch
    
    //copying the final output to h_ans
    cudaMemcpy(h_ans, g_ans, sizeof(long int) * m * n, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    cudaFree(g_ans);
    cudaFree(g_mat);
    cudaFree(g_filter);
    
    
    /*$$$$$$$$$$$$$$$$$$$$$$$$Make sure your final output from the device is stored in h_ans.$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$*/
    std::chrono::duration<double> elapsed1 = end - start;
    /**
     * 
     * DO NOT CHANGE ANYTHING BELOW THIS LINE
     * 
    */


    
    std::ofstream file("cuda.out");
    if (file.is_open()) {
        for (long int i = 0; i < m; i++) {
            for (long int j = 0; j < n; j++) {
                file << h_ans[i * n + j] << " ";
            }
            file << "\n";
        }
        file.close();
    } else {
        std::cout << "Unable to open file";
    }

    std::ofstream file2("cuda_timing.out");
    if(file2.is_open()) {
        file2 << elapsed1.count() << "\n";
        file2.close();
    } else {
        std::cout << "Unable to open file";
    }

    return 0;
}