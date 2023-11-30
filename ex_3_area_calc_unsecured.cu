#include <iostream>
#include <typeinfo>
#include <cmath>
#include <chrono>
#include <fstream>
#include <sstream>

__global__ void calculatePolygonArea(float* area, float* x, float* y, int n_points) {
    __shared__ float sharedArea[1];
    sharedArea[0] = 0.0f;
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int next_idx = (idx + 1) % n_points;

    float term = (x[idx] * y[next_idx] - x[next_idx] * y[idx]);
    
    atomicAdd(sharedArea, term);
    __syncthreads();

    if (threadIdx.x == 0) {
        atomicAdd(area, 0.5f * sharedArea[0]);
    }
}

int main() {

    using std::chrono::high_resolution_clock;
    using std::chrono::duration;
    using std::chrono::milliseconds;

    int n_points = 299;
    float h_x[n_points]; // Host arrays for x-coordinates
    float h_y[n_points]; // Host arrays for y-coordinates 37.7624

    // read from /data/points.txt file
    // the file format has 299 rows contains two float values separated with a space:
    // a1 b1
    // a2 b2
    // ...
    // first column is for h_f_x and the second column is for h_f_y
    std::ifstream infile("data/points.txt");
    std::string line;
    int i = 0;
    while (std::getline(infile, line))
    {
        std::istringstream iss(line);
        float a, b;
        if (!(iss >> a >> b)) { break; } // error
        h_x[i] = a;
        h_y[i] = b;
        i++;
    }

    
    double avg_time = 0; 
    for(int i = 0; i < 50; i++){
        auto t1 = high_resolution_clock::now();

        float* d_x;     // Device array for x-coordinates
        float* d_y;     // Device array for y-coordinates

        float h_area = 0;        // Host variable to store the area
        // Allocate memory on the device
        float* d_area;  // Device variable to store the area
        cudaMalloc((void**)&d_area, sizeof(float));
        cudaMalloc((void**)&d_x, sizeof(float) * n_points);
        cudaMalloc((void**)&d_y, sizeof(float) * n_points);

        // Copy data from host to device
        cudaMemcpy(d_area, &h_area, sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_x, h_x, sizeof(float) * n_points, cudaMemcpyHostToDevice);
        cudaMemcpy(d_y, h_y, sizeof(float) * n_points, cudaMemcpyHostToDevice);

        // Launch the CUDA kernel
        calculatePolygonArea<<<1, n_points>>>(d_area, d_x, d_y, n_points);

        // Copy the result back from the device
        cudaMemcpy(&h_area, d_area, sizeof(float), cudaMemcpyDeviceToHost);

        auto t2 = high_resolution_clock::now();
        // Free device memory
        cudaFree(d_area);
        cudaFree(d_x);
        cudaFree(d_y);

        std::cout << "Area of the polygon: " << abs(h_area) << std::endl;

        duration<double, std::milli> ms_double = t2 - t1;
        avg_time += ms_double.count();
        std::cout << "Time: " << avg_time << std::endl;
        cudaDeviceReset();
    }
    std::cout << "Average Time: " << avg_time / 50 << std::endl;
    return 0;
}
