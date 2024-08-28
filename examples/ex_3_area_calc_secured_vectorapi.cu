/*
Calculating secure area computation using available vector API from LITE.
Unfortunately, the final summation that not implemented in LITE should run in CPU
*/
#include <iostream>
#include <typeinfo>
#include <cmath>

#include <cuda.h>
#include "../lib/lite.cu"

#include <chrono>
#include <fstream>
#include <sstream>


float* shiftLeft(const float arr[], size_t size) {
    if (size <= 1) {
        float* shifted = new float[size];
        for (size_t i = 0; i < size; i++) shifted[i] = arr[i];
        return shifted;
    }
    float* shifted = new float[size];
    int first_element = arr[0];
    for (size_t i = 0; i < size - 1; i++) shifted[i] = arr[i + 1];
    shifted[size - 1] = first_element; // Wrap around to the first element
    return shifted;
}

int main() {

    
    using std::chrono::high_resolution_clock;
    using std::chrono::duration;
    using std::chrono::milliseconds;

    const int n_points = 299;
    float h_f_x[n_points]; // Host arrays for Float x-coordinates
    float h_f_y[n_points]; // Host arrays for Float y-coordinates 48.555

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
        h_f_x[i] = a;
        h_f_y[i] = b;
        i++;
    }
    
    // 0. Initiate key in CPU
    uchar key[] = { 0x00, 0x00, 0x00, 0x00,
                    0x00, 0x00, 0x00, 0x00,
                    0x00, 0x00, 0x00, 0x00,
                    0x00, 0x00, 0x00, 0x00 };
    uint keySize = 16;
    int Nr=10;
    uint e_sched[4*(MAXNR + 1)];
    uint d_sched[4*(MAXNR + 1)];
    makeKey(key, keySize << 3, DIR_BOTH, e_sched, d_sched, Nr);

    float *shifted_h_f_x = shiftLeft(h_f_x, n_points);
    float *shifted_h_f_y = shiftLeft(h_f_y, n_points);

    float *c = new float[n_points];
    float *d = new float[n_points];
    int gridSize=1;
    int blockSize=min(128, 4*((n_points+3)/4)); // find closest divible number to 4 that larger or equal to n_points, and less than 128
                                                // why 128? because we limit the maximum number of threads per block to avoid register spilling
                                                // check lite_vector.cu for more details

    cout << "gridSize: " << gridSize << endl;
    cout << "blockSize: " << blockSize << endl;

    double avg_time = 0; 
    for(int ii=0; ii<50; ii++){
        auto t1 = high_resolution_clock::now();
        liteMultiplication(d, shifted_h_f_x, h_f_y, n_points, e_sched, d_sched, Nr, gridSize, blockSize);
        liteMultiplication(c, h_f_x, shifted_h_f_y, n_points, e_sched, d_sched, Nr, gridSize, blockSize);
        liteSubstraction(d, c, d, n_points, e_sched, d_sched, Nr, gridSize, blockSize);
        int total = 0;
        for(int i=0;i<n_points;i++){
            total+=d[i];
        }
        auto t2 = high_resolution_clock::now();

        std::cout << "Area: " << (float)abs(total)*0.5 << std::endl;

        duration<double, std::milli> ms_double = t2 - t1;
        avg_time += ms_double.count();
        std::cout << avg_time << std::endl;
        cudaDeviceReset();
    }
    avg_time /= 50;
    std::cout << "Average time: " << avg_time << " ms" << std::endl;

    return 0;
}
