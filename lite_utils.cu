
///////////////////////////////////////
//0. Debugging
///////////////////////////////////////
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true){
   if (code != cudaSuccess){
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

///////////////////////////////////////
//0. Utils
///////////////////////////////////////
__device__ unsigned int* floatToUint(float  *input){
    unsigned char  * temp1 = reinterpret_cast<unsigned char  *>(input);       
    unsigned int  * output = reinterpret_cast<unsigned int  *>(temp1);    
    return output;
}
__device__ float * uintToFloat(unsigned int  *input){
    unsigned char  * temp1 = reinterpret_cast<unsigned char  *>(input);       
    float  * output = reinterpret_cast<float  *>(temp1);    
    return output;
}
void floatToUintCPU(uint *dest, const float *source, int N) {
    for(int i=0; i<N; i++) memcpy(&dest[i], &source[i], sizeof(float));
}
void uintToFloatCPU(float *dest, const uint *source, int N) {
    for(int i=0; i<N; i++) memcpy(&dest[i], &source[i], sizeof(uint));
}
int padArray(uint* arr, int N) {
    int paddingSize = 0;
    int remainder = N % 4;
    if (remainder != 0) {
        paddingSize = 4 - remainder;
        int newSize = N + paddingSize;
        uint* tempArr = new uint[newSize];
        for (int i = 0; i < N; i++) tempArr[i] = arr[i];
        for (int i = N; i < newSize; i++) tempArr[i] = 0;
        arr = tempArr;
    }
    return paddingSize;
}
void removePadArray(uint* arr, int N, int paddingSize) {
    int newSize = N - paddingSize;
    uint* tempArr = new uint[newSize];
    for (int i = 0; i < newSize; i++) tempArr[i] = arr[i];
    arr = tempArr;
}

int padMatrix(uint* matrix, int& width, int& height) {
    int paddingSize = 0;
    int widthRemainder = width % 4;
    int heightRemainder = height % 4;

    if (widthRemainder != 0) {
        int newWidth = width + (4 - widthRemainder);
        paddingSize += 4 - widthRemainder;

        uint* tempMatrix = new uint[newWidth * height];
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                tempMatrix[i * newWidth + j] = matrix[i * width + j];
            }
            for (int j = width; j < newWidth; j++) {
                tempMatrix[i * newWidth + j] = 0;  // Pad with zeros
            }
        }
        matrix = tempMatrix;
        width = newWidth;
    }

    if (heightRemainder != 0) {
        int newHeight = height + (4 - heightRemainder);
        paddingSize += (4 - heightRemainder) * width;

        uint* tempMatrix = new uint[width * newHeight];
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                tempMatrix[i * width + j] = matrix[i * width + j];
            }
        }
        for (int i = height; i < newHeight; i++) {
            for (int j = 0; j < width; j++) {
                tempMatrix[i * width + j] = 0;  // Pad with zeros
            }
        }
        matrix = tempMatrix;
        height = newHeight;
    }

    return paddingSize;
}

void removePadMatrix(int* matrix, int& width, int& height, int paddingSize) {
    int newWidth = width - (paddingSize % width);
    int newHeight = height - (paddingSize / width);

    int* tempMatrix = new int[newWidth * newHeight];
    for (int i = 0; i < newHeight; i++) {
        for (int j = 0; j < newWidth; j++) {
            tempMatrix[i * newWidth + j] = matrix[i * width + j];
        }
    }

    delete[] matrix;
    matrix = tempMatrix;
    width = newWidth;
    height = newHeight;
}