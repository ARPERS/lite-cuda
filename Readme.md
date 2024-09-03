# LITE

version 1.0.0

## File Organization:


<table><tbody><tr><th><p><strong>LITE/lib/</strong></p></th><th><p>Contains main LITE files:</p><ul><li>`lite_encdec` : contains encryption and decryptin APIs</li><li>lite_matrix : contains matrix multiplication APIs</li><li>lite_vector : contains vector operation APIs</li><li>lite_utils : contains tools used in lite (.e.g, floatToInt())</li></ul></th></tr><tr><td><p><strong>LITE/examples/</strong></p></td><td><p>Contains several files of LITE APIs usage examples. Some files contain unsecure implementation (no LITE) used for comparison (as microbenchmark)</p></td></tr><tr><td><p><strong>LITE/benchmark/</strong></p></td><td><p>Contains secured and unsecured Parboil’s Simple-MM implementation</p></td></tr></tbody></table>

## Setup

- **Requirements:** Nvidia CUDA compiler (NVCC)
- The main LITE library is in the /lib/ folder. To make it easily accessible it is recommended to create symbolic links in the system to lib/lite.cu, lib/lite.h, and lib/
- In the project’s makefile, include the path to the /lib/ folder, for example:

```
NVCCFLAGS += -I/usr/local/cuda_libs
```

Or check the makefile in the /examples/

- Test if the program can access LITE files by compile (run make in the /examples) and run the examples binary file

## API

Details of all the available APIs can be found in the source codes. Here we show some important APIs:

**Encryption Decryption API**

```
ltEncryptGPU(uint \*ct, uint \*pt, uint \*rek, uint Nr, int N)
ltEncryptGPU(uint \*ct, float \*pt, uint \*rek, uint Nr, int N)
```
Lite encryption API. It encrypts N data in GPU using AES algorithm. N should be divisible by 4

*Parameters*

- ct : cipher text (for storing the output)
- pt : plain text (the input), if the plain text is float it will be converted into uint
- rek : AES encryption key
- Nr : AES parameters (unused with default value)
- N : Total length of the cipher text / plain text

*Output*

Encrypted plain text is stored in ct
___

```
ltDecryptGPU(uint \*pt, uint \*ct, uint \*rek, uint Nr, int N)
ltDecryptGPU(float \*pt, uint \*ct, uint \*rek, uint Nr, int N)
```
Lite decryption API. It decrypts N data in GPU using AES algorithm. N should be divisible by 4

*Parameters*

- pt :plain text (for storing the output)
- ct : cipher text (the input)
- rek : AES decryption key
- Nr : AES parameters (unused with default value)
- N : Total length of the cipher text / plain text

*Output*

Decrypted cipher text is stored in pt
___
```        
ltEncryptCPU(uint \*pt, uint \*ct, uint \*rek, uint Nr, int N)
ltEncryptCPU(float \*pt, uint \*ct, uint \*rek, uint Nr, int N)
```
Lite encryption API. It decrypts N data in CPU using AES algorithm. N should be divisible by 4

*Parameters*

- ct : cipher text (for storing the output)
- pt : plain text (the input), if the plain text is float it will be converted into uint
- rek : AES encryption key
- Nr : AES parameters (unused with default value)
- N : Total length of the cipher text / plain text

*Output*

Encrypted plain text is stored in ct
___
```
ltDecryptCPU(uint \*pt, uint \*ct, uint \*rek, uint Nr, int N)
ltDecryptCPU(float \*pt, uint \*ct, uint \*rek, uint Nr, int N)
```
Lite decryption API. It decrypts N data in CPU using AES algorithm. N should be divisible by 4

*Parameters*

- pt :plain text (for storing the output)
- ct : cipher text (the input)
- rek : AES decryption key
- Nr : AES parameters (unused with default value)
- N : Total length of the cipher text / plain text

*Output*

Decrypted cipher text is stored in pt
___

**Vector Operation API**
```
ltVecAdd (uint \*result, uint \*a, uint \*b, int N, uint \*enc_sched, uint \*dec_sched, int Nr, int gridSize, int blockSize)
ltVecAdd (float \*result, float \*a, float \*b, int N, uint \*enc_sched, uint \*dec_sched, int Nr, int gridSize, int blockSize)
```

Lite API for vector addition. It receives two N data arrays in CPU, a and b, then it will securely do vector addition in GPU. It also apply to other vector operation API: ltVecSub, ltVecMul, ltVecDiv.

*Parameters*

- result : result of vector addition
- a and b : the vector to be added, if they are float, they will be converted into uint
- N : Total length of the vector
- enc_sched : encryption key
- dec_sched : decryption key
- Nr : AES parameters (unused with default value)
- gridSize : GPU Grid size / number of block that will be used for computation
- blockSize: GPU Block size / number of thread that will be used for computation

**Matrix Multiplication API**
```
- ltMatMul (uint \*result, uint \*A, uint \*B, int N, uint \*enc_sched, uint \*dec_sched, int Nr)
- ltMatMul (float \*result, float \*A, float \*B, int N, uint \*enc_sched, uint \*dec_sched, int Nr)
```
Lite API for vector matrix multiplicatoin. It receives two one-dimensional NxN data arrays in CPU, A and B, then it will securely do matrix multiplication in GPU. Grid size and block size will be computed automatically in ltMatrixMultiplication function.

*Parameters*

- result : result of matrix multiplication
- A and B : the matrix to be added, if they are float, they will be converted into uint
- N : The matrix size for one dimension. The matrix size is N x N, where N is divisible by 4.
- enc_sched : encryption key
- dec_sched : decryption key
- Nr : AES parameters (unused with default value)

## Limitation

Current LITE library does not covered all optimizations presented in the paper. That is because most of this optimization are case-based optimizations. We implemented padding for vector operation APIs but not in the matrix multiplication. We also hard coded the size of the shared variables in the matrix multiplication based on AES capability, modification of this size may improve LITE performance, with risk some data can be spilled to the GPU memory if it is too large