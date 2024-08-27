# objects = ex_2_matrix_multiplication_uint.o

# all: $(objects)
# 	nvcc -arch=sm_86 $(objects) -o app

# %.o: %.cu
# 	nvcc -arch=sm_86 -I. -dc $< -o $@

# clean:
# 	rm -f *.o app

objects = matrixMul.o

all: clean $(objects)
	nvcc -arch=sm_86 $(objects) -o app

%.o: %.cu
	nvcc -arch=sm_86 -I. -dc $< -o $@

clean:
	rm -f *.o app