# Define object files for each program
objects1 = matrixMul.o
objects2 = secure_matrixMul.o

# Default target
all: clean app secure_app

# Target to build the first application
app: $(objects1)
	nvcc -arch=sm_86 $(objects1) -o app

# Target to build the second application
secure_app: $(objects2)
	nvcc -arch=sm_86 $(objects2) -o secure_app

# Pattern rule to compile .cu files into .o files
%.o: %.cu
	nvcc -arch=sm_86 -I. -dc $< -o $@

# Clean up object files and binaries
clean:
	rm -f *.o app secure_app
