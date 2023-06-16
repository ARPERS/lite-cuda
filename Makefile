CXXFLAGS = -o
EXE_NAME = test_literian
CPP_SOURCES = AES/AES.cpp AES/AES_encrypt.cpp AES/AES_decrypt.cpp

CUDA_CXXFLAGS = -o
CUDA_EXE_NAME = test_literian
CUDA_SOURCES = vector_addition_secured.cu
NVCCLFLAGS = -arch=sm_70 -cudart=shared -rdc=true

CPP_OBJECTS = $(CPP_SOURCES:.cpp=.o)
CUDA_OBJECTS = $(CUDA_SOURCES:.cu=.o)

EXECUTABLE = program

all: $(EXECUTABLE)

$(EXECUTABLE): $(CPP_OBJECTS) $(CU_OBJECTS)
	$(CC)  $(CPP_OBJECTS) $(CU_OBJECTS) -o $@

%.o: %.cpp
	g++ $< -o $@

%.o: %.cu
	nvcc $< -o $@

clean:
	rm -f $(CPP_OBJECTS) $(CU_OBJECTS) $(EXECUTABLE)