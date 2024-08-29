NVCC = nvcc
OUTPUT_DIR = bin

EX_FILES = $(wildcard ex*.cu)
EXECUTABLES = $(patsubst %.cu, $(OUTPUT_DIR)/%, $(EX_FILES))

all: clean $(EXECUTABLES)

$(OUTPUT_DIR)/%: %.cu
	$(NVCC) -I. -I/usr/local/cuda_lite -o $@ $< 

clean:
	rm -f $(EXECUTABLES)