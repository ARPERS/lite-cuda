NVCC = nvcc

OUTPUT_DIR = bin

EX_FILES = $(wildcard $(SRC_DIR)/ex*.cu)

EXECUTABLES = $(patsubst %.cu,$(OUTPUT_DIR)/%,$(EX_FILES))

all: $(EXECUTABLES)

$(OUTPUT_DIR)/%: $(SRC_DIR)/%.cu
	$(NVCC) -o $@ $<

clean:
	rm -f $(EXECUTABLES)
