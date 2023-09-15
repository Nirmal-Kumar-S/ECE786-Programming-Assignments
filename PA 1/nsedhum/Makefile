# Location of the CUDA Toolkit
NVCC := /usr/local/cuda/bin/nvcc
CCFLAGS := -O2

build: quamsimV1 quamsimV2

quamsimV1.o:quamsimV1.cu
	$(NVCC) $(INCLUDES) $(CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

quamsimV1: quamsimV1.o
	$(NVCC) $(LDFLAGS) $(GENCODE_FLAGS) -o $@ $+ $(LIBRARIES)

quamsimV2.o:quamsimV2.cu
	$(NVCC) $(INCLUDES) $(CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

quamsimV2: quamsimV2.o
	$(NVCC) $(LDFLAGS) $(GENCODE_FLAGS) -o $@ $+ $(LIBRARIES)

run: build
	$(EXEC) ./quamsimV1
	$(EXEC) ./quamsimV2

clean:
	rm -f quamsimV1 quamsimV2 *.o