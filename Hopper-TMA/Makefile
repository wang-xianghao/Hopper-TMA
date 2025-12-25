NVCC = nvcc
NVCC_FLAGS = -O2 -lineinfo -arch=native
NCU = ncu
NCU_FLAGS = -f --import-source yes --set full
# NCU_FLAGS = -f --import-source yes --set full

TARGET = main
SRC = main.cu

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(NVCC_FLAGS) -o $@ $^

profile: $(TARGET)
	$(NCU) $(NCU_FLAGS) -o $< $<	

clean:
	rm -f $(TARGET)
	rm -rf $(TARGET).ncu-rep
	rm -f *.o