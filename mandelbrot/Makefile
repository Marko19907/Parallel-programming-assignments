CC = gcc
NVCC = nvcc
CFLAGS = -O3
NVCCFLAGS = -O3
TARGET = mandel

all: $(TARGET)

$(TARGET): mandel.cu
	$(NVCC) $(NVCCFLAGS) -o $(TARGET) mandel.cu

check: $(TARGET)
	./$(TARGET) 1

clean:
	rm -f $(TARGET)
