run: bin/benchmark
	@echo "Image size = 128*128"
	OMP_NUM_THREADS=8 ./bin/benchmark 128 5 25
	@echo "--------------------------------"
	@echo "Image size = 256*256"
	OMP_NUM_THREADS=8 ./bin/benchmark 256 5 25
	@echo "--------------------------------"
	@echo "Image size = 512*512"
	OMP_NUM_THREADS=8 ./bin/benchmark 512 5 25
	@echo "--------------------------------"
	@echo "Image size = 1024*1024"
	OMP_NUM_THREADS=8 ./bin/benchmark 1024 5 25
	@echo "--------------------------------"
	@echo "Image size = 2048*2048"
	OMP_NUM_THREADS=8 ./bin/benchmark 2048 5 25
	@echo "--------------------------------"
	@echo "Image size = 4096*4096"
	OMP_NUM_THREADS=8 ./bin/benchmark 4096 5 25
	@echo "--------------------------------"
	@echo "Image size = 8192*8192"
	OMP_NUM_THREADS=8 ./bin/benchmark 8192 5 25

bin/benchmark: bin/intimg.o bin/omp.o bin/cuda1.o bin/cuda2.o bin/utils.o
	nvcc src/benchmark.cu bin/intimg.o bin/omp.o bin/cuda1.o bin/cuda2.o bin/utils.o -lgomp -o bin/benchmark

bin/utils.o: src/utils.cpp src/utils.h
	mkdir -p bin
	g++ -c src/utils.cpp -o bin/utils.o

bin/intimg.o: src/intimg.cpp src/intimg.h
	mkdir -p bin
	g++ -c src/intimg.cpp -o bin/intimg.o

bin/omp.o: src/intimg_omp.cpp src/intimg_omp.h
	mkdir -p bin
	g++ -c src/intimg_omp.cpp -fopenmp -o bin/omp.o

bin/cuda1.o: src/intimg.cu src/intimg.cuh bin/utils.o
	mkdir -p bin
	nvcc -c src/intimg.cu -o bin/cuda1.o

bin/cuda2.o: src/intimg2.cu src/intimg2.cuh bin/utils.o
	mkdir -p bin
	nvcc -c src/intimg2.cu -o bin/cuda2.o

all: bin/benchmark

clean:
	rm -rf *.o *.out benchmark bin
