#include <cstdio>
#include <assert.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cuda/barrier>

template <int NUM_PER_THREADS>
__global__ void copy_to_shared_sequential(int *data)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    extern __shared__ int shm[];

    for (int i = 0; i < NUM_PER_THREADS; ++i)
    {
        shm[threadIdx.x * NUM_PER_THREADS + i] = data[idx * NUM_PER_THREADS + i];
    }
    __syncthreads();
}

template <int NUM_PER_THREADS>
__global__ void copy_to_shared_async(int *data)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    auto block = cooperative_groups::this_thread_block();
    extern __shared__ int shm[];

    __shared__ cuda::barrier<cuda::thread_scope::thread_scope_block> barrier;
    if (block.thread_rank() == 0)
    {
        init(&barrier, block.size());
    }
    block.sync();

    for (int i = 0; i < NUM_PER_THREADS; ++i)
    {
        cuda::memcpy_async(block, shm + threadIdx.x * NUM_PER_THREADS + i, data + idx * NUM_PER_THREADS + i, sizeof(int), barrier);
    }
    barrier.arrive_and_wait();
}

int main()
{
    int N = 1024 * 1024;
    int size_data = N * sizeof(int);

    // Allocate memory
    int *data_host, *data_device;
    cudaMallocHost(&data_host, size_data);
    cudaMalloc(&data_device, size_data);

    // Initialize data
    for (int i = 0; i < N; ++i)
        data_host[i] = i % 17;
    cudaMemcpy(data_device, data_host, size_data, cudaMemcpyHostToDevice);

    // Configure kernel
    constexpr int BLOCK_SIZE = 1024;
    constexpr int NUM_PER_THREADS = 8;
    constexpr int SHM_SIZE = BLOCK_SIZE * NUM_PER_THREADS * sizeof(int);
    assert(N % (BLOCK_SIZE * NUM_PER_THREADS) == 0);
    dim3 blockDim(BLOCK_SIZE);
    dim3 gridDim(N / BLOCK_SIZE / NUM_PER_THREADS);

    // Run
    copy_to_shared_sequential<NUM_PER_THREADS><<<gridDim, blockDim, SHM_SIZE>>>(data_device);
    copy_to_shared_async<NUM_PER_THREADS><<<gridDim, blockDim, SHM_SIZE>>>(data_device);

    // printf("%s\n", cudaGetErrorString(cudaGetLastError()));
    return 0;
}