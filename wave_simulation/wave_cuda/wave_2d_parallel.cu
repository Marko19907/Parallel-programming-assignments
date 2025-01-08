#define _XOPEN_SOURCE 600
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <sys/time.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;


// Convert 'struct timeval' into seconds in double prec. floating point
#define WALLTIME(t) ((double)(t).tv_sec + 1e-6 * (double)(t).tv_usec)

// Option to change numerical precision
typedef int64_t int_t;
typedef double real_t;

// Simulation parameters: size, step count, and how often to save the state
int_t
    N = 128,
    M = 128,
    max_iteration = 1000000,
    snapshot_freq = 1000;

// Wave equation parameters, time step is derived from the space step
const real_t
    c  = 1.0,
    dx = 1.0,
    dy = 1.0;
real_t
    dt;

// Buffers for three time steps, indexed with 2 ghost points for the boundary
// Host buffers start with h_, device buffers with d_
real_t *h_buffers[3] = { NULL, NULL, NULL };

real_t *d_buffers[3] = { NULL, NULL, NULL };

#define U_prv(i,j) h_buffers[0][((i)+1)*(N+2)+(j)+1]
#define U(i,j)     h_buffers[1][((i)+1)*(N+2)+(j)+1]
#define U_nxt(i,j) h_buffers[2][((i)+1)*(N+2)+(j)+1]


#define cudaErrorCheck(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess) {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}


// Rotate the time step buffers.
void move_buffer_window ( void )
{
    // Rotate device buffers
    real_t *temp_d = d_buffers[0];
    d_buffers[0] = d_buffers[1];
    d_buffers[1] = d_buffers[2];
    d_buffers[2] = temp_d;
}


// Save the present time step in a numbered file under 'data/'
void domain_save ( int_t step )
{
    char filename[256];
    sprintf ( filename, "data/%.5ld.dat", step );
    FILE *out = fopen ( filename, "wb" );
    for ( int_t i=0; i<M; i++ )
    {
        fwrite ( &U(i,0), sizeof(real_t), N, out );
    }
    fclose ( out );
}


// Get rid of all the memory allocations
void domain_finalize ( void )
{
    free ( h_buffers[0] );
    free ( h_buffers[1] );
    // buffers[2] was not allocated on host

    // Free device memory
    cudaFree(d_buffers[0]);
    cudaFree(d_buffers[1]);
    cudaFree(d_buffers[2]);
}


// Neumann (reflective) boundary condition
__global__ void boundary_condition_kernel(real_t *d_U, int M, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Process vertical boundaries (left and right)
    if (idx < M)
    {
        int i = idx;

        // Left boundary j = -1
        d_U[(i+1)*(N+2) + 0] = d_U[(i+1)*(N+2) + 2]; // U(i,-1) = U(i,1);

        // Right boundary j = N
        d_U[(i+1)*(N+2) + (N+1)] = d_U[(i+1)*(N+2) + (N-1)]; // U(i,N) = U(i,N-2);
    }

    // Process horizontal boundaries (top and bottom)
    if (idx < N)
    {
        int j = idx;

        // Top boundary i = -1
        d_U[0*(N+2) + (j+1)] = d_U[2*(N+2) + (j+1)]; // U(-1,j) = U(1,j);

        // Bottom boundary i = M
        d_U[(M+1)*(N+2) + (j+1)] = d_U[(M-1)*(N+2) + (j+1)]; // U(M,j) = U(M-2,j);
    }
}


// Integration formula
__global__ void time_step_kernel(real_t *d_U_prv, real_t *d_U, real_t *d_U_nxt, int M, int N, real_t dt, real_t c, real_t dx, real_t dy)
{
    cg::thread_block cta = cg::this_thread_block();

    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    #define D_U_prv(i,j) d_U_prv[((i)+1)*(N+2)+(j)+1]
    #define D_U(i,j)     d_U[((i)+1)*(N+2)+(j)+1]
    #define D_U_nxt(i,j) d_U_nxt[((i)+1)*(N+2)+(j)+1]

    if (i < M && j < N)
    {
        D_U_nxt(i,j) = -D_U_prv(i,j) + 2.0*D_U(i,j)
             + (dt*dt*c*c)/(dx*dy) * (
                D_U(i-1,j)+D_U(i+1,j)+D_U(i,j-1)+D_U(i,j+1)-4.0*D_U(i,j)
             );
    }

    cta.sync();
}


// Main time integration.
void simulate( void )
{
    // Set up grid and block dimensions
    dim3 blockDim(16, 16);
    dim3 gridDim( (N + blockDim.x - 1) / blockDim.x,
                  (M + blockDim.y - 1) / blockDim.y );

    int threadsPerBlock = 256;
    int blocksPerGrid = (M > N ? M : N + threadsPerBlock - 1) / threadsPerBlock;

    // Go through each time step
    for ( int_t iteration=0; iteration<=max_iteration; iteration++ )
    {
        if ( (iteration % snapshot_freq)==0 )
        {
            // Copy d_buffers[1] to buffers[1] on the host
            cudaMemcpy(h_buffers[1], d_buffers[1], (M+2)*(N+2)*sizeof(real_t), cudaMemcpyDeviceToHost);

            domain_save ( iteration / snapshot_freq );
        }

        // Derive step t+1 from steps t and t-1
        // Launch boundary_condition_kernel
        boundary_condition_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_buffers[1], M, N);
        cudaErrorCheck(cudaGetLastError());

        // Launch time_step_kernel
        time_step_kernel<<<gridDim, blockDim>>>(d_buffers[0], d_buffers[1], d_buffers[2], M, N, dt, c, dx, dy);
        cudaErrorCheck(cudaGetLastError());

        // Rotate the time step buffers
        move_buffer_window();
    }
}


// GPU occupancy
void occupancy( void )
{
    cudaDeviceProp deviceProp;
    int devID;
    cudaGetDevice(&devID);
    cudaGetDeviceProperties(&deviceProp, devID);

    int blockSize = 16 * 16; // Number of threads per block in our time_step_kernel
    int minGridSize;
    int blockSizeSuggested;
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSizeSuggested, (void*)time_step_kernel, 0, 0);

    int maxActiveBlocksPerMultiprocessor;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &maxActiveBlocksPerMultiprocessor,
        (void*)time_step_kernel,
        blockSize,
        0); // 0 for dynamic shared memory

    int maxThreadsPerMultiprocessor = deviceProp.maxThreadsPerMultiProcessor;
    int activeThreadsPerMultiprocessor = maxActiveBlocksPerMultiprocessor * blockSize;

    int maxWarpsPerMultiprocessor = maxThreadsPerMultiprocessor / deviceProp.warpSize;
    int activeWarpsPerMultiprocessor = activeThreadsPerMultiprocessor / deviceProp.warpSize;

    float occupancy = (float)activeWarpsPerMultiprocessor / maxWarpsPerMultiprocessor;

    printf("Grid size set to: %d\n", minGridSize);
    printf("Launched blocks of size: %d\n", blockSize);
    printf("Theoretical occupancy:\n");
    printf("%d.\n", activeWarpsPerMultiprocessor);
    printf("%d.\n", maxWarpsPerMultiprocessor);
    printf("%f\n", occupancy);
}


// Make sure at least one CUDA-capable device exists
static bool init_cuda()
{
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess || deviceCount == 0) {
        printf("No CUDA devices found\n");
        return false;
    }
    printf("CUDA device count: %d\n", deviceCount);

    int devID = 0; // Select the first device, for now
    err = cudaSetDevice(devID);
    if (err != cudaSuccess) {
        printf("Failed to set device %d\n", devID);
        return false;
    }

    cudaDeviceProp deviceProp;
    err = cudaGetDeviceProperties(&deviceProp, devID);
    if (err != cudaSuccess) {
        printf("Failed to get device properties\n");
        return false;
    }

    printf("CUDA device #%d:\n", devID);
    printf(" Name: %s\n", deviceProp.name);
    printf(" Compute capability: %d.%d\n", deviceProp.major, deviceProp.minor);
    printf(" Supports cooperative groups: %s\n", deviceProp.cooperativeLaunch ? "Yes" : "No");
    printf(" Multiprocessors: %d\n", deviceProp.multiProcessorCount);
    printf(" Warp size: %d\n", deviceProp.warpSize);
    printf(" Global memory: %.1fGiB bytes\n", (float)deviceProp.totalGlobalMem / (1024 * 1024 * 1024));
    printf(" Per-block shared memory: %.1fKiB\n", (float)deviceProp.sharedMemPerBlock / 1024);
    printf(" Per-block registers: %d\n", deviceProp.regsPerBlock);

    return true;
}


// Set up our three buffers, and fill two with an initial perturbation
void domain_initialize ( void )
{
    bool locate_cuda = init_cuda();
    if (!locate_cuda)
    {
        exit( EXIT_FAILURE );
    }

    // Allocate host memory for buffers[0] and buffers[1]
    h_buffers[0] = (real_t *) malloc ( (M+2)*(N+2)*sizeof(real_t) );
    h_buffers[1] = (real_t *) malloc ( (M+2)*(N+2)*sizeof(real_t) );

    // Allocate device memory for d_buffers[0..2]
    cudaMalloc(&d_buffers[0], (M+2)*(N+2)*sizeof(real_t));
    cudaMalloc(&d_buffers[1], (M+2)*(N+2)*sizeof(real_t));
    cudaMalloc(&d_buffers[2], (M+2)*(N+2)*sizeof(real_t));

    // Initialize buffers[0] and buffers[1] on the host
    for ( int_t i=0; i<M; i++ )
    {
        for ( int_t j=0; j<N; j++ )
        {
            // Calculate delta (radial distance) adjusted for M x N grid
            real_t delta = sqrt ( ((i - M/2.0) * (i - M/2.0)) / (real_t)M +
                                ((j - N/2.0) * (j - N/2.0)) / (real_t)N );
            U_prv(i,j) = U(i,j) = exp ( -4.0*delta*delta );
        }
    }

    // Copy h_buffers[0] and h_buffers[1] to device
    cudaMemcpy(d_buffers[0], h_buffers[0], (M+2)*(N+2)*sizeof(real_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_buffers[1], h_buffers[1], (M+2)*(N+2)*sizeof(real_t), cudaMemcpyHostToDevice);

    // Set the time step for 2D case
    dt = dx*dy / (c * sqrt (dx*dx+dy*dy));
}


int main ( void )
{
    // Set up the initial state of the domain
    domain_initialize();

    struct timeval t_start, t_end;

    gettimeofday ( &t_start, NULL );
    simulate();
    gettimeofday ( &t_end, NULL );

    printf ( "Total elapsed time: %lf seconds\n",
        WALLTIME(t_end) - WALLTIME(t_start)
    );

    occupancy();

    // Clean up and shut down
    domain_finalize();
    exit ( EXIT_SUCCESS );
}
