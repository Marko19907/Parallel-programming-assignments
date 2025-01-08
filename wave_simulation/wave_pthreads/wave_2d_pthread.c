#define _XOPEN_SOURCE 600
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <errno.h>
#include <sys/time.h>

#include <pthread.h>

// Option to change numerical precision
typedef int64_t int_t;
typedef double real_t;


// Pthread management
int_t n_threads = 1;
pthread_barrier_t barrier;

// Performance measurement
struct timeval t_start, t_end;
#define WALLTIME(t) ((double)(t).tv_sec + 1e-6 * (double)(t).tv_usec)

// Simulation parameters: size, step count, and how often to save the state
const int_t
    N = 1024,
    max_iteration = 4000,
    snapshot_freq = 20;

// Wave equation parameters, time step is derived from the space step
const real_t
    c  = 1.0,
    h  = 1.0;
real_t
    dt;

// Buffers for three time steps, indexed with 2 ghost points for the boundary
real_t
    *buffers[3] = { NULL, NULL, NULL };

#define U_prv(i,j) buffers[0][((i)+1)*(N+2)+(j)+1]
#define U(i,j)     buffers[1][((i)+1)*(N+2)+(j)+1]
#define U_nxt(i,j) buffers[2][((i)+1)*(N+2)+(j)+1]


// Rotate the time step buffers.
void move_buffer_window ( void )
{
    real_t *temp = buffers[0];
    buffers[0] = buffers[1];
    buffers[1] = buffers[2];
    buffers[2] = temp;
}


// Set up our three buffers, and fill two with an initial perturbation
void domain_initialize ( void )
{
    buffers[0] = malloc ( (N+2)*(N+2)*sizeof(real_t) );
    buffers[1] = malloc ( (N+2)*(N+2)*sizeof(real_t) );
    buffers[2] = malloc ( (N+2)*(N+2)*sizeof(real_t) );

    for ( int_t i=0; i<N; i++ )
    {
        for ( int_t j=0; j<N; j++ )
        {
            real_t delta = sqrt ( ((i-N/2)*(i-N/2)+(j-N/2)*(j-N/2))/(real_t)N );
            U_prv(i,j) = U(i,j) = exp ( -4.0*delta*delta );
        }
    }

    // Set the time step
    dt = (h*h) / (4.0*c*c);
}


// Get rid of all the memory allocations
void domain_finalize ( void )
{
    free ( buffers[0] );
    free ( buffers[1] );
    free ( buffers[2] );
}


// Integration formula
void time_step ( int_t thread_id )
{
    int_t chunk_size = N / n_threads;
    int_t i_start = thread_id * chunk_size;
    int_t i_end = (thread_id == n_threads -1) ? N : i_start + chunk_size;

    for ( int_t i=i_start; i<i_end; i+=1 )
        for ( int_t j=0; j<N; j++ )
            U_nxt(i,j) = -U_prv(i,j) + 2.0*U(i,j)
                     + (dt*dt*c*c)/(h*h) * (
                        U(i-1,j)+U(i+1,j)+U(i,j-1)+U(i,j+1)-4.0*U(i,j)
                     );
}


// Neumann (reflective) boundary condition
void boundary_condition ( int_t thread_id )
{
    int_t chunk_size = N / n_threads;
    int_t i_start = thread_id * chunk_size;
    int_t i_end = (thread_id == n_threads -1) ? N : i_start + chunk_size;

    // Update boundary conditions for rows
    for ( int_t i=i_start; i<i_end; i++ )
    {
        U(i,-1) = U(i,1);
        U(i,N)  = U(i,N-2);
    }

    // Synchronize before updating columns
    pthread_barrier_wait(&barrier);

    int_t j_start = thread_id * chunk_size;
    int_t j_end = (thread_id == n_threads -1) ? N : j_start + chunk_size;

    // Update boundary conditions for columns
    for ( int_t j=j_start; j<j_end; j++ )
    {
        U(-1,j) = U(1,j);
        U(N,j)  = U(N-2,j);
    }
}


// Save the present time step in a numbered file under 'data/'
void domain_save ( int_t step )
{
    char filename[256];
    sprintf ( filename, "data/%.5ld.dat", step );
    FILE *out = fopen ( filename, "wb" );
    for ( int_t i=0; i<N; i++ )
        fwrite ( &U(i,0), sizeof(real_t), N, out );
    fclose ( out );
}


// Main loop
void *simulate ( void *id )
{
    int_t thread_id = *((int_t *)id);

    // Go through each time step
    for ( int_t iteration=0; iteration<=max_iteration; iteration++ )
    {
        if ( (iteration % snapshot_freq)==0 )
        {
            // Ensure only one thread saves the domain state
            if (thread_id == 0)
                domain_save ( iteration / snapshot_freq );
        }

        // Derive step t+1 from steps t and t-1
        boundary_condition(thread_id);

        // Synchronize threads before proceeding to time_step
        pthread_barrier_wait(&barrier);

        // Now perform the time step
        time_step(thread_id);

        // Synchronize threads before moving buffer window
        pthread_barrier_wait(&barrier);

        // Rotate the time step buffers
        // Ensure only one thread rotates the buffers
        if (thread_id == 0)
            move_buffer_window();

        // Synchronize before next iteration
        pthread_barrier_wait(&barrier);
    }

    return NULL;
}


// Main time integration loop
int main ( int argc, char **argv )
{
    // Number of threads is an optional argument, sanity check its value
    if ( argc > 1 )
    {
        n_threads = strtol ( argv[1], NULL, 10 );
        if ( errno == EINVAL )
            fprintf ( stderr, "'%s' is not a valid thread count\n", argv[1] );
        if ( n_threads < 1 )
        {
            fprintf ( stderr, "Number of threads must be >0\n" );
            exit ( EXIT_FAILURE );
        }
    }

    // Initialise pthreads
    pthread_barrier_init(&barrier, NULL, n_threads);

    // Set up the initial state of the domain
    domain_initialize();

    // Time the execution
    gettimeofday ( &t_start, NULL );

    // Run the integration loop
    pthread_t threads[n_threads];
    int_t thread_ids[n_threads];

    for (int_t t=0; t<n_threads; t++) {
        thread_ids[t] = t;
        pthread_create(&threads[t], NULL, simulate, (void *)&thread_ids[t]);
    }

    for (int_t t=0; t<n_threads; t++) {
        pthread_join(threads[t], NULL);
    }

    // Report how long we spent in the integration stage
    gettimeofday ( &t_end , NULL );
    printf ( "%lf seconds elapsed with %ld threads\n",
        WALLTIME(t_end)-WALLTIME(t_start),
        n_threads
    );

    // Clean up and shut down
    domain_finalize();

    // Finalise pthreads
    pthread_barrier_destroy(&barrier);

    exit ( EXIT_SUCCESS );
}
