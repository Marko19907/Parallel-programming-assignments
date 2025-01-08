#define _XOPEN_SOURCE 600
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <sys/time.h>

#include "argument_utils.h"

#include <mpi.h>


// Option to change numerical precision
typedef int64_t int_t;
typedef double real_t;

// Declare variables each MPI process will need
int world_size;
int world_rank;

MPI_Comm cart_comm;
int cart_rank;
int coords[2];
int dims[2];
int periods[2] = {0, 0};

int rank_left, rank_right, rank_up, rank_down;

int local_M;
int local_N;

// Simulation parameters: size, step count, and how often to save the state
int_t M = 256, N = 256;
int_t max_iteration = 4000;
int_t snapshot_freq = 20;

real_t
    c  = 1.0,
    dx = 1.0,
    dy = 1.0;
real_t dt;

// Buffers for three time steps, indexed with 2 ghost points for the boundary
real_t *buffers[3] = { NULL, NULL, NULL };

// Macros for accessing the buffers
#define U_prv(i,j) buffers[0][((i)+1)*(local_N+2)+(j)+1]
#define U(i,j)     buffers[1][((i)+1)*(local_N+2)+(j)+1]
#define U_nxt(i,j) buffers[2][((i)+1)*(local_N+2)+(j)+1]

// Convert 'struct timeval' into seconds in double prec. floating point
#define WALLTIME(t) ((double)(t).tv_sec + 1e-6 * (double)(t).tv_usec)

// Rotate the time step buffers.
void move_buffer_window ( void )
{
    real_t *temp = buffers[0];
    buffers[0] = buffers[1];
    buffers[1] = buffers[2];
    buffers[2] = temp;
}

// Set up our three buffers, and fill two with an initial perturbation
// and set the time step.
void domain_initialize ( void )
{
    local_M = M / dims[0];
    local_N = N / dims[1];

    buffers[0] = malloc ( (local_M+2)*(local_N+2)*sizeof(real_t) );
    buffers[1] = malloc ( (local_M+2)*(local_N+2)*sizeof(real_t) );
    buffers[2] = malloc ( (local_M+2)*(local_N+2)*sizeof(real_t) );

    // Compute the global starting indices
    int global_i_start = coords[0] * local_M;
    int global_j_start = coords[1] * local_N;

    for ( int_t i=0; i<local_M; i++ )
    {
        for ( int_t j=0; j<local_N; j++ )
        {
            // Global indices
            int_t gi = global_i_start + i;
            int_t gj = global_j_start + j;

            // Calculate delta (radial distance) adjusted for M x N grid
            real_t delta = sqrt ( ((gi - M/2.0) * (gi - M/2.0)) / (real_t)M +
                                  ((gj - N/2.0) * (gj - N/2.0)) / (real_t)N );
            U_prv(i,j) = U(i,j) = exp ( -4.0*delta*delta );
        }
    }

    // Set the time step for 2D case
    dt = dx*dy / (c * sqrt (dx*dx+dy*dy));
}

// Get rid of all the memory allocations
void domain_finalize ( void )
{
    free ( buffers[0] );
    free ( buffers[1] );
    free ( buffers[2] );
}

// Integration formula
void time_step ( void )
{
    for ( int_t i=0; i<local_M; i++ )
    {
        for ( int_t j=0; j<local_N; j++ )
        {
            U_nxt(i,j) = -U_prv(i,j) + 2.0*U(i,j)
                     + (dt*dt*c*c)/(dx*dy) * (
                        U(i-1,j)+U(i+1,j)+U(i,j-1)+U(i,j+1)-4.0*U(i,j)
                    );
        }
    }
}

// Communicate the border between processes.
void border_exchange ( void )
{
    MPI_Status status;
    MPI_Request request[8];

    // Create MPI datatypes for row and column
    MPI_Datatype row_type;
    MPI_Type_contiguous(local_N, MPI_DOUBLE, &row_type);
    MPI_Type_commit(&row_type);

    MPI_Datatype column_type;
    MPI_Type_vector(local_M, 1, local_N+2, MPI_DOUBLE, &column_type);
    MPI_Type_commit(&column_type);

    // Send up, receive from down
    MPI_Isend(&U(0,0), 1, row_type, rank_up, 0, cart_comm, &request[0]);
    MPI_Irecv(&U(local_M,0), 1, row_type, rank_down, 0, cart_comm, &request[1]);

    // Send down, receive from up
    MPI_Isend(&U(local_M-1,0), 1, row_type, rank_down, 1, cart_comm, &request[2]);
    MPI_Irecv(&U(-1,0), 1, row_type, rank_up, 1, cart_comm, &request[3]);

    // Send left, receive from right
    MPI_Isend(&U(0,0), 1, column_type, rank_left, 2, cart_comm, &request[4]);
    MPI_Irecv(&U(0,local_N), 1, column_type, rank_right, 2, cart_comm, &request[5]);

    // Send right, receive from left
    MPI_Isend(&U(0,local_N-1), 1, column_type, rank_right, 3, cart_comm, &request[6]);
    MPI_Irecv(&U(0,-1), 1, column_type, rank_left, 3, cart_comm, &request[7]);

    MPI_Waitall(8, request, MPI_STATUSES_IGNORE);

    // Free the datatypes
    MPI_Type_free(&row_type);
    MPI_Type_free(&column_type);
}

// Neumann (reflective) boundary condition
void boundary_condition ( void )
{
    // Apply Neumann boundary conditions only at the physical boundaries
    if (coords[1] == 0) { // Left boundary
        for ( int_t i=0; i<local_M; i++ )
        {
            U(i, -1) = U(i, 1);
        }
    }
    if (coords[1] == dims[1] - 1) { // Right boundary
        for ( int_t i=0; i<local_M; i++ )
        {
            U(i, local_N) = U(i, local_N - 2);
        }
    }
    if (coords[0] == 0) { // Top boundary
        for ( int_t j=0; j<local_N; j++ )
        {
            U(-1, j) = U(1, j);
        }
    }
    if (coords[0] == dims[0] -1) { // Bottom boundary
        for ( int_t j=0; j<local_N; j++ )
        {
            U(local_M, j) = U(local_M - 2, j);
        }
    }
}

// Save the present time step in a numbered file under 'data/'
void domain_save ( int_t step )
{
    char filename[256];
    sprintf ( filename, "data/%.5ld.dat", step );

    MPI_File fh;
    MPI_Status status;

    // Open the file in parallel
    MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fh);

    // Define the file view
    int gsizes[2] = { (int) M, (int) N };
    int lsizes[2] = { local_M, local_N };
    int starts[2] = { coords[0]*local_M, coords[1]*local_N };

    MPI_Datatype filetype;
    MPI_Type_create_subarray(2, gsizes, lsizes, starts, MPI_ORDER_C, MPI_DOUBLE, &filetype);
    MPI_Type_commit(&filetype);

    // Define the memory datatype (interior data)
    int l_gsizes[2] = { local_M+2, local_N+2 };
    int l_lsizes[2] = { local_M, local_N };
    int l_starts[2] = {1,1};

    MPI_Datatype memtype;
    MPI_Type_create_subarray(2, l_gsizes, l_lsizes, l_starts, MPI_ORDER_C, MPI_DOUBLE, &memtype);
    MPI_Type_commit(&memtype);

    // Set the file view
    MPI_File_set_view(fh, 0, MPI_DOUBLE, filetype, "native", MPI_INFO_NULL);

    // Write the data
    MPI_File_write_all(fh, buffers[1], 1, memtype, &status);

    // Clean up
    MPI_File_close(&fh);
    MPI_Type_free(&filetype);
    MPI_Type_free(&memtype);
}

// Main time integration.
void simulate( void )
{
    // Go through each time step
    for ( int_t iteration=0; iteration<=max_iteration; iteration++ )
    {
        if ( (iteration % snapshot_freq)==0 )
        {
            domain_save ( iteration / snapshot_freq );
        }

        // Derive step t+1 from steps t and t-1
        border_exchange();
        boundary_condition();
        time_step();

        // Rotate the time step buffers
        move_buffer_window();
    }
}

int main ( int argc, char **argv )
{
    // Initialise MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    // Distribute the user arguments to all the processes
    if (world_rank == 0) {
        OPTIONS *options = parse_args( argc, argv );
        if ( !options )
        {
            fprintf( stderr, "Argument parsing failed\n" );
            exit( EXIT_FAILURE );
        }

        M = options->M;
        N = options->N;
        max_iteration = options->max_iteration;
        snapshot_freq = options->snapshot_frequency;
    }

    // Broadcast the parameters to all processes
    MPI_Bcast(&M, 1, MPI_LONG_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&N, 1, MPI_LONG_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&max_iteration, 1, MPI_LONG_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&snapshot_freq, 1, MPI_LONG_LONG, 0, MPI_COMM_WORLD);

    // Set up Cartesian communicator
    dims[0] = dims[1] = 0;
    MPI_Dims_create(world_size, 2, dims);
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 1, &cart_comm);
    MPI_Comm_rank(cart_comm, &cart_rank);
    MPI_Cart_coords(cart_comm, cart_rank, 2, coords);

    // Get ranks of neighboring processes
    MPI_Cart_shift(cart_comm, 0, 1, &rank_up, &rank_down);
    MPI_Cart_shift(cart_comm, 1, 1, &rank_left, &rank_right);

    // Set up the initial state of the domain
    domain_initialize();

    struct timeval t_start, t_end;

    // Time the code
    gettimeofday(&t_start, NULL);
    simulate();
    gettimeofday(&t_end, NULL);

    double elapsed_time = WALLTIME(t_end) - WALLTIME(t_start);
    if (world_rank == 0) {
        printf("Total simulation time: %lf seconds\n", elapsed_time);
    }

    // Clean up and shut down
    domain_finalize();

    // Finalise MPI
    MPI_Finalize();

    exit ( EXIT_SUCCESS );
}
