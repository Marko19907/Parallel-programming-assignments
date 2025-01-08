#define _XOPEN_SOURCE 600
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>


// Option to change numerical precision.
typedef int64_t int_t;
typedef double real_t;

// Simulation parameters: size, step count, and how often to save the state.
const int_t
    N = 1024,
    max_iteration = 4000,
    snapshot_freq = 10;

// Wave equation parameters, time step is derived from the space step.
const real_t
    c  = 1.0,
    dx  = 1.0;
real_t
    dt;

// Buffers for three time steps, indexed with 2 ghost points for the boundary.
real_t
    *buffers[3] = { NULL, NULL, NULL };


#define U_prv(i) buffers[0][(i)+1]
#define U(i)     buffers[1][(i)+1]
#define U_nxt(i) buffers[2][(i)+1]


// Save the present time step in a numbered file under 'data/'.
void domain_save ( int_t step )
{
    char filename[256];
    sprintf ( filename, "data/%.5ld.dat", step );
    FILE *out = fopen ( filename, "wb" );
    fwrite ( &U(0), sizeof(real_t), N, out );
    fclose ( out );
}


// Set up our three buffers, fill two with an initial cosine wave,
// and set the time step.
void domain_initialize ( void )
{
    // Allocate memory for the three buffers (N+2 to account for ghost points)
    for (int i = 0; i < 3; i++) {
        buffers[i] = (real_t *)malloc((N + 2) * sizeof(real_t));
        if (buffers[i] == NULL) {
            perror("Failed to allocate memory for buffers");
            exit(EXIT_FAILURE);
        }
    }

    // Initialize the buffers with the initial condition u(x, 0) = cos(2 * pi * x)
    for (int_t i = 0; i < N; i++) {
        real_t x = (real_t)i / N; // Normalize x to the range [0, 1]
        U_prv(i) = cos(2 * M_PI * x); // Set initial condition for previous buffer
        U(i) = cos(2 * M_PI * x);     // Set initial condition for current buffer
    }

    // Set the time step using the CFL condition: dt = dx / c
    dt = dx / c;
}


// Return the memory to the OS.
void domain_finalize ( void )
{
    // Free the memory allocated for the three buffers
    for (int i = 0; i < 3; i++) {
        if (buffers[i] != NULL) {
            free(buffers[i]);
            buffers[i] = NULL; // Set pointer to NULL to avoid dangling pointers
        }
    }
}


// Rotate the time step buffers.
void rotate_buffers( void )
{
    // Rotate the buffers by swapping the pointers, think this should be fine since the next buffer will be overwritten anyway
    real_t *temp = buffers[0];
    buffers[0] = buffers[1];
    buffers[1] = buffers[2];
    buffers[2] = temp;
}


// Derive step t+1 from steps t and t-1.
void calculate_next_step( void )
{
    // Calculate the coefficient (dt^2 * c^2) / dx^2, which is used in the wave equation
    real_t coeff = (dt * dt * c * c) / (dx * dx);

    // Loop over each spatial point, excluding ghost points
    for (int_t i = 0; i < N; i++) {
        // Apply the wave equation to calculate the next time step at each point
        U_nxt(i) = -U_prv(i) + 2 * U(i) + coeff * (U(i-1) + U(i+1) - 2 * U(i));
    }
}


// Neumann (reflective) boundary condition.
void apply_neumann_boundary_conditions( void )
{
    // Left boundary (ghost point at index -1)
    U(-1) = U(0);  // Reflect the first interior point to the first ghost point

    // Right boundary (ghost point at index N)
    U(N) = U(N-1); // Reflect the last interior point to the last ghost point
}


// Main time integration.
void simulate( void )
{
    // int_t iteration=0;
    // domain_save ( iteration / snapshot_freq );

    for (int_t iteration = 0; iteration <= max_iteration; iteration++) {
        // Apply Neumann (reflective) boundary conditions
        apply_neumann_boundary_conditions();

        // Calculate the next time step
        calculate_next_step();

        // Rotate the buffers so that the new step becomes the current step
        rotate_buffers();

        // Save the current state at regular intervals (every snapshot_freq iterations)
        if (iteration % snapshot_freq == 0) {
            domain_save(iteration);
        }
    }
}


int main ( void )
{
    domain_initialize();

    simulate();

    domain_finalize();
    exit ( EXIT_SUCCESS );
}
