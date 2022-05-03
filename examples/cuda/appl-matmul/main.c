// Copyright (c) 2019, University of Washington All rights reserved.
// 
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
// 
// Redistributions of source code must retain the above copyright notice, this list
// of conditions and the following disclaimer.
// 
// Redistributions in binary form must reproduce the above copyright notice, this
// list of conditions and the following disclaimer in the documentation and/or
// other materials provided with the distribution.
// 
// Neither the name of the copyright holder nor the names of its contributors may
// be used to endorse or promote products derived from this software without
// specific prior written permission.
// 
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
// ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
// WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
// ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
// ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include <bsg_manycore_tile.h>
#include <bsg_manycore_errno.h>
#include <bsg_manycore_tile.h>
#include <bsg_manycore_loader.h>
#include <bsg_manycore_cuda.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <stdio.h>
#include <bsg_manycore_regression.h>

#define ALLOC_NAME "default_allocator"
#define MAX_WORKERS 128
#define HB_L2_CACHE_LINE_WORDS 16
#define BUF_FACTOR 2049
#define BUF_SIZE (MAX_WORKERS * HB_L2_CACHE_LINE_WORDS * BUF_FACTOR)

#define N MATRIX_N
#define GRAIN_SIZE 32

#define REAL float

void print( REAL* A, int n )
{
  int i, j;

  for ( i = 0; i < n; i++ ) {
    for ( j = 0; j < n; j++ ) {
      printf("%f ", A[i * n + j]);
    }
    printf("\n");
  }
}

void zero( REAL* A, int n )
{
  int i, j;

  for ( i = 0; i < n; i++ ) {
    for ( j = 0; j < n; j++ ) {
      A[i * n + j] = 0.0;
    }
  }
}

void init( REAL* A, int n )
{
  int i, j;

  for ( i = 0; i < n; i++ ) {
    for ( j = 0; j < n; j++ ) {
      A[i * n + j] = (REAL)rand() / (REAL)(RAND_MAX / 5.0f);
    }
  }
}

double maxerror( REAL* A, REAL* B, int n )
{
  int    i, j;
  double error = 0.0;

  for ( i = 0; i < n; i++ ) {
    for ( j = 0; j < n; j++ ) {
      double diff = ( A[i * n + j] - B[i * n + j] ) / A[i * n + j];
      if ( diff < 0 )
        diff = -diff;
      if ( diff > error )
        error = diff;
    }
  }
  return error;
}

void iter_matmul( REAL* A, REAL* B, REAL* C, int n )
{
  int i, j, k;

  for ( i = 0; i < n; i++ )
    for ( k = 0; k < n; k++ ) {
      REAL c = 0.0;
      for ( j = 0; j < n; j++ )
        c += A[i * n + j] * B[j * n + k];
      C[i * n + k] = c;
    }
}

/* Function to check if x is power of 2*/
int isPowerOfTwo( int n )
{
  return ( ceil( log2( n ) ) == floor( log2( n ) ) );
}

int kernel_appl_matmul (int argc, char **argv) {
        int rc;
        char *bin_path, *test_name;
        struct arguments_path args = {NULL, NULL};

        argp_parse (&argp_path, argc, argv, 0, 0, &args);
        bin_path = args.path;
        test_name = args.name;

        bsg_pr_test_info("Running the Cilk5 MatMul WS Kernel on one %dx%d tile groups.\n\n", bsg_tiles_X, bsg_tiles_Y);

        srand(time);

        /*****************************************************************************************************************
        * Define path to binary.
        * Initialize device, load binary and unfreeze tiles.
        ******************************************************************************************************************/
        hb_mc_device_t device;
        BSG_CUDA_CALL(hb_mc_device_init(&device, test_name, 0));

        hb_mc_pod_id_t pod;
        hb_mc_device_foreach_pod_id(&device, pod)
        {
                bsg_pr_info("Loading program for test %s onto pod %d\n", test_name, pod);
                BSG_CUDA_CALL(hb_mc_device_set_default_pod(&device, pod));
                BSG_CUDA_CALL(hb_mc_device_program_init(&device, bin_path, ALLOC_NAME, 0));

                /*****************************************************************************************************************
                 * Allocate memory on the device for A, B and C.
                 ******************************************************************************************************************/
                if ( !isPowerOfTwo( N ) ) {
                  printf("Input size must be a power of two!");
                  return HB_MC_FAIL;
                }

                if ( GRAIN_SIZE < 4 || !isPowerOfTwo( GRAIN_SIZE ) ) {
                  printf("Grain size must >= 4 and is power of 2");
                  return HB_MC_FAIL;
                }

                eva_t A_device, B_device, C_device;
                BSG_CUDA_CALL(hb_mc_device_malloc(&device, N * N * sizeof(REAL), &A_device)); /* allocate A[N*N] on the device */
                BSG_CUDA_CALL(hb_mc_device_malloc(&device, N * N * sizeof(REAL), &B_device)); /* allocate B[N*N] on the device */
                BSG_CUDA_CALL(hb_mc_device_malloc(&device, N * N * sizeof(REAL), &C_device)); /* allocate C[N*N] on the device */

                eva_t dram_buffer;
                BSG_CUDA_CALL(hb_mc_device_malloc(&device, BUF_SIZE * sizeof(uint32_t), &dram_buffer));

                /*****************************************************************************************************************
                 * Allocate memory on the host for A & B and initialize with random values.
                 ******************************************************************************************************************/
                REAL * A, *B, *C1, *C2;
                A  = (REAL*)malloc( N * N * sizeof( REAL ) );
                B  = (REAL*)malloc( N * N * sizeof( REAL ) );
                C1 = (REAL*)malloc( N * N * sizeof( REAL ) );
                C2 = (REAL*)malloc( N * N * sizeof( REAL ) );

                init( A, N );
                init( B, N );
                zero( C1, N );
                zero( C2, N );

                print( A, N );
                printf("----\n");
                print( B, N );

                /*****************************************************************************************************************
                 * Copy A & B from host onto device DRAM.
                 ******************************************************************************************************************/
                hb_mc_dma_htod_t htod_A = {
                  .d_addr = A_device,
                  .h_addr = A,
                  .size   = N * N * sizeof(REAL)
                };
                BSG_CUDA_CALL(hb_mc_device_dma_to_device(&device, &htod_A, 1));

                hb_mc_dma_htod_t htod_B = {
                  .d_addr = B_device,
                  .h_addr = B,
                  .size   = N * N * sizeof(REAL)
                };
                BSG_CUDA_CALL(hb_mc_device_dma_to_device(&device, &htod_B, 1));

                /*****************************************************************************************************************
                 * Define block_size_x/y: amount of work for each tile group
                 * Define tg_dim_x/y: number of tiles in each tile group
                 * Calculate grid_dim_x/y: number of tile groups needed based on block_size_x/y
                 ******************************************************************************************************************/
                hb_mc_dimension_t tg_dim = { .x = bsg_tiles_X, .y = bsg_tiles_Y};
                hb_mc_dimension_t grid_dim = { .x = 1, .y = 1};

                /*****************************************************************************************************************
                 * Prepare list of input arguments for kernel.
                 ******************************************************************************************************************/
                int cuda_argv[6] = {A_device, B_device, C_device, N, GRAIN_SIZE, dram_buffer};

                /*****************************************************************************************************************
                 * Enquque grid of tile groups, pass in grid and tile group dimensions, kernel name, number and list of input arguments
                 ******************************************************************************************************************/
                BSG_CUDA_CALL(hb_mc_kernel_enqueue (&device, grid_dim, tg_dim, "kernel_appl_matmul", 6, cuda_argv));

                /*****************************************************************************************************************
                 * Launch and execute all tile groups on device and wait for all to finish.
                 ******************************************************************************************************************/
                BSG_CUDA_CALL(hb_mc_device_tile_groups_execute(&device));

                /*****************************************************************************************************************
                 * Copy result matrix back from device DRAM into host memory.
                 ******************************************************************************************************************/
                hb_mc_dma_dtoh_t dtoh_C = {
                  .d_addr = C_device,
                  .h_addr = C1,
                  .size   = N * N * sizeof(REAL)
                };
                BSG_CUDA_CALL(hb_mc_device_dma_to_host(&device, &dtoh_C, 1));

                /*****************************************************************************************************************
                 * Freeze the tiles and memory manager cleanup.
                 ******************************************************************************************************************/
                BSG_CUDA_CALL(hb_mc_device_program_finish(&device));

                /*****************************************************************************************************************
                 * Calculate the expected result using host code and compare the results.
                 ******************************************************************************************************************/
                iter_matmul( A, B, C2, N );
                double err = maxerror( C1, C2, N );
                printf("max error = %f\n", err);
                print(C1, N);
                printf("----\n");
                print(C2, N);

                if (err > 0.01) {
                        return HB_MC_FAIL;
                }
        }
        BSG_CUDA_CALL(hb_mc_device_finish(&device));

        return HB_MC_SUCCESS;
}

declare_program_main("test_appl_matmul", kernel_appl_matmul);
