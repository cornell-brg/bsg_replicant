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
#define BUF_FACTOR 16385
#define BUF_SIZE (MAX_WORKERS * HB_L2_CACHE_LINE_WORDS * BUF_FACTOR)

#define N MATRIX_N
typedef uint32_t ELM;

#define swap( a, b )                                                     \
  {                                                                      \
    ELM tmp;                                                             \
    tmp = a;                                                             \
    a   = b;                                                             \
    b   = tmp;                                                           \
  }

/* Function to check if x is power of 2*/
int isPowerOfTwo( int n )
{
  return ( ceil( log2( n ) ) == floor( log2( n ) ) );
}

static unsigned long rand_nxt = 0;

static inline unsigned long my_rand( void )
{
  rand_nxt = rand_nxt * 1103515245 + 12345;
  return rand_nxt;
}

static inline void my_srand( unsigned long seed )
{
  rand_nxt = seed;
}

void scramble_array( ELM* arr, unsigned long size )
{
  unsigned long i;
  unsigned long j;

  for ( i = 0; i < size; ++i ) {
    j = my_rand();
    j = j % size;
    swap( arr[i], arr[j] );
  }
}

void fill_array( ELM* arr, unsigned long size )
{
  unsigned long i;

  my_srand( 1 );
  /* first, fill with integers 1..size */
  for ( i = 0; i < size; ++i ) {
    arr[i] = i;
  }

  /* then, scramble randomly */
  scramble_array( arr, size );
}

int kernel_static_mergesort (int argc, char **argv) {
        int rc;
        char *bin_path, *test_name;
        struct arguments_path args = {NULL, NULL};

        argp_parse (&argp_path, argc, argv, 0, 0, &args);
        bin_path = args.path;
        test_name = args.name;

        bsg_pr_test_info("Running the STATIC MergeSort Kernel on one %dx%d tile groups.\n\n", bsg_tiles_X, bsg_tiles_Y);

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
                if (!isPowerOfTwo(bsg_tiles_X * bsg_tiles_Y)) {
                  bsg_pr_err(BSG_RED("Error: ") "tile group size has to be power of 2\n");
                  return HB_MC_FAIL;
                }

                bsg_pr_info("Loading program for test %s onto pod %d\n", test_name, pod);
                BSG_CUDA_CALL(hb_mc_device_set_default_pod(&device, pod));
                BSG_CUDA_CALL(hb_mc_device_program_init(&device, bin_path, ALLOC_NAME, 0));

                /*****************************************************************************************************************
                 * Allocate memory on the device for A, B and C.
                 ******************************************************************************************************************/

                eva_t A_device, B_device, C_device;
                BSG_CUDA_CALL(hb_mc_device_malloc(&device, N * sizeof(ELM), &A_device)); /* allocate A[N*N] on the device */
                BSG_CUDA_CALL(hb_mc_device_malloc(&device, N * sizeof(ELM), &B_device)); /* allocate B[N*N] on the device */

                eva_t dram_buffer;
                BSG_CUDA_CALL(hb_mc_device_malloc(&device, BUF_SIZE * sizeof(uint32_t), &dram_buffer));

                /*****************************************************************************************************************
                 * Allocate memory on the host for A & B and initialize with random values.
                 ******************************************************************************************************************/
                ELM* array  = (ELM*)malloc( N * sizeof( ELM ) );
                ELM* tmp    = (ELM*)malloc( N * sizeof( ELM ) );
                ELM* result = (ELM*)malloc( N * sizeof( ELM ) );

                fill_array( array, N );

                /*****************************************************************************************************************
                 * Copy A & B from host onto device DRAM.
                 ******************************************************************************************************************/
                hb_mc_dma_htod_t htod = {
                  .d_addr = A_device,
                  .h_addr = array,
                  .size   = N * sizeof(ELM)
                };
                BSG_CUDA_CALL(hb_mc_device_dma_to_device(&device, &htod, 1));

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
                int cuda_argv[4] = {A_device, B_device, N, dram_buffer};

                /*****************************************************************************************************************
                 * Enquque grid of tile groups, pass in grid and tile group dimensions, kernel name, number and list of input arguments
                 ******************************************************************************************************************/
                BSG_CUDA_CALL(hb_mc_kernel_enqueue (&device, grid_dim, tg_dim, "kernel_static_mergesort", 4, cuda_argv));

                /*****************************************************************************************************************
                 * Launch and execute all tile groups on device and wait for all to finish.
                 ******************************************************************************************************************/
                BSG_CUDA_CALL(hb_mc_device_tile_groups_execute(&device));

                /*****************************************************************************************************************
                 * Copy result matrix back from device DRAM into host memory.
                 ******************************************************************************************************************/
                int32_t iters  = 0;
                int32_t factor = 1;
                while( factor != bsg_tiles_X * bsg_tiles_Y) {
                  ++iters;
                  factor = factor * 2;
                }
                if (iters % 2 == 1) {
                  // merged data is in tmp buffer
                  A_device = B_device;
                }
                hb_mc_dma_dtoh_t dtoh_C = {
                  .d_addr = A_device,
                  .h_addr = result,
                  .size   = N * sizeof(ELM)
                };
                BSG_CUDA_CALL(hb_mc_device_dma_to_host(&device, &dtoh_C, 1));

                /*****************************************************************************************************************
                 * Freeze the tiles and memory manager cleanup.
                 ******************************************************************************************************************/
                BSG_CUDA_CALL(hb_mc_device_program_finish(&device));

                /*****************************************************************************************************************
                 * Calculate the expected result using host code and compare the results.
                 ******************************************************************************************************************/
                int success = 1;
                for ( int i = 0; i < N; ++i ) {
                  printf("result[%d] = %d\n", i, result[i]);
                  if ( result[i] != i ) {
                    success = 0;
                    bsg_pr_err(BSG_RED("Mismatch: ") "result[%d]: %d != %d\n",
                        i, result[i], i);
                  }
                }
                if ( !success ) {
                  return HB_MC_FAIL;
                }
        }
        BSG_CUDA_CALL(hb_mc_device_finish(&device));

        return HB_MC_SUCCESS;
}

declare_program_main("test_static_mergesort", kernel_static_mergesort);
