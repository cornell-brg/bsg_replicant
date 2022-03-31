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
#define N 64
#define GBASE 16

#define MAX_WORKERS 128
#define HB_L2_CACHE_LINE_WORDS 16
#define BUF_FACTOR 129
#define BUF_SIZE (MAX_WORKERS * HB_L2_CACHE_LINE_WORDS * BUF_FACTOR)

#ifndef RAND_MAX
#define RAND_MAX 32767
#endif

#define REAL float

#define BASE_ROW  GBASE
#define BASE_COL  GBASE

unsigned long rand_nxt = 0;

int cilk_rand( void )
{
  int result;
  rand_nxt = rand_nxt * 1103515245 + 12345;
  result = (rand_nxt >> 16) % ((unsigned int) RAND_MAX + 1);
  return result;
}

void init( REAL *A, int n )
{
  int i, j;
  for (i = 0; i < n; i++) {
    for (j = 0; j < n; j++) {
      A[i * n + j] = (REAL)cilk_rand();
    }
  }
}

void mat_transpose( REAL *A, REAL *B, int m, int n, int r_off, int col_off ) {
  int i, j;
  if (m < BASE_COL && n < BASE_ROW) {
    int r_max = r_off + n;
    int c_max = col_off + m;
    for ( i = r_off; i < r_max; i++ ) {
      for ( j = col_off; j < c_max; j++ ) {
        B[j * N + i] = A[i * N + j];
        //printf("B[%d] = A[%d] = %f\n", j * N + i, i * N + j, A[i * N + j]);
      }
    }
  } else {
    if (n >= m) {
      int split = n/2;
      mat_transpose( A, B, m, split, r_off, col_off );
      mat_transpose( A, B, m, split, r_off, col_off + split );
    }
    else {
      int split = m/2;
      mat_transpose( A, B, split, n, r_off, col_off );
      mat_transpose( A, B, split, n, r_off + split, col_off );
    }
  }
  return;
}

int kernel_mattranspose (int argc, char **argv) {
        int rc;
        char *bin_path, *test_name;
        struct arguments_path args = {NULL, NULL};

        argp_parse (&argp_path, argc, argv, 0, 0, &args);
        bin_path = args.path;
        test_name = args.name;

        bsg_pr_test_info("Running the mattranspose Kernel on one %dx%d tile groups.\n\n", bsg_tiles_X, bsg_tiles_Y);

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

                eva_t A_device, B_device;
                BSG_CUDA_CALL(hb_mc_device_malloc(&device, N * N * sizeof(REAL), &A_device)); /* allocate A[N] on the device */
                BSG_CUDA_CALL(hb_mc_device_malloc(&device, N * N * sizeof(REAL), &B_device)); /* allocate B[N] on the device */

                eva_t dram_buffer;
                BSG_CUDA_CALL(hb_mc_device_malloc(&device, BUF_SIZE * sizeof(uint32_t), &dram_buffer));

                /*****************************************************************************************************************
                 * Allocate memory on the host for A & B and initialize with random values.
                 ******************************************************************************************************************/
                REAL A_host[N * N]; /* allocate A[N] on the host */
                init( A_host, N );

                for (int i = 0; i < N * N; i++) {
                        printf("A_host[%d] = %f\n", i, A_host[i]);
                }
                /*****************************************************************************************************************
                 * Copy A & B from host onto device DRAM.
                 ******************************************************************************************************************/
                void *dst = (void *) ((intptr_t) A_device);
                void *src = (void *) &A_host[0];
                BSG_CUDA_CALL(hb_mc_device_memcpy (&device, dst, src, N * N * sizeof(REAL), HB_MC_MEMCPY_TO_DEVICE)); /* Copy A to the device  */

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
                int cuda_argv[5] = {A_device, B_device, N, GBASE, dram_buffer};

                /*****************************************************************************************************************
                 * Enquque grid of tile groups, pass in grid and tile group dimensions, kernel name, number and list of input arguments
                 ******************************************************************************************************************/
                BSG_CUDA_CALL(hb_mc_kernel_enqueue (&device, grid_dim, tg_dim, "kernel_appl_mattranspose", 5, cuda_argv));

                /*****************************************************************************************************************
                 * Launch and execute all tile groups on device and wait for all to finish.
                 ******************************************************************************************************************/
                BSG_CUDA_CALL(hb_mc_device_tile_groups_execute(&device));

                /*****************************************************************************************************************
                 * Copy result matrix back from device DRAM into host memory.
                 ******************************************************************************************************************/
                REAL B_host[N * N];
                src = (void *) ((intptr_t) B_device);
                dst = (void *) &B_host[0];
                BSG_CUDA_CALL(hb_mc_device_memcpy (&device, (void *) dst, src, N * N * sizeof(REAL), HB_MC_MEMCPY_TO_HOST)); /* copy C to the host */

                /*****************************************************************************************************************
                 * Freeze the tiles and memory manager cleanup.
                 ******************************************************************************************************************/
                BSG_CUDA_CALL(hb_mc_device_program_finish(&device));

                /*****************************************************************************************************************
                 * Calculate the expected result using host code and compare the results.
                 ******************************************************************************************************************/
                REAL B_expected[N * N];
                mat_transpose(A_host, B_expected, N, N, 0, 0);

                int mismatch = 0;
                for (int i = 0; i < N * N; i++) {
                        // printf("B[%d] = %f\n", i, B_host[i]);
                        if (B_expected[i] != B_host[i]) {
                                bsg_pr_err(BSG_RED("Mismatch: ") "B[%d]:  %f != %f\n", i, B_host[i], B_expected[i]);
                                mismatch = 1;
                        }
                }

                if (mismatch) {
                        return HB_MC_FAIL;
                }
        }
        BSG_CUDA_CALL(hb_mc_device_finish(&device));

        return HB_MC_SUCCESS;
}

declare_program_main("test_appl_mattranspose", kernel_mattranspose);
