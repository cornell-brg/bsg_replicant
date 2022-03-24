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
#define BUF_FACTOR 129
#define BUF_SIZE (MAX_WORKERS * HB_L2_CACHE_LINE_WORDS * BUF_FACTOR)

/* Define the size of a block. */
#ifndef BLOCK_SIZE
#define BLOCK_SIZE 4
#endif

/* Define the default matrix size. */
#ifndef DEFAULT_SIZE
#define DEFAULT_SIZE (16 * BLOCK_SIZE)
#endif

/* A block is a 2D array of floats. */
typedef float Block[BLOCK_SIZE][BLOCK_SIZE];
#define BLOCK(B,I,J) (B[I][J])

/* A matrix is a 1D array of blocks. */
typedef Block *Matrix;
#define MATRIX(M,I,J) ((M)[(I)*nBlocks+(J)])

/* Matrix size in blocks. */
static int nBlocks;

#define N 32

/****************************************************************************\
 * Utility routines.
\****************************************************************************/

/*
 * init_matrix - Fill in matrix M with random values.
 */
static void init_matrix(Matrix M, int nb)
{
  int I, J, K, i, j, k;

  /* Initialize random number generator. */
  srand(1);

  /* For each element of each block, fill in random value. */
  for (I = 0; I < nb; I++) {
    for (J = 0; J < nb; J++) {
      for (i = 0; i < BLOCK_SIZE; i++) {
        for (j = 0; j < BLOCK_SIZE; j++) {
          BLOCK(MATRIX(M, I, J), i, j) =
            ((float)rand()) / (float)RAND_MAX;
        }
      }
    }
  }

  /* Inflate diagonal entries. */
  for (K = 0; K < nb; K++)
    for (k = 0; k < BLOCK_SIZE; k++)
      BLOCK(MATRIX(M, K, K), k, k) *= 10.0;
}

/*
 * print_matrix - Print matrix M.
 */
static void print_matrix(Matrix M, int nb)
{
  int i, j;

  /* Print out matrix. */
  for (i = 0; i < nb * BLOCK_SIZE; i++) {
    for (j = 0; j < nb * BLOCK_SIZE; j++)
      printf(" %6.4f",
             BLOCK(MATRIX(M, i / BLOCK_SIZE, j / BLOCK_SIZE),
                   i % BLOCK_SIZE, j % BLOCK_SIZE));
    printf("\n");
  }
}

/*
 * test_result - Check that matrix LU contains LU decomposition of M.
 */
static int test_result(Matrix LU, Matrix M, int nb)
{
  int I, J, K, i, j, k;
  float diff, max_diff;
  float v;

  /* Initialize test. */
  max_diff = 0.0;

  /* Find maximum difference between any element of LU and M. */
  for (i = 0; i < nb * BLOCK_SIZE; i++)
    for (j = 0; j < nb * BLOCK_SIZE; j++) {
      I = i / BLOCK_SIZE;
      J = j / BLOCK_SIZE;
      v = 0.0;
      for (k = 0; k < i && k <= j; k++) {
        K = k / BLOCK_SIZE;
        v += BLOCK(MATRIX(LU, I, K), i % BLOCK_SIZE,
                   k % BLOCK_SIZE) *
          BLOCK(MATRIX(LU, K, J), k % BLOCK_SIZE,
                j % BLOCK_SIZE);
      }
      if (k == i && k <= j) {
        K = k / BLOCK_SIZE;
        v += BLOCK(MATRIX(LU, K, J), k % BLOCK_SIZE,
                   j % BLOCK_SIZE);
      }
      diff = fabs(BLOCK(MATRIX(M, I, J), i % BLOCK_SIZE,
                        j % BLOCK_SIZE) - v);
      if (diff > max_diff) {
        max_diff = diff;
        printf("update max diff -> %f at (%d, %d)\n", max_diff, i, j);
      }
    }

  /* Check maximum difference against threshold. */
  if (max_diff > 0.00001)
    return 0;
  else
    return 1;
}

int kernel_lu (int argc, char **argv) {
        int rc;
        char *bin_path, *test_name;
        struct arguments_path args = {NULL, NULL};

        argp_parse (&argp_path, argc, argv, 0, 0, &args);
        bin_path = args.path;
        test_name = args.name;

        bsg_pr_test_info("Running the LU Kernel on one %dx%d tile groups.\n\n", bsg_tiles_X, bsg_tiles_Y);

        srand(time);

        /*****************************************************************************************************************
        * Define path to binary.
        * Initialize device, load binary and unfreeze tiles.
        ******************************************************************************************************************/
        hb_mc_device_t device;
        BSG_CUDA_CALL(hb_mc_device_init(&device, test_name, 0));

        if ( N < BLOCK_SIZE ) {
          printf("n needs to be at least %d\n", BLOCK_SIZE);
          return HB_MC_FAIL;
        }

        /* Check that matrix is power-of-2 sized. */
        int v = N;
        while (!((unsigned) v & (unsigned) 1)) {
          v >>= 1;
        }
        if (v != 1) {
          printf("n needs to be a power of 2");
          return HB_MC_FAIL;
        }

        hb_mc_pod_id_t pod;
        hb_mc_device_foreach_pod_id(&device, pod)
        {
                bsg_pr_info("Loading program for test %s onto pod %d\n", test_name, pod);
                BSG_CUDA_CALL(hb_mc_device_set_default_pod(&device, pod));
                BSG_CUDA_CALL(hb_mc_device_program_init(&device, bin_path, ALLOC_NAME, 0));

                /*****************************************************************************************************************
                 * Allocate memory on the device for A, B and C.
                 ******************************************************************************************************************/

                nBlocks = N / BLOCK_SIZE;
                eva_t M_device;
                BSG_CUDA_CALL(hb_mc_device_malloc(&device, N * N * sizeof(float), &M_device)); /* allocate M on the device */

                eva_t dram_buffer;
                BSG_CUDA_CALL(hb_mc_device_malloc(&device, BUF_SIZE * sizeof(uint32_t), &dram_buffer));

                /*****************************************************************************************************************
                 * Allocate memory on the host for A & B and initialize with random values.
                 ******************************************************************************************************************/
                Matrix M_host = (Matrix) malloc(N * N * sizeof(float));
                init_matrix(M_host, nBlocks);

                print_matrix(M_host, nBlocks);

                /*****************************************************************************************************************
                 * Copy A & B from host onto device DRAM.
                 ******************************************************************************************************************/
                void *dst = (void *) ((intptr_t) M_device);
                void *src = (void *) &M_host[0];
                BSG_CUDA_CALL(hb_mc_device_memcpy (&device, dst, src, N * N * sizeof(float), HB_MC_MEMCPY_TO_DEVICE)); /* Copy A to the device  */

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
                int cuda_argv[3] = {M_device, N, dram_buffer};

                /*****************************************************************************************************************
                 * Enquque grid of tile groups, pass in grid and tile group dimensions, kernel name, number and list of input arguments
                 ******************************************************************************************************************/
                BSG_CUDA_CALL(hb_mc_kernel_enqueue (&device, grid_dim, tg_dim, "kernel_appl_lu", 3, cuda_argv));

                /*****************************************************************************************************************
                 * Launch and execute all tile groups on device and wait for all to finish.
                 ******************************************************************************************************************/
                BSG_CUDA_CALL(hb_mc_device_tile_groups_execute(&device));

                /*****************************************************************************************************************
                 * Copy result matrix back from device DRAM into host memory.
                 ******************************************************************************************************************/
                Matrix LU_host = (Matrix) malloc(N * N * sizeof(float));
                src = (void *) ((intptr_t) M_device);
                dst = (void *) &LU_host[0];
                BSG_CUDA_CALL(hb_mc_device_memcpy (&device, (void *) dst, src, N * N * sizeof(float), HB_MC_MEMCPY_TO_HOST)); /* copy C to the host */

                /*****************************************************************************************************************
                 * Freeze the tiles and memory manager cleanup.
                 ******************************************************************************************************************/
                BSG_CUDA_CALL(hb_mc_device_program_finish(&device));

                /*****************************************************************************************************************
                 * Calculate the expected result using host code and compare the results.
                 ******************************************************************************************************************/

                print_matrix(LU_host, nBlocks);
                int success = test_result(LU_host, M_host, nBlocks);
                free(M_host);
                free(LU_host);
                if (!success) {
                        return HB_MC_FAIL;
                }
        }
        BSG_CUDA_CALL(hb_mc_device_finish(&device));

        return HB_MC_SUCCESS;
}

declare_program_main("test_appl_lu", kernel_lu);
