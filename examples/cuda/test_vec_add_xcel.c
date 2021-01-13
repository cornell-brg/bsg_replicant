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

#include "test_vec_add_xcel.h"

#define ALLOC_NAME "default_allocator"

/* #define NUM_VEC_ADD_XCEL    16 */
#define NUM_VEC_ADD_XCEL    1
#define VEC_ADD_XCEL_Y_CORD 9
#define ELEM_PER_VEC_XCEL   8

/*!
 * Runs the vector addition a one 1x1 tile group. A[N] + B[N] --> C[N]
 * Grid dimensions are prefixed at 1x1.
 * This tests uses the software/spmd/bsg_cuda_lite_runtime/vec_add_xcel/ device
 * code in the bsg_manycore repository.
*/


void host_vec_add (int *A, int *B, int *C, int N) { 
        for (int i = 0; i < N; i ++) { 
                C[i] = A[i] + B[i];
        }
        return;
}


int kernel_vec_add_xcel (int argc, char **argv) {
        int rc;
        char *bin_path, *test_name;
        struct arguments_path args = {NULL, NULL};

        argp_parse (&argp_path, argc, argv, 0, 0, &args);
        bin_path = args.path;
        test_name = args.name;

        bsg_pr_test_info("Running the CUDA Vector Addition Xcel Kernel on one 8x16 manycore.\n\n");

        srand(time); 

        /*****************************************************************************************************************
        * Define path to binary.
        * Initialize device, load binary and unfreeze tiles.
        ******************************************************************************************************************/

        hb_mc_device_t device;
        rc = hb_mc_device_init(&device, test_name, 0);
        if (rc != HB_MC_SUCCESS) { 
                bsg_pr_err("failed to initialize device.\n");
                return rc;
        }

        rc = hb_mc_device_program_init(&device, bin_path, ALLOC_NAME, 0);
        if (rc != HB_MC_SUCCESS) { 
                bsg_pr_err("failed to initialize program.\n");
                return rc;
        }

        /*****************************************************************************************************************
        * Allocate memory on the device for A, B and C.
        ******************************************************************************************************************/

        uint32_t N = ELEM_PER_VEC_XCEL * NUM_VEC_ADD_XCEL;
        bsg_pr_test_info("Allocating memory for ABC\n");

        eva_t A_device, B_device, C_device; 
        rc = hb_mc_device_malloc(&device, N * sizeof(uint32_t), &A_device); /* allocate A[N] on the device */
        if (rc != HB_MC_SUCCESS) { 
                bsg_pr_err("failed to allocate memory on device.\n");
                return rc;
        }


        rc = hb_mc_device_malloc(&device, N * sizeof(uint32_t), &B_device); /* allocate B[N] on the device */
        if (rc != HB_MC_SUCCESS) { 
                bsg_pr_err("failed to allocate memory on device.\n");
                return rc;
        }


        rc = hb_mc_device_malloc(&device, N * sizeof(uint32_t), &C_device); /* allocate C[N] on the device */
        if (rc != HB_MC_SUCCESS) { 
                bsg_pr_err("failed to allocate memory on device.\n");
                return rc;
        }


        /*****************************************************************************************************************
        * Allocate memory on the host for A & B and initialize with random values.
        ******************************************************************************************************************/

        bsg_pr_test_info("Filling ABC with random data\n");

        uint32_t A_host[N]; /* allocate A[N] on the host */ 
        uint32_t B_host[N]; /* allocate B[N] on the host */
        for (int i = 0; i < N; i++) { /* fill A with arbitrary data */
                /* A_host[i] = rand() & 0xFFFF; */
                /* B_host[i] = rand() & 0xFFFF; */
                A_host[i] = 0xcafe + i;
                B_host[i] = 0xb00c + i;
        }

        /*****************************************************************************************************************
        * Copy A & B from host onto device DRAM.
        ******************************************************************************************************************/

        bsg_pr_test_info("Copying data into HB DRAM\n");

        void *dst = (void *) ((intptr_t) A_device);
        void *src = (void *) &A_host[0];
        rc = hb_mc_device_memcpy (&device, dst, src, N * sizeof(uint32_t), HB_MC_MEMCPY_TO_DEVICE); /* Copy A to the device  */ 
        if (rc != HB_MC_SUCCESS) { 
                bsg_pr_err("failed to copy memory to device.\n");
                return rc;
        }


        dst = (void *) ((intptr_t) B_device);
        src = (void *) &B_host[0];
        rc = hb_mc_device_memcpy (&device, dst, src, N * sizeof(uint32_t), HB_MC_MEMCPY_TO_DEVICE); /* Copy B to the device */ 
        if (rc != HB_MC_SUCCESS) { 
                bsg_pr_err("failed to copy memory to device.\n");
                return rc;
        }


        /*****************************************************************************************************************
        * Define tg_dim_x/y: number of tiles in each tile group
        * Calculate grid_dim_x/y: number of tile groups needed
        ******************************************************************************************************************/

        // PP: sinec the accelerator will do the work a 1x1 tile group should be enough?
        hb_mc_dimension_t tg_dim = { .x = 2, .y = 2};

        hb_mc_dimension_t grid_dim = { .x = 1, .y = 1};


        /*****************************************************************************************************************
        * Prepare list of input arguments for kernel.
        ******************************************************************************************************************/

        int cuda_argv[5] = {A_device, B_device, C_device, N, ELEM_PER_VEC_XCEL};

        /*****************************************************************************************************************
        * Enquque grid of tile groups, pass in grid and tile group dimensions, kernel name, number and list of input arguments
        ******************************************************************************************************************/

        bsg_pr_test_info("Enqueuing kernel to HB\n");

        rc = hb_mc_kernel_enqueue (&device, grid_dim, tg_dim, "kernel_vec_add_xcel", 5, cuda_argv);
        if (rc != HB_MC_SUCCESS) { 
                bsg_pr_err("failed to initialize grid.\n");
                return rc;
        }

        /*****************************************************************************************************************
        * Launch and execute all tile groups on device and wait for all to finish. 
        ******************************************************************************************************************/

        bsg_pr_test_info("HB kernel execution starts\n");

        rc = hb_mc_device_tile_groups_execute(&device);
        if (rc != HB_MC_SUCCESS) { 
                bsg_pr_err("failed to execute tile groups.\n");
                return rc;
        }

        /*****************************************************************************************************************
        * Copy result matrix back from device DRAM into host memory. 
        ******************************************************************************************************************/

        bsg_pr_test_info("Copying result C from HB DRAM into host...\n");

        uint32_t C_host[N];
        src = (void *) ((intptr_t) C_device);
        dst = (void *) &C_host[0];
        rc = hb_mc_device_memcpy (&device, (void *) dst, src, N * sizeof(uint32_t), HB_MC_MEMCPY_TO_HOST); /* copy C to the host */
        if (rc != HB_MC_SUCCESS) { 
                bsg_pr_err("failed to copy memory from device.\n");
                return rc;
        }

        /*****************************************************************************************************************
        * Freeze the tiles and memory manager cleanup. 
        ******************************************************************************************************************/

        bsg_pr_test_info("Finalizing HB device...\n");

        rc = hb_mc_device_finish(&device); 
        if (rc != HB_MC_SUCCESS) { 
                bsg_pr_err("failed to de-initialize device.\n");
                return rc;
        }

        /*****************************************************************************************************************
        * Calculate the expected result using host code and compare the results. 
        ******************************************************************************************************************/

        uint32_t C_expected[N]; 
        host_vec_add (A_host, B_host, C_expected, N); 


        int mismatch = 0; 
        for (int i = 0; i < N; i++) {
                if (A_host[i] + B_host[i] != C_host[i]) {
                        bsg_pr_err(BSG_RED("Mismatch: ") "C[%d]:  0x%08" PRIx32 " + 0x%08" PRIx32 " = 0x%08" PRIx32 "\t Expected: 0x%08" PRIx32 "\n", i , A_host[i], B_host[i], C_host[i], C_expected[i]);
                        mismatch = 1;
                }
        } 

        if (mismatch) { 
                return HB_MC_FAIL;
        }
        return HB_MC_SUCCESS;
}

#ifdef VCS
int vcs_main(int argc, char ** argv) {
#else
int main(int argc, char ** argv) {
#endif
        bsg_pr_test_info("test_vec_add_xcel Regression Test\n");
        int rc = kernel_vec_add_xcel(argc, argv);
        bsg_pr_test_pass_fail(rc == HB_MC_SUCCESS);
        return rc;
}