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

#include "test_smu_toplevel_xcel.h"

#define ALLOC_NAME "default_allocator"

// Matrix data dimension (16KB, stripped across all L2 banks)
#define NROWS 64
#define NCOLS 64

/*!
 * Allocate a large array on host, copy it to HB DRAM.
 * Configure SMU to stream data from DRAM back to V5 DMEM.
 * V5 core verifies that the data is as expected.
 * Grid dimensions are prefixed at 1x1.
 * This tests uses the software/spmd/bsg_cuda_lite_runtime/smu_toplevel_xcel/ device
 * code in the bsg_manycore repository.
*/


int kernel_smu_toplevel_xcel (int argc, char **argv) {
        int rc;
        char *bin_path, *test_name;
        struct arguments_path args = {NULL, NULL};

        argp_parse (&argp_path, argc, argv, 0, 0, &args);
        bin_path = args.path;
        test_name = args.name;

        bsg_pr_test_info("Running the CUDA SMU Toplevel Xcel Kernel on one 11x18 manycore.\n\n");

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
        * Allocate memory on the device for array
        ******************************************************************************************************************/

        uint32_t N = NROWS * NCOLS;
        bsg_pr_test_info("Allocating memory for array\n");

        eva_t array_device; 
        rc = hb_mc_device_malloc(&device, N * sizeof(uint32_t), &array_device); /* allocate array[N] on the device */
        if (rc != HB_MC_SUCCESS) { 
                bsg_pr_err("failed to allocate memory on device.\n");
                return rc;
        }

        /*****************************************************************************************************************
        * Initialize array with regular data
        ******************************************************************************************************************/

        bsg_pr_test_info("Filling arary with data\n");

        uint32_t* array_host = malloc(N * sizeof(uint32_t)); /* allocate array on the host */
        for (int i = 0; i < N; i++) {
                array_host[i] = 0xdead0000 + i;
        }

        /*****************************************************************************************************************
        * Copy array from host onto device DRAM.
        ******************************************************************************************************************/

        bsg_pr_test_info("Copying data into HB DRAM\n");

        void *dst = (void *) ((intptr_t) array_device);
        void *src = (void *) &array_host[0];
        rc = hb_mc_device_memcpy (&device, dst, src, N * sizeof(uint32_t), HB_MC_MEMCPY_TO_DEVICE);
        if (rc != HB_MC_SUCCESS) { 
                bsg_pr_err("failed to copy memory to device.\n");
                return rc;
        }

        /*****************************************************************************************************************
        * Define tg_dim_x/y: number of tiles in each tile group
        * Calculate grid_dim_x/y: number of tile groups needed
        ******************************************************************************************************************/

        // PP: since the accelerator will do the work a 1x1 tile group should be enough?
        hb_mc_dimension_t tg_dim = { .x = 2, .y = 2};

        hb_mc_dimension_t grid_dim = { .x = 1, .y = 1};


        /*****************************************************************************************************************
        * Prepare list of input arguments for kernel.
        ******************************************************************************************************************/

        int cuda_argv[1] = {array_device};

        /*****************************************************************************************************************
        * Enquque grid of tile groups, pass in grid and tile group dimensions, kernel name, number and list of input arguments
        ******************************************************************************************************************/

        bsg_pr_test_info("Enqueuing kernel to HB\n");

        rc = hb_mc_kernel_enqueue (&device, grid_dim, tg_dim, "kernel_smu_toplevel_xcel", 1, cuda_argv);
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
        * Copy array back from device DRAM into host memory. 
        ******************************************************************************************************************/

        bsg_pr_test_info("Copying array from HB DRAM into host...\n");

        uint32_t* result = malloc(N * sizeof(uint32_t));
        src = (void *) ((intptr_t) array_device);
        dst = (void *) &result[0];
        rc = hb_mc_device_memcpy (&device, dst, src, N * sizeof(uint32_t), HB_MC_MEMCPY_TO_HOST);
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

        int mismatch = 0;
        for (int i = 0; i < N; i++) {
                if (array_host[i] != result[i]) {
                        bsg_pr_err(BSG_RED("Mismatch: ") "Result[%d]:  0x%08" PRIx32 "\t Expected: 0x%08" PRIx32 "\n", i , result[i], array_host[i]);
                        mismatch = 1;
                }
        } 

        free(array_host);
        free(result);

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
        bsg_pr_test_info("test_smu_toplevel_xcel Regression Test\n");
#ifdef SMU_TOPLEVEL_XCEL
        bsg_pr_test_info("SMU_TOPLEVEL_XCEL defined!\n");
#else
        bsg_pr_test_info("SMU_TOPLEVEL_XCEL NOT defined! Please check the build system...\n");
        return HB_MC_FAIL;
#endif
        int rc = kernel_smu_toplevel_xcel(argc, argv);
        bsg_pr_test_pass_fail(rc == HB_MC_SUCCESS);
        return rc;
}
