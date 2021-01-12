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

#include "test_cgra_xcel_hb_tapeout.h"

#define ALLOC_NAME "default_allocator"

/*!
 * Runs a specific configuration on an 8x8 CGRA
 * Grid dimensions are prefixed at 1x1.
 * This tests uses the software/spmd/bsg_cuda_lite_runtime/cgra_xcel_hb_tapeout/ device
 * code in the bsg_manycore repository.
*/


int kernel_cgra_xcel_hb_tapeout (int argc, char **argv) {
        int rc;
        char *bin_path, *test_name;
        struct arguments_path args = {NULL, NULL};

        argp_parse (&argp_path, argc, argv, 0, 0, &args);
        bin_path = args.path;
        test_name = args.name;

        bsg_pr_test_info("Running the CUDA-Lite 8x8 CGRA vvadd Kernel.\n\n");

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
        * Allocate memory on the device for bitstream and arguments
        ******************************************************************************************************************/

        bsg_pr_test_info("Allocating memory for bitstream and arguments\n");

        uint32_t bstrm_size = FP32_OS_GEMM_BITSTREAM_SIZE;

        eva_t bstrm_device; 

        rc = hb_mc_device_malloc(&device, bstrm_size * sizeof(uint32_t), &bstrm_device);
        if (rc != HB_MC_SUCCESS) { 
                bsg_pr_err("failed to allocate memory on device.\n");
                return rc;
        }

        // Allocate space for the arguments

        uint32_t inst_size = FP32_OS_GEMM_INSTRUCTION_SIZE;
        uint32_t arg_size = FP32_OS_GEMM_CONFIG_INSTRUCTION_SIZE;
        uint32_t result_size = FP32_OS_GEMM_REF_SIZE;

        int verif_base_addr = FP32_OS_GEMM_VERIF_BASE_ADDR;

        eva_t inst_device, arg0_device, arg1_device, arg2_device, arg3_device,
              result_device;

        rc = hb_mc_device_malloc(&device, inst_size * sizeof(int), &inst_device);
        if (rc != HB_MC_SUCCESS) { 
                bsg_pr_err("failed to allocate memory on device.\n");
                return rc;
        }

        rc = hb_mc_device_malloc(&device, arg_size * sizeof(int), &arg0_device);
        if (rc != HB_MC_SUCCESS) { 
                bsg_pr_err("failed to allocate memory on device.\n");
                return rc;
        }
        rc = hb_mc_device_malloc(&device, arg_size * sizeof(int), &arg1_device);
        if (rc != HB_MC_SUCCESS) { 
                bsg_pr_err("failed to allocate memory on device.\n");
                return rc;
        }
        rc = hb_mc_device_malloc(&device, arg_size * sizeof(int), &arg2_device);
        if (rc != HB_MC_SUCCESS) { 
                bsg_pr_err("failed to allocate memory on device.\n");
                return rc;
        }
        rc = hb_mc_device_malloc(&device, arg_size * sizeof(int), &arg3_device);
        if (rc != HB_MC_SUCCESS) { 
                bsg_pr_err("failed to allocate memory on device.\n");
                return rc;
        }

        // Allocate space for results

        rc = hb_mc_device_malloc(&device, result_size * sizeof(int), &result_device);
        if (rc != HB_MC_SUCCESS) { 
                bsg_pr_err("failed to allocate memory on device.\n");
                return rc;
        }

        /*****************************************************************************************************************
        * Copy bitstream & arguments from host onto device DRAM.
        ******************************************************************************************************************/

        bsg_pr_test_info("Copying data into HB DRAM\n");

        void *dst = (void *) ((intptr_t) bstrm_device);
        void *src = (void *) &Bitstream[0];
        rc = hb_mc_device_memcpy (&device, dst, src, bstrm_size * sizeof(uint32_t), HB_MC_MEMCPY_TO_DEVICE);
        if (rc != HB_MC_SUCCESS) { 
                bsg_pr_err("failed to copy memory to device.\n");
                return rc;
        }

        dst = (void *) ((intptr_t) inst_device);
        src = (void *) &Instructions[0];
        rc = hb_mc_device_memcpy (&device, dst, src, inst_size * sizeof(uint32_t), HB_MC_MEMCPY_TO_DEVICE);
        if (rc != HB_MC_SUCCESS) { 
                bsg_pr_err("failed to copy memory to device.\n");
                return rc;
        }
        dst = (void *) ((intptr_t) arg0_device);
        src = (void *) &Arg0[0];
        rc = hb_mc_device_memcpy (&device, dst, src, arg_size * sizeof(uint32_t), HB_MC_MEMCPY_TO_DEVICE);
        if (rc != HB_MC_SUCCESS) { 
                bsg_pr_err("failed to copy memory to device.\n");
                return rc;
        }
        dst = (void *) ((intptr_t) arg1_device);
        src = (void *) &Arg1[0];
        rc = hb_mc_device_memcpy (&device, dst, src, arg_size * sizeof(uint32_t), HB_MC_MEMCPY_TO_DEVICE);
        if (rc != HB_MC_SUCCESS) { 
                bsg_pr_err("failed to copy memory to device.\n");
                return rc;
        }
        dst = (void *) ((intptr_t) arg2_device);
        src = (void *) &Arg2[0];
        rc = hb_mc_device_memcpy (&device, dst, src, arg_size * sizeof(uint32_t), HB_MC_MEMCPY_TO_DEVICE);
        if (rc != HB_MC_SUCCESS) { 
                bsg_pr_err("failed to copy memory to device.\n");
                return rc;
        }
        dst = (void *) ((intptr_t) arg3_device);
        src = (void *) &Arg3[0];
        rc = hb_mc_device_memcpy (&device, dst, src, arg_size * sizeof(uint32_t), HB_MC_MEMCPY_TO_DEVICE);
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

        int cuda_argv[11] = { bstrm_device, bstrm_size,
                              inst_device, arg0_device, arg1_device,
                              arg2_device, arg3_device,
                              inst_size,
                              result_device,
                              verif_base_addr,
                              result_size };

        /*****************************************************************************************************************
        * Enquque grid of tile groups, pass in grid and tile group dimensions, kernel name, number and list of input arguments
        ******************************************************************************************************************/

        bsg_pr_test_info("Enqueuing kernel to HB\n");

        rc = hb_mc_kernel_enqueue (&device, grid_dim, tg_dim, "kernel_cgra_xcel_hb_tapeout", 11, cuda_argv);
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

        bsg_pr_test_info("Copying result from HB DRAM into host...\n");

        uint32_t result_host[result_size];
        src = (void *) ((intptr_t) result_device);
        dst = (void *) &result_host[0];
        rc = hb_mc_device_memcpy (&device, (void *) dst, src, result_size * sizeof(uint32_t), HB_MC_MEMCPY_TO_HOST);
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
        * Dump the results. 
        ******************************************************************************************************************/

        for (int i = 0; i < result_size; i++) {
          bsg_pr_test_info("Result[%d] = 0x%08" PRIx32 "\n", i, result_host[i]);
        }

        /*****************************************************************************************************************
        * Compare the results. 
        ******************************************************************************************************************/

        int mismatch = 0; 
        for (int i = 0; i < result_size; i++) {
                if (Reference[i] != result_host[i]) {
                        bsg_pr_err(BSG_RED("Mismatch: ") "Result[%d]: 0x%08" PRIx32 "\t Expected: 0x%08" PRIx32 "\n", i, result_host[i], Reference[i]);
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
        bsg_pr_test_info("test_cgra_xcel_hb_tapeout Regression Test\n");
        int rc = kernel_cgra_xcel_hb_tapeout(argc, argv);
        bsg_pr_test_pass_fail(rc == HB_MC_SUCCESS);
        return rc;
}
