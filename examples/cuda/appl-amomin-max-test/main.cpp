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
#include <iostream>

// Ligra headers
#include "ligra.h"

#define ALLOC_NAME "default_allocator"
#define MAX_WORKERS 128
#define HB_L2_CACHE_LINE_WORDS 16
#define BUF_FACTOR 129
#define BUF_SIZE (MAX_WORKERS * HB_L2_CACHE_LINE_WORDS * BUF_FACTOR)

int kernel_appl_amo_test (int argc, char **argv) {
        int rc;

        /*****************************************************************************************************************
        * Define path to binary.
        * Initialize device, load binary and unfreeze tiles.
        ******************************************************************************************************************/
        std::string bin_path  = argv[1];
        std::string test_name = argv[2];
        std::string iFile     = argv[3];
        uint32_t grain_size   = atoi(argv[4]);
        uint32_t symmetric    = atoi(argv[5]);
        uint32_t rounds       = atoi(argv[6]);

        // debug
        std::cout << "Ligra command line parsed -- iFile:" << iFile << " grain_size=" << grain_size
                  << " symmetric?=" << symmetric << " rounds=" << rounds << std::endl;
        std::cout << "size of symmetricVertex " << sizeof(symmetricVertex) << std::endl;

        hb_mc_device_t device;
        BSG_CUDA_CALL(hb_mc_device_init(&device, test_name.c_str(), 0));

        bsg_pr_test_info("Running the Ligra AMO tests on one %dx%d tile groups.\n\n", bsg_tiles_X, bsg_tiles_Y);

        hb_mc_pod_id_t pod;
        hb_mc_device_foreach_pod_id(&device, pod)
        {

                bsg_pr_info("Loading program for test %s onto pod %d\n", test_name.c_str(), pod);
                BSG_CUDA_CALL(hb_mc_device_set_default_pod(&device, pod));
                BSG_CUDA_CALL(hb_mc_device_program_init(&device, bin_path.c_str(), ALLOC_NAME, 0));

                /*****************************************************************************************************************
                 * Ligra host code
                 ******************************************************************************************************************/
                graph<symmetricVertex> G = readGraph<symmetricVertex>(
                    iFile.c_str(), false, (bool)symmetric, false, false, device);

                /*****************************************************************************************************************
                 * Allocate memory on the device.
                 ******************************************************************************************************************/

                eva_t device_result;
                BSG_CUDA_CALL(hb_mc_device_malloc(&device, G.n * sizeof(uint32_t), &device_result)); // buffer for return results
                eva_t dram_buffer;
                BSG_CUDA_CALL(hb_mc_device_malloc(&device, BUF_SIZE * sizeof(uint32_t), &dram_buffer));

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
                const uint32_t cuda_argv[5] = {device_result, G.hb_V, G.n, G.m, dram_buffer};

                /*****************************************************************************************************************
                 * Enquque grid of tile groups, pass in grid and tile group dimensions, kernel name, number and list of input arguments
                 ******************************************************************************************************************/
                BSG_CUDA_CALL(hb_mc_kernel_enqueue (&device, grid_dim, tg_dim, "kernel_appl_amo_test", 5, cuda_argv));

                /*****************************************************************************************************************
                 * Launch and execute all tile groups on device and wait for all to finish.
                 ******************************************************************************************************************/
                BSG_CUDA_CALL(hb_mc_device_tile_groups_execute(&device));

                /*****************************************************************************************************************
                 * Copy result back from device DRAM into host memory.
                 ******************************************************************************************************************/
                uint32_t host_result[G.n];
                void *src = (void *) ((intptr_t) device_result);;
                void *dst = (void *) &host_result[0];
                BSG_CUDA_CALL(hb_mc_device_memcpy (&device, (void *) dst, src, G.n * sizeof(uint32_t), HB_MC_MEMCPY_TO_HOST));

                /*****************************************************************************************************************
                 * Freeze the tiles and memory manager cleanup.
                 ******************************************************************************************************************/
                BSG_CUDA_CALL(hb_mc_device_program_finish(&device));

                int golden[4] = {14850, 61801, 0, -1};
                for (int i = 0; i < 4; i++) {
                  if (host_result[i] != golden[i]) {
                     bsg_pr_err(BSG_RED("Mismatch: ") "result[%d]: 0x%08" PRIx32 " != golden[%d]: 0x%08" PRIx32 "\n",
                                i, host_result[i], i, golden[i]);
                    return HB_MC_FAIL;
                  }
                }

        }
        BSG_CUDA_CALL(hb_mc_device_finish(&device));

        return HB_MC_SUCCESS;
}

declare_program_main("test_appl_amo_test", kernel_appl_amo_test);