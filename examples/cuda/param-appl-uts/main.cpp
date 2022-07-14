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

#define PLATFORM_BYTE_ORDER IS_BIG_ENDIAN
#include "brg_sha1.h"
#include "uts-datasets.hpp"
#include "uts-common.hpp"
#include "uts-scalar.hpp"

#define ALLOC_NAME "default_allocator"
#define MAX_WORKERS 128
#define HB_L2_CACHE_LINE_WORDS 16
#define BUF_FACTOR 16385
#define BUF_SIZE (MAX_WORKERS * HB_L2_CACHE_LINE_WORDS * BUF_FACTOR)

int kernel_appl_uts (int argc, char **argv) {
        int rc;
        char *bin_path, *test_name;
        struct arguments_path args = {NULL, NULL};

        argp_parse (&argp_path, argc, argv, 0, 0, &args);
        bin_path = args.path;
        test_name = args.name;

        bsg_pr_test_info("Running the UTS Kernel on one %dx%d tile groups.\n\n", bsg_tiles_X, bsg_tiles_Y);

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
                 * Allocate memory on the device.
                 ******************************************************************************************************************/

                eva_t device_result;
                BSG_CUDA_CALL(hb_mc_device_malloc(&device, 64 * sizeof(uint32_t), &device_result)); // buffer for return results
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
                Dataset* dataset_ptr = choose_dataset("test");
                // debug print of the dataset
                printf("Using Dataset %s:\n", dataset_ptr->str);

                int t, a, g;

                nonLeafProb = dataset_ptr->nonLeafProb;
                nonLeafBF   = dataset_ptr->nonLeafBF;
                rootId      = dataset_ptr->rootId;
                t           = dataset_ptr->t;
                a           = dataset_ptr->a;
                b_0         = dataset_ptr->b_0;
                gen_mx      = dataset_ptr->gen_mx;
                shiftDepth  = dataset_ptr->shiftDepth;
                g           = dataset_ptr->g;

                // Replace -1's with defaults

                if ( nonLeafProb < 0 )
                  nonLeafProb = 0.234375;
                if ( nonLeafBF < 0 )
                  nonLeafBF = 4;
                if ( rootId < 0 )
                  rootId = 0;
                if ( t < 0 )
                  t = 1;
                if ( a < 0 )
                  a = 0;
                if ( b_0 < 0 )
                  b_0 = 4.0;
                if ( gen_mx < 0 )
                  gen_mx = 6;
                if ( shiftDepth < 0 )
                  shiftDepth = 0.5;
                if ( g < 0 )
                  g = 1;

                type               = (tree_t)t;
                shape_fn           = (geoshape_t)a;
                computeGranularity = std::max( 1, g );

                std::cout << "Got args:" << std::endl;
                std::cout << "nonLeafProb = " << nonLeafProb << std::endl;
                std::cout << "nonLeafBF   = " << nonLeafBF << std::endl;
                std::cout << "rootId      = " << rootId << std::endl;
                std::cout << "t           = " << t << std::endl;
                std::cout << "a           = " << a << std::endl;
                std::cout << "b_0         = " << b_0 << std::endl;
                std::cout << "gen_mx      = " << gen_mx << std::endl;
                std::cout << "shiftDepth  = " << shiftDepth << std::endl;
                std::cout << "g           = " << g << std::endl;

                // execute locally
                verify = true;
                numNodes  = 0;
                maxHeight = 0;
                numLeaves = 0;

                uts_scalar();

                UTSResults scalar_result( "host scalar", numNodes, maxHeight, numLeaves );

                verify_results( &scalar_result, dataset_ptr->str );

                int N = FIB_IN;
                int gsize = FIB_GSIZE;
                const uint32_t cuda_argv[4] = {device_result, N, gsize, dram_buffer};

                /*****************************************************************************************************************
                 * Enquque grid of tile groups, pass in grid and tile group dimensions, kernel name, number and list of input arguments
                 ******************************************************************************************************************/
                BSG_CUDA_CALL(hb_mc_kernel_enqueue (&device, grid_dim, tg_dim, "kernel_appl_uts", 4, cuda_argv));

                /*****************************************************************************************************************
                 * Launch and execute all tile groups on device and wait for all to finish.
                 ******************************************************************************************************************/
                BSG_CUDA_CALL(hb_mc_device_tile_groups_execute(&device));

                /*****************************************************************************************************************
                 * Copy result back from device DRAM into host memory.
                 ******************************************************************************************************************/
                uint32_t host_result[64];
                void *src = (void *) ((intptr_t) device_result);;
                void *dst = (void *) &host_result[0];
                BSG_CUDA_CALL(hb_mc_device_memcpy (&device, (void *) dst, src, 64 * sizeof(uint32_t), HB_MC_MEMCPY_TO_HOST));

                struct state_t mystate;
                for (int i = 0; i < 20; i++) {
                  mystate.state[i] = i;
                }
                rng_init( mystate.state, 14850 );
                for (int i = 0; i < 20; i++) {
                  printf("state[%d] = %d\n", i, mystate.state[i]);
                }
                int rand1 = rng_nextrand( mystate.state );
                int rand2 = rng_nextrand( mystate.state );
                printf("rng_nextrand = %d\n", rand1);
                printf("rng_nextrand = %d\n", rand2);
                if (rand1 != host_result[0] || rand2 != host_result[1]) {
                  return HB_MC_FAIL;
                }
                /*****************************************************************************************************************
                 * Freeze the tiles and memory manager cleanup.
                 ******************************************************************************************************************/
                BSG_CUDA_CALL(hb_mc_device_program_finish(&device));

        }
        BSG_CUDA_CALL(hb_mc_device_finish(&device));

        return HB_MC_SUCCESS;
}

declare_program_main("test_appl_uts", kernel_appl_uts);
