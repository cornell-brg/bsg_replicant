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
#include <vector>

// Ligra headers
#include "ligra.h"

#define ALLOC_NAME "default_allocator"
#define MAX_WORKERS 128
#define HB_L2_CACHE_LINE_WORDS 16
#define BUF_FACTOR 16385
#define BUF_SIZE (MAX_WORKERS * HB_L2_CACHE_LINE_WORDS * BUF_FACTOR)

struct BFS_F {
  uintE* Parents;
  uintE* bfsLvls;
  uintE  lvl;
  BFS_F( uintE* _Parents, uintE* _bfsLvls, uintE  _lvl )
    : Parents( _Parents ), bfsLvls( _bfsLvls ), lvl( _lvl ) {}

  inline bool update( uintE s, uintE d )
  { // Update
    if ( Parents[d] == UINT_E_MAX ) {
      Parents[d] = s;
      bfsLvls[d] = lvl;
      return 1;
    }
    else
      return 0;
  }
  inline bool updateAtomic( uintE s, uintE d )
  { // atomic version of Update
    return update( s, d );
  }
  // cond function checks if vertex has been visited yet
  inline bool cond( uintE d ) { return ( Parents[d] == UINT_E_MAX ); }
};

static int symbol_to_eva(hb_mc_device_t *dev, const char *symbol, hb_mc_eva_t *eva)
{
    hb_mc_pod_t *pod = &dev->pods[dev->default_pod_id];
    hb_mc_program_t *prog = pod->program;
    BSG_CUDA_CALL(hb_mc_loader_symbol_to_eva(
                      prog->bin
                      ,prog->bin_size
                      ,symbol
                      ,eva));
}
int kernel_appl_bfs (int argc, char **argv) {
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
        uint32_t iter         = atoi(argv[6]);

        // debug
        std::cout << "Ligra command line parsed -- iFile:" << iFile << " grain_size=" << grain_size
                  << " symmetric?=" << symmetric << " iter=" << iter << std::endl;
        std::cout << "size of symmetricVertex " << sizeof(symmetricVertex) << std::endl;

        hb_mc_device_t device;
        BSG_CUDA_CALL(hb_mc_device_init(&device, test_name.c_str(), 0));

        bsg_pr_test_info("Running the Ligra BFS on one %dx%d tile groups.\n\n", bsg_tiles_X, bsg_tiles_Y);

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
                 * Run BFS natively
                 ******************************************************************************************************************/

                int32_t start = START_VERTEX;
                int32_t n     = G.n;
                uintE* Parents = newA( uintE, n );
                uintE* bfsLvls = newA( uintE, n );
                for (int32_t i = 0; i < n; i++) {
                  Parents[i] = UINT_E_MAX;
                  bfsLvls[i] = UINT_E_MAX;
                }

                bsg_pr_info("Starting Ligra BFS with start index %d.\n\n", start);

                Parents[start] = start;
                vertexSubset Frontier( n, start ); // creates initial frontier

                uintE lvl = 0;
                //while ( !Frontier.isEmpty() ) {    // loop until frontier is empty
                while (lvl < iter) {
                  vertexSubset output = edgeMap( G, Frontier, BFS_F( Parents, bfsLvls, lvl ) );
                  Frontier.del();
                  Frontier = output; // set new frontier
                  lvl++;
                }

                // calculate the solution
                uintE *ParentsNext = newA(uintE, n);
                uintE *bfsLvlsNext = newA(uintE, n);
                memcpy(ParentsNext, Parents, n*sizeof(uintE));
                memcpy(bfsLvlsNext, bfsLvls, n*sizeof(uintE));
                vertexSubset FrontierNext = edgeMap(G, Frontier, BFS_F(ParentsNext, bfsLvlsNext, lvl));

                // write the current frontier, parent array, lvls arrays, and lvl to device
                // write the current frontier
                hb_mc_eva_t d_Frontier, d_Parents, d_bfsLvls, d_sparseFrontier;
                hb_mc_eva_t g_Frontier_ptr, g_Parents_ptr, g_bfsLvls_ptr, g_sparseFrontier_ptr;

                // record #nonzero
                uint32_t nonZeroes = Frontier.numNonzeros();

                bsg_pr_info("numNonZero in this iteration: %u\n\n", nonZeroes);

                // (1) find global symbols and allocate memory
                // Parents
                BSG_CUDA_CALL(symbol_to_eva(&device, "g_Parents", &g_Parents_ptr));
                BSG_CUDA_CALL(hb_mc_device_malloc(&device, n*sizeof(uintE), &d_Parents));
                // bfsLvls
                BSG_CUDA_CALL(symbol_to_eva(&device, "g_bfsLvls", &g_bfsLvls_ptr));
                BSG_CUDA_CALL(hb_mc_device_malloc(&device, n*sizeof(uintE), &d_bfsLvls));
                // Frontier
                BSG_CUDA_CALL(symbol_to_eva(&device, "g_Frontier", &g_Frontier_ptr));
                BSG_CUDA_CALL(hb_mc_device_malloc(&device, n*sizeof(bool), &d_Frontier));
                // Frontier
                BSG_CUDA_CALL(symbol_to_eva(&device, "g_sparseFrontier", &g_sparseFrontier_ptr));
                BSG_CUDA_CALL(hb_mc_device_malloc(&device, nonZeroes*sizeof(uintE), &d_sparseFrontier));

                // record density
                uint32_t isDense = (uint32_t)Frontier.dense();

                // debug
                if (isDense) {
                  bsg_pr_info(" this frontier is dense \n\n");
                }
                symmetricVertex* v_5786 = &(G.V[5786]);
                uintE d = v_5786->getInDegree();
                for ( size_t j = 0; j < d; j++ ) {
                  bsg_pr_info(" 5786 is neighbor with %u which is set to %u \n\n", v_5786->getInNeighbor( j ), Frontier.d[ v_5786->getInNeighbor( j )]);
                }
                bsg_pr_info(" 5786 in parent is %u\n\n", Parents[5786]);
                bsg_pr_info(" after iter 5786 in parent is %u\n\n", ParentsNext[5786]);
                bsg_pr_info(" 5786 in bfsLvls is %u\n\n", bfsLvls[5786]);
                bsg_pr_info(" after iter 5786 in bfsLvls is %u\n\n", bfsLvlsNext[5786]);
                bsg_pr_info(" d_bfsLvls addr is 0x%x\n\n", d_bfsLvls);

                // DMA to device
                if (isDense) {
                  hb_mc_dma_htod_t htod [] = {
                      {.d_addr = d_Parents, .h_addr = Parents, .size = n*sizeof(uintE) }
                      ,{.d_addr = d_bfsLvls, .h_addr = bfsLvls, .size = n*sizeof(uintE) }
                      ,{.d_addr = d_Frontier, .h_addr = Frontier.d, .size = n*sizeof(bool) }
                      ,{.d_addr = g_Frontier_ptr, .h_addr = &d_Frontier, .size = sizeof(d_Frontier) }
                      ,{.d_addr = g_Parents_ptr, .h_addr = &d_Parents, .size = sizeof(d_Parents) }
                      ,{.d_addr = g_bfsLvls_ptr, .h_addr = &d_bfsLvls, .size = sizeof(d_bfsLvls) }
                  };
                  BSG_CUDA_CALL(hb_mc_device_dma_to_device(&device, htod, sizeof(htod)/sizeof(htod[0])));
                } else {
                  hb_mc_dma_htod_t htod [] = {
                      {.d_addr = d_Parents, .h_addr = Parents, .size = n*sizeof(uintE) }
                      ,{.d_addr = d_bfsLvls, .h_addr = bfsLvls, .size = n*sizeof(uintE) }
                      ,{.d_addr = d_sparseFrontier, .h_addr = Frontier.s, .size = nonZeroes*sizeof(uintE) }
                      ,{.d_addr = g_sparseFrontier_ptr, .h_addr = &d_sparseFrontier, .size = sizeof(d_sparseFrontier) }
                      ,{.d_addr = g_Parents_ptr, .h_addr = &d_Parents, .size = sizeof(d_Parents) }
                      ,{.d_addr = g_bfsLvls_ptr, .h_addr = &d_bfsLvls, .size = sizeof(d_bfsLvls) }
                  };
                  BSG_CUDA_CALL(hb_mc_device_dma_to_device(&device, htod, sizeof(htod)/sizeof(htod[0])));
                }

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
                const uint32_t cuda_argv[7] = {device_result, G.hb_V, G.n, G.m, nonZeroes, isDense, dram_buffer};

                /*****************************************************************************************************************
                 * Enquque grid of tile groups, pass in grid and tile group dimensions, kernel name, number and list of input arguments
                 ******************************************************************************************************************/
                BSG_CUDA_CALL(hb_mc_kernel_enqueue (&device, grid_dim, tg_dim, "kernel_appl_bfs", 7, cuda_argv));

                /*****************************************************************************************************************
                 * Launch and execute all tile groups on device and wait for all to finish.
                 ******************************************************************************************************************/
                BSG_CUDA_CALL(hb_mc_device_tile_groups_execute(&device));

                /*****************************************************************************************************************
                 * Copy result back from device DRAM into host memory.
                 ******************************************************************************************************************/
                std::vector<uint32_t> host_result(G.n);
                hb_mc_dma_dtoh_t dtoh = {
                    .d_addr = d_bfsLvls
                    ,.h_addr = &host_result[0]
                    ,.size = G.n*sizeof(uint32_t)
                };
                BSG_CUDA_CALL(hb_mc_device_dma_to_host(&device, &dtoh, 1));

                /*****************************************************************************************************************
                 * Freeze the tiles and memory manager cleanup.
                 ******************************************************************************************************************/
                BSG_CUDA_CALL(hb_mc_device_program_finish(&device));

                bool failed = false;
                for (int i = 0; i < G.n; i++) {
                  printf("result[%d] = %d\n", i, host_result[i]);
                  if (host_result[i] != bfsLvlsNext[i]) {
                     bsg_pr_err(BSG_RED("Mismatch: ") "result[%d]: 0x%08" PRIx32 " != bfsLvls[%d]: 0x%08" PRIx32 "\n",
                                i, host_result[i], i, bfsLvlsNext[i]);
                     failed = true;
                  }
                }
                if (failed) {
                  return HB_MC_FAIL;
                }

        }
        BSG_CUDA_CALL(hb_mc_device_finish(&device));

        return HB_MC_SUCCESS;
}

declare_program_main("test_appl_bfs", kernel_appl_bfs);
