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

struct BFS_F {
  uintE* Parents;
  BFS_F( uintE* _Parents ) : Parents( _Parents ) {}
  inline bool update( uintE s, uintE d )
  { // Update
    if ( Parents[d] == UINT_E_MAX ) {
      Parents[d] = s;
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

template <class vertex>
void Compute( graph<vertex>& GA, char* out_file, char* ref_file )
{
  int32_t start = 0;
  int32_t n     = GA.n;
  // creates Parents array, initialized to all -1, except for start
  uintE* Parents = newA( uintE, n );
  appl::parallel_for( 0, n,
                      [&]( int32_t i ) { Parents[i] = UINT_E_MAX; } );
  Parents[start] = start;
  vertexSubset Frontier( n, start ); // creates initial frontier
  while ( !Frontier.isEmpty() ) {    // loop until frontier is empty
    vertexSubset output = edgeMap( GA, Frontier, BFS_F( Parents ) );
    Frontier.del();
    Frontier = output; // set new frontier
  }

  // verify or dump outputs
  if ( ref_file ) {
    verify<uintE>( Parents, ref_file, n );
  }
  else if ( out_file ) {
    output<uintE>( Parents, out_file, n );
  }

  Frontier.del();
  free( Parents );
}

uint32_t host_fib(uint32_t n) {
  if (n < 2)
    return n;
  else
    return host_fib(n-1) + host_fib(n-2);
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
        uint32_t rounds       = atoi(argv[6]);

        // debug
        std::cout << "Ligra command line parsed -- iFile:" << iFile << " grain_size=" << grain_size
                  << " symmetric?=" << symmetric << " rounds=" << rounds << std::endl;
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
                Compute(G, NULL, NULL);
                std::cout << "G.V[1].getOutDegree() = " << G.V[1].getOutDegree() << std::endl;
                for (int i = 0; i < G.V[1].getOutDegree(); i++) {
                  std::cout << "G.V[1].getInNeighbor(" << i << ") = " << G.V[1].getInNeighbor(i) << std::endl;
                }

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
                int N = FIB_IN;
                int gsize = FIB_GSIZE;
                const uint32_t cuda_argv[5] = {device_result, G.hb_V, G.n, G.m, dram_buffer};

                /*****************************************************************************************************************
                 * Enquque grid of tile groups, pass in grid and tile group dimensions, kernel name, number and list of input arguments
                 ******************************************************************************************************************/
                BSG_CUDA_CALL(hb_mc_kernel_enqueue (&device, grid_dim, tg_dim, "kernel_appl_bfs", 5, cuda_argv));

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

                /*****************************************************************************************************************
                 * Freeze the tiles and memory manager cleanup.
                 ******************************************************************************************************************/
                BSG_CUDA_CALL(hb_mc_device_program_finish(&device));

                int32_t expected = host_fib(N);

                if (host_result[0] != expected) {
                  bsg_pr_err(BSG_RED("Mismatch: ") "fib %d = %d != expected %d\n", N, host_result[0], expected);
                  return HB_MC_FAIL;
                }

        }
        BSG_CUDA_CALL(hb_mc_device_finish(&device));

        return HB_MC_SUCCESS;
}

declare_program_main("test_appl_bfs", kernel_appl_bfs);
