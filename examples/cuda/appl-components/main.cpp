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
#define BUF_FACTOR 2049
#define BUF_SIZE (MAX_WORKERS * HB_L2_CACHE_LINE_WORDS * BUF_FACTOR)

struct CC_F {
  uintE *IDs, *prevIDs;
  CC_F( uintE* _IDs, uintE* _prevIDs ) : IDs( _IDs ), prevIDs( _prevIDs )
  {
  }
  inline bool update( uintE s, uintE d )
  { // Update function writes min ID
    uintE origID = IDs[d];
    if ( IDs[s] < origID ) {
      IDs[d] = min( origID, IDs[s] );
      if ( origID == prevIDs[d] )
        return 1;
    }
    return 0;
  }
  inline bool updateAtomic( uintE s, uintE d )
  { // atomic Update
    uintE origID = IDs[d];
    return ( writeMin( &IDs[d], IDs[s] ) && origID == prevIDs[d] );
  }
  inline bool cond( uintE d ) { return cond_true( d ); } // does nothing
};

// function used by vertex map to sync prevIDs with IDs
struct CC_Vertex_F {
  uintE *IDs, *prevIDs;
  CC_Vertex_F( uintE* _IDs, uintE* _prevIDs )
      : IDs( _IDs ), prevIDs( _prevIDs )
  {
  }
  inline bool operator()( uintE i )
  {
    prevIDs[i] = IDs[i];
    return 1;
  }
};

template <class vertex>
void Compute( graph<vertex>& GA, uintE* goldenIDs )
{
  size_t   n   = GA.n;
  uintE *IDs = newA( uintE, n ), *prevIDs = newA( uintE, n );
  appl::parallel_for( size_t( 0 ), n, [&]( size_t i ) { IDs[i] = i; } );

  bool* frontier = newA( bool, n );
  appl::parallel_for( size_t( 0 ), n, [&]( size_t i ) { frontier[i] = 1; } );
  vertexSubset Frontier(
      n, n, frontier ); // initial frontier contains all vertices

  while ( !Frontier.isEmpty() ) { // iterate until IDS converge
    vertexMap( Frontier, CC_Vertex_F( IDs, prevIDs ) );
    vertexSubset output = edgeMap( GA, Frontier, CC_F( IDs, prevIDs ) );
    Frontier.del();
    Frontier = output;
  }

  // verify or dump outputs
  for (size_t i = 0; i < n; i++) {
    goldenIDs[i] = IDs[i];
  }

  Frontier.del();
  free( IDs );
  free( prevIDs );
}

int kernel_appl_components (int argc, char **argv) {
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

        bsg_pr_test_info("Running the Ligra Components on one %dx%d tile groups.\n\n", bsg_tiles_X, bsg_tiles_Y);

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
                 * Run Components natively
                 ******************************************************************************************************************/
                uintE* goldenIDs = newA( uintE, G.n );
                Compute(G, goldenIDs);

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
                BSG_CUDA_CALL(hb_mc_kernel_enqueue (&device, grid_dim, tg_dim, "kernel_appl_components", 5, cuda_argv));

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

                for (int i = 0; i < G.n; i++) {
                  printf("IDs[%d] = %d\n", i, host_result[i]);
                  if (host_result[i] != goldenIDs[i]) {
                     bsg_pr_err(BSG_RED("Mismatch: ") "result[%d]: 0x%08" PRIx32 " != golden[%d]: 0x%08" PRIx32 "\n",
                                i, host_result[i], i, goldenIDs[i]);
                    return HB_MC_FAIL;
                  }
                }

        }
        BSG_CUDA_CALL(hb_mc_device_finish(&device));

        return HB_MC_SUCCESS;
}

declare_program_main("test_appl_components", kernel_appl_components);
