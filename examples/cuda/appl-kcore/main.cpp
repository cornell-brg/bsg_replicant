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

struct Update_Deg {
  intE* Degrees;
  Update_Deg( intE* _Degrees ) : Degrees( _Degrees ) {}
  inline bool update( uintE s, uintE d )
  {
    Degrees[d]--;
    return 1;
  }
  inline bool updateAtomic( uintE s, uintE d )
  {
    writeAdd( &Degrees[d], -1 );
    return 1;
  }
  inline bool cond( uintE d ) { return Degrees[d] > 0; }
};

template <class vertex>
struct Deg_LessThan_K {
  vertex* V;
  uintE*  coreNumbers;
  intE*   Degrees;
  uintE   k;
  Deg_LessThan_K( vertex* _V, intE* _Degrees, uintE* _coreNumbers,
                  uintE _k )
      : V( _V ), k( _k ), Degrees( _Degrees ), coreNumbers( _coreNumbers )
  {
  }
  inline bool operator()( uintE i )
  {
    if ( Degrees[i] < k ) {
      coreNumbers[i] = k - 1;
      Degrees[i]     = 0;
      return true;
    }
    else
      return false;
  }
};

template <class vertex>
struct Deg_AtLeast_K {
  vertex* V;
  intE*   Degrees;
  uintE   k;
  Deg_AtLeast_K( vertex* _V, intE* _Degrees, uintE _k )
      : V( _V ), k( _k ), Degrees( _Degrees )
  {
  }
  inline bool operator()( uintE i ) { return Degrees[i] >= k; }
};

// assumes symmetric graph
// 1) iterate over all remaining active vertices
// 2) for each active vertex, remove if induced degree < k. Any vertex
// removed has
//    core-number (k-1) (part of (k-1)-core, but not k-core)
// 3) stop once no vertices are removed. Vertices remaining are in the
// k-core.
template <class vertex>
uint32_t Compute( graph<vertex>& GA ) {
  const uint32_t n      = GA.n;
  bool*      active = newA( bool, n );
  appl::parallel_for( uint32_t( 0 ), n, [&]( uint32_t i ) { active[i] = 1; } );
  vertexSubset Frontier( n, n, active );
  uintE*       coreNumbers = newA( uintE, n );
  intE*        Degrees     = newA( intE, n );
  {
    appl::parallel_for( uint32_t( 0 ), n, [&]( uint32_t i ) {
      coreNumbers[i] = 0;
      Degrees[i]     = GA.V[i].getOutDegree();
    } );
  }
  uint32_t largestCore = -1;
  for ( uint32_t k = 1; k <= n; k++ ) {
    while ( true ) {
      vertexSubset toRemove = vertexFilter(
          Frontier,
          Deg_LessThan_K<vertex>( GA.V, Degrees, coreNumbers, k ) );
      vertexSubset remaining = vertexFilter(
          Frontier, Deg_AtLeast_K<vertex>( GA.V, Degrees, k ) );
      Frontier.del();
      Frontier = remaining;
      if ( 0 == toRemove.numNonzeros() ) { // fixed point. found k-core
        toRemove.del();
        break;
      }
      else {
        edgeMap( GA, toRemove, Update_Deg( Degrees ), -1, no_output );
        toRemove.del();
      }
    }
    if ( Frontier.numNonzeros() == 0 ) {
      largestCore = k - 1;
      break;
    }
  }
  // cout << "largestCore was " << largestCore << endl;

  Frontier.del();
  free( coreNumbers );
  free( Degrees );

  return largestCore;
}

int kernel_appl_kcore (int argc, char **argv) {
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

        bsg_pr_test_info("Running the Ligra KCore on one %dx%d tile groups.\n\n", bsg_tiles_X, bsg_tiles_Y);

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
                 * Run KCore natively
                 ******************************************************************************************************************/
                uint32_t largestCore = Compute(G);

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
                BSG_CUDA_CALL(hb_mc_kernel_enqueue (&device, grid_dim, tg_dim, "kernel_appl_kcore", 5, cuda_argv));

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

                if (host_result[0] != largestCore) {
                  bsg_pr_err(BSG_RED("Mismatch: ") "result: 0x%08" PRIx32 " != largestCore: 0x%08" PRIx32 "\n",
                             host_result[0], largestCore);
                  return HB_MC_FAIL;
                }

        }
        BSG_CUDA_CALL(hb_mc_device_finish(&device));

        return HB_MC_SUCCESS;
}

declare_program_main("test_appl_kcore", kernel_appl_kcore);
