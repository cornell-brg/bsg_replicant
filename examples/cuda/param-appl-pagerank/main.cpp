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
#include "math.h"

#define ALLOC_NAME "default_allocator"
#define MAX_WORKERS 128
#define HB_L2_CACHE_LINE_WORDS 16
#define BUF_FACTOR 16385
#define BUF_SIZE (MAX_WORKERS * HB_L2_CACHE_LINE_WORDS * BUF_FACTOR)

template <class vertex>
struct PR_F {
  float *p_curr, *p_next;
  vertex* V;
  PR_F( float* _p_curr, float* _p_next, vertex* _V )
      : p_curr( _p_curr ), p_next( _p_next ), V( _V )
  {
  }
  inline bool update( uintE s, uintE d )
  { // update function applies PageRank equation
    p_next[d] += p_curr[s] / V[s].getOutDegree();
    return 1;
  }
  inline bool updateAtomic( uintE s, uintE d )
  { // atomic Update
    writeAdd( &p_next[d], p_curr[s] / V[s].getOutDegree() );
    return 1;
  }
  inline bool cond( intT d ) { return cond_true( d ); }
};

// vertex map function to update its p value according to PageRank
// equation
struct PR_Vertex_F {
  float  damping;
  float  addedConstant;
  float* p_curr;
  float* p_next;
  PR_Vertex_F( float* _p_curr, float* _p_next, float _damping, intE n )
      : p_curr( _p_curr ), p_next( _p_next ), damping( _damping ),
        addedConstant( ( 1 - _damping ) * ( 1 / (float)n ) )
  {
  }
  inline bool operator()( uintE i )
  {
    p_next[i] = damping * p_next[i] + addedConstant;
    return 1;
  }
};

// resets p
struct PR_Vertex_Reset {
  float* p_curr;
  PR_Vertex_Reset( float* _p_curr ) : p_curr( _p_curr ) {}
  inline bool operator()( uintE i )
  {
    p_curr[i] = 0.0;
    return 1;
  }
};

int kernel_appl_pagerank (int argc, char **argv) {
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

        bsg_pr_test_info("Running the Ligra PageRank on one %dx%d tile groups.\n\n", bsg_tiles_X, bsg_tiles_Y);

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
                 * Run PageRank natively
                 ******************************************************************************************************************/

                uint32_t maxIters   = MAX_ITER;
                const intE n        = G.n;
                const float damping = 0.85;
                const float epsilon = 0.0000001;

                float  one_over_n = 1 / (float)n;
                float* p_curr     = newA( float, n );
                appl::parallel_for( intE( 0 ), n,
                                    [&]( intE i ) { p_curr[i] = one_over_n; } );
                float* p_next = newA( float, n );
                appl::parallel_for( intE( 0 ), n, [&]( intE i ) { p_next[i] = 0; } );
                bool* frontier = newA( bool, n );
                appl::parallel_for( intE( 0 ), n, [&]( intE i ) { frontier[i] = 1; } );

                vertexSubset Frontier( n, n, frontier );

                bsg_pr_info("Starting Ligra PageRank with max iter %d.\n\n", maxIters);

                uint32_t iter = 0;
                while ( iter++ < maxIters ) {
                  edgeMap( G, Frontier, PR_F<symmetricVertex>( p_curr, p_next, G.V ), 0,
                           no_output );
                  vertexMap( Frontier, PR_Vertex_F( p_curr, p_next, damping, n ) );
                  // compute L1-norm between p_curr and p_next
                  {
                    appl::parallel_for( intE( 0 ), n, [&]( intE i ) {
                      p_curr[i] = fabs( p_curr[i] - p_next[i] );
                    } );
                  }
                  bsg_pr_info("iter %d\n\n", iter);
                  for (size_t i = 0; i < n; i++) {
                    bsg_pr_info("PageRank at vertex %d = %f\n", i, p_next[i]);
                  }
                  float L1_norm = sequence::plusReduce( p_curr, n );
                  if ( L1_norm < epsilon )
                    break;
                  // reset p_curr
                  vertexMap( Frontier, PR_Vertex_Reset( p_curr ) );
                  swap( p_curr, p_next );
                }

                Frontier.del();

                for (size_t i = 0; i < n; i++) {
                  bsg_pr_info("PageRank at vertex %d = %f\n", i, p_curr[i]);
                }

                bsg_pr_info("Starting kernel...\n\n");
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
                const uint32_t cuda_argv[6] = {device_result, G.hb_V, G.n, G.m, maxIters, dram_buffer};

                /*****************************************************************************************************************
                 * Enquque grid of tile groups, pass in grid and tile group dimensions, kernel name, number and list of input arguments
                 ******************************************************************************************************************/
                BSG_CUDA_CALL(hb_mc_kernel_enqueue (&device, grid_dim, tg_dim, "kernel_appl_pagerank", 6, cuda_argv));

                /*****************************************************************************************************************
                 * Launch and execute all tile groups on device and wait for all to finish.
                 ******************************************************************************************************************/
                BSG_CUDA_CALL(hb_mc_device_tile_groups_execute(&device));

                /*****************************************************************************************************************
                 * Copy result back from device DRAM into host memory.
                 ******************************************************************************************************************/
                float host_result[G.n];
                hb_mc_dma_dtoh_t dtoh = {
                  .d_addr = device_result,
                  .h_addr = (&host_result[0]),
                  .size   = G.n * sizeof(float)
                };
                BSG_CUDA_CALL(hb_mc_device_dma_to_host(&device, &dtoh, 1));


                /*****************************************************************************************************************
                 * Freeze the tiles and memory manager cleanup.
                 ******************************************************************************************************************/
                BSG_CUDA_CALL(hb_mc_device_program_finish(&device));

                double error = 0.0;
                for (size_t i = 0; i < G.n; i++) {
                  bsg_pr_info("PageRank at vertex %d = %f : %f\n", i, host_result[i], p_curr[i]);
                  error += fabs(host_result[i] - p_curr[i]);
                }
                if (error > 0.0001) {
                  return HB_MC_FAIL;
                }

        }
        BSG_CUDA_CALL(hb_mc_device_finish(&device));

        return HB_MC_SUCCESS;
}

declare_program_main("test_appl_pagerank", kernel_appl_pagerank);
