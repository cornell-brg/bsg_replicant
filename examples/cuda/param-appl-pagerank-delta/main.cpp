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
  vertex* V;
  float *Delta, *nghSum;
  PR_F( vertex* _V, float* _Delta, float* _nghSum )
      : V( _V ), Delta( _Delta ), nghSum( _nghSum )
  {
  }
  inline bool update( uintE s, uintE d )
  {
    float oldVal = nghSum[d];
    nghSum[d] += Delta[s] / V[s].getOutDegree();
    return oldVal == 0;
  }
  inline bool updateAtomic( uintE s, uintE d )
  {
    volatile float oldV, newV;
    do { // basically a fetch-and-add
      oldV = nghSum[d];
      newV = oldV + Delta[s] / V[s].getOutDegree();
    } while ( !CAS( &nghSum[d], oldV, newV ) );
    return oldV == 0.0;
  }
  inline bool cond( uintE d ) { return cond_true( d ); }
};

struct PR_Vertex_F_FirstRound {
  float  damping, addedConstant, one_over_n, epsilon2;
  float *p, *Delta, *nghSum;
  PR_Vertex_F_FirstRound( float* _p, float* _Delta, float* _nghSum,
                          float _damping, float _one_over_n,
                          float _epsilon2 )
      : p( _p ), damping( _damping ), Delta( _Delta ), nghSum( _nghSum ),
        one_over_n( _one_over_n ),
        addedConstant( ( 1 - _damping ) * _one_over_n ),
        epsilon2( _epsilon2 )
  {
  }
  inline bool operator()( uintE i )
  {
    Delta[i] = damping * nghSum[i] + addedConstant;
    p[i] += Delta[i];
    Delta[i] -= one_over_n; // subtract off delta from initialization
    return ( fabs( Delta[i] ) > epsilon2 * p[i] );
  }
};

struct PR_Vertex_F {
  float  damping, epsilon2;
  float *p, *Delta, *nghSum;
  PR_Vertex_F( float* _p, float* _Delta, float* _nghSum,
               float _damping, float _epsilon2 )
      : p( _p ), damping( _damping ), Delta( _Delta ), nghSum( _nghSum ),
        epsilon2( _epsilon2 )
  {
  }
  inline bool operator()( uintE i )
  {
    Delta[i] = nghSum[i] * damping;
    if ( fabs( Delta[i] ) > epsilon2 * p[i] ) {
      p[i] += Delta[i];
      return 1;
    }
    else
      return 0;
  }
};

struct PR_Vertex_Reset {
  float* nghSum;
  PR_Vertex_Reset( float* _nghSum ) : nghSum( _nghSum ) {}
  inline bool operator()( uintE i )
  {
    nghSum[i] = 0.0;
    return 1;
  }
};

int kernel_appl_pagerank_delta (int argc, char **argv) {
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

        bsg_pr_test_info("Running the Ligra PageRankDelta on one %dx%d tile groups.\n\n", bsg_tiles_X, bsg_tiles_Y);

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
                 * Run PageRankDelta natively
                 ******************************************************************************************************************/

                uint32_t maxIters    = MAX_ITER;
                const uintE n       = G.n;
                const float damping  = 0.85;
                const float epsilon  = 0.0000001;
                const float epsilon2 = 0.01;

                float  one_over_n = 1 / (float)n;
                float*          p = newA( float, n );
                float*      Delta = newA( float, n );
                float*     nghSum = newA( float, n );
                bool*    frontier = newA( bool,  n );
                bool*        all  = newA( bool, n );

                appl::parallel_for( uintE( 0 ), n, [&]( uintE i ) {
                    p[i]        = 0.0;
                    Delta[i]    = one_over_n;
                    nghSum[i]   = 0.0;
                    frontier[i] = 1;
                    all[i]      = 1;
                } );

                vertexSubset Frontier( n, n, frontier );
                vertexSubset All( n, n, all );

                bsg_pr_info("Starting Ligra PageRankDelta with max iter %d.\n\n", maxIters);

                uint32_t iter = 0;
                while ( iter++ < maxIters ) {
                  edgeMap( G, Frontier, PR_F<symmetricVertex>( G.V, Delta, nghSum ), 0,
                           no_output | dense_forward );
                  vertexSubset active =
                    ( iter == 1 )
                      ? vertexFilter(
                          All, PR_Vertex_F_FirstRound( p, Delta, nghSum, damping,
                                                       one_over_n, epsilon2 ) )
                      : vertexFilter( All, PR_Vertex_F( p, Delta, nghSum, damping,
                                                        epsilon2 ) );
                  // compute L1-norm (use nghSum as temp array)
                  {
                    appl::parallel_for( uintE( 0 ), n, [&]( uintE i ) {
                        nghSum[i] = fabs( Delta[i] );
                    } );
                  }
                  bsg_pr_info("iter %d\n\n", iter);
                  for (size_t i = 0; i < n; i++) {
                    bsg_pr_info("PageRank at vertex %d = %f\n", i, p[i]);
                  }
                  float L1_norm = sequence::plusReduce( nghSum, n );
                  if ( L1_norm < epsilon )
                    break;
                  // reset
                  vertexMap( All, PR_Vertex_Reset( nghSum ) );
                  Frontier.del();
                  Frontier = active;
                }

                Frontier.del();

                for (size_t i = 0; i < n; i++) {
                  bsg_pr_info("PageRank at vertex %d = %f\n", i, p[i]);
                }

                exit(1);
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
                BSG_CUDA_CALL(hb_mc_kernel_enqueue (&device, grid_dim, tg_dim, "kernel_appl_pagerank_delta", 6, cuda_argv));

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
                  bsg_pr_info("PageRank at vertex %d = %f : %f\n", i, host_result[i], p[i]);
                  error += fabs(host_result[i] - p[i]);
                }
                if (error > 0.0001) {
                  return HB_MC_FAIL;
                }

        }
        BSG_CUDA_CALL(hb_mc_device_finish(&device));

        return HB_MC_SUCCESS;
}

declare_program_main("test_appl_pagerank_delta", kernel_appl_pagerank_delta);
