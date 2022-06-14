#include <stdint.h>
#include "bsg_manycore.h"
#include "appl.hpp"
#include "ligra.h"

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

extern "C" __attribute__ ((noinline))
int kernel_appl_pagerank_delta(int* results, symmetricVertex* V, int n, int m, uint32_t maxIters, int* dram_buffer) {

  appl::runtime_init(dram_buffer);
  appl::sync();
  bsg_cuda_print_stat_kernel_start();

  if (__bsg_id == 0) {
    graph<symmetricVertex> G = graph<symmetricVertex>(V, n, m, nullptr);
    Compute(G, maxIters, (float*)(intptr_t)results);
  } else {
    appl::worker_thread_init();
  }

  appl::runtime_end();
  bsg_cuda_print_stat_kernel_end();
  appl::sync();

  return 0;
}
