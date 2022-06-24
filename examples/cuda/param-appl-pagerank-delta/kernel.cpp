#include <stdint.h>
#include "bsg_manycore.h"
#include "appl.hpp"
#include "ligra.h"

float fabs( float x ) {
  if (x > 0) {
    return x;
  } else {
    return -x;
  }
}

template <class vertex>
struct PR_F {
  vertex* V;
  float *Delta, *nghSum;
  int* locks;
  PR_F( vertex* _V, float* _Delta, float* _nghSum, int* _locks )
      : V( _V ), Delta( _Delta ), nghSum( _nghSum ),
        locks( _locks )
  {
  }
  inline bool update( uintE s, uintE d ) const
  {
    float oldVal = nghSum[d];
    nghSum[d] += Delta[s] / V[s].getOutDegree();
    return oldVal == 0;
  }
  inline bool updateAtomic( uintE s, uintE d ) const
  {
    int* lock_ptr = &(locks[d]);
    // lock
    int lock_val = 1;
    do {
      lock_val = bsg_amoswap_aq(lock_ptr, 1);
    } while (lock_val != 0);
    asm volatile("": : :"memory");
    float oldV = nghSum[d];
    nghSum[d] += Delta[s] / V[s].getOutDegree();
    // unlock
    asm volatile("": : :"memory");
    bsg_amoswap_rl(lock_ptr, 0);
    return oldV == 0.0;
  }
  inline bool cond( uintE d ) const { return cond_true( d ); }
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
  inline bool operator()( uintE i ) const
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
  inline bool operator()( uintE i ) const
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
  inline bool operator()( uintE i ) const
  {
    nghSum[i] = 0.0;
    return 1;
  }
};

template <class vertex>
void Compute( graph<vertex>& GA, uint32_t maxIters, float* results )
{
  const uintE n        = GA.n;
  const float damping  = 0.85;
  const float epsilon  = 0.0000001;
  const float epsilon2 = 0.01;

  float  one_over_n = 1 / (float)n;
  float*          p = newA( float, n );
  float*      Delta = newA( float, n );
  float*     nghSum = newA( float, n );
  bool*    frontier = newA( bool,  n );
  bool*         all = newA( bool,  n );
  int*        locks = newA( int,   n );

  appl::parallel_for( uintE( 0 ), n, [&]( uintE i ) {
    p[i]        = 0.0;        // one_over_n;
    Delta[i]    = one_over_n; // initial delta propagation from each vertex
    nghSum[i]   = 0.0;
    frontier[i] = 1;
    all[i]      = 1;
    locks       = 0;
  } );

  vertexSubset Frontier( n, n, frontier );
  vertexSubset All( n, n, all ); // all vertices

  uint32_t round = 0;
  while ( round++ < maxIters ) {
    edgeMap( GA, Frontier, PR_F<vertex>( GA.V, Delta, nghSum, locks ), 0,
             no_output );
    vertexSubset active =
        ( round == 1 )
            ? vertexFilter(
                  All, PR_Vertex_F_FirstRound( p, Delta, nghSum, damping,
                                               one_over_n, epsilon2 ) )
            : vertexFilter( All, PR_Vertex_F( p, Delta, nghSum, damping,
                                              epsilon2 ) );
    // compute L1-norm (use nghSum as temp array)
    {
      appl::parallel_for(
          uintE( 0 ), n, [&]( uintE i ) { nghSum[i] = fabs( Delta[i] ); } );
    }

    // float L1_norm = sequence::plusReduce( nghSum, n );
    float L1_norm = appl::parallel_reduce(uintE(0), n, 0.0f,
        [&](uintE start, uintE end, float initV) {
          float psum = initV;
          for (uintE i = start; i < end; i++) {
            psum += nghSum[i];
          }
          return psum;
        },
        [](float x, float y) { return x + y; }
    );

    if ( L1_norm < epsilon )
      break;
    // reset
    vertexMap( All, PR_Vertex_Reset( nghSum ) );
    Frontier.del();
    Frontier = active;
  }

  // dump
  for (size_t i = 0; i < n; i++) {
    bsg_print_float(p[i]);
    results[i] = p[i];
  }

  Frontier.del();
  All.del();
}

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
