#include <stdint.h>
#include "bsg_manycore.h"
#include "appl.hpp"

#define PLATFORM_BYTE_ORDER IS_LITTLE_ENDIAN

#include "brg_sha1.h"
#include "uts-hb-common.hpp"
#include <cmath>

template <class ET>
ET writeMax(ET* p, ET val) {
  ET result;
  asm volatile ("amomax.w %[result], %[val], 0(%[p])" \
                : [result] "=r" (result) \
                : [p] "r" (p), [val] "r" (val));
  return result;
}

void uts_v3_kernel( Node* parent )
{
  if ( verify )
    bsg_amoadd(&numNodes, 1);

  // Calculate how many children this node should have

  int numChildren, childType;

  numChildren = uts_numChildren( parent );
  childType   = uts_childType( parent );

  // Record number of children in parent

  parent->numChildren = numChildren;
  bsg_print_int(numChildren);

  // Construct children and push them onto stack

  int parentHeight = parent->height;

  if ( numChildren > 0 ) {

    // Give a SHA-1 hash to each child

    // Define all nodes, args, tasks on the stack to avoid dynamic
    // memory management complexity. Need to put all definitions up here
    // so that they stay in scope. This function is parallelized with
    // run() and run_and_wait(). After run_and_wait() finishes, these
    // definitions go out of scope and are automatically cleaned up.

    Node all_children[numChildren];

    // Node creation

    for ( int i = 0; i < numChildren; i++ ) {
      bsg_print_int(i);
      Node* child = &all_children[i];
      initNode( child );
      child->height = parentHeight + 1;
      child->type   = childType;

      for ( int j = 0; j < computeGranularity; j++ ) {
        // computeGranularity controls number of rng_spawn calls per node
        rng_spawn( parent->state.state, child->state.state, i );
      }

      // Track tree height for verification

      if ( verify ) {
        writeMax(&maxHeight, child->height);
      }
    }

    appl::parallel_for( 0, numChildren, [&]( int i ) {
        bsg_print_int(10088);
        uts_v3_kernel( &all_children[i] );
      } );
  }

  // No children

  else {
    // Track num leaves for verification
    if ( verify )
      bsg_amoadd(&numLeaves, 1);
  }
}

void uts_v3()
{
  Node root;
  uts_initRoot( &root, type );
  uts_v3_kernel( &root );
}
extern "C" __attribute__ ((noinline))
//int kernel_appl_uts(int* results, float _nonLeafProb, int _nonLeafBF, int _rootId,
//                    int _t, int _a, float _b_0, int _gen_mx, float _shiftDepth,
//                    int _g, int* dram_buffer) {
int kernel_appl_uts(int* results, int* dram_buffer) {

  type     = (tree_t)1;
  shape_fn = (geoshape_t)3;
  computeGranularity = 1;

  b_0         = 4.0;// _b_0;
  rootId      = 19;// _rootId;
  nonLeafBF   = 4;// _nonLeafBF;
  nonLeafProb = 0.234375;// _nonLeafProb;
  gen_mx      = 3;// _gen_mx;
  shiftDepth  = 0.5;// _shiftDepth;

  verify = true;

  // debug print
  if (__bsg_id == 0) {
    bsg_print_float(nonLeafProb);
    bsg_print_int(nonLeafBF);
    bsg_print_int(rootId);
    bsg_print_int((int)type);
    bsg_print_int((int)shape_fn);
    bsg_print_float(b_0);
    bsg_print_int(gen_mx);
    bsg_print_float(shiftDepth);
    bsg_print_int(computeGranularity);
  }

  // --------------------- kernel ------------------------
  appl::runtime_init(dram_buffer);

  // sync
  appl::sync();
  bsg_cuda_print_stat_kernel_start();

  if (__bsg_id == 0) {
    // sha1 test
    struct state_t mystate;
    for (int i = 0; i < 20; i++) {
      mystate.state[i] = i;
    }
    rng_init( mystate.state, 14850 );
    for (int i = 0; i < 20; i++) {
      bsg_print_int(mystate.state[i]);
    }
    bsg_print_int(14850);
    int rand1 = (rng_nextrand( mystate.state ));
    int rand2 = (rng_nextrand( mystate.state ));
    bsg_print_int(rand1);
    bsg_print_int(rand2);
    results[0] = rand1;
    results[1] = rand2;

    // uts
    uts_v3();

    results[2] = bsg_amoor(&numNodes, 0);
    results[3] = bsg_amoor(&numLeaves, 0);
    results[4] = bsg_amoor(&maxHeight, 0);

    bsg_print_int(14853);
    bsg_print_int(results[2]);
    bsg_print_int(results[3]);
    bsg_print_int(results[4]);
  } else {
    appl::worker_thread_init();
  }
  appl::runtime_end();
  // --------------------- end of kernel -----------------

  bsg_cuda_print_stat_kernel_end();

  appl::sync();
  return 0;
}
