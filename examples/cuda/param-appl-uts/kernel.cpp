#include <stdint.h>
#include "bsg_manycore.h"
#include "appl.hpp"

#define PLATFORM_BYTE_ORDER IS_LITTLE_ENDIAN

#include "brg_sha1.h"
#include "uts-hb-common.hpp"
#include <cmath>

uint32_t buf_head = 0;
uint32_t buf_tail = 0;
Node*    buf;

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

    // Node all_children[numChildren];
    Node* all_children = (Node*)appl::appl_malloc( numChildren * sizeof(Node) );

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

// applrts and navie baseline
void uts_v3()
{
  Node root;
  uts_initRoot( &root, type );
  uts_v3_kernel( &root );
}

void uts_v4_kernel( Node* parent )
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

    Node* all_children = &(buf[buf_tail]);
    buf_tail += numChildren;

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
  }

  // No children

  else {
    // Track num leaves for verification
    if ( verify )
      bsg_amoadd(&numLeaves, 1);
  }
}

void uts_v4() {
  buf = (Node*)appl::appl_malloc( 1024 * sizeof(Node) );
  Node* root = &(buf[buf_tail++]);
  uts_initRoot( root, type );
  while (buf_tail - buf_head < appl::get_nthreads() && buf_tail != buf_head && buf_tail < 1024) {
    bsg_print_int(12303);
    bsg_print_int(buf_head);
    bsg_print_int(buf_tail);
    uts_v4_kernel( &(buf[buf_head++]) );
  }
  if (buf_tail >= 1024) {
    // buffer overflow
    bsg_print_int(7904);
  }
  // still have work to do
  if (buf_tail != buf_head && buf_tail < 1024) {
    Node* nodes = &(buf[buf_head]);
    appl::parallel_for_1(size_t(0), size_t(buf_tail - buf_head), [nodes](size_t i) {
        uts_v3_kernel( &(nodes[i]) );
    });
  }
}


struct param_t {
  float nonLeafProb;
  int   nonLeafBF;
  int   rootId;
  int   t;
  int   a;
  float b_0;
  int   gen_mx;
  float shiftDepth;
  int   g;
};

extern "C" __attribute__ ((noinline))
int kernel_appl_uts(int* results, int* dram_buffer, int* _param) {

  struct param_t* param = (struct param_t*)(intptr_t)_param;

  type     = (tree_t)param->t;
  shape_fn = (geoshape_t)param->a;
  computeGranularity = param->g;

  b_0         = param->b_0;
  rootId      = param->rootId;
  nonLeafBF   = param->nonLeafBF;
  nonLeafProb = param->nonLeafProb;
  gen_mx      = param->gen_mx;
  shiftDepth  = param->shiftDepth;

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

  // test SHA1
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
    int rand1 = (rng_nextrand( mystate.state ));
    int rand2 = (rng_nextrand( mystate.state ));
    results[0] = rand1;
    results[1] = rand2;
  }

  // sync
  appl::sync();
  bsg_cuda_print_stat_kernel_start();

  if (__bsg_id == 0) {
    // uts
    uts_v4();

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
