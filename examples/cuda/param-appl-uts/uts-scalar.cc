//========================================================================
// uts-scalar
//========================================================================
// We can use the optimize function attribute to force the compiler to
// unroll this loop. We get better code if we use the special
// __restrict__ keyword so that the compiler knows the arrays don't
// overlap, and if we use pointer bumps as opposed to array indexing.

#include "uts-scalar.h"
#include "uts-common.h"

#include <cmath>
#include <cstring>
#include <deque>
#include <iostream>

__attribute__( ( noinline, optimize( "unroll-loops" ) ) ) void
uts_scalar()
{
  // Create and init root node
  // Allocate on the heap so we don't have to worry about scoping

  Node* root = new Node();
  uts_initRoot( root, type );

  // Push node onto the queue

  std::deque<Node*> taskq;
  taskq.push_front( root );

  // Run search using task queue to store node pointers. Each iteration
  // pops a node from the queue, processes it (generates its children),
  // and then pushes each child onto the front of the queue.
  // The search is over when the queue is empty.

  while ( !taskq.empty() ) {

    if ( debug > 1 )
      std::cout << "taskq size: " << taskq.size() << std::endl;

    // Get parent

    Node* parent = taskq.front();
    taskq.pop_front();

    // Track num nodes for verification

    if ( verify )
      numNodes++;

    // Generate children

    genChildren( parent, taskq );

    // Free parent

    delete ( parent );
  }
}
