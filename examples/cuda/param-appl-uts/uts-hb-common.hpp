//========================================================================
// uts-common.h
//========================================================================

#ifndef UTS_COMMON_H
#define UTS_COMMON_H

#include "brg_sha1.h"

//----------------------------------------------------------------------------
// Node
//----------------------------------------------------------------------------

#define MAXNUMCHILDREN 100 // cap on children (BIN root is exempt)

struct node_t {
  int type;             // distribution governing number of children
  int height;           // depth of this node in the tree
  int numChildren;      // number of children, -1 => not yet determined
  struct state_t state; // RNG state associated with this node
};

typedef struct node_t Node;

/* Tree type
 *   Trees are generated using a Galton-Watson process, in
 *   which the branching factor of each node is a random
 *   variable.
 *
 *   The random variable can follow a binomial distribution
 *   or a geometric distribution.  Hybrid tree are
 *   generated with geometric distributions near the
 *   root and binomial distributions towards the leaves.
 */

enum uts_trees_e { BIN = 0, GEO, HYBRID, BALANCED };
enum uts_geoshape_e { LINEAR = 0, EXPDEC, CYCLIC, FIXED };

typedef enum uts_trees_e    tree_t;
typedef enum uts_geoshape_e geoshape_t;

// Tree parameters

extern tree_t     type;
extern float      b_0;
extern int        rootId;
extern int        nonLeafBF;
extern float      nonLeafProb;
extern int        gen_mx;
extern geoshape_t shape_fn;
extern float      shiftDepth;

// Benchmark parameters

extern int computeGranularity;

// Verification parameters

extern bool verify;

extern int numNodes __attribute__ ((section (".dram")));
extern int numLeaves __attribute__ ((section (".dram")));
extern int maxHeight __attribute__ ((section (".dram")));

//----------------------------------------------------------------------------
// UTS Functions
//----------------------------------------------------------------------------

// Tree functions

void uts_initRoot( Node* root, int type );
void initNode( Node* child );
int  uts_childType( Node* parent );
int  uts_numChildren_bin( Node* parent );
int  uts_numChildren_geo( Node* parent );
int  uts_numChildren( Node* parent );

// Misc functions

float rng_toProb( int n );

#endif
