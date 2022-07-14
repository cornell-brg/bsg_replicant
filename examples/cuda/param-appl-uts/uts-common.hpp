//========================================================================
// uts-common.h
//========================================================================

#ifndef UTS_COMMON_H
#define UTS_COMMON_H

#include <atomic>
#include <deque>
#include <mutex>

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

// Results struct for verification

struct UTSResults {
  std::string impl_str;
  int         numNodes;
  int         maxHeight;
  int         numLeaves;
  UTSResults() : numNodes( 0 ), maxHeight( 0 ), numLeaves( 0 ) {}
  UTSResults( std::string _s, int _n, int _h, int _l )
      : impl_str( _s ), numNodes( _n ), maxHeight( _h ), numLeaves( _l )
  {
  }
};

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
extern int debug;

// Verification parameters

extern bool verify;

extern std::atomic<int> numNodes;
extern std::atomic<int> numLeaves;
extern std::atomic<int> maxHeight;

// Global lock

extern std::mutex g_lock;

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
void genChildren( Node* parent, std::deque<Node*>& taskq );

void genChildren_kernel( void* args_vptr, void* null_ptr );

// Misc functions

float rng_toProb( int n );

// Verify

void verify_results( UTSResults* result, const char* cdataset );

#endif
