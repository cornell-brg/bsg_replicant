//========================================================================
// uts-common.cc
//========================================================================
// Defines global variables and common functions used by all
// implementations

#include <atomic>
#include <cmath>
#include <cstring>
#include <deque>
#include <iostream>
#include <mutex>
#include <string>

#include "uts-common.hpp"
#include "uts-datasets.hpp"

//------------------------------------------------------------------------
// UTS Global Variables
//------------------------------------------------------------------------

// Tree parameters

tree_t     type;
float      b_0;
int        rootId;
int        nonLeafBF;
float      nonLeafProb;
int        gen_mx;
geoshape_t shape_fn;
float      shiftDepth;

// Benchmark parameters

int computeGranularity;
int debug;

// Verification parameters

bool verify;

std::atomic<int> numNodes;
std::atomic<int> maxHeight;
std::atomic<int> numLeaves;

// Global lock

std::mutex g_lock;

//------------------------------------------------------------------------
// UTS Functions
//------------------------------------------------------------------------

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
// rng_toProb
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
// Interpret 32 bit positive integer as value on [0,1)

float rng_toProb( int n )
{
  if ( n < 0 ) {
    printf( "*** toProb: rand n = %d out of range\n", n );
  }
  return ( ( n < 0 ) ? 0.0 : ( (float)n ) / 2147483648.0 );
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
// uts_initRoot
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
// Initialize type, height, numChildren, state of root

void uts_initRoot( Node* root, int type )
{
  root->type        = type;
  root->height      = 0;
  root->numChildren = -1; // means not yet determined
  rng_init( root->state.state, rootId );

  if ( debug > 1 )
    printf( "root node of type %d at %p\n", type, root );
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
// initNode
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
// Initialize type, height, numChildren

void initNode( Node* child )
{
  child->type        = -1;
  child->height      = -1;
  child->numChildren = -1; // not yet determined
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
// uts_childType
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
// Return child type (this is an enum)

int uts_childType( Node* parent )
{
  switch ( type ) {
  case BIN:
    return BIN;
  case GEO:
    return GEO;
  case HYBRID:
    if ( parent->height < shiftDepth * gen_mx )
      return GEO;
    else
      return BIN;
  case BALANCED:
    return BALANCED;
  default:
    std::cerr << "uts_childType(): Unknown tree type" << std::endl;
    return -1;
  }
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
// uts_numChildren_bin
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
// Use binomial distribution to determine how many children a node has

int uts_numChildren_bin( Node* parent )
{
  // Distribution is identical everywhere below root
  int   v = rng_rand( parent->state.state );
  float d = rng_toProb( v );

  return ( d < nonLeafProb ) ? nonLeafBF : 0;
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
// uts_numChildren_geo
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
// Use geometric distribution to determine how many children a node has

int uts_numChildren_geo( Node* parent )
{
  float b_i   = b_0;
  int   depth = parent->height;
  int   numChildren, h;
  float p, u;

  // use shape function to compute target b_i
  if ( depth > 0 ) {
    switch ( shape_fn ) {

    // expected size polynomial in depth
    case EXPDEC:
      b_i = b_0 *
            std::pow( (float)depth, -log( b_0 ) / log( (float)gen_mx ) );
      break;

    // cyclic tree size
    case CYCLIC:
      if ( depth > 5 * gen_mx ) {
        b_i = 0.0;
        break;
      }
      b_i = std::pow( b_0, std::sin( 2.0 * 3.141592653589793 *
                                     (float)depth / (float)gen_mx ) );
      break;

    // identical distribution at all nodes up to max depth
    case FIXED:
      b_i = ( depth < gen_mx ) ? b_0 : 0;
      break;

    // linear decrease in b_i
    case LINEAR:
    default:
      b_i = b_0 * ( 1.0 - (float)depth / (float)gen_mx );
      break;
    }
  }

  // given target b_i, find prob p so expected value of
  // geometric distribution is b_i.
  p = 1.0 / ( 1.0 + b_i );

  // get uniform random number on [0,1)
  h = rng_rand( parent->state.state );
  u = rng_toProb( h );

  // max number of children at this cumulative probability
  // (from inverse geometric cumulative density function)
  numChildren = (int)std::floor( log( 1 - u ) / log( 1 - p ) );

  return numChildren;
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
// uts_numChildren
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
// Wrapper around uts_numChildren_geo and uts_numChildren_bin

int uts_numChildren( Node* parent )
{
  int numChildren = 0;

  switch ( type ) {
  case BIN:
    if ( parent->height == 0 )
      numChildren = (int)std::floor( b_0 );
    else
      numChildren = uts_numChildren_bin( parent );
    break;

  case GEO:
    numChildren = uts_numChildren_geo( parent );
    break;

  case HYBRID:
    if ( parent->height < shiftDepth * gen_mx )
      numChildren = uts_numChildren_geo( parent );
    else
      numChildren = uts_numChildren_bin( parent );
    break;
  case BALANCED:
    if ( parent->height < gen_mx )
      numChildren = (int)b_0;
    break;
  default:
    std::cerr << "parTreeSearch(): Unknown tree type" << std::endl;
  }

  // limit number of children
  // only a BIN root can have more than MAXNUMCHILDREN
  if ( parent->height == 0 && parent->type == BIN ) {
    int rootBF = (int)std::ceil( b_0 );
    if ( numChildren > rootBF ) {
      printf( "*** Number of children of root truncated from %d to %d\n",
              numChildren, rootBF );
      numChildren = rootBF;
    }
  }
  else if ( type != BALANCED ) {
    if ( numChildren > MAXNUMCHILDREN ) {
      printf( "*** Number of children truncated from %d to %d\n",
              numChildren, MAXNUMCHILDREN );
      numChildren = MAXNUMCHILDREN;
    }
  }

  return numChildren;
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
// genChildren
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
// Generate all children for the given node and push them on the task
// queue

void genChildren( Node* parent, std::deque<Node*>& taskq )
{

  // Calculate how many children this node should have

  int numChildren, childType;

  numChildren = uts_numChildren( parent );
  childType   = uts_childType( parent );

  // Record number of children in parent

  parent->numChildren = numChildren;

  // Construct children and push them onto stack

  int parentHeight = parent->height;

  if ( numChildren > 0 ) {

    // Give a SHA-1 hash to each child

    for ( int i = 0; i < numChildren; i++ ) {
      Node* child = new Node();
      initNode( child );
      child->height = parentHeight + 1;
      child->type   = childType;

      for ( int j = 0; j < computeGranularity; j++ ) {
        // computeGranularity controls number of rng_spawn calls per node
        rng_spawn( parent->state.state, child->state.state, i );
      }

      // Track tree height for verification

      if ( verify )
        maxHeight = std::max( maxHeight.load(), child->height );

      // Push child onto task queue

      taskq.push_front( child );
    }
  }

  // No children

  else {
    // Track num leaves for verification
    if ( verify )
      numLeaves++;
  }
}

//------------------------------------------------------------------------
// verify_results
//------------------------------------------------------------------------

void verify_results( UTSResults* result, const char* cdataset )
{
  numNodes  = result->numNodes;
  maxHeight = result->maxHeight;
  numLeaves = result->numLeaves;

  std::string name( result->impl_str );
  std::string dataset( cdataset );

  float leafPercent = numLeaves / (float)numNodes * 100.0;

  std::cout << "Verify : " << std::endl;
  std::cout << "Nodes  = " << numNodes << std::endl;
  std::cout << "Height = " << maxHeight << std::endl;
  std::cout << "Leaves = " << numLeaves << std::endl;

  std::cout.precision( 2 );
  std::cout << std::fixed;
  std::cout << "Leaf % = " << leafPercent << std::endl;

  // Get reference data

  Ref* ref_ptr = choose_ref( dataset );

  // Verify the dataset

  if ( numNodes != ref_ptr->numNodes || maxHeight != ref_ptr->maxHeight ||
       numLeaves != ref_ptr->numLeaves ) {
    std::cout << "  [ FAILED ] " << name << ", " << dataset << " : "
              << "(numNodes, maxHeight, numLeaves)"
              << " was ( " << numNodes << ", " << maxHeight << ", "
              << numLeaves << " ), expected ( " << ref_ptr->numNodes
              << ", " << ref_ptr->maxHeight << ", " << ref_ptr->numLeaves
              << " )" << std::endl;
    return;
  }

  std::cout << "  [ passed ] " << name << ", " << dataset << std::endl;
}
