//========================================================================
// uts-common.cc
//========================================================================
// Defines global variables and common functions used by all
// implementations

#include <cmath>

#include "uts-hb-common.hpp"
#include "uts-datasets.hpp"

#include "bsg_manycore.h"

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

// Verification parameters

bool verify;

int numNodes __attribute__ ((section (".dram")))  = 0;
int numLeaves __attribute__ ((section (".dram"))) = 0;
int maxHeight __attribute__ ((section (".dram"))) = 0;

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
    bsg_print_int(7900);
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
    bsg_print_int(7901);
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
    bsg_print_int(7902);
  }

  // limit number of children
  // only a BIN root can have more than MAXNUMCHILDREN
  if ( parent->height == 0 && parent->type == BIN ) {
    int rootBF = (int)std::ceil( b_0 );
    if ( numChildren > rootBF ) {
      bsg_print_int(7903);
      numChildren = rootBF;
    }
  }
  else if ( type != BALANCED ) {
    if ( numChildren > MAXNUMCHILDREN ) {
      bsg_print_int(7904);
      numChildren = MAXNUMCHILDREN;
    }
  }

  return numChildren;
}
