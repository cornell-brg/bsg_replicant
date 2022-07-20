//========================================================================
// uts-datasets.h
//========================================================================

#ifndef UTS_DATASETS_H
#define UTS_DATASETS_H

#include <string>

struct Dataset {
  const char* str;
  float       nonLeafProb; // q: Non-leaf probability
  int         nonLeafBF;   // m: Non-leaf branching factor
  int         rootId;      // r: Root ID
  int         t;           // t: Tree type
  int         a;           // a: Shape Function
  float       b_0;         // b: Root branching factor
  int         gen_mx;      // d: Max tree depth
  float       shiftDepth;  // f: Shift depth
  int         g;           // g: Compute granularity
};

struct Ref {
  const char* str;
  int         numNodes;  // size (number of nodes) of the tree
  int         maxHeight; // max height of the tree
  int         numLeaves; // number of leaves in the tree
};

extern Dataset dataset_table[];
extern Ref     ref_table[];

//------------------------------------------------------------------------
// Helper functions
//------------------------------------------------------------------------

std::string mk_dataset_list_str();
Dataset*    choose_dataset( std::string str );
Ref*        choose_ref( std::string str );

#endif
