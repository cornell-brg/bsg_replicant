//========================================================================
// uts-datasets.cc
//========================================================================

#include <cstdlib>
#include <cstring>
#include <iostream>

#include "uts-datasets.hpp"

//------------------------------------------------------------------------
// Dataset Table
//------------------------------------------------------------------------
// This table contains the values used for each of the various datasets.
// The following datasets are small workloads of ~4 million nodes from
// sample_trees.sh in the UTS distribution code (but would take half a
// month to run on gem5):
//
// (T1) Geometric [fixed]
// Tree size = 4130071, tree depth = 10, num leaves = 3305118 (80.03%)
//
// (T2) Geometric [cyclic]
// Tree size = 4117769, tree depth = 81, num leaves = 2342762 (56.89%)
//
// (T3) Binomial
// Tree size = 4112897, tree depth = 1572, num leaves = 3599034 (87.51%)
//
// (T4) Hybrid
// Tree size = 4132453, tree depth = 134, num leaves = 3108986 (75.23%)
//
// (T5) Geometric [linear dec.]
// Tree size = 4147582, tree depth = 20, num leaves = 2181318 (52.59%)
//
// We create the following datasets by modulating the tree depth:
//
// - test     : identical to test-t1
// - test-t1  : decrease depth of T1 from 16 to 3
// - tiny-t1  : decrease depth of T1 from 16 to 4
// - small-t1 : decrease depth of T1 from 16 to 5
// - test-t2  : decrease depth of T2 from 16 to 2
// - tiny-t2  : decrease depth of T2 from 16 to 3
// - small-t2 : decrease depth of T2 from 16 to 6
// - test-t3  : decrease root branching factor of T3 from 2000 to 10
// - tiny-t3  : decrease root branching factor of T3 from 2000 to 50
// - small-t3 : decrease root branching factor of T3 from 2000 to 200
// - test-t4  : decrease m to 0.20 and d to 4
// - tiny-t4  : decrease m to 0.23 and d to 4
// - small-t4 : decrease m to 0.23 and d to 5
// - test-t5  : decrease depth of T5 from 20 to 6
// - tiny-t5  : decrease depth of T5 from 20 to 7
// - small-t5 : decrease depth of T5 from 20 to 10
//
//
// Each dataset has a corresponding ref used in verify_results().
//
// To add a new dataset, add an entry to both dataset and ref tables.

// Dataset Table
// Value of -1 means assign default

Dataset dataset_table[] =
{
  // name          q     m     r     t     a     b     d     f     g
  { "test",       -1,   -1,   19,    1,    3,    4,    3,   -1,   -1 },
  { "test-t1",    -1,   -1,   19,    1,    3,    4,    3,   -1,   -1 },
  { "tiny-t1",    -1,   -1,   19,    1,    3,    4,    4,   -1,   -1 },
  { "small-t1",   -1,   -1,   19,    1,    3,    4,    5,   -1,   -1 },
  { "test-t2",    -1,   -1,  502,    1,    2,    6,    2,   -1,   -1 },
  { "tiny-t2",    -1,   -1,  502,    1,    2,    6,    3,   -1,   -1 },
  { "small-t2",   -1,   -1,  502,    1,    2,    6,    6,   -1,   -1 },
  { "test-t3",  0.12,    8,   42,    0,   -1,   10,   -1,   -1,   -1 },
  { "tiny-t3",  0.12,    8,   42,    0,   -1,   50,   -1,   -1,   -1 },
  { "small-t3", 0.12,    8,   42,    0,   -1,  200,   -1,   -1,   -1 },
  { "test-t4",  0.20,    4,    1,    2,    0,    6,    4,   -1,   -1 },
  { "tiny-t4",  0.23,    4,    1,    2,    0,    6,    4,   -1,   -1 },
  { "small-t4", 0.23,    4,    1,    2,    0,    6,    5,   -1,   -1 },
  { "medium-t4",0.25,    4,    1,    2,    0,    6,    6,   -1,   -1 }, // TODO: add to Ref Table
  { "test-t5",    -1,   -1,   34,    1,    0,    4,    6,   -1,   -1 },
  { "tiny-t5",    -1,   -1,   34,    1,    0,    4,    7,   -1,   -1 },
  { "small-t5",   -1,   -1,   34,    1,    0,    4,   10,   -1,   -1 },
  { "",            0,    0,    0,    0,    0,    0,    0,    0,    0 },
};

//------------------------------------------------------------------------
// Ref Table
//------------------------------------------------------------------------

Ref ref_table[] = {
    // name       numNodes  maxHeight  numLeaves
    {"test", 254, 3, 201},
    {"test-t1", 254, 3, 201},
    {"tiny-t1", 944, 4, 744},
    {"small-t1", 3987, 5, 3232},
    {"test-t2", 227, 11, 124},
    {"tiny-t2", 996, 16, 705},
    {"small-t2", 4988, 31, 3086},
    {"test-t3", 371, 15, 325},
    {"tiny-t3", 723, 17, 638},
    {"small-t3", 4649, 62, 4092},
    {"test-t4", 374, 16, 286},
    {"tiny-t4", 910, 22, 688},
    {"small-t4", 4699, 72, 3530},
    {"test-t5", 320, 6, 192},
    {"tiny-t5", 736, 7, 434},
    {"small-t5", 5577, 10, 3111},
    {"", 0, 0, 0},
};

//------------------------------------------------------------------------
// Helper functions
//------------------------------------------------------------------------

std::string mk_dataset_list_str()
{
  std::string dataset_list_str;
  Dataset*    dataset_ptr = &dataset_table[0];
  dataset_list_str += "{";
  dataset_list_str += dataset_ptr->str;
  dataset_ptr++;
  while ( strlen( dataset_ptr->str ) != 0 ) {
    dataset_list_str += ",";
    dataset_list_str += dataset_ptr->str;
    dataset_ptr++;
  }
  dataset_list_str += "}";
  return dataset_list_str;
}

Dataset* choose_dataset( std::string str )
{
  Dataset* dataset_ptr = &dataset_table[0];
  bool     found       = false;
  while ( strlen( dataset_ptr->str ) != 0 ) {
    if ( strcmp( str.c_str(), dataset_ptr->str ) == 0 ) {
      found = true;
      break;
    }
    dataset_ptr++;
  }

  if ( !found ) {
    std::cout << "\n ERROR: dataset \"" << str << "\" is not valid"
              << std::endl;
    exit( 1 );
  }

  return dataset_ptr;
}

Ref* choose_ref( std::string str )
{
  Ref* ref_ptr = &ref_table[0];
  bool found   = false;
  while ( strlen( ref_ptr->str ) != 0 ) {
    if ( strcmp( str.c_str(), ref_ptr->str ) == 0 ) {
      found = true;
      break;
    }
    ref_ptr++;
  }

  if ( !found ) {
    std::cout << "\n ERROR: ref \"" << str << "\" is not valid"
              << std::endl;
    exit( 1 );
  }

  return ref_ptr;
}
