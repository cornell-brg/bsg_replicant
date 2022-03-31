#ifndef GRAPH_H
#define GRAPH_H
#include "parallel.h"
#include "vertex.h"
#include <fstream>
#include <iostream>
#include <stdlib.h>
using namespace std;

// **************************************************************
//    ADJACENCY ARRAY REPRESENTATION
// **************************************************************

// Class that handles implementation specific freeing of memory
// owned by the graph
struct Deletable {
public:
  virtual void del() = 0;
};

template <class vertex>
struct Uncompressed_Mem : public Deletable {
public:
  vertex* V;
  int    n;
  int    m;
  void *  allocatedInplace, *inEdges;

  Uncompressed_Mem( vertex* VV, int nn, int mm, void* ai,
                    void* _inEdges = NULL )
      : V( VV ), n( nn ), m( mm ), allocatedInplace( ai ),
        inEdges( _inEdges )
  {
  }

  void del()
  {
    if ( allocatedInplace == NULL )
      for ( int i = 0; i < n; i++ )
        V[i].del();
    else
      free( allocatedInplace );
    free( V );
    if ( inEdges != NULL )
      free( inEdges );
  }
};

template <class vertex>
struct Compressed_Mem : public Deletable {
public:
  vertex* V;
  char*   s;

  Compressed_Mem( vertex* _V, char* _s ) : V( _V ), s( _s ) {}

  void del()
  {
    free( V );
    free( s );
  }
};

template <class vertex>
struct graph {
  vertex*    V;
  eva_t      hb_V;
  int       n;
  int       m;
  bool       transposed;
  uintE*     flags;
  Deletable* D;

  graph( vertex* _V, eva_t _hb_V, int _n, int _m, Deletable* _D )
      : V( _V ), hb_V( _hb_V), n( _n ), m( _m ), D( _D ), flags( NULL ), transposed( 0 )
  {
  }

  graph( vertex* _V, eva_t _hb_V, int _n, int _m, Deletable* _D, uintE* _flags )
      : V( _V ), hb_V( _hb_V), n( _n ), m( _m ), D( _D ), flags( _flags ),
        transposed( 0 )
  {
  }

  void del()
  {
    if ( flags != NULL )
      free( flags );
    D->del();
    free( D );
  }

  void transpose()
  {
    if ( sizeof( vertex ) == sizeof( asymmetricVertex ) ) {
      appl::parallel_for( int( 0 ), n,
                          [&]( int i ) { V[i].flipEdges(); } );
      transposed = !transposed;
    }
  }
};
#endif
