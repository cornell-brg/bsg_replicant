// This code is part of the project "Ligra: A Lightweight Graph Processing
// Framework for Shared Memory", presented at Principles and Practice of
// Parallel Programming, 2013.
// Copyright (c) 2013 Julian Shun and Guy Blelloch
//
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the
// "Software"), to deal in the Software without restriction, including
// without limitation the rights (to use, copy, modify, merge, publish,
// distribute, sublicense, and/or sell copies of the Software, and to
// permit persons to whom the Software is furnished to do so, subject to
// the following conditions:
//
// The above copyright notice and this permission notice shall be included
// in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
// OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
// LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
// OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
// WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#include <cmath>
#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include "blockRadixSort.h"
#include "graph.h"
#include "parallel.h"
#include "quickSort.h"
#include "utils.h"

#include <bsg_manycore_cuda.h>
#include <bsg_manycore_tile.h>
#include <bsg_manycore_errno.h>
#include <bsg_manycore_loader.h>

using namespace std;

typedef pair<uintE, uintE>             intPair;
typedef pair<uintE, pair<uintE, intE>> intTriple;

template <class E>
struct pairFirstCmp {
  bool operator()( pair<uintE, E> a, pair<uintE, E> b )
  {
    return a.first < b.first;
  }
};

template <class E>
struct getFirst {
  uintE operator()( pair<uintE, E> a ) { return a.first; }
};

template <class IntType>
struct pairBothCmp {
  bool operator()( pair<uintE, IntType> a, pair<uintE, IntType> b )
  {
    if ( a.first != b.first )
      return a.first < b.first;
    return a.second < b.second;
  }
};

// A structure that keeps a sequence of strings all allocated from
// the same block of memory
struct words {
  int   n;       // total number of characters
  char*  Chars;   // array storing all strings
  int   m;       // number of substrings
  char** Strings; // pointers to strings (all should be null terminated)
  words() {}
  words( char* C, int nn, char** S, int mm )
      : Chars( C ), n( nn ), Strings( S ), m( mm )
  {
  }
  void del()
  {
    free( Chars );
    free( Strings );
  }
};

inline bool isSpace( char c )
{
  switch ( c ) {
  case '\r':
  case '\t':
  case '\n':
  case 0:
  case ' ':
    return true;
  default:
    return false;
  }
}

_seq<char> mmapStringFromFile( const char* filename )
{
  struct stat sb;
  int         fd = open( filename, O_RDONLY );
  if ( fd == -1 ) {
    perror( "open" );
    exit( -1 );
  }
  if ( fstat( fd, &sb ) == -1 ) {
    perror( "fstat" );
    exit( -1 );
  }
  if ( !S_ISREG( sb.st_mode ) ) {
    perror( "not a file\n" );
    exit( -1 );
  }
  char* p = static_cast<char*>(
      mmap( 0, sb.st_size, PROT_READ, MAP_PRIVATE, fd, 0 ) );
  if ( p == MAP_FAILED ) {
    perror( "mmap" );
    exit( -1 );
  }
  if ( close( fd ) == -1 ) {
    perror( "close" );
    exit( -1 );
  }
  size_t n = sb.st_size;
  //  char *bytes = newA(char, n);
  //  for(size_t i=0; i<n; i++) {
  //    bytes[i] = p[i];
  //  }
  //  if (munmap(p, sb.st_size) == -1) {
  //    perror("munmap");
  //    exit(-1);
  //  }
  //  cout << "mmapped" << endl;
  //  free(bytes);
  //  exit(0);
  return _seq<char>( p, n );
}

_seq<char> readStringFromFile( const char* fileName )
{
  ifstream file( fileName, ios::in | ios::binary | ios::ate );
  if ( !file.is_open() ) {
    std::cout << "Unable to open file: " << fileName << std::endl;
    abort();
  }
  int end = file.tellg();
  file.seekg( 0, ios::beg );
  int  n     = end - file.tellg();
  char* bytes = newA( char, n + 1 );
  file.read( bytes, n );
  file.close();
  return _seq<char>( bytes, n );
}

// parallel code for converting a string to words
words stringToWords( char* Str, int n )
{
  {
    for( int i = 0; i < n; i++ ) if ( isSpace( Str[i] ) ) Str[i] =
        0;
  }

  // mark start of words
  bool* FL = newA( bool, n );
  FL[0]    = Str[0];
  {
    for( int i = 1; i < n; i++ ) FL[i] = Str[i] && !Str[i - 1];
  }

  // offset for each start of word
  _seq<int> Off     = sequence::packIndex<int>( FL, n );
  int       m       = Off.n;
  int*      offsets = Off.A;

  // pointer to each start of word
  char** SA = newA( char*, m );
  {
    for( int j = 0; j < m; j++ ) SA[j] = Str + offsets[j];
  }

  free( offsets );
  free( FL );
  return words( Str, n, SA, m );
}

template <class vertex>
graph<vertex> readGraphFromFile( const char* fname, bool isSymmetric,
                                 bool mmap, hb_mc_device_t& device )
{
  words W;
  if ( mmap ) {
    _seq<char> S     = mmapStringFromFile( fname );
    char*      bytes = newA( char, S.n );
    // Cannot mutate the graph unless we copy.
    for( size_t i = 0; i < S.n; i++ ) { bytes[i] = S.A[i]; }
    if ( munmap( S.A, S.n ) == -1 ) {
      perror( "munmap" );
      exit( -1 );
    }
    S.A = bytes;
    W   = stringToWords( S.A, S.n );
  }
  else {
    _seq<char> S = readStringFromFile( fname );
    W            = stringToWords( S.A, S.n );
  }
  if ( W.Strings[0] != ( string ) "AdjacencyGraph" ) {
    cout << "Bad input file" << endl;
    abort();
  }

  int len = W.m - 1;
  int n   = atol( W.Strings[1] );
  int m   = atol( W.Strings[2] );
  if ( len != n + m + 2 ) {
    cout << "Bad input file" << endl;
    abort();
  }

  uintT* offsets = newA( uintT, n );
  uintE* edges = newA( uintE, m );
  eva_t hb_edges;
  (hb_mc_device_malloc(&device, m * sizeof(uintE), &hb_edges));

  {
    for( int i = 0; i < n; i++ ) offsets[i] =
        atol( W.Strings[i + 3] );
  }
  {
    for( int i = 0; i < m; i++ )
    {
      edges[i] = atol( W.Strings[i + n + 3] );
    }
  }
  // W.del(); // to deal with performance bug in malloc

  vertex* v      = newA( vertex, n );
  vertex* host_v = newA( vertex, n );
  eva_t hb_v;
  (hb_mc_device_malloc(&device, n * sizeof(vertex), &hb_v));

  {
    for( uintT i = 0; i < n; i++ )
    {
      uintT o = offsets[i];
      uintT l = ( ( i == n - 1 ) ? m : offsets[i + 1] ) - offsets[i];
      v[i].setOutDegree( l );
      host_v[i].setOutDegree( l );
      v[i].setOutNeighbors( edges + o );
      host_v[i].setOutNeighbors( (uintE*)(intptr_t)(hb_edges + o * sizeof(uintE)) );
    }
    // copy edges and v
    hb_mc_dma_htod_t htod[2] = {{
      .d_addr = hb_edges,
      .h_addr = (&edges[0]),
      .size   = m * sizeof(uintE)
    }, {
      .d_addr = hb_v,
      .h_addr = (&host_v[0]),
      .size   = n * sizeof(vertex)
    }};
    (hb_mc_device_dma_to_device(&device, htod, 2));
  }

  if ( !isSymmetric ) {
    abort();
  }
  else {
    free( offsets );
    Uncompressed_Mem<vertex>* mem =
        new Uncompressed_Mem<vertex>( v, n, m, edges );
    return graph<vertex>( v, hb_v, n, m, mem );
  }
}

template <class vertex>
graph<vertex> readGraph( const char* iFile, bool compressed, bool symmetric,
                         bool binary, bool mmap, hb_mc_device_t& device )
{
  if ( binary )
    abort();
  else
    return readGraphFromFile<vertex>( iFile, symmetric, mmap, device );
}
