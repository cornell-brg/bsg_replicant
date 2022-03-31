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
                                 bool mmap )
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
#ifndef WEIGHTED
  if ( W.Strings[0] != ( string ) "AdjacencyGraph" ) {
#else
  if ( W.Strings[0] != ( string ) "WeightedAdjacencyGraph" ) {
#endif
    cout << "Bad input file" << endl;
    abort();
  }

  int len = W.m - 1;
  int n   = atol( W.Strings[1] );
  int m   = atol( W.Strings[2] );
#ifndef WEIGHTED
  if ( len != n + m + 2 ) {
#else
  if ( len != n + 2 * m + 2 ) {
#endif
    cout << "Bad input file" << endl;
    abort();
  }

  uintT* offsets = newA( uintT, n );
#ifndef WEIGHTED
  uintE* edges = newA( uintE, m );
#else
  intE* edges = newA( intE, 2 * m );
#endif

  {
    for( int i = 0; i < n; i++ ) offsets[i] =
        atol( W.Strings[i + 3] );
  }
  {
    for( int i = 0; i < m; i++ )
    {
#ifndef WEIGHTED
      edges[i] = atol( W.Strings[i + n + 3] );
#else
      edges[2 * i]     = atol( W.Strings[i + n + 3] );
      edges[2 * i + 1] = atol( W.Strings[i + n + m + 3] );
#endif
    }
  }
  // W.del(); // to deal with performance bug in malloc

  vertex* v = newA( vertex, n );

  {
    for( uintT i = 0; i < n; i++ )
    {
      uintT o = offsets[i];
      uintT l = ( ( i == n - 1 ) ? m : offsets[i + 1] ) - offsets[i];
      v[i].setOutDegree( l );
#ifndef WEIGHTED
      v[i].setOutNeighbors( edges + o );
#else
      v[i].setOutNeighbors( edges + 2 * o );
#endif
    }
  }

  if ( !isSymmetric ) {
    uintT* tOffsets = newA( uintT, n );
    {
      for( int i = 0; i < n; i++ ) tOffsets[i] = INT_T_MAX;
    }
#ifndef WEIGHTED
    intPair* temp = newA( intPair, m );
#else
    intTriple* temp = newA( intTriple, m );
#endif
    {
      for( int i = 0; i < n; i++ )
      {
        uintT o = offsets[i];
        for ( uintT j = 0; j < v[i].getOutDegree(); j++ ) {
#ifndef WEIGHTED
          temp[o + j] = make_pair( v[i].getOutNeighbor( j ), i );
#else
          temp[o + j] =
              make_pair( v[i].getOutNeighbor( j ),
                         make_pair( i, v[i].getOutWeight( j ) ) );
#endif
        }
      }
    }
    free( offsets );

#ifndef WEIGHTED
#ifndef LOWMEM
    intSort::iSort( temp, m, n + 1, getFirst<uintE>() );
#else
    quickSort( temp, m, pairFirstCmp<uintE>() );
#endif
#else
#ifndef LOWMEM
    intSort::iSort( temp, m, n + 1, getFirst<intPair>() );
#else
    quickSort( temp, m, pairFirstCmp<intPair>() );
#endif
#endif

    tOffsets[temp[0].first] = 0;
#ifndef WEIGHTED
    uintE* inEdges = newA( uintE, m );
    inEdges[0]     = temp[0].second;
#else
    intE* inEdges = newA( intE, 2 * m );
    inEdges[0]    = temp[0].second.first;
    inEdges[1]    = temp[0].second.second;
#endif
    {
      for( int i = 1; i < m; i++ )
      {
#ifndef WEIGHTED
        inEdges[i] = temp[i].second;
#else
        inEdges[2 * i]     = temp[i].second.first;
        inEdges[2 * i + 1] = temp[i].second.second;
#endif
        if ( temp[i].first != temp[i - 1].first ) {
          tOffsets[temp[i].first] = i;
        }
      }
    }

    free( temp );

    // fill in offsets of degree 0 vertices by taking closest non-zero
    // offset to the right
    sequence::scanIBack( tOffsets, tOffsets, n, minF<uintT>(), (uintT)m );

    {
      for( int i = 0; i < n; i++ )
      {
        uintT o = tOffsets[i];
        uintT l = ( ( i == n - 1 ) ? m : tOffsets[i + 1] ) - tOffsets[i];
        v[i].setInDegree( l );
#ifndef WEIGHTED
        v[i].setInNeighbors( inEdges + o );
#else
        v[i].setInNeighbors( inEdges + 2 * o );
#endif
      }
    }

    free( tOffsets );
    Uncompressed_Mem<vertex>* mem =
        new Uncompressed_Mem<vertex>( v, n, m, edges, inEdges );
    return graph<vertex>( v, n, m, mem );
  }
  else {
    free( offsets );
    Uncompressed_Mem<vertex>* mem =
        new Uncompressed_Mem<vertex>( v, n, m, edges );
    return graph<vertex>( v, n, m, mem );
  }
}

template <class vertex>
graph<vertex> readGraphFromBinary( const char* iFile, bool isSymmetric )
{
  char* config = (char*)".config";
  char* adj    = (char*)".adj";
  char* idx    = (char*)".idx";
  char  configFile[strlen( iFile ) + strlen( config ) + 1];
  char  adjFile[strlen( iFile ) + strlen( adj ) + 1];
  char  idxFile[strlen( iFile ) + strlen( idx ) + 1];
  *configFile = *adjFile = *idxFile = '\0';
  strcat( configFile, iFile );
  strcat( adjFile, iFile );
  strcat( idxFile, iFile );
  strcat( configFile, config );
  strcat( adjFile, adj );
  strcat( idxFile, idx );

  ifstream in( configFile, ifstream::in );
  int     n;
  in >> n;
  in.close();

  ifstream in2( adjFile, ifstream::in | ios::binary ); // stored as uints
  in2.seekg( 0, ios::end );
  int size = in2.tellg();
  in2.seekg( 0 );
#ifdef WEIGHTED
  int m = size / ( 2 * sizeof( uint ) );
#else
  int m = size / sizeof( uint );
#endif
  char* s = (char*)malloc( size );
  in2.read( s, size );
  in2.close();
  uintE* edges = (uintE*)s;

  ifstream in3( idxFile, ifstream::in | ios::binary ); // stored as ints
  in3.seekg( 0, ios::end );
  size = in3.tellg();
  in3.seekg( 0 );
  if ( n != size / sizeof( intT ) ) {
    cout << "File size wrong\n";
    abort();
  }

  char* t = (char*)malloc( size );
  in3.read( t, size );
  in3.close();
  uintT* offsets = (uintT*)t;

  vertex* v = newA( vertex, n );
#ifdef WEIGHTED
  intE* edgesAndWeights = newA( intE, 2 * m );
  {
    for( int i = 0; i < m; i++ )
    {
      edgesAndWeights[2 * i]     = edges[i];
      edgesAndWeights[2 * i + 1] = edges[i + m];
    }
  }
  // free(edges);
#endif
  {
    for( int i = 0; i < n; i++ )
    {
      uintT o = offsets[i];
      uintT l = ( ( i == n - 1 ) ? m : offsets[i + 1] ) - offsets[i];
      v[i].setOutDegree( l );
#ifndef WEIGHTED
      v[i].setOutNeighbors( (uintE*)edges + o );
#else
      v[i].setOutNeighbors( edgesAndWeights + 2 * o );
#endif
    }
  }

  if ( !isSymmetric ) {
    uintT* tOffsets = newA( uintT, n );
    {
      for( int i = 0; i < n; i++ ) tOffsets[i] = INT_T_MAX;
    }
#ifndef WEIGHTED
    intPair* temp = newA( intPair, m );
#else
    intTriple* temp = newA( intTriple, m );
#endif
    {
      for( intT i = 0; i < n; i++ )
      {
        uintT o = offsets[i];
        for ( uintT j = 0; j < v[i].getOutDegree(); j++ ) {
#ifndef WEIGHTED
          temp[o + j] = make_pair( v[i].getOutNeighbor( j ), i );
#else
          temp[o + j] =
              make_pair( v[i].getOutNeighbor( j ),
                         make_pair( i, v[i].getOutWeight( j ) ) );
#endif
        }
      }
    }
    free( offsets );
#ifndef WEIGHTED
#ifndef LOWMEM
    intSort::iSort( temp, m, n + 1, getFirst<uintE>() );
#else
    quickSort( temp, m, pairFirstCmp<uintE>() );
#endif
#else
#ifndef LOWMEM
    intSort::iSort( temp, m, n + 1, getFirst<intPair>() );
#else
    quickSort( temp, m, pairFirstCmp<intPair>() );
#endif
#endif
    tOffsets[temp[0].first] = 0;
#ifndef WEIGHTED
    uintE* inEdges = newA( uintE, m );
    inEdges[0]     = temp[0].second;
#else
    intE* inEdges = newA( intE, 2 * m );
    inEdges[0]    = temp[0].second.first;
    inEdges[1]    = temp[0].second.second;
#endif
    {
      for( int i = 1; i < m; i++ )
      {
#ifndef WEIGHTED
        inEdges[i] = temp[i].second;
#else
        inEdges[2 * i]     = temp[i].second.first;
        inEdges[2 * i + 1] = temp[i].second.second;
#endif
        if ( temp[i].first != temp[i - 1].first ) {
          tOffsets[temp[i].first] = i;
        }
      }
    }
    free( temp );
    // fill in offsets of degree 0 vertices by taking closest non-zero
    // offset to the right
    sequence::scanIBack( tOffsets, tOffsets, n, minF<uintT>(), (uintT)m );
    {
      for( int i = 0; i < n; i++ )
      {
        uintT o = tOffsets[i];
        uintT l = ( ( i == n - 1 ) ? m : tOffsets[i + 1] ) - tOffsets[i];
        v[i].setInDegree( l );
#ifndef WEIGHTED
        v[i].setInNeighbors( (uintE*)inEdges + o );
#else
        v[i].setInNeighbors( (intE*)( inEdges + 2 * o ) );
#endif
      }
    }
    free( tOffsets );
#ifndef WEIGHTED
    Uncompressed_Mem<vertex>* mem =
        new Uncompressed_Mem<vertex>( v, n, m, edges, inEdges );
    return graph<vertex>( v, n, m, mem );
#else
    Uncompressed_Mem<vertex>* mem =
        new Uncompressed_Mem<vertex>( v, n, m, edgesAndWeights, inEdges );
    return graph<vertex>( v, n, m, mem );
#endif
  }
  free( offsets );
#ifndef WEIGHTED
  Uncompressed_Mem<vertex>* mem =
      new Uncompressed_Mem<vertex>( v, n, m, edges );
  return graph<vertex>( v, n, m, mem );
#else
  Uncompressed_Mem<vertex>* mem =
      new Uncompressed_Mem<vertex>( v, n, m, edgesAndWeights );
  return graph<vertex>( v, n, m, mem );
#endif
}

template <class vertex>
graph<vertex> readGraph( const char* iFile, bool compressed, bool symmetric,
                         bool binary, bool mmap )
{
  if ( binary )
    return readGraphFromBinary<vertex>( iFile, symmetric );
  else
    return readGraphFromFile<vertex>( iFile, symmetric, mmap );
}
