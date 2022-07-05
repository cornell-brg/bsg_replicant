#include <stdint.h>
#include "bsg_manycore.h"
#include "appl.hpp"

#define merge_size 64
#define quick_size 64
#define INSERTIONSIZE 20

typedef uint32_t ELM;

void ELM_memcpy(ELM* dest, ELM* src, size_t n) {
  bsg_unroll(32)
  for( size_t i = 0; i < n; i++ ) {
    dest[i] = src[i];
  }
}

static inline ELM med3( ELM a, ELM b, ELM c )
{
  if ( a < b ) {
    if ( b < c ) {
      return b;
    }
    else {
      if ( a < c )
        return c;
      else
        return a;
    }
  }
  else {
    if ( b > c ) {
      return b;
    }
    else {
      if ( a > c )
        return c;
      else
        return a;
    }
  }
}

static inline ELM choose_pivot( ELM* low, ELM* high )
{
  return med3( *low, *high, low[( high - low ) / 2] );
}

static ELM* seqpart( ELM* low, ELM* high )
{
  ELM  pivot;
  ELM  h, l;
  ELM* curr_low  = low;
  ELM* curr_high = high;

  pivot = choose_pivot( low, high );

  while ( 1 ) {
    while ( ( h = *curr_high ) > pivot )
      curr_high--;

    while ( ( l = *curr_low ) < pivot )
      curr_low++;

    if ( curr_low >= curr_high )
      break;

    *curr_high-- = l;
    *curr_low++  = h;
  }

  /*
   * I don't know if this is really necessary.
   * The problem is that the pivot is not always the
   * first element, and the partition may be trivial.
   * However, if the partition is trivial, then
   * *high is the largest element, whence the following
   * code.
   */
  if ( curr_high < high )
    return curr_high;
  else
    return curr_high - 1;
}

#define swap( a, b )                                                     \
  {                                                                      \
    ELM tmp;                                                             \
    tmp = a;                                                             \
    a   = b;                                                             \
    b   = tmp;                                                           \
  }

static void insertion_sort( ELM* low, ELM* high )
{
  ELM *p, *q;
  ELM  a, b;

  for ( q = low + 1; q <= high; ++q ) {
    a = q[0];
    for ( p = q - 1; p >= low && ( b = p[0] ) > a; p-- )
      p[1] = b;
    p[1] = a;
  }
}

/*
 * tail-recursive quicksort, almost unrecognizable :-)
 */
void seqquick( ELM* low, ELM* high )
{
  ELM* p;

  while ( high - low >= INSERTIONSIZE ) {
    p = seqpart( low, high );
    seqquick( low, p );
    low = p + 1;
  }

  insertion_sort( low, high );
}

void seqmerge( ELM* low1, ELM* high1, ELM* low2, ELM* high2,
               ELM* lowdest )
{
  ELM a1, a2;

  /*
   * The following 'if' statement is not necessary
   * for the correctness of the algorithm, and is
   * in fact subsumed by the rest of the function.
   * However, it is a few percent faster.  Here is why.
   *
   * The merging loop below has something like
   *   if (a1 < a2) {
   *        *dest++ = a1;
   *        ++low1;
   *        if (end of array) break;
   *        a1 = *low1;
   *   }
   *
   * Now, a1 is needed immediately in the next iteration
   * and there is no way to mask the latency of the load.
   * A better approach is to load a1 *before* the end-of-array
   * check; the problem is that we may be speculatively
   * loading an element out of range.  While this is
   * probably not a problem in practice, yet I don't feel
   * comfortable with an incorrect algorithm.  Therefore,
   * I use the 'fast' loop on the array (except for the last
   * element) and the 'slow' loop for the rest, saving both
   * performance and correctness.
   */

  if ( low1 < high1 && low2 < high2 ) {
    a1 = *low1;
    a2 = *low2;
    for ( ;; ) {
      if ( a1 < a2 ) {
        *lowdest++ = a1;
        a1         = *++low1;
        if ( low1 >= high1 )
          break;
      }
      else {
        *lowdest++ = a2;
        a2         = *++low2;
        if ( low2 >= high2 )
          break;
      }
    }
  }
  if ( low1 <= high1 && low2 <= high2 ) {
    a1 = *low1;
    a2 = *low2;
    for ( ;; ) {
      if ( a1 < a2 ) {
        *lowdest++ = a1;
        ++low1;
        if ( low1 > high1 )
          break;
        a1 = *low1;
      }
      else {
        *lowdest++ = a2;
        ++low2;
        if ( low2 > high2 )
          break;
        a2 = *low2;
      }
    }
  }
  if ( low1 > high1 ) {
    ELM_memcpy( lowdest, low2, ( high2 - low2 + 1 ) );
  }
  else {
    ELM_memcpy( lowdest, low1, ( high1 - low1 + 1 ) );
  }
}

#define swap_indices( a, b )                                             \
  {                                                                      \
    ELM* tmp;                                                            \
    tmp = a;                                                             \
    a   = b;                                                             \
    b   = tmp;                                                           \
  }

ELM* binsplit( ELM val, ELM* low, ELM* high )
{
  /*
   * returns index which contains greatest element <= val.  If val is
   * less than all elements, returns low-1
   */
  ELM* mid;

  while ( low != high ) {
    mid = low + ( ( high - low + 1 ) >> 1 );
    if ( val <= *mid )
      high = mid - 1;
    else
      low = mid;
  }

  if ( *low > val )
    return low - 1;
  else
    return low;
}

void cilkmerge( ELM* low1, ELM* high1, ELM* low2, ELM* high2,
                ELM* lowdest )
{
  /*
   * Cilkmerge: Merges range [low1, high1] with range [low2, high2]
   * into the range [lowdest, ...]
   */

  ELM *split1, *split2; /*
                         * where each of the ranges are broken for
                         * recursive merge
                         */
  int32_t lowsize;      /*
                         * total size of lower halves of two
                         * ranges - 2
                         */

  /*
   * We want to take the middle element (indexed by split1) from the
   * larger of the two arrays.  The following code assumes that split1
   * is taken from range [low1, high1].  So if [low1, high1] is
   * actually the smaller range, we should swap it with [low2, high2]
   */

  if ( high2 - low2 > high1 - low1 ) {
    swap_indices( low1, low2 );
    swap_indices( high1, high2 );
  }
  if ( high1 < low1 ) {
    /* smaller range is empty */
    ELM_memcpy( lowdest, low2, ( high2 - low2 ) );
    return;
  }
  if ( high2 - low2 < merge_size ) {
    seqmerge( low1, high1, low2, high2, lowdest );
    return;
  }
  /*
   * Basic approach: Find the middle element of one range (indexed by
   * split1). Find where this element would fit in the other range
   * (indexed by split 2). Then merge the two lower halves and the two
   * upper halves.
   */

  split1  = ( ( high1 - low1 + 1 ) / 2 ) + low1;
  split2  = binsplit( *split1, low2, high2 );
  lowsize = split1 - low1 + split2 - low2;

  /*
   * directly put the splitting element into
   * the appropriate location
   */
  *( lowdest + lowsize + 1 ) = *split1;

  appl::parallel_invoke(
      [&] { cilkmerge( low1, split1 - 1, low2, split2, lowdest ); },
      [&] {
        cilkmerge( split1 + 1, high1, split2 + 1, high2,
                   lowdest + lowsize + 2 );
      } );

  return;
}

void cilksort( ELM* low, ELM* tmp, int32_t size )
{
  /*
   * divide the input in four parts of the same size (A, B, C, D)
   * Then:
   *   1) recursively sort A, B, C, and D (in parallel)
   *   2) merge A and B into tmp1, and C and D into tmp2 (in parallel)
   *   3) merbe tmp1 and tmp2 into the original array
   */
  int32_t quarter = size / 4;
  ELM *A, *B, *C, *D, *tmpA, *tmpB, *tmpC, *tmpD;

  if ( size < quick_size ) {
    /* quicksort when less than 1024 elements */
    seqquick( low, low + size - 1 );
    return;
  }
  A    = low;
  tmpA = tmp;
  B    = A + quarter;
  tmpB = tmpA + quarter;
  C    = B + quarter;
  tmpC = tmpB + quarter;
  D    = C + quarter;
  tmpD = tmpC + quarter;

  appl::parallel_invoke(
      [&] { cilksort( A, tmpA, quarter ); },
      [&] { cilksort( B, tmpB, quarter ); },
      [&] { cilksort( C, tmpC, quarter ); },
      [&] { cilksort( D, tmpD, size - 3 * quarter ); } );

  appl::parallel_invoke(
      [&] { cilkmerge( A, A + quarter - 1, B, B + quarter - 1, tmpA ); },
      [&] { cilkmerge( C, C + quarter - 1, D, low + size - 1, tmpC ); } );

  cilkmerge( tmpA, tmpC - 1, tmpC, tmpA + size - 1, A );
}


extern "C" __attribute__ ((noinline))
int kernel_appl_cilksort(ELM* array, ELM* tmp, int n, int* dram_buffer) {

  // debug print
  if (__bsg_id == 0) {
    bsg_print_int(n);
  }

  // --------------------- kernel ------------------------
  appl::runtime_init(dram_buffer);

  // sync
  appl::sync();

  bsg_cuda_print_stat_kernel_start();
  if (__bsg_id == 0) {
    cilksort( array, tmp, n );
  } else {
    appl::worker_thread_init();
  }
  appl::runtime_end();
  bsg_cuda_print_stat_kernel_end();
  // --------------------- end of kernel -----------------

  appl::sync();
  return 0;
}
