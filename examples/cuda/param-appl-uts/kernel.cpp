#include <stdint.h>
#include "bsg_manycore.h"
#include "appl.hpp"
#include "brg_sha1.h"

extern "C" __attribute__ ((noinline))
int kernel_appl_uts(int* results, int n, int grain_size, int* dram_buffer) {

  // debug print
  if (__bsg_id == 0) {
    bsg_print_int(n);
    bsg_print_int(grain_size);
  }

  // --------------------- kernel ------------------------
  appl::runtime_init(dram_buffer, 2);

  // sync
  appl::sync();
  bsg_cuda_print_stat_kernel_start();

  if (__bsg_id == 0) {
    struct state_t mystate;
    for (int i = 0; i < 20; i++) {
      mystate.state[i] = i;
    }
    rng_init( mystate.state, 14850 );
    for (int i = 0; i < 20; i++) {
      bsg_print_int(mystate.state[i]);
    }
    bsg_print_int(14850);
    bsg_print_int(rng_nextrand( mystate.state ));
    bsg_print_int(rng_nextrand( mystate.state ));
  } else {
    appl::worker_thread_init();
  }
  appl::runtime_end();
  // --------------------- end of kernel -----------------

  bsg_cuda_print_stat_kernel_end();

  appl::sync();
  return 0;
}
