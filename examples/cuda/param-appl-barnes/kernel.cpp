#include <stdint.h>
#include "bsg_manycore.h"
#include "appl.hpp"

extern "C" __attribute__ ((noinline))
int kernel_appl_barnes(int* results, int n, int grain_size, int* dram_buffer) {

  // debug print
  if (__bsg_id == 0) {
    bsg_print_int(n);
    bsg_print_int(grain_size);
  }

  // output
  int32_t result     = -1;

  // --------------------- kernel ------------------------
  appl::runtime_init(dram_buffer, 2);

  // sync
  appl::sync();
  bsg_cuda_print_stat_kernel_start();

  if (__bsg_id == 0) {
    results[0] = result;
  } else {
    appl::worker_thread_init();
  }
  appl::runtime_end();
  // --------------------- end of kernel -----------------

  bsg_cuda_print_stat_kernel_end();
  bsg_print_int(result);

  appl::sync();
  return 0;
}
