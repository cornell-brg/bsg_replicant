#include <stdint.h>
#include "bsg_manycore.h"
#include "appl.hpp"

typedef uint32_t ELM;

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

  } else {
    appl::worker_thread_init();
  }
  appl::runtime_end();
  bsg_cuda_print_stat_kernel_end();
  // --------------------- end of kernel -----------------

  appl::sync();
  return 0;
}
