#include <bsg_manycore_tile.h>
#include <bsg_manycore_errno.h>
#include <bsg_manycore_tile.h>
#include <bsg_manycore_loader.h>
#include <bsg_manycore_cuda.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <stdio.h>
#include <bsg_manycore_regression.h>

#define ALLOC_NAME "default_allocator"
#define MAX_WORKERS 128
#define HB_L2_CACHE_LINE_WORDS 16
#define BUF_FACTOR 2049
#define BUF_SIZE (MAX_WORKERS * HB_L2_CACHE_LINE_WORDS * BUF_FACTOR)

int nsolutions = 0;

/*
 * <a> contains array of <n> queen positions.  Returns 1
 * if none of the queens conflict, and returns 0 otherwise.
 */
int ok(int n, char *a)
{
  int i, j;
  char p, q;

  for (i = 0; i < n; i++) {
    p = a[i];

    for (j = i + 1; j < n; j++) {
      q = a[j];
      if (q == p || q == p - (j - i) || q == p + (j - i))
        return 0;
    }
  }
  return 1;
}

/*
 * <a> is an array of <j> numbers.  The entries of <a> contain
 * queen positions already set.  If there is any extension of <a>
 * to a complete <n> queen setting, returns one of these queen
 * settings (allocated from the heap).  Otherwise, returns NULL.
 * Does not side-effect <a>.
 */
void nqueens(int n, int j, char *a)
{

  if (n == j) {
    nsolutions++;
  }

  /* try each possible position for queen <j> */
  for( int i = 0; i < n; i++) {
    /* allocate a temporary array and copy <a> into it */
    // char* b = (char*)malloc((j + 1) * sizeof(char));
    char b[j+1];
    for ( int k = 0; k < j; k++ ) {
      b[k] = a[k];
    }
    b[j] = i;
    if ( ok(j + 1, b) ) {
      nqueens(n, j + 1, b);
    }
  }
}

int kernel_appl_nqueens (int argc, char **argv) {
        int rc;
        char *bin_path, *test_name;
        struct arguments_path args = {NULL, NULL};

        argp_parse (&argp_path, argc, argv, 0, 0, &args);
        bin_path = args.path;
        test_name = args.name;

        bsg_pr_test_info("Running the nqueens WS Kernel on one %dx%d tile groups.\n\n", bsg_tiles_X, bsg_tiles_Y);

        srand(time);

        /*****************************************************************************************************************
        * Define path to binary.
        * Initialize device, load binary and unfreeze tiles.
        ******************************************************************************************************************/
        hb_mc_device_t device;
        BSG_CUDA_CALL(hb_mc_device_init(&device, test_name, 0));

        hb_mc_pod_id_t pod;
        hb_mc_device_foreach_pod_id(&device, pod)
        {
                bsg_pr_info("Loading program for test %s onto pod %d\n", test_name, pod);
                BSG_CUDA_CALL(hb_mc_device_set_default_pod(&device, pod));
                BSG_CUDA_CALL(hb_mc_device_program_init(&device, bin_path, ALLOC_NAME, 0));

                /*****************************************************************************************************************
                 * Allocate memory on the device.
                 ******************************************************************************************************************/

                eva_t device_result;
                BSG_CUDA_CALL(hb_mc_device_malloc(&device, 64 * sizeof(uint32_t), &device_result)); // buffer for return results

                eva_t dram_buffer;
                BSG_CUDA_CALL(hb_mc_device_malloc(&device, BUF_SIZE * sizeof(uint32_t), &dram_buffer));

                /*****************************************************************************************************************
                 * Define block_size_x/y: amount of work for each tile group
                 * Define tg_dim_x/y: number of tiles in each tile group
                 * Calculate grid_dim_x/y: number of tile groups needed based on block_size_x/y
                 ******************************************************************************************************************/
                hb_mc_dimension_t tg_dim = { .x = bsg_tiles_X, .y = bsg_tiles_Y};
                hb_mc_dimension_t grid_dim = { .x = 1, .y = 1};

                /*****************************************************************************************************************
                 * Prepare list of input arguments for kernel.
                 ******************************************************************************************************************/
                int N = NQUEENS_IN;
                int gsize = NQUEENS_GSIZE;
                int cuda_argv[4] = {device_result, N, gsize, dram_buffer};

                /*****************************************************************************************************************
                 * Enquque grid of tile groups, pass in grid and tile group dimensions, kernel name, number and list of input arguments
                 ******************************************************************************************************************/
                BSG_CUDA_CALL(hb_mc_kernel_enqueue (&device, grid_dim, tg_dim, "kernel_appl_nqueens", 4, cuda_argv));

                /*****************************************************************************************************************
                 * Launch and execute all tile groups on device and wait for all to finish.
                 ******************************************************************************************************************/
                BSG_CUDA_CALL(hb_mc_device_tile_groups_execute(&device));

                /*****************************************************************************************************************
                 * Copy result back from device DRAM into host memory.
                 ******************************************************************************************************************/
                uint32_t host_result[64];
                void *src = (void *) ((intptr_t) device_result);;
                void *dst = (void *) &host_result[0];
                BSG_CUDA_CALL(hb_mc_device_memcpy (&device, (void *) dst, src, 64 * sizeof(uint32_t), HB_MC_MEMCPY_TO_HOST));

                /*****************************************************************************************************************
                 * Freeze the tiles and memory manager cleanup.
                 ******************************************************************************************************************/
                BSG_CUDA_CALL(hb_mc_device_program_finish(&device));


                char* a = (char*)malloc(N * sizeof(char));
                nqueens(N, 0, a);

                int expected = nsolutions;
                if (host_result[0] != expected) {
                  bsg_pr_err(BSG_RED("Mismatch: ") "nqueens %d = %d != expected %d\n", N, host_result[0], expected);
                  return HB_MC_FAIL;
                }

        }
        BSG_CUDA_CALL(hb_mc_device_finish(&device));

        return HB_MC_SUCCESS;
}

declare_program_main("test_appl_nqueens", kernel_appl_nqueens);
