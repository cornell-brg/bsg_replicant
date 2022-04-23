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

#define __BSG_STRINGIFY(arg) #arg
#define BSG_STRINGIFY(arg) __BSG_STRINGIFY(arg)

#define ALLOC_NAME "default_allocator"
#define MAX_WORKERS 128
#define HB_L2_CACHE_LINE_WORDS 16
#define BUF_FACTOR 129
#define BUF_SIZE (MAX_WORKERS * HB_L2_CACHE_LINE_WORDS * BUF_FACTOR)

/* every item in the knapsack has a weight and a value */
#define MAX_ITEMS 256

struct item {
  int value;
  int weight;
};

int compare( struct item* a, struct item* b ) {
  float c =
      ( (float)a->value / a->weight ) - ( (float)b->value / b->weight );

  if ( c > 0 )
    return -1;
  if ( c < 0 )
    return 1;
  return 0;
}

int read_input( const char* filename, struct item* items, int* capacity,
                int* n )
{
  int   i;
  FILE* f;

  if ( filename == NULL )
    filename = "\0";
  f = fopen( filename, "r" );
  if ( f == NULL ) {
    fprintf( stderr, "open_input(\"%s\") failed\n", filename );
    return -1;
  }
  /* format of the input: #items capacity value1 weight1 ... */
  fscanf( f, "%d", n );
  fscanf( f, "%d", capacity );

  for ( i = 0; i < *n; ++i )
    fscanf( f, "%d %d", &items[i].value, &items[i].weight );

  fclose( f );

  /* sort the items on decreasing order of value/weight */
  /* cilk2c is fascist in dealing with pointers, whence the ugly cast */
  qsort( items, *n, sizeof( struct item ),
         (int ( * )( const void*, const void* ))compare );

  return 0;
}

int kernel_appl_knapsack (int argc, char **argv) {
        int rc;
        char *bin_path, *test_name;
        struct arguments_path args = {NULL, NULL};

        argp_parse (&argp_path, argc, argv, 0, 0, &args);
        bin_path = args.path;
        test_name = args.name;

        bsg_pr_test_info("Running the Cilk Knapsack WS Kernel on one %dx%d tile groups.\n\n", bsg_tiles_X, bsg_tiles_Y);

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
                int N = -1;
                int capacity = -1;
                struct item items[MAX_ITEMS];
                char input_file [] = BSG_STRINGIFY(INPUT);
                bsg_pr_info("Opening input file '%s\n", input_file);
                if (read_input(input_file, items, &capacity, &N )) {
                    return HB_MC_FAIL;
                }

                eva_t items_device;
                BSG_CUDA_CALL(hb_mc_device_malloc(&device, N * sizeof(struct item), &items_device));

                void *dst = (void *) ((intptr_t) items_device);
                void *src = (void *) &items[0];
                BSG_CUDA_CALL(hb_mc_device_memcpy (&device, dst, src, N * sizeof(struct item), HB_MC_MEMCPY_TO_DEVICE));

                int cuda_argv[5] = {device_result, items_device, N, capacity, dram_buffer};

                /*****************************************************************************************************************
                 * Enquque grid of tile groups, pass in grid and tile group dimensions, kernel name, number and list of input arguments
                 ******************************************************************************************************************/
                BSG_CUDA_CALL(hb_mc_kernel_enqueue (&device, grid_dim, tg_dim, "kernel_appl_knapsack", 5, cuda_argv));

                /*****************************************************************************************************************
                 * Launch and execute all tile groups on device and wait for all to finish.
                 ******************************************************************************************************************/
                BSG_CUDA_CALL(hb_mc_device_tile_groups_execute(&device));

                /*****************************************************************************************************************
                 * Copy result back from device DRAM into host memory.
                 ******************************************************************************************************************/
                uint32_t host_result[64];
                src = (void *) ((intptr_t) device_result);;
                dst = (void *) &host_result[0];
                BSG_CUDA_CALL(hb_mc_device_memcpy (&device, (void *) dst, src, 64 * sizeof(uint32_t), HB_MC_MEMCPY_TO_HOST));

                /*****************************************************************************************************************
                 * Freeze the tiles and memory manager cleanup.
                 ******************************************************************************************************************/
                BSG_CUDA_CALL(hb_mc_device_program_finish(&device));
        }
        BSG_CUDA_CALL(hb_mc_device_finish(&device));

        return HB_MC_SUCCESS;
}

declare_program_main("test_appl_knapsack", kernel_appl_knapsack);
