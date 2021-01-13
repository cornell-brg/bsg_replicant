// Copyright (c) 2019, University of Washington All rights reserved.
// 
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
// 
// Redistributions of source code must retain the above copyright notice, this list
// of conditions and the following disclaimer.
// 
// Redistributions in binary form must reproduce the above copyright notice, this
// list of conditions and the following disclaimer in the documentation and/or
// other materials provided with the distribution.
// 
// Neither the name of the copyright holder nor the names of its contributors may
// be used to endorse or promote products derived from this software without
// specific prior written permission.
// 
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
// ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
// WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
// ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
// ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include "test_cgra_xcel_hb_tapeout.h"

#define ALLOC_NAME "default_allocator"

int get_num_tests() {
  // Get the number of test vectors based on the number of files under the
  // given directory CGRA_TEST_VECTOR_PATH
  DIR* directory = opendir(CGRA_TEST_VECTOR_PATH);
  struct dirent* entry;

  int num_tests = 0;

  if ( directory ) {
    while ((entry = readdir(directory)) != NULL) {
      /* bsg_pr_test_info("Found %s\n\n", entry->d_name); */
      if (entry->d_name[0] != '.')
        num_tests++;
    }
    closedir(directory);
    return num_tests;
  } else {
    bsg_pr_err("failed to open CGRA test vector directory %s!\n", CGRA_TEST_VECTOR_PATH);
    return -1;
  }
}

int* process_test_vector_item( FILE* fp, const char* format, int* size ) {
  char lineptr[1024];
  int addr;
  // Read one item from the given input file descriptor
  fscanf(fp, format, size);

  if (*size < 1) {
    bsg_pr_err("Format %s gives invalid size %d!\n", format, *size);
  }

  int* array = (int*) malloc((*size) * sizeof(int));

  for (int i = 0; i < *size; i++) {
    while (1) {

      fgets(lineptr, 1024, fp);

      /* bsg_pr_test_info("Line: %s, i = %d, format %s, size %d\n", lineptr, i, format, *size); */

      if ((lineptr[0] == '/') && (lineptr[1] == '/')) {
        // skip comment lines
        continue;
      } else if (lineptr[0] == '\n') {
        // skip empty lines
        continue;
      } else if ((lineptr[0] == '/') && lineptr[1] == '*') {
        // found an address inside /**/ comment
        sscanf(lineptr, "/* 0x%x */ %x", &addr, &array[i]);
        break;
      } else {
        // assume it's just data on this line
        sscanf(lineptr, "0x%x", &array[i]);
        break;
      }
    }
  }

  return array;
}

TestVector* create_test_vector( char* filename ) {
  // Create a test vector object from the given file
  char path[1024] = CGRA_TEST_VECTOR_PATH;
  int len = strlen(path);
  for (int i = 0; i < strlen(filename); i++)
    path[i+len] = filename[i];
  FILE* fp = fopen( path, "r" );

  TestVector* tv = (TestVector*) malloc(sizeof(TestVector));

  if (!fp) {
    bsg_pr_err("failed to open file %s!\n", path);
    return NULL;
  }

  bsg_pr_test_info("Looking at %s\n", path);

  // Assume the test vector is presented strictly in the following order:
  // bitstream, instruction, config arg0~3, verif_base_addr, reference

  tv->bitstream = process_test_vector_item(fp, "bitstream %d\n", &tv->bitstream_size);
  tv->instructions = process_test_vector_item(fp, "\ninstructions %d\n", &tv->instruction_size);
  tv->arg0 = process_test_vector_item(fp, "\narg0 %d\n", &tv->config_instruction_size);
  tv->arg1 = process_test_vector_item(fp, "\narg1 %d\n", &tv->config_instruction_size);
  tv->arg2 = process_test_vector_item(fp, "\narg2 %d\n", &tv->config_instruction_size);
  tv->arg3 = process_test_vector_item(fp, "\narg3 %d\n", &tv->config_instruction_size);
  fscanf(fp, "\nverif_base_addr 0x%x\n", &tv->verif_base_addr);
  tv->reference = process_test_vector_item(fp, "\nreference %d\n", &tv->reference_size);

  return tv;
}

int build_test_list( TestList* list ) {
  // Iterate over all files under CGRA_TEST_VECTOR_PATH
  DIR* directory = opendir(CGRA_TEST_VECTOR_PATH);
  struct dirent* entry;
  int rc, idx = 0;

  rc = get_num_tests();

  if (rc < 0) {
    return HB_MC_FAIL;
  }

  list->num_tests = rc;
  list->test_names = (char**) malloc(list->num_tests * sizeof(char*));
  list->test_vectors = (TestVector**) malloc(list->num_tests * sizeof(TestVector*));

  if ( directory ) {
    while ((entry = readdir(directory)) != NULL) {
      if (entry->d_name[0] == '.')
        continue;
      // Copy name to list->test_names
      int test_name_len = strlen(entry->d_name);
      int strcpy_idx = 0;
      list->test_names[idx] = (char*) malloc((test_name_len+1) * sizeof(char));

      while (entry->d_name[strcpy_idx] != '\0') {
        list->test_names[idx][strcpy_idx] = entry->d_name[strcpy_idx];
        strcpy_idx++;
      }

      // Create test vector
      list->test_vectors[idx] = create_test_vector( entry->d_name );

      idx++;
    }
    closedir(directory);

    return 0;
  } else {
    bsg_pr_err("failed to open CGRA test vector directory %s!\n", CGRA_TEST_VECTOR_PATH);
    return -1;
  }
}

void destory_test_list( TestList* list ) {
  for (int i = 0; i < list->num_tests; i++)
    free(list->test_names[i]);
  free(list->test_names);

  for (int i = 0; i < list->num_tests; i++) {
    free(list->test_vectors[i]->bitstream);
    free(list->test_vectors[i]->instructions);
    free(list->test_vectors[i]->arg0);
    free(list->test_vectors[i]->arg1);
    free(list->test_vectors[i]->arg2);
    free(list->test_vectors[i]->arg3);
    free(list->test_vectors[i]);
  }
}

void display_test_vector(TestVector* vec) {
  printf("******************************\n");
  printf("bstrm size %d, inst size %d, cinst size %d\n",
      vec->bitstream_size, vec->instruction_size, vec->config_instruction_size);
  printf("verif base addr %d, ref size %d\n",
      vec->verif_base_addr, vec->reference_size);

  printf("bitstream:\n");
  for (int i = 0; i < vec->bitstream_size; i++)
    printf("%x\n", vec->bitstream[i]);
  printf("\n");

  printf("inst:\n");
  for (int i = 0; i < vec->instruction_size; i++)
    printf("%x\n", vec->instructions[i]);
  printf("\n");

  printf("arg0:\n");
  for (int i = 0; i < vec->config_instruction_size; i++)
    printf("%x\n", vec->arg0[i]);
  printf("\n");

  printf("arg1:\n");
  for (int i = 0; i < vec->config_instruction_size; i++)
    printf("%x\n", vec->arg1[i]);
  printf("\n");

  printf("arg2:\n");
  for (int i = 0; i < vec->config_instruction_size; i++)
    printf("%x\n", vec->arg2[i]);
  printf("\n");

  printf("arg3:\n");
  for (int i = 0; i < vec->config_instruction_size; i++)
    printf("%x\n", vec->arg3[i]);
  printf("\n");

  printf("ref:\n");
  for (int i = 0; i < vec->reference_size; i++)
    printf("%x\n", vec->reference[i]);
  printf("\n");

  printf("******************************\n");
}

int run_test(int idx, TestList* list, char* bin_path, char* test_name) {

        int rc;

        printf("\n\n");
        printf("********** Test Case #%d %s **********\n\n", idx, list->test_names[idx]);

        /* display_test_vector(list->test_vectors[idx]); */

        //---------------------------------------------------------------------
        // Initialize the underlying hardware
        //---------------------------------------------------------------------

        // Define path to binary.
        // Initialize device, load binary and unfreeze tiles.

        hb_mc_device_t device;
        rc = hb_mc_device_init(&device, test_name, 0);
        if (rc != HB_MC_SUCCESS) { 
                bsg_pr_err("failed to initialize device.\n");
                return rc;
        }

        rc = hb_mc_device_program_init(&device, bin_path, ALLOC_NAME, 0);
        if (rc != HB_MC_SUCCESS) { 
                bsg_pr_err("failed to initialize program.\n");
                return rc;
        }

        // Allocate memory on the device for bitstream and arguments

        bsg_pr_test_info("Allocating memory for bitstream and arguments\n");

        uint32_t bstrm_size = list->test_vectors[idx]->bitstream_size;

        eva_t bstrm_device;

        rc = hb_mc_device_malloc(&device, bstrm_size * sizeof(uint32_t), &bstrm_device);
        if (rc != HB_MC_SUCCESS) { 
                bsg_pr_err("failed to allocate memory on device.\n");
                return rc;
        }

        // Allocate space for the arguments

        uint32_t inst_size = list->test_vectors[idx]->instruction_size;
        uint32_t arg_size = list->test_vectors[idx]->config_instruction_size;
        uint32_t result_size = list->test_vectors[idx]->reference_size;

        int verif_base_addr = list->test_vectors[idx]->verif_base_addr;

        eva_t inst_device, arg0_device, arg1_device, arg2_device, arg3_device,
              result_device;

        rc = hb_mc_device_malloc(&device, inst_size * sizeof(int), &inst_device);
        if (rc != HB_MC_SUCCESS) { 
                bsg_pr_err("failed to allocate memory on device.\n");
                return rc;
        }

        rc = hb_mc_device_malloc(&device, arg_size * sizeof(int), &arg0_device);
        if (rc != HB_MC_SUCCESS) { 
                bsg_pr_err("failed to allocate memory on device.\n");
                return rc;
        }
        rc = hb_mc_device_malloc(&device, arg_size * sizeof(int), &arg1_device);
        if (rc != HB_MC_SUCCESS) { 
                bsg_pr_err("failed to allocate memory on device.\n");
                return rc;
        }
        rc = hb_mc_device_malloc(&device, arg_size * sizeof(int), &arg2_device);
        if (rc != HB_MC_SUCCESS) { 
                bsg_pr_err("failed to allocate memory on device.\n");
                return rc;
        }
        rc = hb_mc_device_malloc(&device, arg_size * sizeof(int), &arg3_device);
        if (rc != HB_MC_SUCCESS) { 
                bsg_pr_err("failed to allocate memory on device.\n");
                return rc;
        }

        // Allocate space for results

        rc = hb_mc_device_malloc(&device, result_size * sizeof(int), &result_device);
        if (rc != HB_MC_SUCCESS) { 
                bsg_pr_err("failed to allocate memory on device.\n");
                return rc;
        }

        // Copy bitstream & arguments from host onto device DRAM.

        bsg_pr_test_info("Copying data into HB DRAM\n");

        void *dst = (void *) ((intptr_t) bstrm_device);
        void *src = (void *) list->test_vectors[idx]->bitstream;
        rc = hb_mc_device_memcpy (&device, dst, src, bstrm_size * sizeof(uint32_t), HB_MC_MEMCPY_TO_DEVICE);
        if (rc != HB_MC_SUCCESS) { 
                bsg_pr_err("failed to copy memory to device.\n");
                return rc;
        }

        dst = (void *) ((intptr_t) inst_device);
        src = (void *) list->test_vectors[idx]->instructions;
        rc = hb_mc_device_memcpy (&device, dst, src, inst_size * sizeof(uint32_t), HB_MC_MEMCPY_TO_DEVICE);
        if (rc != HB_MC_SUCCESS) { 
                bsg_pr_err("failed to copy memory to device.\n");
                return rc;
        }

        dst = (void *) ((intptr_t) arg0_device);
        src = (void *) list->test_vectors[idx]->arg0;
        rc = hb_mc_device_memcpy (&device, dst, src, arg_size * sizeof(uint32_t), HB_MC_MEMCPY_TO_DEVICE);
        if (rc != HB_MC_SUCCESS) { 
                bsg_pr_err("failed to copy memory to device.\n");
                return rc;
        }

        dst = (void *) ((intptr_t) arg1_device);
        src = (void *) list->test_vectors[idx]->arg1;
        rc = hb_mc_device_memcpy (&device, dst, src, arg_size * sizeof(uint32_t), HB_MC_MEMCPY_TO_DEVICE);
        if (rc != HB_MC_SUCCESS) { 
                bsg_pr_err("failed to copy memory to device.\n");
                return rc;
        }

        dst = (void *) ((intptr_t) arg2_device);
        src = (void *) list->test_vectors[idx]->arg2;
        rc = hb_mc_device_memcpy (&device, dst, src, arg_size * sizeof(uint32_t), HB_MC_MEMCPY_TO_DEVICE);
        if (rc != HB_MC_SUCCESS) { 
                bsg_pr_err("failed to copy memory to device.\n");
                return rc;
        }

        dst = (void *) ((intptr_t) arg3_device);
        src = (void *) list->test_vectors[idx]->arg3;
        rc = hb_mc_device_memcpy (&device, dst, src, arg_size * sizeof(uint32_t), HB_MC_MEMCPY_TO_DEVICE);
        if (rc != HB_MC_SUCCESS) { 
                bsg_pr_err("failed to copy memory to device.\n");
                return rc;
        }

        // Define tg_dim_x/y: number of tiles in each tile group
        // Calculate grid_dim_x/y: number of tile groups needed

        // PP: sinec the accelerator will do the work a 1x1 tile group should be enough?
        hb_mc_dimension_t tg_dim = { .x = 2, .y = 2};

        hb_mc_dimension_t grid_dim = { .x = 1, .y = 1};

        // Prepare list of input arguments for kernel.

        int cuda_argv[11] = { bstrm_device, bstrm_size,
                              inst_device, arg0_device, arg1_device,
                              arg2_device, arg3_device,
                              inst_size,
                              result_device,
                              verif_base_addr,
                              result_size };

        // Enquque grid of tile groups, pass in grid and tile group dimensions, kernel name, number and list of input arguments

        bsg_pr_test_info("Enqueuing kernel to HB\n");

        rc = hb_mc_kernel_enqueue (&device, grid_dim, tg_dim, "kernel_cgra_xcel_hb_tapeout", 11, cuda_argv);
        if (rc != HB_MC_SUCCESS) { 
                bsg_pr_err("failed to initialize grid.\n");
                return rc;
        }

        // Launch and execute all tile groups on device and wait for all to finish. 

        bsg_pr_test_info("HB kernel execution starts\n");

        rc = hb_mc_device_tile_groups_execute(&device);
        if (rc != HB_MC_SUCCESS) { 
                bsg_pr_err("failed to execute tile groups.\n");
                return rc;
        }

        // Copy result matrix back from device DRAM into host memory.

        bsg_pr_test_info("Copying result from HB DRAM into host...\n");

        uint32_t result_host[result_size];
        src = (void *) ((intptr_t) result_device);
        dst = (void *) &result_host[0];
        rc = hb_mc_device_memcpy (&device, (void *) dst, src, result_size * sizeof(uint32_t), HB_MC_MEMCPY_TO_HOST);
        if (rc != HB_MC_SUCCESS) { 
                bsg_pr_err("failed to copy memory from device.\n");
                return rc;
        }

        // Free the allocated space on device DRAM
        // NOTE: since we do device_finish this is not necessary...

        // rc = hb_mc_device_free(&device, bstrm_device);
        // if (rc != HB_MC_SUCCESS) {
        //         bsg_pr_err("failed to free bstrm.\n");
        //         return rc;
        // }

        // rc = hb_mc_device_free(&device, inst_device);
        // if (rc != HB_MC_SUCCESS) {
        //         bsg_pr_err("failed to free inst.\n");
        //         return rc;
        // }

        // rc = hb_mc_device_free(&device, arg0_device);
        // if (rc != HB_MC_SUCCESS) {
        //         bsg_pr_err("failed to free arg0.\n");
        //         return rc;
        // }

        // rc = hb_mc_device_free(&device, arg1_device);
        // if (rc != HB_MC_SUCCESS) {
        //         bsg_pr_err("failed to free arg1.\n");
        //         return rc;
        // }

        // rc = hb_mc_device_free(&device, arg2_device);
        // if (rc != HB_MC_SUCCESS) {
        //         bsg_pr_err("failed to free arg2.\n");
        //         return rc;
        // }

        // rc = hb_mc_device_free(&device, arg3_device);
        // if (rc != HB_MC_SUCCESS) {
        //         bsg_pr_err("failed to free arg3.\n");
        //         return rc;
        // }

        // rc = hb_mc_device_free(&device, result_device);
        // if (rc != HB_MC_SUCCESS) {
        //         bsg_pr_err("failed to free result.\n");
        //         return rc;
        // }

        // Freeze the tiles and memory manager cleanup.

        bsg_pr_test_info("Finalizing HB device...\n");

        rc = hb_mc_device_finish(&device);
        if (rc != HB_MC_SUCCESS) { 
                bsg_pr_err("failed to de-initialize device.\n");
                return rc;
        }

        // Dump the results. 

        for (int i = 0; i < result_size; i++) {
          bsg_pr_test_info("Result[%d] = 0x%08" PRIx32 "\n", i, result_host[i]);
        }

        // Compare the results. 

        int mismatch = 0; 
        for (int i = 0; i < result_size; i++) {
                if (list->test_vectors[idx]->reference[i] != result_host[i]) {
                        bsg_pr_err(BSG_RED("Mismatch: ") "Result[%d]: 0x%08" PRIx32 "\t Expected: 0x%08" PRIx32 "\n", i, result_host[i], list->test_vectors[idx]->reference[i]);
                        mismatch = 1;
                }
        } 

        if (mismatch) { 
                bsg_pr_err(BSG_RED("[FAILED]\n"));
                return HB_MC_FAIL;
        }
        bsg_pr_test_info(BSG_GREEN("[passed]\n"));
        return HB_MC_SUCCESS;
}

/*!
 * Runs a specific configuration on an 8x8 CGRA
 * Grid dimensions are prefixed at 1x1.
 * This tests uses the software/spmd/bsg_cuda_lite_runtime/cgra_xcel_hb_tapeout/ device
 * code in the bsg_manycore repository.
*/


int kernel_cgra_xcel_hb_tapeout (int argc, char **argv) {
        int rc, all_pass = 1;
        char *bin_path, *test_name;
        struct arguments_path args = {NULL, NULL};

        argp_parse (&argp_path, argc, argv, 0, 0, &args);
        bin_path = args.path;
        test_name = args.name;

        bsg_pr_test_info("Running the CUDA-Lite 8x8 CGRA vvadd Kernel.\n\n");

        srand(time); 

        //---------------------------------------------------------------------
        // Build test vector list
        //---------------------------------------------------------------------

        TestList list;

        bsg_pr_test_info("Building test vector list...\n\n");
        rc = build_test_list(&list);

        if ( rc < 0 ) {
          return HB_MC_FAIL;
        }

        //---------------------------------------------------------------------
        // Try to get test names from argv
        //---------------------------------------------------------------------

        if ( argc == 3 ) {
          bsg_pr_test_info("No extra C_ARGS specified. Run all test cases.\n\n");
          for (int i = 0; i < list.num_tests; i++) {
            rc = run_test(i, &list, bin_path, test_name);
            if ( rc == HB_MC_FAIL )
              all_pass = 0;
          }
        } else if ( argc > 3 ) {
          bsg_pr_test_info("Extra C_ARGS detected!\n\n");
          for (int arg_i = 3; arg_i < argc; arg_i++) {
              bsg_pr_test_info("Looking for %s...\n", argv[arg_i]);
              for (int i = 0; i < list.num_tests; i++) {
                if ( strcmp(argv[arg_i], list.test_names[i]) == 0 ) {
                rc = run_test(i, &list, bin_path, test_name);
                if ( rc == HB_MC_FAIL )
                  all_pass = 0;
              }
            }
          }
        }

        //---------------------------------------------------------------------
        // Finish
        //---------------------------------------------------------------------

        destory_test_list(&list);

        if ( all_pass )
          return HB_MC_SUCCESS;

        return HB_MC_FAIL;
}

#ifdef VCS
int vcs_main(int argc, char ** argv) {
#else
int main(int argc, char ** argv) {
#endif
        bsg_pr_test_info("test_cgra_xcel_hb_tapeout Regression Test\n");
        int rc = kernel_cgra_xcel_hb_tapeout(argc, argv);
        bsg_pr_test_pass_fail(rc == HB_MC_SUCCESS);
        return rc;
}
