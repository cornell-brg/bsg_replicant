#include "test_matrix_mul_shared_mem.h"

#define TEST_NAME "test_matrix_mul_shared_mem"

/*!
 * Runs the matrix multiplication with shared memory on a grid of 2x2 tile groups. A[M][N] * B[N][P] --> C[M][P]
 * Grid dimensions are determines by how much of a load we want for each tile group (block_size_y/x)
 * This tests uses the software/spmd/bsg_cuda_lite_runtime/matrix_mul_shared_mem/ Manycore binary in the dev_cuda_v4 branch of the BSG Manycore bitbucket repository.  
*/



/*! 
 * Matrix multiplication code on the host side to compare the results
 */
void matrix_mult (uint32_t *A, uint32_t *B, uint32_t *C, int M, int N, int P) { 
	for (int y = 0; y < M; y ++) { 
		for (int x = 0; x < P; x ++) { 
			int res = 0;
			for (int k = 0; k < N; k++) { 
				res += A[y * N + k] * B[k * P + x];
			}
			C[y * P + x] = res;
		}
	}
	return;
}
				


int kernel_matrix_mul_shared_mem () {
	fprintf(stderr, "Running the CUDA Matrix Multiplication With Shared Memory Kernel on a grid of 4 2x2 tile groups.\n\n");

	srand(time); 


	/*****************************************************************************************************************
	* Define the dimension of tile pool.
	* Define path to binary.
	* Initialize device, load binary and unfreeze tiles.
	******************************************************************************************************************/
	device_t device;
	hb_mc_dimension_t mesh_dim = { .x = 4, .y = 4 };
	hb_mc_device_init(&device, TEST_NAME, 0, mesh_dim);

	char* elf = BSG_STRINGIFY(BSG_MANYCORE_DIR) "/software/spmd/bsg_cuda_lite_runtime" "/matrix_mul_shared_mem/main.riscv";
	hb_mc_device_program_init(&device, elf);


	/*****************************************************************************************************************
	* Allocate memory on the device for A, B and C.
	******************************************************************************************************************/
	uint32_t M = 64;   
	uint32_t N = 128; 
	uint32_t P = 32; 

	eva_t A_device, B_device, C_device; 
	hb_mc_device_malloc(&device, M * N * sizeof(uint32_t), &A_device); /* allocate A[M][N] on the device */
	hb_mc_device_malloc(&device, N * P * sizeof(uint32_t), &B_device); /* allocate B[N][P] on the device */
	hb_mc_device_malloc(&device, M * P * sizeof(uint32_t), &C_device); /* allocate C[M][P] on the device */



	/*****************************************************************************************************************
	* Allocate memory on the host for A & B and initialize with random values.
	******************************************************************************************************************/
	uint32_t A_host[M * N]; /* allocate A[M][N] on the host */ 
	uint32_t B_host[N * P]; /* allocate B[N][P] on the host */
	for (int i = 0; i < M * N; i++) { /* fill A with arbitrary data */
		A_host[i] = rand() & 0xFFFF;
	}
	for (int i = 0; i < N * P; i++) { /* fill B with arbitrary data */
		B_host[i] = rand() & 0xFFFF;
	}




	/*****************************************************************************************************************
	* Copy A & B from host onto device DRAM.
	******************************************************************************************************************/
	void *dst = (void *) ((intptr_t) A_device);
	void *src = (void *) &A_host[0];
	hb_mc_device_memcpy (&device, dst, src, (M * N) * sizeof(uint32_t), hb_mc_memcpy_to_device); /* Copy A1 to the device  */	
	dst = (void *) ((intptr_t) B_device);
	src = (void *) &B_host[0];
	hb_mc_device_memcpy (&device, dst, src, (N * P) * sizeof(uint32_t), hb_mc_memcpy_to_device); /* Copy B1 to the device */ 



	/*****************************************************************************************************************
	* Define block_size_x/y: amount of work for each tile group
	* Define tg_dim_x/y: number of tiles in each tile group
	* Calculate grid_dim_x/y: number of tile groups needed based on block_size_x/y
	******************************************************************************************************************/
	uint32_t block_size_x = 16;
	uint32_t block_size_y = 16;

	hb_mc_dimension_t tg_dim = { .x = 2, .y = 2 };

	hb_mc_dimension_t grid_dim = { .x = P / block_size_x, .y = M / block_size_y };


	/*****************************************************************************************************************
	* Prepare list of input arguments for kernel.
	******************************************************************************************************************/
	int argv[8] = {A_device, B_device, C_device, M, N, P, block_size_y, block_size_x};

	/*****************************************************************************************************************
	* Enquque grid of tile groups, pass in grid and tile group dimensions, kernel name, number and list of input arguments
	******************************************************************************************************************/
	hb_mc_grid_init (&device, grid_dim, tg_dim, "kernel_matrix_mul_shared_mem", 8, argv);

	/*****************************************************************************************************************
	* Launch and execute all tile groups on device and wait for all to finish. 
	******************************************************************************************************************/
	hb_mc_device_tile_groups_execute(&device);
	


	/*****************************************************************************************************************
	* Copy result matrix back from device DRAM into host memory. 
	******************************************************************************************************************/
	uint32_t C_result[M * P];
	src = (void *) ((intptr_t) C_device);
	dst = (void *) &C_result[0];
	hb_mc_device_memcpy (&device, (void *) dst, src, (M * P) * sizeof(uint32_t), hb_mc_memcpy_to_host); /* copy C to the host */



	/*****************************************************************************************************************
	* Freeze the tiles and memory manager cleanup. 
	******************************************************************************************************************/
	hb_mc_device_finish(&device); 


	/*****************************************************************************************************************
	* Calculate the expected result matrix using host code and compare the results. 
	******************************************************************************************************************/
	uint32_t C_expected[M * P]; 
	matrix_mult (A_host, B_host, C_expected, M, N, P); 



	int mismatch = 0; 
	for (int i = 0; i < M * P; i++) {
		if ( C_expected[i] != C_result[i]) {
			mismatch = 1;
			break; 
		}
	}

	fprintf(stderr, "Expected Result:\n");
	for (int y = 0; y < M; y++) { 
		for (int x = 0; x < P; x++) {
			fprintf(stderr, "%d\t", C_expected[y * P + x]);
		}
		fprintf(stderr, "\n");
	}
		
	fprintf(stderr, "Manycore Result:\n");
	for (int y = 0; y < M; y++) { 
		for (int x = 0; x < P; x++) {
			fprintf(stderr, "%d\t", C_result[y * P + x]);
		}
		fprintf(stderr, "\n");
	}

	if (mismatch) { 
		fprintf(stderr, "Failed: matrix mismatch.\n");
		return HB_MC_FAIL;
	}
	fprintf(stderr, "Success: matrix match.\n");
	return HB_MC_SUCCESS;
}

#ifdef COSIM
void test_main(uint32_t *exit_code) {	
	bsg_pr_test_info("test_matrix_mul_shared_mem Regression Test (COSIMULATION)\n");
	int rc = kernel_matrix_mul_shared_mem();
	*exit_code = rc;
	bsg_pr_test_pass_fail(rc == HB_MC_SUCCESS);
	return;
}
#else
int main() {
	bsg_pr_test_info("test_matrix_mul_shared_mem Regression Test (F1)\n");
	int rc = kernel_matrix_mul_shared_mem();
	bsg_pr_test_pass_fail(rc == HB_MC_SUCCESS);
	return rc;
}
#endif

