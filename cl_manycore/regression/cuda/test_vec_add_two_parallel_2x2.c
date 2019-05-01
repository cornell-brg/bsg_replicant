#include "test_vec_add_two_parallel_2x2.h"

/*!
 * Runs the addition kernel on a 2x2 tile group at (0, 1). 
 * This tests uses the software/spmd/bsg_cuda_lite_runtime/vec_add_2x2/ Manycore binary in the dev_cuda_tile_group_refactored branch of the BSG Manycore bitbucket repository.  
*/
int kernel_vec_add () {
	fprintf(stderr, "Running the CUDA Vector Addition Kernel on two 2x2 tile groups in parallel.\n\n");

	device_t device;
	uint8_t grid_dim_x = 4;
	uint8_t grid_dim_y = 4;
	uint8_t grid_origin_x = 0;
	uint8_t grid_origin_y = 1;
	eva_id_t eva_id = 0;
	char* ELF_PATH = BSG_STRINGIFY(BSG_MANYCORE_DIR) "/software/spmd/bsg_cuda_lite_runtime" "/vec_add_two_parallel_2x2/main.riscv";

	hb_mc_device_init(&device, eva_id, ELF_PATH, grid_dim_x, grid_dim_y, grid_origin_x, grid_origin_y);


	tile_group_t tg_1; 
	tile_group_id_t tg_id_1 = 0;
	uint8_t tg_dim_x_1 = 2;
	uint8_t tg_dim_y_1 = 2;
	hb_mc_tile_group_allocate(&device, &tg_1, tg_id_1, tg_dim_x_1, tg_dim_y_1); 


	tile_group_t tg_2; 
	tile_group_id_t tg_id_2 = 1;
	uint8_t tg_dim_x_2 = 2;
	uint8_t tg_dim_y_2 = 2;
	hb_mc_tile_group_allocate(&device, &tg_2, tg_id_2, tg_dim_x_2, tg_dim_y_2); 



	uint32_t size_buffer = 8; 
	eva_t A_device_1, B_device_1, C_device_1; 
	hb_mc_device_malloc(&device, size_buffer * sizeof(uint32_t), &A_device_1); /* allocate A on the device */
	hb_mc_device_malloc(&device, size_buffer * sizeof(uint32_t), &B_device_1); /* allocate B on the device */
	hb_mc_device_malloc(&device, size_buffer * sizeof(uint32_t), &C_device_1); /* allocate C on the device */

	uint32_t A_host_1[size_buffer]; /* allocate A on the host */ 
	uint32_t B_host_1[size_buffer]; /* allocate B on the host */
	srand(0);
	for (int i = 0; i < size_buffer; i++) { /* fill A and B with arbitrary data */
		A_host_1[i] = rand() % ((1 << 16) - 1); /* avoid overflow */
		B_host_1[i] = rand() % ((1 << 16) - 1); 
	}

	void *dst = (void *) ((intptr_t) A_device_1);
	void *src = (void *) &A_host_1[0];
	hb_mc_device_memcpy (&device, dst, src, size_buffer * sizeof(uint32_t), hb_mc_memcpy_to_device); /* Copy A1 to the device  */	
	dst = (void *) ((intptr_t) B_device_1);
	src = (void *) &B_host_1[0];
	hb_mc_device_memcpy (&device, dst, src, size_buffer * sizeof(uint32_t), hb_mc_memcpy_to_device); /* Copy B2 to the device */ 





	eva_t A_device_2, B_device_2, C_device_2; 
	hb_mc_device_malloc(&device, size_buffer * sizeof(uint32_t), &A_device_2); /* allocate A on the device */
	hb_mc_device_malloc(&device, size_buffer * sizeof(uint32_t), &B_device_2); /* allocate B on the device */
	hb_mc_device_malloc(&device, size_buffer * sizeof(uint32_t), &C_device_2); /* allocate C on the device */

	uint32_t A_host_2[size_buffer]; /* allocate A on the host */ 
	uint32_t B_host_2[size_buffer]; /* allocate B on the host */
	for (int i = 0; i < size_buffer; i++) { /* fill A and B with arbitrary data */
		A_host_2[i] = rand() % ((1 << 16) - 1); /* avoid overflow */
		B_host_2[i] = rand() % ((1 << 16) - 1); 
	}

	dst = (void *) ((intptr_t) A_device_2);
	src = (void *) &A_host_2[0];
	hb_mc_device_memcpy (&device, dst, src, size_buffer * sizeof(uint32_t), hb_mc_memcpy_to_device); /* Copy A to the device  */	
	dst = (void *) ((intptr_t) B_device_2);
	src = (void *) &B_host_2[0];
	hb_mc_device_memcpy (&device, dst, src, size_buffer * sizeof(uint32_t), hb_mc_memcpy_to_device); /* Copy B to the device */ 



	int argv_1[4] = {A_device_1, B_device_1, C_device_1, size_buffer / (tg_1.dim_x * tg_1.dim_y)};
	uint32_t finish_signal_addr_1 = 0xC0DA;

	int argv_2[4] = {A_device_2, B_device_2, C_device_2, size_buffer / (tg_2.dim_x * tg_2.dim_y)};
	uint32_t finish_signal_addr_2 = 0xC0DB;



	hb_mc_tile_group_init (&device, &tg_1, "kernel_vec_add", 4, argv_1, finish_signal_addr_1);
	hb_mc_tile_group_init (&device, &tg_2, "kernel_vec_add", 4, argv_2, finish_signal_addr_2);

	hb_mc_tile_group_launch(&device, &tg_1); 
	hb_mc_tile_group_launch(&device, &tg_2); 

	
	hb_mc_tile_group_sync(&device, &tg_1);
	hb_mc_tile_group_sync(&device, &tg_2);


	hb_mc_tile_group_deallocate(&device, &tg_1); 
	hb_mc_tile_group_deallocate(&device, &tg_2);
	




	uint32_t C_host_1[size_buffer];
	src = (void *) ((intptr_t) C_device_1);
	dst = (void *) &C_host_1[0];
	hb_mc_device_memcpy (&device, (void *) dst, src, size_buffer * sizeof(uint32_t), hb_mc_memcpy_to_host); /* copy C to the host */

	uint32_t C_host_2[size_buffer];
	src = (void *) ((intptr_t) C_device_2);
	dst = (void *) &C_host_2[0];
	hb_mc_device_memcpy (&device, (void *) dst, src, size_buffer * sizeof(uint32_t), hb_mc_memcpy_to_host); /* copy C to the host */



		

	int mismatch = 0; 
	for (int i = 0; i < size_buffer; i++) {
		if (A_host_1[i] + B_host_1[i] == C_host_1[i]) {
			fprintf(stderr, "Success -- A1[%d] + B1[%d] =  0x%x + 0x%x = 0x%x\n", i, i , A_host_1[i], B_host_1[i], C_host_1[i]);
		}
		else {
			fprintf(stderr, "Failed -- A1[%d] + B1[%d] =  0x%x + 0x%x != 0x%x\n", i, i , A_host_1[i], B_host_1[i], C_host_1[i]);
			mismatch = 1;
		}
	}	



	for (int i = 0; i < size_buffer; i++) {
		if (A_host_2[i] + B_host_2[i] == C_host_2[i]) {
			fprintf(stderr, "Success -- A2[%d] + B2[%d] =  0x%x + 0x%x = 0x%x\n", i, i , A_host_2[i], B_host_2[i], C_host_2[i]);
		}
		else {
			fprintf(stderr, "Failed -- A2[%d] + B2[%d] =  0x%x + 0x%x != 0x%x\n", i, i , A_host_2[i], B_host_2[i], C_host_2[i]);
			mismatch = 1;
		}
	}	


	hb_mc_device_finish(&device); /* freeze the tiles and memory manager cleanup */
	

	if (mismatch)
		return HB_MC_FAIL;
	return HB_MC_SUCCESS;
}

#ifdef COSIM
void test_main(uint32_t *exit_code) {	
	bsg_pr_test_info("test_bsg_cuda_lite_runtime_vec_add Regression Test (COSIMULATION)\n");
	int rc = kernel_vec_add();
	*exit_code = rc;
	bsg_pr_test_pass_fail(rc == HB_MC_SUCCESS);
	return;
}
#else
int main() {
	bsg_pr_test_info("test_bsg_cuda_lite_runtime_vec_add Regression Test (F1)\n");
	int rc = kernel_vec_add();
	bsg_pr_test_pass_fail(rc == HB_MC_SUCCESS);
	return rc;
}
#endif
