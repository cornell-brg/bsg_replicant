#ifndef BSG_MANYCORE_DRIVER_H
#define BSG_MANYCORE_DRIVER_H

#ifndef _BSD_SOURCE
	#define _BSD_SOURCE
#endif
#ifndef _XOPEN_SOURCE
	#define _XOPEN_SOURCE 500
#endif


#include <stdint.h>
#include <stdbool.h>
#include <limits.h>

static char *hb_mc_mmap_ocl (uint8_t fd);
static void hb_mc_write (uint8_t fd, uint32_t ofs, uint32_t val, uint8_t reg_size);
static uint32_t hb_mc_read (uint8_t fd, uint32_t ofs, uint8_t reg_size);
static bool hb_mc_check_fd (uint8_t fd);
bool hb_mc_init_host (char *dev_path, uint8_t *fd);
void hb_mc_close_host (uint8_t fd); 
bool hb_mc_check_dim (uint8_t fd);      
bool hb_mc_write_fifo (uint8_t fd, uint8_t n, uint32_t *val);
uint32_t *hb_mc_read_fifo (uint8_t fd, uint8_t n, uint32_t *val);
void hb_mc_clear_int (uint8_t fd, uint8_t n);
uint32_t hb_mc_get_host_credits (uint8_t fd);
bool hb_mc_all_host_req_complete(uint8_t fd);
uint32_t hb_mc_get_recv_vacancy (uint8_t fd);
bool hb_mc_can_read (uint8_t fd, uint32_t size);
bool hb_mc_check_device (uint8_t fd);

extern char *MANYCORE_DEVICE_PATH;
 
static uint8_t NUM_X = 4; /*! Number of columns of tiles. */

extern uint8_t NUM_Y;  /*! Number of rows of tiles */

/*
 * packet format: {addr, op, op_ex, data, src_y_cord, src_x_cord, y_cord, x_cord)
 * */

/*!
 * Helper function that gets bits of an int.
 * @param data value to get bits from. 
 * @param start starting bit. 
 * @param size number of bits to retrieve.
 * @return desired bits of data. They are right-shifted to the LSB.
 * */



static const uint8_t NUM_FIFO = 2; /* Make sure to change HOST_RECV_VACANCY, HOST_CREDITS */

/* fd[i] = [fd][char *ocl_base] of ith device */
static uint8_t num_dev = 0;
static uint32_t fd_table[8] = {-1, -1, -1, -1, -1, -1, -1, -1};	
static char *ocl_table[8] = {(char *) 0, (char *) 0, (char *) 0, (char *) 0, (char *) 0, (char *) 0, (char *) 0, (char *) 0};	

static const uint32_t fifo[10][8] = {{0xC, 0x10, 0x14 , 0x1C, 0x20, 0x24, 0x0, 0x4} 
						, {0xC + 0x100, 0x10 + 0x100, 0x14 + 0x100, 0x1C + 0x100, 0x20 + 0x100, 0x24 + 0x100, 0x0 + 0x100, 0x4 + 0x100} 
						, {0xC + 0x200, 0x10 + 0x200, 0x14 + 0x200, 0x1C + 0x200, 0x20 + 0x200, 0x24 + 0x200, 0x0 + 0x200, 0x4 + 0x200}
						, {0xC + 0x300, 0x10 + 0x300, 0x14 + 0x300, 0x1C + 0x300, 0x20 + 0x300, 0x24 + 0x300, 0x0 + 0x300, 0x4 + 0x300} 
						, {0xC + 0x400, 0x10 + 0x400, 0x14 + 0x400, 0x1C + 0x400, 0x20 + 0x400, 0x24 + 0x400, 0x0 + 0x400, 0x4 + 0x400} 
						, {0xC + 0x500, 0x10 + 0x500, 0x14 + 0x500, 0x1C + 0x500, 0x20 + 0x500, 0x24 + 0x500, 0x0 + 0x500, 0x4 + 0x500} 
						, {0xC + 0x600, 0x10 + 0x600, 0x14 + 0x600, 0x1C + 0x600, 0x20 + 0x600, 0x24 + 0x600, 0x0 + 0x600, 0x4 + 0x600} 
						, {0xC + 0x700, 0x10 + 0x700, 0x14 + 0x700, 0x1C + 0x700, 0x20 + 0x700, 0x24 + 0x700, 0x0 + 0x700, 0x4 + 0x700} 
						, {0xC + 0x800, 0x10 + 0x800, 0x14 + 0x800, 0x1C + 0x800, 0x20 + 0x800, 0x24 + 0x800, 0x0 + 0x800, 0x4 + 0x800} 
						, {0xC + 0x900, 0x10 + 0x900, 0x14 + 0x900, 0x1C + 0x900, 0x20 + 0x900, 0x24 + 0x900, 0x0 + 0x900, 0x4 + 0x900}}; 

 
static const uint8_t FIFO_VACANCY = 0, FIFO_WRITE = 1, FIFO_TRANSMIT_LENGTH = 2, FIFO_OCCUPANCY = 3, FIFO_READ = 4, FIFO_RECEIVE_LENGTH = 5, FIFO_ISR = 6, FIFO_IER = 7; 

static const uint32_t HOST_RECV_VACANCY = 0x200;
static const uint32_t HOST_CREDITS = 2 * 0x100 + 0x10;
static const uint32_t MANYCORE_NUM_X = 0x220;
static const uint32_t MANYCORE_NUM_Y = 0x224;

static const uint32_t MAX_CREDITS = 16;

#endif