_REPO_ROOT ?= $(shell git rev-parse --show-toplevel)
-include $(_REPO_ROOT)/environment.mk
-include $(BSG_MACHINE_PATH)/Makefile.machine.include

RISCV_CXXFLAGS += -I$(BSG_MANYCORE_DIR)/software/spmd/applrts/
RISCV_CXXFLAGS += -I$(BSG_MANYCORE_DIR)/software/spmd/appl/
RISCV_CXXFLAGS += -I$(BSG_MANYCORE_DIR)/software/spmd/ligra/
RISCV_CXXFLAGS += -I$(BSG_MANYCORE_DIR)/software/spmd/applstatic/
BSG_ELF_STACK_PTR ?= 0x0003fffc

vpath %.cpp $(BSG_MANYCORE_DIR)/software/spmd/appl
vpath %.c   $(BSG_MANYCORE_DIR)/software/spmd/appl
vpath %.S   $(BSG_MANYCORE_DIR)/software/spmd/appl
vpath %.cpp $(BSG_MANYCORE_DIR)/software/spmd/applrts
vpath %.c   $(BSG_MANYCORE_DIR)/software/spmd/applrts
vpath %.S   $(BSG_MANYCORE_DIR)/software/spmd/applrts
vpath %.cpp $(BSG_MANYCORE_DIR)/software/spmd/ligra
vpath %.c   $(BSG_MANYCORE_DIR)/software/spmd/ligra
vpath %.S   $(BSG_MANYCORE_DIR)/software/spmd/ligra
vpath %.cpp $(BSG_MANYCORE_DIR)/software/spmd/applstatic
vpath %.c   $(BSG_MANYCORE_DIR)/software/spmd/applstatic
vpath %.S   $(BSG_MANYCORE_DIR)/software/spmd/applstatic

# APPL implemenation
ifeq ($(APPL_IMPL), APPL_IMPL_APPLRTS)
	RISCV_CXXFLAGS  +=-DAPPL_IMPL_APPLRTS
	RISCV_CXXFLAGS  +=-fno-rtti
	RISCV_TARGET_OBJECTS += appl-runtime.rvo
	RISCV_TARGET_OBJECTS += appl-malloc.rvo
	RISCV_TARGET_OBJECTS += applrts-config.rvo
	RISCV_TARGET_OBJECTS += applrts-runtime.rvo
	RISCV_TARGET_OBJECTS += applrts-scheduler.rvo
	RISCV_TARGET_OBJECTS += applrts-stats.rvo
endif

ifeq ($(APPL_IMPL), APPL_IMPL_SERIAL)
	RISCV_CXXFLAGS += -DAPPL_IMPL_SERIAL
	RISCV_TARGET_OBJECTS += appl-malloc.rvo
endif

ifeq ($(APPL_IMPL), APPL_IMPL_STATIC)
	RISCV_CXXFLAGS += -DAPPL_IMPL_STATIC
	RISCV_CXXFLAGS  +=-fno-rtti
	RISCV_TARGET_OBJECTS += appl-runtime.rvo
	RISCV_TARGET_OBJECTS += appl-malloc.rvo
	RISCV_TARGET_OBJECTS += applstatic-config.rvo
	RISCV_TARGET_OBJECTS += applstatic-runtime.rvo
endif

ifeq ($(APPL_IMPL), APPL_IMPL_CELLO)
	CELLO_DIR := $(CL_DIR)/../cello
	RISCV_CXXFLAGS +=-DAPPL_IMPL_CELLO
	RISCV_CXXFLAGS +=-I$(CELLO_DIR)/include
	RISCV_CXXFLAGS +=-I$(CELLO_DIR)/arch/hammerblade/include
	RISCV_CXXFLAGS += -fno-threadsafe-statics
	RISCV_TARGET_OBJECTS += appl-runtime.rvo
endif

# include riscv builddefs
include $(_REPO_ROOT)/examples/cuda/riscv.mk
RISCV_LD = $(_RISCV_GXX)
