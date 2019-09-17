# This Makefile fragment defines all of the regression tests (and the
# source path) for this sub-directory.

# Makefile.machine.include defines the Manycore hardware
# configuration.
include $(CL_DIR)/Makefile.machine.include

REGRESSION_TESTS_TYPE = python
SRC_PATH=$(REGRESSION_PATH)/$(REGRESSION_TESTS_TYPE)/

# "Unified tests" all use the generic test top-level:
# test_unified_main.c
UNIFIED_TESTS = test_python

# "Independent Tests" use a per-test <test_name>.c file
INDEPENDENT_TESTS := 

# REGRESSION_TESTS is a list of all regression tests to run.
REGRESSION_TESTS = $(UNIFIED_TESTS) $(INDEPENDENT_TESTS)

PYTHON_TEST_PATH = $(abspath $(BSG_F1_DIR)/regression/python/)

DEFINES += -DBSG_MANYCORE_DIR=$(abspath $(BSG_MANYCORE_DIR))
DEFINES += -DCL_MANYCORE_DIM_X=$(CL_MANYCORE_DIM_X) 
DEFINES += -DCL_MANYCORE_DIM_Y=$(CL_MANYCORE_DIM_Y)
DEFINES += -DBSG_PYTHON_TEST_PATH=$(PYTHON_TEST_PATH)
DEFINES += -D_XOPEN_SOURCE=500 -D_BSD_SOURCE

CDEFINES   += $(DEFINES)
CXXDEFINES += $(DEFINES)

FLAGS     = -g -Wall $(shell python3.6-config --cflags) -O1 
CFLAGS   += -std=c99 $(FLAGS)
CXXFLAGS += -std=c++11 $(FLAGS) 
LDFLAGS  += $(shell python3.6-config --ldflags) 