# all tests run with grain size = 1
# APPLRTS
# 2x2 tests
TESTS += $(call test-name,1,1,2,2,APPLRTS)
TESTS += $(call test-name,2,1,2,2,APPLRTS)
TESTS += $(call test-name,3,1,2,2,APPLRTS)
TESTS += $(call test-name,4,1,2,2,APPLRTS)
TESTS += $(call test-name,5,1,2,2,APPLRTS)
# 16x8 tests
TESTS += $(call test-name,1,1,16,8,APPLRTS)
TESTS += $(call test-name,2,1,16,8,APPLRTS)
TESTS += $(call test-name,3,1,16,8,APPLRTS)
TESTS += $(call test-name,4,1,16,8,APPLRTS)
TESTS += $(call test-name,5,1,16,8,APPLRTS)

# CELLO
# 2x2 tests
TESTS += $(call test-name,1,1,2,2,CELLO)
TESTS += $(call test-name,2,1,2,2,CELLO)
TESTS += $(call test-name,3,1,2,2,CELLO)
TESTS += $(call test-name,4,1,2,2,CELLO)
TESTS += $(call test-name,5,1,2,2,CELLO)
# 16x8 tests
TESTS += $(call test-name,1,1,16,8,CELLO)
TESTS += $(call test-name,2,1,16,8,CELLO)
TESTS += $(call test-name,3,1,16,8,CELLO)
TESTS += $(call test-name,4,1,16,8,CELLO)
TESTS += $(call test-name,5,1,16,8,CELLO)

# SERIAL
# 2x2 tests
TESTS += $(call test-name,1,1,2,2,SERIAL)
TESTS += $(call test-name,2,1,2,2,SERIAL)
TESTS += $(call test-name,3,1,2,2,SERIAL)
TESTS += $(call test-name,4,1,2,2,SERIAL)
TESTS += $(call test-name,5,1,2,2,SERIAL)
# 16x8 tests
TESTS += $(call test-name,1,1,16,8,SERIAL)
TESTS += $(call test-name,2,1,16,8,SERIAL)
TESTS += $(call test-name,3,1,16,8,SERIAL)
TESTS += $(call test-name,4,1,16,8,SERIAL)
TESTS += $(call test-name,5,1,16,8,SERIAL)
