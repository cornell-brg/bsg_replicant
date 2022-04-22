# all tests run with grain size = 1
# 2x2 tests
TESTS += $(call test-name,1,1,2,2)
TESTS += $(call test-name,2,1,2,2)
TESTS += $(call test-name,3,1,2,2)
TESTS += $(call test-name,4,1,2,2)
TESTS += $(call test-name,5,1,2,2)
# 16x8 tests
TESTS += $(call test-name,1,1,16,8)
TESTS += $(call test-name,2,1,16,8)
TESTS += $(call test-name,3,1,16,8)
TESTS += $(call test-name,4,1,16,8)
TESTS += $(call test-name,5,1,16,8)
