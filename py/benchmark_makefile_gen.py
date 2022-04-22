import os
import sys


parameters = sys.argv[1:]

test_name_str = 'test-name = ' + '__'.join(
    ["{}_$({})".format(parameter, pidx+1) for (pidx,parameter) in enumerate(parameters)]
)

parameter_getters = '\n'.join([
    'get-{} = $(lastword $(subst _, ,$(filter {}_%,$(subst __, ,$(1)))))'.format(
        parameter,parameter
    ) for parameter in parameters
])

create_parameters_mk = '\n'.join([
    "\t@echo {} = $(call get-{},$*) >> $@".format(parameter, parameter)
    for parameter in parameters
])

makefile_str = """REPLICANT_PATH = $(shell git rev-parse --show-toplevel)

.PHONY: all
all: generate

# call to generate a test name
{define_test_name}

# call to get parameter from test name
{define_parameter_getters}

# defines tests
TESTS =
include tests.mk

TESTS_DIRS = $(TESTS)

$(addsuffix /parameters.mk,$(TESTS_DIRS)): %/parameters.mk:
\t@echo Creating $@
\t@mkdir -p $(dir $@)
\t@touch $@
\t@echo test-name  = $* >> $@
{create_parameters_mk}

$(addsuffix /app_path.mk,$(TESTS_DIRS)): %/app_path.mk: app_path.mk
\t@echo Creating $@
\t@mkdir -p $(dir $@)
\t@cp $< $@

$(addsuffix /Makefile,$(TESTS_DIRS)): %/Makefile: template.mk
\t@echo Creating $@
\t@mkdir -p $(dir $@)
\t@cp template.mk $@

$(TESTS_DIRS): %: %/app_path.mk
$(TESTS_DIRS): %: %/parameters.mk
$(TESTS_DIRS): %: %/Makefile

generate: $(TESTS_DIRS)

$(addsuffix .profile,$(TESTS)): %.profile: %
	$(MAKE) -C $< profile.log

$(addsuffix .exec,$(TESTS)): %.exec: %
	$(MAKE) -C $< exec.log

.PHONY: profile exec
profile: $(addsuffix .profile,$(TESTS))
exec:    $(addsuffix .exec,$(TESTS))
purge:
\trm -rf $(TESTS_DIRS)

""".format(
    define_test_name=test_name_str
    ,define_parameter_getters=parameter_getters
    ,create_parameters_mk=create_parameters_mk
)

print(makefile_str)
