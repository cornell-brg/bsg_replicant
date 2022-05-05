REPLICANT_PATH = $(shell git rev-parse --show-toplevel)

.PHONY: all
all: generate

# call to generate a test name
test-name = graph-input_$(1)__iter_$(2)__grain-size_$(3)__tgx_$(4)__tgy_$(5)__appl-impl_$(6)

# call to get parameter from test name
get-graph-input = $(lastword $(subst _, ,$(filter graph-input_%,$(subst __, ,$(1)))))
get-iter = $(lastword $(subst _, ,$(filter iter_%,$(subst __, ,$(1)))))
get-grain-size = $(lastword $(subst _, ,$(filter grain-size_%,$(subst __, ,$(1)))))
get-tgx = $(lastword $(subst _, ,$(filter tgx_%,$(subst __, ,$(1)))))
get-tgy = $(lastword $(subst _, ,$(filter tgy_%,$(subst __, ,$(1)))))
get-appl-impl = $(lastword $(subst _, ,$(filter appl-impl_%,$(subst __, ,$(1)))))

# defines tests
TESTS =
include tests.mk

TESTS_DIRS = $(TESTS)

$(addsuffix /parameters.mk,$(TESTS_DIRS)): %/parameters.mk:
	@echo Creating $@
	@mkdir -p $(dir $@)
	@touch $@
	@echo test-name  = $* >> $@
	@echo graph-input = $(call get-graph-input,$*) >> $@
	@echo iter = $(call get-iter,$*) >> $@
	@echo grain-size = $(call get-grain-size,$*) >> $@
	@echo tgx = $(call get-tgx,$*) >> $@
	@echo tgy = $(call get-tgy,$*) >> $@
	@echo appl-impl = $(call get-appl-impl,$*) >> $@

$(addsuffix /app_path.mk,$(TESTS_DIRS)): %/app_path.mk: app_path.mk
	@echo Creating $@
	@mkdir -p $(dir $@)
	@cp $< $@

$(addsuffix /Makefile,$(TESTS_DIRS)): %/Makefile: template.mk
	@echo Creating $@
	@mkdir -p $(dir $@)
	@cp template.mk $@

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
	rm -rf $(TESTS_DIRS)


