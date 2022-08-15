REPLICANT_PATH = $(shell git rev-parse --show-toplevel)
MANYCORE_COMMIT = $(shell cd $(REPLICANT_PATH)/../bsg_manycore/ && git log -1 --pretty=format:"%h")

.PHONY: all
all: generate

# call to generate a test name
test-name = nqueens-in_$(1)__grain-size_$(2)__tgx_$(3)__tgy_$(4)__appl-impl_$(5)__mc_$(MANYCORE_COMMIT)

# call to get parameter from test name
get-nqueens-in = $(lastword $(subst _, ,$(filter nqueens-in_%,$(subst __, ,$(1)))))
get-grain-size = $(lastword $(subst _, ,$(filter grain-size_%,$(subst __, ,$(1)))))
get-tgx = $(lastword $(subst _, ,$(filter tgx_%,$(subst __, ,$(1)))))
get-tgy = $(lastword $(subst _, ,$(filter tgy_%,$(subst __, ,$(1)))))
get-appl-impl = $(lastword $(subst _, ,$(filter appl-impl_%,$(subst __, ,$(1)))))
get-manycore-commit = $(lastword $(subst _, ,$(filter mc_%,$(subst __, ,$(1)))))

# defines tests
TESTS =
include tests.mk

TESTS_DIRS = $(TESTS)

$(addsuffix /parameters.mk,$(TESTS_DIRS)): %/parameters.mk:
	@echo Creating $@
	@mkdir -p $(dir $@)
	@touch $@
	@echo test-name  = $* >> $@
	@echo nqueens-in = $(call get-nqueens-in,$*) >> $@
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

$(addsuffix /sbatch.sh,$(TESTS_DIRS)): %/sbatch.sh: sbatch.sh
	@echo Creating $@
	@mkdir -p $(dir $@)
	@sed 's/jOBnAme/$*/g' sbatch.sh > $@

$(TESTS_DIRS): %: %/app_path.mk
$(TESTS_DIRS): %: %/parameters.mk
$(TESTS_DIRS): %: %/Makefile
$(TESTS_DIRS): %: %/sbatch.sh

generate: $(TESTS_DIRS)

$(addsuffix .profile,$(TESTS)): %.profile: %
	$(MAKE) -C $< profile.log

$(addsuffix .exec,$(TESTS)): %.exec: %
	$(MAKE) -C $< exec.log

$(addsuffix .stats,$(TESTS)): %.stats: %
	$(MAKE) -C $< stats

$(addsuffix .batch-exec,$(TESTS)): %.batch-exec: %
	sbatch $</sbatch.sh exec.log

$(addsuffix .batch-profile,$(TESTS)): %.batch-profile: %
	sbatch $</sbatch.sh profile.log

$(addsuffix .batch-stats,$(TESTS)): %.batch-stats: %
	sbatch $</sbatch.sh stats

.PHONY: profile exec
profile: $(addsuffix .profile,$(TESTS))
exec:    $(addsuffix .exec,$(TESTS))
stats:   $(addsuffix .stats,$(TESTS))
batch-exec: $(addsuffix .batch-exec,$(TESTS))
batch-profile: $(addsuffix .batch-profile,$(TESTS))
batch-stats: $(addsuffix .batch-stats,$(TESTS))
purge:
	rm -rf $(TESTS_DIRS)
	rm -rf *$(MANYCORE_COMMIT).log

