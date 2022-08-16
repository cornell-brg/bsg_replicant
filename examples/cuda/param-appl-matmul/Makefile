REPLICANT_PATH = $(shell git rev-parse --show-toplevel)
MANYCORE_COMMIT = $(shell cd $(REPLICANT_PATH)/../bsg_manycore/ && git log -1 --pretty=format:"%h")

.PHONY: all
all: generate

# call to generate a test name
test-name = matrix-n_$(1)__tgx_$(2)__tgy_$(3)__appl-impl_$(4)__mc_$(MANYCORE_COMMIT)

# call to get parameter from test name
get-matrix-n = $(lastword $(subst _, ,$(filter matrix-n_%,$(subst __, ,$(1)))))
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
	@echo matrix-n = $(call get-matrix-n,$*) >> $@
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

$(addsuffix .pc-histogram,$(TESTS)): %.pc-histogram: %
	$(MAKE) -C $< pc-histogram.log

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
pc-histogram: $(addsuffix .pc-histogram,$(TESTS))
exec:    $(addsuffix .exec,$(TESTS))
stats:	 $(addsuffix .stats,$(TESTS))
batch-exec: $(addsuffix .batch-exec,$(TESTS))
batch-profile: $(addsuffix .batch-profile,$(TESTS))
batch-stats: $(addsuffix .batch-stats,$(TESTS))
purge:
	rm -rf $(TESTS_DIRS)
	rm -rf *$(MANYCORE_COMMIT).log


