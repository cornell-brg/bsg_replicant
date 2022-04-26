REPLICANT_PATH = $(shell git rev-parse --show-toplevel)
BENCHMARK  ?= unnamed_benchmark
PARAMETERS ?= p0 p1 p2

.PHONY: all
all: $(BENCHMARK)

$(BENCHMARK): $(BENCHMARK)/Makefile
$(BENCHMARK): $(BENCHMARK)/template.mk
$(BENCHMARK): $(BENCHMARK)/app_path.mk
$(BENCHMARK): $(BENCHMARK)/tests.mk
$(BENCHMARK): $(BENCHMARK)/main.c
$(BENCHMARK): $(BENCHMARK)/kernel.cpp
$(BENCHMARK): $(BENCHMARK)/.gitignore

$(BENCHMARK)/Makefile: $(REPLICANT_PATH)/py/benchmark_makefile_gen.py
	mkdir -p $(dir $@)
	python3 $< $(PARAMETERS) > $@

$(BENCHMARK)/template.mk: $(REPLICANT_PATH)/mk/benchmark_template.mk
	mkdir -p $(dir $@)
	cp $< $@

$(BENCHMARK)/app_path.mk:
	mkdir -p $(dir $@)
	echo 'APP_PATH = $$(REPLICANT_PATH)/examples/cuda/$(BENCHMARK)' > $@

$(BENCHMARK)/tests.mk:
	mkdir -p $(dir $@)
	echo "# TESTS += $$(call test-name,p0,p1,...)" > $@

$(BENCHMARK)/main.c:
	mkdir -p $(dir $@)
	touch $@

$(BENCHMARK)/kernel.cpp:
	mkdir -p $(dir $@)
	touch $@

$(BENCHMARK)/.gitignore:
	mkdir -p $(dir $@)
	echo "*/" > $@
