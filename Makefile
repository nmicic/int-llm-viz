# Copyright 2026 Nenad Mićić
# SPDX-License-Identifier: Apache-2.0
#
# Build pipeline: C trace harness -> data/*.json -> self-contained site/ pages.
# Every product is deterministic; `make test` proves it by rebuilding twice.

CC      ?= cc
CFLAGS  ?= -O2 -std=c11 -Wall -Wextra
PYTHON  ?= python3

MODEL   := third_party/int-llm/model.mgw
RUNTIME := third_party/int-llm/microgpt_int.c third_party/int-llm/fp_math.h

.PHONY: all pages check test release-check clean

all: pages check

build/trace_harness: tools/trace_harness.c $(RUNTIME)
	mkdir -p build
	$(CC) $(CFLAGS) -o $@ tools/trace_harness.c -lm

data/trace.json: build/trace_harness $(MODEL)
	./build/trace_harness $(MODEL) $@

data/geometry.json: tools/derive_geometry.py $(MODEL)
	$(PYTHON) tools/derive_geometry.py

data/layouts.json: tools/derive_layouts.py $(MODEL) data/trace.json \
                   build/trace_harness third_party/layout-set/summary.json \
                   $(wildcard third_party/layout-set/*/mlp-order.json)
	$(PYTHON) tools/derive_layouts.py

data/hidden.json: tools/derive_hidden.py $(MODEL) data/trace.json \
                  build/trace_harness \
                  third_party/hidden-swap/hidden-order.json \
                  third_party/hidden-swap/viz-manifest.json
	$(PYTHON) tools/derive_hidden.py

pages: data/trace.json data/geometry.json data/layouts.json \
       data/hidden.json tools/build_site.py \
       web/shared/theme.css web/shared/mgw.js $(wildcard web/*.html)
	$(PYTHON) tools/build_site.py

check:
	$(PYTHON) tools/check_site.py

test:
	bash tests/run_tests.sh

# Release gate: the browser step must RUN (not skip), and regeneration
# must reproduce the committed data/ and site/ byte-for-byte.
release-check:
	REQUIRE_BROWSER=1 bash tests/run_tests.sh
	git diff --exit-code -- data site
	@echo "release-check: suite green (browser required), no regeneration drift"

clean:
	rm -rf build
