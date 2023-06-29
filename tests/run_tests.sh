#!/bin/bash

CURRENT_DIR="$( dirname -- "$0"; )";

echo -e "Running Profiler Tests"

echo -e "running unit tests for rocprofiler"
eval ${CURRENT_DIR}/tests/unittests/runUnitTests

echo -e "running feature tests for rocprofiler"
eval ${CURRENT_DIR}/tests/featuretests/profiler/runFeatureTests

echo -e "Running Tracer Tests"
eval ${CURRENT_DIR}/tests/featuretests/tracer/runTracerFeatureTests