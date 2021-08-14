#!/bin/bash -x

################################################################################
# Copyright (c) 2018 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
################################################################################

# test filter input
test_filter=-1
if [ -n "$1" ] ; then
  test_filter=$1
fi

# test check routin
test_status=0
test_runnum=0
test_number=0
failed_tests="Failed tests:"

xeval_test() {
  test_number=$test_number
}

eval_test() {
  label=$1
  cmdline=$2
  if [ $test_filter = -1  -o $test_filter = $test_number ] ; then
    echo "$label: \"$cmdline\""
    test_runnum=$((test_runnum + 1))
    eval "$cmdline"
    if [ $? != 0 ] ; then
      echo "$label: FAILED"
      failed_tests="$failed_tests\n  $test_number: \"$label\""
      test_status=$(($test_status + 1))
    else
      echo "$label: PASSED"
    fi
  fi
  test_number=$((test_number + 1))
}

# paths to ROC profiler and oher libraries
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PWD:$PWD/../../lib:/home/jenkins/compute-package/lib

# enable tools load failure reporting
export HSA_TOOLS_REPORT_LOAD_FAILURE=1
# enable error messages logging to '/tmp/rocprofiler_log.txt'
export ROCPROFILER_LOG=1
# enable error messages logging to '/tmp/aql_profile_log.txt'
export HSA_VEN_AMD_AQLPROFILE_LOG=1
# test trace
export ROC_TEST_TRACE=1
# enable V3 code object support
export ROCP_OBJ_TRACKING=1

# Disabple profiler own proxy queue
unset ROCP_PROXY_QUEUE
# ROC profiler metrics config file
export ROCP_METRICS=metrics.xml

## C test
eval_test "C test" ./test/c_test

## Standalone sampling usage model test
unset HSA_TOOLS_LIB
unset ROCP_TOOL_LIB
eval_test "Standalone sampling usage model test" ./test/standalone_test
# Standalone intercepting test
# ROC profiler library loaded by HSA runtime
export HSA_TOOLS_LIB=librocprofiler64.so.1
# enable intercepting mode in rocprofiler
export ROCP_HSA_INTERCEPT=2
# test macro for kernel iterations number
export ROCP_KITER=20
# test macro for per-kernel dispatching number
export ROCP_DITER=10
eval_test "Standalone intercepting test" ./test/stand_intercept_test
unset ROCP_HSA_INTERCEPT

## Intercepting usage model test
# tool library loaded by ROC profiler
export ROCP_TOOL_LIB=./test/libintercept_test.so
export ROCP_KITER=20
export ROCP_DITER=20
export ROCP_AGENTS=1
export ROCP_THRS=3
eval_test "Intercepting usage model test" ./test/ctrl

## Libtool test
# tool library loaded by ROC profiler
export ROCP_TOOL_LIB=libtool.so
# ROC profiler kernels timing
export ROCP_TIMESTAMP_ON=1
# output directory for the tool library, for metrics results file 'results.txt'
export ROCP_OUTPUT_DIR=./RESULTS

if [ ! -e $ROCP_TOOL_LIB ] ; then
  export ROCP_TOOL_LIB=test/libtool.so
fi

export ROCP_KITER=20
export ROCP_DITER=20
export ROCP_AGENTS=1
export ROCP_THRS=1
export ROCP_INPUT=pmc_input.xml
eval_test "'rocprof' libtool PMC test" ./test/ctrl

export ROCP_KITER=20
export ROCP_DITER=20
export ROCP_AGENTS=1
export ROCP_THRS=10
export ROCP_INPUT=pmc_input.xml
eval_test "'rocprof' libtool PMC n-thread test" ./test/ctrl

export ROCP_OPT_MODE=1
export ROCP_KITER=20
export ROCP_DITER=20
export ROCP_AGENTS=1
export ROCP_THRS=10
export ROCP_INPUT=pmc_input.xml
eval_test "'rocprof' libtool PMC n-thread opt test" ./test/ctrl
unset ROCP_OPT_MODE

export ROCP_KITER=20
export ROCP_DITER=20
export ROCP_AGENTS=1
export ROCP_THRS=1
export ROCP_INPUT=pmc_input1.xml
eval_test "'rocprof' libtool PMC test1" ./test/ctrl

export ROCP_KITER=20
export ROCP_DITER=20
export ROCP_AGENTS=1
export ROCP_THRS=10
export ROCP_INPUT=pmc_input1.xml
eval_test "'rocprof' libtool PMC n-thread test1" ./test/ctrl

unset ROCP_MCOPY_TRACKING
# enable HSA intercepting
export ROCP_HSA_INTERC=1

export ROCP_KITER=10
export ROCP_DITER=10
eval_test "libtool test, counter sets" ./test/ctrl

## OpenCL test
#eval_test "libtool test, OpenCL sample" ./test/ocl/SimpleConvolution

#valgrind --leak-check=full $tbin
#valgrind --tool=massif $tbin
#ms_print massif.out.<N>

echo "$test_number tests total / $test_runnum tests run / $test_status tests failed"
if [ $test_status != 0 ] ; then
  echo $failed_tests
fi
exit $test_status
