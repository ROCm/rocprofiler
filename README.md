# ROC-profiler

ROC profiler library. Profiling with perf-counters and derived metrics. Library supports GFX8/GFX9.

HW specific low-level performance analysis interface for profiling of GPU compute applications. The profiling includes HW performance counters with complex performance metrics and HW traces

The library source tree:
 - doc  - Documentation
 - inc/rocprofiler.h - Library public API
 - src  - Library sources
   - core - Library API sources
   - util - Library utils sources
   - xml - XML parser
 - test - Library test suite
   - ctrl - Test controll
   - util - Test utils
   - simple_convolution - Simple convolution test kernel

## Build environment:
```
  export CMAKE_PREFIX_PATH=<path to hsa-runtime includes>:<path to hsa-runtime library>
  export CMAKE_BUILD_TYPE=<debug|release> # release by default
  export CMAKE_DEBUG_TRACE=1 # to enable debug tracing
```

## To build with the current installed ROCM:
```
  cd .../rocprofiler
  mkdir build
  cd build
  cmake -DCMAKE_PREFIX_PATH=/opt/rocm/lib:/opt/rocm/include/hsa ..
  make
```

## To run the test:
```
  cd .../rocprofiler/build
  export LD_LIBRARY_PATH=.:<other paths> # paths to ROC profiler and oher libraries
  export HSA_TOOLS_LIB=librocprofiler64.so # ROC profiler library loaded by HSA runtime
  export ROCP_TOOL_LIB=test/libtool.so # tool library loaded by ROC profiler
  export ROCP_METRICS=metrics.xml # ROC profiler metrics config file
  export ROCP_INPUT=input.xml # input file for the tool library
  export ROCP_OUTPUT_DIR=./ # output directory for the tool library, for metrics results file 'results.txt'
  <your test>
```

## Internal 'simple_convolution' test run script:
```
  cd .../rocprofiler/build
  run.sh
```

## To enable error messages logging to '/tmp/rocprofiler_log.txt':
```
  export ROCPROFILER_LOG=1
```

## To enable verbose tracing:
```
  export ROCPROFILER_TRACE=1
```
