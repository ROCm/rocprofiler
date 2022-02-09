# ROC-profiler
ROC profiler library. Profiling with perf-counters and derived metrics. Library supports GFX8/GFX9.

HW specific low-level performance analysis interface for profiling of GPU compute applications. The
profiling includes HW performance counters with complex performance metrics.

To use the rocProfiler API you need the API header and to link your application with roctracer .so librray:
 - the API header: /opt/rocm/rocprofiler/include/rocprofiler.h
 - the .so library: /opt/rocm/lib/librocprofiler64.so

## Documentation
- ['rocprof' cmdline tool specification](doc/rocprof.md)
- ['rocprofiler' profiling C API specification](doc/rocprofiler_spec.md)

## Metrics
[The link to profiler default metrics XML specification](test/tool/metrics.xml)


## Source tree
```
 - bin
   - rocprof - Profiling tool run script
 - doc - Documentation
 - inc/rocprofiler.h - Library public API
 - src  - Library sources
   - core - Library API sources
   - util - Library utils sources
   - xml - XML parser
 - test - Library test suite
   - tool - Profiling tool
     - tool.cpp - tool sources
     - metrics.xml - metrics config file
   - ctrl - Test controll
   - util - Test utils
   - simple_convolution - Simple convolution test kernel
```

## Build environment:
```
  export CMAKE_PREFIX_PATH=<path to hsa-runtime includes>:<path to hsa-runtime library>
  export CMAKE_BUILD_TYPE=<debug|release> # release by default
  export CMAKE_DEBUG_TRACE=1 # to enable debug tracing
```

## To build with the current installed ROCM:
```
 - ROCm is required.
  ROCr-runtime and roctracer are needed
  
 - Python is required.
  The required modules: CppHeaderParser, argparse, sqlite3
  To install:
  sudo pip install CppHeaderParser argparse sqlite3
 
 - To build and install to /opt/rocm/rocprofiler
  Please use release branches/tags of 'amd-master' branch for development version.
 
  export CMAKE_PREFIX_PATH=/opt/rocm/include/hsa:/opt/rocm

  cd .../rocprofiler
  ./build.sh
```

## Internal 'simple_convolution' test run script:
```
  cd .../rocprofiler/build
  make mytest
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

## Profiling utility usage:
```
rocprof [-h] [--list-basic] [--list-derived] [-i <input .txt/.xml file>] [-o <output CSV file>] <app command line>

Options:
  -h - this help
  --verbose - verbose mode, dumping all base counters used in the input metrics
  --list-basic - to print the list of basic HW counters
  --list-derived - to print the list of derived metrics with formulas
  --cmd-qts <on|off> - quoting profiled cmd-line [on]

  -i <.txt|.xml file> - input file
      Input file .txt format, automatically rerun application for every pmc line:

        # Perf counters group 1
        pmc : Wavefronts VALUInsts SALUInsts SFetchInsts FlatVMemInsts LDSInsts FlatLDSInsts GDSInsts FetchSize
        # Perf counters group 2
        pmc : VALUUtilization,WriteSize L2CacheHit
        # Filter by dispatches range, GPU index and kernel names
        # supported range formats: "3:9", "3:", "3"
        range: 1 : 4
        gpu: 0 1 2 3
        kernel: simple Pass1 simpleConvolutionPass2

      Input file .xml format, for single profiling run:

        # Metrics list definition, also the form "<block-name>:<event-id>" can be used
        # All defined metrics can be found in the 'metrics.xml'
        # There are basic metrics for raw HW counters and high-level metrics for derived counters
        <metric name=SQ:4,SQ_WAVES,VFetchInsts
        ></metric>

        # Filter by dispatches range, GPU index and kernel names
        <metric
          # range formats: "3:9", "3:", "3"
          range=""
          # list of gpu indexes "0,1,2,3"
          gpu_index=""
          # list of matched sub-strings "Simple1,Conv1,SimpleConvolution"
          kernel=""
        ></metric>

  -o <output file> - output CSV file [<input file base>.csv]
    The output CSV file columns meaning in the columns order:
      Index - kernels dispatch order index
      KernelName - the dispatched kernel name
      gpu-id - GPU id the kernel was submitted to
      queue-id - the ROCm queue unique id the kernel was submitted to
      queue-index - The ROCm queue write index for the submitted AQL packet
      tid - system application thread id which submitted the kernel
      grd - the kernel's grid size
      wgr - the kernel's work group size
      lds - the kernel's LDS memory size
      scr - the kernel's scratch memory size
      vgpr - the kernel's VGPR size
      sgpr - the kernel's SGPR size
      fbar - the kernel's barriers limitation
      sig - the kernel's completion signal
      ... - The columns with the counters values per kernel dispatch
      DispatchNs/BeginNs/EndNs/CompleteNs - timestamp columns if time-stamping was enabled
      
  -d <data directory> - directory where profiler store profiling data including thread treaces [/tmp]
      The data directory is renoving autonatically if the directory is matching the temporary one, which is the default.
  -t <temporary directory> - to change the temporary directory [/tmp]
      By changing the temporary directory you can prevent removing the profiling data from /tmp or enable removing from not '/tmp' directory.

  --basenames <on|off> - to turn on/off truncating of the kernel full function names till the base ones [off]
  --timestamp <on|off> - to turn on/off the kernel dispatches timestamps, dispatch/begin/end/complete [off]
    Four kernel timestamps in nanoseconds are reported:
        DispatchNs - the time when the kernel AQL dispatch packet was written to the queue
        BeginNs - the kernel execution begin time
        EndNs - the kernel execution end time
        CompleteNs - the time when the completion signal of the AQL dispatch packet was received

  --ctx-limit <max number> - maximum number of outstanding contexts [0 - unlimited]
  --heartbeat <rate sec> - to print progress heartbeats [0 - disabled]
  --obj-tracking <on|off> - to turn on/off kernels code objects tracking [on]
    To support V3 code-object.

  --stats - generating kernel execution stats, file <output name>.stats.csv
  
  --roctx-trace - to enable rocTX application code annotation trace, "Markers and Ranges" JSON trace section.
  --sys-trace - to trace HIP/HSA APIs and GPU activity, generates stats and JSON trace chrome-tracing compatible
  --hip-trace - to trace HIP, generates API execution stats and JSON file chrome-tracing compatible
  --hsa-trace - to trace HSA, generates API execution stats and JSON file chrome-tracing compatible
  --kfd-trace - to trace KFD, generates API execution stats and JSON file chrome-tracing compatible
    Generated files: <output name>.<domain>_stats.txt <output name>.json
    Traced API list can be set by input .txt or .xml files.
    Input .txt:
      hsa: hsa_queue_create hsa_amd_memory_pool_allocate
    Input .xml:
      <trace name="HSA">
        <parameters list="hsa_queue_create, hsa_amd_memory_pool_allocate">
        </parameters>
      </trace>

  --trace-start <on|off> - to enable tracing on start [on]
  --trace-period <dealy:length:rate> - to enable trace with initial delay, with periodic sample length and rate
    Supported time formats: <number(m|s|ms|us)>

Configuration file:
  You can set your parameters defaults preferences in the configuration file 'rpl_rc.xml'. The search path sequence: .:$HOME:<package path>
  First the configuration file is looking in the current directory, then in your home, and then in the package directory.
  Configurable options: 'basenames', 'timestamp', 'ctx-limit', 'heartbeat', 'obj-tracking'.
  An example of 'rpl_rc.xml':
    <defaults
      basenames=off
      timestamp=off
      ctx-limit=0
      heartbeat=0
      obj-tracking=on
    ></defaults>
```


## Known Issues:
- For workloads where the hip application might make more than 10 million HIP API calls, the application might crash with the error - "Profiling data corrupted"
  - Suggested Workaround - Instead of profiling for the complete run, it is suggested to run profiling in parts by using the --trace-period option.
- When the same kernel is launched back to back multiple times on a GPU, the cache hit rate from rocprofiler is reported as 0% or very low. This also causes FETCH_SIZE to be not usable for repeatable kernel.
