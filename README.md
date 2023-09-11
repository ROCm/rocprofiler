## DISCLAIMER

The information presented in this document is for informational purposes only and may contain technical inaccuracies, omissions, and typographical errors. The information contained herein is subject to change and may be rendered inaccurate for many reasons, including but not limited to product and roadmap changes, component and motherboard version changes, new model and/or product releases, product differences between differing manufacturers, software changes, BIOS flashes, firmware upgrades, or the like. Any computer system has risks of security vulnerabilities that cannot be completely prevented or mitigated. AMD assumes no obligation to update or otherwise correct or revise this information. However, AMD reserves the right to revise this information and to make changes from time to time to the content hereof without obligation of AMD to notify any person of such revisions or changes.THIS INFORMATION IS PROVIDED ‘AS IS.” AMD MAKES NO REPRESENTATIONS OR WARRANTIES WITH RESPECT TO THE CONTENTS HEREOF AND ASSUMES NO RESPONSIBILITY FOR ANY INACCURACIES, ERRORS, OR OMISSIONS THAT MAY APPEAR IN THIS INFORMATION. AMD SPECIFICALLY DISCLAIMS ANY IMPLIED WARRANTIES OF NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR ANY PARTICULAR PURPOSE. IN NO EVENT WILL AMD BE LIABLE TO ANY PERSON FOR ANY RELIANCE, DIRECT, INDIRECT, SPECIAL, OR OTHER CONSEQUENTIAL DAMAGES ARISING FROM THE USE OF ANY INFORMATION CONTAINED HEREIN, EVEN IF AMD IS EXPRESSLY ADVISED OF THE POSSIBILITY OF SUCH DAMAGES. AMD, the AMD Arrow logo, and combinations thereof are trademarks of Advanced Micro Devices, Inc. Other product names used in this publication are for identification purposes only and may be trademarks of their respective companies.

© 2022 Advanced Micro Devices, Inc. All Rights Reserved.

## ROC Profiler v1

## Introduction

Profiling with metrics and traces based on perfcounters (PMC) and traces (SPM).
Implementation is based on AqlProfile HSA extension.
Library supports GFX8/GFX9.
The last API library version for ROCProfiler v1 is 8.0.0

The library source tree:

- doc  - Documentation
- include/rocprofiler/rocprofiler.h - Library public API
- include/rocprofiler/v2/rocprofiler.h - V2 Beta Library public API
- include/rocprofiler/v2/rocprofiler_plugins.h - V2 Beta Tool's Plugins Library public API
- src  - Library sources
  - core - Library API sources
  - util - Library utils sources
  - xml - XML parser
- test - Library test suite
  - ctrl - Test controll
  - util - Test utils
  - simple_convolution - Simple convolution test kernel

## Build environment

Roctracer & Rocprofiler need to be installed in the same directory.

```bash
export CMAKE_PREFIX_PATH=<path to hsa-runtime includes>:<path to hsa-runtime library>
export CMAKE_BUILD_TYPE=<debug|release> # release by default
export CMAKE_DEBUG_TRACE=1 # 1 to enable debug tracing
```

To build with the current installed ROCM:

 ```bash
cd .../rocprofiler
./build.sh ## (for clean build use `-cb`)
```

To run the test:

```bash
$ cd .../rocprofiler/build
$ export LD_LIBRARY_PATH=.:<other paths> # paths to ROC profiler and oher libraries
$ export HSA_TOOLS_LIB=librocprofiler64.so.1 # ROC profiler library loaded by HSA runtime
$ export ROCP_TOOL_LIB=test/librocprof-tool.so # tool library loaded by ROC profiler
$ export ROCP_METRICS=metrics.xml # ROC profiler metrics config file
$ export ROCP_INPUT=input.xml # input file for the tool library
$ export ROCP_OUTPUT_DIR=./ # output directory for the tool library, for metrics results file 'results.txt' and trace files
$ <your test>

Internal 'simple_convolution' test run script:
$ cd .../rocprofiler/build
$ ./run.sh

To enabled error messages logging to '/tmp/rocprofiler_log.txt':

$ export ROCPROFILER_LOG=1

To enable verbose tracing:

$ export ROCPROFILER_TRACE=1
```

## ROCProfiler v2

## Introduction

ROCProfilerV2 is a newly developed design for AMD’s tooling infrastructure that provides a hardware specific low level performance analysis interface for profiling of GPU compute applications.
The first API library version for ROCProfiler v2 is 9.0.0

#### Note: ROCProfilerV2 is currently considered a beta version and is subject to change in future releases

### ROCProfilerV2 Modules

- Counters
- Hardware
- Generic Buffer
- Session
- Filter
- Tools
- Plugins
- Samples
- Tests

## Getting started

### Requirements

- makecache
- Gtest Development Package (Ubuntu: libgtest-dev)
- libsystemd-dev, libelf-dev, libnuma-dev, libpciaccess-dev on ubuntu or their corresponding packages on any other OS
- Cppheaderparser, websockets, matplotlib, lxml, barectf Python3 Packages
- Python packages can be installed using:
  ```bash
  pip3 install -r requirements.txt
  ```

### Build

The user has two options for building:

- Option 1 (It will install in the path saved in ROCM_PATH environment variable or /opt/rocm if ROCM_PATH is empty):

  - Run

  ```bash
  # Normal Build
  ./build.sh --build OR ./build.sh -b
  # Clean Build
  ./build.sh --clean-build OR ./build.sh -cb
  ```

- Option 2 (Where ROCM_PATH envronment need to be set with the current installation directory of rocm), run the following:

  ```bash
  # Creating the build directory
  mkdir build && cd build

  # Configuring the rocprofv2 build
  cmake -DCMAKE_PREFIX_PATH=$ROCM_PATH -DCMAKE_MODULE_PATH=$ROCM_PATH/hip/cmake <CMAKE_OPTIONS> ..

  # Building the main runtime of the rocprofv2 project
  cmake --build . -- -j

  # Optionally, for building API documentation
  cmake --build . -- -j doc

  # Optionally, for building ROCProfiler V2 samples
  cmake --build . -- -j samples

  # Optionally, for building packages (DEB, RPM, TGZ)
  cmake --build . -- -j tests

  # Optionally, for building packages (DEB, RPM, TGZ)
  # Note: Requires rpm package on ubuntu
  cmake --build . -- -j package
  ```

### Install

- Optionally, run the following to install

  ```bash
  # Install rocprofv2 in the ROCM_PATH path
  ./rocprofv2 --install
  ```

  OR, if you are using option 2 in building

  ```bash
  cd build
  # Install rocprofv2 in the ROCM_PATH path
  cmake --build . -- -j install
  ```

## Features & Usage

  - ### rocsys
    ##### A command line utility to control a session (launch/start/stop/exit), with the required application to be traced or profiled in a rocprofv2 context. Usage:

      ```bash
      # Launch the application with the required profiling and tracing options with giving a session identifier to be used later
      rocsys --session session_name launch mpiexec -n 2 ./rocprofv2 -i samples/input.txt Histogram

      # Start a session with a given identifier created at launch
      rocsys --session session_name start

      # Stop a session with a given identifier created at launch
      rocsys –session session_name stop

      # Exit a session with a given identifier created at launch
      rocsys –session session_name exit
      ```

  - ### Counters and Metric Collection
    HW counters and derived metrics can be collected using following option:

      ```bash
      rocprofv2 -i samples/input.txt <app_relative_path>
      input.txt
      ```

      input.txt content Example (Details of what is needed inside input.txt will be mentioned with every feature):

      ```bash
      pmc: SQ_WAVES GRBM_COUNT GRBM_GUI_ACTIVE SQ_INSTS_VALU
      ```

  - ### Application Trace Support
    Different trace options are available while profiling an app:

      ```bash
      # HIP API & asynchronous activity tracing
      rocprofv2 --hip-api <app_relative_path> ## For synchronous HIP API Activity tracing
      rocprofv2 --hip-activity <app_relative_path> ## For both Synchronous & ASynchronous HIP API Activity tracing
      rocprofv2 --hip-trace <app_relative_path> ## Same as --hip-activity, added for backward compatibility

      # HSA API & asynchronous activity tracing
      rocprofv2 --hsa-api <app_relative_path> ## For synchronous HSA API Activity tracing
      rocprofv2 --hsa-activity <app_relative_path> ## For both Synchronous & ASynchronous HSA API Activity tracing
      rocprofv2 --hsa-trace <app_relative_path> ## Same as --hsa-activity, added for backward compatibility

      # Kernel dispatches tracing
      rocprofv2 --kernel-trace <app_relative_path> ## Kernel Dispatch Tracing

      # HIP & HSA API and asynchronous activity and kernel dispatches tracing
      rocprofv2 --sys-trace <app_relative_path> ## Same as combining --hip-trace & --hsa-trace & --kernel-trace
      ```

      For complete usage options, please run rocprofv2 help

      ```bash
      rocprofv2 --help
      ```

  - ### Plugin Support
    We have a template for adding new plugins. New plugins can be written on top of rocprofv2 to support the desired output format using include/rocprofiler/v2/rocprofiler_plugins.h header file. These plugins are modular in nature and can easily be decoupled from the code based on need. Installation files:

    ```bash
    rocprofiler-plugins_9.0.0-local_amd64.deb
    rocprofiler-plugins-9.0.0-local.x86_64.rpm
    ```
    - file plugin: outputs the data in txt files.
    - Perfetto plugin: outputs the data in protobuf format.
      - Protobuf files can be viewed using ui.perfetto.dev or using trace_processor
    - ATT (Advanced thread tracer) plugin: advanced hardware traces data in binary format. Please refer ATT section.
    - CTF plugin: Outputs the data in ctf format(a binary trace format)
      - CTF binary output can be viewed using TraceCompass or babeltrace.

    Usage:

    ```bash
    # plugin_name can be file, perfetto , ctf
    ./rocprofv2 --plugin plugin_name -i samples/input.txt -d output_dir <app_relative_path> # -d is optional, but can be used to define the directory output for output results
    ```

    Both the output directory and filenames allow for simple environment variable substitution via a special syntax %q{var} -> $var, e.g.:
    ```bash
    export var="FOO"
    rocprofv2 --plugin perfetto -o file_%q{var}_name
    # Generates file names: file_FOO_name[...].pftrace
    ```

    - #### (ATT) Advanced Thread Trace
      Tool used to collect fine-grained hardware metrics. Provides ISA-level instruction hotspot analysis via hardware tracing.

        ```bash
        # ATT(Advanced Thread Trace) needs some preparation before running.

        # 1. Make sure to generate the assembly file for application by executing the following before compiling your HIP Application
        # This can be achieved globally by following environment variable
        export HIPCC_COMPILE_FLAGS_APPEND="--save-temps -g"
        # Similarly, the --save-temps -g flags can be added per file for better ISA generation control.

        # 2. Install plugin package
        # see Plugin Support section for installation

        # 3. Run the following to view the trace
        # Att-specific options must come right after the assembly file
        rocprofv2 -i input.txt --plugin att <app_assembly_file> --mode network <app_relative_path>
        ```
        ```bash
        # Example for vectoradd on navi31.
        # Special attention to gfx1100.s==navi31 in the ISA file name. 
        # Use gfx1030 for navi21, gfx90a for MI200 and gfx940 for MI300
        hipcc -g --save-temps vectoradd_hip.cpp -o vectoradd_hip.exe
        # for csv mode
        rocprofv2 -i input.txt -o out.csv --plugin att vectoradd_hip-hip-amdgcn-amd-amdhsa-gfx1100.s --mode csv ./vectoradd_hip.exe
        # for browser mode
        rocprofv2 -i input.txt --plugin att vectoradd_hip-hip-amdgcn-amd-amdhsa-gfx1100.s --mode network ./vectoradd_hip.exe
        # Then open the browser at http://localhost:8000
        # The ISA can also be obtained from llvm/roc objdump, however, annotations will be different
        ```
      - ##### app_assembly_file_relative_path
        AMDGCN ISA file with .s extension generated in 1st step
      - ##### app_relative_path
        Path for the running application
      - ##### ATT plugin optional parameters
        - --depth [n]: How many waves per slot to parse (maximum).
        - --mpi [proc]: Parse with this many mpi processes, for greater analysis speed. Does not change results. Requires mpi4py.
        - --att_kernel "filename": Kernel filename to use (instead of ATT asking which one to use).
        - --trace_file "files": glob (wildcards allowed) of traces files to parse. Requires quotes for use with wildcards.
        - --mode [network, file, csv, off (default)]
          - ##### network
            Opens the server with the browser UI.
            att needs 2 ports available (e.g. 8000, 18000). There is an option (default: --ports "8000,18000") to change these.
            In case rocprofv2 is running on a different machine, use port forwarding "ssh -L 8000:localhost:8000 <user@IP>" so the browser can be used locally. For docker, use --network=host --ipc=host -p8000:8000 -p18000:18000
          - ##### file
            Dumps the analyzed json files to disk for vieweing at a later time. Run python3 httpserver.py from within the generated ui/ folder to view the trace, similarly to network mode. The folder can be copied to another machine, and will run without rocm.
          - ##### csv
            Generates CSV file (-o filename.csv) with the assembly and instruction latency tables.
          - ##### off
            Runs trace collection but not analysis, so it can be analyzed at a later time. Run rocprofv2 ATT [network, file] with the same parameters, removing the application binary, to analyze previously generated traces.
      - ##### input.txt 
        Required. Used to select specific compute units and other trace parameters.
        For first time users, we recommend compiling and running vectorAdd with
        ```bash
        att: TARGET_CU=1
        SE_MASK=0x1
        SIMD_MASK=0x3
        ```
        and histogram with
        ```bash
        att: TARGET_CU=0
        SE_MASK=0xFF
        SIMD_MASK=0xF // 0xF for GFX9, SIMD_MASK=0 for Navi
        ```
        Possible contents:
        - att: TARGET_CU=1 //or some other CU [0,15] - WGP for Navi [0,8]
        - SE_MASK=0x1 // bitmask of shader engines. The fewer, the easier on the hardware. Default enables 1 out of 4 shader engines.
        - SIMD_MASK=0xF // GFX9: bitmask of SIMDs. Navi: SIMD Index [0-3].
<<<<<<< HEAD   (bc7174 SWDEV-415057: Fixing warning messages for masked simds)
        - DISPATCH=ID,RN // collect trace only for the given dispatch_ID and MPI rank RN. RN ignored for single processes. Multiple lines with varying combinations of RN and ID can be added.
=======
        - DISPATCH=ID,RN // collect trace only for the given dispatch_ID (from --kernel-trace) and MPI rank RN. RN is optional and ignored for single processes. Multiple lines with varying combinations of RN and ID can be added.
>>>>>>> CHANGE (157eac DCGPUBU-44: Added arbitrary envvars to file/dir names. Squas)
        - KERNEL=kernname // Profile only kernels containing the string kernname (c++ mangled name). Multiple lines can be added.
        - PERFCOUNTERS_COL_PERIOD=0x3 // Multiplier period for counter collection [0~31]. 0=fastest (usually once every 16 cycles). GFX9 only. Counters will be shown in a graph over time in the browser UI.
        - PERFCOUNTER=counter_name // Add a SQ counter to be collected with ATT; period defined by PERFCOUNTERS_COL_PERIOD. GFX9 only.
        - BUFFER_SIZE=[size] // Sets size of the ATT buffer collection, per dispatch, in megabytes (shared among all shader engines).
  
  - ### Flush Interval
    Flush interval can be used to control the interval time in milliseconds between the buffers flush for the tool. However, if the buffers are full the flush will be called on its own. This can be used as in the next example:
    ```bash
    rocprofv2 --flush-interval <TIME_INTERVAL_IN_MILLISECONDS> <rest_of_rocprofv2_arguments> <app_relative_path>
    ```

  - ### Trace Period
    Trace period can be used to control when the profiling or tracing is enabled using two arguments, the first one is the delay time, which is the time spent idle without tracing or profiling. The second argument is the profiling or the tracing time, which is the active time where the profiling and tracing are working, so basically, the session will work in the following timeline:
    ```
    # <DELAY_TIME> => <PROFILING_OR_TRACING_SESSION_START> => <ACTIVE_PROFILING_OR_TRACING_TIME> => <PROFILING_OR_TRACING_SESSION_STOP>
    ```
    This feature can be used using the following command:
    ```bash
    rocprofv2 --trace-period <delay>:<active_time>:<interval> <rest_of_rocprofv2_arguments> <app_relative_path>
    ```
    - delay: Time delay to start profiling (ms).
    - active_time: How long to profile for (ms).
    - interval: If set, profiling sessions will start (loop) every "interval", and run for "active_time", until the application ends. Must be higher than "active_time".

- Device Profiling: A device profiling session allows the user to profile the GPU device for counters irrespective of the running applications on the GPU. This is different from application profiling. device profiling session doesn't care about the host running processes and threads. It directly provides low level profiling information.

- Session Support: A session is a unique identifier for a profiling/tracing/pc-sampling task. A ROCProfilerV2 Session has enough information about what needs to be collected or traced and it allows the user to start/stop profiling/tracing whenever required. More details on the API can be found in the API specification documentation that can be installed using rocprofiler-doc package. Samples also can be found for how to use the API in samples directory.

## Tests

 We make use of the GoogleTest (Gtest) framework to automatically find and add test cases to the CMAKE testing environment. ROCProfilerV2 testing is categorized as following:

- unittests (Gtest Based) : These includes tests for core classes. Any newly added functionality should have a unit test written to it.

- featuretests (standalone and Gtest Based): These includes both API tests and tool tests. Tool is tested against different applications to make sure we have right output in evry run.

- memorytests (standalone): This includes running address sanitizer for memory leaks, corruptions.

  installation:

  ```bash
  rocprofiler-tests_9.0.0-local_amd64.deb
  rocprofiler-tests-9.0.0-local.x86_64.rpm
  ```

- Optionally, for tests: run the following:

- Option 1, using rocprofv2 script:

    ```bash
    cd build && ./rocprofv2 -t
    ```

- Option 2, using cmake directly:

    ```bash
    cd build && cmake --build . -- -j check
    ```
## Logging

To enable error messages logging to '/tmp/rocprofiler_log.txt':
   ```bash
   $ export ROCPROFILER_LOG=1
   ```
## Documentation

We make use of doxygen to automatically generate API documentation. Generated document can be found in the following path:

   ```bash
   # ROCM_PATH by default is /opt/rocm
   # It can be set by the user in different location if needed.
   <ROCM_PATH>/share/doc/rocprofv2
   ```

   installation:

   ```bash
   rocprofiler-docs_9.0.0-local_amd64.deb
   rocprofiler-docs-9.0.0-local.x86_64.rpm
   ```

## Samples

- Profiling: Profiling Samples depending on replay mode
- Tracing: Tracing Samples

installation:

```bash
rocprofiler-samples_9.0.0-local_amd64.deb
rocprofiler-samples-9.0.0-local.x86_64.rpm
```

usage:

samples can be run as independent executables once installed

## Project Structure

- bin: ROCProf scripts along with V1 post processing scripts
- doc: Documentation settings for doxygen, V1 API Specifications pdf document.
- include:
  - rocprofiler.h: V1 API Header File
  - v2:
    - rocprofiler.h: V2 API Header File
    - rocprofiler_plugin.h: V2 Tool Plugins API
- plugin
  - file: File Plugin
  - perfetto: Perfetto Plugin
  - att: Adavced thread tracer Plugin
  - ctf: CTF Plugin
- samples: Samples of how to use the API, and also input.txt input file samples for counter collection and ATT.
- script: Scripts needed for tracing
- src: Source files of the project
  - api: API implementation for rocprofv2
  - core: Core source files needed for the V1/V2 API
    - counters: Basic and Derived Counters
    - hardware: Hardware support
    - hsa: Provides support for profiler and tracer to communicate with HSA
      - queues: Intercepting HSA Queues
      - packets: Packets Preparation for profiling
    - memory: Memory Pool used in buffers that saves the output data
    - session: Session Logic
      - filter: Type of profiling or tracing and its properties
      - tracer: Tracing support of the session
      - profiler: Profiling support of the session
      - spm: SPM support of the session
      - att: ATT support of the session
  - tools: Tools needed to run profiling and tracing
    - rocsys: Controlling Session from another CLI
  - utils: Utilities needed by the project
- tests: Tests folder
- CMakeLists.txt: Handles cmake list for the whole project
- build.sh: To easily build and compile rocprofiler
- CHANGELOG.md: Changes that are happening per release

## Support

Please report in the Github Issues

## Limitations
- ##### Navi3x requires a stable power state for counter collection.
  Currently, this state needs to be set by the user.
  To do so, set "power_dpm_force_performance_level" to be writeable for non-root users, then set performance level to profile_standard:
  ```bash
  sudo chmod 777 /sys/class/drm/card0/device/power_dpm_force_performance_level
  echo profile_standard >> /sys/class/drm/card0/device/power_dpm_force_performance_level
  ```
  Recommended: "profile_standard" for counter collection and "auto" for all other profiling. Use rocm-smi to verify the current power state. For multiGPU systems (includes integrated graphics), replace "card0" by the desired card.
