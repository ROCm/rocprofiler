## DISCLAIMER

The information presented in this document is for informational purposes only and may contain technical inaccuracies, omissions, and typographical errors. The information contained herein is subject to change and may be rendered inaccurate for many reasons, including but not limited to product and roadmap changes, component and motherboard version changes, new model and/or product releases, product differences between differing manufacturers, software changes, BIOS flashes, firmware upgrades, or the like. Any computer system has risks of security vulnerabilities that cannot be completely prevented or mitigated. AMD assumes no obligation to update or otherwise correct or revise this information. However, AMD reserves the right to revise this information and to make changes from time to time to the content hereof without obligation of AMD to notify any person of such revisions or changes.THIS INFORMATION IS PROVIDED ‘AS IS.” AMD MAKES NO REPRESENTATIONS OR WARRANTIES WITH RESPECT TO THE CONTENTS HEREOF AND ASSUMES NO RESPONSIBILITY FOR ANY INACCURACIES, ERRORS, OR OMISSIONS THAT MAY APPEAR IN THIS INFORMATION. AMD SPECIFICALLY DISCLAIMS ANY IMPLIED WARRANTIES OF NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR ANY PARTICULAR PURPOSE. IN NO EVENT WILL AMD BE LIABLE TO ANY PERSON FOR ANY RELIANCE, DIRECT, INDIRECT, SPECIAL, OR OTHER CONSEQUENTIAL DAMAGES ARISING FROM THE USE OF ANY INFORMATION CONTAINED HEREIN, EVEN IF AMD IS EXPRESSLY ADVISED OF THE POSSIBILITY OF SUCH DAMAGES. AMD, the AMD Arrow logo, and combinations thereof are trademarks of Advanced Micro Devices, Inc. Other product names used in this publication are for identification purposes only and may be trademarks of their respective companies.

© 2022 Advanced Micro Devices, Inc. All Rights Reserved.

## ROC Profiler library version 1.0

## Introduction
Profiling with metrics and traces based on perfcounters (PMC) and traces (SPM).
Implementation is based on AqlProfile HSA extension.
Library supports GFX8/GFX9.

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

Roctracer & Rocprofiler need to be installed in the same directory.
```bash
$ export CMAKE_PREFIX_PATH=<path to hsa-runtime includes>:<path to hsa-runtime library>
$ export CMAKE_BUILD_TYPE=<debug|release> # release by default
$ export CMAKE_DEBUG_TRACE=1 # 1 to enable debug tracing
```
To build with the current installed ROCM:
 ```bash
$ cd .../rocprofiler
$ export CMAKE_PREFIX_PATH=/opt/rocm/include/hsa:/opt/rocm
$ mkdir build
$ cd build
$ cmake ..
$ make
```
To run the test:
```bash
$ cd .../rocprofiler/build
$ export LD_LIBRARY_PATH=.:<other paths> # paths to ROC profiler and oher libraries
$ export HSA_TOOLS_LIB=librocprofiler64.so # ROC profiler library loaded by HSA runtime
$ export ROCP_TOOL_LIB=test/libtool.so # tool library loaded by ROC profiler
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

## ROC Profiler library version 2.0

## Introduction

ROCProfilerV2 is a newly developed design for AMD’s tooling infrastructure that provides a hardware specific low level performance analysis interface for profiling of GPU compute applications.


### ROCProfilerV2 Modules

- Counters
- Hardware
- Buffer Pool
- Session
- Filter
- Tools
- Plugins
- Samples
- Tests

## Getting started

### Requirements

- Makecache
- Gtest Development Package (Ubuntu: libgtest-dev)
- Cppheaderparser Python 3 Package
- Lxml Python 3 Package
- Systemd Development Package (Ubuntu: libsystemd-dev)

### Build

The user has two options for building:

- Option 1 (It will install in the path saved in ROCM_PATH environment variable or /opt/rocm if ROCM_PATH is empty):

  - Run

  ```bash
  # Normal Build
  ./build.sh --build 
  # Clean Build
  ./build.sh --clean-build
  ```

  - Optionally, For testing, run the following

  ```bash
  ./rocprofv2 -t
  ```

- Option 2 (Where ROCM_PATH envronment need to be set with the current installation directory of rocm), run the following:

  ```bash
  # Creating the build directory
  mkdir build && cd build

  # Configuring the rocprofv2 build
  cmake -DCMAKE_MODULE_PATH=$ROCM_PATH/hip/cmake <CMAKE_OPTIONS> ..

  # Building the main runtime of the rocprofv2 project
  cmake --build . -- runtime

  # Optionally, for building API documentation
  cmake --build . -- doc
  
  # Optionally, for building packages (DEB, RPM, TGZ)
  cmake --build . -- package
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
  cmake --build . -- install
  ```

### Test

- Optionally, for tests: run the following:

  ```bash
  cmake --build . -- check
  ```

## Usage

### Features

- Tools:

  - rocsys: This is a frontend command line utility to launch/start/stop/exit a session with the required application to be traced or profiled in rocprofv2 context. Usage:

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

  - Device Profiling: A device profiling session allows the user to profile the GPU device for counters irrespective of the running applications on the GPU. This is different from application profiling. device profiling session doesn't care about the host running processes and threads. It directly provides low level profiling information.

  - rocprofv2:

    - Counters and Metric Collection: HW counters and derived metrics can be collected using following option:

      ```bash
      rocprofv2 -i samples/input.txt <app_relative_path>
      input.txt
      ```

      input.txt content:

      ```bash
      pmc: SQ_WAVES GRBM_COUNT GRBM_GUI_ACTIVE SQ_INSTS_VALU
      ```

    - Application Trace Support: Differnt trace options are available while profiling an app:

      ```bash
      # HIP API & asynchronous activity tracing
      rocprofv2 --hip-api <app_relative_path>
      rocprofv2 --hip-activity <app_relative_path>

      # HSA API & asynchronous activity tracing
      rocprofv2 --hsa-api <app_relative_path>
      rocprofv2 --hsa-activity <app_relative_path>

      # Kernel dispatches tracing
      rocprofv2 --kernel-trace <app_relative_path>

      # HIP & HSA API and asynchronous activity and kernel dispatches tracing
      rocprofv2 --sys-trace <app_relative_path>
      ```
      
      For complete usage options, please run rocprofv2 help
      ```bash
      rocprofv2 --help
      ``` 
    - Advanced Thread Trace: It can collect kernel running time, granular hardware metrics per kernel dispatch and provide hotspot analysis at source code level via hardware tracing.

      ```bash
      # ATT(Advanced Thread Trace) needs few proeconditions before running.
      #1. Make sure to generate the assembly file for application
      export HIPCC_COMPILE_FLAGS_APPEND="--save-temps -g"
      
      #2. Install plugin package 
      rocprofiler-plugins_2.0.0-local_amd64.deb
      
      #3. Additionally you might need to install few python packages.e.g:
      pip3 install websockets
      pip3 install matplotlib 

      # Run the following to view the trace
      rocprofv2 --plugin att <app_relative_path_assembly_file> -i input.txt <app_relative_path>
      
      # app_assembly_file_relative_path is the assembly file with .s extension generated in 1st step
      # app_relative_path is the path for the application binary
      # input.txt gives flexibility to to target the compute unit and provide filters.
            # input.txt contents: att: TARGET_CU=0 
      # att needs 2 ports opened (8000, 18000), In case the browser is running on a different machine.
      ```

    - Plugin Support: We have a template for adding new plugins. New plugins can be written on top of rocprofv2 to support the desired output format. These plugins are modular in nature and can easily be decoupled from the code based on need. E.g.
      - file plugin: outputs the data in txt files.
      - Perfetto plugin: outputs the data in protobuf format.
      - Adavced thread tracer plugin: advanced hardware traces data in binary format.
      - Grafana plugin: streaming the data to Prometheus and Jaeger services, so that it can be used by Grafana ROCProfilerV2 dashboard, for more details please refer to [Grafana Plugin Documentation](plugins/grafana/README.md)

      usage:

      ```bash
      # plugin_name can be file, perfetto, att or grafana
      ./rocprofv2 --plugin plugin_name -i samples/input.txt <app_relative_path>
      ```

- Profile Replay Modes: Different replay modes are provided for flexibility to support kernel profiling. The API provides functionality for profiling GPU applications in kernel and application and user mode and also with no replay mode at all and it provides the records pool support with an easy sequence of calls, so the user can be able to profile and trace in easy small steps. Currently, Kernel replay mode is the only supported mode.

- Session Support: A session is a unique identifier for a profiling/tracing/pc-sampling task. A ROCProfilerV2 Session has enough information about what needs to be collected or traced and it allows the user to start/stop profiling/tracing whenever required. A simple session API usage:

   ```c++
   // Initialize the tools
   rocprofiler_initialize();

   // Creating the session with given replay mode
   rocprofiler_session_id_t session_id;
   rocprofiler_create_session(rocprofiler_KERNEL_REPLAY_MODE, &session_id);
   
   // Start Session 
   rocprofiler_start_session(session_id);
    
   // profile a kernel -kernelA
   hipLaunchKernelGGL(kernelA, dim3(1), dim3(1), 0, 0);
       
   // Deactivating session
   rocprofiler_terminate_session(session_id);

   // Destroy sessions
   rocprofiler_destroy_session(session_id);
    
   // Destroy all profiling related objects
   rocprofiler_finalize();
   ```

- Quality Control: We make use of the GoogleTest (Gtest) framework to automatically find and add test cases to the CMAKE testing environment. ROCProfilerV2 testing is categorized as following:
  - unittests (Gtest Based) : These includes tests for core classes. Any newly added functionality should have a unit test written to it.

  - featuretests (standalone and Gtest Based): These includes both API tests and tool tests. Tool is tested against different applications to make sure we have right output in evry run.

  - memorytests (standalone): This includes running address sanitizer for memory leaks, corruptions.

- Documentation: We make use of doxygen to autmatically generate API documentation. Generated document can be found in the following path:

   ```bash
   # ROCM_PATH by default is /opt/rocm
   # It can be set by the user in different location if needed.
   <ROCM_PATH>/share/doc/rocprofv2
   ```

## Project Structure

- Doc: Documentation settings for doxygen
- Plugins
  - File Plugin
  - Perfetto Plugin
  - Adavced thread tracer Plugin
  - Grafana Plugin
- Samples: Samples of how to use the API
- Script: Scripts needed for tracing
- Src: Source files of the project
  - API: API implementation for rocprofv2
  - Core: Core source files needed for the API
    - Counters: Basic and Derived Counters
    - Hardware: Hardware support
    - HSA: Provides support for profiler and tracer to communicate with HSA
      - Queues: Intercepting HSA Queues
      - Packets: Packets Preparation for profiling
    - Memory: Memory Pool used in buffers that saves the output data
    - Session: Session Logic
      - Filter: Type of profiling or tracing and its properties
      - Tracer: Tracing support of the session
      - Profiler: Profiling support of the session
  - Tools: Tools needed to run profiling and tracing
    - rocsys: Controling Session from another CLI
    - rocprofv2: Binary version of rocprofv2 script (Not yet supported at the moment)
  - Utils: Utilities needed by the project
- Tests: Tests folder
- CMakeLists.txt: Handles cmake list for the whole project

## Samples

- Profiling: Profiling Samples depending on replay mode
- Tracing: Tracing Samples

## Support

Please report in the Github Issues

## Limitations
