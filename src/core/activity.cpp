/******************************************************************************
Copyright (c) 2018 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*******************************************************************************/

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <atomic>
#include <string>

// Tracer messages protocol
#include <prof_protocol.h>

#include "core/context.h"
#include "inc/rocprofiler.h"
#include "util/hsa_rsrc_factory.h"

#define PUBLIC_API __attribute__((visibility("default")))

// Error handler
void fatal(const std::string msg) {
  fflush(stdout);
  fprintf(stderr, "%s\n\n", msg.c_str());
  fflush(stderr);
  abort();
}

// Check returned HSA API status
void check_status(hsa_status_t status) {
  if (status != HSA_STATUS_SUCCESS) {
    const char* error_string = NULL;
    rocprofiler_error_string(&error_string);
    fprintf(stderr, "ERROR: %s\n", error_string);
    abort();
  }
}

extern "C" {
PUBLIC_API const char* GetOpName(uint32_t op) { return strdup("PCSAMPLE"); }

PUBLIC_API bool RegisterApiCallback(uint32_t op, void* callback, void* arg) { return true; }

PUBLIC_API bool RemoveApiCallback(uint32_t op) { return true; }

PUBLIC_API bool InitActivityCallback(void* callback, void* arg) {
  return true;
}

PUBLIC_API bool EnableActivityCallback(uint32_t op, bool enable) {
  return true;
}
}  // extern "C"
