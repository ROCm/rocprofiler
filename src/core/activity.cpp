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

#define ROCP_INTERNAL_BUILD
#include "activity.h"

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

struct evt_cb_entry_t {
  void* callback;
  void* arg;
};
typedef std::atomic<evt_cb_entry_t> evt_cb_entry_atomic_t;
evt_cb_entry_atomic_t evt_cb_table[HSA_EVT_ID_NUMBER]{};

hsa_status_t codeobj_evt_callback(
  rocprofiler_hsa_cb_id_t id,
  const rocprofiler_hsa_callback_data_t* cb_data,
  void* arg)
{
  evt_cb_entry_t evt = evt_cb_table[id].load(std::memory_order_relaxed);
  activity_rtapi_callback_t evt_callback = (activity_rtapi_callback_t)evt.callback;

  if (evt_callback != NULL) {
    evt_callback(ACTIVITY_DOMAIN_HSA_EVT, id, cb_data, evt.arg);
  }

  return HSA_STATUS_SUCCESS;
}

PUBLIC_API const char* GetEvtName(uint32_t op) { return strdup("CODEOBJ"); }

PUBLIC_API bool RegisterEvtCallback(uint32_t op, void* callback, void* arg) {
  evt_cb_table[op].store(evt_cb_entry_t{callback, arg}, std::memory_order_relaxed);

  rocprofiler_hsa_callbacks_t ocb{};
  switch (op) {
    case HSA_EVT_ID_ALLOCATE:
      ocb.allocate = codeobj_evt_callback;
      break;
    case HSA_EVT_ID_DEVICE:
      ocb.device = codeobj_evt_callback;
      break;
    case HSA_EVT_ID_MEMCOPY:
      ocb.memcopy = codeobj_evt_callback;
      break;
    case HSA_EVT_ID_SUBMIT:
      ocb.submit = codeobj_evt_callback;
      break;
    case HSA_EVT_ID_KSYMBOL:
      ocb.ksymbol = codeobj_evt_callback;
      break;
    case HSA_EVT_ID_CODEOBJ:
      ocb.codeobj = codeobj_evt_callback;
      break;
    default:
      fatal("invalid activity opcode");
  }
  rocprofiler_set_hsa_callbacks(ocb, NULL);

  return true;
}

PUBLIC_API bool RemoveEvtCallback(uint32_t op) {
  rocprofiler_hsa_callbacks_t ocb{};
  rocprofiler_set_hsa_callbacks(ocb, NULL);
  return true;
}
}  // extern "C"
