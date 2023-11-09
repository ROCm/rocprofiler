/* Copyright (c) 2022 Advanced Micro Devices, Inc.

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
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE SOFTWARE. */

#include "queue.h"

#include <atomic>
#include <map>
#include <mutex>
#include <optional>
#include <string>
#include <vector>
#include <utility>
#include <algorithm>
#include <numa.h>

#include "rocprofiler.h"
#include "src/api/rocprofiler_singleton.h"
#include "src/core/hsa/packets/packets_generator.h"
#include "src/core/hsa/hsa_support.h"
#include "src/utils/helper.h"
#include "src/core/isa_capture/code_object_track.hpp"


#define CHECK_HSA_STATUS(msg, status)                                                              \
  do {                                                                                             \
    if ((status) != HSA_STATUS_SUCCESS && status != HSA_STATUS_INFO_BREAK) {                       \
      try {                                                                                        \
        const char* emsg = nullptr;                                                                \
        hsa_status_string(status, &emsg);                                                          \
        if (!emsg) emsg = "<Unknown HSA Error>";                                                   \
        std::cerr << msg << std::endl;                                                             \
        std::cerr << emsg << std::endl;                                                            \
      } catch (std::exception & e) {                                                               \
      }                                                                                            \
      abort();                                                                                     \
    }                                                                                              \
  } while (0)

#define __NR_gettid 186

#define DEFAULT_ATT_BUFFER_SIZE 0x40000000

std::mutex sessions_pending_signal_lock;

namespace rocprofiler {

std::atomic<uint32_t> ACTIVE_INTERRUPT_SIGNAL_COUNT{0};

uint32_t GetCurrentActiveInterruptSignalsCount() {
  return ACTIVE_INTERRUPT_SIGNAL_COUNT.load(std::memory_order_relaxed);
}

typedef std::vector<hsa_ven_amd_aqlprofile_info_data_t> pmc_callback_data_t;

static inline bool IsEventMatch(const hsa_ven_amd_aqlprofile_event_t& event1,
                                const hsa_ven_amd_aqlprofile_event_t& event2) {
  return (event1.block_name == event2.block_name) && (event1.block_index == event2.block_index) &&
      (event1.counter_id == event2.counter_id);
}

typedef std::vector<hsa_ven_amd_aqlprofile_info_data_t> att_trace_callback_data_t;


void AddKernelName(uint64_t handle, std::string name) {
  HSASupport_Singleton& hsasupport_singleton = HSASupport_Singleton::GetInstance();
  std::lock_guard<std::mutex> lock(hsasupport_singleton.ksymbol_map_lock);
  hsasupport_singleton.ksymbols->emplace(handle, name);
}
void RemoveKernelName(uint64_t handle) {
  HSASupport_Singleton& hsasupport_singleton = HSASupport_Singleton::GetInstance();
  std::lock_guard<std::mutex> lock(hsasupport_singleton.ksymbol_map_lock);
  hsasupport_singleton.ksymbols->erase(handle);
}
std::string GetKernelNameFromKsymbols(uint64_t handle) {
  HSASupport_Singleton& hsasupport_singleton = HSASupport_Singleton::GetInstance();
  std::lock_guard<std::mutex> lock(hsasupport_singleton.ksymbol_map_lock);
  if (hsasupport_singleton.ksymbols->find(handle) != hsasupport_singleton.ksymbols->end())
    return hsasupport_singleton.ksymbols->at(handle);
  else
    return "Unknown Kernel!";
}

void AddKernelNameWithDispatchID(std::string name, uint64_t id) {
  HSASupport_Singleton& hsasupport_singleton = HSASupport_Singleton::GetInstance();
  std::lock_guard<std::mutex> lock(hsasupport_singleton.kernel_names_map_lock);
  if (hsasupport_singleton.kernel_names->find(name) == hsasupport_singleton.kernel_names->end())
    hsasupport_singleton.kernel_names->emplace(name, std::vector<uint64_t>());
  hsasupport_singleton.kernel_names->at(name).push_back(id);
}
std::string GetKernelNameUsingDispatchID(uint64_t given_id) {
  HSASupport_Singleton& hsasupport_singleton = HSASupport_Singleton::GetInstance();
  std::lock_guard<std::mutex> lock(hsasupport_singleton.kernel_names_map_lock);
  for (auto kernel_name : (*hsasupport_singleton.kernel_names)) {
    for (auto dispatch_id : kernel_name.second) {
      if (dispatch_id == given_id) return kernel_name.first;
    }
  }
  return "Unknown Kernel!";
}


struct kernel_descriptor_t {
  uint8_t reserved0[16];
  int64_t kernel_code_entry_byte_offset;
  uint8_t reserved1[20];
  uint32_t compute_pgm_rsrc3;
  uint32_t compute_pgm_rsrc1;
  uint32_t compute_pgm_rsrc2;
  uint16_t kernel_code_properties;
  uint8_t reserved2[6];
};
// AMD Compute Program Resource Register Three.
using amd_compute_pgm_rsrc_three32_t = uint32_t;
enum amd_compute_gfx9_pgm_rsrc_three_t {
  AMD_HSA_BITS_CREATE_ENUM_ENTRIES(AMD_COMPUTE_PGM_RSRC_THREE_ACCUM_OFFSET, 0, 5),
  AMD_HSA_BITS_CREATE_ENUM_ENTRIES(AMD_COMPUTE_PGM_RSRC_THREE_TG_SPLIT, 16, 1)
};
enum amd_compute_gfx10_gfx11_pgm_rsrc_three_t {
  AMD_HSA_BITS_CREATE_ENUM_ENTRIES(AMD_COMPUTE_PGM_RSRC_THREE_SHARED_VGPR_COUNT, 0, 4),
  AMD_HSA_BITS_CREATE_ENUM_ENTRIES(AMD_COMPUTE_PGM_RSRC_THREE_INST_PREF_SIZE, 4, 6),
  AMD_HSA_BITS_CREATE_ENUM_ENTRIES(AMD_COMPUTE_PGM_RSRC_THREE_TRAP_ON_START, 10, 1),
  AMD_HSA_BITS_CREATE_ENUM_ENTRIES(AMD_COMPUTE_PGM_RSRC_THREE_TRAP_ON_END, 11, 1),
  AMD_HSA_BITS_CREATE_ENUM_ENTRIES(AMD_COMPUTE_PGM_RSRC_THREE_IMAGE_OP, 31, 1)
};

// Kernel code properties.
enum amd_kernel_code_property_t {
  AMD_HSA_BITS_CREATE_ENUM_ENTRIES(AMD_KERNEL_CODE_PROPERTY_ENABLE_SGPR_PRIVATE_SEGMENT_BUFFER, 0,
                                   1),
  AMD_HSA_BITS_CREATE_ENUM_ENTRIES(AMD_KERNEL_CODE_PROPERTY_ENABLE_SGPR_DISPATCH_PTR, 1, 1),
  AMD_HSA_BITS_CREATE_ENUM_ENTRIES(AMD_KERNEL_CODE_PROPERTY_ENABLE_SGPR_QUEUE_PTR, 2, 1),
  AMD_HSA_BITS_CREATE_ENUM_ENTRIES(AMD_KERNEL_CODE_PROPERTY_ENABLE_SGPR_KERNARG_SEGMENT_PTR, 3, 1),
  AMD_HSA_BITS_CREATE_ENUM_ENTRIES(AMD_KERNEL_CODE_PROPERTY_ENABLE_SGPR_DISPATCH_ID, 4, 1),
  AMD_HSA_BITS_CREATE_ENUM_ENTRIES(AMD_KERNEL_CODE_PROPERTY_ENABLE_SGPR_FLAT_SCRATCH_INIT, 5, 1),
  AMD_HSA_BITS_CREATE_ENUM_ENTRIES(AMD_KERNEL_CODE_PROPERTY_ENABLE_SGPR_PRIVATE_SEGMENT_SIZE, 6, 1),
  AMD_HSA_BITS_CREATE_ENUM_ENTRIES(AMD_KERNEL_CODE_PROPERTY_RESERVED0, 7, 3),
  AMD_HSA_BITS_CREATE_ENUM_ENTRIES(AMD_KERNEL_CODE_PROPERTY_ENABLE_WAVEFRONT_SIZE32, 10,
                                   1),  // GFX10+
  AMD_HSA_BITS_CREATE_ENUM_ENTRIES(AMD_KERNEL_CODE_PROPERTY_USES_DYNAMIC_STACK, 11, 1),
  AMD_HSA_BITS_CREATE_ENUM_ENTRIES(AMD_KERNEL_CODE_PROPERTY_RESERVED1, 12, 4),
};

static const kernel_descriptor_t* GetKernelCode(uint64_t kernel_object) {
  const kernel_descriptor_t* kernel_code = NULL;
  rocprofiler::HSASupport_Singleton& hsasupport_singleton =
      rocprofiler::HSASupport_Singleton::GetInstance();
  hsa_status_t status =
      hsasupport_singleton.GetHSALoaderApi().hsa_ven_amd_loader_query_host_address(
          reinterpret_cast<const void*>(kernel_object),
          reinterpret_cast<const void**>(&kernel_code));
  if (HSA_STATUS_SUCCESS != status) {
    kernel_code = reinterpret_cast<kernel_descriptor_t*>(kernel_object);
  }
  return kernel_code;
}

static uint32_t arch_vgpr_count(const std::string_view& name,
                                const kernel_descriptor_t& kernel_code) {
  if (strcmp(name.data(), "gfx90a") == 0 || strncmp(name.data(), "gfx94", 5) == 0)
    return (AMD_HSA_BITS_GET(kernel_code.compute_pgm_rsrc3,
                             AMD_COMPUTE_PGM_RSRC_THREE_ACCUM_OFFSET) +
            1) *
        4;

  return (AMD_HSA_BITS_GET(kernel_code.compute_pgm_rsrc1,
                           AMD_COMPUTE_PGM_RSRC_ONE_GRANULATED_WORKITEM_VGPR_COUNT) +
          1) *
      (AMD_HSA_BITS_GET(kernel_code.kernel_code_properties,
                        AMD_KERNEL_CODE_PROPERTY_ENABLE_WAVEFRONT_SIZE32)
           ? 8
           : 4);
}
static uint32_t accum_vgpr_count(const std::string_view& name,
                                 const kernel_descriptor_t& kernel_code) {
  std::string info_name(name.data(), name.size());
  if (strcmp(info_name.c_str(), "gfx908") == 0) return arch_vgpr_count(name, kernel_code);
  if (strcmp(info_name.c_str(), "gfx90a") == 0 || strncmp(info_name.c_str(), "gfx94", 5) == 0)
    return (AMD_HSA_BITS_GET(kernel_code.compute_pgm_rsrc1,
                             AMD_COMPUTE_PGM_RSRC_ONE_GRANULATED_WORKITEM_VGPR_COUNT) +
            1) *
        8 -
        arch_vgpr_count(name, kernel_code);

  return 0;
}

static uint32_t sgpr_count(const std::string_view& name, const kernel_descriptor_t& kernel_code) {
  // GFX10 and later always allocate 128 sgprs.

  // TODO(srnagara): Recheck the extraction of gfxip from gpu name
  const char* name_data = name.data();
  const size_t gfxip_label_len = std::min(name.size() - 2, size_t{63});
  if (gfxip_label_len > 0 && strnlen(name_data, gfxip_label_len + 1) >= gfxip_label_len) {
    char gfxip[gfxip_label_len + 1];
    memcpy(gfxip, name_data, gfxip_label_len);
    gfxip[gfxip_label_len] = '\0';
    // TODO(srnagara): Check if it is hardcoded
    if (std::stoi(&gfxip[3]) >= 10) return 128;
    return (AMD_HSA_BITS_GET(kernel_code.compute_pgm_rsrc1,
                             AMD_COMPUTE_PGM_RSRC_ONE_GRANULATED_WAVEFRONT_SGPR_COUNT) /
                2 +
            1) *
        16;
  }
  return 0;
}

rocprofiler_kernel_properties_t set_kernel_properties(hsa_kernel_dispatch_packet_t packet,
                                                      hsa_agent_t agent) {
  const uint64_t kernel_object = packet.kernel_object;
  rocprofiler_kernel_properties_t kernel_properties_ptr = {};
  const kernel_descriptor_t* kernel_code = GetKernelCode(kernel_object);
  uint64_t grid_size = packet.grid_size_x * packet.grid_size_y * packet.grid_size_z;
  if (grid_size > UINT32_MAX) abort();
  kernel_properties_ptr.grid_size = grid_size;
  uint64_t workgroup_size =
      packet.workgroup_size_x * packet.workgroup_size_y * packet.workgroup_size_z;
  if (workgroup_size > UINT32_MAX) abort();
  kernel_properties_ptr.workgroup_size = (uint32_t)workgroup_size;
  kernel_properties_ptr.lds_size = packet.group_segment_size;
  kernel_properties_ptr.scratch_size = packet.private_segment_size;
  HSAAgentInfo agent_info = HSASupport_Singleton::GetInstance().GetHSAAgentInfo(agent.handle);
  kernel_properties_ptr.arch_vgpr_count =
      arch_vgpr_count(agent_info.GetDeviceInfo().getName(), *kernel_code);
  kernel_properties_ptr.accum_vgpr_count =
      accum_vgpr_count(agent_info.GetDeviceInfo().getName(), *kernel_code);
  kernel_properties_ptr.sgpr_count = sgpr_count(agent_info.GetDeviceInfo().getName(), *kernel_code);
  kernel_properties_ptr.wave_size =
      AMD_HSA_BITS_GET(kernel_code->kernel_code_properties,
                       AMD_KERNEL_CODE_PROPERTY_ENABLE_WAVEFRONT_SIZE32)
      ? 32
      : 64;
  kernel_properties_ptr.signal_handle = packet.completion_signal.handle;

  return kernel_properties_ptr;
}

namespace queue {

hsa_status_t pmcCallback(hsa_ven_amd_aqlprofile_info_type_t info_type,
                         hsa_ven_amd_aqlprofile_info_data_t* info_data, void* data) {
  hsa_status_t status = HSA_STATUS_SUCCESS;
  pmc_callback_data_t* passed_data = reinterpret_cast<pmc_callback_data_t*>(data);

  pmc_callback_data_t::iterator data_it;
  if (info_data->sample_id == 0) {
    passed_data->emplace_back(*info_data);
  } else {
    for (data_it = passed_data->begin(); data_it != passed_data->end(); ++data_it) {
      if (info_type == HSA_VEN_AMD_AQLPROFILE_INFO_PMC_DATA) {
        if (IsEventMatch(info_data->pmc_data.event, data_it->pmc_data.event)) {
          data_it->pmc_data.result += info_data->pmc_data.result;
        }
      }
    }
  }
  return status;
}

hsa_status_t attTraceDataCallback(hsa_ven_amd_aqlprofile_info_type_t info_type,
                                  hsa_ven_amd_aqlprofile_info_data_t* info_data, void* data) {
  hsa_status_t status = HSA_STATUS_SUCCESS;
  att_trace_callback_data_t* passed_data = reinterpret_cast<att_trace_callback_data_t*>(data);
  passed_data->push_back(*info_data);
  // TODO: clear output buffers after copying
  // either copy here or in ::AddAttRecord

  return status;
}

void AddRecordCounters(rocprofiler_record_profiler_t* record, const pending_signal_t* pending) {
  record->counters_count =
      rocprofiler_record_counters_instances_count_t{pending->context->metrics_list.size()};
  size_t counters_list_size =
      record->counters_count.value * sizeof(rocprofiler_record_counter_instance_t);
  rocprofiler_record_counter_instance_t* counters =
      static_cast<rocprofiler_record_counter_instance_t*>(malloc(counters_list_size));
  for (size_t i = 0; i < pending->context->metrics_list.size(); i++) {
    const rocprofiler::Metric* metric = pending->context->metrics_list[i];
    double value = 0;
    std::string metric_name = metric->GetName();
    auto it = pending->context->results_map.find(metric_name);
    if (it != pending->context->results_map.end()) {
      value = it->second->val_double;
    }
    counters[i] = (rocprofiler_record_counter_instance_t{
        // TODO(aelwazir): Moving to span once C++20 is adopted, strdup can be
        // removed after that
        rocprofiler_counter_id_t{rocprofiler::profiler::GetCounterID(metric_name)},
        rocprofiler_record_counter_value_t{value}});
  }
  record->counters = counters;
  rocprofiler::Session* session =
      rocprofiler::ROCProfiler_Singleton::GetInstance().GetSession(pending->session_id);
  void* initial_handle = const_cast<rocprofiler_record_counter_instance_t*>(record->counters);
  if (session->FindBuffer(pending->buffer_id)) {
    Memory::GenericBuffer* buffer = session->GetBuffer(pending->buffer_id);
    buffer->AddRecord(*record, record->counters, counters_list_size,
                      [initial_handle](auto& record, const void* data) {
                        if (record.counters == initial_handle && data != initial_handle) {
                          free(initial_handle);
                        }
                        record.counters =
                            static_cast<const rocprofiler_record_counter_instance_t*>(data);
                      });
  }
}

void AddAttRecord(rocprofiler_record_att_tracer_t* record, hsa_agent_t gpu_agent,
                  att_pending_signal_t& pending) {
  HSASupport_Singleton& hsasupport_singleton = HSASupport_Singleton::GetInstance();
  HSAAgentInfo agent_info = hsasupport_singleton.GetHSAAgentInfo(gpu_agent.handle);
  att_trace_callback_data_t data;
  hsa_status_t status =
      hsa_ven_amd_aqlprofile_iterate_data(pending.profile, attTraceDataCallback, &data);

  if ((status & HSA_STATUS_ERROR_OUT_OF_RESOURCES) == HSA_STATUS_ERROR_OUT_OF_RESOURCES)
    rocprofiler::warning("Warning: ATT buffer full!\n");
  if ((status & HSA_STATUS_ERROR_EXCEPTION) == HSA_STATUS_ERROR_EXCEPTION)
    rocprofiler::warning("Warning: ATT received a UTC memory error!\n");
  if (status == HSA_STATUS_ERROR) fatal("Thread Trace Error!");

  // Allocate memory for shader_engine_data
  record->shader_engine_data = static_cast<rocprofiler_record_se_att_data_t*>(
      calloc(data.size(), sizeof(rocprofiler_record_se_att_data_t)));

  att_trace_callback_data_t::iterator trace_data_it;

  uint32_t se_index = 0;
  // iterate over the trace data collected from each shader engine
  for (trace_data_it = data.begin(); trace_data_it != data.end(); trace_data_it++) {
    const void* data_ptr = trace_data_it->trace_data.ptr;
    const uint32_t data_size = trace_data_it->trace_data.size;

    void* buffer = NULL;
    if (data_size != 0) {
      // Allocate buffer on CPU to copy out trace data
      buffer = Packet::AllocateSysMemory(gpu_agent, data_size, &agent_info.cpu_pool_);
      if (buffer == NULL) fatal("Trace data buffer allocation failed");

      auto status =
          hsasupport_singleton.GetCoreApiTable().hsa_memory_copy_fn(buffer, data_ptr, data_size);
      if (status != HSA_STATUS_SUCCESS) fatal("Trace data memcopy to host failed");

      record->shader_engine_data[se_index].buffer_ptr = buffer;
      record->shader_engine_data[se_index].buffer_size = data_size;
      ++se_index;

      // TODO: clear output buffers after copying
    }
  }
  record->shader_engine_data_count = data.size();
}

/*
    Function name: enable_dispatch
    Argument : pointer to the the Queue class object
    Description: This function asserts if the mutex is not already
                 locked by the calling thread. It enable the kernel dispatch
                 from the given queue by setting its block signal to 0.
                 Finally, it updates the serializer queue with the given queue.
*/
void enable_dispatch(Queue* dispatch_queue) {
  // ToDO(srnagara): Find a way to assert if the mutex is already locked.
  // assert(!rocmtools::GetSerializer()->serializer_mutex.try_lock());
  profiler_serializer_t& serializer =
      rocprofiler::ROCProfiler_Singleton::GetInstance().GetSerializer();
  assert(serializer.dispatch_queue == nullptr);
  HSASupport_Singleton::GetInstance().GetCoreApiTable().hsa_signal_store_screlease_fn(
      dispatch_queue->GetBlockSignal(), 0);
  serializer.dispatch_queue = dispatch_queue;
}

/*
    Function name:  AsyncSignalReadyHandler
    Argument:    hsa signal value for which the async handler was called
                 and pointer to the data.
    Description: This async handler is invoked when the queue is ready
                 to launch a kernel. It first, resets the queue's ready signal to 1.
                 It then checks if there is any queue which has a kernel currently dispatched.
                 If yes, it pushes the queue to the dispatch ready else
                 it enables the dispatch for the given queue.
    Return :     It returns true since we need this handler to be invoked
                 each time the queue's ready signal (used for entire queue) is set to 0.
                 If we had a separate signal for every dispatch in the queue then we don't
                 need this to be invoked more than once in which case we would return false.
*/

bool AsyncSignalReadyHandler(hsa_signal_value_t signal_value, void* data) {
  HSASupport_Singleton& hsasupport_singleton = HSASupport_Singleton::GetInstance();
  profiler_serializer_t& serializer =
      rocprofiler::ROCProfiler_Singleton::GetInstance().GetSerializer();
  std::lock_guard<std::mutex> serializer_lock(serializer.serializer_mutex);
  auto queue = static_cast<Queue*>(data);
  std::lock_guard<std::mutex> queue_lock(queue->qw_mutex);
  /* If is_destroy is set by the destructor then unreg_async_handler is set
  ready signal is destroyed and
  the destructor is notified and the handler is unregistered by returning false
  */
  if (queue->state == is_destroy::to_destroy) {
    {
      queue->state = done_destroy;
      hsasupport_singleton.GetCoreApiTable().hsa_signal_destroy_fn(queue->GetReadySignal());
    }
    queue->cv_ready_signal.notify_one();
    return false;
  }
  hsasupport_singleton.GetCoreApiTable().hsa_signal_store_screlease_fn(queue->GetReadySignal(), 1);
  if (serializer.dispatch_queue == nullptr)
    enable_dispatch(queue);
  else
    serializer.dispatch_ready.push_back(queue);
  return true;
}
/*
    Function name:  SignalAsyncReadyHandler.
    Argument :      The signal value and pointer to the data to
                    pass to the handler.
    Description :   Registers a asynchronous callback function
                    for the ready signal to be invoked when it goes to zero.
*/
void SignalAsyncReadyHandler(const hsa_signal_t& signal, void* data) {
  hsa_status_t status =
      HSASupport_Singleton::GetInstance().GetAmdExtTable().hsa_amd_signal_async_handler_fn(
          signal, HSA_SIGNAL_CONDITION_EQ, 0, AsyncSignalReadyHandler, data);
  if (status != HSA_STATUS_SUCCESS) fatal("hsa_amd_signal_async_handler failed");
}
bool AsyncSignalHandler(hsa_signal_value_t signal_value, void* data) {
  auto queue_info_session = static_cast<queue_info_session_t*>(data);
  rocprofiler::ROCProfiler_Singleton& rocprofiler_singleton =
      rocprofiler::ROCProfiler_Singleton::GetInstance();
  rocprofiler::HSASupport_Singleton& hsasupport_singleton =
      rocprofiler::HSASupport_Singleton::GetInstance();
  if (!queue_info_session || !rocprofiler_singleton.GetSession(queue_info_session->session_id) ||
      !rocprofiler_singleton.GetSession(queue_info_session->session_id)->GetProfiler())
    return true;
  rocprofiler::Session* session = rocprofiler_singleton.GetSession(queue_info_session->session_id);
  std::lock_guard<std::mutex> lock(session->GetSessionLock());
  rocprofiler::profiler::Profiler* profiler = session->GetProfiler();
  std::vector<pending_signal_t*> pending_signals = const_cast<std::vector<pending_signal_t*>&>(
      profiler->GetPendingSignals(queue_info_session->writer_id));

  if (!pending_signals.empty()) {
    for (auto it = pending_signals.begin(); it != pending_signals.end();
         it = pending_signals.erase(it)) {
      auto& pending = *it;
      if (hsasupport_singleton.GetCoreApiTable().hsa_signal_load_relaxed_fn(pending->new_signal))
       return true;
      hsa_amd_profiling_dispatch_time_t time;
      hsasupport_singleton.GetAmdExtTable().hsa_amd_profiling_get_dispatch_time_fn(
          queue_info_session->agent, pending->new_signal, &time);
      {
        std::lock_guard<std::mutex> lock(hsasupport_singleton.signals_timestamps_map_lock);
        hsasupport_singleton.signals_timestamps[pending->original_signal.handle].time =
            std::make_optional(time);
      }
      //hsasupport_singleton.GetCoreApiTable().hsa_signal_destroy_fn(pending->new_signal);
      uint32_t record_count = 1;
      bool is_individual_xcc_mode = false;
      uint32_t xcc_count = queue_info_session->xcc_count;
      if (xcc_count > 1) {  // for MI300
        const char* str = getenv("ROCPROFILER_INDIVIDUAL_XCC_MODE");
        if (str != NULL) is_individual_xcc_mode = (atol(str) > 0);
        // for individual xcc mode, there will be xcc_count records for each dispatch
        // for accumulation mode, there will be only one record for a dispatch
        if (is_individual_xcc_mode) record_count = xcc_count;
      }
      for (uint32_t xcc_id = 0; xcc_id < record_count; xcc_id++) {
        rocprofiler_record_profiler_t record{};
        // TODO: (sauverma) gpu-id will need to support xcc like so- 1.1, 1.2, 1.3 ... 1.5 for
        // different xcc
        record.gpu_id = rocprofiler_agent_id_t{(uint64_t)queue_info_session->gpu_index};
        record.kernel_properties = pending->kernel_properties;
        record.thread_id = rocprofiler_thread_id_t{pending->thread_id};
        record.queue_idx = rocprofiler_queue_index_t{pending->queue_index};
        record.timestamps = rocprofiler_record_header_timestamp_t{time.start, time.end};
        record.queue_id = rocprofiler_queue_id_t{queue_info_session->queue_id};
        // Kernel Descriptor is the right record id generated in the WriteInterceptor function and
        // will be used to handle the kernel name of that dispatch
        record.header = rocprofiler_record_header_t{
            ROCPROFILER_PROFILER_RECORD, rocprofiler_record_id_t{pending->kernel_descriptor}};
        record.kernel_id = rocprofiler_kernel_id_t{pending->kernel_descriptor};
        record.correlation_id = rocprofiler_correlation_id_t{pending->correlation_id};

        if (pending->session_id.handle == 0) {
          pending->session_id = rocprofiler_singleton.GetCurrentSessionId();
        }
        if (pending->counters_count > 0) {
          if (xcc_id == 0 && pending->context && pending->context->metrics_list.size() > 0 &&
              pending->profile)  // call to GetCounterData() is required only once for a dispatch
            rocprofiler::metrics::GetCounterData(pending->profile, queue_info_session->agent,
                                                 pending->context->results_list);
          if (is_individual_xcc_mode)
            rocprofiler::metrics::GetCountersAndMetricResultsByXcc(
                xcc_id, pending->context->results_list, pending->context->results_map,
                pending->context->metrics_list, time.end - time.start);
          else
            rocprofiler::metrics::GetMetricsData(pending->context->results_map,
                                                 pending->context->metrics_list,
                                                 time.end - time.start);
          AddRecordCounters(&record, pending);
        } else {
          if (session->FindBuffer(pending->buffer_id)) {
            Memory::GenericBuffer* buffer = session->GetBuffer(pending->buffer_id);
            buffer->AddRecord(record);
          }
        }
      }
      if (pending->counters_count > 0 && pending->profile && pending->profile->events) {
        // TODO(aelwazir): we need a better way of distributing events and free them
        // if (pending->profile->output_buffer.ptr)
        //   numa_free(pending->profile->output_buffer.ptr, pending->profile->output_buffer.size);
        hsa_status_t status = hsasupport_singleton.GetAmdExtTable().hsa_amd_memory_pool_free_fn(
            (pending->profile->output_buffer.ptr));
        CHECK_HSA_STATUS("Error: Couldn't free output buffer memory", status);
        // if (pending->profile->command_buffer.ptr)
        //   numa_free(pending->profile->command_buffer.ptr, pending->profile->command_buffer.size);
        status = hsasupport_singleton.GetAmdExtTable().hsa_amd_memory_pool_free_fn(
            (pending->profile->command_buffer.ptr));
        CHECK_HSA_STATUS("Error: Couldn't free command buffer memory", status);
        delete pending->profile;
        for (auto& it : pending->context->results_map) {
          delete it.second;
        }
        delete pending->context;
          /*
      Check if the dispatch ready is empty, If so, there is no more
      dispatches to be launched and we return. Else, dispatch the
      kernel of the queue in the front of the dispatch_ready.
      */

        profiler_serializer_t& serializer =
          rocprofiler::ROCProfiler_Singleton::GetInstance().GetSerializer();
      std::lock_guard<std::mutex> serializer_lock(serializer.serializer_mutex);
      assert(serializer.dispatch_queue != nullptr);
      hsasupport_singleton.GetCoreApiTable().hsa_signal_store_screlease_fn(
          queue_info_session->block_signal, 1);
      serializer.dispatch_queue = nullptr;
      if (serializer.dispatch_ready.empty()) return false;
      Queue* queue = serializer.dispatch_ready.front();
      serializer.dispatch_ready.erase(serializer.dispatch_ready.begin());
      enable_dispatch(queue);

      }

    

      if (pending->new_signal.handle)
       hsasupport_singleton.GetCoreApiTable().hsa_signal_destroy_fn(pending->new_signal);
      if (queue_info_session->interrupt_signal.handle)
        hsasupport_singleton.GetCoreApiTable().hsa_signal_destroy_fn(
            queue_info_session->interrupt_signal);
    }
  }
  delete queue_info_session;
  ACTIVE_INTERRUPT_SIGNAL_COUNT.fetch_sub(1, std::memory_order_relaxed);
  return false;
}

bool AsyncSignalHandlerATT(hsa_signal_value_t /* signal */, void* data) {
  auto queue_info_session = static_cast<queue_info_session_t*>(data);
  rocprofiler::ROCProfiler_Singleton& rocprofiler_singleton =
      rocprofiler::ROCProfiler_Singleton::GetInstance();
  rocprofiler::HSASupport_Singleton& hsasupport_singleton =
      rocprofiler::HSASupport_Singleton::GetInstance();
  if (!queue_info_session || !rocprofiler_singleton.GetSession(queue_info_session->session_id) ||
      !rocprofiler_singleton.GetSession(queue_info_session->session_id)->GetAttTracer())
    return true;
  rocprofiler::Session* session = rocprofiler_singleton.GetSession(queue_info_session->session_id);
  rocprofiler::att::AttTracer* att_tracer = session->GetAttTracer();
  if (!session->GetAttTracer()) return true;

  std::vector<att_pending_signal_t>& pending_signals =
      const_cast<std::vector<att_pending_signal_t>&>(
          att_tracer->GetPendingSignals(queue_info_session->writer_id));

  if (!pending_signals.empty()) {
    for (auto it = pending_signals.begin(); it != pending_signals.end();
         it = pending_signals.erase(it)) {
      auto& pending = *it;
      std::lock_guard<std::mutex> lock(session->GetSessionLock());
      if (hsasupport_singleton.GetCoreApiTable().hsa_signal_load_relaxed_fn(pending.new_signal))
        return true;
      rocprofiler_record_att_tracer_t record{};
      record.kernel_id = rocprofiler_kernel_id_t{pending.kernel_descriptor};
      record.gpu_id = rocprofiler_agent_id_t{(uint64_t)queue_info_session->gpu_index};
      record.kernel_properties = pending.kernel_properties;
      record.thread_id = rocprofiler_thread_id_t{pending.thread_id};
      record.queue_idx = rocprofiler_queue_index_t{pending.queue_index};
      record.queue_id = rocprofiler_queue_id_t{queue_info_session->queue_id};
      record.writer_id = queue_info_session->writer_id;

      if (/*pending.counters_count > 0 && */ pending.profile) {
        AddAttRecord(&record, queue_info_session->agent, pending);
      }
      // July/01/2023 -> Changed this to queue_info_session->writer_id
      // so we can correlate to dispatches. kernel_id already has the descriptor.
      record.header = {ROCPROFILER_ATT_TRACER_RECORD,
                       rocprofiler_record_id_t{pending.kernel_descriptor}};

      record.intercept_list = codeobj_record::get_capture(record.header.id);
      std::atomic_thread_fence(std::memory_order_release);

      if (pending.session_id.handle == 0) {
        pending.session_id = rocprofiler_singleton.GetCurrentSessionId();
      }
      if (session->FindBuffer(pending.buffer_id)) {
        Memory::GenericBuffer* buffer = session->GetBuffer(pending.buffer_id);
        buffer->AddRecord(record);
        buffer->Flush();
      }
      codeobj_record::free_capture(record.header.id);

      hsa_status_t status = hsasupport_singleton.GetAmdExtTable().hsa_amd_memory_pool_free_fn(
          (pending.profile->output_buffer.ptr));
      CHECK_HSA_STATUS("Error: Couldn't free output buffer memory", status);
      status = hsasupport_singleton.GetAmdExtTable().hsa_amd_memory_pool_free_fn(
          (pending.profile->command_buffer.ptr));
      CHECK_HSA_STATUS("Error: Couldn't free command buffer memory", status);
      delete pending.profile;
    }
  }
  delete queue_info_session;

  std::atomic_thread_fence(std::memory_order_seq_cst);
  return false;
}

/*
    Function name: CreateBarrierPacket.
    Argument :     The list of transformed packets to add the
                   barrier packet to. Pointer to the completion signal
                   and the input signal of the barrier packet to be created.
    Description :  This packet creates the barrier packet with the given
                   completion signal and dependency signal. It then adds to
                   the transformed packets list.
*/
void CreateBarrierPacket(std::vector<Packet::packet_t>* transformed_packets,
                         const hsa_signal_t* packet_dependency_signal,
                         const hsa_signal_t* packet_completion_signal
                         ) {
  hsa_barrier_and_packet_t barrier{0};
  barrier.header = HSA_PACKET_TYPE_BARRIER_AND << HSA_PACKET_HEADER_TYPE |
      (1 << HSA_PACKET_HEADER_BARRIER) |
      (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_ACQUIRE_FENCE_SCOPE) |
      (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_RELEASE_FENCE_SCOPE);
  if (packet_completion_signal != nullptr) barrier.completion_signal = *packet_completion_signal;
  if (packet_dependency_signal != nullptr) barrier.dep_signal[0] = *packet_dependency_signal;
  void* barrier_ptr = &barrier;
  transformed_packets->emplace_back(*reinterpret_cast<Packet::packet_t*>(barrier_ptr));
}

void AddVendorSpecificPacket(const Packet::packet_t* packet,
                             std::vector<Packet::packet_t>* transformed_packets,
                             const hsa_signal_t& packet_completion_signal) {
  transformed_packets->emplace_back(*packet).completion_signal = packet_completion_signal;
}

void SignalAsyncHandler(const hsa_signal_t& signal, void* data) {
  hsa_status_t status =
      HSASupport_Singleton::GetInstance().GetAmdExtTable().hsa_amd_signal_async_handler_fn(
          signal, HSA_SIGNAL_CONDITION_EQ, 0, AsyncSignalHandler, data);
  CHECK_HSA_STATUS("Error: hsa_amd_signal_async_handler failed", status);
}

void signalAsyncHandlerATT(const hsa_signal_t& signal, void* data) {
  hsa_status_t status =
      HSASupport_Singleton::GetInstance().GetAmdExtTable().hsa_amd_signal_async_handler_fn(
          signal, HSA_SIGNAL_CONDITION_EQ, 0, AsyncSignalHandlerATT, data);
  CHECK_HSA_STATUS("Error: hsa_amd_signal_async_handler for ATT failed", status);
}

void CreateSignal(uint32_t attribute, hsa_signal_t* signal) {
  hsa_status_t status =
      HSASupport_Singleton::GetInstance().GetAmdExtTable().hsa_amd_signal_create_fn(
          1, 0, nullptr, attribute, signal);
  CHECK_HSA_STATUS("Error: hsa_amd_signal_create failed", status);
}

template <typename Integral = uint64_t> constexpr Integral bit_mask(int first, int last) {
  assert(last >= first && "Error: hsa_support::bit_mask -> invalid argument");
  size_t num_bits = last - first + 1;
  return ((num_bits >= sizeof(Integral) * 8) ? ~Integral{0}
                                             /* num_bits exceed the size of Integral */
                                             : ((Integral{1} << num_bits) - 1))
      << first;
}

/* Extract bits [last:first] from t.  */
template <typename Integral> constexpr Integral bit_extract(Integral x, int first, int last) {
  return (x >> first) & bit_mask<Integral>(0, last - first);
}

rocprofiler_session_id_t session_id = rocprofiler_session_id_t{0};
// Counter Names declaration
std::vector<std::string> session_data;

rocprofiler_buffer_id_t buffer_id;

uint64_t session_data_count = 0;

bool is_counter_collection_mode = false;
bool is_timestamp_collection_mode = false;
bool is_att_collection_mode = false;
bool is_pc_sampling_collection_mode = false;
std::vector<rocprofiler_att_parameter_t> att_parameters_data;
uint32_t replay_mode_count = 0;
std::vector<std::string> kernel_profile_names;
std::vector<uint64_t> kernel_profile_dispatch_ids;
std::vector<std::string> att_counters_names;

rocprofiler::Session* session = nullptr;

void ResetSessionID(rocprofiler_session_id_t id) { session_id = id; }

void CheckNeededProfileConfigs() {
  rocprofiler_session_id_t internal_session_id;
  // Getting Session ID
  rocprofiler::ROCProfiler_Singleton& rocprofiler_singleton =
      rocprofiler::ROCProfiler_Singleton::GetInstance();
  internal_session_id = rocprofiler_singleton.GetCurrentSessionId();


  if (session_id.handle == 0 || internal_session_id.handle != session_id.handle) {
    session_id = internal_session_id;
    // Getting Counters count from the Session
    if (session_id.handle > 0) {
      session = rocprofiler_singleton.GetSession(session_id);
      if (session && session->FindFilterWithKind(ROCPROFILER_COUNTERS_COLLECTION)) {
        rocprofiler_filter_id_t filter_id =
            session->GetFilterIdWithKind(ROCPROFILER_COUNTERS_COLLECTION);
        rocprofiler::Filter* filter = session->GetFilter(filter_id);
        session_data = filter->GetCounterData();
        is_counter_collection_mode = true;
        session_data_count = session_data.size();
        buffer_id = filter->GetBufferId();
      } else if (session &&
                 session->FindFilterWithKind(ROCPROFILER_DISPATCH_TIMESTAMPS_COLLECTION)) {
        is_timestamp_collection_mode = true;
        rocprofiler_filter_id_t filter_id =
            session->GetFilterIdWithKind(ROCPROFILER_DISPATCH_TIMESTAMPS_COLLECTION);
        rocprofiler::Filter* filter = session->GetFilter(filter_id);
        buffer_id = filter->GetBufferId();
      } else if (session && session->FindFilterWithKind(ROCPROFILER_ATT_TRACE_COLLECTION)) {
        rocprofiler_filter_id_t filter_id =
            session->GetFilterIdWithKind(ROCPROFILER_ATT_TRACE_COLLECTION);
        rocprofiler::Filter* filter = session->GetFilter(filter_id);
        att_parameters_data = filter->GetAttParametersData();
        is_att_collection_mode = true;
        buffer_id =
            session->GetFilter(session->GetFilterIdWithKind(ROCPROFILER_ATT_TRACE_COLLECTION))
                ->GetBufferId();

        att_counters_names = filter->GetCounterData();
        kernel_profile_names = std::get<std::vector<std::string>>(
            filter->GetProperty(ROCPROFILER_FILTER_KERNEL_NAMES));
        kernel_profile_dispatch_ids =
            std::get<std::vector<uint64_t>>(filter->GetProperty(ROCPROFILER_FILTER_DISPATCH_IDS));
      } else if (session && session->FindFilterWithKind(ROCPROFILER_PC_SAMPLING_COLLECTION)) {
        is_pc_sampling_collection_mode = true;
      }
    }
  }
}

static int KernelInterceptCount = 0;
std::atomic<uint32_t> WRITER_ID{0};

std::pair<std::vector<bool>, bool> GetAllowedProfilesList(const void* packets, int pkt_count) {
  std::vector<bool> can_profile_packet;
  bool b_can_profile_anypacket = false;
  can_profile_packet.reserve(pkt_count);
  rocprofiler::HSASupport_Singleton& hsasupport_singleton =
      rocprofiler::HSASupport_Singleton::GetInstance();
  std::lock_guard<std::mutex> lock(hsasupport_singleton.ksymbol_map_lock);
  assert(hsasupport_singleton.ksymbols);

  uint32_t current_writer_id = WRITER_ID.load(std::memory_order_relaxed);

  for (int i = 0; i < pkt_count; ++i) {
    auto& original_packet = static_cast<const hsa_barrier_and_packet_t*>(packets)[i];
    bool b_profile_this_object = false;

    // Skip packets other than kernel dispatch packets.
    if (bit_extract(original_packet.header, HSA_PACKET_HEADER_TYPE,
                    HSA_PACKET_HEADER_TYPE + HSA_PACKET_HEADER_WIDTH_TYPE - 1) ==
        HSA_PACKET_TYPE_KERNEL_DISPATCH) {
      auto& kdispatch = static_cast<const hsa_kernel_dispatch_packet_s*>(packets)[i];

      // If Dispatch IDs specified, profile based on dispatch ID
      for (auto id : kernel_profile_dispatch_ids) b_profile_this_object |= id == current_writer_id;
      try {
        // Can throw
        const std::string& kernel_name = hsasupport_singleton.ksymbols->at(kdispatch.kernel_object);

        // If no filters specified, auto profile this kernel
        if (kernel_profile_names.size() == 0 && kernel_profile_dispatch_ids.size() == 0 &&
            kernel_name.find("__amd_rocclr_") == std::string::npos)
          b_profile_this_object = true;

        // Try to match the mangled kernel name with given matches in input.txt
        // We want to initiate att profiling if a match exists
        for (const std::string& kernel_matches : kernel_profile_names)
          if (kernel_name.find(kernel_matches) != std::string::npos) b_profile_this_object = true;
      } catch (...) {
        rocprofiler::warning("Warning: Unknown name for object %lu\n", kdispatch.kernel_object);
      }
      current_writer_id += 1;
    }
    b_can_profile_anypacket |= b_profile_this_object;
    can_profile_packet.push_back(b_profile_this_object);
  }
  // If we're going to skip all packets, need to update writer ID
  if (!b_can_profile_anypacket) WRITER_ID.store(current_writer_id, std::memory_order_release);
  return {can_profile_packet, b_can_profile_anypacket};
}

std::pair<hsa_ven_amd_aqlprofile_profile_t*, rocprofiler_codeobj_capture_mode_t> ProcessATTParams(
    Packet::packet_t& start_packet, Packet::packet_t& stop_packet, Queue& queue_info,
    rocprofiler::HSAAgentInfo& agentInfo) {
  std::vector<hsa_ven_amd_aqlprofile_parameter_t> att_params;
  int num_att_counters = 0;
  uint32_t att_buffer_size = DEFAULT_ATT_BUFFER_SIZE;
  rocprofiler_codeobj_capture_mode_t capture_mode = ROCPROFILER_CAPTURE_SYMBOLS_ONLY;

  for (rocprofiler_att_parameter_t& param : att_parameters_data) {
    switch (param.parameter_name) {
      case ROCPROFILER_ATT_PERFCOUNTER_NAME:
        break;
      case ROCPROFILER_ATT_CAPTURE_MODE:
        capture_mode = static_cast<rocprofiler_codeobj_capture_mode_t>(param.value);
        break;
      case ROCPROFILER_ATT_BUFFER_SIZE:
        att_buffer_size =
            std::max(96l << 10l, std::min(int64_t(param.value) << 20l, (1l << 32l) - (3l << 20)));
        break;  // Clip to [96KB, 4GB)
      case ROCPROFILER_ATT_PERFCOUNTER:
        num_att_counters += 1;
        break;
      default:
        att_params.push_back(
            {static_cast<hsa_ven_amd_aqlprofile_parameter_name_t>(int(param.parameter_name)),
             param.value});
    }
  }

  if (att_counters_names.size() > 0) {
    MetricsDict* metrics_dict_ = MetricsDict::Create(&agentInfo);

    for (const std::string& counter_name : att_counters_names) {
      const Metric* metric = metrics_dict_->Get(counter_name);
      const BaseMetric* base = dynamic_cast<const BaseMetric*>(metric);
      if (!base) rocprofiler::fatal("Invalid base metric value: %s\n", counter_name.c_str());

      std::vector<const counter_t*> counters;
      base->GetCounters(counters);
      hsa_ven_amd_aqlprofile_event_t event = counters[0]->event;
      if (event.block_name != HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_SQ)
        rocprofiler::fatal("Only events from the SQ block can be selected for ATT.\n");

      att_params.push_back(
          {static_cast<hsa_ven_amd_aqlprofile_parameter_name_t>(int(ROCPROFILER_ATT_PERFCOUNTER)),
           event.counter_id | (event.counter_id ? (0xF << 24) : 0)});
      num_att_counters += 1;
    }

    hsa_ven_amd_aqlprofile_parameter_t zero_perf = {
        static_cast<hsa_ven_amd_aqlprofile_parameter_name_t>(int(ROCPROFILER_ATT_PERFCOUNTER)), 0};

    // Fill other perfcounters with 0's
    for (; num_att_counters < 16; num_att_counters++) att_params.push_back(zero_perf);
  }
  // Get the PM4 Packets using packets_generator
  return {Packet::GenerateATTPackets(queue_info.GetCPUAgent(), queue_info.GetGPUAgent(), att_params,
                                     &start_packet, &stop_packet, att_buffer_size),
          capture_mode};
}

/**
 * @brief This function is a queue write interceptor. It intercepts the
 * packet write function. Creates an instance of packet class with the raw
 * pointer. invoke the populate function of the packet class which returns a
 * pointer to the packet. This packet is written into the queue by this
 * interceptor by invoking the writer function.
 */
void Queue::WriteInterceptor(const void* packets, uint64_t pkt_count, uint64_t user_pkt_index,
                             void* data, hsa_amd_queue_intercept_packet_writer writer) {
  static const char* env_MAX_ATT_PROFILES = getenv("ROCPROFILER_MAX_ATT_PROFILES");
  static int MAX_ATT_PROFILES = env_MAX_ATT_PROFILES ? atoi(env_MAX_ATT_PROFILES) : 1;

  const Packet::packet_t* packets_arr = reinterpret_cast<const Packet::packet_t*>(packets);
  std::vector<Packet::packet_t> transformed_packets;

  CheckNeededProfileConfigs();
  rocprofiler_session_id_t session_id_snapshot = session_id;

  if (session_id_snapshot.handle > 0 && pkt_count > 0 &&
      (is_counter_collection_mode || is_timestamp_collection_mode ||
       is_pc_sampling_collection_mode) &&
      session) {
    // Getting Queue Data and Information
    auto& queue_info = *reinterpret_cast<Queue*>(data);
    std::lock_guard<std::mutex> lk(queue_info.qw_mutex);


    // hsa_ven_amd_aqlprofile_profile_t* profile;
    std::vector<std::pair<rocprofiler::profiling_context_t*, hsa_ven_amd_aqlprofile_profile_t*>>
        profiles;
    // Searching accross all the packets given during this write
    for (size_t i = 0; i < pkt_count; ++i) {
      auto& original_packet = static_cast<const hsa_barrier_and_packet_t*>(packets)[i];

      // +Skip kernel dispatch IDs not wanted
      // Skip packets other than kernel dispatch packets.
      if (session_id_snapshot.handle == 0 ||
          bit_extract(original_packet.header, HSA_PACKET_HEADER_TYPE,
                      HSA_PACKET_HEADER_TYPE + HSA_PACKET_HEADER_WIDTH_TYPE - 1) !=
              HSA_PACKET_TYPE_KERNEL_DISPATCH) {
        transformed_packets.emplace_back(packets_arr[i]);
        continue;
      }

      // If counters found in the session
      if (session_data_count > 0 && is_counter_collection_mode) {
        // Get the PM4 Packets using packets_generator
        profiles = Packet::InitializeAqlPackets(queue_info.GetCPUAgent(), queue_info.GetGPUAgent(),
                                                session_data, session_id_snapshot);
        replay_mode_count = profiles.size();

      }

      uint32_t profile_id = 0;
      // do {

      std::pair<rocprofiler::profiling_context_t*, hsa_ven_amd_aqlprofile_profile_t*> profile;
      if (profiles.size() > 0 && replay_mode_count > 0) {
         profile = profiles.at(profile_id);
        hsa_signal_t ready_signal = queue_info.GetReadySignal();
        hsa_signal_t block_signal = queue_info.GetBlockSignal();

      /*
       Creates a barrier packet with its completion signal as the
       queue's ready signal.
       */
      CreateBarrierPacket(&transformed_packets, nullptr, &ready_signal);
      /*
      Creates a barrier packet with queue's blocksignal as its input and
      completion signal.This will ensure it is no longer 0 so a later barrier
      packet waiting on it to be 0 will be blocked
      */
      CreateBarrierPacket(&transformed_packets, &block_signal, &block_signal);
      }


      uint32_t writer_id = WRITER_ID.fetch_add(1, std::memory_order_release);

      if (session_data_count > 0 && is_counter_collection_mode && profiles.size() > 0 &&
          replay_mode_count > 0 && profile.first && profile.first->start_packet) {
        // Adding start packet and its barrier with a dummy signal
        hsa_signal_t dummy_signal{};
        dummy_signal.handle = 0;
        profile.first->start_packet->header = HSA_PACKET_TYPE_VENDOR_SPECIFIC
            << HSA_PACKET_HEADER_TYPE;
        AddVendorSpecificPacket(profile.first->start_packet, &transformed_packets, dummy_signal);

        CreateBarrierPacket(&transformed_packets, &profile.first->start_packet->completion_signal,
                            nullptr);
      }

      auto& packet = transformed_packets.emplace_back(packets_arr[i]);
      auto& dispatch_packet = reinterpret_cast<hsa_kernel_dispatch_packet_t&>(packet);
      uint64_t correlation_id = dispatch_packet.reserved2;

      CreateSignal(HSA_AMD_SIGNAL_AMD_GPU_ONLY, &packet.completion_signal);

      // Adding the dispatch packet newly created signal to the pending signals
      // list to be processed by the signal interrupt
      rocprofiler_kernel_properties_t kernel_properties =
          set_kernel_properties(dispatch_packet, queue_info.GetGPUAgent());
      if (session) {
        uint64_t record_id = rocprofiler::ROCProfiler_Singleton::GetInstance().GetUniqueRecordId();
        AddKernelNameWithDispatchID(GetKernelNameFromKsymbols(dispatch_packet.kernel_object),
                                    record_id);
        if (session_data_count > 0 && profile.second) {
          session->GetProfiler()->AddPendingSignals(
              writer_id, record_id, original_packet.completion_signal, packet.completion_signal,
              session_id, buffer_id, profile.first, session_data_count, profile.second,
              kernel_properties, (uint32_t)syscall(__NR_gettid), user_pkt_index, correlation_id);
        } else {
          session->GetProfiler()->AddPendingSignals(
              writer_id, record_id, original_packet.completion_signal, packet.completion_signal,
              session_id, buffer_id, nullptr, session_data_count, nullptr, kernel_properties,
              (uint32_t)syscall(__NR_gettid), user_pkt_index, correlation_id);
        }
      }

      // Make a copy of the original packet, adding its signal to a barrier
      // packet and create a new signal for it to get timestamps
      if (original_packet.completion_signal.handle) {
        hsa_barrier_and_packet_t barrier{};
        barrier.header = (HSA_PACKET_TYPE_BARRIER_AND << HSA_PACKET_HEADER_TYPE) |
            (1 << HSA_PACKET_HEADER_BARRIER) |
            (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_ACQUIRE_FENCE_SCOPE) |
            (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_RELEASE_FENCE_SCOPE);
        Packet::packet_t* __attribute__((__may_alias__)) pkt =
            (reinterpret_cast<Packet::packet_t*>(&barrier));
        transformed_packets.emplace_back(*pkt).completion_signal =
            original_packet.completion_signal;

        {
          std::lock_guard<std::mutex> lock(
              HSASupport_Singleton::GetInstance().signals_timestamps_map_lock);
          HSASupport_Singleton::GetInstance()
              .signals_timestamps[original_packet.completion_signal.handle] =
              new_signal_timestamp_t{packet.completion_signal, std::nullopt};
        }
      }

      hsa_signal_t interrupt_signal{};
      // Adding a barrier packet with the original packet's completion signal.
      CreateSignal(0, &interrupt_signal);

      // Adding Stop and Read PM4 Packets
      if (session_data_count > 0 && is_counter_collection_mode && profiles.size() > 0 &&
          profile.first && profile.first->stop_packet) {
        hsa_signal_t dummy_signal{};
        profile.first->stop_packet->header = HSA_PACKET_TYPE_VENDOR_SPECIFIC
            << HSA_PACKET_HEADER_TYPE;
        AddVendorSpecificPacket(profile.first->stop_packet, &transformed_packets, dummy_signal);
        profile.first->read_packet->header = HSA_PACKET_TYPE_VENDOR_SPECIFIC
            << HSA_PACKET_HEADER_TYPE;
        AddVendorSpecificPacket(profile.first->read_packet, &transformed_packets, interrupt_signal);

        // Added Interrupt Signal with barrier and provided handler for it
        CreateBarrierPacket( &transformed_packets, &interrupt_signal, nullptr);


      }
      else
        CreateBarrierPacket( &transformed_packets, nullptr, &interrupt_signal);
      rocprofiler::HSAAgentInfo& agentInfo =
       rocprofiler::HSASupport_Singleton::GetInstance().GetHSAAgentInfo(
         queue_info.GetGPUAgent().handle);
      //  Creating Async Handler to be called every time the interrupt signal is
      //  marked complete
      SignalAsyncHandler(
          interrupt_signal,
          new queue_info_session_t{
              queue_info.GetGPUAgent(), session_id_snapshot, queue_info.GetQueueID(), writer_id,
              interrupt_signal, agentInfo.GetDeviceInfo().getGPUId(),
              agentInfo.GetDeviceInfo().getXccCount(), queue_info.GetBlockSignal()});
      ACTIVE_INTERRUPT_SIGNAL_COUNT.fetch_add(1, std::memory_order_relaxed);
      // profile_id++;
      // } while (replay_mode_count > 0 && profile_id < replay_mode_count);  // Profiles loop end

  }
    /* Write the transformed packets to the hardware queue.  */
    writer(&transformed_packets[0], transformed_packets.size());
  } else if (session_id_snapshot.handle > 0 && pkt_count > 0 && is_att_collection_mode && session &&
             KernelInterceptCount < MAX_ATT_PROFILES) {
    // att start
    // Getting Queue Data and Information
    auto& queue_info = *static_cast<Queue*>(data);
    std::lock_guard<std::mutex> lk(queue_info.qw_mutex);
    rocprofiler::HSAAgentInfo& agentInfo =
        rocprofiler::HSASupport_Singleton::GetInstance().GetHSAAgentInfo(
            queue_info.GetGPUAgent().handle);

    bool can_profile_anypacket = false;
    std::vector<bool> can_profile_packet;
    std::tie(can_profile_packet, can_profile_anypacket) =
        GetAllowedProfilesList(packets, pkt_count);

    if (!can_profile_anypacket) {
      /* Write the original packets to the hardware if no patch will be profiled */
      writer(packets, pkt_count);
      return;
    }

    // Preparing att Packets
    Packet::packet_t start_packet{};
    Packet::packet_t stop_packet{};
    hsa_ven_amd_aqlprofile_profile_t* profile = nullptr;
    rocprofiler_codeobj_capture_mode_t capture_mode = ROCPROFILER_CAPTURE_SYMBOLS_ONLY;

    if (att_parameters_data.size() > 0) {
      std::tie(profile, capture_mode) =
          ProcessATTParams(start_packet, stop_packet, queue_info, agentInfo);
    }

    // Searching across all the packets given during this write
    for (size_t i = 0; i < pkt_count; ++i) {
      auto& original_packet = static_cast<const hsa_barrier_and_packet_t*>(packets)[i];
      uint32_t writer_id = 0;

      // Skip all packets marked with !can_profile
      if (i >= can_profile_packet.size() || can_profile_packet[i] == false) {
        transformed_packets.emplace_back(packets_arr[i]);

        // increment writer ID for every packet
        if (bit_extract(original_packet.header, HSA_PACKET_HEADER_TYPE,
                        HSA_PACKET_HEADER_TYPE + HSA_PACKET_HEADER_WIDTH_TYPE - 1) ==
            HSA_PACKET_TYPE_KERNEL_DISPATCH)
          writer_id = WRITER_ID.fetch_add(1, std::memory_order_release);

        continue;
      }
      KernelInterceptCount += 1;
      writer_id = WRITER_ID.fetch_add(1, std::memory_order_release);

      if (!att_parameters_data.empty() && profile) {
        // Adding start packet and its barrier with a dummy signal
        hsa_signal_t dummy_signal{};
        dummy_signal.handle = 0;
        start_packet.header = HSA_PACKET_TYPE_VENDOR_SPECIFIC << HSA_PACKET_HEADER_TYPE;
        AddVendorSpecificPacket(&start_packet, &transformed_packets, dummy_signal);
        CreateBarrierPacket(&transformed_packets, &start_packet.completion_signal, nullptr) ;
      }

      auto& packet = transformed_packets.emplace_back(packets_arr[i]);
      auto& dispatch_packet = reinterpret_cast<hsa_kernel_dispatch_packet_t&>(packet);

      CreateSignal(HSA_AMD_SIGNAL_AMD_GPU_ONLY, &packet.completion_signal);

      // Adding the dispatch packet newly created signal to the pending signals
      // list to be processed by the signal interrupt
      rocprofiler_kernel_properties_t kernel_properties =
          set_kernel_properties(dispatch_packet, queue_info.GetGPUAgent());
      uint64_t record_id = rocprofiler::ROCProfiler_Singleton::GetInstance().GetUniqueRecordId();
      AddKernelNameWithDispatchID(GetKernelNameFromKsymbols(dispatch_packet.kernel_object),
                                  record_id);

      session->GetAttTracer()->AddPendingSignals(
          writer_id, record_id, original_packet.completion_signal, packet.completion_signal,
          session_id_snapshot, buffer_id, profile, kernel_properties,
          (uint32_t)syscall(__NR_gettid), user_pkt_index);

      uint64_t userdata = HSASupport_Singleton::GetInstance()
                          .GetHSAAgentInfo(queue_info.GetGPUAgent().handle)
                          .GetDeviceInfo()
                          .getName()
                          .find("gfx9") != std::string::npos;
      codeobj_record::make_capture(rocprofiler_record_id_t{record_id}, capture_mode, userdata);
      codeobj_record::start_capture(rocprofiler_record_id_t{record_id});
      codeobj_record::stop_capture(rocprofiler_record_id_t{record_id});

      // Make a copy of the original packet, adding its signal to a barrier packet
      if (original_packet.completion_signal.handle != 0U) {
        hsa_barrier_and_packet_t barrier{};
        barrier.header = HSA_PACKET_TYPE_BARRIER_AND << HSA_PACKET_HEADER_TYPE;
        barrier.dep_signal[0] = packet.completion_signal;
        Packet::packet_t* __attribute__((__may_alias__)) pkt =
            (reinterpret_cast<Packet::packet_t*>(&barrier));
        transformed_packets.emplace_back(*pkt).completion_signal =
            original_packet.completion_signal;
      }

      // Adding a barrier packet with the original packet's completion signal.
      hsa_signal_t interrupt_signal;
      CreateSignal(0, &interrupt_signal);

      // Adding Stop PM4 Packets
      if (att_parameters_data.size() > 0 && is_att_collection_mode && profile) {
        stop_packet.header = HSA_PACKET_TYPE_VENDOR_SPECIFIC << HSA_PACKET_HEADER_TYPE;
        AddVendorSpecificPacket(&stop_packet, &transformed_packets, interrupt_signal);

        // Added Interrupt Signal with barrier and provided handler for it
        CreateBarrierPacket(&transformed_packets, &interrupt_signal, nullptr);
      } else {
        hsa_barrier_and_packet_t barrier{};
        barrier.header = HSA_PACKET_TYPE_BARRIER_AND << HSA_PACKET_HEADER_TYPE;
        barrier.completion_signal = interrupt_signal;
        Packet::packet_t* __attribute__((__may_alias__)) pkt =
            (reinterpret_cast<Packet::packet_t*>(&barrier));
        transformed_packets.emplace_back(*pkt);
      }

      // Creating Async Handler to be called every time the interrupt signal is
      // marked complete
      signalAsyncHandlerATT(
          interrupt_signal,
          new queue_info_session_t{queue_info.GetGPUAgent(), session_id_snapshot,
                                   queue_info.GetQueueID(), writer_id, interrupt_signal});
    }
    /* Write the transformed packets to the hardware queue.  */
    writer(&transformed_packets[0], transformed_packets.size());
    // ATT end
  } else {
    /* Write the original packets to the hardware queue if no profiling session
     * is active  */
    writer(packets, pkt_count);
  }
}


Queue::Queue(const hsa_agent_t cpu_agent, const hsa_agent_t gpu_agent, hsa_queue_t* queue)
    : cpu_agent_(cpu_agent), gpu_agent_(gpu_agent), intercept_queue_(queue) {
  state = is_destroy::normal;
  CreateSignal(0, &block_signal_);
  CreateSignal(0, &ready_signal_);
  SignalAsyncReadyHandler(ready_signal_, this);
}

Queue::~Queue() {
  std::unique_lock<std::mutex> queue_lock(qw_mutex);
  {
    profiler_serializer_t& serializer =
        rocprofiler::ROCProfiler_Singleton::GetInstance().GetSerializer();
    // std::cout << GetROCMToolObj()->
    std::lock_guard<std::mutex> serializer_lock(serializer.serializer_mutex);
    for (auto it = serializer.dispatch_ready.begin(); it != serializer.dispatch_ready.end();) {
      if ((*it)->GetQueueID() == GetQueueID()) {
        /*Deletes the queue to be destructed from the dispatch ready.*/
        serializer.dispatch_ready.erase(it);
        if (serializer.dispatch_queue->GetQueueID() == GetQueueID())
          // ToDO [srnagara]: Need to find a solution rather than abort.
          fatal("Queue is being destroyed while kernel launch is still active");
      }
    }
    state = is_destroy::to_destroy;

    rocprofiler::HSASupport_Singleton::GetInstance()
        .GetCoreApiTable()
        .hsa_signal_store_screlease_fn(ready_signal_, 0);
  }
  this->cv_ready_signal.wait(queue_lock, [this] { return state == is_destroy::done_destroy; });

  if (block_signal_.handle)
    rocprofiler::HSASupport_Singleton::GetInstance().GetCoreApiTable().hsa_signal_destroy_fn(
        block_signal_);
}

hsa_queue_t* Queue::GetCurrentInterceptQueue() { return intercept_queue_; }

hsa_agent_t Queue::GetGPUAgent() { return gpu_agent_; }

hsa_agent_t Queue::GetCPUAgent() { return cpu_agent_; }

uint64_t Queue::GetQueueID() { return intercept_queue_->id; }

void CheckPacketReqiurements() {
  Packet::CheckPacketReqiurements();
}
hsa_signal_t Queue::GetReadySignal() { return ready_signal_; }

hsa_signal_t Queue::GetBlockSignal() { return block_signal_; }





}  // namespace queue
}  // namespace rocprofiler
