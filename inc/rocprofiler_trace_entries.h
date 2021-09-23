#ifndef INC_ROCTRACER_TRACE_ENTRIES_H_
#define INC_ROCTRACER_TRACE_ENTRIES_H_

#include <cstdint>

struct metric_trace_entry_t {
  uint32_t dispatch;
  const char* name;
  uint64_t result;
};

struct kernel_trace_entry_t {
  uint32_t dispatch;
  uint32_t gpu_id;
  uint32_t queue_id;
  uint64_t queue_index;
  uint32_t pid;
  uint32_t tid;
  uint32_t grid_size;
  uint32_t workgroup_size;
  uint32_t lds_size;
  uint32_t scratch_size;
  uint32_t vgpr;
  uint32_t sgpr;
  uint32_t fbarrier_count;
  uint64_t signal_handle;
  uint64_t object;
  const char* kernel_name;
  bool record;
  uint64_t dispatch_time;
  uint64_t begin;
  uint64_t end;
  uint64_t complete;
};

#endif