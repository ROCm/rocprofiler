#ifndef INC_ROCTRACER_TRACE_ENTRIES_H_
#define INC_ROCTRACER_TRACE_ENTRIES_H_

#include <cstdint>

struct metric_trace_entry_t {
  uint32_t dispatch;
  const char* name;
  uint64_t result;
};

#endif