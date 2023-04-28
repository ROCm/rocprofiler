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

#ifndef SRC_TOOLS_TRACER_TRACER_H_
#define SRC_TOOLS_TRACER_TRACER_H_

#include <atomic>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "rocprofiler.h"
#include "src/roctracer.h"

typedef bool is_filtered_domain_t;

typedef struct {
  rocprofiler_sync_callback_t user_sync_callback;
  rocprofiler_session_id_t session_id;
} api_callback_data_t;

namespace rocmtools {
namespace tracer {

class Tracer {
 public:
  // Getting Buffer and/or sync callback
  Tracer(rocprofiler_session_id_t session_id, rocprofiler_sync_callback_t callback,
         rocprofiler_buffer_id_t buffer_id, std::vector<rocprofiler_tracer_activity_domain_t> domains);
  ~Tracer();

  rocprofiler_tracer_api_data_handle_t AddROCTxApiData(std::string api_data);
  rocprofiler_tracer_api_data_handle_t AddHSAApiData(hsa_api_data_t api_data);
  rocprofiler_tracer_api_data_handle_t AddHIPApiData(hip_api_data_t api_data);

  bool FindROCTxApiData(rocprofiler_tracer_api_data_handle_t api_data_handler);
  bool FindHSAApiData(rocprofiler_tracer_api_data_handle_t api_data_handler);
  bool FindHIPApiData(rocprofiler_tracer_api_data_handle_t api_data_handler);

  size_t GetROCTxApiDataInfoSize(rocprofiler_tracer_roctx_api_data_info_t kind,
                                 rocprofiler_tracer_api_data_handle_t api_data_id,
                                 rocprofiler_tracer_operation_id_t operation_id);
  size_t GetHSAApiDataInfoSize(rocprofiler_tracer_hsa_api_data_info_t kind,
                               rocprofiler_tracer_api_data_handle_t api_data_id,
                               rocprofiler_tracer_operation_id_t operation_id);
  size_t GetHIPApiDataInfoSize(rocprofiler_tracer_hip_api_data_info_t kind,
                               rocprofiler_tracer_api_data_handle_t api_data_id,
                               rocprofiler_tracer_operation_id_t operation_id);

  char* GetROCTxApiDataInfo(rocprofiler_tracer_roctx_api_data_info_t kind,
                            rocprofiler_tracer_api_data_handle_t api_data_id,
                            rocprofiler_tracer_operation_id_t operation_id);
  char* GetHSAApiDataInfo(rocprofiler_tracer_hsa_api_data_info_t kind,
                          rocprofiler_tracer_api_data_handle_t api_data_id,
                          rocprofiler_tracer_operation_id_t operation_id);
  char* GetHIPApiDataInfo(rocprofiler_tracer_hip_api_data_info_t kind,
                          rocprofiler_tracer_api_data_handle_t api_data_id,
                          rocprofiler_tracer_operation_id_t operation_id);

  void InitRoctracer(
      const std::map<rocprofiler_tracer_activity_domain_t, is_filtered_domain_t>& domains,
      const std::vector<std::string>& api_filter_data_vector);

  std::mutex& GetTracerLock();
  void DisableRoctracer();
  void StartRoctracer();
  void StopRoctracer();

 private:
  std::atomic<bool> is_active_{false};
  std::atomic<bool> roctracer_initiated_{false};
  std::atomic<int (*)(rocprofiler_tracer_activity_domain_t domain, uint32_t operation_id, void* data)>
      roctx_report_activity_;

  std::vector<rocprofiler_tracer_activity_domain_t> domains_;
  bool is_sync_;
  rocprofiler_sync_callback_t callback_;
  rocprofiler_buffer_id_t buffer_id_;
  rocprofiler_session_id_t session_id_;
  api_callback_data_t callback_data_;
  std::mutex tracer_lock_;
};

}  // namespace tracer
}  // namespace rocmtools

#endif  // SRC_TOOLS_TRACER_TRACER_H_
