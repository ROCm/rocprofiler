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

#ifndef SRC_CORE_HARDWARE_HSA_INFO_H_
#define SRC_CORE_HARDWARE_HSA_INFO_H_

#include <hsa/hsa.h>
#include <hsa/hsa_api_trace.h>

#include <atomic>
#include <map>
#include <mutex>
#include <string>
#include <string_view>
#include <vector>

namespace Agent {

static const uint32_t LDS_BLOCK_SIZE = 128 * 4;

// XXX TODO: This should be merged into rocmtools::hsa_support::AgentInfo and
// this file should be removed entirely, as it's completely redundant
class AgentInfo {
 public:
  AgentInfo();
  AgentInfo(const hsa_agent_t agent, ::CoreApiTable* table);

  int getIndex() const;
  hsa_device_type_t getType() const;
  uint64_t getHandle() const;
  const std::string_view getName() const;

  std::string getGfxip() const;
  uint32_t getMaxWaveSize() const;
  uint32_t getMaxQueueSize() const;
  uint32_t getCUCount() const;
  uint32_t getSimdCountPerCU() const;
  uint32_t getShaderEngineCount() const;
  uint32_t getShaderArraysPerSE() const;
  uint32_t getMaxWavesPerCU() const;
  uint32_t getCUCountPerSH() const;
  uint32_t getWaveSlotsPerSimd() const;
  uint32_t getPCIDomain() const;
  uint32_t getPCILocationID() const;

  void setIndex(int index);
  void setType(hsa_device_type_t type);
  void setHandle(uint64_t handle);
  void setName(const std::string& name);

 private:
  int index_;
  hsa_device_type_t type_;  // Agent type - Cpu = 0, Gpu = 1 or Dsp = 2
  uint64_t handle_;
  char name_[64];
  char gfxip_[64];
  uint32_t max_wave_size_;
  uint32_t max_queue_size_;
  uint32_t cu_num_;
  uint32_t simds_per_cu_;
  uint32_t se_num_;
  uint32_t shader_arrays_per_se_;
  uint32_t waves_per_cu_;
  // CUs per SH/SA
  uint32_t compute_units_per_sh_;
  uint32_t wave_slots_per_simd_;

  uint32_t pci_domain_;
  uint32_t pci_location_id_;
};

// XXX TODO: This should be moved somewhere else so this file can be deleted
class CounterHardwareInfo {
 public:
  CounterHardwareInfo(uint64_t event_id, const char* block_id);
  int64_t getNumInstances();

 private:
  uint64_t register_offset_;
  uint64_t register_address_;
  int64_t num_instances_;
  hsa_agent_t agent_;
  uint64_t event_id_;
  const char* block_id_;
};

// XXX TODO: This too
bool getHardwareInfo(uint64_t event_id, const char* block_id,
                     CounterHardwareInfo* counter_hardware_info);

}  // namespace Agent

#endif  // SRC_CORE_HARDWARE_HSA_INFO_H_
