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

#include <hsa.h>
#include <stdio.h>
#include <string.h>
#include <iostream>
#include <thread>

#include "ctrl/run_kernel.h"
#include "ctrl/test_aql.h"
#include "dummy_kernel/dummy_kernel.h"
#include "simple_convolution/simple_convolution.h"
#include <future>
#include <memory>

bool thread_fun(const int kiter, const int diter, const uint32_t agents_number) {
  bool result = true;
  const AgentInfo* agent_info[agents_number];
  std::vector<std::unique_ptr<hsa_queue_t, decltype(&hsa_queue_destroy)>> queue;
  HsaRsrcFactory* rsrc = &HsaRsrcFactory::Instance();

  for (uint32_t n = 0; n < agents_number; ++n) {
    uint32_t agent_id = n % rsrc->GetCountOfGpuAgents();
    if (rsrc->GetGpuAgentInfo(agent_id, &agent_info[n]) == false) {
      fprintf(stderr, "AgentInfo failed\n");
      abort();
    }
    queue.emplace_back(rsrc->CreateQueue(agent_info[n], 128), &hsa_queue_destroy);
    if (!queue.back()) {
      fprintf(stderr, "CreateQueue failed\n");
      abort();
    }
  }

  for (int i = 0; i < kiter && result; ++i) {
    for (uint32_t n = 0; n < agents_number && result; ++n) {
      // RunKernel<DummyKernel, TestAql>(0, NULL, agent_info[n], queue[n], diter);
      result &= RunKernel<SimpleConvolution, TestAql>(0, NULL, agent_info[n], queue[n].get(), diter);
    }
  }

  return result;
}

int main(int argc, char** argv) {
  bool result = true;
  const char* kiter_s = getenv("ROCP_KITER");
  const char* diter_s = getenv("ROCP_DITER");
  const char* agents_s = getenv("ROCP_AGENTS");
  const char* thrs_s = getenv("ROCP_THRS");

  const int kiter = (kiter_s != NULL) ? atol(kiter_s) : 1;
  const int diter = (diter_s != NULL) ? atol(diter_s) : 1;
  const uint32_t agents_number = (agents_s != NULL) ? (uint32_t)atol(agents_s) : 1;
  const int thrs = (thrs_s != NULL) ? atol(thrs_s) : 1;

  TestHsa::HsaInstantiate();

  std::vector<std::future<bool>> futures;
  for (int n = 0; n < thrs; ++n) {
    futures.emplace_back(std::async(thread_fun, kiter, diter, agents_number));
  }
  
  for (auto &f : futures) {
    result &= f.get();
  }

  TestHsa::HsaShutdown();
  return result ? 0 : 1;
}
