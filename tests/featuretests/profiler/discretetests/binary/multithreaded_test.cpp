/*
Copyright (c) 2015-2016 Advanced Micro Devices, Inc. All rights reserved.

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
*/

/** \mainpage ROC Profiler Binary Test
 *
 * \section introduction Introduction
 *
 * The goal of this test is to test ROC profiler as a binary against a
 * multithreaded application.Test application launches an empty kernel
 * on multiple threads.
 *
 * The test then parses the csv and verifies if the nuber of kernel dispatches
 *  are equal to number of threads launched in test application.
 *
 * Test also does some basic verification if counter values are non-negative
 */

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "utils/csv_parser.h"
#include "utils/test_utils.h"

// kernel dispatch count test
int DispatchCountTest(std::string profiler_output) {
  CSVParser parser;
  parser.ParseCSV(profiler_output);
  countermap counter_map = parser.GetCounterMap();

  int dispatch_counter = 0;
  for (auto i = 0; i < counter_map.size(); i++) {
    std::string* dispatch_id = parser.ReadCounter(i, 1);
    if (dispatch_id != nullptr) {
      if (dispatch_id->find("dispatch") != std::string::npos) {
        dispatch_counter++;
      }
    }
  }

  // dispatch count test: Number of dispatches must be equal to
  // number of kernel launches in test_app
  if (dispatch_counter == GetNumberOfCores()) {
    return 0;
  }
  return -1;
}

std::string ReadProfilerBuffer(const char* cmd) {
  std::vector<char> buffer(1028);
  std::string profiler_output;
  std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(cmd, "r"), pclose);
  if (!pipe) {
    throw std::runtime_error("popen() failed!");
  }
  while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
    profiler_output += buffer.data();
  }
  return profiler_output;
}

std::string InitCounterTest() {
  std::string input_path = GetRunningPath("profiler_multithreaded_test");
  std::string rocprofv2_path = GetRunningPath(
      "build/tests/featuretests/profiler/profiler_multithreaded_test");
  std::stringstream command;
  command << rocprofv2_path + "./rocprofv2 -i "
          << input_path + "basic_metrics.txt "
          << input_path + "multithreaded_testapp";
  std::string result = ReadProfilerBuffer(command.str().c_str());
  return result;
}

int main(int argc, char** argv) {
  int test_status = -1;

  // initialize kernel dispatch test
  std::string profiler_output = InitCounterTest();
  // kernel dispatch count test
  test_status = DispatchCountTest(profiler_output);

  return test_status;
}