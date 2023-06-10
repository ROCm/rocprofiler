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
#include <gtest/gtest.h>
#include <ostream>
#include <vector>
#include "tracer_gtest.h"
#include "../utils/test_utils.h"

/**
 * Sets application enviornment by seting HSA_TOOLS_LIB.
 */
void ApplicationParser::SetApplicationEnv(const char* app_name, const char* trace_option) {
  std::string app_path = GetRunningPath("tests-v2/featuretests/tracer/runTracerFeatureTests");

  std::stringstream hsa_tools_lib_path;
  hsa_tools_lib_path << app_path << "librocprofiler_tool.so";
  setenv("LD_PRELOAD", hsa_tools_lib_path.str().c_str(), true);

  std::string trace_type{trace_option};

  if (trace_type.find("hip") != std::string::npos) {
    // set --hip-api option
    setenv("ROCPROFILER_HIP_API_TRACE", "1", true);
  }

  if (trace_type.find("hsa") != std::string::npos) {
    // set --hsa-api and --hsa-activity
    setenv("ROCPROFILER_HSA_API_TRACE", "1", true);
    setenv("ROCPROFILER_HSA_ACTIVITY_TRACE", "1", true);
  }


  std::stringstream os;
  os << app_path << "tests-v2/featuretests/tracer/apps/" << app_name;
  ProcessApplication(os);
}

/**
 * Parses kernel-info after running tracer against curent application
 * and saves them in a vector.
 */
void ApplicationParser::GetKernelInfoForRunningApplication(
    std::vector<KernelInfo>* kernel_info_output) {
  KernelInfo kinfo;
  for (std::string line : output_lines) {
    // if (std::regex_match(line, std::regex("(Record)(.*)"))) {
    // Record id
    size_t found = line.find("Record");
    if (found != std::string::npos) {
      int spos = found;
      int epos = line.find(")", spos);
      int length = std::string("Record").length();
      std::string sub = line.substr(spos + length + 1, epos - spos - length - 1);

      kinfo.record_id = sub;
    }

    // Kernel-Name
    found = line.find("Function");
    if (found != std::string::npos) {
      int spos = found;
      int epos = line.find(")", spos);
      int length = std::string("Function").length();
      std::string sub = line.substr(spos + length + 1, epos - spos - length - 1);
      kinfo.function = sub;
    }

    // corelation-ids
    found = line.find("Correlation_ID");
    if (found != std::string::npos) {
      int spos = found;
      int epos = line.find(")", spos);
      int length = std::string("Correlation_ID").length();
      std::string sub = line.substr(spos + length + 1, epos - spos - length - 1);
      kinfo.corelation_id = sub;
    }

    if (kinfo.record_id != "") {
      kernel_info_output->push_back(kinfo);
    }
  }
}

/**
 * Parses kernel-names from a pre-saved golden out files
 * and saves them in a vector.
 */
void ApplicationParser::GetKernelInfoForGoldenOutput(const char* app_name, std::string file_name,
                                                     std::vector<KernelInfo>* kernel_info_output) {
  std::string entry;
  std::string path = GetRunningPath("runTracerFeatureTests");
  entry = path.append("apps/goldentraces/") + file_name;

  // parse kernel info fields for golden output
  ParseKernelInfoFields(entry, kernel_info_output);
}

/**
 * Runs a given appllication and saves tracer output.
 * These output lines can be letter passed for kernel informations
 * i.e: kernel_names
 */
void ApplicationParser::ProcessApplication(std::stringstream& ss) {
  FILE* handle = popen(ss.str().c_str(), "r");
  ASSERT_NE(handle, nullptr);

  char* ln{NULL};
  std::string temp{""};
  size_t len{0};

  while (getline(&ln, &len, handle) != -1) {
    temp = temp + std::string(ln);
  }

  free(ln);
  size_t pos{0};
  std::string delimiter{"\n"};
  while ((pos = temp.find(delimiter)) != std::string::npos) {
    output_lines.push_back(temp.substr(0, pos));
    temp.erase(0, pos + delimiter.length());
  }

  pclose(handle);
}

/**
 * Parses kernel-info for golden output file
 * and saves them in a vector.
 */
void ApplicationParser::ParseKernelInfoFields(const std::string& s,
                                              std::vector<KernelInfo>* kernel_info_output) {
  std::string line;
  KernelInfo kinfo;

  std::ifstream golden_file(s);
  while (!golden_file.eof()) {
    getline(golden_file, line);
    // if (std::regex_match(line, std::regex("(Record)(.*)"))) {
    // Record id
    size_t found = line.find("Record");
    if (found != std::string::npos) {
      int spos = found;
      int epos = line.find(")", spos);
      int length = std::string("Record").length();
      std::string sub = line.substr(spos + length + 1, epos - spos - length - 1);

      kinfo.record_id = sub;
      // kernel_info_output->push_back(kinfo);
    }

    // Kernel-Name
    found = line.find("Function");
    if (found != std::string::npos) {
      int spos = found;
      int epos = line.find(")", spos);
      int length = std::string("Function").length();
      std::string sub = line.substr(spos + length + 1, epos - spos - length - 1);

      kinfo.function = sub;
      // kernel_info_output->push_back(kinfo);
    }

    // corealtion-ids
    found = line.find("Correlation_ID");
    if (found != std::string::npos) {
      int spos = found;
      int epos = line.find(")", spos);
      int length = std::string("Correlation_ID").length();
      std::string sub = line.substr(spos + length + 1, epos - spos - length - 1);

      kinfo.corelation_id = sub;
      // kernel_info_output->push_back(kinfo);
    }
    //}
    if (kinfo.record_id != "") {
      kernel_info_output->push_back(kinfo);
    }
  }
  golden_file.close();
}
/*
 * ###################################################
 * ############ HelloWorld HIP Tests ################
 * ###################################################
 */

constexpr auto kGoldenOutputHelloworld = "hip_helloworld_golden_traces.txt";

class HelloWorldTest : public Tracertest {
 protected:
  std::vector<KernelInfo> golden_kernel_info;
  void SetUp() {
    Tracertest::SetUp("tracer_hip_helloworld", "--hip-api ");
    GetKernelInfoForGoldenOutput("tracer_hip_helloworld", kGoldenOutputHelloworld,
                                 &golden_kernel_info);
  }
  void TearDown() { output_lines.clear(); }
};

// Test:1 Compares total num of kernel-names in golden output against current
// tracer output
TEST_F(HelloWorldTest, WhenRunningTracerWithAppThenKernelInfoMatchWithGoldenOutput) {
  // kernel info in current profler run
  std::vector<KernelInfo> current_kernel_info;

  GetKernelInfoForRunningApplication(&current_kernel_info);
  ASSERT_TRUE(current_kernel_info.size());

  EXPECT_EQ(golden_kernel_info.size(), current_kernel_info.size());
}

// Test:2 Compares order of kernel-names in golden output against current
// tracer output
TEST_F(HelloWorldTest, WhenRunningTracerWithAppThenFunctionNamessMatchWithGoldenOutput) {
  // kernel info in current tracer run
  std::vector<KernelInfo> current_kernel_info;
  GetKernelInfoForRunningApplication(&current_kernel_info);

  ASSERT_TRUE(current_kernel_info.size());

  EXPECT_EQ(golden_kernel_info[0].function, current_kernel_info[0].function);
  EXPECT_EQ(golden_kernel_info[1].function, current_kernel_info[1].function);
}

// Test:3 Compares order of kernel-names in golden output against current
// tracer output
TEST_F(HelloWorldTest, WhenRunningTracerWithAppThenKernelDurationShouldBePositive) {
  // kernel info in current tracer run
  std::vector<KernelInfo> current_kernel_info;

  GetKernelInfoForRunningApplication(&current_kernel_info);
  ASSERT_TRUE(current_kernel_info.size());

  EXPECT_GT(current_kernel_info.size(), 0);
}


/*
 * ###################################################
 * ############ Async COopy HSA Tests ################
 * ###################################################
 */

class AsyncCopyTest : public Tracertest {
 protected:
  void SetUp() { Tracertest::SetUp("copy_on_engine", "--hsa-api --hsa-activity"); }
  void TearDown() { output_lines.clear(); }
};

// Test:1 Compares total num of kernel-names in golden output against current
// tracer output
TEST_F(AsyncCopyTest, WhenRunningTracerWithAppThenAsyncCopyOutputIsgenerated) {
  // kernel info in current profler run
  std::vector<KernelInfo> current_kernel_info;

  GetKernelInfoForRunningApplication(&current_kernel_info);
  ASSERT_TRUE(current_kernel_info.size());
}

// Test:2 Matches coelation Ids
TEST_F(AsyncCopyTest, WhenRunningTracerWithAppThenAsyncCorelationCountIsCorrect) {
  // kernel info in current profler run
  std::vector<KernelInfo> current_kernel_info;

  GetKernelInfoForRunningApplication(&current_kernel_info);
  ASSERT_TRUE(current_kernel_info.size());

  std::vector<std::pair<std::string, std::string>> corelation_pair{};
  for (const auto& itr : current_kernel_info) {
    if (itr.function.find("async_copy_on_engine") != std::string::npos) {
      corelation_pair.push_back({itr.record_id, itr.corelation_id});
      break;  // we just want first occurance to test
    }
  }
  ASSERT_TRUE(corelation_pair.size());

  uint32_t corelation_count = 0;
  // check if same corelation id appears again but as a different ops record
  for (size_t i = 0; i < corelation_pair.size(); i++) {
    for (const auto& itr : current_kernel_info) {
      if ((itr.corelation_id == corelation_pair[i].second) &&
          (itr.record_id != corelation_pair[i].first)) {
        corelation_count++;
      }
    }
  }

  EXPECT_EQ(corelation_count, corelation_pair.size());
}