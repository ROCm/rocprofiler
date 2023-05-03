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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <hip/hip_runtime.h>
#include "utils/test_helper.h"

#include <fstream>
#include <iostream>
#include <string>

#define SUCCESS 0
#define FAILURE 1

__global__ void helloworld(char* in, char* out) {
  int num = hipThreadIdx_x + hipBlockDim_x * hipBlockIdx_x;
  out[num] = in[num] + 1;
}

int main(int argc, char* argv[]) {
  hipDeviceProp_t devProp;
  HIP_RC(hipGetDeviceProperties(&devProp, 0));
  std::cout << " System minor " << devProp.minor << std::endl;
  std::cout << " System major " << devProp.major << std::endl;
  std::cout << " agent prop name " << devProp.name << std::endl;

  /* Initial input,output for the host and create memory objects for the
   * kernel*/
  const char* input = "GdkknVnqkc";
  size_t strlength = strlen(input);
  std::cout << "input string:" << std::endl;
  std::cout << input << std::endl;
  char* output = reinterpret_cast<char*>(malloc(strlength + 1));

  char* inputBuffer;
  char* outputBuffer;
  HIP_RC(hipMalloc(reinterpret_cast<void**>(&inputBuffer), (strlength + 1) * sizeof(char)));
  HIP_RC(hipMalloc(reinterpret_cast<void**>(&outputBuffer), (strlength + 1) * sizeof(char)));

  HIP_RC(hipMemcpy(inputBuffer, input, (strlength + 1) * sizeof(char), hipMemcpyHostToDevice));

  HIP_KL(hipLaunchKernelGGL(helloworld, dim3(1), dim3(strlength), 0, 0, inputBuffer, outputBuffer));

  HIP_RC(hipMemcpy(output, outputBuffer, (strlength + 1) * sizeof(char), hipMemcpyDeviceToHost));

  HIP_RC(hipFree(inputBuffer));
  HIP_RC(hipFree(outputBuffer));

  output[strlength] = '\0';  // Add the terminal character to the end of output.
  std::cout << "\noutput string:" << std::endl;
  std::cout << output << std::endl;

  free(output);

  std::cout << "Passed!\n";

  return SUCCESS;
}
