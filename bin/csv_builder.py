#!/usr/bin/python

################################################################################
# Copyright (c) 2018 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
################################################################################

import os, sys, re

class CSV:

  def __init__(self, file_name):
    self.file_name = file_name
    self.fd = open(self.file_name, mode='w')

  def __del__(self):
    self.fd.close()

  # dump CSV results
  def dump_csv_fromtable(self, var_list, var_table):
    #global var_list
    keys = sorted(var_table.keys(), key=int)

    with open(self.file_name, mode='w') as self.fd:
      self.fd.write(','.join(var_list) + '\n');
      for ind in keys:
        entry = var_table[ind]
        dispatch_number = entry['Index']
        if ind != dispatch_number: fatal("Dispatch #" + ind + " index mismatch (" + dispatch_number + ")\n")
        val_list = [entry[var] for var in var_list]
        self.fd.write(','.join(val_list) + '\n');

    print("File '" + self.file_name + "' is generating")

  # dump CSV table
  def dump_csv_fromdb(self, table_name):
    file_name = self.file_name
    if not re.search(r'\.csv$', file_name):
      raise Exception('wrong output file type: "' + file_name + '"' )

    fields = self._get_fields(table_name)
    with open(file_name, mode='w') as fd:
      fd.write(','.join(fields) + '\n')
      for raw in self._get_raws(table_name):
        fd.write(reduce(lambda a, b: str(a) + ',' + str(b), raw) + '\n')
