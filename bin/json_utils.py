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
import commands

class JSON:

  def __init__(self, file_name):
    self.file_name = file_name
    self.section_index = 0
    self.fd = open(self.file_name, mode='w')

  def __del__(self):
    self.fd.close()

  # dump JSON trace
  def open_json(self):
    file_name = self.file_name
    if not re.search(r'\.json$', file_name):
      raise Exception('wrong output file type: "' + file_name + '"' )
    with open(file_name, mode='w') as fd:
      fd.write('{ "traceEvents":[{}\n');

  def close_json(self):
    file_name = self.file_name
    if not re.search(r'\.json$', file_name):
      raise Exception('wrong output file type: "' + file_name + '"' )
    with open(file_name, mode='a') as fd:
      fd.write(']}\n');

  def label_json(self, pid, label):
    file_name = self.file_name
    if not re.search(r'\.json$', file_name):
      raise Exception('wrong output file type: "' + file_name + '"' )
    with open(file_name, mode='a') as fd:
      fd.write(',{"args":{"name":"%s %s"},"ph":"M","pid":%s,"name":"process_name"}\n' %(self.section_index, label, pid));
    self.section_index += 1

  def flow_json(self, base_id, from_pid, from_tid, from_us_list, to_pid, to_us_dict, corr_id_list, start_us):
    file_name = self.file_name
    if not re.search(r'\.json$', file_name):
      raise Exception('wrong output file type: "' + file_name + '"' )
    with open(file_name, mode='a') as fd:
      dep_id = base_id
      for ind in range(len(from_tid)):
        if (len(corr_id_list) != 0): corr_id = corr_id_list[ind]
        else: corr_id = ind
        if corr_id in to_us_dict:
          from_ts = from_us_list[ind] - start_us
          to_ts = to_us_dict[corr_id] - start_us
          if from_ts > to_ts: from_ts = to_ts
          fd.write(',{"ts":%d,"ph":"s","cat":"DataFlow","id":%d,"pid":%s,"tid":%s,"name":"dep"}\n' % (from_ts, dep_id, str(from_pid), from_tid[ind]))
          fd.write(',{"ts":%d,"ph":"t","cat":"DataFlow","id":%d,"pid":%s,"tid":0,"name":"dep"}\n' % (to_ts, dep_id, str(to_pid)))
          dep_id += 1

  def dump_json(self, table_name, data_name, db):
    file_name = self.file_name
    if not re.search(r'\.json$', file_name):
      raise Exception('wrong output file type: "' + file_name + '"' )

    sub_ptrn = re.compile(r'(^"|"$)')
    name_ptrn = re.compile(r'(name|Name)')

    table_fields = db._get_fields(table_name)
    table_raws = db._get_raws(table_name)
    data_fields = db._get_fields(data_name)
    data_raws = db._get_raws(data_name)

    with open(file_name, mode='a') as fd:
      table_raws_len = len(table_raws)
      for raw_index in range(table_raws_len):
        if (raw_index == table_raws_len - 1) or (raw_index % 1000 == 0):
          sys.stdout.write( \
            "\rdump json " + str(raw_index) + ":" + str(len(table_raws))  + " "*100 \
          )

        vals_list = []
        values = list(table_raws[raw_index])
        for value_index in range(len(values)):
          label = table_fields[value_index]
          value = values[value_index]
          if name_ptrn.search(label): value = sub_ptrn.sub(r'', value)
          if label != '"Index"': vals_list.append('%s:"%s"' % (label, value))

        args_list = []
        data = list(data_raws[raw_index])
        for value_index in range(len(data)):
          label = data_fields[value_index]
          value = data[value_index]
          if name_ptrn.search(label): value = sub_ptrn.sub(r'', value)
          if label != '"Index"': args_list.append('%s:"%s"' % (label, value))

        fd.write(',{"ph":"%s",%s,\n  "args":{\n    %s\n  }\n}\n' % ('X', ','.join(vals_list), ',\n    '.join(args_list)))

    sys.stdout.write('\n')

