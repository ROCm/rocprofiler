#!/usr/bin/env python3
import sys
if sys.version_info[0] < 3:
    raise Exception("Must be using Python 3")

import os
import argparse
from pathlib import Path
from ctypes import *
import ctypes
from copy import deepcopy
from trace_view import view_trace
import sys
import glob
import numpy as np
from stitch import stitch
import gc

class PerfEvent(ctypes.Structure):
    _fields_ = [
        ('time', c_uint64),
        ('event0', c_uint16),
        ('event1', c_uint16),
        ('event2', c_uint16),
        ('event3', c_uint16),
        ('cu', c_uint32),
        ('bank', c_uint32),
    ]
    def toTuple(self):
        return (int(self.time), int(self.event0), int(self.event1),
            int(self.event2), int(self.event3), int(self.cu), int(self.bank))


class CodeWrapped(ctypes.Structure):
    """ Matches CodeWrapped on the python side """
    _fields_ = [('line', ctypes.c_char_p),
                            ('loc', ctypes.c_char_p),
                            ('value', ctypes.c_int),
                            ('to_line', ctypes.c_int),
                            ('index', ctypes.c_int),
                            ('line_num', ctypes.c_int)]


class KvPair(ctypes.Structure):
    """ Matches pair<int, int> = (key, value) on the python side """
    _fields_ = [('key', ctypes.c_int),
                            ('value', ctypes.c_int)]


class ReturnAssemblyInfo(ctypes.Structure):
    """ Matches ReturnAssemblyInfo on the python side """
    _fields_ = [('code', POINTER(CodeWrapped)),
                            ('jumps', POINTER(KvPair)),
                            ('code_len', ctypes.c_int),
                            ('jumps_len', ctypes.c_int)]


rocprofv2_att_lib = os.getenv('ROCPROFV2_ATT_LIB_PATH')
if rocprofv2_att_lib is None:
    print("ATT Lib path not set. Use export ROCPROFV2_ATT_LIB_PATH=/path/to/librocprofv2_att.so")
    quit()
path_to_parser = os.path.abspath(rocprofv2_att_lib)
SO = CDLL(path_to_parser)

SO.wrapped_parse_binary.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
SO.wrapped_parse_binary.restype = ReturnAssemblyInfo

def parse_binary(filename, kernel=None):
    if kernel is None or kernel == '':
        kernel = ctypes.c_char_p(0)
        print('Parsing all kernels')
    else:
        with open(glob.glob(kernel)[0], 'r') as file:
            kernel = file.readlines()
        print('Parsing kernel:', kernel[0].split(': ')[0])
        kernel = kernel[0].split(': ')[1].split('.kd')[0]
        kernel = str(kernel).encode('utf-8')
    filename = os.path.abspath(str(filename))
    info = SO.wrapped_parse_binary(str(filename).encode('utf-8'), kernel)

    code = []
    for k in range(info.code_len):
        code_entry = info.code[k]

        line = deepcopy(code_entry.line.decode("utf-8"))
        loc = deepcopy(code_entry.loc.decode("utf-8"))

        to_line = int(code_entry.to_line) if (code_entry.to_line >= 0) else None
        loc = loc if len(loc) > 0 else None

        code.append([line, int(code_entry.value), to_line, loc,
                    int(code_entry.index), int(code_entry.line_num), 0, 0]) # hitcount + cycles

    jumps = {}
    for k in range(info.jumps_len):
        jumps[info.jumps[k].key] = info.jumps[k].value

    return code, jumps


def getWaves_binary(name, shader_engine_data_dict):
    filename = os.path.abspath(str(name))
    with open(filename, 'rb') as f:
        data = f.read()

    if len(data)%8 != 0:
        data = data[:len(data)-(len(data)%8)]

    header = np.frombuffer(data[:32], dtype=c_uint64).astype(np.int64)
    flags = 'navi' if (header[0] & 0x1) else 'vega'
    num_occupancy = header[1]
    num_events = header[2]
    num_waves = header[3]
    wave_sizes = np.frombuffer(data[32:32+8*num_waves], dtype=ctypes.c_uint64).astype(np.int64)

    data = data[32+8*num_waves:]
    wavedata = np.frombuffer(data[:np.sum(wave_sizes)*24], dtype=ctypes.c_uint64)
    data = data[np.sum(wave_sizes)*24:]

    waves_python = []
    for wid in range(num_waves):
        #pwave = wavedata[:wave_sizes[wid]*3].reshape((-1,3)).astype(np.int64)
        pwave = [tuple([int(wavedata[3*k+m]) for m in range(3)]) for k in range(wave_sizes[wid])]
        wavedata = wavedata[wave_sizes[wid]*3:]
        if len(pwave) < 2:
            continue
        waves_python.append( pwave )

    events, occupancy = [], [0]
    try:
        if num_events > 0:
            events = np.frombuffer(data[:24*num_events], dtype=PerfEvent)
    except:
        pass
    try:
        if num_occupancy > 0:
            occupancy = [int(m) for m in np.frombuffer(data[24*num_events:], dtype=c_uint64)]
            ev0 = occupancy[0] & ~0xFFFFF
            occupancy = [k-ev0 for k in occupancy]
    except:
        pass

    shader_engine_data_dict[name] = (waves_python, events, occupancy, flags)


def getWaves_stitch(SIMD, code, jumps, flags, latency_map, hitcount_map):
    for k in range(len(SIMD)):
        SIMD[k] = stitch(SIMD[k], code, jumps, flags)

        for inst in SIMD[k][0]:
            hitcount_map[inst[-1]] += inst[0]
            latency_map[inst[-1]] += inst[2]

def mem_max(array):
    mem_dict = {}
    for SE in array:
        for wave in SE:
            for inst in wave:
                try:
                    mem_dict[inst[0]][0] = max(mem_dict[inst[0]][0], inst[1])
                except:
                    mem_dict[inst[0]] = inst[1:]
                assert(mem_dict[inst[0]][1] == inst[2])

    return mem_dict

def lgk(count):
    return 'lgkmcnt({0})'.format(count)
def vmc(count):
    return 'vmcnt({0})'.format(count)
def both_cnt(count):
    return lgk(count)+' '+vmc(count)

def insert_waitcnt(flight_count, assembly_code):
    flight_count = mem_max(flight_count)
    for key in sorted(flight_count):
        line_n = key
        issue_amount, waitcnt_amount, = flight_count[key]
        if 'vmcnt' in assembly_code[line_n] and 'lgkmcnt' in assembly_code[line_n]:
            counter_type = both_cnt
        elif 'vmcnt' in assembly_code[line_n]:
            counter_type = vmc
        elif 'lgkmcnt' in assembly_code[line_n]:
            counter_type = lgk
        else:
            print('Error: Line mismatch')
            exit(-1)

        for count in range(waitcnt_amount+1, issue_amount):
            print('Inserted line: '+str(line_n))
            as_index = line_n - count/(issue_amount+1)
            assembly_code[as_index] = \
                '\ts_waitcnt {0}\t\t; Timing analysis.'.format(counter_type(count))
            as_index += 0.5/(issue_amount+1)
            assembly_code[as_index] = '\ts_nop 0\t\t\t\t\t\t; Counters: '+str(issue_amount)

    return assembly_code

if __name__ == "__main__":
    pathenv = os.getenv('OUTPUT_PATH')
    if pathenv is None:
        pathenv = "."
    parser = argparse.ArgumentParser()
    parser.add_argument("assembly_code", help="Path to the assembly code. Must be the first parameter.")
    parser.add_argument("--depth", help="Maximum number of parsed waves per slot", default=100, type=int)
    parser.add_argument("--trace_file", help="Filter for trace files", default=None, type=str)
    parser.add_argument("--att_kernel", help="Kernel file",
                        type=str, default=pathenv+'/*_kernel.txt')
    parser.add_argument("--ports", help="Server and websocket ports, default: 8000,18000")
    parser.add_argument("--genasm",
                        help="Generate post-processed asm file at this path", type=str, default="")
    parser.add_argument("--mode", help='''ATT analysis modes:\n
                        off: Only run ATT collection, disable analysis.\n
                        file: dump json files to disk.\n
                        network: Open att server over the network.''', type=str, default="off")
    args = parser.parse_args()

    CSV_MODE = False
    if args.mode.lower() == 'csv':
        CSV_MODE = True
    elif args.mode.lower() == 'file':
        args.dumpfiles = True
    elif args.mode.lower() == 'network':
        args.dumpfiles = False
    else:
        print('Skipping analysis.')
        quit()

    with open(os.getenv("COUNTERS_PATH"), 'r') as f:
        lines = [l.split('//')[0] for l in f.readlines()]

        EVENT_NAMES = []
        clean = lambda x: x.split('=')[1].split(' ')[0].split('\n')[0]
        for line in lines:
            if 'PERFCOUNTER_ID=' in line:
                EVENT_NAMES += ['id: '+clean(line)]
            elif 'att: TARGET_CU' in line:
                args.target_cu = int(clean(line))
        for line in lines:
            if 'PERFCOUNTER=' in line:
                EVENT_NAMES += [clean(line).split('SQ_')[1].lower()]
    if args.target_cu is None:
        args.target_cu = 1

    # Assembly parsing
    path = Path(args.assembly_code)
    if not path.is_file():
        print("Invalid assembly_code('{0}')!".format(args.assembly_code))
        sys.exit(1)

    att_kernel = glob.glob(args.att_kernel)

    if len(att_kernel) == 0:
        print('Could not find att output kernel:', args.att_kernel)
        exit(1)
    elif len(att_kernel) > 1:
        print('Found multiple kernel matching given filters:')
        for n, k in enumerate(att_kernel):
            print('\t', n, '->', k)

        bValid = False
        while bValid == False:
            try:
                args.att_kernel = att_kernel[int(input("Please select number: "))]
                bValid = True
            except KeyboardInterrupt:
                exit(0)
            except:
                print('Invalid option.')
    else:
        args.att_kernel = att_kernel[0]

    # Trace Parsing
    if args.trace_file is None:
        filenames = glob.glob(args.att_kernel.split('_kernel.txt')[0]+'_*.att')
    else:
        filenames = glob.glob(args.trace_file)
    assert(len(filenames) > 0)

    print('Att kernel:', args.att_kernel)
    code, jumps = parse_binary(args.assembly_code, args.att_kernel)

    DBFILES = []
    TIMELINES = [np.zeros(int(1E4),dtype=np.int16) for k in range(5)]
    EVENTS = []
    OCCUPANCY = []
    GFXV = []
    analysed_filenames = []

    shader_engine_data_dict = {}
    for name in filenames:
        getWaves_binary(name, shader_engine_data_dict)

    gc.collect()
    latency_map = np.zeros((len(code)), dtype=np.int64)
    hitcount_map = np.zeros((len(code)), dtype=np.int32)
    for name in filenames:
        SIMD, perfevents, occupancy, gfxv = shader_engine_data_dict[name]
        getWaves_stitch(SIMD, code, jumps, gfxv, latency_map, hitcount_map)
        if len(SIMD) == 0:
            print("Error parsing ", name)
            continue
        analysed_filenames.append(name)
        EVENTS.append(perfevents)
        DBFILES.append( SIMD )
        OCCUPANCY.append( occupancy )
        GFXV.append(gfxv)

    gc.collect()

    gathered_filenames = analysed_filenames

    for k in range(len(code)):
        code[k][-2] = int(hitcount_map[k])
        code[k][-1] = int(latency_map[k])

    gc.collect()

    if CSV_MODE:
        from att_to_csv import dump_csv
        dump_csv(code)
        quit()

    drawinfo = {'TIMELINES':TIMELINES, 'EVENTS':EVENTS, 'EVENT_NAMES':EVENT_NAMES, 'OCCUPANCY': OCCUPANCY, 'ShaderNames': gathered_filenames}
    if args.genasm and len(args.genasm) > 0:
        flight_count = view_trace(args, code, DBFILES, analysed_filenames, True, OCCUPANCY, args.dumpfiles, gfxv, drawinfo)
        with open(args.assembly_code, 'r') as file:
            lines = file.readlines()
        assembly_code = {l+1.0: lines[l][:-1] for l in range(len(lines))}
        assembly_code = insert_waitcnt(flight_count, assembly_code)

        with open(args.genasm, 'w') as file:
            keys = sorted(assembly_code.keys())
            for k in keys:
                file.write(assembly_code[k]+'\n')
    else:
        view_trace(args, code, DBFILES, analysed_filenames, False, OCCUPANCY, args.dumpfiles, gfxv, drawinfo)
