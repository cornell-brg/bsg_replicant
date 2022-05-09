# get stall cycles from profiling data
import pandas as pd
import numpy as np

size = 64
impl = "APPLRTS"

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', -1)

configs = [(1, 1), (2, 2), (4, 4), (8, 8), (16, 8)]

stallInROI = ['stall_depend_dram_load',
              'stall_depend_dram_amo',
              'stall_depend_group_load',
              'stall_depend_group_amo',
              'stall_depend_local_load',
              'stall_amo_aq',
              'stall_amo_rl',
              'stall_remote_req',
              'stall_ifetch_wait']

class DataPoint:
  def __init__(self, filename, numActiveCores):
    self.dp = {}

    self.dp["active_cores"] = numActiveCores

    header = parseAggrFromFile(filename).split()
    self.dp["abs_total_cycle"] = int(header[6])

    for stall in parseStallFromFile(filename):
      entry = stall.split()
      self.dp[entry[0]] = int(entry[1])

    for instr in parseInstrFromFile(filename):
      entry = instr.split()
      self.dp[entry[0]] = int(entry[1])

def parseStallFromFile(filename):
  with open(filename, 'r') as f:
    raw = f.readlines()
    start = -1
    end   = -1
    for i in range(len(raw)):
      if (raw[i].startswith('stall_depend_dram_load')):
        start = i
      if (raw[i].startswith('not_stall')):
        end = i
        break
    return raw[start:end+1]

def parseInstrFromFile(filename):
  with open(filename, 'r') as f:
    raw = f.readlines()
    start = -1
    end   = -1
    for i in range(len(raw)):
      if (raw[i].startswith('instr_fadd')):
        start = i
      if (raw[i].startswith('instr_total')):
        end = i
        break
    return raw[start:end+1]

def parseAggrFromFile(filename):
  with open(filename, 'r') as f:
    raw = f.readlines()[3]
    return raw


if __name__ == "__main__":
  dpoints = {}
  for config in configs:
    filename = "matrix-n_{0}__tgx_{1}__tgy_{2}__appl-impl_{3}/stats/manycore_stats.log".format(size, config[0], config[1], impl)
    testCase = DataPoint(filename, config[0] * config[1])
    dpoints["{0}x{1}".format(config[0], config[1])] = testCase.dp

  df = pd.DataFrame(dpoints)

  print(df)
  # print("abs total cycles")
  # print(df[df.index.str.startswith("abs_total")])
  # print()

  # # stall cycles
  # print("stall cycles")
  # print(df[df.index.str.startswith("stall")])
  # print()

  # print("instr count")
  # print(df[df.index.str.startswith("instr_")])
  # print()
  print()
  print("---------------------------------")
  print("per core: ")
  print(df.div(df.loc['active_cores']).apply(np.ceil).astype('int64'))

  print()
  print("---------------------------------")
  print("per group ld cost: ")
  print(df.loc["stall_depend_group_load"].div(df.loc["instr_remote_ld_group"]))

  print()
  print("---------------------------------")
  print("per group amo cost: ")
  print(df.loc["stall_depend_group_amo"].div(df.loc["instr_amoswap"]))
