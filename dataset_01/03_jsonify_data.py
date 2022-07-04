#!/usr/bin/env python3

import os
import csv
import struct

# to read in Windows Terminal:
# data/beacon/training/1645433807.48_raw_ulf406025000_dlf1544185000_gain65_inputsr1500000_outputsr37500_1.cf32 | more

filename = 'data/beacon/training/1645433807.48_raw_ulf406025000_dlf1544185000_gain65_inputsr1500000_outputsr37500_1.cf32'

with open(filename, mode='rb') as bin_file: 
    for chunk in iter(lambda: bin_file.read(4), ''):
        int_val = struct.unpack('<I', chunk)[0]
        print(int_val)
        break