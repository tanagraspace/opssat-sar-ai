#!/usr/bin/env python3

import os
import csv
import shutil

# the file to read that will be used as a reference move data files intto dedicated folder 
# based on whether or not they are label as beacon
data_dir = 'data'
csv_filename = '1645433807.48_raw_ulf406025000_dlf1544185000_gain65_inputsr1500000_outputsr37500.csv'

# make expected directories
os.system('mkdir -p data/beacon')
os.system('mkdir -p data/nobeacon')
os.system('mkdir -p data/unknown')

# read csv file: loop row by row
with open(data_dir + '/' + csv_filename) as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for row in reader:

        # the csv contains filename and beacon label
        data_filename = row[0]
        beacon_label = int(row[1])

        # if beacon
        if beacon_label == 1:
            #print(data_dir + '/' + data_filename + " --> " + data_dir + '/beacon/' + data_filename)
            shutil.move(data_dir + '/' + data_filename, data_dir + '/beacon/' + data_filename)

        elif beacon_label == 0: # if no beacon
            #print(data_dir + '/' + data_filename + " --> " + data_dir + '/nobeacon/' + data_filename)
            shutil.move(data_dir + '/' + data_filename, data_dir + '/nobeacon/' + data_filename)

        else: # should not happen
            print("ERROR")

# Move the rest to the unknown subfolder
os.system("mv data/*.cf32 data/unknown/")