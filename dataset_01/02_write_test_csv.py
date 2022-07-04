#!/usr/bin/env python3

import os
import csv

# read the csv fle with all expected labels and create a new csv file for the test data
csv_filename = '1645433807.48_raw_ulf406025000_dlf1544185000_gain65_inputsr1500000_outputsr37500.csv'

# read csv file: loop row by row
with open('data/' + csv_filename) as csvfile:

    # write new csv file for test data set 
    with open('data/test_dataset_expected.csv', 'w', newline='') as csvfile_testset:

        # the writer
        writer = csv.writer(csvfile_testset, delimiter=',')

        # the reader
        reader = csv.reader(csvfile, delimiter=',')

        # the loop
        for row in reader:

            # the csv contains filename and beacon label
            data_filename = row[0]
            beacon_label = int(row[1])

            if os.path.exists('data/beacon/test/' + data_filename) or os.path.exists('data/nobeacon/test/' + data_filename):
                writer.writerow([data_filename, str(beacon_label)])