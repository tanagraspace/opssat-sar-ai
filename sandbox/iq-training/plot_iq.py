import os
import math
import struct
import numpy as np
import matplotlib.pyplot as plt

# Training data directories
TRAINING_DIR_BEACON = '../../dataset_01/data/beacon/training'
TRAINING_DIR_NOBEACON = '../../dataset_01/data/nobeacon/training'
TEST_DIR_BEACON = '../../dataset_01/data/beacon/test'
TEST_DIR_NOBEACON = '../../dataset_01/data/nobeacon/test'

PLOT_DIR_BEACON = 'output/plots/beacon'
PLOT_DIR_NOBEACON = 'output/plots/nobeacon'

NORMALIZE = False
MAX_VAL = 0xFFFFFFFF

BIG_ENDIAN = True
SHOW_PLOTS = False

def read_phasor_cartesian_coordinates(filepath):
  iq_pair_count = 0
  index = 1

  i_array = []
  q_array = []


  with open(filepath, mode='rb') as bin_file: 
      for chunk in iter(lambda: bin_file.read(4), ''):

          if chunk == b'':
              #print("Done")
              break

          # get chunk value
          val = None
          
          if BIG_ENDIAN:
            # read as big-endian
            val = struct.unpack('>I', chunk)[0]
          elif not BIG_ENDIAN:
            # read as little-endian
            val = struct.unpack('<I', chunk)[0]

          # normalize the data
          if NORMALIZE:
            val = val / MAX_VAL
          
          if index % 2 != 0:
              # get the I value
              #print("I: " + str(hex(val)).upper())
              i_array.append(val)
          
          else:
              # get the Q value
              #print("Q: " + str(hex(val)).upper())
              q_array.append(val)
              iq_pair_count = iq_pair_count + 1

          # increment index
          index = index + 1

  return (i_array, q_array)


def plot(x, y, xlabel, ylabel, title, filename, index=0, show=False):
  # draw plot
  fig, ax = plt.subplots()
  ax.scatter(x, y, s=[1]*len(y))
  ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
  ax.grid()

  # save figure to file
  fig.savefig(filename + '.' + str(index) + '.png')

  if show:
    # show figure
    plt.show()


# calculate amplitude
def amplitude(i, q):
  return math.sqrt(i^2 + q^2)

# calculate phase
def phase(i, q):
  return math.atan(q/i)


def do_all_the_doings(iq_data_dir_path, plots_dir_path, show_plots, stop_at=1):
  files_counter = 0

  # plot entire I/Q data in given directory
  for filename in os.listdir(iq_data_dir_path):

    # plot phasor diagram

    # get phasor cartesian coordinates
    # x = I and y = Q
    i_array, q_array = read_phasor_cartesian_coordinates(iq_data_dir_path + "/" + filename)
    
    # plot I/Q Phasor
    plot(i_array, q_array, 'I', 'Q', 'I/Q Phasor', plots_dir_path + "/" + filename + ".phasor", files_counter+1, show_plots)

    # plot amplitude and phase

    # calculate amplitude and phase
    amplitude_array = []
    phase_array = []

    for index in range(0, len(i_array)):
      amplitude_array.append(amplitude(i_array[index], q_array[index]))
      phase_array.append(phase(i_array[index], q_array[index]))

    # plot Amplitude / Phase'
    plot(phase_array, amplitude_array, 'φ', 'A', 'Amplitude / Phase', plots_dir_path + "/" + filename + ".amplitude_and_phase", files_counter+1, show_plots)

    # plot Amplitude / Time
    #plot(range(0, len(amplitude_array)), amplitude_array, 't', 'A', 'Amplitude / Time', plots_dir_path + "/" + filename + ".amplitude_and_time", files_counter+1, show_plots)

    # plot Phase / Time
    #plot(range(0, len(phase_array)), phase_array, 't', 'φ', 'Phase / Time', plots_dir_path + "/" + filename + ".phase_and_time", files_counter+1, show_plots)

    # only process the asked number of I/Q files 
    files_counter = files_counter + 1
    if files_counter == stop_at:
      break


# now do all the doings
do_all_the_doings(TRAINING_DIR_BEACON, PLOT_DIR_BEACON, SHOW_PLOTS, 2)
do_all_the_doings(TRAINING_DIR_NOBEACON, PLOT_DIR_NOBEACON, SHOW_PLOTS, 2)


'''
# plot segments of the I/Q data
step = 10000
index_max = 5

for filename in os.listdir(TRAINING_DIR_BEACON):
  
  # get phasor cartesian coordinates
  # x = I and y = Q
  x, y = read_phasor_cartesian_coordinates(TRAINING_DIR_BEACON + "/" + filename)

  plot(x[1:step], y[1:step], filename, 1)
  for i in range(1, index_max+1):
    plot(x[step * i:step * (i+1)], y[step * i:step * (i+1)], filename, i)
    

  # only do this for the first file we encounter.
  break
'''