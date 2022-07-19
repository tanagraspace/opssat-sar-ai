import os
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


def plot(x, y, filename, index=0, show=False):
  # draw plot
  fig, ax = plt.subplots()
  ax.scatter(x, y, s=[1]*len(y))

  ax.set(xlabel='I', ylabel='Q',
        title='I/Q Phasor')

  ax.grid()

  # save figure to file
  fig.savefig(PLOT_DIR_BEACON + '/' + filename + '_' + str(index) + '.png')

  if show:
    # show figure
    plt.show()



# plot entire I/Q data
# read I/Q files that have a beacon
for filename in os.listdir(TRAINING_DIR_BEACON):
  
  # get phasor cartesian coordinates
  # x = I and y = Q
  i_array, q_array = read_phasor_cartesian_coordinates(TRAINING_DIR_BEACON + "/" + filename)

  plot(i_array, q_array, filename)


  # only do this for the first file we encounter.
  break



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