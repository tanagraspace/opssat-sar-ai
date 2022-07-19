import os
import struct
import tensorflow as tf
import numpy as np
from PIL import Image

# Training data directories
TRAINING_DIR_BEACON = '../../dataset_01/data/beacon/training'
TRAINING_DIR_NOBEACON = '../../dataset_01/data/nobeacon/training'
TEST_DIR_BEACON = '../../dataset_01/data/beacon/test'
TEST_DIR_NOBEACON = '../../dataset_01/data/nobeacon/test'

MAX_VAL = 0xFFFFFFFF
NORMALIZE_MAX = 250

TEST_MODE = True

if TEST_MODE is True:
  IMG_DIR_BEACON = 'output/images/test/beacon'
  IMG_DIR_NOBEACON = 'output/images/test/nobeacon'
  
  WRITE_IMAGES = True
  SINGLE_TEST_RUN = True
  TRAIN_MODEL = False

else:
  IMG_DIR_BEACON = 'output/images/beacon/be'
  IMG_DIR_NOBEACON = 'output/images/nobeacon/be'

  WRITE_IMAGES = False
  SINGLE_TEST_RUN = False
  TRAIN_MODEL = True

# this array will collect all I/Q matrices
iq_data = []


# a super simple model definition
def create_model():
  model = tf.keras.models.Sequential([
    # the first layer flattens the input into a NORMALIZE_MAX x NORMALIZE_MAX sized vector
    tf.keras.layers.Flatten(input_shape=(NORMALIZE_MAX, NORMALIZE_MAX)), 

    # add a hidden layer with 50 neurons
    #tf.keras.layers.Dense(units=350, activation='relu'),
    tf.keras.layers.Dense(units=50, activation='sigmoid'), 

    # the final layer has 2 neurons, one for each label in our dataset (i.e. beacon and no beacon)
    #tf.keras.layers.Dense(units=2, activation=tf.nn.softmax)
    tf.keras.layers.Dense(units=2)
  ])

  # print model summary
  print(model.summary())

  return model
  

# build I/Q data matrix from an I/Q file
def build_iq_matrix(filepath):
  iq_pair_count = 0
  index = 1

  iq_mat = np.zeros((NORMALIZE_MAX, NORMALIZE_MAX), dtype=np.dtype('?'))
  i_index = None
  q_index = None

  with open(filepath, mode='rb') as bin_file: 
    for chunk in iter(lambda: bin_file.read(4), ''):

      if chunk == b'':
        #print("Done")
        break

      # get chunk value
      # little-endian
      #int_val = struct.unpack('<I', chunk)[0]

      # big-endian
      int_val = struct.unpack('>I', chunk)[0]

      # normalize and round
      int_val = round((int_val / MAX_VAL) * NORMALIZE_MAX)
      
      if index % 2 != 0:
        # get the I value
        #print("I: " + str(hex(int_val)).upper())
        i_index = int_val
      
      else:
        # get the Q value
        #print("Q: " + str(hex(int_val)).upper())
        q_index = int_val
        iq_mat[i_index-1, q_index-1] = 1 

        i_index = None
        q_index = None

        iq_pair_count = iq_pair_count + 1

      # increment index
      index = index + 1

  #print(iq_pair_count)
  return iq_mat

def write_image(iq_matrix, img_filepath):

  im = Image.new("RGB", (NORMALIZE_MAX, NORMALIZE_MAX), "#FFFFFF")
  pixels = im.load()

  for i in range(0, NORMALIZE_MAX):
    for j in range(0, NORMALIZE_MAX):
      if iq_matrix[i, j] == 1:
        pixels[i, j] = (0, 0, 0)

  im.save(img_filepath)
  #im.show()

# build I/Q tensors
def build_tensors():

  # BEACON: iterate directory containing i/q data with beacon
  count_beacon = 0
  for filename in os.listdir(TRAINING_DIR_BEACON):
      iq_mat = build_iq_matrix(TRAINING_DIR_BEACON + "/" + filename)
      iq_data.append(iq_mat)
      count_beacon = count_beacon + 1

      if WRITE_IMAGES is True:
        write_image(iq_mat, IMG_DIR_BEACON + '/' + filename + '.png')

      if SINGLE_TEST_RUN is True:
        break


  # NO BEACON: iterate directory containing i/q data with no beacon
  count_nobeacon = 0
  for filename in os.listdir(TRAINING_DIR_NOBEACON):
      iq_mat = build_iq_matrix(TRAINING_DIR_NOBEACON + "/" + filename)
      iq_data.append(iq_mat)
      count_nobeacon = count_nobeacon + 1

      if WRITE_IMAGES is True:
        write_image(iq_mat, IMG_DIR_NOBEACON + '/' + filename + '.png')

      if SINGLE_TEST_RUN is True:
        break

  # build the input and output tensor
  input_tensor3d = tf.constant(iq_data)
  output_tensor0d = tf.constant([1] * count_beacon + [0] * count_nobeacon)

  # shuffle the tensor indices
  indices = tf.range(start=0, limit=input_tensor3d.shape[0], dtype=tf.int32)
  shuffled_indices = tf.random.shuffle(indices)

  # shuffle the tensors using the suffled indices
  shuffled_input_tensor3d = tf.gather(input_tensor3d, shuffled_indices)
  shuffled_output_tensor0d = tf.gather(output_tensor0d, shuffled_indices)

  # print info on the input tensor
  print("\nInput Tensor for Training")
  print("  Type of every element:", shuffled_input_tensor3d.dtype)
  print("  Number of axes:", shuffled_input_tensor3d.ndim)
  print("  Shape of tensor:", shuffled_input_tensor3d.shape)
  print("  Elements along axis 0 of tensor:", shuffled_input_tensor3d.shape[0])
  print("  Elements along the last axis of tensor:", shuffled_input_tensor3d.shape[-1])
  print("  Total number of elements: ", tf.size(shuffled_input_tensor3d).numpy())

  #print(shuffled_input_tensor3d)
  #print(shuffled_output_tensor0d)

  return (shuffled_input_tensor3d, shuffled_output_tensor0d)


###########################
#     Train the model     #
###########################

# create input and output tensors
input_tensor, output_tensor = build_tensors()

if TRAIN_MODEL is True:
  # create and train a model
  model = create_model()
  model.compile(optimizer="adam", loss="mse", metrics=["accuracy"])
  model.fit(input_tensor, output_tensor, epochs=5)