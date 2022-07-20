# resources
# - https://www.tensorflow.org/guide/tensor
# - https://towardsdatascience.com/create-image-classification-models-with-tensorflow-in-10-minutes-d0caef7ca011

import os
import struct
import tensorflow as tf
import numpy as np

# Training data directories
TRAINING_DIR_BEACON = '../../dataset_01/data/beacon/training'
TRAINING_DIR_NOBEACON = '../../dataset_01/data/nobeacon/training'
TEST_DIR_BEACON = '../../dataset_01/data/beacon/test'
TEST_DIR_NOBEACON = '../../dataset_01/data/nobeacon/test'

NORMALIZE = True
MAX_VAL = 0xFFFFFFFF
#MAX_VAL = 1000000000

INPUT_TENSOR_DATA_STRUCTURE_TYPE = 1

# this array will collect all I/Q matrices
iq_data = []
test_iq_data = []


# a super simple model definition
def create_model():


  model = None
  
  if INPUT_TENSOR_DATA_STRUCTURE_TYPE == 1:
    model = tf.keras.models.Sequential([
      # the first layer flattens the input into a 2*52,375=104,750 sized vector
      tf.keras.layers.Flatten(input_shape=(2, 52375)), 

      # add a hidden layer with 50 neurons
      #tf.keras.layers.Dense(units=350, activation='relu'),
      tf.keras.layers.Dense(units=50, activation='sigmoid'), 

      # the final layer has 2 neurons, one for each label in our dataset (i.e. beacon and no beacon)
      #tf.keras.layers.Dense(units=2, activation=tf.nn.softmax)
      tf.keras.layers.Dense(units=2)
    ])

  elif INPUT_TENSOR_DATA_STRUCTURE_TYPE == 2:
    model = tf.keras.models.Sequential([
      # the first layer flattens the input into a 52,375*2=104,750 sized vector
      tf.keras.layers.Flatten(input_shape=(52375, 2)), 

      # add a hidden layer with 50 neurons
      #tf.keras.layers.Dense(units=350, activation='relu'),
      tf.keras.layers.Dense(units=50, activation='sigmoid'), 

      # the final layer has 2 neurons, one for each label in our dataset (i.e. beacon and no beacon)
      #tf.keras.layers.Dense(units=2, activation=tf.nn.softmax)
      tf.keras.layers.Dense(units=2)
    ])

  elif INPUT_TENSOR_DATA_STRUCTURE_TYPE == 3:
    model = tf.keras.models.Sequential([
      # the first layer flattens the input into a 104,750 sized vector
      tf.keras.layers.Flatten(input_shape=(104750, 1)), 

      # add a hidden layer with 50 neurons
      #tf.keras.layers.Dense(units=350, activation='relu'),
      tf.keras.layers.Dense(units=50, activation='sigmoid'), 

      # the final layer has 2 neurons, one for each label in our dataset (i.e. beacon and no beacon)
      #tf.keras.layers.Dense(units=2, activation=tf.nn.softmax)
      tf.keras.layers.Dense(units=2)
    ])

  # print model summary
  print(model.summary())

  # return model
  return model


# build I/Q data matrix from an I/Q file
# [
#    [i1, i2, i3, i4… i52,375],
#    [q1, q2, q3, q4… q52,375]
# ]
def build_iq_matrix_type1(filepath):
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
          # little-endian
          #val = struct.unpack('<I', chunk)[0]

          # big-endian
          val = struct.unpack('>I', chunk)[0]

          # todo: understand and use the normalize function instead
          # https://www.tensorflow.org/api_docs/python/tf/linalg/normalize
          # or a rescale layer: https://www.tensorflow.org/api_docs/python/tf/keras/layers/Rescaling
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

  return [i_array, q_array]


# build I/Q data matrix from an I/Q file
# [
#    [i1, q1],
#    [i2, q2],
#    [i3, q3],
#    [i4, q4],
#    ...
#    [i52375, q52375]
# ] 
def build_iq_matrix_type2(filepath):
  iq_pair_count = 0
  index = 1

  iq_array = []
  iq_pair = []


  with open(filepath, mode='rb') as bin_file: 
      for chunk in iter(lambda: bin_file.read(4), ''):

          if chunk == b'':
              #print("Done")
              break

          # get chunk value
          # little-endian
          #val = struct.unpack('<I', chunk)[0]

          # big-endian
          val = struct.unpack('>I', chunk)[0]

          # todo: understand and use the normalize function instead
          # https://www.tensorflow.org/api_docs/python/tf/linalg/normalize
          # or a rescale layer: https://www.tensorflow.org/api_docs/python/tf/keras/layers/Rescaling
          if NORMALIZE:
            val = val / MAX_VAL
          
          if index % 2 != 0:
              # get the I value
              #print("I: " + str(hex(val)).upper())
              iq_pair.append(val)
          
          else:
              # get the Q value
              #print("Q: " + str(hex(val)).upper())
              iq_pair.append(val)
              iq_array.append(iq_pair)
              iq_pair = [] # reset
              iq_pair_count = iq_pair_count + 1

          # increment index
          index = index + 1

  return iq_array


# build I/Q data matrix from an I/Q file
# [
#    [i1, q1, i2, q2, i3, q3, i4, q4… i104750, q104750]
# ]
def build_iq_matrix_type3(filepath):
  iq_pair_count = 0
  index = 1

  iq_array = []

  with open(filepath, mode='rb') as bin_file: 
      for chunk in iter(lambda: bin_file.read(4), ''):

          if chunk == b'':
              #print("Done")
              break

          # get chunk value
          # little-endian
          #val = struct.unpack('<I', chunk)[0]

          # big-endian
          val = struct.unpack('>I', chunk)[0]

          # todo: understand and use the normalize function instead
          # https://www.tensorflow.org/api_docs/python/tf/linalg/normalize
          # or a rescale layer: https://www.tensorflow.org/api_docs/python/tf/keras/layers/Rescaling
          if NORMALIZE:
            val = val / MAX_VAL
          
          if index % 2 != 0:
              # get the I value
              #print("I: " + str(hex(val)).upper())
              iq_array.append(val)
          
          else:
              # get the Q value
              #print("Q: " + str(hex(val)).upper())
              iq_array.append(val)
              iq_pair_count = iq_pair_count + 1

          # increment index
          index = index + 1

  return iq_array


def build_iq_matrix(filepath, t):
  if t == 1:
    return build_iq_matrix_type1(filepath)
  elif t == 2:
    return build_iq_matrix_type2(filepath)
  elif t == 3:
    return build_iq_matrix_type3(filepath)
  else:
    return None

# build I/Q tensors
def build_tensors():

  # BEACON: iterate directory containing i/q data with beacon
  count_beacon = 0
  for filename in os.listdir(TRAINING_DIR_BEACON):
      iq_mat = build_iq_matrix(TRAINING_DIR_BEACON + "/" + filename, INPUT_TENSOR_DATA_STRUCTURE_TYPE)
      iq_data.append(iq_mat)
      count_beacon = count_beacon + 1

  # NO BEACON: iterate directory containing i/q data with no beacon
  count_nobeacon = 0
  for filename in os.listdir(TRAINING_DIR_NOBEACON):
      iq_mat = build_iq_matrix(TRAINING_DIR_NOBEACON + "/" + filename, INPUT_TENSOR_DATA_STRUCTURE_TYPE)
      iq_data.append(iq_mat)
      count_nobeacon = count_nobeacon + 1

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

  print(shuffled_input_tensor3d)
  #print(shuffled_output_tensor0d)

  return (shuffled_input_tensor3d, shuffled_output_tensor0d)


# build test I/Q tensors
def build_test_tensors():

  # repeat data matrices building but for test data
  test_count_beacon = 0
  for test_filename in os.listdir(TEST_DIR_BEACON):
      test_iq_mat = build_iq_matrix(TEST_DIR_BEACON + "/" + test_filename, INPUT_TENSOR_DATA_STRUCTURE_TYPE)
      test_iq_data.append(test_iq_mat)
      test_count_beacon = test_count_beacon + 1

  test_count_nobeacon = 0
  for test_filename in os.listdir(TEST_DIR_NOBEACON):
      test_iq_mat = build_iq_matrix(TEST_DIR_NOBEACON + "/" + test_filename, INPUT_TENSOR_DATA_STRUCTURE_TYPE)
      test_iq_data.append(test_iq_mat)
      test_count_nobeacon = test_count_nobeacon + 1


  # build the input and output tensor
  test_input_tensor3d = tf.constant(test_iq_data)
  test_output_tensor0d = tf.constant([1] * test_count_beacon + [0] * test_count_nobeacon)

  # shuffle the tensor indices
  test_indices = tf.range(start=0, limit=test_input_tensor3d.shape[0], dtype=tf.int32)
  test_shuffled_indices = tf.random.shuffle(test_indices)

  # shuffle the tensors using the suffled indices
  test_shuffled_input_tensor3d = tf.gather(test_input_tensor3d, test_shuffled_indices)
  test_shuffled_output_tensor0d = tf.gather(test_output_tensor0d, test_shuffled_indices)

  # print info on the input tensor
  print("\nInput Tensor for Testing")
  print("  Type of every element:", test_shuffled_input_tensor3d.dtype)
  print("  Number of axes:", test_shuffled_input_tensor3d.ndim)
  print("  Shape of tensor:", test_shuffled_input_tensor3d.shape)
  print("  Elements along axis 0 of tensor:", test_shuffled_input_tensor3d.shape[0])
  print("  Elements along the last axis of tensor:", test_shuffled_input_tensor3d.shape[-1])
  print("  Total number of elements: ", tf.size(test_shuffled_input_tensor3d).numpy())

  #print(test_shuffled_input_tensor3d)
  #print(test_shuffled_output_tensor0d)

  return (test_shuffled_input_tensor3d, test_shuffled_output_tensor0d)


###########################
#     Train the model     #
###########################

# create input and output tensors
input_tensor, output_tensor = build_tensors()

# create and train a model
model = create_model()
model.compile(optimizer="adam", loss="mse", metrics=["accuracy"])
model.fit(input_tensor, output_tensor, epochs=5)

# neural networks are prone to overfitting
# EarlyStopping monitors validation loss during training
# if validation loss stops decreasing for a specified amount of epochs (called patience), the training immediately halts
# https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/EarlyStopping
# early_stop_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', restore_best_weights= True, patience=5, verbose=1)


##############################
#     Evaluate the model     #
##############################

# create input and output test tensors
input_test_tensor, output_test_tensor = build_test_tensors()

# evaluate the model with test data.
model.evaluate(input_test_tensor, output_test_tensor)