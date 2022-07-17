import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"
from tflite_model_maker.config import ExportFormat, QuantizationConfig
from tflite_model_maker import model_spec
from tflite_model_maker import object_detector
from wai.tmm.common.hyper import add_hyper_parameters
from tflite_support import metadata

from tflite_support import flatbuffers
from tflite_support import metadata as _metadata
from tflite_support import metadata_schema_py_generated as _metadata_fb

import argparse
import sys

import tensorflow as tf
assert tf.__version__.startswith('2')

tf.get_logger().setLevel('INFO')
from absl import logging
logging.set_verbosity(logging.INFO)


def write_labels(data, output_dir):
    """
    Writes the labels to disk.
    
    :param data: the training data to get the labels from
    :param output_dir: the output directory to store the labels in (labels.json, labels.txt)
    """
    keys = list(data.label_map.keys())
    keys.sort()
    labels = {}
    for k in keys:
        labels[k] = str(data.label_map[k])
    with open(output_dir + "/labels.json", "w") as f:
        json.dump(labels, f)
    with open(output_dir + "/labels.txt", "w") as f:
        for k in keys:
            f.write(labels[k])
            f.write("\n")

if __name__ == "__main__":
    cliParser = argparse.ArgumentParser(description='Train a TF object detection model and export to .tflite format')    
    cliParser.add_argument('--labels_file', type=str, help='absolute path to csv file containg the labels', required=True)
    cliParser.add_argument('--model_type', type=str, help='which model to use', required=True)
    cliParser.add_argument('--hyper_params', type=str, help='absolute path to YAML file specifying hyper parameters', required=True)
    cliParser.add_argument('--model_filename', type=str, help='filename of .tflite model to generate (file will be put in /output)', required=True)
    args = cliParser.parse_args()

    # disable GPU usage as a test
    #os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    gpu_available = tf.test.is_gpu_available()
    if gpu_available:
        print("Found GPU")
    else:
        print("No GPU found, or disabled")
        sys.exit()
       
    # limit GPU memory allocation
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
        

    # load the csv with filepaths to spectra, and labelled bounding boxes
    train_data, val_data, test_data = object_detector.DataLoader.from_csv(args.labels_file)

    # load model architecture 'efficientdet_lite1'
    spec = model_spec.get(args.model_type)

    # override hyper params
    overrides = {
        'num_classes' : 3,
        'tflite_max_detections' : 10,
        'verbose' : 1
    }
    add_hyper_parameters(spec, overrides)
    print(spec.config)

    model = object_detector.create(train_data, model_spec=spec, train_whole_model=True, epochs=25, batch_size=16, validation_data=val_data)
    model.export(export_dir='/output', tflite_filename=args.model_filename)
    write_labels(train_data, '/output')