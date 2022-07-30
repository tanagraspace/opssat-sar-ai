import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"

# https://stackoverflow.com/questions/47910681/tensorflow-setting-allow-growth-to-true-does-still-allocate-memory-of-all-my-gp
# limit GPU memory alloc
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
from tflite_model_maker.config import ExportFormat, QuantizationConfig
from tflite_model_maker import model_spec
from tflite_model_maker import object_detector
from wai.tmm.common.hyper import add_hyper_parameters
from tflite_support import metadata

from tflite_support import flatbuffers
from tflite_support import metadata as _metadata
from tflite_support import metadata_schema_py_generated as _metadata_fb
import json
import hashlib
import datetime
import uuid

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

def get_md5(filename):
    with open(filename, 'rb') as file_to_check:
        # read contents of the file
        data = file_to_check.read()    
        # pipe contents of the file through
        return hashlib.md5(data).hexdigest()

if __name__ == "__main__":
    cliParser = argparse.ArgumentParser(description='Train a TF object detection model and export to .tflite format')    
    cliParser.add_argument('--labels_file', type=str, help='absolute path to csv file containg the labels', required=True)
    cliParser.add_argument('--model_type', type=str, help='which model to use', required=True)
    cliParser.add_argument('--num_epochs', type=int, help='number of epochs to train for', required=True)
    cliParser.add_argument('--use_augmix', type=str, help='augmix policy', required=False)
    cliParser.add_argument('--model_filename', type=str, help='filename of .tflite model to generate (file will be put in /output)', required=True)
    args = cliParser.parse_args()
    
    MODEL_FILENAME = str(uuid.uuid4())[:8] + "_" + args.model_filename
    print("Running for: {}".format(MODEL_FILENAME))

    EXPORT_DIR = '/output'

    # disable GPU usage as a test
    #os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    gpu_available = tf.test.is_gpu_available()
    if gpu_available:
        print("Found GPU")
        hw = 'GPU'
        # limit GPU memory allocation
        gpus = tf.config.experimental.list_physical_devices('GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    else:
        print("No GPU found, continuing with CPU...")
        hw = 'CPU'

        #sys.exit()
       
    start = datetime.datetime.utcnow()
    # load the csv with filepaths to spectra, and labelled bounding boxes
    train_data, validation_data, test_data = object_detector.DataLoader.from_csv(args.labels_file)

    print("Samples: {} train, {} validation, {} test".format(len(train_data), len(validation_data), len(test_data)))

    # load model architecture 'efficientdet_lite1'
    spec = model_spec.get(args.model_type)

    # override hyper params
    overrides = {
        'num_classes' : 1,
        'tflite_max_detections' : 15,
        'verbose' : 1,
        'num_epochs' : args.num_epochs,
        'batch_size' : 16,
        'max_instances_per_image' : 8,
        'learning_rate': 0.08
    }
    if args.use_augmix is not None:
        overrides['use_augmix'] = args.use_augmix
        
    add_hyper_parameters(spec, overrides)
    print(spec.config)

    # train model and evaluate with test_data afterwards the non-quantized model
    #model = object_detector.create(train_data, model_spec=spec, do_train=False, train_whole_model = True)
    #model.train(train_data=train_data,  validation_data=validation_data, epochs=overrides['num_epochs'], batch_size=overrides['batch_size'])
    
    # train model and evaluate with test_data afterwards the non-quantized model
    model = object_detector.create(model_spec=spec, do_train=True, train_whole_model = True, train_data=train_data,  validation_data=validation_data, epochs=overrides['num_epochs'], batch_size=overrides['batch_size'])
    metrics = model.model.history.history
    print(metrics)


    
    
    model_evaluation = model.evaluate(test_data)
    print(str(model_evaluation))
     
    #config = QuantizationConfig.for_float16()
    
    # export tflite and re-asses model with test_data

    model.export(export_dir='/output', tflite_filename=MODEL_FILENAME)
    #tflite_evaluation = model.evaluate_tflite('/output/' + MODEL_FILENAME, test_data)
    
    write_labels(train_data, '/output')
    
    stop = datetime.datetime.utcnow()

    # generate a report
    model_generation_report = {}
    model_generation_report['filename'] = MODEL_FILENAME
    model_generation_report['md5sum'] = get_md5(EXPORT_DIR + '/' + MODEL_FILENAME)
    model_generation_report['size'] = os.path.getsize(EXPORT_DIR + '/' + MODEL_FILENAME) 
    model_generation_report['generated'] = str(datetime.datetime.utcnow())
    model_generation_report['hparams'] = spec.config.as_dict()
    model_generation_report['hardware'] = hw
    model_generation_report['labelset'] = {'file' : args.labels_file, 'samples_train' : len(train_data), 'samples_test' : len(test_data), 'samples_validation' : len(validation_data) }
    model_generation_report['model_evaluation'] = {k : float(v) for k, v in model_evaluation.items()}
    model_generation_report['metrics'] = metrics
    print(metrics)
    model_generation_report['train_time'] = int((stop - start).total_seconds())

    with open('{}/{}.json'.format(EXPORT_DIR, MODEL_FILENAME), 'w') as fp:
        json.dump(model_generation_report, fp)