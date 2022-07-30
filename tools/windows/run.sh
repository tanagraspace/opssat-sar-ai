#!/bin/bash

python3 learn.py --labels_file /input/master_labelset_1class_train.csv --model_type efficientdet_lite0 --num_epochs 20 --model_filename efficientdet_lite0.tflite >> /output/20220729.log
#python3 learn.py --labels_file /input/master_labelset_1class_train.csv --model_type efficientdet_lite1 --num_epochs 25 --model_filename efficientdet_lite1.tflite >> /output/20220728.log
python3 learn.py --labels_file /input/master_labelset_1class_train.csv --model_type efficientdet_lite0 --num_epochs 20 --use_augmix v1 --model_filename efficientdet_lite0.tflite >> /output/20220729.log
#python3 learn.py --labels_file /input/master_labelset_1class_train.csv --model_type efficientdet_lite1 --num_epochs 25 --use_augmix v1 --model_filename efficientdet_lite1.tflite >> /output/20220728.log