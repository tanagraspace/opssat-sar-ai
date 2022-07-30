import argparse
import traceback
import numpy as np
import tensorflow as tf
from wai.tmm.objdet.predict_utils import box_to_bbox_polygon
#from wai.tmm.common.predict_utils import preprocess_image
from wai.tmm.common.io import load_model, load_classes
import matplotlib.pyplot as plt
import sys
from datetime import datetime
from opex import ObjectPredictions, ObjectPrediction, BBox, Polygon
import cv2
import glob

import tensorflow as tf

from wai.tmm.common.predict_utils import set_input_tensor, get_output_tensor

def detect_objects(interpreter, image, image_size, threshold=0.1, labels=None):
    """
    Returns a list of detection results, each a dictionary of object info.
    :param interpreter: the model to use use
    :param image: the preprocessed image to make a prediction for
    :type image: np.ndarray
    :param image_size: the image size tuple (height, width)
    :type image_size: tuple
    :param threshold: the probability threshold to use
    :type threshold: float
    :param labels: the class labels
    :type labels: list
    :return: the predicted objects
    :rtype: ObjectPredictions
    """

    start = datetime.now()
    timestamp = str(start)

    # Feed the input image to the model
    set_input_tensor(interpreter, image)
    interpreter.invoke()

    end = datetime.now()
    meta = {"prediction_time": str((end-start).total_seconds())}

    # Get all outputs from the model
    boxes = get_output_tensor(interpreter, 0)
    classes = get_output_tensor(interpreter, 1)
    scores = get_output_tensor(interpreter, 2)
    count = int(get_output_tensor(interpreter, 3))

    objs = []
    for i in range(count):
        if scores[i] >= threshold:
            label = str(classes[i])
            if labels is not None:
                label = labels[int(classes[i])]
            bbox, poly = box_to_bbox_polygon(boxes[i], image_size)
            obj = ObjectPrediction(score=float(scores[i]), label=label, bbox=bbox, polygon=poly)
            objs.append(obj)

    result = ObjectPredictions(timestamp=timestamp, id="", objects=objs, meta=meta)
    return result

def preprocess_image(image_path, input_size):
    """
    Preprocess the input image to feed to the TFLite model.
    :param image_path: the image to load
    :type image_path: str
    :param input_size: the tuple (height, width) to resize to
    :type input_size: tuple
    :return: the preprocessed image
    :rtype: np.ndarray
    """
    img = tf.io.read_file(image_path)
    img = tf.io.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.uint8)
    original_image = img
    resized_img = tf.image.resize(img, input_size)
    resized_img = resized_img[tf.newaxis, :]
    return resized_img, original_image

def predict(model, labels, image, threshold, output=None):
    """
    Uses an object detection model to make a prediction for a single image.
    :param model: the model to load
    :type model: str
    :param labels: the text file with the labels (one label per line)
    :type labels: str
    :param image: the image to make the predictions for
    :type image: str
    :param threshold: the probability threshold to use
    :type threshold: float
    :param output: the JSON file to store the predictions in, gets output to stdout if None
    :type output: str
    """

    interpreter = load_model(model)
    _, input_height, input_width, _ = interpreter.get_input_details()[0]['shape']

    #print("Model input: {}".format(interpreter.get_input_details()[0]['shape']))
    #print("Model output: {}".format(interpreter.get_output_details()[0]['shape']))

    #print(interpreter.get_input_details())
    classes = load_classes(labels)
    #print(classes)

    preprocessed_image, original_image = preprocess_image(image, (input_height, input_width))

    image_height, image_width, _ = original_image.shape
    #print("original height: {} width: {}".format(image_height, image_width))
    results = detect_objects(interpreter, preprocessed_image, (image_height, image_width), threshold=threshold, labels=classes)
    if output is None:
        print(results.to_json_string())
    else:
        with open(output, "w") as f:
            f.write(results.to_json_string())

    return results




if __name__ == "__main__":

    cliParser = argparse.ArgumentParser(description='Train a TF object detection model and export to .tflite format')    
    cliParser.add_argument('--image', type=str, help='absolute path to input image', required=True)
    cliParser.add_argument('--model', type=str, help='absolute path to .tflite model', required=True)
    cliParser.add_argument('--labels', type=str, help='absolute path to labels file', required=True)
    args = cliParser.parse_args()

    #for i in range(0, 1):
    #for filepath in glob.iglob('/input/sdr_iq*'):
    #INPUT_IMAGE = "/input/1645389029.12_raw_ulf406025000_dlf1544185000_gain65_inputsr1500000_outputsr37500_{}.cf32.jpg".format(str(i).zfill(4))
    #INPUT_IMAGE = filepath

    if '*' in args.image:
        for filepath in sorted(glob.iglob(args.image)):
            INPUT_IMAGE = filepath
            OUTPUT_IMAGE = INPUT_IMAGE.replace('.jpg', '.pred.jpg').replace('/input/', '/output/')

            print("Running inference on image: {}".format(INPUT_IMAGE))

            results = predict(  model=args.model, 
                                labels=args.labels, 
                                image=INPUT_IMAGE, 
                                threshold=0.25, 
                                output=OUTPUT_IMAGE.replace('.jpg', '.json'))

            annotated_image = cv2.imread(INPUT_IMAGE)

            font                   = cv2.FONT_HERSHEY_SIMPLEX
            fontScale              = 0.3
            fontColor              = (255,0,0)
            thickness              = 1
            lineType               = 2

            for box in results['objects']:
                
                x1 = box['bbox']['left']
                y1 = box['bbox']['top']
                x2 = box['bbox']['right']
                y2 = box['bbox']['bottom']

                if round((100 * box['score'] * 2), 1) > 50.0:

                    cv2.rectangle(annotated_image,(x1,y1),(x2,y2),(0,255,0),1)
                    cv2.putText(    annotated_image, '{}: {}'.format(box['label'], round((100 * box['score'] * 2), 1)), 
                                    (x1, y1 - 10), 
                                    font, 
                                    fontScale,
                                    fontColor,
                                    thickness,
                                    lineType)
                else:

                    cv2.rectangle(annotated_image,(x1,y1),(x2,y2),(255,0,0),1)
                    cv2.putText(    annotated_image, '{}: {}'.format(box['label'], round((100 * box['score'] * 2), 1)), 
                                    (x1, y1 - 10), 
                                    font, 
                                    fontScale,
                                    fontColor,
                                    thickness,
                                    lineType)
                    

            cv2.imwrite(OUTPUT_IMAGE, annotated_image)
            print("Done writing image: {}".format(OUTPUT_IMAGE))

    else:

        INPUT_IMAGE = args.image
        OUTPUT_IMAGE = INPUT_IMAGE.replace('.jpg', '.pred.jpg').replace('/input/', '/output/')

        print("Running inference on image: {}".format(INPUT_IMAGE))

        results = predict(  model=args.model, 
                            labels=args.labels, 
                            image=INPUT_IMAGE, 
                            threshold=0.15, 
                            output=OUTPUT_IMAGE.replace('.jpg', '.json'))

        annotated_image = cv2.imread(INPUT_IMAGE)

        font                   = cv2.FONT_HERSHEY_SIMPLEX
        fontScale              = 0.3
        fontColor              = (255,0,0)
        thickness              = 1
        lineType               = 2

        for box in results['objects']:
            
            x1 = box['bbox']['left']
            y1 = box['bbox']['top']
            x2 = box['bbox']['right']
            y2 = box['bbox']['bottom']

            if round((100 * box['score'] * 2), 1) > 50.0:

                cv2.rectangle(annotated_image,(x1,y1),(x2,y2),(0,255,0),1)
                cv2.putText(    annotated_image, '{}: {}'.format(box['label'], round((100 * box['score'] * 2), 1)), 
                                (x1, y1 - 10), 
                                font, 
                                fontScale,
                                fontColor,
                                thickness,
                                lineType)
            else:

                cv2.rectangle(annotated_image,(x1,y1),(x2,y2),(255,0,0),1)
                cv2.putText(    annotated_image, '{}: {}'.format(box['label'], round((100 * box['score'] * 2), 1)), 
                                (x1, y1 - 10), 
                                font, 
                                fontScale,
                                fontColor,
                                thickness,
                                lineType)
                

        cv2.imwrite(OUTPUT_IMAGE, annotated_image)
        print("Done writing image: {}".format(OUTPUT_IMAGE))

