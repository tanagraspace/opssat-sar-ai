import json
import matplotlib.pyplot as plt
import numpy as np
import re
import csv

# path with json files indicating all the predictied bounding boxes for teh given image
# each line in this file is a beacon, use this as main iterator
GROUND_TRUTH_CSV = '../labelling/master_labelset_1class_train.csv'

# model for which to plot the precision recall curves
MODEL_ID = '4bd18f4a'

# path to the csv file containing the predictions by this model for the TEST dataset
PREDICTION_CSV = '../ml/output/predictions_{}_efficientdet_lite0.tflite.csv'.format(MODEL_ID)



def get_iou(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
    if interArea == 0:
        return 0
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = abs((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = abs((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou

def get_tp_fp(filename, prediction, IoU, score, pred_class):

    # do not take prediction into account if below confidence threshold
    if score > confidence_threshold:
        # lookup the ground truth items associated with this prediction (via the filename)
        ground_truth_items = [gt for gt in ground_truth_dict if gt['filename'] == filename]

        # loop over the ground truth items where the filename matches this prediction
        for gt in ground_truth_items:

            # if for a certairn ground truth (gt), the prediction has a larger than IoU and the prediction class is a beacon, mark it as found: the prediction is correct
            if get_iou([gt['xmin'], gt['ymin'],gt['xmax'],gt['ymax'],] , [prediction[1], prediction[0], prediction[3], prediction[2]]) >= IoU and pred_class == 0 and gt['found'] == False:
                # mark as found in the global ground truth dict
                for d in ground_truth_dict:
                    if d['filename'] == filename and d['xmin'] == gt['xmin'] and d['ymax'] == gt['ymax']:
                        d['found'] = True
                        # return tp=1, fp=0
                        return 1, 0

            # else if the IoU is less than, and the prediciton class is a beacon, the prediction is not correct
            elif get_iou([gt['xmin'], gt['ymin'],gt['xmax'],gt['ymax'],] , [prediction[1], prediction[0], prediction[3], prediction[2]]) < IoU and pred_class == 0:
                # return tp=0, fp=1
                return 0, 1
            
        # anything past here is either background class prediction or has already been found
        return 0, 0
    else:
        # do not take prediction into account due to too low threshold
        return 0, 0
    

def get_fn(predictions, gt, IoU):

    j = 0
    for prediction in predictions['bboxes']:

        if predictions['scores'][j] > confidence_threshold:
            if get_iou([gt['xmin'], gt['ymin'],gt['xmax'],gt['ymax'],] , [prediction[1], prediction[0], prediction[3], prediction[2]]) >= IoU and predictions['classes'][j] == 0:
                # gt overlaps with on of the predictions - not a false negative
                return 0
        j += 1

    # if we get here none of the predictions overlapped the ground truth - increment false negative count
    return 1





if __name__ == "__main__":

    predictions_dict = []

    #[{'filename': '/input/1645435895.98_raw_ulf406025000_dlf1544185000_gain65_inputsr1500000_outputsr37500_0042.cf32.jpg', 
    #    'bboxes': [[0.5715447, 0.29249182, 0.85586256, 0.3716286], [-0.13623482, 0.002878368, 1.4231448, 0.6329854], [-0.079885185, 0.33115098, 1.4385226, 0.96125793], [-0.546404, 0.16849136, 1.2118299, 0.8789505], [-0.6872189, -0.20025958, 1.6182811, 0.6830252], [0.5919069, -0.07469796, 1.1761422, 0.33859992], [-0.63639104, 0.50594425, 1.0450335, 1.0477587], [-0.22204071, -0.17326117, 0.7346929, 2.258338], [-0.18804525, 0.45370305, 0.5778605, 1.5518547], [-0.4000766, -0.12554511, 0.62690675, 0.4855043], [-0.24439806, 0.39965078, 1.9708376, 1.2371296], [0.025077105, -0.117828056, 1.3479469, 0.3889907], [0.7113235, 0.8329519, 0.9669174, 0.914224], [0.36001801, -0.109809995, 1.3963082, 2.1351142], [-0.10409978, -0.019434452, 0.5315695, 0.8799318]], 
    #    'classes': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0], 
    #    'scores': [0.41796875, 0.12890625, 0.0703125, 0.0625, 0.046875, 0.046875, 0.04296875, 0.03515625, 0.03125, 0.03125, 0.03125, 0.03125, 0.02734375, 0.02734375, 0.02734375]}



    # iterate over predictions to load them into memory for fast lookup
    with open(PREDICTION_CSV, 'r') as file:
        reader = csv.reader(file)
        predictions_list = [row for row in reader]
        for pred in predictions_list:
            filename = pred[0]
            bboxes = [float(item) for item in pred[1:61]]
            classes = [float(item) for item in pred[62:77]]
            scores = [float(item) for item in pred[78:93]]

            bboxes = [bboxes[i:i+4] for i in range(0, len(bboxes), 4)]

            predictions_dict.append({'filename' : filename, 'bboxes' : bboxes, 'classes' : classes, 'scores' : scores})


    ground_truth_dict = []

    #[{'filename': '/input/1645435895.98_raw_ulf406025000_dlf1544185000_gain65_inputsr1500000_outputsr37500_0042.cf32.jpg', 
    #    'xmin': 0.3115460577856194, 
    #    'ymin': 0.5628100841733863, 
    #    'xmax': 0.3545148077856194, 
    #    'ymax': 0.8863394959380921},

    with open(GROUND_TRUTH_CSV, 'r') as file:
        reader = csv.reader(file)
        test_rows = [row for row in reader]

        for beacon in test_rows:
            kind = beacon[0]
            filename = beacon[1]
            xmin = float(beacon[3])
            ymin = float(beacon[4])
            xmax = float(beacon[7])
            ymax = float(beacon[8])
            ground_truth_dict.append({'kind' : kind, 'filename' : filename, 'xmin' : xmin, 'ymin' : ymin, 'xmax' : xmax, 'ymax' : ymax, 'found' : False})


    precisions = []
    recalls = []

    for IoU in np.arange (0.38, 0.52, 0.02):

        print("Running for IOU {}".format(IoU))

        OUTPUT_CSV = 'precision_recall_iou{:.2f}_{}_efficientdet_lite0.tflite.csv'.format(round(IoU, 2), MODEL_ID)

        with open(OUTPUT_CSV, 'w') as file:
            writer = csv.writer(file)
            writer.writerow(['confidence', 'precision', 'recall', 'fp', 'tp', 'fn'])
            
            # increment the confidence threshold
            for confidence_threshold in np.arange (0, 0.50, 0.01):
                print('Running for treshold = {}'.format(confidence_threshold))

                # Determine true and false positives
                # TP/FP
                # we need to look from the point of view from the predictions, and for each of them determine 

                tp_total = 0
                fp_total = 0
                fn_total = 0
                for gt in ground_truth_dict:
                    gt['found'] = False

                for predictions in predictions_dict:
                    filename = predictions['filename']

                    i = 0
                    for bbox in predictions['bboxes']:

                        tp, fp = get_tp_fp(filename, bbox, IoU, predictions['scores'][i], predictions['classes'][i])
                        tp_total = tp_total + tp
                        fp_total = fp_total + fp
                        i += 1

                for gt in ground_truth_dict:
                    if gt['kind'] == 'TEST':
                        # look in the entire ground truth and check if any of our predictions are overlapping with correct class and good score
                        # first fetch our predictions that match the current ground truth
                        predictions_o = [pred for pred in predictions_dict if pred['filename'] == gt['filename']]

                        if len(predictions_o) != 0:
                            predictions = predictions_o[0]
                            fn = get_fn(predictions, gt, IoU)
                            fn_total = fn_total + fn

                if fn_total != 0 and fp_total != 0:
                    precision = float(tp_total) / (tp_total + fp_total)
                    recall = float(tp_total) / (tp_total + fn_total)

                    precisions.append(precision)
                    recalls.append(recall)

                    print("confidence: {}  precision: {}  recall:{}".format(confidence_threshold, precision, recall))
                    writer.writerow([str(confidence_threshold*2), str(precision), str(recall), str(fp_total), str(tp_total), str(fn_total)])
                else:
                    print("error: fn_total is 0")

        print("DONE with IOU {}".format(IoU))



        


