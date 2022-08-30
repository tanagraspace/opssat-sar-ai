import csv
import numpy as np

# this script calculates the average precision and average recall (area under precision recall curves) and dthe F-score and performs linear interpolation
# based on a set recall requirement, it will calculate the corresponding precision and the confidence level which should be used to accept predictions

# paths to csv files containing the precision-recall curves for each of the 4 models
MODEL_A = 'precision_recall_iou0.38_41d6171e_efficientdet_lite0.tflite.csv'
MODEL_B = 'precision_recall_iou0.38_08f16b89_efficientdet_lite0.tflite.csv'
MODEL_C = 'precision_recall_iou0.38_b5abe1d2_efficientdet_lite0.tflite.csv'
MODEL_D = 'precision_recall_iou0.38_4bd18f4a_efficientdet_lite0.tflite.csv'

def calc_average_precision(recalls, precisions):
    cumul = 0
    past_recall = 0
    try:
        for i in range(len(recalls)):
            AP_tmp = (recalls[i] - recalls[i+1])*precisions[i]
            past_recall = recalls[i]
            cumul += AP_tmp

    except Exception as e:
        pass

    return cumul


def calc_average_recall(recalls, precisions):
    cumul = 0
    past_precision = 0
    try:
        for i in range(len(recalls)):
            AR_tmp = (precisions[i+1] - precisions[i])*recalls[i]
            cumul += AR_tmp

    except Exception as e:
        pass

    return cumul

def read_precision_recall(filename):

    with open(filename, 'r') as file:
        reader = csv.reader(file)
        next(reader)
        rows = [row for row in reader]

        confidences = np.array([float(item[0]) for item in rows], dtype=np.float)
        precisions = np.array([float(item[1]) for item in rows], dtype=np.float)
        recalls = np.array([float(item[2]) for item in rows], dtype=np.float)
        fp= np.array([int(item[3]) for item in rows], dtype=np.uint32)
        tp = np.array([int(item[4]) for item in rows], dtype=np.uint32)
        fn = np.array([int(item[5]) for item in rows], dtype=np.uint32)

    return confidences, precisions, recalls, fp, tp, fn, filename

def print_results(confidences, precisions, recalls, fp, tp, fn, modelname):

    print("{}: at {} recall: ".format(modelname, RECALL))
    average_precision = calc_average_precision(recalls, precisions)
    average_recall = calc_average_recall(recalls, precisions)
    precision = np.interp(RECALL, recalls[::-1], precisions[::-1])
    confidence = np.interp(RECALL, recalls[::-1], confidences[::-1])
    fscore = 2*(precision*RECALL) / (precision + RECALL)
    print("Precision: {}, F-score: {}, Confidence: {}, AP: {}, AR: {}".format(precision, fscore, confidence, average_precision, average_recall))


if __name__ == "__main__":

    RECALL = 0.90
    print_results(*read_precision_recall(MODEL_A))
    print_results(*read_precision_recall(MODEL_B))
    print_results(*read_precision_recall(MODEL_C))
    print_results(*read_precision_recall(MODEL_D))

















