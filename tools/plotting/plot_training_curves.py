import json
import matplotlib.pyplot as plt
import numpy as np
import re
import csv

first_run = True

data_objects = []
current_data_object = {}

float_re = re.compile('-?\ *[0-9]+\.[0-9]*(?:[Ee]\ *-?\ *[0-9]+)?')

with open('../ml/models/20220729.log') as topo_file:
    for line in topo_file:
        if 'Running for:' in line:
            if first_run:
                first_run = False
            else:
                data_objects.append(current_data_object)

            model = line.split(' ')[-1].rstrip()
            print(model)

            current_data_object = {}
            current_data_object['model'] = model
            current_data_object['metrics'] = []
            #current_data_object['val_loss'] = []


        if 'step - det_loss:' in line:

            metrics = re.findall(float_re, line)
            metrics = [m.replace(' ', '') for m in metrics]

            current_data_object['metrics'].append(metrics)

data_objects.append(current_data_object)

header = ['det_loss', 'cls_loss', 'box_loss', 'reg_l2_loss', 'loss', 'learning_rate', 'gradient_norm',  'val_det_loss', 'val_cls_loss', 'val_box_loss', 'val_reg_l2_loss', 'val_loss']

for d in data_objects:

    filename = d['model'].replace('.tflite', '.csv')

    with open('../ml/models/{}'.format(filename), 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for row in d['metrics']:
            writer.writerow(row)

    val_loss = [float(x[0]) for x in d['metrics']]
    det_loss = [float(x[-1]) for x in d['metrics']]
    plt.plot(val_loss, label='val_loss ' + d['model'])
    plt.plot(det_loss, label='det_loss ' + d['model'])

plt.legend()
plt.show()   

