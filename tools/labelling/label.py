#!/usr/bin/env python3

__maintainer__ = "Tom Mladenov"

import argparse
from sys import byteorder
import numpy as np
import sys
from os.path import exists

from PyQt5 import QtWidgets, QtCore, uic, QtGui
from pyqtgraph import PlotWidget, plot
import pyqtgraph as pg
import glob, os
import subprocess
import imageio
import csv
import logging

ROI_HEIGHT_BEACON_LONG = 77
ROI_HEIGHT_BEACON_SHORT = 66

class MainWindow(QtWidgets.QMainWindow):

    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)

        uic.loadUi('/tools/labelling/gui.ui', self)
        self.first = True

        # list with CSV lines
        self.csv_content = []

        # current file index
        self.current_index = 0
        self.current_filename = ""

        # holds numpy data for one image at a time
        self.data = []

        # list of currently highlighed beacons, contains qgraphicsrectitems
        self.selected_beacons_per_frame = []

        self.beacon_graphics = []

        self.p1 = self.win.addPlot(title="")
        self.img = pg.ImageItem(axisOrder='row-major')
        self.img.hoverEvent = self.imageHoverEvent
        self.p1.addItem(self.img)

        self.selector_roi = pg.ROI([0, 0], [22, ROI_HEIGHT_BEACON_SHORT], pen='r')
        self.selector_roi.setAcceptedMouseButtons(QtCore.Qt.MouseButton.LeftButton)
        self.selector_roi.sigRegionChanged.connect(self.update_zoom)
        self.selector_roi.sigClicked.connect(self.tag_beacon)
        self.p1.addItem(self.selector_roi)

        self.zoom = pg.ImageItem(axisOrder='row-major')
        self.p1.addItem(self.zoom)

        self.p1.invertY()

    # function triggered by pressing B key, toggles for a long beacon or short beacon selection
    def toggle_beacon_length(self):
        w, h = self.selector_roi.size()
        if h == ROI_HEIGHT_BEACON_SHORT:
            #current ROI selector is for a short beacon, make it a long one
            self.selector_roi.setSize((22, ROI_HEIGHT_BEACON_LONG))
        else:
            #current selector ROI is for a long beacon, make it a short one
            self.selector_roi.setSize((22, ROI_HEIGHT_BEACON_SHORT))

    # function that updates the zoomed in view of currently selected beacon when mouse moves
    def update_zoom(self, roi):
        self.zoom.setImage(self.selector_roi.getArrayRegion(self.data, self.zoom))
 
    def set_file_list(self, file_list):
        self.file_list = file_list

    def set_output_file(self, output_file):
        self.output_file = output_file

    # process key events for app control
    def keyPressEvent (self, eventQKeyEvent):
        if type(eventQKeyEvent) == QtGui.QKeyEvent:
            key = eventQKeyEvent.key()
            if key == QtCore.Qt.Key_A:
                self.previous()
            elif key == QtCore.Qt.Key_F:
                self.next()
            elif key == QtCore.Qt.Key_B:
                self.toggle_beacon_length()
            elif key == QtCore.Qt.Key_R:
                self.reset()         

    # clears all selected beacons, triggerd by pressing R key (RESET)
    def reset(self):
        for b in self.beacon_graphics:
            self.p1.removeItem(b)

        self.beacon_graphics = []
        self.selected_beacons_per_frame = []

    # triggered when clicking, will add a rectangle where the ROI is currently located
    def tag_beacon(self, event):
        x, y = self.selector_roi.pos()
        w, h = self.selector_roi.size()
        rectangle = pg.QtGui.QGraphicsRectItem(x, y, w, h)
        rectangle.setPen(pg.mkPen('g'))
        self.p1.addItem(rectangle)
        beacon = {}
        beacon['xmin'] = int(x)
        beacon['ymin'] = int(y)
        beacon['xmax'] = int(x + w)
        beacon['ymax'] = int(y + h)
        self.selected_beacons_per_frame.append(beacon)
        self.beacon_graphics.append(rectangle)



    def imageHoverEvent(self, event):
        """Show the position, pixel, and value under the mouse cursor.
        """
        if event.isExit():
            self.p1.setTitle("")
            return
        pos = event.pos()
        i, j = pos.y(), pos.x()
        i = int(np.clip(i, 0, self.data.shape[0] - 1))
        j = int(np.clip(j, 0, self.data.shape[1] - 1))
        val = self.data[i, j]
        ppos = self.img.mapToParent(pos)
        x, y = ppos.x(), ppos.y()
        self.p1.setTitle("pos: (%0.2f, %0.2f)  pixel: (%d, %d)" % (x, y, j, i))
        self.selector_roi.setPos(x - 11, y - 23)

    def write_content(self):
        with open(self.output_file, 'w') as f: 
            write = csv.writer(f) 
            write.writerow(['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax'])
            write.writerows(self.csv_content)
        print(self.csv_content)

    def closeEvent(self, event):
        self.write_content()

    # function to move to next raw file
    def next(self):

        if not self.first:
            
            image_height, image_width, level = self.data.shape

            # first check if the user tagged some beacons!
            if len(self.selected_beacons_per_frame) == 0:
                # no beacons :(
                csv_line = [self.current_filename, str(image_width), str(image_height), "", "", "", "", ""]
                self.csv_content.append(csv_line)
            else:
                # 1 or more beacons :)))
                for beacon in self.selected_beacons_per_frame:
                    if beacon['ymax'] - beacon['ymin'] == ROI_HEIGHT_BEACON_LONG:
                        csv_line = [self.current_filename, str(image_width), str(image_height), "beacon_long", beacon['xmin'], beacon['ymin'], beacon['xmax'], beacon['ymax']]
                    else:
                        csv_line = [self.current_filename, str(image_width), str(image_height), "beacon_short", beacon['xmin'], beacon['ymin'], beacon['xmax'], beacon['ymax']]
                    self.csv_content.append(csv_line)

        if self.first:
            self.first = False

        self.reset()

        self.current_index += 1
        try:
            print("loading image: {}".format(self.file_list[self.current_index]))
            self.show_image(self.file_list[self.current_index])
        except Exception as e:
            print(e)

    # function to move to previous raw file (will overwrite output)
    def previous(self):

        self.reset()

        self.current_index -= 1
        try:
            file = self.file_list[self.current_index]
            self.csv_content = [line for line in self.csv_content if line[0] != file]
            self.show_image(file)
        except Exception as e:
            print(e)

    # function that is triggering rendering of the FFT
    def show_image(self, file):
        png = file + ".png"
        if not exists(png):
            os.system("renderfall -n 512 -v -f float32 -l 256 -w hann {}".format(file))
        
        self.data = imageio.imread(png)
        self.img.setImage(self.data)
        self.p1.setTitle(png)
        self.current_filename = file



def main():

    cliParser = argparse.ArgumentParser(description='Plots quadrature IQ signals')    
    
    cliParser.add_argument('output_csv', type=str, help='output csv file containing raw filenames and labelled data')
    args = cliParser.parse_args()

    app = QtWidgets.QApplication(sys.argv)
    main = MainWindow()

    main.set_file_list(sorted(glob.glob("/input/*.cf32"), key=os.path.getmtime))
    main.set_output_file(args.output_csv)
    main.show()

    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
    