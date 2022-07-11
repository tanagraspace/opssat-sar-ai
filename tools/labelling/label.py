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
        if x < 0 or \
           x + w > self.image_width or \
           y < 0 or \
           y + h > self.image_height:
            print("ROI boundaries outside of image limits!")
        else:

            x, y = self.selector_roi.pos()
            w, h = self.selector_roi.size()
            rectangle = pg.QtGui.QGraphicsRectItem(x, y, w, h)
            rectangle.setPen(pg.mkPen('g'))
            self.p1.addItem(rectangle)
            beacon = {}
            beacon['xmin'] = x
            beacon['ymin'] = y
            beacon['xmax'] = x + w
            beacon['ymax'] = y + h
            self.selected_beacons_per_frame.append(beacon)
            self.beacon_graphics.append(rectangle)
            print(beacon)



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
        self.selector_roi.setPos(x - 11, y - 23) #offset roi from mouse
        
        #print("Mouse_position = {}x {}y".format(x, y))
        

        x, y = self.selector_roi.pos()
        w, h = self.selector_roi.size()

        #print("ROI coords = {}x {}y {}w {}h".format(x, y, w, h))

        if x < 0 or \
           x + w > self.image_width or \
           y < 0 or \
           h + y > self.image_height:
            self.selector_roi.setPen(pg.mkPen('r'))
        else:
            self.selector_roi.setPen(pg.mkPen('y'))
            #color yellow




    def write_content(self):
        with open(self.output_file, 'w') as f: 
            write = csv.writer(f) 
            write.writerow(['tag', 'filename', 'class', 'xmin', 'ymin', '', '', 'xmax', 'ymax', '', ''])
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
                csv_line = [    
                                "TRAINING", 
                                self.current_filename_image,
                                "", 
                                "",
                                "",
                                "",
                                "",
                                "",
                                "",
                                "",
                                ""
                            ]
                
                #self.csv_content.append(csv_line)
            else:
                # 1 or more beacons :)))
                for beacon in self.selected_beacons_per_frame:
                    if beacon['ymax'] - beacon['ymin'] == ROI_HEIGHT_BEACON_LONG:
                        csv_line = [    "TRAINING", 
                                        self.current_filename_image,
                                        "beacon_long", 
                                        float(beacon['xmin']/image_width), 
                                        float(beacon['ymin']/image_height),
                                        "",
                                        "",
                                        float(beacon['xmax']/image_width), 
                                        float(beacon['ymax']/image_height),
                                        "",
                                        ""
                                    ]
                    else:
                        csv_line = [    "TRAINING", 
                                        self.current_filename_image,
                                        "beacon_short", 
                                        float(beacon['xmin']/image_width), 
                                        float(beacon['ymin']/image_height),
                                        "",
                                        "",
                                        float(beacon['xmax']/image_width), 
                                        float(beacon['ymax']/image_height),
                                        "",
                                        ""
                                    ]
                    self.csv_content.append(csv_line)

                    print("Image width: {}px   height {}px".format(image_width, image_height))
                    print("writing line: {}".format(csv_line))

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
        jpg = file + ".jpg"
        if not exists(png) or not exists(jpg):
            os.system("renderfall -n 512 -v -f float32 -l 256 -w hann {} && pngtopnm {} | ppmtojpeg > {}".format(file, png, jpg))
        
        self.data = imageio.imread(jpg)
        self.image_height, self.image_width, self.level = self.data.shape
        self.img.setImage(self.data)
        self.p1.setTitle(jpg)
        self.current_filename = file
        self.current_filename_image = jpg



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
    