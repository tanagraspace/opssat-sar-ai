#!/usr/bin/env python3

__maintainer__ = "Tom Mladenov"

import argparse
from sys import byteorder
import numpy as np
import sys

from PyQt5 import QtWidgets, QtCore, uic, QtGui
from pyqtgraph import PlotWidget, plot
import pyqtgraph as pg
import glob, os
import subprocess
import imageio


class MainWindow(QtWidgets.QMainWindow):

    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)

        uic.loadUi('/tools/labelling/gui.ui', self)
        self.current_index = 0
        self.data = []
        self.selected_beacons = []

        #self.imv = pg.ImageView()
        print('ok')

        self.p1 = self.win.addPlot(title="")
        self.img = pg.ImageItem(axisOrder='row-major')
        self.img.hoverEvent = self.imageHoverEvent
        #self.img.mouseClickEvent = self.mouseClickEvent
        self.p1.addItem(self.img)

        self.selector_roi = pg.ROI([0, 0], [22, 66], pen='r')
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
        if h == 66:
            #current ROI selector is for a short beacon, make it a long one
            self.selector_roi.setSize((22, 77))
        else:
            #current selector ROI is for a long beacon, make it a short one
            self.selector_roi.setSize((22, 66))

    # function that updates the zoomed in view of currently selected beacon when mouse moves
    def update_zoom(self, roi):
        self.zoom.setImage(self.selector_roi.getArrayRegion(self.data, self.zoom))
 
    def set_file_list(self, file_list):
        self.file_list = file_list

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
        for b in self.selected_beacons:
            self.p1.removeItem(b) 

    # triggered when clicking, will add a rectangle where the ROI is currently located
    def tag_beacon(self, event):
        x, y = self.selector_roi.pos()
        w, h = self.selector_roi.size()
        r1 = pg.QtGui.QGraphicsRectItem(x, y, w, h)
        #r1.setPen(pg.mkPen(None))
        r1.setPen(pg.mkPen('g'))
        self.p1.addItem(r1)
        self.selected_beacons.append(r1)



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
        self.p1.setTitle("pos: (%0.2f, %0.2f)  pixel: (%d, %d)" % (y, x, j, i))
        self.selector_roi.setPos(x - 11, y - 23)

    # function to move to next raw file
    def next(self):
        self.current_index += 1
        try:
            print("laoding image: {}".format(self.file_list[self.current_index]))
            self.show_image(self.file_list[self.current_index])
        except Exception as e:
            print(e)

    # function to move to previous raw file (will overwrite output)
    def previous(self):
        self.current_index -= 1
        try:
            self.show_image(self.file_list[self.current_index])
        except Exception as e:
            print(e)

    # function that is triggering rendering of the FFT
    def show_image(self, file):
        os.system("renderfall -n 512 -v -f float32 -l 256 -w hann {}".format(file))
        png = file + ".png"
        self.data = imageio.imread(png)
        self.img.setImage(self.data)
        self.p1.setTitle(png)



def main():

    cliParser = argparse.ArgumentParser(description='Plots quadrature IQ signals')    
    
    cliParser.add_argument('input_dir', type=str, help='input directory containing the raw .cf32 files')
    cliParser.add_argument('output_csv', type=str, help='output csv file containing raw filenames and labelled data')
    args = cliParser.parse_args()

    app = QtWidgets.QApplication(sys.argv)
    main = MainWindow()

    os.chdir(args.input_dir)
    main.set_file_list(sorted(glob.glob("*.cf32"), key=os.path.getmtime))
    main.show()

    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
    