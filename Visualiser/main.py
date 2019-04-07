import sys
import csv
import numpy as np
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from PyQt4 import QtGui, QtCore
import cv2

def getPos(line, x=0, y=1, z=2):
    pos = np.zeros([6, 3], dtype=np.float32)
    pos[0,:] = [x_dir * float(line[2 + x]), y_dir * float(line[2 + y]), z_dir * float(line[2 + z])] # Palm
    pos[1,:] = [x_dir * float(line[5 + x]), y_dir * float(line[5 + y]), z_dir * float(line[5 + z])] # Thumb
    pos[2, :] = [x_dir * float(line[8 + x]), y_dir * float(line[8 + y]), z_dir * float(line[8 + z])]  # Index
    pos[3, :] = [x_dir * float(line[11 + x]), y_dir * float(line[11 + y]), z_dir * float(line[11 + z])]  # Middle
    pos[4, :] = [x_dir * float(line[14 + x]), y_dir * float(line[14 + y]), z_dir * float(line[14 + z])]  # Ring
    pos[5, :] = [x_dir * float(line[17 + x]), y_dir * float(line[17 + y]), z_dir * float(line[17 + z])]  # Pinky
    return pos

def getLines(line, x=0, y=1, z=2):
    pos = getPos(line, x=x, y=y, z=z)
    lines = np.zeros([10, 3], dtype=np.float32)
    lines[0:11:2,:] = pos[0,:] # Palm base for each finger
    lines[1,:] = pos[1,:] # Thumb
    lines[3, :] = pos[2, :]  # Index
    lines[5, :] = pos[3, :]  # Middle
    lines[7, :] = pos[4, :]  # Ring
    lines[9, :] = pos[5, :]  # Pinky
    return lines

def update():
    global currentLine, points, currentLine
    if currentLine < len(dataFile):
        points.setData(pos=getPos(dataFile[currentLine], x=0, y=2, z=1))
        lines.setData(pos=getLines(dataFile[currentLine], x=0, y=2, z=1))
        leftImage.setImage(cv2.imread("../DataGathering/Images/" + str(dataFile[currentLine][0]) + "_0.jpg", cv2.IMREAD_GRAYSCALE).T)
        #rightImage.setImage(cv2.imread("testImage.jpg", cv2.IMREAD_GRAYSCALE).T)
        currentLine += 1
    else:
        print("Done")
        sys.exit(0)

headings = []
dataFile = []
currentLine = 0

# Read in all the data
with open("testData.csv", "r") as file:
    reader = csv.reader(file)
    for i, line in enumerate(reader):
        if (i == 0):
            headings.append(line)
        else:
            dataFile.append(line)
file.close()

x_dir = -1 # Leap x-axis
y_dir = 1 # Leap z-axis
z_dir = 1 # Leap y-axis

app = QtGui.QApplication([])
window = QtGui.QWidget()
window.showMaximized()

layout = QtGui.QGridLayout()
window.setLayout(layout)

viewer = gl.GLViewWidget()
viewer.setCameraPosition(distance=2000, azimuth=90)
viewer.setSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
layout.addWidget(viewer, 0, 0, 1, 2)

g = gl.GLGridItem()
g.scale(50, 50, 1)
viewer.addItem(g)

points = gl.GLScatterPlotItem(pos=np.zeros([6, 3], dtype=np.float32))
lines = gl.GLLinePlotItem(pos=np.zeros([10, 3], dtype=np.float32))
viewer.addItem(points)
viewer.addItem(lines)

leftImage = pg.ImageView()
leftImage.ui.histogram.hide()
leftImage.ui.roiBtn.hide()
leftImage.ui.menuBtn.hide()
layout.addWidget(leftImage, 2, 0)

rightImage = pg.ImageView()
rightImage.ui.histogram.hide()
rightImage.ui.roiBtn.hide()
rightImage.ui.menuBtn.hide()
layout.addWidget(rightImage, 2, 1)

timer = QtCore.QTimer()
timer.timeout.connect(update)
timer.start(10)


if __name__ == "__main__":
    app.exec_()