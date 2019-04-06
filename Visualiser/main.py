import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
import time
import numpy as np
import csv

x_dir = 1 # Leap x-axis
y_dir = -1 # Leap z-axis
z_dir = 1 # Leap y-axis

t_1 = 0

def plotPalm(ax, line, x, y, z):
    start = 2
    ax.scatter(x_dir * float(line[start + x]), y_dir * float(line[start + y]), z_dir * float(line[start + z]))

def plotThumb(ax, line, x, y, z):
    start = 5
    ax.scatter(x_dir * float(line[start + x]), y_dir * float(line[start + y]), z_dir * float(line[start + z]))
    ax.plot([x_dir * float(line[2]), x_dir * float(line[start + x])], [y_dir * float(line[4]), y_dir * float(line[start + y])], [z_dir * float(line[3]), z_dir * float(line[start + z])])

def plotIndex(ax, line, x, y, z):
    start = 8
    ax.scatter(x_dir * float(line[start + x]), y_dir * float(line[start + y]), z_dir * float(line[start + z]))
    ax.plot([x_dir * float(line[2]), x_dir * float(line[start + x])], [y_dir * float(line[4]), y_dir * float(line[start + y])], [z_dir * float(line[3]), z_dir * float(line[start + z])])

def plotMiddle(ax, line, x, y, z):
    start = 11
    ax.scatter(x_dir * float(line[start + x]), y_dir * float(line[start + y]), z_dir * float(line[start + z]))
    ax.plot([x_dir * float(line[2]), x_dir * float(line[start + x])], [y_dir * float(line[4]), y_dir * float(line[start + y])], [z_dir * float(line[3]), z_dir * float(line[start + z])])

def plotRing(ax, line, x, y, z):
    start = 14
    ax.scatter(x_dir * float(line[start + x]), y_dir * float(line[start + y]), z_dir * float(line[start + z]))
    ax.plot([x_dir * float(line[2]), x_dir * float(line[start + x])], [y_dir * float(line[4]), y_dir * float(line[start + y])], [z_dir * float(line[3]), z_dir * float(line[start + z])])

def plotPinky(ax, line, x, y, z):
    start = 17
    ax.scatter(x_dir * float(line[start + x]), y_dir * float(line[start + y]), z_dir * float(line[start + z]))
    ax.plot([x_dir * float(line[2]), x_dir * float(line[start + x])], [y_dir * float(line[4]), y_dir * float(line[start + y])], [z_dir * float(line[3]), z_dir * float(line[start + z])])

def plotHand(ax, line, x=0, y=1, z=2):
    plotPalm(ax, line, x, y, z)
    plotThumb(ax, line, x, y, z)
    plotIndex(ax, line, x, y, z)
    plotMiddle(ax, line, x, y, z)
    plotRing(ax, line, x, y, z)
    plotPinky(ax, line, x, y, z)

def handle_figure_close(evt):
    sys.exit(0)

def main():
    fig = plt.figure()
    fig.canvas.mpl_connect('close_event', handle_figure_close)
    plt.ion()
    ax = Axes3D(fig)

    t_1 = time.time()

    try:
        with open("testData.csv", "r") as file:
            reader = csv.reader(file)
            for i, line in enumerate(reader):
                if (i != 0): # Ignore headings
                    ax.clear()
                    ax.set_xlim3d(-300, 300) # Leap x-axis
                    ax.set_ylim3d(-300, 300) # Leap z-axis
                    ax.set_zlim3d(0, 300)    # Leap y-axis
                    ax.set_xlabel("x-axis (left/right)")
                    ax.set_ylabel("z-axis (fwd/back)")
                    ax.set_zlabel("y-axit (up/down)")
                    plt.autoscale(False)

                    plotHand(ax, line, 0, 2, 1) # Swapping y- and z-axes to convert Leap axes to matplotlib axes
                    print("Interval: " + str(time.time() - t_1))
                    t_1 = time.time()
                    plt.pause(0.001)
    except KeyboardInterrupt:
        pass
    finally:
        file.close()

if __name__ == "__main__":
    main()