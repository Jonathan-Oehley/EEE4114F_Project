import os, sys, inspect, time
import Leap
import cv2, matplotlib.pyplot as plt
import numpy as np
import ctypes
import csv

class listenForFrame(Leap.Listener):

    def on_connect(self, controller):
        print("Connected")
        controller.set_policy(controller.POLICY_IMAGES)
        headings = ['FrameID', 'PalmX', 'PalmY', 'PalmZ']
        with open('testData.csv', 'w') as file:
            writer = csv.writer(file)
            writer.writerow(headings)
        file.close()

    def on_frame(self, controller):
        frame = controller.frame()
        print("FPS: " + str(frame.current_frames_per_second))
        hands = frame.hands
        if len(hands) != 0:
            entry = [frame.id, hands[0].palm_position[0], hands[0].palm_position[2], hands[0].palm_position[2]]
            with open('testData.csv', 'a') as file:
                writer = csv.writer(file)
                writer.writerow(entry)
            file.close()