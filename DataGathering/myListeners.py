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
        os.system("rm -rf " + os.path.join(os.getcwd(), "Images").replace(" ", "\ "))
        os.mkdir(os.path.join(os.getcwd(), "Images"))
        #headings = ['FrameID', 'PalmX', 'PalmY', 'PalmZ']
        headings = ['IMG_1', 'IMG_2']
        with open('testData.csv', 'w') as file:
            writer = csv.writer(file)
            writer.writerow(headings)
        file.close()

    def on_frame(self, controller):
        frame = controller.frame()
        print("FPS: " + str(frame.current_frames_per_second))
        # hands = frame.hands
        # if len(hands) != 0:
        #     entry = [frame.id, hands[0].palm_position[0], hands[0].palm_position[2], hands[0].palm_position[2]]
        #     with open('testData.csv', 'a') as file:
        #         writer = csv.writer(file)
        #         writer.writerow(entry)
        #     file.close()
        images = frame.images
        for image in images:
            ptr = image.data_pointer
            arr_def = ctypes.c_ubyte * image.width * image.height
            c_arr = arr_def.from_address(int(ptr))
            np_arr = np.ctypeslib.as_array(c_arr)
            # print(np_arr)
            #             # with open('testData.csv', 'a') as file:
            #             #     writer = csv.writer(file)
            #             #     writer.writerow(np_arr)
            #             # file.close()

            cv2.imwrite(os.path.join(os.getcwd(), "Images/" + str(frame.id) + "_" + str(image.id) + '.jpg'), np_arr)