import os, sys, inspect, time
import Leap
import cv2, matplotlib.pyplot as plt
import numpy as np
import ctypes
import csv


class listenForFrame(Leap.Listener):

    def on_connect(self, controller):
        # Allow Images to be passed through to python
        controller.set_policy(controller.POLICY_IMAGES)

        # Remove previous images
        os.system("rm -rf " + os.path.join(os.getcwd(), "Images").replace(" ", "\ "))
        os.mkdir(os.path.join(os.getcwd(), "Images"))

        # Write the required headings to the CSV file
        headings = ['FrameID', 'Right', 'PalmX', 'PalmY', 'PalmZ', 'ThumbX', 'ThumbY', 'ThumbZ', 'IndexX', 'IndexY', 'IndexZ', 'MiddleX', 'MiddleY', 'MiddleZ', 'RingX', 'RingY', 'RingZ', 'PinkyX', 'PinkyY', 'PinkyZ']
        with open('testData.csv', 'w') as file:
            writer = csv.writer(file)
            writer.writerow(headings)
        file.close()

        # Ready to capture data
        print("Connected")

    def on_frame(self, controller):
        # Get new frame
        frame = controller.frame()
        #print("FPS: " + str(frame.current_frames_per_second))

        # only act on frames which contain a hand
        if not frame.hands.is_empty:
            hand = frame.hands[0]
            fingers = hand.fingers
            palm = hand.palm_position
            thumb = fingers.finger_type(Leap.Finger.TYPE_THUMB)[0].bone(3).next_joint
            index = fingers.finger_type(Leap.Finger.TYPE_INDEX)[0].bone(3).next_joint
            middle = fingers.finger_type(Leap.Finger.TYPE_MIDDLE)[0].bone(3).next_joint
            ring = fingers.finger_type(Leap.Finger.TYPE_RING)[0].bone(3).next_joint
            pinky = fingers.finger_type(Leap.Finger.TYPE_PINKY)[0].bone(3).next_joint

            # Save tracking data to CSV
            entry = [frame.id,  hand.is_right,
                     palm[0],   palm[1],    palm[2],
                     thumb[0], thumb[1], thumb[2],
                     index[0],  index[1],   index[2],
                     middle[0], middle[1],  middle[2],
                     ring[0],   ring[1],    ring[2],
                     pinky[0],  pinky[1],   pinky[2]]
            with open('testData.csv', 'a') as file:
                writer = csv.writer(file)
                writer.writerow(entry)
            file.close()

            # Save both images to file as JPEG
            images = frame.images
            for image in images:
                # Could maybe streamline this process
                ptr = image.data_pointer
                arr_def = ctypes.c_ubyte * image.width * image.height
                c_arr = arr_def.from_address(int(ptr))
                np_arr = np.ctypeslib.as_array(c_arr)
                cv2.imwrite(os.path.join(os.getcwd(), "Images/" + str(frame.id) + "_" + str(image.id) + '.jpg'), np_arr)