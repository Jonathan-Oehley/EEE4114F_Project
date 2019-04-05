import os, sys, inspect, time
import Leap
import myListeners
import cv2, matplotlib.pyplot as plt

def main():
    listener = myListeners.listenForFrame()
    controller = Leap.Controller()

    controller.add_listener(listener)

    # Keep this process running until Enter is pressed
    print("Press Enter to quit...")
    try:
        sys.stdin.readline()
    except KeyboardInterrupt:
        pass
    finally:
        # Remove the sample listener when done
        controller.remove_listener(listener)

if __name__ == "__main__":
    main()