"""

Reads a coupon from the webcam and stores it in a local buffer that's
eventually written to the file system.

"""

import os

import cv2
import imutils
import numpy as np
from pyzbar import pyzbar


# Simple enum for containing key presses.
class Keys:
    ESC = 27
    SPACE = 32


def get_webcam():
    """ Opens the user's webcam.

    :return: (VideoCapture) The camera object
    """
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise IOError("Cannot open webcam!")
    return cap


def find_barcode(frame):
    """ Tries finding a barcode in a provided cv2 frame.

    :param frame: (np.ndarray) RGB data for the frame.
    :return: TODO
    """
    # Convert to grayscale.
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Find Scharr gradient magnitude representation in x and y directions.
    ddepth = cv2.cv.CV_32F if imutils.is_cv2() else cv2.CV_32F
    grad_x = cv2.Sobel(gray, ddepth=ddepth, dx=1, dy=0, ksize=-1)
    grad_y = cv2.Sobel(gray, ddepth=ddepth, dx=0, dy=1, ksize=-1)

    # Subtract y gradient from x gradient.
    grad = cv2.subtract(grad_x, grad_y)
    grad = cv2.convertScaleAbs(grad)

    # Blur and threshold the image.
    blur = cv2.blur(grad, (9, 9))
    _, thresh = cv2.threshold(blur, 225, 225, cv2.THRESH_BINARY)

    # Construct a closing kernel and apply to threshold.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    closed = cv2.erode(closed, None, iterations=4)
    closed = cv2.dilate(closed, None, iterations=4)

    # Find contours in threshold, then sort and keep largest.
    # Terminate early if contour too small or there are none.
    contours = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    if len(contours) == 0:
        return
    barcode_contour = sorted(contours, key=cv2.contourArea, reverse=True)[0]
    if cv2.contourArea(barcode_contour) < 5000:
        return

    # Compute rotated bounding box.
    rect = cv2.minAreaRect(barcode_contour)
    box = cv2.cv.BoxPoints(rect) if imutils.is_cv2() else cv2.boxPoints(rect)
    box = np.int0(box)

    cv2.drawContours(frame, [box], -1, (0, 255, 0), 3)


def main():
    """ Main loop for reading the webcam.

    :return: None
    """
    # Loop through the webcam infinitely until we stop.
    cam = get_webcam()
    while True:
        # Read a frame and resize it.
        ret, frame = cam.read()
        frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)

        # Show the frame and then try finding a barcode.
        #cv2.imshow("SWP Scanner", frame)
        find_barcode(frame)
        cv2.imshow("SWP Scanner", frame)

        #
        c = cv2.waitKey(1)
        if c == Keys.ESC:
            break

    # Get rid of the camera and close everything, as well as delete temporary files.
    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()