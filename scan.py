"""

Reads a coupon from the webcam and stores it in a local buffer that's
eventually written to the file system.

"""

import math
import os
import time

import barcode
import cv2
import imutils
import numpy as np
import pytesseract

from imutils.object_detection import non_max_suppression
from pyzbar import pyzbar


# Path to EAST text detector.
EAST_PATH = "frozen_east_text_detection.pb"

# Config for Tesseract.
CONFIG = "-l eng --oem 1 --psm 7"


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
    :return: (np.ndarray) bounding box for barcode
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
        return None
    barcode_contour = sorted(contours, key=cv2.contourArea, reverse=True)[0]
    if cv2.contourArea(barcode_contour) < 4000:
        return None

    # Compute rotated bounding box and add to the frame.
    rect = cv2.minAreaRect(barcode_contour)
    box = cv2.cv.BoxPoints(rect) if imutils.is_cv2() else cv2.boxPoints(rect)
    box = np.int0(box)
    cv2.drawContours(frame, [box], -1, (0, 255, 0), 3)
    return box


def find_text_bounding_boxes(scores, geometry):
    """ Finds boxes that bound the text in a provided frame.

    :param scores:
    :param geometry:
    :return: (list) np.ndarray of bounding boxes
    """
    # Initialize bounding box rectangles.
    n_rows, n_cols = scores.shape[2:4]
    rects, confidences = [], []

    # Go over each row and extract probabilities with geometric data.
    for i in range(n_rows):
        # Fetch geometry and scores.
        scores_data = scores[0, 0, i]
        x0 = geometry[0, 0, i]
        x1 = geometry[0, 1, i]
        x2 = geometry[0, 2, i]
        x3 = geometry[0, 3, i]
        angles = geometry[0, 4, i]

        # Go through columns.
        for j in range(n_cols):
            if scores_data[j] < 0.5:
                continue

            # Find offset and rotation angle.
            delta_x, delta_y = 4.0*j, 4.0*i
            angle = angles[j]
            cos = np.cos(angle)
            sin = np.sin(angle)

            # Create the bounding box.
            h, w = x0[j] + x2[j], x1[j] + x3[j]
            end_x = int(delta_x + (cos*x1[j]) + (sin*x2[j]))
            end_y = int(delta_y - (sin*x1[j]) + (cos*x2[j]))
            start_x = int(end_x-w)
            start_y = int(end_y-h)

            # Add in the box.
            rects.append((start_x, start_y, end_x, end_y))
            confidences.append(scores_data[j])

    # Suppress weak overlaps and return.
    return non_max_suppression(np.array(rects), probs=confidences)


def find_barcode_text(frame, box):
    """ Given a barcode box, finds the region of text right above it.
    
    :param frame: (np.ndarray) The overall frame to detect text in
    :param box: (np.ndarray) Bounding box
    :return: (np.ndarray) Region of text
    """
    # Load layer names and open EAST.
    layerNames = ["feature_fusion/Conv_7/Sigmoid",
                  "feature_fusion/concat_3"]
    east = cv2.dnn.readNet(EAST_PATH)

    # Make blob from image then forward pass in model.
    w, h = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (w, h), (123.68, 116.78, 103.94),
                                 swapRB=True, crop=False)
    east.setInput(blob)
    scores, geometry = east.forward(layerNames)
    boxes = find_text_bounding_boxes(scores, geometry)

    # Loop over the bounding boxes and find which is closest.
    x0, y0, xf, yf = 0, 0, 0, 0
    distance = float("inf")
    for start_x, start_y, end_x, end_y in boxes:
        norm = math.sqrt((box[0][0]-start_x)**2+(box[0][1]-start_y)**2)
        if norm < distance:
            x0, y0, xf, yf = start_x, start_y, end_x, end_y

    # Display the closest box and show the text.
    cv2.rectangle(frame, (x0, y0), (xf, yf), (0, 255, 0), 2)
    rect = frame[y0:yf, x0:xf]
    text = pytesseract.image_to_string(rect, config=CONFIG)
    return text


def main():
    """ Main loop for reading the webcam.

    :return: None
    """
    # Loop through the webcam infinitely until we stop.
    cam = get_webcam()
    while True:
        # Read a frame and resize it.
        ret, frame = cam.read()

        # h, w = frame.shape[:2]
        # h, w = h // 2, w // 2
        # h, w = h - (h % 32), w - (w % 32)
        # frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
        # frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)

        # Try finding the barcode and it corresponding text, then show the frame.
        box = find_barcode(frame)
        # if box is not None:
        #     min_x, min_y, max_x, max_y = float("inf"), float("inf"), float("-inf"), float("-inf")
        #     for point in box:
        #         if point[0] < min_x:
        #             min_x = point[0]
        #         if point[0] > max_x:
        #             max_x = point[0]
        #         if point[1] < min_y:
        #             min_y = point[1]
        #         if point[1] > max_y:
        #             max_y = point[1]
        #     min_x = max(0, min_x-50)
        #     min_y = max(0, min_y-50)
        #     max_x = min(w, max_x+50)
        #     max_y = min(h, max_y+50)
        #     print(min_y, max_y, min_x, max_x)
        #     clipped = frame[min_y:max_y, min_x:max_x]
        #     barcodes = pyzbar.decode(clipped)
        #     print(barcodes)
        #     #find_barcode_text(frame, box)
        #     pass
        cv2.imshow("SWP Scanner", frame)

        # Exit if we press escape.
        c = cv2.waitKey(1)
        if c == Keys.ESC:
            break
        elif c == Keys.SPACE:
            barcodes = pyzbar.decode(frame)
            if barcodes:
                print(str(barcodes[0].data, encoding="utf-8"))
            time.sleep(1)

    # Get rid of the camera and close everything, as well as delete temporary files.
    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
