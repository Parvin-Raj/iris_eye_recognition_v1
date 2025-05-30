import cv2
import numpy as np

def segment_iris(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
   
    blurred = cv2.medianBlur(gray, 5)

    
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=gray.shape[0]/8,
                               param1=200, param2=30, minRadius=30, maxRadius=80)
    if circles is None:
        raise Exception("No circles detected!")
    circles = np.uint16(np.around(circles))

    
    iris_circle = circles[0][0]  

    
    x, y, r = iris_circle
    x1, y1 = max(0, x - r), max(0, y - r)
    x2, y2 = min(image.shape[1], x + r), min(image.shape[0], y + r)
    iris_cropped = image[y1:y2, x1:x2]
    return iris_cropped
