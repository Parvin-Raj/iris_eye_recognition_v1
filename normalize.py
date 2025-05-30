import cv2
import numpy as np

def normalize_iris(iris_img, output_size=(128, 128)):
    
    norm_img = cv2.resize(iris_img, output_size, interpolation=cv2.INTER_CUBIC)
    return norm_img
