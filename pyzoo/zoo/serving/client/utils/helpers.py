import numpy as np
import base64
import sys
import cv2
import time


def base64_encode_image(image_array):
    # base64 encode the input NumPy array
    return base64.b64encode(image_array).decode("utf-8")
