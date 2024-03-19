import cv2
import numpy as np


def shadow(origin_image, window_size):
    S = []
    if len(origin_image.shape) == 3:
        for ch in range(len(origin_image.shape)):
            kernel = np.ones((window_size, window_size), np.uint8)
            min_image = cv2.erode(origin_image[:, :, ch], kernel)
            S.append(int(min_image.min()))
    else:
        kernel = np.ones((window_size, window_size), np.uint8)
        min_image = cv2.erode(origin_image, kernel)
        S.append(int(min_image.min()))
    return S
