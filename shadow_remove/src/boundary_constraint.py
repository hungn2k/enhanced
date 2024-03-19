import cv2
import numpy as np


def boundary_constraint(origin_image, shadow, C0, C1, window_size):
    if len(origin_image.shape) == 3:

        t_b = np.maximum((shadow[0] - origin_image[:, :, 0].astype(np.float)) / (shadow[0] - C0),
                         (shadow[0] - origin_image[:, :, 0].astype(np.float)) / (shadow[0] - C1))
        t_g = np.maximum((shadow[1] - origin_image[:, :, 1].astype(np.float)) / (shadow[1] - C0),
                         (shadow[1] - origin_image[:, :, 1].astype(np.float)) / (shadow[1] - C1))
        t_r = np.maximum((shadow[2] - origin_image[:, :, 2].astype(np.float)) / (shadow[2] - C0),
                         (shadow[2] - origin_image[:, :, 2].astype(np.float)) / (shadow[2] - C1))

        max_value = np.maximum(np.maximum(t_b, t_g), t_r)

        transmission = np.minimum(max_value, 1)
    else:
        transmission = np.maximum((shadow[0] - origin_image.astype(np.float)) / (shadow[0] - C0),
                                  (shadow[0] - origin_image.astype(np.float)) / (shadow[0] - C1))
        transmission = np.minimum(transmission, 1)

    kernel = np.ones((window_size, window_size), np.float)
    transmission = cv2.morphologyEx(transmission, cv2.MORPH_CLOSE, kernel=kernel)
    return transmission