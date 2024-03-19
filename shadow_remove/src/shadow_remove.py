import cv2 as cv
import numpy as np
from src.boundary_constraint import boundary_constraint
from src.contextual_regularization import contextual_regularization
from src.atmospheric_scattering_model import atmospheric_scattering_model
from src.shadow import shadow
from libs.gf import guided_filter
from libs.fast_bilateral_solver.smothing import smoothing
from settings import SHADOW_REMOVE


class ShadowRemove:
    def __init__(self):
        self.window_size = SHADOW_REMOVE.get("WINDOW_SIZE")
        self.C0 = SHADOW_REMOVE.get("C0")
        self.C1 = SHADOW_REMOVE.get("C1")
        self.regularize_lambda = SHADOW_REMOVE.get("REGULARIZE_LAMBDA")
        self.sigma = SHADOW_REMOVE.get("SIGMA")
        self.delta = SHADOW_REMOVE.get("DELTA")
        self.epsilon = SHADOW_REMOVE.get("EPSILON")
        self.origin_image = None
        self.shadow_remove_image = None
        self.shadow = None
        self.trans = None
        self.Transmission = None

    def start(self, use_cont_regula=True, resize=None, filter_type="fbs"):
        if resize is not None:
            self.origin_image = cv.resize(self.origin_image, resize)
        self.shadow = shadow(self.origin_image, self.window_size)
        self.trans = boundary_constraint(self.origin_image, self.shadow, self.C0, self.C1, self.window_size)
        if use_cont_regula:
            if filter_type == "cr":
                self.Transmission = contextual_regularization(self.origin_image, self.trans, self.regularize_lambda,
                                                              self.sigma)
            elif filter_type == "guided_filter":
                self.Transmission = guided_filter(self.trans, self.trans, r=SHADOW_REMOVE.get("FILTER_R"),
                                                  eps=SHADOW_REMOVE.get("FILTER_EPS"))
            elif filter_type == "fbs":
                ref_img = np.array(list(map(lambda i: list(map(lambda j: [j], i)), self.trans)))
                new_trans = smoothing(
                    ref_img=ref_img,
                    lambd=SHADOW_REMOVE.get("FILTER_LAMBDA"), sigma_xy=SHADOW_REMOVE.get("FILTER_SIGMA_XY"),
                    sigma_l=SHADOW_REMOVE.get("FILTER_SIGMA_L"),
                    sigma_s=None, sigma_r=None
                )
                self.Transmission = new_trans.reshape(new_trans.shape[:2])

    def run(self):
        if self.Transmission is not None:
            self.shadow_remove_image = atmospheric_scattering_model(
                self.origin_image, self.Transmission, self.shadow, self.delta, self.epsilon)
        else:
            self.shadow_remove_image = atmospheric_scattering_model(
                self.origin_image, self.trans, self.shadow, self.delta, self.epsilon)

    def show(self):
        cv.imshow("origin image", self.origin_image)
        if self.shadow_remove_image is not None:
            cv.imshow("shadow remove image", self.shadow_remove_image)
            cv.waitKey(0)
        else:
            print("shadow remove image is None.")

    def write(self, name):
        if self.shadow_remove_image is not None:
            cv.imwrite(f"{name}", self.shadow_remove_image)
        else:
            print("shadow remove image is None.")

    def write_trans(self, name):
        if self.trans is not None:
            cv.imwrite(f"{name}-bc.jpg", self.trans * 255)
        if self.Transmission is not None:
            cv.imwrite(f"{name}-bc-cr.jpg", self.Transmission * 255)
