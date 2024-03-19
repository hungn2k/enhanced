
import numpy as np
import copy


def atmospheric_scattering_model(origin_image, transmission, shadow, delta=0.85, epsilon=0.1):
    '''
    :param origin_image: Shadow input image
    :param transmission: estimated transmission
    :param shadow: estimated Shadow
    :param delta: fineTuning parameter for shadow --> default = 0.85
    :return: result --> Shadow removed image
    '''

    # This function will implement equation(3) in the paper
    # " https://www.cv-foundation.org/openaccess/content_iccv_2013/papers/Meng_Efficient_Image_Dehazing_2013_ICCV_paper.pdf "

    Transmission = pow(np.maximum(abs(transmission), epsilon), delta)

    shadow_corrected_image = copy.deepcopy(origin_image)
    if len(origin_image.shape) == 3:
        for ch in range(len(origin_image.shape)):
            temp = ((origin_image[:, :, ch].astype(float) - shadow[ch]) / Transmission) + shadow[ch]
            temp = np.maximum(np.minimum(temp, 255), 0)
            shadow_corrected_image[:, :, ch] = temp
    else:
        temp = ((origin_image.astype(float) - shadow[0]) / Transmission) + shadow[0]
        temp = np.maximum(np.minimum(temp, 255), 0)
        shadow_corrected_image = temp
    return shadow_corrected_image
