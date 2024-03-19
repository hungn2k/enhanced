import numpy as np
import cv2


def contextual_regularization(origin_image, transmission, regularize_lambda, sigma):
    rows, cols = transmission.shape
    kirsch_filters = load_filter_bank()

    # Normalize the filters
    for idx, currentFilter in enumerate(kirsch_filters):
        kirsch_filters[idx] = kirsch_filters[idx] / np.linalg.norm(currentFilter)
    # Calculate Weighting function --> [rows, cols. numFilters] --> One Weighting function for every filter
    weight_func = []
    for idx, currentFilter in enumerate(kirsch_filters):
        weight_func.append(calculate_weighting_function(origin_image, currentFilter, sigma))
    # Precompute the constants that are later needed in the optimization step
    tF = np.fft.fft2(transmission)
    DS = 0

    for i in range(len(kirsch_filters)):
        D = psf2otf(kirsch_filters[i], (rows, cols))
        DS = DS + (abs(D) ** 2)

    # Cyclic loop for refining t and u --> Section III in the paper
    beta = 1  # Start Beta value --> selected from the paper
    beta_max = 2 ** 8  # Selected from the paper --> Section III --> "Scene Transmission Estimation"
    beta_rate = 2 * np.sqrt(2)  # Selected from the paper

    while beta < beta_max:
        gamma = regularize_lambda / beta

        # Fixing t first and solving for u
        DU = 0
        for i in range(len(kirsch_filters)):
            dt = circularConvFilt(transmission, kirsch_filters[i])
            u = np.maximum((abs(dt) - (weight_func[i] / (len(kirsch_filters) * beta))), 0) * np.sign(dt)
            DU = DU + np.fft.fft2(circularConvFilt(u, cv2.flip(kirsch_filters[i], -1)))

        # Fixing u and solving t --> Equation 26 in the paper
        # Note: In equation 26, the Numerator is the "DU" calculated in the above part of the code
        # In the equation 26, the Denominator is the DS which was computed as a constant in the above code

        transmission = np.abs(np.fft.ifft2((gamma * tF + DU) / (gamma + DS)))
        beta = beta * beta_rate
    return transmission


def load_filter_bank():
    kirsch_filters = []
    kirsch_filters.append(np.array([[-3, -3, -3], [-3, 0, 5], [-3, 5, 5]]))
    kirsch_filters.append(np.array([[-3, -3, -3], [-3, 0, -3], [5, 5, 5]]))
    kirsch_filters.append(np.array([[-3, -3, -3], [5, 0, -3], [5, 5, -3]]))
    kirsch_filters.append(np.array([[5, -3, -3], [5, 0, -3], [5, -3, -3]]))
    kirsch_filters.append(np.array([[5, 5, -3], [5, 0, -3], [-3, -3, -3]]))
    kirsch_filters.append(np.array([[5, 5, 5], [-3, 0, -3], [-3, -3, -3]]))
    kirsch_filters.append(np.array([[-3, 5, 5], [-3, 0, 5], [-3, -3, -3]]))
    kirsch_filters.append(np.array([[-3, -3, 5], [-3, 0, 5], [-3, -3, 5]]))
    kirsch_filters.append(np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]))
    return kirsch_filters


def calculate_weighting_function(origin_image, filters, sigma):
    # Computing the weight function... Eq (17) in the paper

    shadow_image_double = origin_image.astype(float) / 255.0
    if len(origin_image.shape) == 3:
        red = shadow_image_double[:, :, 2]
        d_r = circularConvFilt(red, filters)

        green = shadow_image_double[:, :, 1]
        d_g = circularConvFilt(green, filters)

        blue = shadow_image_double[:, :, 0]
        d_b = circularConvFilt(blue, filters)

        weight_func = np.exp(-((d_r ** 2) + (d_g ** 2) + (d_b ** 2)) / (2 * sigma * sigma))
    else:
        d = circularConvFilt(shadow_image_double, filters)
        weight_func = np.exp(-((d ** 2) + (d ** 2) + (d ** 2)) / (2 * sigma * sigma))
    return weight_func


def circularConvFilt(image, filters):
    filter_height, filter_width = filters.shape
    assert (filter_height == filter_width), 'Filter must be square in shape --> Height must be same as width'
    assert (filter_height % 2 == 1), 'Filter dimension must be a odd number.'

    filter_hals_size = int((filter_height - 1) / 2)
    rows, cols = image.shape
    padded_image = cv2.copyMakeBorder(image, filter_hals_size, filter_hals_size, filter_hals_size, filter_hals_size,
                                      borderType=cv2.BORDER_WRAP)
    filtered_image = cv2.filter2D(padded_image, -1, filters)
    result = filtered_image[filter_hals_size:rows + filter_hals_size, filter_hals_size:cols + filter_hals_size]

    return result


##################
def psf2otf(psf, shape):
    """
    Convert point-spread function to optical transfer function.
    Compute the Fast Fourier Transform (FFT) of the point-spread
    function (PSF) array and creates the optical transfer function (OTF)
    array that is not influenced by the PSF off-centering.
    By default, the OTF array is the same size as the PSF array.
    To ensure that the OTF is not altered due to PSF off-centering, PSF2OTF
    post-pads the PSF array (down or to the right) with zeros to match
    dimensions specified in OUTSIZE, then circularly shifts the values of
    the PSF array up (or to the left) until the central pixel reaches (1,1)
    position.
    Parameters
    ----------
    psf : `numpy.ndarray`
        PSF array
    shape : int
        Output shape of the OTF array
    Returns
    -------
    otf : `numpy.ndarray`
        OTF array
    Notes
    -----
    Adapted from MATLAB psf2otf function
    """
    if np.all(psf == 0):
        return np.zeros_like(psf)

    inshape = psf.shape
    # Pad the PSF to outsize
    psf = zero_pad(psf, shape, position='corner')

    # Circularly shift OTF so that the 'center' of the PSF is
    # [0,0] element of the array
    for axis, axis_size in enumerate(inshape):
        psf = np.roll(psf, -int(axis_size / 2), axis=axis)

    # Compute the OTF
    otf = np.fft.fft2(psf)

    # Estimate the rough number of operations involved in the FFT
    # and discard the PSF imaginary part if within roundoff error
    # roundoff error  = machine epsilon = sys.float_info.epsilon
    # or np.finfo().eps
    n_ops = np.sum(psf.size * np.log2(psf.shape))
    otf = np.real_if_close(otf, tol=n_ops)

    return otf


def zero_pad(image, shape, position='corner'):
    """
    Extends image to a certain size with zeros
    Parameters
    ----------
    image: real 2d `numpy.ndarray`
        Input image
    shape: tuple of int
        Desired output shape of the image
    position : str, optional
        The position of the input image in the output one:
            * 'corner'
                top-left corner (default)
            * 'center'
                centered
    Returns
    -------
    padded_img: real `numpy.ndarray`
        The zero-padded image
    """
    shape = np.asarray(shape, dtype=int)
    imshape = np.asarray(image.shape, dtype=int)

    if np.alltrue(imshape == shape):
        return image

    if np.any(shape <= 0):
        raise ValueError("ZERO_PAD: null or negative shape given")

    dshape = shape - imshape
    if np.any(dshape < 0):
        raise ValueError("ZERO_PAD: target size smaller than source one")

    pad_img = np.zeros(shape, dtype=image.dtype)

    idx, idy = np.indices(imshape)

    if position == 'center':
        if np.any(dshape % 2 != 0):
            raise ValueError("ZERO_PAD: source and target shapes "
                             "have different parity.")
        offx, offy = dshape // 2
    else:
        offx, offy = (0, 0)

    pad_img[idx + offx, idy + offy] = image

    return pad_img
