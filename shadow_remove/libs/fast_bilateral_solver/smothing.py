
import numpy as np

from libs.fast_bilateral_solver.FastBilateralSolver import BilateralGrid
from libs.fast_bilateral_solver.utils \
    import solve, domain_transform


def smoothing(
        ref_img,
        # Bilateral solver parameters
        lambd, sigma_xy, sigma_l=None, sigma_rgb=None,
        # Domain transform parameters
        sigma_s=None, sigma_r=None, dt_it=None
):
    # Choose the right set of standard deviations
    if ref_img.shape[2] == 1:
        sigma = np.array([sigma_xy, sigma_xy, sigma_l])
    elif ref_img.shape[2] == 3:
        sigma = np.array([sigma_xy, sigma_xy, sigma_rgb, sigma_rgb, sigma_rgb])

    # Create the bilateral grid
    bilateral_grid = BilateralGrid(ref_img, sigma)
    S, B = bilateral_grid.S, bilateral_grid.B

    # The target image is the same as the reference image
    target_img = ref_img
    new_img = np.empty_like(target_img)

    # Perform smoothing channel by channel
    for channel in range(target_img.shape[2]):
        T = target_img[:, :, channel].flatten()
        # The confidence in each pixel of the target is the same
        C = np.ones_like(T)

        # Compute bilateral space solution
        y = solve(bilateral_grid, C, T, lambd, channel=channel)
        # Go back to pixel space
        x = S.T.dot(y).reshape(ref_img.shape[:2])

        # Fill the corresponding channel of the new image
        new_img[:, :, channel] = x

    # Apply domain transform if needed
    if sigma_s is not None and sigma_r is not None:
        new_img = domain_transform(
            I0=new_img, I_ref=new_img,
            sigma_s=sigma_s, sigma_r=sigma_r, N_it=dt_it,
        )

    return new_img