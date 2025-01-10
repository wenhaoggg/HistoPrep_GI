from typing import Optional, Union

import cv2
import numpy as np
from PIL import Image
from typing import Union, Tuple
from skimage.filters import threshold_otsu # pylint: disable=no-name-in-module
from scipy.signal import convolve2d
from ._check import check_image

ERROR_THRESHOLD = "Threshold should be in range [0, 255], got {}."
ERROR_BLOOD = "Blood filter should be a boolean, got {}."

MAX_THRESHOLD = 255
WHITE_PIXEL = 255
BLACK_PIXEL = 0
SIGMA_NO_OP = 0.0
GRAY_NDIM = 2


def get_tissue_mask(
    image: Union[Image.Image, np.ndarray],
    *,
    threshold: Union[int, str] = "H&E_Otsu", #None
    blood: bool = True,
    multiplier: float = 1.0,
    sigma: float = 1.0,
) -> np.ndarray: #tuple[int, np.ndarray]:
    """Detect tissue from image.

    Args:
        image: Input image.
        threshold: "H&E_Otsu" or not. Threshold for tissue detection. If set, will detect tissue by
            global thresholding, and otherwise Otsu's method is used to find
            a threshold. Defaults to None.
        blood: Blood Filter or not. Defaults to True.
        multiplier: Otsu's method is used to find an optimal threshold by
            minimizing the weighted within-class variance. This threshold is
            then multiplied with `multiplier`. Ignored if `threshold` is not None.
            Defaults to 1.0.
        sigma: Sigma for gaussian blurring. Defaults to 1.0.

    Raises:
        ValueError: Threshold not between 0 and 255.

    Returns:
        np.ndarray `tissue_mask` (0=background and 1=tissue).
        #Tuple with `threshold` and `tissue_mask` (0=background and 1=tissue).
    """
    # Check image and convert to array.
    image = check_image(image)
    # Check arguments.
    if threshold is not None and (not isinstance(threshold, str) and not 0 <= threshold <= MAX_THRESHOLD) and threshold != "H&E_Otsu":
        raise ValueError(ERROR_THRESHOLD.format(threshold))
    if not isinstance(blood, bool):
        raise ValueError(ERROR_BLOOD.format(blood))
    # Convert to grayscale.
    gray = image if image.ndim == GRAY_NDIM else cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # Gaussian blurring.
    blur = _gaussian_blur(image=gray, sigma=sigma, truncate=3.5)
    # Get threshold.
    if threshold == "H&E_Otsu":
        mask = _h_and_e_otsu_threshold(image=image, nbins=256, hist=None)   
    else:
        if threshold is None:
            threshold = _otsu_threshold(gray=blur)
            threshold = max(min(255, int(threshold * max(0.0, multiplier) + 0.5)), 0)
        # Global thresholding.
        thrsh, mask = cv2.threshold(blur, threshold, 1, cv2.THRESH_BINARY_INV)

    #Blood filter
    if blood:
        blood_mask = _blood_mask(image=image)
        # Filter blood from tissue mask, to get "TRUE" tissue mask
        mask = np.where(blood_mask==1, 0, mask)

    return mask #int(thrsh), mask


def clean_tissue_mask(
    tissue_mask: np.ndarray,
    min_area_pixel: int = 10,
    max_area_pixel: Optional[int] = None,
    min_area_relative: float = 0.2,
    max_area_relative: Optional[float] = 2.0,
) -> np.ndarray:
    """Remove too small/large contours from tissue mask.

    Args:
        tissue_mask: Tissue mask to be cleaned.
        min_area_pixel: Minimum pixel area for contours. Defaults to 10.
        max_area_pixel: Maximum pixel area for contours. Defaults to None.
        min_area_relative: Relative minimum contour area, calculated from the median
            contour area after filtering contours with `[min,max]_pixel` arguments
            (`min_area_relative * median(contour_areas)`). Defaults to 0.2.
        max_area_relative: Relative maximum contour area, calculated from the median
            contour area after filtering contours with `[min,max]_pixel` arguments
            (`max_area_relative * median(contour_areas)`). Defaults to 2.0.

    Returns:
        Tissue mask with too small/large contours removed.
    """
    contours, __ = cv2.findContours(
        tissue_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
    )
    if len(contours) == 0:
        return tissue_mask
    contour_areas = np.array([cv2.contourArea(cnt) for cnt in contours])
    # Filter based on pixel values.
    selection = contour_areas >= min_area_pixel
    if max_area_pixel is not None:
        selection = selection & (contour_areas <= max_area_pixel)
    if selection.sum() == 0:
        # Nothing to draw
        return np.zeros_like(tissue_mask)
    # Define relative min/max values.
    area_median = np.median(contour_areas[selection])
    area_min = area_median * min_area_relative
    area_max = None if max_area_relative is None else area_median * max_area_relative
    # Draw new mask.
    new_mask = np.zeros_like(tissue_mask)
    for select, area, cnt in zip(selection, contour_areas, contours):
        if select and area >= area_min and (area_max is None or area <= area_max):
            cv2.drawContours(new_mask, [cnt], -1, 1, -1)
    return new_mask


def _otsu_threshold(*, gray: np.ndarray) -> int:
    """Helper function to calculate Otsu's thresold from a grayscale image."""
    values = gray.flatten()
    values = values[(values != WHITE_PIXEL) & (values != BLACK_PIXEL)]
    threshold, __ = cv2.threshold(
        values, None, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    return threshold

def _h_and_e_otsu_threshold(
    *,
    image: np.ndarray,
    nbins: int = 256,
    hist: Union[np.ndarray, Tuple[np.ndarray, np.ndarray]] = None
) -> np.ndarray:
    """Calculates the H&E Otsu threshold.

    Args:
        image (np.ndarray): An RGB image
        #blur_kernel_width (int): Blurs the mask with the given blur kernel
        nbins (int, optional): Number of bins used to calculate histogram.
                            This value is ignored for integer arrays.
                            Defaults to 256.
        hist (Union[np.ndarray, Tuple[np.ndarray,np.ndarray]], optional):
                            Histogram from which to determine the threshold, and optionally a
                            corresponding array of bin center intensities. If no hist provided,
                            this function will compute it from the image. Default to None.

    Returns:
        #int: The calculated threshold, this threshold is different from Normal Otsu threshold, not in the same scale.
        Tuple[np.ndarray, float]: Tissue mask after gaussian blur. All pixels with an
        intensity higher than this value are assumed to be tissue. 
        (0 for non-tissue, 1 for tissue mask.)
    """
    red_channel = image[:, :, 0].astype(float)
    green_channel = image[:, :, 1].astype(float)
    blue_channel = image[:, :, 2].astype(float)

    red_to_green_mask = np.maximum(red_channel - green_channel, 0)
    blue_to_green_mask = np.maximum(blue_channel - green_channel, 0)

    tissue_heatmap = red_to_green_mask * blue_to_green_mask

    threshold = threshold_otsu(
        red_to_green_mask * blue_to_green_mask, nbins=nbins, hist=hist
    )
    
    mask = tissue_heatmap > threshold
    mask = np.where(mask, 1, 0)
    return mask

def _blood_mask(
        *,
        image: np.ndarray
) -> np.ndarray:
    """
    Blood mask generation.
    Args:
        image: Input image.
    Returns:
        Blood mask (0=background and 1=blood).
    """
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    lower_red = np.array([0, 110, 70])
    upper_red = np.array([20, 255, 255])
    mask_1 = cv2.inRange(hsv_image, lower_red, upper_red)

    lower_red = np.array([170, 120, 70])
    upper_red = np.array([200, 255, 255])
    mask_2 = cv2.inRange(hsv_image, lower_red, upper_red)
    
    blood_mask = mask_1 + mask_2
    blood_mask = np.array(blood_mask)
    blood_mask = np.where(blood_mask == 255, 1, 0)
    return blood_mask


def _gaussian_blur(
    *, image: np.ndarray, sigma: float, truncate: float = 3.5
) -> np.ndarray:
    """Apply gaussian blurring."""
    if sigma <= SIGMA_NO_OP:
        return image
    ksize = int(truncate * sigma + 0.5)
    if ksize % 2 == 0:
        ksize += 1
    return cv2.GaussianBlur(image, ksize=(ksize, ksize), sigmaX=sigma, sigmaY=sigma)
