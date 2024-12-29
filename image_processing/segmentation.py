import numpy as np
import cv2
def otsu_thresholding(image):
    hist, _ = np.histogram(image.flatten(), bins=256, range=[0, 256])
    total = image.size
    sum_total = np.sum(np.arange(256) * hist)
    sum_bg, weight_bg, max_variance = 0, 0, 0
    threshold = 0

    for t in range(256):
        weight_bg += hist[t]
        if weight_bg == 0:
            continue
        weight_fg = total - weight_bg
        if weight_fg == 0:
            break
        sum_bg += t * hist[t]
        mean_bg = sum_bg / weight_bg
        mean_fg = (sum_total - sum_bg) / weight_fg
        variance = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
        if variance > max_variance:
            max_variance = variance
            threshold = t
    return (image > threshold).astype(np.uint8) * 255

def otsu_threshold_opencv(image):
    """
    Áp dụng phân ngưỡng Otsu sử dụng OpenCV.

    :param image: Ảnh đầu vào (grayscale).
    :return: Ảnh nhị phân sau khi áp dụng phân ngưỡng Otsu.
    """
    # Áp dụng phân ngưỡng Otsu
    _, thresholded_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    return thresholded_image