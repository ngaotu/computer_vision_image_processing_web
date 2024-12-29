import numpy as np
import cv2
def histogram_equalization(image):
    """
    Hàm cân bằng histogram cho ảnh grayscale.
    :param image: Mảng numpy 2D đại diện cho ảnh grayscale.
    :return: Ảnh sau khi cân bằng histogram.
    """
    total_pixels = image.size
    # Bước 1: Tính histogram
    hist = [0] * 256 
    for pixel_value in image.flatten(): 
        hist[pixel_value] += 1 

    # Bước 2: Tính hàm phân phối tích lũy (CDF)
    cdf = []  
    cumulative_sum = 0  
    for frequency in hist:
        cumulative_sum += frequency 
        cdf.append(cumulative_sum)  

    # Bước 3: Tìm giá trị CDF nhỏ nhất(cdf_min)
    cdf_min = None
    for value in cdf:
        if value > 0:
            cdf_min = value  
            break

    # Bước 4: Chuẩn hóa CDF
    cdf_normalized = []
    for value in cdf:
        normalized_value = (value - cdf_min) / (total_pixels - cdf_min) * 255
        cdf_normalized.append(normalized_value)

    # Bước 5: Ánh xạ giá trị pixel theo CDF đã chuẩn hóa
    equalized_image = [] 
    for pixel_value in image.flatten():
        new_pixel_value = cdf_normalized[pixel_value]
        equalized_image.append(new_pixel_value)
    equalized_image = np.array(equalized_image).reshape(image.shape).astype(np.uint8)

    return equalized_image

def histogram_equalization_opencv(image):
    """
    Hàm cân bằng histogram sử dụng OpenCV.
    :param image: Mảng numpy 2D đại diện cho ảnh grayscale.
    :return: Ảnh sau khi cân bằng histogram.
    """
    return cv2.equalizeHist(image)