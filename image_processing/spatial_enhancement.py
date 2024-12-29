import numpy as np
import cv2
def mean_filter(image, kernel_size=3):
    """
    Lọc trung bình để làm mượt ảnh.
    Args:
        image (ndarray): Ảnh đầu vào.
        kernel_size (int): Kích thước kernel (phải là số lẻ).
    Returns:
        ndarray: Ảnh sau xử lý.
    """
    pad_size = kernel_size // 2
    padded_image = np.pad(image, pad_size, mode='constant')
    output = np.zeros_like(image)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            region = padded_image[i:i + kernel_size, j:j + kernel_size]
            output[i, j] = np.mean(region)

    return output

def median_filter(image, kernel_size=3):
    """
    Lọc trung vị để giảm nhiễu "Salt and Pepper".
    Args:
        image (ndarray): Ảnh đầu vào.
        kernel_size (int): Kích thước kernel (phải là số lẻ).
    Returns:
        ndarray: Ảnh sau xử lý.
    """
    pad_size = kernel_size // 2
    padded_image = np.pad(image, pad_size, mode='constant')
    output = np.zeros_like(image)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            region = padded_image[i:i + kernel_size, j:j + kernel_size]
            output[i, j] = np.median(region)

    return output

def median_filter_opencv(image, kernel_size=3):
    """
    Lọc trung vị sử dụng OpenCV.
    Args:
        image (ndarray): Ảnh đầu vào.
        kernel_size (int): Kích thước kernel (phải là số lẻ).
    Returns:
        ndarray: Ảnh sau xử lý.
    """
    return cv2.medianBlur(image, kernel_size)




def laplacian_filter(image):
    """
    Áp dụng bộ lọc Laplacian để phát hiện biên mà không dùng OpenCV.
    Args:
        image (ndarray): Ảnh đầu vào (grayscale).
    Returns:
        ndarray: Ảnh sau khi áp dụng bộ lọc Laplacian (kiểu uint8).
    """
    # Kernel Laplacian 3x3
    laplacian_kernel = np.array([[0, -1, 0],
                                 [-1, 4, -1],
                                 [0, -1, 0]])

    # Padding để giữ kích thước ảnh không thay đổi
    pad_size = 1
    padded_image = np.pad(image, pad_size, mode='constant')

    # Khởi tạo mảng kết quả
    output = np.zeros_like(image, dtype=np.float64)

    # Lặp qua từng pixel và áp dụng kernel Laplacian
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            region = padded_image[i:i + 3, j:j + 3]  # Lấy vùng 3x3
            output[i, j] = np.sum(region * laplacian_kernel)  # Áp dụng kernel

    # Lấy giá trị tuyệt đối để loại bỏ giá trị âm
    output_abs = np.abs(output)

    # Chuẩn hóa giá trị về khoảng [0, 255]
    output_normalized = (output_abs - output_abs.min()) / (output_abs.max() - output_abs.min()) * 255

    # Chuyển đổi về kiểu uint8
    output_uint8 = output_normalized.astype(np.uint8)

    return output_uint8


def laplacian_filter_opencv(image):
    """
    Áp dụng bộ lọc Laplacian để phát hiện biên sử dụng OpenCV.
    Args:
        image (ndarray): Ảnh đầu vào (grayscale).
    Returns:
        ndarray: Ảnh sau khi áp dụng bộ lọc Laplacian (kiểu uint8).
    """
    # Áp dụng bộ lọc Laplacian trực tiếp từ OpenCV
    laplacian_output = cv2.Laplacian(image, cv2.CV_64F, ksize=3)

    # Lấy giá trị tuyệt đối để loại bỏ giá trị âm
    laplacian_abs = np.abs(laplacian_output)

    # Chuẩn hóa giá trị về khoảng [0, 255]
    laplacian_normalized = cv2.normalize(laplacian_abs, None, 0, 255, cv2.NORM_MINMAX)

    # Chuyển đổi về kiểu uint8
    laplacian_uint8 = np.uint8(laplacian_normalized)

    return laplacian_uint8