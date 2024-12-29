import numpy as np
import cv2

def dilation(image, kernel_size=3):
    """
    Thực hiện phép dilation trên ảnh grayscale hoặc nhị phân.
    
    :param image: Ảnh đầu vào (grayscale hoặc nhị phân).
    :param kernel_size: Kích thước kernel (mặc định là 3x3).
    :return: Ảnh sau khi áp dụng dilation.
    """
    # Tạo kernel toàn số 1
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    
    # Lấy kích thước ảnh và kernel
    img_height, img_width = image.shape
    kernel_height, kernel_width = kernel.shape
    
    # Tính toán padding
    pad_height = kernel_height // 2
    pad_width = kernel_width // 2
    
    # Thêm padding vào ảnh
    padded_image = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width)), mode='constant', constant_values=0)
    
    # Tạo ảnh đầu ra
    output = np.zeros_like(image)
    
    # Áp dụng phép dilation
    for i in range(img_height):
        for j in range(img_width):
            # Lấy vùng ảnh tương ứng với kernel
            region = padded_image[i:i + kernel_height, j:j + kernel_width]
            # Áp dụng phép max với kernel
            output[i, j] = np.max(region[kernel == 1])
    
    return output

def dilation_opencv(image, kernel_size=3):
    """
    Thực hiện phép dilation sử dụng OpenCV.
    
    :param image: Ảnh đầu vào (grayscale).
    :param kernel_size: Kích thước kernel (mặc định là 3x3).
    :return: Ảnh sau khi áp dụng dilation.
    """
    # Tạo kernel (cấu trúc hình chữ nhật)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    
    # Áp dụng dilation
    dilated_image = cv2.dilate(image, kernel, iterations=1)
    
    return dilated_image

def erosion(image, kernel=None, kernel_size=3):
    """
    Thực hiện phép erosion trên ảnh grayscale hoặc nhị phân.
    
    :param image: Ảnh đầu vào (grayscale hoặc nhị phân).
    :param kernel: Phần tử cấu trúc (kernel). Nếu None, sử dụng kernel hình chữ nhật kích thước kernel_size x kernel_size.
    :param kernel_size: Kích thước kernel mặc định nếu kernel không được cung cấp.
    :return: Ảnh sau khi áp dụng erosion.
    """
    # Kiểm tra và tạo kernel nếu không được cung cấp
    if kernel is None:
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
    
    # Lấy kích thước ảnh và kernel
    img_height, img_width = image.shape
    kernel_height, kernel_width = kernel.shape
    
    # Tính toán padding
    pad_height = kernel_height // 2
    pad_width = kernel_width // 2
    
    # Thêm padding vào ảnh
    padded_image = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width)), mode='constant', constant_values=0)
    
    # Tạo ảnh đầu ra
    output = np.zeros_like(image)
    
    # Áp dụng phép erosion
    for i in range(img_height):
        for j in range(img_width):
            # Lấy vùng ảnh tương ứng với kernel
            region = padded_image[i:i + kernel_height, j:j + kernel_width]
            # Áp dụng phép min với kernel
            output[i, j] = np.min(region[kernel == 1])
    
    return output

def erosion_opencv(image, kernel_size=3):
    """
    Thực hiện phép erosion sử dụng OpenCV.
    :param image: Ảnh đầu vào (grayscale).
    :param kernel_size: Kích thước kernel (mặc định là 3x3).
    :return: Ảnh sau khi áp dụng erosion.
    """
    # Tạo kernel (cấu trúc hình chữ nhật)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    
    # Áp dụng erosion
    eroded_image = cv2.erode(image, kernel, iterations=1)
    
    return eroded_image