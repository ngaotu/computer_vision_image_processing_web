import numpy as np

def distance_matrix(shape):
    """
    Tính ma trận khoảng cách từ tâm ảnh trong miền tần số.
    Args:
        shape (tuple): Kích thước ảnh (rows, cols).
    Returns:
        ndarray: Ma trận khoảng cách.
    """
    rows, cols = shape
    crow, ccol = rows // 2, cols // 2
    u, v = np.meshgrid(range(cols), range(rows))
    return np.sqrt((u - ccol) ** 2 + (v - crow) ** 2)

def ideal_low_pass_filter(image, radius):
    """
    Bộ lọc thông thấp lý tưởng (Ideal Low-Pass Filter).
    Args:
        image (ndarray): Ảnh đầu vào.
        radius (int): Bán kính cut-off.
    Returns:
        ndarray: Ảnh đã xử lý.
    """
    fft_image = np.fft.fft2(image)
    fft_shift = np.fft.fftshift(fft_image)

    D = distance_matrix(image.shape)
    H = (D <= radius).astype(np.float32)

    filtered_fft = fft_shift * H
    filtered_image = np.fft.ifft2(np.fft.ifftshift(filtered_fft))
    return np.abs(filtered_image)

def gaussian_low_pass_filter(image, radius):
    """
    Bộ lọc thông thấp Gaussian.
    Args:
        image (ndarray): Ảnh đầu vào.
        radius (int): Bán kính cut-off.
    Returns:
        ndarray: Ảnh đã xử lý.
    """
    fft_image = np.fft.fft2(image)
    fft_shift = np.fft.fftshift(fft_image)

    D = distance_matrix(image.shape)
    H = np.exp(-(D ** 2) / (2 * (radius ** 2)))

    filtered_fft = fft_shift * H
    filtered_image = np.fft.ifft2(np.fft.ifftshift(filtered_fft))
    return np.abs(filtered_image)

def butterworth_low_pass_filter(image, radius, order):
    """
    Bộ lọc thông thấp Butterworth.
    Args:
        image (ndarray): Ảnh đầu vào.
        radius (int): Bán kính cut-off.
        order (int): Bậc của bộ lọc.
    Returns:
        ndarray: Ảnh đã xử lý.
    """
    fft_image = np.fft.fft2(image)
    fft_shift = np.fft.fftshift(fft_image)

    D = distance_matrix(image.shape)
    H = 1 / (1 + (D / radius) ** (2 * order))

    filtered_fft = fft_shift * H
    filtered_image = np.fft.ifft2(np.fft.ifftshift(filtered_fft))
    return np.abs(filtered_image)
def ideal_high_pass_filter(image, radius):
    """
    Bộ lọc thông cao lý tưởng (Ideal High-Pass Filter).
    Args:
        image (ndarray): Ảnh đầu vào.
        radius (int): Bán kính cut-off.
    Returns:
        ndarray: Ảnh đã xử lý.
    """
    fft_image = np.fft.fft2(image)
    fft_shift = np.fft.fftshift(fft_image)

    D = distance_matrix(image.shape)
    H = (D > radius).astype(np.float32)

    filtered_fft = fft_shift * H
    filtered_image = np.fft.ifft2(np.fft.ifftshift(filtered_fft))
    return np.abs(filtered_image)

def gaussian_high_pass_filter(image, radius):
    """
    Bộ lọc thông cao Gaussian.
    Args:
        image (ndarray): Ảnh đầu vào.
        radius (int): Bán kính cut-off.
    Returns:
        ndarray: Ảnh đã xử lý.
    """
    fft_image = np.fft.fft2(image)
    fft_shift = np.fft.fftshift(fft_image)

    D = distance_matrix(image.shape)
    H = 1 - np.exp(-(D ** 2) / (2 * (radius ** 2)))

    filtered_fft = fft_shift * H
    filtered_image = np.fft.ifft2(np.fft.ifftshift(filtered_fft))
    return np.abs(filtered_image)

def butterworth_high_pass_filter(image, radius, order):
    """
    Bộ lọc thông cao Butterworth.
    Args:
        image (ndarray): Ảnh đầu vào.
        radius (int): Bán kính cut-off.
        order (int): Bậc của bộ lọc.
    Returns:
        ndarray: Ảnh đã xử lý.
    """
    fft_image = np.fft.fft2(image)
    fft_shift = np.fft.fftshift(fft_image)

    D = distance_matrix(image.shape)
    H = 1 / (1 + (radius / D) ** (2 * order))

    filtered_fft = fft_shift * H
    filtered_image = np.fft.ifft2(np.fft.ifftshift(filtered_fft))
    return np.abs(filtered_image)
