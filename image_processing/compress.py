import numpy as np
from PIL import Image
from scipy.fftpack import dct, idct  # Sử dụng FFT để tính DCT nhanh hơn
import cv2
# Ma trận lượng tử hóa chuẩn JPEG
JPEG_QUANTIZATION_TABLE = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99]
])

# Bước 1: Chuyển đổi ảnh sang grayscale (nếu cần)
def convert_to_grayscale(image_array):
    """
    Chuyển đổi ảnh về định dạng grayscale.
    """
    if len(image_array.shape) == 2:  # Ảnh đã là grayscale
        return image_array
    elif len(image_array.shape) == 3:  # Ảnh có nhiều kênh (RGB hoặc RGBA)
        return np.dot(image_array[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)
    else:
        raise ValueError("Unsupported image format.")

# Bước 2: Chuyển dữ liệu từ [0, 255] sang [-128, 127]
def shift_range(image_array):
    """
    Chuyển dữ liệu từ [0, 255] sang [-128, 127].
    """
    return image_array - 128

# Bước 3: Chia ảnh thành các khối 8x8 (có padding nếu cần)
def split_into_blocks(image_array):
    """
    Chia ảnh thành các khối 8x8.
    Nếu kích thước ảnh không chia hết cho 8, thêm padding.
    """
    height, width = image_array.shape
    # Tính toán số lượng hàng và cột cần thêm để chia hết cho 8
    pad_height = (8 - height % 8) % 8
    pad_width = (8 - width % 8) % 8
    
    # Thêm padding vào ảnh
    padded_image = np.pad(image_array, ((0, pad_height), (0, pad_width)), mode='constant', constant_values=0)
    
    # Chia ảnh thành các khối 8x8
    blocks = []
    for i in range(0, padded_image.shape[0], 8):
        for j in range(0, padded_image.shape[1], 8):
            block = padded_image[i:i+8, j:j+8]
            blocks.append(block)
    return blocks, height, width

# Bước 4: Áp dụng DCT lên mỗi khối 8x8 (sử dụng FFT để tối ưu)
def apply_dct(block):
    """
    Áp dụng DCT 2D lên một khối 8x8.
    """
    return dct(dct(block.T, norm='ortho').T, norm='ortho')

# Bước 5: Lượng tử hóa
def quantize(dct_block, quantization_table):
    """
    Lượng tử hóa khối DCT sử dụng ma trận lượng tử hóa.
    """
    return np.round(dct_block / quantization_table).astype(int)

# Bước 6: Quét Zig-Zag
def zigzag_scan(block):
    """
    Quét khối 8x8 theo thứ tự Zig-Zag.
    """
    zigzag_order = [
        0, 1, 5, 6, 14, 15, 27, 28,
        2, 4, 7, 13, 16, 26, 29, 42,
        3, 8, 12, 17, 25, 30, 41, 43,
        9, 11, 18, 24, 31, 40, 44, 53,
        10, 19, 23, 32, 39, 45, 52, 54,
        20, 22, 33, 38, 46, 51, 55, 56,
        21, 34, 37, 47, 50, 57, 58, 59,
        35, 36, 48, 49, 60, 61, 62, 63
    ]
    return block.flatten()[zigzag_order]

# Bước 7: Mã hóa Run-Length Coding (RLC)
def run_length_encoding(quantized_block):
    """
    Mã hóa Run-Length Coding (RLC) cho khối lượng tử hóa.
    """
    rlc = []
    count = 1
    for i in range(1, len(quantized_block)):
        if quantized_block[i] == quantized_block[i - 1]:
            count += 1
        else:
            rlc.append((quantized_block[i - 1], count))
            count = 1
    rlc.append((quantized_block[-1], count))
    return rlc

# Bước 8: Giải mã Run-Length Coding (RLC)
def run_length_decoding(rlc_block):
    """
    Giải mã Run-Length Coding (RLC) để khôi phục khối lượng tử hóa.
    """
    quantized_block = []
    for value, count in rlc_block:
        quantized_block.extend([value] * count)
    return np.array(quantized_block)

# Bước 9: Quét Zig-Zag ngược
def inverse_zigzag_scan(zigzag_block):
    """
    Quét Zig-Zag ngược để khôi phục khối 8x8.
    """
    block = np.zeros((8, 8))
    zigzag_order = [
        0, 1, 5, 6, 14, 15, 27, 28,
        2, 4, 7, 13, 16, 26, 29, 42,
        3, 8, 12, 17, 25, 30, 41, 43,
        9, 11, 18, 24, 31, 40, 44, 53,
        10, 19, 23, 32, 39, 45, 52, 54,
        20, 22, 33, 38, 46, 51, 55, 56,
        21, 34, 37, 47, 50, 57, 58, 59,
        35, 36, 48, 49, 60, 61, 62, 63
    ]
    for i, value in enumerate(zigzag_block):
        block[zigzag_order[i] // 8, zigzag_order[i] % 8] = value
    return block

# Bước 10: Giải lượng tử hóa
def dequantize(quantized_block, quantization_table):
    """
    Giải lượng tử hóa khối DCT.
    """
    return quantized_block * quantization_table

# Bước 11: Áp dụng Inverse DCT (IDCT) lên mỗi khối 8x8 (sử dụng FFT để tối ưu)
def apply_idct(block):
    """
    Áp dụng Inverse DCT (IDCT) lên một khối 8x8.
    """
    return idct(idct(block.T, norm='ortho').T, norm='ortho')

# Bước 12: Khôi phục ảnh từ các khối 8x8 (loại bỏ padding nếu có)
def reconstruct_image(blocks, original_height, original_width):
    """
    Khôi phục ảnh từ các khối 8x8.
    Loại bỏ padding nếu có.
    """
    # Tính toán kích thước ảnh đã được padding
    padded_height = (original_height + 7) // 8 * 8
    padded_width = (original_width + 7) // 8 * 8
    
    # Tạo ảnh mới với kích thước đã được padding
    padded_image = np.zeros((padded_height, padded_width))
    
    # Gán các khối vào ảnh
    block_index = 0
    for i in range(0, padded_height, 8):
        for j in range(0, padded_width, 8):
            padded_image[i:i+8, j:j+8] = blocks[block_index]
            block_index += 1
    
    # Loại bỏ padding để khôi phục kích thước ảnh gốc
    reconstructed_image = padded_image[:original_height, :original_width]
    return reconstructed_image

# Hàm thử nghiệm toàn bộ quy trình nén và giải nén JPEG
def jpeg_encode_decode_grayscale(image_array):
    """
    Thực hiện nén và giải nén JPEG trên ảnh grayscale.
    """
    # Bước 1: Chuyển đổi ảnh sang grayscale (nếu cần)
    image_array = convert_to_grayscale(image_array)
    height, width = image_array.shape
    
    # Bước 2: Chuyển dữ liệu từ [0, 255] sang [-128, 127]
    image_array = shift_range(image_array)
    
    # Bước 3: Chia ảnh thành các khối 8x8 (có padding nếu cần)
    blocks, original_height, original_width = split_into_blocks(image_array)
    encoded_blocks = []
    
    # Bước 4-7: Xử lý từng khối (nén)
    for block in blocks:
        dct_block = apply_dct(block)
        quantized_block = quantize(dct_block, JPEG_QUANTIZATION_TABLE)
        zigzag_block = zigzag_scan(quantized_block)
        rlc_block = run_length_encoding(zigzag_block)
        encoded_blocks.append(rlc_block)
    
    # Bước 8-11: Xử lý từng khối (giải nén)
    decoded_blocks = []
    for rlc_block in encoded_blocks:
        zigzag_block = run_length_decoding(rlc_block)
        quantized_block = inverse_zigzag_scan(zigzag_block)
        dct_block = dequantize(quantized_block, JPEG_QUANTIZATION_TABLE)
        idct_block = apply_idct(dct_block)
        decoded_blocks.append(idct_block)
    
    # Bước 12: Khôi phục ảnh (loại bỏ padding nếu có)
    reconstructed_image = reconstruct_image(decoded_blocks, original_height, original_width)
    
    # Bước 13: Chuyển dữ liệu từ [-128, 127] sang [0, 255]
    reconstructed_image = np.clip(reconstructed_image + 128, 0, 255).astype(np.uint8)
    
    return reconstructed_image


def compress_jpeg_with_opencv(image_array, output_path, quality=50):
    """
    Nén ảnh JPEG sử dụng OpenCV.
    """
    try:
        # Đảm bảo image_array là uint8 và có giá trị pixel từ 0 đến 255
        if image_array.dtype != np.uint8:
            image_array = image_array.astype(np.uint8)

        # Nén và lưu ảnh
        success = cv2.imwrite(output_path, image_array, [cv2.IMWRITE_JPEG_QUALITY, quality])
        return success
    except Exception as e:
        print(f"Error during JPEG compression: {e}")
        return False