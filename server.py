from flask import Flask, render_template, request, send_file
from http import HTTPStatus
import os
import numpy as np
from PIL import Image
import cv2
from image_processing.histogram import histogram_equalization, histogram_equalization_opencv
from image_processing.morphological import dilation, erosion, erosion_opencv, dilation_opencv
from image_processing.spatial_enhancement import mean_filter, median_filter, laplacian_filter, laplacian_filter_opencv, median_filter_opencv
from image_processing.segmentation import otsu_thresholding, otsu_threshold_opencv
from image_processing.compress import jpeg_encode_decode_grayscale, compress_jpeg_with_opencv  

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER

def convert_to_grayscale(image_array):
    """
    Chuyển đổi ảnh về định dạng grayscale.
    """
    if len(image_array.shape) == 2:  # Ảnh đã là grayscale
        return image_array
    elif len(image_array.shape) == 3:  # Ảnh có nhiều kênh (RGB hoặc RGBA)
        return cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    else:
        raise ValueError("Unsupported image format.")



@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process_image():
    if 'file' not in request.files or request.files['file'].filename == '':
        return "No file provided", HTTPStatus.BAD_REQUEST

    file = request.files['file']
    method_type = request.form.get('methodType', '')
    kernel_size = request.form.get('kernelSize', '3')  # Lấy giá trị kernelSize từ form, mặc định là 3

    if not method_type:
        return "No method selected", HTTPStatus.BAD_REQUEST

    try:
        kernel_size = int(kernel_size)

        filename = file.filename
        original_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(original_path)

        # Đọc ảnh và chuyển đổi sang grayscale
        image = Image.open(original_path)
        image_array = np.array(image)
        image_array = convert_to_grayscale(image_array)

        # Đảm bảo image_array là uint8 và có giá trị pixel từ 0 đến 255
        if image_array.dtype != np.uint8:
            image_array = image_array.astype(np.uint8)

        # Xử lý các phương pháp xử lý ảnh grayscale
        if method_type == 'equalize':
            image_array = histogram_equalization(image_array)
        elif method_type == 'equalize_opencv':
            image_array = histogram_equalization_opencv(image_array)
        elif method_type == 'mean_filter':
            image_array = mean_filter(image_array, kernel_size=kernel_size)
        elif method_type == 'median_filter':
            image_array = median_filter(image_array, kernel_size=kernel_size)
        elif method_type == 'median_filter_opencv':
            image_array = median_filter_opencv(image_array, kernel_size=kernel_size)
        elif method_type == 'laplacian_filter':
            image_array = laplacian_filter(image_array)
        elif method_type == 'laplacian_filter_opencv':
            image_array = laplacian_filter_opencv(image_array)
        elif method_type == 'dilation':
            image_array = dilation(image_array, kernel_size=kernel_size)
        elif method_type == 'dilation_opencv':
            image_array = dilation_opencv(image_array, kernel_size=kernel_size)
        elif method_type == 'erosion':
            image_array = erosion(image_array, kernel_size=kernel_size)
        elif method_type == 'erosion_opencv':
            image_array = erosion_opencv(image_array, kernel_size=kernel_size)
        elif method_type == 'otsu':
            image_array = otsu_thresholding(image_array)
        elif method_type == 'otsu_opencv':
            image_array = otsu_threshold_opencv(image_array)
        elif method_type == 'jpeg_custom_compress':
            # Nén ảnh grayscale bằng thuật toán tùy chỉnh
            reconstructed_image = jpeg_encode_decode_grayscale(image_array)
            compressed_path = os.path.join(app.config['RESULT_FOLDER'], f"compressed_custom_{filename.split('.')[0]}.jpg")
            Image.fromarray(reconstructed_image).save(compressed_path, format="JPEG")
            return render_template('result.html', message="Custom JPEG Compression Complete!", image_url=f"/results/{os.path.basename(compressed_path)}")
        elif method_type == 'jpeg_opencv_compress':
            # Nén ảnh grayscale bằng OpenCV
            compressed_path = os.path.join(app.config['RESULT_FOLDER'], f"compressed_opencv_{filename.split('.')[0]}.jpg")
            success = compress_jpeg_with_opencv(image_array, compressed_path, quality=50)
            if success:
                return render_template('result.html', message="OpenCV JPEG Compression Complete!", image_url=f"/results/{os.path.basename(compressed_path)}")
            else:
                return "Failed to compress the image using OpenCV.", HTTPStatus.INTERNAL_SERVER_ERROR
        else:
            return "Unsupported method", HTTPStatus.BAD_REQUEST

        # Lưu kết quả xử lý ảnh
        result_filename = f"processed_{method_type}_{filename.split('.')[0]}.png"  # Tên file không có đuôi
        processed_path = os.path.join(app.config['RESULT_FOLDER'], result_filename)
        Image.fromarray(image_array).save(processed_path, format='PNG')  # Lưu dưới định dạng PNG

        return render_template('result.html', image_url=f"/results/{os.path.basename(processed_path)}")

    except Exception as e:
        import traceback
        traceback.print_exc()  # In ra thông báo lỗi chi tiết
        return f"An error occurred: {str(e)}", HTTPStatus.INTERNAL_SERVER_ERROR

@app.route('/results/<filename>')
def result_image(filename):
    return send_file(os.path.join(app.config['RESULT_FOLDER'], filename))

if __name__ == '__main__':
    app.run(debug=True)