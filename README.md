# Ứng Dụng Xử Lý Ảnh Với Flask

Dự án này là một ứng dụng web dựa trên Flask, cung cấp các chức năng xử lý ảnh như cải thiện ảnh (cân bằng histogram, sử dụng các bộ lọc không gian như Median, Laplacian), biến đổi hình thái học (dilation, erosion), phân đoạn ảnh (otsu thresholding) và nén JPEG.

## Tính Năng
- **Cân Bằng Histogram**:
  - Tự xây dựng dựa trên thuật toán và sử dụng OpenCV.
- **Biến Đổi Hình Thái Học**:
  - Dilation và Erosion (tự xây dựng và OpenCV).
- **Bộ Lọc Không Gian**:
  - Bộ lọc Mean, Median, và Laplacian (tự xây dựng và OpenCV).
- **Phân Đoạn Ảnh**:
  - Ngưỡng Otsu (tự xây dựng và OpenCV).
- **Nén Ảnh**:
  - Nén JPEG (tự xây dựng và OpenCV).

## Yêu Cầu
- Python 3.x
- Pip (Trình quản lý gói Python)
- Môi trường ảo (không bắt buộc nhưng được khuyến nghị)

## Cài Đặt

1. **Clone kho lưu trữ**:
   ```bash
   git clone <repository-url>
   cd <repository-folder>
   ```

2. **Tạo môi trường ảo (không bắt buộc)**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # Trên Windows, sử dụng `venv\Scripts\activate`
   ```

3. **Cài đặt các thư viện cần thiết**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Tạo các thư mục cần thiết**:
   ```bash
   mkdir uploads results
   ```

## Sử Dụng

1. **Chạy ứng dụng**:
   ```bash
   python app.py
   ```

2. **Truy cập ứng dụng**:
   Mở trình duyệt web và truy cập: `http://127.0.0.1:5000/`

3. **Xử lý ảnh**:
   - Tải lên một file ảnh.
   - Chọn phương pháp xử lý.
   - (Tùy chọn) Nhập kích thước kernel nếu cần.
   - Gửi biểu mẫu để xử lý ảnh.

4. **Xem kết quả**:
   - Ảnh đã xử lý sẽ được hiển thị.
   - Bạn có thể tải xuống ảnh đã xử lý hoặc nén từ thư mục kết quả.

## Cấu Trúc Thư Mục
```
.
├── app.py                       # File ứng dụng chính
├── templates/
│   ├── index.html              # Template trang chính
│   ├── result.html             # Template hiển thị kết quả
├── static/
├── uploads/                    # Thư mục chứa ảnh tải lên
├── results/                    # Thư mục chứa ảnh đã xử lý/nén
├── image_processing/           # Các hàm xử lý ảnh tự xây dựng
│   ├── histogram.py
│   ├── morphological.py
│   ├── spatial_enhancement.py
│   ├── segmentation.py
│   ├── compress.py
├── requirements.txt            # Các thư viện cần thiết
├── README.md                   # Tài liệu hướng dẫn
```

## Các Phương Pháp Xử Lý Có Sẵn
| Phương Pháp           | Mô Tả                                              |
|-----------------------|----------------------------------------------------|
| `equalize`            | Cân bằng histogram (triển khai tự xây dựng).      |
| `equalize_opencv`     | Cân bằng histogram sử dụng OpenCV.                |
| `mean_filter`         | Bộ lọc Mean để làm mịn ảnh.                       |
| `median_filter`       | Bộ lọc Median để giảm nhiễu.                      |
| `median_filter_opencv`| Bộ lọc Median sử dụng OpenCV.                     |
| `laplacian_filter`    | Bộ lọc Laplacian để phát hiện biên cạnh.          |
| `laplacian_filter_opencv`| Bộ lọc Laplacian sử dụng OpenCV.                |
| `dilation`            | Dilation ảnh (triển khai tự xây dựng).            |
| `dilation_opencv`     | Dilation ảnh sử dụng OpenCV.                      |
| `erosion`             | Erosion ảnh (triển khai tự xây dựng).             |
| `erosion_opencv`      | Erosion ảnh sử dụng OpenCV.                       |
| `otsu`                | Ngưỡng Otsu (triển khai tự xây dựng).             |
| `otsu_opencv`         | Ngưỡng Otsu sử dụng OpenCV.                       |
| `jpeg_custom_compress`| Nén JPEG với thuật toán tự xây dựng.              |
| `jpeg_opencv_compress`| Nén JPEG sử dụng OpenCV.                          |



