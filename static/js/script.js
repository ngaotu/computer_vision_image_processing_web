const fileInput = document.getElementById('file');
const imageInfo = document.getElementById('imageInfo');
const imageName = document.getElementById('imageName');
const imageResolution = document.getElementById('imageResolution');
const imagePreview = document.getElementById('imagePreview');
const previewImage = document.getElementById('previewImage');

fileInput.addEventListener('change', (event) => {
    const file = event.target.files[0];
    if (file) {
        // Hiển thị tên ảnh
        imageName.textContent = file.name;

        // Tạo một đối tượng Image để lấy độ phân giải
        const img = new Image();
        img.src = URL.createObjectURL(file);

        img.onload = () => {
            // Hiển thị độ phân giải
            imageResolution.textContent = `${img.width} x ${img.height}`;

            // Hiển thị ảnh
            previewImage.src = img.src;
            imagePreview.style.display = 'block';
            imageInfo.style.display = 'block';
        };
    } else {
        // Ẩn thông tin và ảnh nếu không có tệp được chọn
        imageInfo.style.display = 'none';
        imagePreview.style.display = 'none';
    }
});

const form = document.getElementById('imageProcessingForm');
const loading = document.getElementById('loading');
const dimmedBackground = document.getElementById('dimmedBackground');
const kernelInput = document.getElementById('kernelInput');
const kernelSizeInput = document.getElementById('kernelSize');
const kernelError = document.getElementById('kernelError');
const methodOptions = document.querySelectorAll('input[name="methodType"]');

// Hiển thị/ẩn trường nhập kernel dựa trên phương pháp được chọn
methodOptions.forEach(option => {
    option.addEventListener('change', () => {
        const selectedMethod = option.value;
        const methodsRequiringKernel = [
            'mean_filter', 'median_filter', 'median_filter_opencv', 'dilation', 'dilation_opencv', 'erosion', 'erosion_opencv'
        ];

        if (methodsRequiringKernel.includes(selectedMethod)) {
            kernelInput.style.display = 'block';
        } else {
            kernelInput.style.display = 'none';
        }
    });
});

// Kiểm tra kích thước kernel trước khi gửi form
form.addEventListener('submit', (event) => {
    const selectedMethod = document.querySelector('input[name="methodType"]:checked').value;
    const methodsRequiringKernel = [
        'spatial', 'median_filter', 'dilation', 'erosion', 'erosion_opencv'
    ];

    if (methodsRequiringKernel.includes(selectedMethod)) {
        const kernelSize = parseInt(kernelSizeInput.value);

        if (kernelSize % 2 === 0) {
            // Hiển thị thông báo lỗi và ngăn form gửi đi
            kernelError.style.display = 'block';
            event.preventDefault(); // Ngăn form gửi đi
            return;
        } else {
            kernelError.style.display = 'none';
        }
    }

    // Hiển thị loading animation
    loading.style.display = 'flex';
    dimmedBackground.style.display = 'block';
});