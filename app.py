import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import gdown
import os

# URL của file mô hình trên Google Drive
url = 'https://drive.google.com/file/d/1vjrd7w4bLy3aK3MrhIWrAF7sSSFdpR1D/view?usp=sharing'

# Tên file lưu trữ mô hình trên máy chủ
output = 'model1.h5'

# Kiểm tra nếu file mô hình chưa tồn tại, tải về
if not os.path.exists(output):
    gdown.download(url, output, quiet=False)

# Tải mô hình đã được huấn luyện
model = load_model('model1.h5')

# Định nghĩa hàm xử lý và dự đoán ảnh tải lên
def predict_image(image):
    size = (256, 256)  # Kích thước ảnh sau khi thay đổi kích thước
    image = ImageOps.fit(image, size, Image.ANTIALIAS)
    img = np.asarray(image)
    img = img / 255.0  # Chuẩn hóa ảnh
    img = np.expand_dims(img, axis=0)  # Thêm chiều batch

    # Dự đoán
    prediction = model.predict(img)
    return "Motorcycle" if prediction[0][0] < 0.5 else "Car"

# Cấu hình trang
st.set_page_config(page_title="Image Classification", layout="centered")

# Tiêu đề
st.title("Phân loại ảnh: Xe máy hoặc Ô tô")

# Tải ảnh lên
st.subheader("Tải ảnh lên")
uploaded_file = st.file_uploader("Chọn một ảnh...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Hiển thị ảnh đã tải lên
    image = Image.open(uploaded_file)
    st.image(image, caption='Ảnh đã tải lên', use_column_width=True)

    # Dự đoán ảnh
    st.write("")
    st.write("Đang phân loại...")
    label = predict_image(image)
    st.write(f"Ảnh đã tải lên là **{label}**.")
