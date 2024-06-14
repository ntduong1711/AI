import streamlit as st
import gdown
from PIL import Image
import numpy as np
import tensorflow as tf
import os

# Đường dẫn chia sẻ của tệp mô hình trên Google Drive
model_url = 'https://drive.google.com/file/d/1MbKaItBuHEa-RcWekA3wd8BBU4t8QTtZ/view?usp=drive_link'
model_path = 'model2.h5'

# Tải mô hình từ Google Drive nếu chưa tồn tại
if not os.path.exists(model_path):
    gdown.download(model_url, model_path, quiet=False)

# Tải mô hình đã huấn luyện
model = tf.keras.models.load_model(model_path)

# Hàm để dự đoán lớp của ảnh
def predict_image(image):
    image = image.resize((224, 224))  # Điều chỉnh kích thước ảnh phù hợp với mô hình
    image = np.array(image) / 255.0  # Chuẩn hóa giá trị ảnh
    image = np.expand_dims(image, axis=0)  # Thêm một chiều để phù hợp với đầu vào của mô hình
    predictions = model.predict(image)
    return predictions

# Giao diện Streamlit
st.title("Nhận diện ô tô và xe máy")
st.write("Vui lòng tải lên một hình ảnh để nhận diện.")

uploaded_file = st.file_uploader("Chọn một tệp ảnh", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Ảnh đã tải lên", use_column_width=True)
    
    st.write("Đang phân tích...")
    
    predictions = predict_image(image)
    class_names = ['Car', 'Motorbike']  # Đặt tên lớp phù hợp với mô hình của bạn
    predicted_class = class_names[np.argmax(predictions)]
    
    st.write(f"Loại phương tiện: {predicted_class}")
    st.write(f"Độ chính xác: {np.max(predictions):.2f}")

