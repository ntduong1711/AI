import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import streamlit as st
from PIL import Image

# Mount Google Drive if needed
# from google.colab import drive
# drive.mount('/content/drive')

# Set paths for data
data_train_path = '/content/drive/MyDrive/AI/Train'
data_valid_path = '/content/drive/MyDrive/AI/Validation'

# Constants
img_width = 256
img_height = 256
batch_size = 32

# Load data
train_datagen = ImageDataGenerator(
    rescale=1./255,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest')

valid_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    data_train_path,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

valid_generator = valid_datagen.flow_from_directory(
    data_valid_path,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

# Load model
model = tf.keras.models.Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(img_width, img_height, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(512, activation=tf.nn.relu))
model.add(Dense(2, activation=tf.nn.softmax))
model.compile(optimizer=RMSprop(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Load history
history = None  # Define history if you want to display the history plot

# Define labels
labels = ['Ô tô', 'Xe máy']

# Streamlit app
st.title('Vehicle Classification')

# Function to make predictions
def make_prediction(image):
    img = image.resize((256, 256))  # Resize image
    img_array = np.array(img) / 255.0  # Convert to numpy array and normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    prediction = model.predict(img_array)
    predicted_label = labels[np.argmax(prediction)]
    return predicted_label

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")
    prediction = make_prediction(image)
    st.write('Prediction:', prediction)

# Display history plot if available
if history is not None:
    st.subheader('Model Accuracy')
    st.line_chart(history.history['accuracy'])
    st.line_chart(history.history['val_accuracy'])
