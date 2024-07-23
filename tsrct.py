import streamlit as st
import cv2
import pytesseract
import numpy as np
from PIL import Image

# Function to detect and correct the orientation of the image
def correct_orientation(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(gray)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    
    coords = np.column_stack(np.where(thresh > 0))
    angle = cv2.minAreaRect(coords)[-1]
    
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    
    return rotated

# Function to preprocess the image
def preprocess_image(image):
    rotated_image = correct_orientation(image)
    
    gray = cv2.cvtColor(rotated_image, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 3)
    gray = cv2.equalizeHist(gray)
    
    # Additional preprocessing steps
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    
    return gray

# Function to extract text using Tesseract
def extract_text(image):
    preprocessed_image = preprocess_image(image)
    custom_config = r'--oem 3 --psm 6'
    text = pytesseract.image_to_string(preprocessed_image, config=custom_config)
    return text

# Streamlit App
st.title('Tesseract Test #1 OCR')

uploaded_file = st.file_uploader('Upload an image', type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image = np.array(image)
    
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    with st.spinner('Processing...'):
        extracted_text = extract_text(image)
    
    st.success('Text Extraction Complete!')
    st.text_area('Extracted Text', extracted_text, height=300)
