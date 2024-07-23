import streamlit as st
import requests
from PIL import Image
import io
import base64
import fireworks.client

# Fireworks API credentials
FIREWORKS_API_KEY = ''
fireworks.client.api_key = FIREWORKS_API_KEY

# Function to encode image to base64
def encode_image(image_data):
    return base64.b64encode(image_data).decode('utf-8')

# Function to process image using Fireworks AI's vision-language model
def process_document(image_data):
    image_base64 = encode_image(image_data)
    image_url = f"data:image/jpeg;base64,{image_base64}"

    response = fireworks.client.ChatCompletion.create(
        model="accounts/fireworks/models/firellava-13b",
        messages=[{
            "role": "user",
            "content": [{
                "type": "text",
                "text": "Can you describe this image?",
            }, {
                "type": "image_url",
                "image_url": {
                    "url": image_url
                },
            }],
        }],
    )

    # Extracting content from response
    if response.choices:
        extracted_data = response.choices[0].message.content
        return extracted_data
    else:
        return "No data extracted. Please check the image and try again."

# Streamlit app
st.title("Test #1 -- Direct to Llava-13B")

uploaded_files = st.file_uploader("Upload your documents", type=['png', 'jpg', 'jpeg'], accept_multiple_files=True)

if uploaded_files:
    for uploaded_file in uploaded_files:
        # Read the uploaded file
        image_data = uploaded_file.read()
        image = Image.open(io.BytesIO(image_data))

        st.image(image, caption='Uploaded Document', use_column_width=True)

        # Process the document
        extracted_data = process_document(image_data)
        
        # Display the extracted data
        st.write(f'Extracted Data for {uploaded_file.name}:', extracted_data)
