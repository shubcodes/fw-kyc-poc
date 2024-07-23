import streamlit as st
import requests
from PIL import Image
import io
import base64
import fireworks.client
import re

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
                "text": """
                Please extract the following information from the document image:
                - Full Name 
                - Date of Birth
                - Document Number
                - Sex
                - Address
                """
                
                ,
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

# Function to parse extracted text for specific fields
def parse_extracted_data(data):
    parsed_data = {}
    # Regular expressions for common fields in IDs and passports
    name_pattern = re.compile(r'(Name|Full Name|Given Name):?\s*(.*)', re.IGNORECASE)
    dob_pattern = re.compile(r'(DOB|Date of Birth|Birth Date):?\s*(\d{2}/\d{2}/\d{4}|\d{4}-\d{2}-\d{2})', re.IGNORECASE)
    doc_num_pattern = re.compile(r'(DLN|Document Number|ID Number|Passport Number):?\s*(\w+)', re.IGNORECASE)
    sex_pattern = re.compile(r'(Sex|Gender):?\s*(M|F)', re.IGNORECASE)
    address_pattern = re.compile(r'(Address|Residence):?\s*(.*\d{5})', re.IGNORECASE)

    # Extracting fields using regex
    name_match = name_pattern.search(data)
    dob_match = dob_pattern.search(data)
    doc_num_match = doc_num_pattern.search(data)
    sex_match = sex_pattern.search(data)
    address_match = address_pattern.search(data)

    if name_match:
        parsed_data['Name'] = name_match.group(2)
    if dob_match:
        parsed_data['DOB'] = dob_match.group(2)
    if doc_num_match:
        parsed_data['Document Number'] = doc_num_match.group(2)
    if sex_match:
        parsed_data['Sex'] = sex_match.group(2)
    if address_match:
        parsed_data['Address'] = address_match.group(2)

    return parsed_data

# Streamlit app
st.title("Test #2 -- Direct to Llava-13B + Post Processing")

uploaded_files = st.file_uploader("Upload your documents", type=['png', 'jpg', 'jpeg'], accept_multiple_files=True)

if uploaded_files:
    for uploaded_file in uploaded_files:
        # Read the uploaded file
        image_data = uploaded_file.read()
        image = Image.open(io.BytesIO(image_data))

        st.image(image, caption='Uploaded Document', use_column_width=True)

        # Process the document
        extracted_data = process_document(image_data)

        # Parse the extracted data for specific fields
        parsed_data = parse_extracted_data(extracted_data)

        # Display the parsed data
        st.write(f'Parsed Data for {uploaded_file.name}:', parsed_data)
