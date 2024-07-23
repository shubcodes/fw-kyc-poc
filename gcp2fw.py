import streamlit as st
from google.cloud import documentai_v1beta3 as documentai
from google.cloud.documentai_v1beta3 import types
import os
import requests
import json
import re
import time

# Set the environment variable to the absolute path of the key file
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "gcp.json"

# Initialize the client
client = documentai.DocumentProcessorServiceClient()

# Define the processor details
#project_id = ''
#location = ''
#processor_id = ''

# The full resource name of the processor
name = client.processor_path(project_id, location, processor_id)

# Streamlit interface
st.title("GCP OCR to Fireworks Mistral 8x22B")

uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    # Read the file into memory
    image_content = uploaded_file.read()

    # Measure time for Google Document AI
    start_time_gcp = time.time()

    # Configure the request
    request = types.ProcessRequest(
        name=name,
        raw_document=types.RawDocument(
            content=image_content,
            mime_type="image/jpeg"
        )
    )

    # Process the document
    result = client.process_document(request=request)

    # Calculate time taken for GCP API
    gcp_api_time = time.time() - start_time_gcp

    # Extract the text from the document
    document_text = result.document.text

    # Extract relevant information from the Document object
    gcp_result = {"text": result.document.text}

    # Measure time for Fireworks API
    start_time_fw = time.time()

    # Prepare the payload for Fireworks API
    url = "https://api.fireworks.ai/inference/v1/chat/completions"
    payload = {
        "model": "accounts/fireworks/models/mixtral-8x22b-instruct",
        "max_tokens": 8192,
        "top_p": 1,
        "top_k": 40,
        "presence_penalty": 0,
        "frequency_penalty": 0,
        "temperature": 0.6,
        "messages": [
            {
                "role": "user",
                "content": f"From the following text dump from a user's identifying documentation please parse all of the user data and provide it in a json format\n\nDocument text: {document_text}. As a strict identification check robot you must make sure to validate that all pieces of information correctly match what is expected of licenses and passports in the Untied States. You must make sure for the address that the city, state, and zipcode are actual USA city states and zip codes. You must be thorough and exact. "
            }
        ]
    }
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": "Bearer API KEY"
    }

    # Send the request to the Fireworks API
    response = requests.post(url, headers=headers, data=json.dumps(payload))

    # Calculate time taken for Fireworks API
    fw_api_time = time.time() - start_time_fw

    # Extract the JSON part from the response
    response_json = response.json()
    content = response_json['choices'][0]['message']['content']

    # Extract the JSON part from the content
    json_str_match = re.search(r'```json\n({.*})\n```', content, re.DOTALL)
    parsed_json = {}
    if json_str_match:
        json_str = json_str_match.group(1)
        parsed_json = json.loads(json_str)

    # Display the results side by side
    col1, col2 = st.columns(2)

    with col1:
        st.header("GCP Document AI Response")
        st.json(gcp_result)

    with col2:
        st.header("Fireworks API Response")
        st.json(parsed_json)

    # Display timing information
    st.write(f"GCP Document AI Processing Time: {gcp_api_time:.2f} seconds")
    st.write(f"Fireworks API Processing Time: {fw_api_time:.2f} seconds")
