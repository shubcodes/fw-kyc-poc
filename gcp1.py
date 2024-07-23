import json
import time
import os
from PIL import Image
import io
from google.cloud import documentai_v1beta3 as documentai
from google.cloud.documentai_v1beta3 import types
import streamlit as st

# Set the environment variable to the absolute path of the key file
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "gcp.json"

# Initialize the client
client = documentai.DocumentProcessorServiceClient()

# Define the processor details
#project_id = ''
#location = 'us'
#processor_id = ''
#processor_version_id = ""  # Optional. Processor version to use

# The full resource name of the processor
name = client.processor_path(project_id, location, processor_id)

# Streamlit app
st.title("GCP Test -- Direct OCR Text")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    start_time = time.time()  # Record start time
    
    image_content = uploaded_file.read()
    image = Image.open(io.BytesIO(image_content))
    
    st.image(image, caption='Uploaded Document', use_column_width=True)

    # Configure the request
    request = types.ProcessRequest(
        name=name,
        raw_document=types.RawDocument(
            content=image_content,
            mime_type="image/jpeg"
        )
    )

    # Measure the response time
    response_start_time = time.time()
    result = client.process_document(request=request)
    response_end_time = time.time()

    # Display the text from the document
    document = result.document
    document_text = "Document text: " + document.text
    
    # Calculate and display metrics
    end_time = time.time()
    
    total_time = end_time - start_time
    response_time = response_end_time - response_start_time

    # Create a dictionary with all information
    info = {
        "Document Text": document.text,
        "Total Execution Time (seconds)": round(total_time, 2),
        "API Response Time (seconds)": round(response_time, 2)
    }
    
    # Convert dictionary to JSON and format it nicely
    info_json = json.dumps(info, indent=4)
    
    # Display JSON formatted information
    st.json(info_json)
