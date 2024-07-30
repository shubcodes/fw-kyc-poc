import streamlit as st
from google.cloud import documentai_v1beta3 as documentai
from google.cloud.documentai_v1beta3 import types
import os
import requests
import json
import re
import time
from fireworks.client import Fireworks
from pydantic import BaseModel, Field


# Set the environment variable to the absolute path of the key file
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "gcp.json"

# Initialize the client
client = documentai.DocumentProcessorServiceClient()

# Define the processor details
project_id = ''
location = ''
processor_id = ''
processor_version_id = ""  # Optional. Processor version to use

# The full resource name of the processor
processor_name = client.processor_path(project_id, location, processor_id)

# Initialize Fireworks client
fireworks_api_key = ""  # Replace with your Fireworks API key
fireworks_client = Fireworks(api_key=fireworks_api_key)

# Streamlit app
st.title("Loan Processing Application")

# Form for user inputs
with st.form("loan_application_form"):
    user_name = st.text_input("Name")
    ssn = st.text_input("SSN")
    address = st.text_input("Address")
    dob = st.date_input("Date of Birth", format="MM/DD/YYYY")
    salary = st.number_input("Salary", min_value=0.0)
    loan_amount = st.number_input("Requested Loan Amount", min_value=0.0)
    
    st.write("Upload proof of ID")
    id_proofs = st.file_uploader("Choose files", type=["jpg", "jpeg", "png", "pdf"], accept_multiple_files=True, key="id_proofs")
    
    st.write("Upload paystubs and bank statements")
    paystubs = st.file_uploader("Choose files", type=["jpg", "jpeg", "png", "pdf"], accept_multiple_files=True, key="paystubs")
    
    submitted = st.form_submit_button("Submit")

if submitted:
    # Collect data
    user_data = {
        "name": user_name,
        "ssn": ssn,
        "address": address,
        "dob": str(dob),
        "salary": salary,
        "loan_amount": loan_amount
    }

    # Save user data to session state
    st.session_state['user_data'] = user_data

    # Process uploaded documents with Google Document AI and Fireworks API
    def process_id_documents(uploaded_files):
        results = []
        for uploaded_file in uploaded_files:
            st.write(f"Processing ID document: {uploaded_file.name}")
            image_content = uploaded_file.read()

            # Measure time for Google Document AI
            start_time_gcp = time.time()

            # Configure the request
            request = types.ProcessRequest(
                name=processor_name,
                raw_document=types.RawDocument(
                    content=image_content,
                    mime_type="application/pdf" if uploaded_file.type == "application/pdf" else "image/jpeg"
                )
            )

            # Process the document
            try:
                result = client.process_document(request=request)
                st.write("GCP API call successful")
            except Exception as e:
                st.error(f"GCP API call failed: {e}")
                continue

            # Calculate time taken for GCP API
            gcp_api_time = time.time() - start_time_gcp

            # Extract the text from the document
            document_text = result.document.text
            st.write(f"Extracted text: {document_text}")

            # Measure time for Fireworks API
            start_time_fw = time.time()

            # Prepare the payload for Fireworks API for ID documents
            payload_id = {
                "model": "accounts/fireworks/models/llama-v3p1-8b-instruct",
                "max_tokens": 8192,
                "response_format": {"type": "json_object"},
                "messages": [{
                    "role": "user",
                    "content": (
                        f"From the following text dump from a user's identifying documentation please parse all of the user data and provide it in a json format. \n\n"
                        f"Document text: {document_text}. As a strict identification check robot you must make sure to validate that all pieces of information correctly match what is expected of licenses and passports in the United States. Respond with a JSON object containing the parsed data.  "
                        f"Analyze the following JSON data, which represents an OCR extract from an identification document. Extract all relevant identification information including but not limited to the following:\n"
                        f"For Identification Documents (IDs):\n"
                        f"Full name\n"
                        f"Date of birth\n"
                        f"Address\n"
                        f"Document number\n"
                        f"Issuing authority\n"
                        f"Issue date\n"
                        f"Expiry date\n"
                        f"Nationality\n"
                        f"Any other relevant details (e.g., gender, photo, signature)\n"
                        f"For Passports:\n"
                        f"Full name\n"
                        f"Passport number\n"
                        f"Nationality\n"
                        f"Date of birth\n"
                        f"Place of birth\n"
                        f"Gender\n"
                        f"Issue date\n"
                        f"Expiry date\n"
                        f"Issuing country\n"
                        f"Any other relevant details (e.g., photo, signature, visa information)\n"
                        f"Provide the extracted information in a structured format."
                    )
                }],
            }

            st.write(f"Fireworks ID payload: {json.dumps(payload_id, indent=2)}")

            try:
                response = fireworks_client.chat.completions.create(**payload_id)
                st.write("Fireworks API call for ID documents successful")
            except Exception as e:
                st.error(f"Fireworks API call for ID documents failed: {e}")
                continue

            # Calculate time taken for Fireworks API
            fw_api_time = time.time() - start_time_fw

            # Extract the response content
            content = response.choices[0].message.content
            st.write(f"Fireworks ID response content: {content}")

            results.append({
                "document_name": uploaded_file.name,
                "gcp_result": document_text,
                "fireworks_result": content,
                "gcp_api_time": gcp_api_time,
                "fw_api_time": fw_api_time
            })

        return results

    def process_paystub_documents(uploaded_files):
        results = []
        for uploaded_file in uploaded_files:
            st.write(f"Processing paystub document: {uploaded_file.name}")
            image_content = uploaded_file.read()

            # Measure time for Google Document AI
            start_time_gcp = time.time()

            # Configure the request
            request = types.ProcessRequest(
                name=processor_name,
                raw_document=types.RawDocument(
                    content=image_content,
                    mime_type="application/pdf" if uploaded_file.type == "application/pdf" else "image/jpeg"
                )
            )

            # Process the document
            try:
                result = client.process_document(request=request)
                st.write("GCP API call successful")
            except Exception as e:
                st.error(f"GCP API call failed: {e}")
                continue

            # Calculate time taken for GCP API
            gcp_api_time = time.time() - start_time_gcp

            # Extract the text from the document
            document_text = result.document.text
            st.write(f"Extracted text: {document_text}")

            # Measure time for Fireworks API
            start_time_fw = time.time()

            # Prepare the payload for Fireworks API for paystub documents
            payload_paystub = {
                "model": "accounts/fireworks/models/llama-v3p1-8b-instruct",
                "max_tokens": 8192,
                "response_format": {"type": "json_object"},
                "messages": [{
                    "role": "user",
                    "content": (
                        f"Extract all financial information from the following paystub document text:\n\n"
                        f"Document text: {document_text}. Respond with a JSON object containing the parsed data. "
                        f"Analyze the following JSON data, which represents an OCR extract from a financial document. "
                        f"Extract all relevant financial information including but not limited to the following:\n"
                        f"For Pay Stubs:\n"
                        f"Employee name\n"
                        f"Employer name\n"
                        f"Pay period dates\n"
                        f"Gross income\n"
                        f"Net income\n"
                        f"Deductions (taxes, insurance, etc.)\n"
                        f"Any other financial details (e.g., bonuses, overtime, reimbursements)\n"
                        f"For Banking Documents:\n"
                        f"Account holder name\n"
                        f"Bank name\n"
                        f"Account number\n"
                        f"Transaction dates\n"
                        f"Transaction amounts\n"
                        f"Transaction types (e.g., deposit, withdrawal, transfer)\n"
                        f"Balance information\n"
                        f"Any other relevant financial details\n"
                        f"Provide the extracted information in a structured format."
                    )
                }],
            }

            st.write(f"Fireworks paystub payload: {json.dumps(payload_paystub, indent=2)}")

            try:
                response = fireworks_client.chat.completions.create(**payload_paystub)
                st.write("Fireworks API call for paystub documents successful")
            except Exception as e:
                st.error(f"Fireworks API call for paystub documents failed: {e}")
                continue

            # Calculate time taken for Fireworks API
            fw_api_time = time.time() - start_time_fw

            # Extract the response content
            content = response.choices[0].message.content
            st.write(f"Fireworks paystub response content: {content}")

            results.append({
                "document_name": uploaded_file.name,
                "gcp_result": document_text,
                "fireworks_result": content,
                "gcp_api_time": gcp_api_time,
                "fw_api_time": fw_api_time
            })

        return results

    id_results = []
    if id_proofs:
        id_results = process_id_documents(id_proofs)
        st.session_state['id_results'] = id_results
        for result in id_results:
            st.write(f"Results for {result['document_name']}")
            st.json(result['fireworks_result'])
            st.write(f"GCP Document AI Processing Time: {result['gcp_api_time']:.2f} seconds")
            st.write(f"Fireworks API Processing Time: {result['fw_api_time']:.2f} seconds")

    paystub_results = []
    if paystubs:
        paystub_results = process_paystub_documents(paystubs)
        st.session_state['paystub_results'] = paystub_results
        for result in paystub_results:
            st.write(f"Results for {result['document_name']}")
            st.json(result['fireworks_result'])
            st.write(f"GCP Document AI Processing Time: {result['gcp_api_time']:.2f} seconds")
            st.write(f"Fireworks API Processing Time: {result['fw_api_time']:.2f} seconds")


# Check if we have results in session state
if 'id_results' in st.session_state and 'paystub_results' in st.session_state:
    verify_submitted = st.button("Verify Information")

    if verify_submitted:
        # Combine the extracted data from IDs and paystubs
        all_documents_text = " ".join([result['fireworks_result'] for result in st.session_state['id_results'] + st.session_state['paystub_results']])

        # Prepare the payload for verification with Fireworks API
        verify_payload = {
            "model": "accounts/fireworks/models/llama-v3p1-8b-instruct",
            "max_tokens": 16384,
            "response_format": {"type": "json_object"},
            "messages": [{
                "role": "user",
                "content": (
                    f"Verify that the following information matches across the form, IDs, and paystubs. Form data: {json.dumps(st.session_state['user_data'])}. "
                    f"Document data: {all_documents_text}. Analyze the following JSON data from various identification documents (e.g., driver's license, passport) and financial documents (e.g., pay stubs, banking documents). Compare specific fields to determine if they refer to the same person. "
                    f"The comparison should be based on the following criteria:\n"
                    f"Full Name:\n"
                    f"Identification Documents: Compare Full Name and the combination of First Name, Middle Name, and Last Name.\n"
                    f"Financial Documents: Compare Employee Name or Account Holder Name.\n"
                    f"Date of Birth:\n"
                    f"Identification Documents: Compare Date of Birth.\n"
                    f"Financial Documents: This information is typically not present but should be cross-referenced where possible.\n"
                    f"Address (if provided in any of the documents):\n"
                    f"Identification Documents: Compare Address.\n"
                    f"Financial Documents: Compare Address.\n"
                    f"Social Security Number:\n"
                    f"Identification Documents: Compare Social Security Number if present.\n"
                    f"Financial Documents: This information is typically not present but should be cross-referenced where possible.\n"
                    f"Other Identifiers:\n"
                    f"Identification Documents: Compare fields such as Gender, Document Number, Nationality, Issue Date, and Expiry Date.\n"
                    f"Financial Documents: Compare any relevant fields like Account Number, Employer Name, etc.\n"
                    f"Consistency Across Documents:\n"
                    f"Ensure that the information provided in the financial documents (e.g., employer, income periods, transaction details) is consistent with the personal details in the identification documents.\n"
                    f"Provide a response in a structured JSON format indicating if these documents correspond to the same person based on the above criteria. Ensure to handle slight variations in name formatting (e.g., middle names, initials) and common discrepancies in addresses.\n"
                    f"Determine if these documents refer to the same person. Respond with a JSON object field isSamePerson that states 'true' or 'false'. Do not respond with anything else."
                )
            }],
        }

        st.write(f"Fireworks verification payload: {json.dumps(verify_payload, indent=2)}")

        try:
            verify_response = fireworks_client.chat.completions.create(**verify_payload)
            st.write("Fireworks verification API call successful")
            verify_content = verify_response.choices[0].message.content
            st.write(f"Fireworks verification response content: {verify_content}")
            st.json(json.loads(verify_content))
        except Exception as e:
            st.error(f"Fireworks verification API call failed: {e}")

    calculate_salary_submitted = st.button("Calculate Salary and Net Dollars")

    if calculate_salary_submitted:
        # Combine the extracted data from paystubs and banking documents
        all_paystub_text = " ".join([result['fireworks_result'] for result in st.session_state['paystub_results']])

        # Prepare the payload for calculating net dollars and yearly salary with Fireworks API
        salary_payload = {
            "model": "accounts/fireworks/models/llama-v3p1-8b-instruct",
            "max_tokens": 8192,
            "response_format": {"type": "json_object"},
            "messages": [{
                "role": "user",
                "content": f"Calculate the yearly salary and net dollars from the following paystub and banking document information. Document data: {all_paystub_text}. Respond with a JSON object field that includes the yearly salary and net dollars. Do not respond with anything else.",
            }],
        }
        st.write(f"Fireworks salary calculation payload: {json.dumps(salary_payload, indent=2)}")

        try:
            salary_response = fireworks_client.chat.completions.create(**salary_payload)
            st.write("Fireworks salary calculation API call successful")
            salary_content = salary_response.choices[0].message.content
            st.write(f"Fireworks salary calculation response content: {salary_content}")
            salary_data = json.loads(salary_content)
            st.json(salary_data)
            st.session_state['yearly_salary'] = salary_data.get("Yearly Salary")
            st.session_state['net_dollars'] = salary_data.get("Yearly Net Dollars")

        except Exception as e:
            st.error(f"Fireworks salary calculation API call failed: {e}")

    # Add an option for credit check after verification
    credit_check_submitted = st.button("Perform Credit Check")

    if credit_check_submitted:
        # Define the credit check function schema
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "perform_credit_check",
                    "description": "Perform a credit check given user details.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "ssn": {"type": "string"},
                            "address": {"type": "string"},
                            "dob": {"type": "string"}
                        },
                        "required": ["name", "ssn", "address", "dob"],
                    },
                },
            }
        ]

        # Prepare the payload for the credit check with FireFunction
        credit_check_payload = {
            "model": "accounts/fireworks/models/firefunction-v2",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant with access to functions. Use them if required."},
                {"role": "user", "content": "Perform a credit check for the following user details."},
                {"role": "user", "content": json.dumps({
                    "name": st.session_state['user_data']['name'],
                    "ssn": st.session_state['user_data']['ssn'],
                    "address": st.session_state['user_data']['address'],
                    "dob": st.session_state['user_data']['dob']
                })}
            ],
            "tools": tools,
            "temperature": 0.1
        }
        st.write(f"FireFunction credit check payload: {json.dumps(credit_check_payload, indent=2)}")

        try:
            credit_check_response = fireworks_client.chat.completions.create(**credit_check_payload)
            st.write("FireFunction credit check API call successful")
            
            # Convert the response to a dictionary
            credit_check_response_dict = credit_check_response.model_dump()
            st.write(f"Full FireFunction response: {json.dumps(credit_check_response_dict, indent=2)}")

            # Extract function call details
            if 'tool_calls' in credit_check_response_dict['choices'][0]['message']:
                function_call = credit_check_response_dict['choices'][0]['message']['tool_calls'][0]['function']
                st.write(f"Function call details: {function_call}")

                # Extract arguments from function call
                credit_check_arguments = json.loads(function_call['arguments'])
                st.write(f"Credit check arguments: {credit_check_arguments}")

                # Make the actual credit check API call
                credit_check_api_url = "http://127.0.0.1:5100/credit_check"
                credit_check_api_response = requests.post(credit_check_api_url, json=credit_check_arguments)
                if credit_check_api_response.status_code == 200:
                    credit_check_result = credit_check_api_response.json()
                    st.session_state['credit_data'] = credit_check_result
                    st.write(f"Credit Check Result: {credit_check_result}")
                    st.json(credit_check_result)
                else:
                    st.error(f"Credit Check API call failed with status code: {credit_check_api_response.status_code} and message: {credit_check_api_response.text}")
            else:
                st.error("No tool calls found in the FireFunction response.")

        except Exception as e:
            st.error(f"FireFunction credit check API call failed: {e}")

    # Add an option for loan approval check after salary calculation and credit check
    loan_approval_submitted = st.button("Evaluate Loan Approval")

    if loan_approval_submitted:
        # Define the loan evaluation function schema
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "evaluate_loan",
                    "description": "Evaluate the loan approval based on user details.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "credit_score": {"type": "integer"},
                            "credit_history": {"type": "string"},
                            "yearly_salary": {"type": "number"},
                            #"net_dollars": {"type": "number"},
                            "loan_amount": {"type": "number"}
                        },
                        "required": ["credit_score", "credit_history", "yearly_salary", "loan_amount"],
                    },
                },
            }
        ]

        # Prepare the payload for the loan approval check with FireFunction
        loan_approval_payload = {
            "model": "accounts/fireworks/models/firefunction-v2",
            "messages": [
                #{"role": "system", "content": "You are a helpful assistant with access to functions. Use them if required."},
                {"role": "user", "content": "Evaluate the loan approval for the following user details."},
                {"role": "user", "content": json.dumps({
                    "credit_score": st.session_state['credit_data']['credit_score'],
                    "credit_history": st.session_state['credit_data']['credit_history'],
                    "yearly_salary": st.session_state['yearly_salary'],
                    #"net_dollars": st.session_state['net_dollars'],
                    "loan_amount": st.session_state['user_data']['loan_amount']
                })}
            ],
            "tools": tools,
            "temperature": 0.1
        }
        st.write(f"FireFunction loan approval payload: {json.dumps(loan_approval_payload, indent=2)}")

        try:
            loan_approval_response = fireworks_client.chat.completions.create(**loan_approval_payload)
            st.write("FireFunction loan approval API call successful")
            
            # Convert the response to a dictionary
            loan_approval_response_dict = loan_approval_response.model_dump()
            st.write(f"Full FireFunction response: {json.dumps(loan_approval_response_dict, indent=2)}")

            # Extract function call details
            if 'tool_calls' in loan_approval_response_dict['choices'][0]['message']:
                function_call = loan_approval_response_dict['choices'][0]['message']['tool_calls'][0]['function']
                st.write(f"Function call details: {function_call}")

                # Extract arguments from function call
                loan_approval_arguments = json.loads(function_call['arguments'])
                st.write(f"Loan approval arguments: {loan_approval_arguments}")

                # Make the actual loan approval API call
                loan_approval_api_url = "http://127.0.0.1:5200/evaluate_loan"
                loan_approval_api_response = requests.post(loan_approval_api_url, json=loan_approval_arguments)
                if loan_approval_api_response.status_code == 200:
                    loan_approval_result = loan_approval_api_response.json()
                    st.write(f"Loan Approval Result: {loan_approval_result}")
                    st.json(loan_approval_result)
                else:
                    st.error(f"Loan Approval API call failed with status code: {loan_approval_api_response.status_code} and message: {loan_approval_api_response.text}")
            else:
                st.error("No tool calls found in the FireFunction response.")

        except Exception as e:
            st.error(f"FireFunction loan approval API call failed: {e}")
