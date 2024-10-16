import streamlit as st
from streamlit_chat import message
import requests
import pandas as pd  # Import pandas to load and display the dataset

# Sidebar for API Key input
user_api_key = st.sidebar.text_input(
    label="#### Your OpenAI API key 👇",
    placeholder="Paste your OpenAI API key",
    type="password"
)

# Select between Chat with your dataset or Compare Datasets
mode = st.sidebar.selectbox(
    "Select Mode:",
    ("Chat with your dataset", "Compare Datasets")
)

# API endpoints for backend
BACKEND_URL = "https://chatbot-mwsb.onrender.com"

# Function to reset session state for new files or new mode
def reset_chat_history():
    st.session_state['history'] = []  # Reset history when a new file is uploaded
    st.session_state['generated'] = ["Hello! Ask me anything about the dataset 🤗"]
    st.session_state['past'] = ["Hey! 👋"]

# Function to upload single file to backend
def upload_single_file_to_backend(file, api_key):
    files = {"file": (file.name, file.getvalue())}
    with st.spinner("Uploading and processing the dataset..."):  # Loader for file uploading
        response = requests.post(
            f"{BACKEND_URL}/uploadfile/",
            files=files,
            data={"api_key": api_key}
        )
    return response.json()

# Function to upload two files to backend for comparison
def upload_files_to_backend(file1, file2, api_key):
    files = {
        "file1": (file1.name, file1.getvalue()),
        "file2": (file2.name, file2.getvalue())
    }
    with st.spinner("Uploading and processing both datasets..."):  # Loader for file uploading
        response = requests.post(
            f"{BACKEND_URL}/uploadfiles/",
            files=files,
            data={"api_key": api_key}
        )
    return response.json()

# Function to send chat request to backend
def conversational_chat(query):
    payload = {
        "query": query,
        "chat_history": st.session_state['history']  # Pass current chat history
    }
    with st.spinner("Processing query..."):  # Loader when executing the query
        response = requests.post(f"{BACKEND_URL}/chat/", json=payload)
    result = response.json()

    # Update the chat history in session state
    st.session_state['history'] = result['chat_history']  # Replace with updated history from backend
    
    return result['answer']

# Function to handle dataset comparison chat
def compare_datasets(query):
    payload = {
        "query": query,
        "chat_history": st.session_state['history']  # Pass current chat history
    }
    with st.spinner("Comparing datasets..."):  # Loader when executing comparison
        response = requests.post(f"{BACKEND_URL}/compare/", json=payload)
    result = response.json()

    # Update the chat history in session state
    st.session_state['history'] = result['chat_history']  # Replace with updated history from backend
    
    return result['answer']

# Function to clear chat messages and reset memory
def clear_chat():
    reset_chat_history()
    st.session_state['past'] = []
    st.session_state['generated'] = []

# Chat with your dataset Mode
if mode == "Chat with your dataset":
    st.title("Chat with your dataset")
    
    # File uploader for a single dataset
    uploaded_file = st.sidebar.file_uploader("Upload a dataset", type=["xlsx", "csv"])

    if uploaded_file:
        if 'file_name' not in st.session_state or st.session_state['file_name'] != uploaded_file.name:
            # Upload file to FastAPI backend
            response = upload_single_file_to_backend(uploaded_file, user_api_key)
            if response.get("status") == "File uploaded and processed successfully":
                reset_chat_history()
                st.session_state['file_name'] = uploaded_file.name
                # Load dataset into a pandas DataFrame
                if uploaded_file.name.endswith('.csv'):
                    st.session_state['data'] = pd.read_csv(uploaded_file)
                else:
                    st.session_state['data'] = pd.read_excel(uploaded_file)
            else:
                st.error("Failed to upload file to the backend.")

        # Show dataset in an expandable DataGrid
        with st.expander("View Dataset"):
            st.dataframe(st.session_state['data'], use_container_width=True)

        # Container for the chat history
        response_container = st.container()

        # Container for the user's text input
        container = st.container()

        with container:
            with st.form(key='my_form', clear_on_submit=True):
                user_input = st.text_input("Query:", placeholder="Ask questions about your data", key='input')
                submit_button = st.form_submit_button(label='Send')

            if submit_button and user_input:
                output = conversational_chat(user_input)
                st.session_state['past'].append(user_input)
                st.session_state['generated'].append(output)

        # Display chat history
        if st.session_state['generated']:
            with response_container:
                for i in range(len(st.session_state['generated'])):
                    message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="adventurer")
                    message(st.session_state["generated"][i], key=str(i), avatar_style="bottts")

        # Add a button to clear chat history
        if st.button("Clear Chat"):
            clear_chat()

# Compare Datasets Mode
elif mode == "Compare Datasets":
    st.title("Compare Datasets")

    # File uploader for two datasets
    uploaded_file1 = st.sidebar.file_uploader("Upload first dataset", type=["xlsx", "csv"])
    uploaded_file2 = st.sidebar.file_uploader("Upload second dataset", type=["xlsx", "csv"])

    if uploaded_file1 and uploaded_file2:
        if 'file_name1' not in st.session_state or st.session_state['file_name1'] != uploaded_file1.name:
            # Upload both files to FastAPI backend for comparison
            response = upload_files_to_backend(uploaded_file1, uploaded_file2, user_api_key)
            if response.get("status") == "Files uploaded and processed successfully":
                reset_chat_history()
                st.session_state['file_name1'] = uploaded_file1.name
                st.session_state['file_name2'] = uploaded_file2.name
                # Load both datasets into pandas DataFrames
                if uploaded_file1.name.endswith('.csv'):
                    st.session_state['data1'] = pd.read_csv(uploaded_file1)
                else:
                    st.session_state['data1'] = pd.read_excel(uploaded_file1)
                if uploaded_file2.name.endswith('.csv'):
                    st.session_state['data2'] = pd.read_csv(uploaded_file2)
                else:
                    st.session_state['data2'] = pd.read_excel(uploaded_file2)
            else:
                st.error("Failed to upload files to the backend.")

        # Show both datasets in expandable DataGrids
        with st.expander("View Dataset 1"):
            st.dataframe(st.session_state['data1'], use_container_width=True)

        with st.expander("View Dataset 2"):
            st.dataframe(st.session_state['data2'], use_container_width=True)

        # Container for the comparison chat history
        response_container = st.container()

        # Container for the user's text input for comparison
        container = st.container()

        with container:
            with st.form(key='compare_form', clear_on_submit=True):
                compare_query = st.text_input("Query for comparison:", placeholder="Ask questions to compare the datasets", key='compare_input')
                compare_submit_button = st.form_submit_button(label='Compare')

            if compare_submit_button and compare_query:
                compare_output = compare_datasets(compare_query)
                st.session_state['past'].append(compare_query)
                st.session_state['generated'].append(compare_output)

        # Display chat history for comparison
        if st.session_state['generated']:
            with response_container:
                for i in range(len(st.session_state['generated'])):
                    message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="adventurer")
                    message(st.session_state["generated"][i], key=str(i), avatar_style="bottts")

        # Add a button to clear comparison chat history
        if st.button("Clear Chat"):
            clear_chat()
