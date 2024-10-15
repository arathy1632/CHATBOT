 # **Chatbot with FastAPI and Streamlit**

This project is a chatbot application designed to provide users with an intuitive interface to interact with datasets using natural language. The application leverages FastAPI for the backend processing and Streamlit for the frontend interface, offering users a seamless way to upload datasets, query them, and even compare two datasets interactively. The combination of cutting-edge technologies ensures that users can engage with their data in a conversational and visual manner.

## **Features**

**Dataset Uploading:** Users can upload one or multiple datasets in various formats (e.g., CSV or Excel files).

**Conversational Interaction:** Ask questions and query uploaded datasets in natural language through a chatbot interface. The backend processes these queries and returns the relevant results or visualizations.

**Dataset Comparison:** Compare two datasets side-by-side to get insights or statistical comparisons based on user-defined queries.

**Streamlit Frontend:** An easy-to-use, interactive web interface that allows users to upload datasets, ask questions, and view results with visual plots.

**FastAPI Backend:** A powerful, fast backend that handles data processing, queries, and manages communication between the frontend and the conversational model.

## **Technologies Used**

**Frontend:** Streamlit (for building a user-friendly web interface)

**Backend:**

FastAPI (for handling API requests)

Uvicorn (for running the FastAPI server)

Retrieval Augmented Generation (RAG):

Utilizes OpenAI GPT-3.5-Turbo to process natural language queries and generate meaningful responses based on the data.

Vector Store: FAISS is used for storing and searching vectorized text embeddings, allowing for efficient information retrieval from the uploaded datasets.

## **Data Processing and Visualization:**

**Pandas:** For handling and manipulating the uploaded datasets.

**PandasAI:** A wrapper around Pandas, used for executing complex queries, calculations, and plotting charts based on natural language questions. This simplifies interaction and analysis by allowing users to ask for plots and calculations directly.


## **Workflow Overview**

**Data Upload:** Users upload datasets via the Streamlit interface.

**Conversational Querying:** Users ask questions in natural language. The backend, powered by FastAPI, processes these queries, with the OpenAI model interpreting them.

**Data Processing:** The PandasAI integration facilitates query execution and plots based on user questions, whether they relate to data filtering, comparison, or statistical operations.

**Visualization:** Streamlit displays any results, plots, or comparisons generated based on user queries.

## **Live Demo**

**Backend (FastAPI):** Deployed on Render: https://chatbot-mwsb.onrender.com

**Frontend (Streamlit):**Deployed in Streamlit Cloud: https://chatbot-fastapi.streamlit.app/

## **How it looks like**

### **Chat with your dataset**

Give your openai api key and dataset in the sidebar 

**1. Chat Interface**

<img width="955" alt="{E34BAC56-2A8D-48FF-BC69-7BE8ED1EC2F4}" src="https://github.com/user-attachments/assets/1483dd56-e108-4317-a8f6-1948370d7602">

**2. Visualization**
   
<img width="958" alt="{8077EE3E-519F-4314-BF0F-F257E7485769}" src="https://github.com/user-attachments/assets/d1265b86-589a-4815-b614-b68d2fc75d8d">

**3. Comparison**

<img width="953" alt="{913E86E4-7B98-4CCB-9592-ADC8293541DE}" src="https://github.com/user-attachments/assets/ccb2268d-9397-4b5e-965b-ae6221b94660">




