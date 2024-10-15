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

**Backend (FastAPI):** https://chatbot-mwsb.onrender.com

**Frontend (Streamlit):** https://chatbot-fastapi.streamlit.app/
