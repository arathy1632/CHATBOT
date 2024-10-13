from fastapi import FastAPI, UploadFile, File, Form
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import CSVLoader
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from pydantic import BaseModel
import os
import pandas as pd
import tempfile

app = FastAPI()

# Request and Response models for FastAPI
class ChatRequest(BaseModel):
    query: str
    chat_history: list  # List of previous interactions ([(question, answer)])

class ChatResponse(BaseModel):
    answer: str
    chat_history: list  # Updated list of interactions ([(question, answer)])

# Function to load the dataset based on file type (CSV or Excel)
def load_dataset(file_path, file_extension):
    if file_extension == '.csv':
        loader = CSVLoader(file_path=file_path, encoding="latin")
        data = loader.load()
    elif file_extension in ['.xlsx', '.xls']:
        df = pd.read_excel(file_path)
        # Convert DataFrame to list of Document objects
        data = [Document(page_content=str(row)) for _, row in df.iterrows()]
    else:
        data = []
    return data

# Initialize global objects for embeddings, vector store, and chain
embeddings = None
vectors_1 = None
vectors_2 = None
chain_1 = None
chain_2 = None
# Define a custom prompt template
basePrompt = """
    You are an intelligent assistant. Your primary goal is to provide accurate and specific answers based strictly on the provided dataset.
    Do not guess or generate information that is not supported by the data. If the dataset does not contain the answer, reply with "I don't know."

    For questions requiring calculations, perform the necessary operations based on the data and provide the exact answer. Always base your answers on the context and data available.

    Ensure that the responses are clear, concise, and directly address the question asked. 

    Dataset context:
    {context}
    
    Question: {question}
    Answer:
"""
PROMPT = PromptTemplate(template=basePrompt, input_variables=["context", "question"])

@app.post("/uploadfile/")
async def upload_file(file: UploadFile, api_key: str = Form(...)):
    os.environ["OPENAI_API_KEY"] = api_key

    # Create a temporary file for the dataset
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(file.file.read())
        tmp_file_path = tmp_file.name

    # Determine file extension
    file_extension = os.path.splitext(file.filename)[1]
    
    # Load data from the file
    global embeddings, vectors_1, chain_1
    data = load_dataset(tmp_file_path, file_extension)
    
    # Initialize embeddings and vector store
    embeddings = OpenAIEmbeddings()
    vectors_1 = FAISS.from_documents(data, embeddings)
    
    # Initialize the conversational retrieval chain for the dataset
    chain_1 = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(temperature=0.0, model_name='gpt-3.5-turbo'),
        retriever=vectors_1.as_retriever(),
        combine_docs_chain_kwargs={"prompt": PROMPT},
    )
   
    return {"status": "File uploaded and processed successfully"}

@app.post("/uploadfiles/")
async def upload_files(file1: UploadFile, file2: UploadFile, api_key: str = Form(...)):
    os.environ["OPENAI_API_KEY"] = api_key

    # Create temporary files for both datasets
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file1, tempfile.NamedTemporaryFile(delete=False) as tmp_file2:
        tmp_file1.write(file1.file.read())
        tmp_file2.write(file2.file.read())
        tmp_file_path1 = tmp_file1.name
        tmp_file_path2 = tmp_file2.name

    # Determine file extensions
    file_extension1 = os.path.splitext(file1.filename)[1]
    file_extension2 = os.path.splitext(file2.filename)[1]
    
    # Load data from both files
    global embeddings, vectors_1, vectors_2, chain_1, chain_2
    data1 = load_dataset(tmp_file_path1, file_extension1)
    data2 = load_dataset(tmp_file_path2, file_extension2)
    
    # Initialize embeddings and vector stores for both datasets
    embeddings = OpenAIEmbeddings()
    vectors_1 = FAISS.from_documents(data1, embeddings)
    vectors_2 = FAISS.from_documents(data2, embeddings)
    
    # Initialize the conversational retrieval chains for both datasets
    chain_1 = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(temperature=0.0, model_name='gpt-3.5-turbo'),
        retriever=vectors_1.as_retriever(),
        combine_docs_chain_kwargs={"prompt": PROMPT},
    )
    
    chain_2 = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(temperature=0.0, model_name='gpt-3.5-turbo'),
        retriever=vectors_2.as_retriever(),
        combine_docs_chain_kwargs={"prompt": PROMPT},
    )

    return {"status": "Files uploaded and processed successfully"}

@app.post("/chat/", response_model=ChatResponse)
async def chat_with_data(request: ChatRequest):
    global chain_1
    if not chain_1:
        return {"error": "File not uploaded"}
    
    chat_history = [(entry[0], entry[1]) for entry in request.chat_history]  # Ensure proper format
    result = chain_1({"question": request.query, "chat_history": chat_history})

    # Append the new interaction to chat history
    updated_history = request.chat_history + [(request.query, result["answer"])]
    
    return ChatResponse(answer=result["answer"], chat_history=updated_history)

@app.post("/compare/", response_model=ChatResponse)
async def compare_datasets(request: ChatRequest):
    global chain_1, chain_2
    if not chain_1 or not chain_2:
        return {"error": "Files not uploaded"}
    
    chat_history = [(entry[0], entry[1]) for entry in request.chat_history]  # Ensure proper format

    # Get response from both datasets
    result_1 = chain_1({"question": request.query, "chat_history": chat_history})
    result_2 = chain_2({"question": request.query, "chat_history": chat_history})

    # Combine the answers from both datasets
    final_answer = f"Dataset 1: {result_1['answer']}\nDataset 2: {result_2['answer']}"

    # Append the new interaction to chat history
    updated_history = request.chat_history + [(request.query, final_answer)]
    
    return ChatResponse(answer=final_answer, chat_history=updated_history)
