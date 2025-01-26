import os
import pickle
import gdown
import streamlit as st
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint
from langchain.memory import ConversationBufferMemory
from tenacity import retry, wait_fixed, stop_after_attempt, RetryError
import time

# Google Drive file IDs for FAISS files
FAISS_FILE_ID = "1y0imcFyEqerDntQY8mMAcEa-W3iauYNN"  # Replace with your actual file ID
PKL_FILE_ID = "1k6lCoW7gmuhIj32m5pVi-3d62zVStPIr"  # Replace with your actual file ID

# Path to store files after download
DB_FAISS_PATH = "vectorstore/db_faiss"

# Download files from Google Drive
def download_from_drive(file_id, destination_path):
    try:
        print(f"Downloading file with ID: {file_id}")
        gdown.download(f'https://drive.google.com/uc?export=download&id={file_id}', destination_path, quiet=False)
        print(f"File downloaded to {destination_path}")
    except Exception as e:
        print(f"Failed to download file: {e}")

@st.cache_resource
def get_vectorstore():
    # Ensure FAISS files are downloaded
    if not os.path.exists(DB_FAISS_PATH):
        os.makedirs(DB_FAISS_PATH)

    # Download the FAISS index and pickle files
    print("Downloading FAISS files...")
    download_from_drive(FAISS_FILE_ID, os.path.join(DB_FAISS_PATH, "index.faiss"))
    download_from_drive(PKL_FILE_ID, os.path.join(DB_FAISS_PATH, "index.pkl"))

    # Check if files exist
    if os.path.exists(os.path.join(DB_FAISS_PATH, "index.faiss")) and os.path.exists(os.path.join(DB_FAISS_PATH, "index.pkl")):
        print("FAISS files are ready for loading.")
    else:
        print("FAISS files are missing!")

    # Load FAISS index
    embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db

def set_custom_prompt(custom_prompt_template):
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt

def load_llm(huggingface_repo_id, HF_TOKEN):
    llm = HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        temperature=0.5,
        model_kwargs={"token": HF_TOKEN, "max_length": 512}
    )
    return llm

# Retry logic for handling 429 rate-limit errors
@retry(wait=wait_fixed(10), stop=stop_after_attempt(5))
def query_llm(qa_chain, prompt):
    response = qa_chain.invoke({'question': prompt})
    return response

def main():
    # Set page configuration
    st.set_page_config(
        page_title="Smart Chat Assistant",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Custom CSS for styling
    st.markdown("""
    <style>
    .main {
        background-color: #f0f2f6;
    }
    .stChatMessage {
        margin-bottom: 15px;
        border-radius: 10px;
        padding: 10px;
    }
    .stChatMessage.user {
        background-color: #e6f2ff;
    }
    .stChatMessage.assistant {
        background-color: #f0f0f0;
    }
    .stTextInput > div > div > input {
        border-radius: 10px;
        border: 2px solid #4a90e2;
    }
    </style>
    """, unsafe_allow_html=True)

    # Title and Welcome Section
    st.title("🤖 DOCPLUS ASSIST")
    st.markdown("### Ask anything about medical topics, diseases, treatments, or health - related doubts, and get reliable insights.")

    # Sidebar for additional information
    with st.sidebar:
        st.header("About")
        st.warning("""**Important Notice:**  
        This Health Care Assist Bot is trained on Harrison's Principles of Internal Medicine .  
        While it provides helpful information, it is not a substitute for professional medical advice, diagnosis, or treatment.  
        Always consult a qualified healthcare provider for medical concerns.""")

    # Initialize session state for messages and memory
    if 'messages' not in st.session_state:
        st.session_state.messages = []
        # Add welcome message
        welcome_msg = "Hello! Welcome to Docplus Assist. How can I help you today?."
        st.session_state.messages.append({'role': 'assistant', 'content': welcome_msg})

    if 'memory' not in st.session_state:
        st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    # Display chat history
    for message in st.session_state.messages:
        st.chat_message(message['role']).markdown(message['content'])

    # Custom prompt template
    CUSTOM_PROMPT_TEMPLATE = """
    Use the pieces of information provided in the context to answer the user's question. 
    When the answer involves multiple aspects (e.g., causes, symptoms, steps, or benefits), 
    format it as a list of points for better understanding. 
    If you don't know the answer, just say that you don't know, and don't try to make up an answer. 
    Do not provide anything outside of the given context. 

    Context: {context} 
    Question: {question} 
    Provide the answer clearly and concisely: 
    1. Use bullet points or numbers for answers with multiple aspects. 
    2. For short answers, respond directly in a sentence.
    """

    # HuggingFace model details
    HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"
    HF_TOKEN = os.environ.get("HF_TOKEN")

    try:
        vectorstore = get_vectorstore()
        if vectorstore is None:
            st.error("Failed to load the vector store")
            return

        # Create Conversational Retrieval Chain
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=load_llm(huggingface_repo_id=HUGGINGFACE_REPO_ID, HF_TOKEN=HF_TOKEN),
            retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
            memory=st.session_state.memory
        )

        # Input for the user's prompt
        prompt = st.chat_input("Ask me anything...")

        if prompt:
            # Display user message
            st.chat_message('user').markdown(prompt)
            st.session_state.messages.append({'role': 'user', 'content': prompt})

            try:
                # Attempt to invoke the chain with retry logic
                response = query_llm(qa_chain, prompt)

                # Extract the answer
                result = response.get("answer", "No answer found.")

                # Display the assistant's response
                st.chat_message('assistant').markdown(result)
                st.session_state.messages.append({'role': 'assistant', 'content': result})

            except RetryError as e:
                st.error("Rate limit exceeded. Please try again later.")
                print(f"Retry failed: {e}")

    except Exception as e:
        st.error(f"Error: {str(e)}")


if __name__ == "__main__":
    main()
