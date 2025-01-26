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
from time import sleep
from tenacity import retry, stop_after_attempt, wait_exponential  # For retry logic

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
    if not os.path.exists(DB_FAISS_PATH):
        os.makedirs(DB_FAISS_PATH)
    download_from_drive(FAISS_FILE_ID, os.path.join(DB_FAISS_PATH, "index.faiss"))
    download_from_drive(PKL_FILE_ID, os.path.join(DB_FAISS_PATH, "index.pkl"))
    embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    return FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

def set_custom_prompt(custom_prompt_template):
    return PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])

@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=2, max=10), reraise=True)
def load_llm(huggingface_repo_id, HF_TOKEN):
    return HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        temperature=0.5,
        model_kwargs={"token": HF_TOKEN, "max_length": 512}
    )

def main():
    st.set_page_config(
        page_title="Smart Chat Assistant",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.title("🤖 DOCPLUS ASSIST")
    st.markdown("### Ask anything about medical topics, diseases, treatments, or health-related doubts, and get reliable insights.")

    with st.sidebar:
        st.header("About")
        st.warning("""
        **Important Notice:**  
        This Health Care Assist Bot is trained on Harrison's Principles of Internal Medicine.  
        Always consult a qualified healthcare provider for medical concerns.
        """)

    if 'messages' not in st.session_state:
        st.session_state.messages = [{'role': 'assistant', 'content': "Hello! Welcome to Docplus Assist. How can I help you today?"}]
    if 'memory' not in st.session_state:
        st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    for message in st.session_state.messages:
        st.chat_message(message['role']).markdown(message['content'])

    CUSTOM_PROMPT_TEMPLATE = """
    Use the pieces of information provided in the context to answer the user's question.
    Context: {context}
    Question: {question}
    Provide the answer clearly and concisely.
    """
    HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"
    HF_TOKEN = os.environ.get("HF_TOKEN")

    try:
        vectorstore = get_vectorstore()
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=load_llm(huggingface_repo_id=HUGGINGFACE_REPO_ID, HF_TOKEN=HF_TOKEN),
            retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
            memory=st.session_state.memory
        )

        prompt = st.chat_input("Ask me anything...")
        if prompt:
            st.chat_message('user').markdown(prompt)
            st.session_state.messages.append({'role': 'user', 'content': prompt})

            try:
                response = qa_chain.invoke({'question': prompt})
                result = response.get("answer", "No answer found.")
            except Exception as e:
                if "429" in str(e):
                    result = "Rate limit exceeded. Please wait and try again."
                else:
                    result = f"An error occurred: {str(e)}"

            st.chat_message('assistant').markdown(result)
            st.session_state.messages.append({'role': 'assistant', 'content': result})

    except Exception as e:
        st.error(f"Error: {str(e)}")


if __name__ == "__main__":
    main()
