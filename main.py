import os
import time
import logging
from dotenv import load_dotenv, find_dotenv
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv(find_dotenv())
HF_TOKEN = os.environ.get("HF_TOKEN")
HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN is not set. Please check your environment variables.")

# Load LLM
def load_llm(huggingface_repo_id):
    llm = HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        task="text2text-generation",
        model_kwargs={
            "headers": {"Authorization": f"Bearer {HF_TOKEN}"},
            "temperature": 0.5,
            "max_length": 512
        },
    )
    return llm

# Custom Prompt Template
CUSTOM_PROMPT_TEMPLATE = """
Use the pieces of information provided in the context to answer the user's question.
If you don't know the answer, just say that you don't know. Do not attempt to fabricate an answer.
Do not provide information outside of the given context.

Context: {context}
Question: {question}

Start the answer directly. No small talk, please.
"""
prompt = PromptTemplate(template=CUSTOM_PROMPT_TEMPLATE, input_variables=["context", "question"])

# FAISS Database
DB_FAISS_PATH = "vectorstore/db_faiss"
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
try:
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
except FileNotFoundError:
    logger.error(f"FAISS database not found at {DB_FAISS_PATH}. Check the file path.")
    exit(1)

# Memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Conversational QA Chain
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=load_llm(HUGGINGFACE_REPO_ID),
    retriever=db.as_retriever(search_kwargs={'k': 3}),
    memory=memory
)

logger.info("Starting Conversational Retrieval Chain...")

# Interactive Loop
while True:
    user_query = input("Write Query Here (or type 'exit' to quit): ")
    if user_query.lower() == "exit":
        print("Exiting the conversation.")
        break

    time.sleep(1)  # Respect API rate limits

    try:
        response = qa_chain.run({'question': user_query})
        print("\nRESULT: ", response)
    except Exception as e:
        logger.error("Error occurred: %s", str(e))
