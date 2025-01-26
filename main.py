import os
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv, find_dotenv

# Load environment variables
load_dotenv(find_dotenv())

# Step 1: Setup LLM (Mistral with HuggingFace)
HF_TOKEN = os.environ.get("HF_TOKEN")
HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"

def load_llm(huggingface_repo_id):
    """
    Load the LLM using HuggingFaceEndpoint.
    """
    llm = HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        temperature=0.5,
        model_kwargs={"token": HF_TOKEN, "max_length": 512}
    )
    return llm

# Step 2: Setup Prompt Template
CUSTOM_PROMPT_TEMPLATE = """
Use the pieces of information provided in the context to answer the user's question.
If you don't know the answer, just say that you don't know. Do not attempt to fabricate an answer.
Do not provide information outside of the given context.

Context: {context}
Question: {question}

Start the answer directly. No small talk, please.
"""

def set_custom_prompt(custom_prompt_template):
    """
    Create a custom prompt template.
    """
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt

# Step 3: Load the FAISS Database
DB_FAISS_PATH = "vectorstore/db_faiss"
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

# Step 4: Initialize Memory for Conversation
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Step 5: Create Conversational QA Chain
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=load_llm(HUGGINGFACE_REPO_ID),
    retriever=db.as_retriever(search_kwargs={'k': 3}),
    memory=memory
)

# Now invoke with a single query
while True:
    user_query = input("Write Query Here (or type 'exit' to quit): ")
    if user_query.lower() == "exit":
        print("Exiting the conversation.")
        break

    # Invoke the chain
    response = qa_chain.invoke({'question': user_query})

    # Debug the full response
    print("\nDEBUG RESPONSE: ", response)

    # Extract the 'answer' key
    if 'answer' in response:
        print("\nRESULT: ", response['answer'])
    else:
        print("\nNo 'answer' found in the response. Full response:", response)
