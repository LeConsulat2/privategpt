import os
from dotenv import load_dotenv
import streamlit as st
from langchain_unstructured import UnstructuredLoader
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain.embeddings import CacheBackedEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.storage import LocalFileStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.callbacks.base import BaseCallbackHandler

# from utils import check_authentication

# Streamlit Page Configuration
st.set_page_config(page_title="PrivateGPT", page_icon="❓")

# User Authentication
# check_authentication()

# Load environment variables (if needed)
# load_dotenv()

# Ensure required credentials are present
# username = os.getenv("username", st.secrets.get("credentials", {}).get("username"))
# password = os.getenv("password", st.secrets.get("credentials", {}).get("password"))

# Stop execution if any credentials are missing
# if not all([username, password]):
#     st.error("Some required environment variables are missing.")
#     st.stop()


# Define callback handler for the LLM
class ChatCallBackHandler(BaseCallbackHandler):
    def __init__(self):
        self.message = ""  # Initialize message as an empty string
        self.message_box = st.empty()  # Placeholder for live updates

    def on_llm_start(self, *args, **kwargs):
        self.message = ""  # Reset message when LLM starts

    def on_llm_end(self, *args, **kwargs):
        save_message(self.message, "ai")  # Save message when LLM ends

    def on_llm_new_token(self, token, *args, **kwargs):
        self.message += token  # Append new tokens to the message
        self.message_box.markdown(self.message)  # Update the message box in real-time


# Initialize the ChatOllama model
llm = ChatOllama(
    model="mistral:latest",
    temperature=0.1,
    streaming=True,
    callbacks=[ChatCallBackHandler()],
)


# Memory class for managing conversational context
class MemoryManager:
    def __init__(self):
        self.context = []  # Initialize context as an empty list

    def update(self, new_context):
        self.context.append(new_context)  # Append new context to the existing list

    def get_full_context(self):
        return "\n\n".join(self.context)  # Join all context entries with newlines


# Initialize memory in session state if not already set
if "memory" not in st.session_state:
    st.session_state.memory = MemoryManager()


@st.cache_data(show_spinner="Embedding file...")
def embed_file(file):
    file_content = file.read()
    file_path = f"./.cache/private_files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)

    cache_dir = LocalFileStore(f"./.cache/private_embeddings/{file.name}")
    text_splitter = CharacterTextSplitter(
        separator="\n\n",
        chunk_size=600,
        chunk_overlap=100,
        length_function=len,
        is_separator_regex=False,
    )

    loader = UnstructuredLoader(file_path)
    docs = loader.load_and_split(text_splitter)

    embeddings = OllamaEmbeddings(model="mistral:latest")
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)

    vectorstore = FAISS.from_documents(docs, cached_embeddings)
    retriever = vectorstore.as_retriever()

    return retriever


# Save message in session state
def save_message(message, role):
    st.session_state.setdefault("messages", []).append(
        {"message": message, "role": role}
    )


# Send a message to the chat interface
def send_message(message, role, save=True):
    st.chat_message(role).markdown(message)  # Display the message
    if save:
        save_message(message, role)  # Optionally save the message


# Render the chat history from session state
def render_chat_history():
    for msg in st.session_state.get("messages", []):
        send_message(msg["message"], msg["role"], save=False)


# Format documents into a single string
def format_documents(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# Define prompt template for the LLM
prompt = PromptTemplate.from_template(
    """
    Answer the question using ONLY the context provided below. If the answer is unknown, respond with 'I don't know'. Do not fabricate information.

    Context: {context}
    Question: {question}
    """
)

# Streamlit UI setup
st.title("DocumentGPT")
st.markdown("Welcome! Use this chatbot to query your uploaded documents.")

with st.sidebar:
    uploaded_file = st.file_uploader(
        "Upload a .txt, .pdf, or .docx file", type=["pdf", "txt", "docx"]
    )

if uploaded_file:
    retriever = embed_file(uploaded_file)
    send_message(
        "Ready to answer your queries based on the uploaded document.", "ai", save=False
    )
    render_chat_history()
    user_input = st.chat_input("Ask anything about your file...")
    if user_input:
        send_message(user_input, "human")
        docs = retriever.similarity_search(user_input)
        formatted_docs = format_documents(docs)
        st.session_state.memory.update(formatted_docs)
        chain_input = {
            "context": st.session_state.memory.get_full_context(),
            "question": user_input,
        }
        chain = prompt | llm
        response = chain.invoke(chain_input)
        send_message(response, "ai")
else:
    st.session_state["messages"] = []  # Reset messages if no file is uploaded
