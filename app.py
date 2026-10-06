import hashlib
import os

import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai.errors import APIError
from pypdf.errors import PdfReadError
from streamlit.errors import StreamlitSecretNotFoundError

from chatbot import (
    SIMILARITY_THRESHOLD,
    ask_gemini,
    index_pdf,
    load_embedding_model,
    search_pdf,
)

load_dotenv()


@st.cache_resource(show_spinner="Loading the document embedding model...")
def get_embedding_model():
    return load_embedding_model()


@st.cache_data(show_spinner="Reading and indexing the PDF...")
def get_document_index(pdf_bytes: bytes):
    return index_pdf(pdf_bytes, get_embedding_model())


def get_api_key():
    api_key = os.getenv("GEMINI_API_KEY")
    if api_key:
        return api_key

    try:
        return st.secrets["GEMINI_API_KEY"]
    except (KeyError, StreamlitSecretNotFoundError):
        return None


st.set_page_config(page_title="PDF Chatbot", page_icon="📄")
st.title("Chat with a PDF")
st.write("Upload a PDF and ask questions about its contents.")

uploaded_pdf = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_pdf is None:
    st.info("Upload a PDF to start chatting.")
    st.stop()

pdf_bytes = uploaded_pdf.getvalue()
document_id = hashlib.sha256(pdf_bytes).hexdigest()
if st.session_state.get("document_id") != document_id:
    st.session_state.document_id = document_id
    st.session_state.messages = []

try:
    chunks, document_embeddings = get_document_index(pdf_bytes)
except PdfReadError as error:
    st.error(f"Could not read this PDF: {error}")
    st.stop()

if not chunks:
    st.error("This PDF contains no readable text.")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.caption(f"Loaded {len(chunks)} text chunks from **{uploaded_pdf.name}**.")
    if st.button("Clear chat"):
        st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

question = st.chat_input("Ask a question about the PDF")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    results = search_pdf(
        question, chunks, document_embeddings, get_embedding_model()
    )
    with st.chat_message("assistant"):
        if results[0]["score"] < SIMILARITY_THRESHOLD:
            answer = "I couldn't find relevant information in the PDF."
            st.markdown(answer)
            st.session_state.messages.append(
                {"role": "assistant", "content": answer}
            )
        else:
            api_key = get_api_key()
            if not api_key:
                st.error(
                    "Set GEMINI_API_KEY in Streamlit app secrets or in a local .env file."
                )
            else:
                try:
                    answer = ask_gemini(
                        question,
                        results,
                        genai.Client(api_key=api_key),
                    )
                    st.markdown(answer)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": answer}
                    )
                except APIError as error:
                    st.error(f"Gemini request failed: {error}")
