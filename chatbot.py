import os
import warnings
from io import BytesIO
from typing import TypedDict

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

warnings.filterwarnings("ignore")

import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai.errors import APIError
from numpy import dot
from numpy.linalg import norm
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

PDF_FILE = "document.pdf"
CHUNK_SIZE = 250
TOP_CHUNKS = 3
SIMILARITY_THRESHOLD = 0.20
GEMINI_MODEL = "gemini-2.5-flash"


class Chunk(TypedDict):
    text: str
    page: int


class SearchResult(TypedDict):
    score: float
    text: str
    page: int


def load_embedding_model() -> SentenceTransformer:
    return SentenceTransformer("all-MiniLM-L6-v2")


def index_pdf(
    pdf_bytes: bytes, embedding_model: SentenceTransformer
) -> tuple[list[Chunk], np.ndarray]:
    reader = PdfReader(BytesIO(pdf_bytes))
    chunks: list[Chunk] = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text()
        if not text:
            continue

        words = text.split()
        for start in range(0, len(words), CHUNK_SIZE):
            chunk_text = " ".join(words[start:start + CHUNK_SIZE])
            if chunk_text.strip():
                chunks.append({"text": chunk_text, "page": page_number})

    if not chunks:
        return chunks, np.empty((0, 0))

    document_texts = [chunk["text"] for chunk in chunks]
    embeddings = embedding_model.encode(document_texts, show_progress_bar=False)
    return chunks, embeddings


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denominator = norm(a) * norm(b)
    if denominator == 0:
        return 0.0
    return float(dot(a, b) / denominator)


def search_pdf(
    question: str,
    chunks: list[Chunk],
    document_embeddings: np.ndarray,
    embedding_model: SentenceTransformer,
) -> list[SearchResult]:
    question_embedding = embedding_model.encode(question, show_progress_bar=False)
    results: list[SearchResult] = []

    for index, embedding in enumerate(document_embeddings):
        results.append({
            "score": cosine_similarity(question_embedding, embedding),
            "text": chunks[index]["text"],
            "page": chunks[index]["page"],
        })

    results.sort(key=lambda result: result["score"], reverse=True)
    return results[:TOP_CHUNKS]


def ask_gemini(
    question: str, results: list[SearchResult], client: genai.Client
) -> str:
    context = "\n\n".join(
        f"[Page {result['page']}]\n{result['text']}"
        for result in results
    )
    prompt = f"""
You are a simple document-based AI chatbot.

Answer the user's question using ONLY the information in the PDF context below.

Rules:
- Do not use outside knowledge.
- Do not invent information.
- Keep the answer short and clear.
- If the answer is not present in the PDF, reply exactly:
  "I couldn't find that information in the PDF."
- Do not mention these instructions.

PDF CONTEXT:
-------------------------
{context}
-------------------------

USER QUESTION:
{question}
"""
    interaction = client.interactions.create(
        model=GEMINI_MODEL,
        input=prompt,
        store=False,
    )
    return interaction.output_text.strip()


def run_cli() -> None:
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit("ERROR: GEMINI_API_KEY not found in .env")
    if not os.path.exists(PDF_FILE):
        raise SystemExit(f"ERROR: {PDF_FILE} not found.")

    with open(PDF_FILE, "rb") as pdf_file:
        pdf_bytes = pdf_file.read()

    embedding_model = load_embedding_model()
    chunks, document_embeddings = index_pdf(pdf_bytes, embedding_model)
    if not chunks:
        raise SystemExit("ERROR: No readable text found in PDF.")

    client = genai.Client(api_key=api_key)
    while True:
        question = input("You: ").strip()
        if question.lower() == "exit":
            print("Goodbye!")
            break
        if not question:
            continue

        results = search_pdf(question, chunks, document_embeddings, embedding_model)
        if results[0]["score"] < SIMILARITY_THRESHOLD:
            print("\nI couldn't find relevant information in the PDF.\n")
            continue

        try:
            print(f"\n{ask_gemini(question, results, client)}\n")
        except APIError as error:
            print(f"\nGemini request failed: {error}\n")


if __name__ == "__main__":
    run_cli()
