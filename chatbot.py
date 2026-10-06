import os
import logging
import warnings

from dotenv import load_dotenv
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from numpy import dot
from numpy.linalg import norm
from google import genai

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)

PDF_FILE = "document.pdf"
CHUNK_SIZE = 250
TOP_CHUNKS = 3
SIMILARITY_THRESHOLD = 0.20
GEMINI_MODEL = "gemini-3.1-flash-lite"

load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    print("ERROR: GEMINI_API_KEY not found in .env")
    raise SystemExit

client = genai.Client(api_key=API_KEY)

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

if not os.path.exists(PDF_FILE):
    print(f"ERROR: {PDF_FILE} not found.")
    raise SystemExit

reader = PdfReader(PDF_FILE)
chunks = []

for page_number, page in enumerate(reader.pages, start=1):
    text = page.extract_text()

    if not text:
        continue

    words = text.split()

    for i in range(0, len(words), CHUNK_SIZE):
        chunk_text = " ".join(words[i:i + CHUNK_SIZE])

        if chunk_text.strip():
            chunks.append({"text": chunk_text, "page": page_number})

if not chunks:
    print("ERROR: No readable text found in PDF.")
    raise SystemExit

document_texts = [chunk["text"] for chunk in chunks]
document_embeddings = embedding_model.encode(document_texts, show_progress_bar=False)


def cosine_similarity(a, b):
    denominator = norm(a) * norm(b)

    if denominator == 0:
        return 0

    return dot(a, b) / denominator


def search_pdf(question):
    question_embedding = embedding_model.encode(question, show_progress_bar=False)
    results = []

    for index, embedding in enumerate(document_embeddings):
        score = cosine_similarity(question_embedding, embedding)
        results.append({
            "score": score,
            "text": chunks[index]["text"],
            "page": chunks[index]["page"]
        })

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:TOP_CHUNKS]


def ask_gemini(question, results):
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
        store=False
    )

    return interaction.output_text.strip()


while True:
    question = input("You: ").strip()

    if question.lower() == "exit":
        print("Goodbye!")
        break

    if not question:
        continue

    results = search_pdf(question)
    best_score = results[0]["score"]

    if best_score < SIMILARITY_THRESHOLD:
        print("\nI couldn't find relevant information in the PDF.\n")
        continue

    try:
        answer = ask_gemini(question, results)
        print(f"\n{answer}\n")
    except Exception:
        print("\nSorry, I couldn't process that question.\n")
        logging.debug("Gemini request failed")