"""
diagnose.py — run this on the server to see exactly what programs_db
contains and what context the bot would use for a given query.

Usage:
    python diagnose.py
"""
import asyncio
import os
import sys

# ── Make sure project imports work ───────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

# Load .env so OPENAI_API_KEY is available (same as main.py does)
from dotenv import load_dotenv
load_dotenv()

from utils.text_utils import CharacterTextSplitter, TextFileLoader, PDFLoader
from utils.vectordatabase import VectorDatabase

MATERIAS_DIR = "materias"
TEST_QUERIES = [
    "que carreras ofrecen",
    "que materias se ofrecen",
    "ciencias computacionales materias",
    "licenciaturas disponibles",
]


async def main():
    print("=" * 60)
    print("STEP 1 — Check materias/ directory")
    print("=" * 60)
    print(f"cwd: {os.getcwd()}")
    print(f"materias/ exists: {os.path.isdir(MATERIAS_DIR)}")
    if os.path.isdir(MATERIAS_DIR):
        files = os.listdir(MATERIAS_DIR)
        print(f"Files ({len(files)}): {files}")
    else:
        print("ERROR: materias/ not found — bot will have no course context!")
        return

    print()
    print("=" * 60)
    print("STEP 2 — Load materias/ files")
    print("=" * 60)
    try:
        loader = TextFileLoader(MATERIAS_DIR)
        docs = loader.load_documents()
        print(f"Loaded {len(docs)} documents")
        for i, doc in enumerate(docs):
            print(f"  doc[{i}]: {len(doc)} chars, first 80: {repr(doc[:80])}")
    except Exception as e:
        print(f"ERROR loading: {e}")
        return

    print()
    print("=" * 60)
    print("STEP 3 — Split into chunks")
    print("=" * 60)
    splitter = CharacterTextSplitter()
    chunks = splitter.split_texts(docs)
    print(f"Total chunks: {len(chunks)}")
    print(f"Sample chunk[0]: {repr(chunks[0][:200])}")

    print()
    print("=" * 60)
    print("STEP 4 — Build VectorDatabase (requires OPENAI_API_KEY)")
    print("=" * 60)
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set — cannot build vector DB")
        return
    print(f"API key found: {api_key[:8]}...")

    try:
        db = VectorDatabase()
        await db.abuild_from_list(chunks)
        print(f"VectorDatabase built with {len(db.vectors)} vectors")
    except Exception as e:
        print(f"ERROR building VectorDatabase: {e}")
        return

    print()
    print("=" * 60)
    print("STEP 5 — Test queries")
    print("=" * 60)
    for query in TEST_QUERIES:
        results = db.search_by_text(query, k=3)
        print(f"\nQuery: '{query}'")
        for i, (text, score) in enumerate(results):
            print(f"  [{i}] score={score:.4f}  text={repr(text[:120])}")


if __name__ == "__main__":
    asyncio.run(main())
