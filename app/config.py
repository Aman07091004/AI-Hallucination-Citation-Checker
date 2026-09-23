import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

RAW_CORPUS_DIR = Path(os.getenv("RAW_CORPUS_DIR", "./data/raw_corpus"))
PROCESSED_DIR = Path(os.getenv("PROCESSED_DIR", "./data/processed"))
VECTOR_DB_DIR = Path(os.getenv("VECTOR_DB_DIR", "./data/processed/chroma"))

TESSERACT_CMD = os.getenv("TESSERACT_CMD")  # e.g. C:\Program Files\Tesseract-OCR\tesseract.exe
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")
COURTLISTENER_API_TOKEN = os.getenv("COURTLISTENER_API_TOKEN")

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)
