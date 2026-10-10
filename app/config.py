from dotenv import load_dotenv

load_dotenv()

from pathlib import Path
import os

UPLOAD_DIR = Path("data/uploads")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "openai/gpt-oss-20b"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K = 5
MIN_SIMILARITY = 0.15
NOT_FOUND_MESSAGE = "I couldn't find this in your uploaded documents."
HEADING_PATTERN = r"(?:Module|Chapter|Unit)\s+\d+(?:\.\d+)?\s*:"
MAX_SECTION_CHARS = 2000
CHROMA_DIR = Path("data/chroma")
API_URL = os.getenv("API_URL", "http://localhost:8000")