import os
from dotenv import load_dotenv

load_dotenv()

# Pinecone Configuration
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "your-api-key")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "us-east1-aws")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "study-assistant")

# Ollama Configuration
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral")

# Embedding Configuration
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384

# Flask Configuration
FLASK_HOST = os.getenv("FLASK_HOST", "0.0.0.0")
FLASK_PORT = int(os.getenv("FLASK_PORT", 5000))
FLASK_DEBUG = os.getenv("FLASK_DEBUG", True)

# Upload Configuration
UPLOAD_FOLDER = os.getenv("UPLOAD_FOLDER", "uploads")
ALLOWED_EXTENSIONS = {"pdf", "txt", "docx"}
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
