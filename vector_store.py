"""
Vector Store Module
Handles Pinecone integration for storing and retrieving embeddings
"""

from typing import List, Dict, Tuple
import pinecone
from sentence_transformers import SentenceTransformer
from config import (
    PINECONE_API_KEY,
    PINECONE_ENVIRONMENT,
    PINECONE_INDEX_NAME,
    EMBEDDING_MODEL,
    EMBEDDING_DIMENSION
)


class VectorStore:
    """Manages vector embeddings and Pinecone operations"""

    def __init__(self):
        self.model = SentenceTransformer(EMBEDDING_MODEL)
        self.index_name = PINECONE_INDEX_NAME
        self._initialize_pinecone()

    def _initialize_pinecone(self):
        """Initialize Pinecone connection"""
        try:
            pinecone.init(
                api_key=PINECONE_API_KEY,
                environment=PINECONE_ENVIRONMENT
            )
            
            # Create index if it doesn't exist
            if self.index_name not in pinecone.list_indexes():
                pinecone.create_index(
                    name=self.index_name,
                    dimension=EMBEDDING_DIMENSION,
                    metric="cosine"
                )
            
            self.index = pinecone.Index(self.index_name)
        except Exception as e:
            raise Exception(f"Failed to initialize Pinecone: {str(e)}")

    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for texts using Sentence Transformers"""
        try:
            embeddings = self.model.encode(texts, convert_to_numpy=True)
            return embeddings.tolist()
        except Exception as e:
            raise Exception(f"Error generating embeddings: {str(e)}")

    def store_embeddings(self, chunks: List[str], document_id: str):
        """Store embeddings in Pinecone"""
        try:
            embeddings = self.generate_embeddings(chunks)
            
            # Prepare vectors for upsert
            vectors = []
            for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                vector_id = f"{document_id}_{idx}"
                metadata = {
                    "text": chunk,
                    "document_id": document_id,
                    "chunk_index": idx
                }
                vectors.append((vector_id, embedding, metadata))
            
            # Upsert to Pinecone
            self.index.upsert(vectors=vectors)
            
        except Exception as e:
            raise Exception(f"Error storing embeddings: {str(e)}")

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Search for similar documents"""
        try:
            query_embedding = self.generate_embeddings([query])[0]
            
            results = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True
            )
            
            retrieved_docs = []
            for match in results.matches:
                retrieved_docs.append({
                    "id": match.id,
                    "score": match.score,
                    "text": match.metadata.get("text", ""),
                    "document_id": match.metadata.get("document_id", "")
                })
            
            return retrieved_docs
        except Exception as e:
            raise Exception(f"Error searching: {str(e)}")

    def delete_document(self, document_id: str):
        """Delete all embeddings for a document"""
        try:
            # Get all vector IDs for the document
            results = self.index.query(
                vector=[0] * EMBEDDING_DIMENSION,
                filter={"document_id": {"$eq": document_id}},
                top_k=10000
            )
            
            # Delete vectors
            if results.matches:
                vector_ids = [match.id for match in results.matches]
                self.index.delete(ids=vector_ids)
                
        except Exception as e:
            raise Exception(f"Error deleting document: {str(e)}")

    def health_check(self) -> bool:
        """Check if Pinecone connection is healthy"""
        try:
            self.index.describe_index_stats()
            return True
        except Exception:
            return False


if __name__ == "__main__":
    store = VectorStore()
    print("Vector store initialized successfully")
