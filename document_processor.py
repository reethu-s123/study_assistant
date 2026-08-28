"""
Document Processor Module
Handles PDF extraction, text processing, and chunking
"""

import os
from typing import List, Tuple
import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter


class DocumentProcessor:
    """Process documents and prepare them for embedding"""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )

    def extract_text_from_pdf(self, file_path: str) -> str:
        """Extract text from PDF file"""
        try:
            text = ""
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text()
            return text
        except Exception as e:
            raise Exception(f"Error extracting PDF: {str(e)}")

    def extract_text_from_txt(self, file_path: str) -> str:
        """Extract text from text file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except Exception as e:
            raise Exception(f"Error reading text file: {str(e)}")

    def extract_text_from_document(self, file_path: str) -> str:
        """Extract text from document based on file type"""
        if file_path.endswith('.pdf'):
            return self.extract_text_from_pdf(file_path)
        elif file_path.endswith('.txt'):
            return self.extract_text_from_txt(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_path}")

    def chunk_text(self, text: str) -> List[str]:
        """Split text into chunks"""
        return self.text_splitter.split_text(text)

    def process_document(self, file_path: str) -> Tuple[List[str], str]:
        """Process document and return chunks with metadata"""
        # Extract text
        text = self.extract_text_from_document(file_path)

        # Clean text
        text = self._clean_text(text)

        # Chunk text
        chunks = self.chunk_text(text)

        return chunks, text

    @staticmethod
    def _clean_text(text: str) -> str:
        """Clean and normalize text"""
        # Remove extra whitespace
        text = ' '.join(text.split())
        # Remove special characters but keep punctuation
        return text


if __name__ == "__main__":
    processor = DocumentProcessor()
    print("Document processor initialized successfully")
