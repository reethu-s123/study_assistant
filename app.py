"""
Flask Application - Study Assistant
Main entry point for the web application
"""

import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
from document_processor import DocumentProcessor
from vector_store import VectorStore
from agent_crew import StudyAssistantCrew
from config import (
    UPLOAD_FOLDER,
    ALLOWED_EXTENSIONS,
    MAX_FILE_SIZE,
    FLASK_HOST,
    FLASK_PORT,
    FLASK_DEBUG
)

app = Flask(__name__)
CORS(app)

# Initialize components
doc_processor = DocumentProcessor()
vector_store = VectorStore()
study_crew = StudyAssistantCrew()

# Configuration
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE


def allowed_file(filename: str) -> bool:
    """Check if file has allowed extension"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    vector_store_healthy = vector_store.health_check()
    
    return jsonify({
        "status": "healthy",
        "vector_store": "connected" if vector_store_healthy else "disconnected"
    }), 200 if vector_store_healthy else 503


@app.route('/upload', methods=['POST'])
def upload_document():
    """Upload and process a document"""
    try:
        # Check if file is present
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400

        file = request.files['file']
        
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400

        if not allowed_file(file.filename):
            return jsonify({
                "error": f"File type not allowed. Allowed types: {', '.join(ALLOWED_EXTENSIONS)}"
            }), 400

        # Save file
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Process document
        chunks, full_text = doc_processor.process_document(file_path)

        # Store embeddings
        document_id = os.path.splitext(filename)[0]
        vector_store.store_embeddings(chunks, document_id)

        return jsonify({
            "status": "success",
            "document_id": document_id,
            "filename": filename,
            "chunks": len(chunks),
            "message": f"Document '{filename}' uploaded and processed successfully"
        }), 201

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/ask', methods=['POST'])
def ask_question():
    """Ask a question about the uploaded documents"""
    try:
        data = request.get_json()
        
        if not data or 'question' not in data:
            return jsonify({"error": "No question provided"}), 400

        query = data['question'].strip()
        
        if not query:
            return jsonify({"error": "Question cannot be empty"}), 400

        # Process query through crew
        result = study_crew.process_query(query)

        return jsonify({
            "status": "success" if result['success'] else "error",
            "question": query,
            "answer": result['answer'],
            "retrieved_documents": result['retrieved_docs'],
            "document_count": len(result['retrieved_docs'])
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/documents', methods=['GET'])
def list_documents():
    """List uploaded documents"""
    try:
        files = []
        if os.path.exists(app.config['UPLOAD_FOLDER']):
            for filename in os.listdir(app.config['UPLOAD_FOLDER']):
                if allowed_file(filename):
                    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                    files.append({
                        "filename": filename,
                        "size": os.path.getsize(file_path),
                        "document_id": os.path.splitext(filename)[0]
                    })

        return jsonify({
            "status": "success",
            "documents": files,
            "count": len(files)
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/documents/<document_id>', methods=['DELETE'])
def delete_document(document_id: str):
    """Delete a document and its embeddings"""
    try:
        # Delete from vector store
        vector_store.delete_document(document_id)

        # Delete file if exists
        upload_folder = app.config['UPLOAD_FOLDER']
        for filename in os.listdir(upload_folder):
            if filename.startswith(document_id) and allowed_file(filename):
                file_path = os.path.join(upload_folder, filename)
                os.remove(file_path)
                break

        return jsonify({
            "status": "success",
            "message": f"Document '{document_id}' deleted successfully"
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/search', methods=['POST'])
def search_documents():
    """Search documents by query"""
    try:
        data = request.get_json()
        
        if not data or 'query' not in data:
            return jsonify({"error": "No query provided"}), 400

        query = data['query'].strip()
        top_k = data.get('top_k', 5)

        if not query:
            return jsonify({"error": "Query cannot be empty"}), 400

        # Search vector store
        results = vector_store.search(query, top_k=top_k)

        return jsonify({
            "status": "success",
            "query": query,
            "results": results,
            "count": len(results)
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({"error": "Endpoint not found"}), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return jsonify({"error": "Internal server error"}), 500


if __name__ == '__main__':
    app.run(
        host=FLASK_HOST,
        port=FLASK_PORT,
        debug=FLASK_DEBUG
    )
