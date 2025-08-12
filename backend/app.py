import os
import time
import json
import logging
from flask import Flask, request, jsonify, Response, stream_with_context
from flask_cors import CORS
from dotenv import load_dotenv
import sys
import uuid
from functools import wraps
from datetime import datetime

# Add parent directory to path to import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from enhanced_rag import EnhancedRAG

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("backend.log"),
        logging.StreamHandler()
    ]
)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Simple in-memory rate limiting
rate_limits = {}
# Simple in-memory session storage
sessions = {}

# Initialize the RAG system with improved settings
rag = EnhancedRAG(
    persist_dir="../chroma_db",
    model_name=os.getenv("OPENAI_MODEL", "gpt-4"),  # Use GPT-4 for better quality if available
    temperature=float(os.getenv("TEMPERATURE", "0.0")),
    streaming=True,
    embedding_model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),  # Keep using small model for compatibility
    k_documents=int(os.getenv("K_DOCUMENTS", "6")),  # Retrieve more documents
    search_type="mmr",  # Use Maximum Marginal Relevance for diversity
    similarity_top_k=10,  # Consider more candidates for MMR
    fetch_k=20  # Fetch more documents initially
)

logging.info(f"Initialized RAG system with model: {rag.model_name}, embedding: {rag.embedding_model}")

# Authentication decorator
def require_api_key(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        api_key = request.headers.get('X-API-Key')
        if not api_key:
            return jsonify({"error": "API key is required"}), 401
            
        # In production, you would validate against a database
        # For now, we'll use a simple environment variable
        if api_key != os.getenv("API_KEY", "test-key"):
            return jsonify({"error": "Invalid API key"}), 401
            
        return f(*args, **kwargs)
    return decorated_function

# Rate limiting decorator
def rate_limit(requests_per_minute=60):
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            client_ip = request.remote_addr
            current_time = time.time()
            
            # Initialize or update rate limit data
            if client_ip not in rate_limits:
                rate_limits[client_ip] = {"count": 0, "reset_time": current_time + 60}
            
            # Reset count if time window has passed
            if current_time > rate_limits[client_ip]["reset_time"]:
                rate_limits[client_ip] = {"count": 0, "reset_time": current_time + 60}
                
            # Check if rate limit exceeded
            if rate_limits[client_ip]["count"] >= requests_per_minute:
                return jsonify({
                    "error": "Rate limit exceeded",
                    "retry_after": int(rate_limits[client_ip]["reset_time"] - current_time)
                }), 429
                
            # Increment request count
            rate_limits[client_ip]["count"] += 1
            
            return f(*args, **kwargs)
        return decorated_function
    return decorator

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        "status": "ok",
        "timestamp": datetime.now().isoformat()
    })

@app.route('/api/chat', methods=['POST'])
@require_api_key
@rate_limit(60)  # 60 requests per minute
def chat():
    """Process a chat message."""
    data = request.json
    
    if not data or 'message' not in data:
        return jsonify({"error": "Message is required"}), 400
        
    # Get or create session
    session_id = data.get('session_id')
    if not session_id or session_id not in sessions:
        session_id = str(uuid.uuid4())
        sessions[session_id] = {"rag": EnhancedRAG(
            persist_dir="../chroma_db",
            model_name=os.getenv("OPENAI_MODEL", "gpt-4"),  # Use GPT-4 for better quality
            temperature=float(os.getenv("TEMPERATURE", "0.0")),
            streaming=False,  # Non-streaming for regular requests
            embedding_model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),  # Keep using small model for compatibility
            k_documents=int(os.getenv("K_DOCUMENTS", "6")),  # Retrieve more documents
            search_type="mmr",  # Use Maximum Marginal Relevance for diversity
            similarity_top_k=10,  # Consider more candidates for MMR
            fetch_k=20  # Fetch more documents initially
        )}
        
        logging.info(f"Created new session for regular chat: {session_id}")
    
    session_rag = sessions[session_id]["rag"]
    
    try:
        # Process the message
        result = session_rag.query(data['message'])
        
        # Process the answer to ensure it's concise
        answer = result["answer"].strip()
        
        # If answer is too long, truncate it (as a fallback)
        if len(answer.split()) > 50:  # Roughly 2-3 sentences
            sentences = answer.split('. ')
            if len(sentences) > 2:
                answer = '. '.join(sentences[:2]) + '.'
        
        # Don't include sources in the response
        # as per user request to completely hide sources
        
        return jsonify({
            "session_id": session_id,
            "answer": answer,
            "sources": [],  # Empty array instead of actual sources
            "processing_time": result["processing_time"]
        })
        
    except Exception as e:
        logging.error(f"Error processing chat: {e}")
        return jsonify({
            "error": "Failed to process your message",
            "details": str(e)
        }), 500

@app.route('/api/chat/stream', methods=['POST'])
@require_api_key
@rate_limit(30)  # Lower rate limit for streaming
def stream_chat():
    """Stream a chat response."""
    data = request.json
    
    if not data or 'message' not in data:
        return jsonify({"error": "Message is required"}), 400
    
    # Get or create session
    session_id = data.get('session_id')
    if not session_id or session_id not in sessions:
        session_id = str(uuid.uuid4())
        sessions[session_id] = {"rag": EnhancedRAG(
            persist_dir="../chroma_db",
            model_name=os.getenv("OPENAI_MODEL", "gpt-4"),  # Use GPT-4 for better quality
            temperature=float(os.getenv("TEMPERATURE", "0.0")),
            streaming=True,  # Enable streaming
            embedding_model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),  # Keep using small model for compatibility
            k_documents=int(os.getenv("K_DOCUMENTS", "6")),  # Retrieve more documents
            search_type="mmr",  # Use Maximum Marginal Relevance for diversity
            similarity_top_k=10,  # Consider more candidates for MMR
            fetch_k=20  # Fetch more documents initially
        )}
        
        logging.info(f"Created new session: {session_id}")
    
    session_rag = sessions[session_id]["rag"]
    query = data['message']
    
    def generate():
        try:
            # Use a simpler approach that doesn't rely on CallbackManager
            from langchain.callbacks.base import BaseCallbackHandler
            
            # Create a custom callback handler that inherits from BaseCallbackHandler
            class CustomCallback(BaseCallbackHandler):
                def __init__(self):
                    super().__init__()
                    self.tokens = []
                    self.content = ""
                
                def on_llm_new_token(self, token, **kwargs):
                    self.content += token
                    # We can't yield directly from here, so we'll just collect the tokens
            
            # Create an instance of our callback
            callback_handler = CustomCallback()
            
            # Use the session RAG to query with streaming
            # Since we can't use the streaming callback directly in this version of LangChain,
            # we'll just get the complete result and then simulate streaming
            result = session_rag.query(query)
            
            # Simulate streaming by sending the complete answer
            yield f"data: {json.dumps({'type': 'token', 'content': result['answer']})}\n\n"
            
            # Extract source documents (limit to top 2 most relevant)
            sources = []
            for doc in result.get("source_documents", [])[:2]:
                if doc.metadata.get("url"):
                    # Clean up title if needed
                    title = doc.metadata.get("title", "Neutrino Tech Systems")
                    if not title or len(title) < 3:
                        title = "Neutrino Tech Systems"
                    
                    # Simplify title if too long
                    if len(title) > 30:
                        title = title[:27] + "..."
                    
                    sources.append({
                        "url": doc.metadata.get("url", ""),
                        "title": title
                    })
            
            # Don't send sources to the frontend at all
            # as per user request to completely hide sources
            
            # Send the final done event
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
            
        except Exception as e:
            logging.error(f"Error in stream: {e}")
            yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"
    
    return Response(stream_with_context(generate()), mimetype='text/event-stream')

@app.route('/api/reset', methods=['POST'])
@require_api_key
def reset_conversation():
    """Reset a conversation."""
    data = request.json
    session_id = data.get('session_id')
    
    if session_id and session_id in sessions:
        sessions[session_id]["rag"].reset_conversation()
        return jsonify({"status": "Conversation reset"})
    elif not session_id:
        # Reset global RAG
        rag.reset_conversation()
        return jsonify({"status": "Global conversation reset"})
    else:
        return jsonify({"error": "Session not found"}), 404

@app.route('/api/crawl', methods=['POST'])
@require_api_key
def trigger_crawl():
    """Trigger a new crawl (admin only)."""
    # In a production environment, you would add additional authentication
    # to ensure only admins can trigger a crawl
    
    data = request.json
    urls = data.get('urls', ["https://neutrinotechsystems.com"])
    max_pages = data.get('max_pages', 100)
    
    try:
        # Import here to avoid circular imports
        from enhanced_web_crawler import EnhancedWebCrawler
        
        # Run crawler in a separate thread (in production, use Celery or similar)
        def run_crawler():
            crawler = EnhancedWebCrawler(
                start_urls=urls,
                max_pages=max_pages
            )
            crawler.crawl()
            crawler.save_to_file()
            
            # Optionally re-embed after crawling
            from enhanced_embedding import EnhancedEmbedding
            processor = EnhancedEmbedding()
            processor.process()
            
        import threading
        thread = threading.Thread(target=run_crawler)
        thread.start()
        
        return jsonify({
            "status": "Crawl started",
            "urls": urls,
            "max_pages": max_pages
        })
        
    except Exception as e:
        logging.error(f"Error triggering crawl: {e}")
        return jsonify({
            "error": "Failed to start crawl",
            "details": str(e)
        }), 500

if __name__ == '__main__':
    # Create directories if they don't exist
    os.makedirs(os.path.dirname("../chroma_db"), exist_ok=True)
    
    # Get port from environment or use default
    port = int(os.getenv("PORT", 5000))
    
    print(f"🚀 Starting Neutrino Chatbot API on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False)