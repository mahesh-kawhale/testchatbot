import os
import json
import logging
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, JSONLoader
from langchain_community.vectorstores import Chroma
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain.schema import Document
import tiktoken

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("embedding.log"),
        logging.StreamHandler()
    ]
)

class EnhancedEmbedding:
    def __init__(self,
                 input_file="website_content.json",
                 persist_dir="./chroma_db",
                 chunk_size=800,  # Larger chunks for more context
                 chunk_overlap=150,  # More overlap to maintain context between chunks
                 embedding_model="text-embedding-3-small"):  # Keep using small model for compatibility
        """
        Initialize the enhanced embedding process.
        
        Args:
            input_file (str): Path to the input file (JSON or TXT)
            persist_dir (str): Directory to persist the vector store
            chunk_size (int): Size of each chunk
            chunk_overlap (int): Overlap between chunks
            embedding_model (str): OpenAI embedding model to use
        """
        # Load environment variables
        load_dotenv()
        
        self.input_file = input_file
        self.persist_dir = persist_dir
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embedding_model = embedding_model
        
        # Check if OpenAI API key is available
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY environment variable is not set")
            
        # Initialize OpenAI embeddings
        self.embedding = OpenAIEmbeddings(
            model=embedding_model,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        
    def _get_tokenizer(self):
        """Get the appropriate tokenizer for the embedding model."""
        if "3" in self.embedding_model:  # For text-embedding-3-*
            return tiktoken.get_encoding("cl100k_base")
        else:  # For text-embedding-ada-002
            return tiktoken.get_encoding("p50k_base")
            
    def _estimate_tokens(self, text):
        """Estimate the number of tokens in a text."""
        tokenizer = self._get_tokenizer()
        return len(tokenizer.encode(text))
        
    def load_documents(self):
        """Load documents from the input file."""
        file_extension = os.path.splitext(self.input_file)[1].lower()
        
        if file_extension == ".json":
            logging.info(f"Loading JSON data from {self.input_file}")
            try:
                with open(self.input_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                documents = []
                for page in data:
                    metadata = page.get("metadata", {})
                    content = page.get("content", "")
                    
                    if content:
                        # Create a document with metadata
                        doc = Document(
                            page_content=content,
                            metadata={
                                "url": metadata.get("url", ""),
                                "title": metadata.get("title", ""),
                                "description": metadata.get("description", ""),
                                "source": "website"
                            }
                        )
                        documents.append(doc)
                
                logging.info(f"Loaded {len(documents)} documents from JSON")
                return documents
                
            except Exception as e:
                logging.error(f"Error loading JSON: {e}")
                # Fallback to text loader
                logging.info("Falling back to text loader")
                return self._load_text_documents()
        else:
            return self._load_text_documents()
            
    def _load_text_documents(self):
        """Load documents from a text file."""
        logging.info(f"Loading text data from {self.input_file}")
        try:
            loader = TextLoader(self.input_file, encoding='utf-8')
            documents = loader.load()
            logging.info(f"Loaded {len(documents)} documents from text file")
            return documents
        except Exception as e:
            logging.error(f"Error loading text file: {e}")
            raise
            
    def split_documents(self, documents):
        """Split documents into chunks using an advanced strategy."""
        logging.info(f"Splitting documents into chunks (size: {self.chunk_size}, overlap: {self.chunk_overlap})")
        
        # Use RecursiveCharacterTextSplitter with improved settings
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=self._estimate_tokens,
            # More granular separators for better semantic chunking
            separators=[
                "\n\n\n",  # Triple line breaks (major section)
                "\n\n",    # Double line breaks (paragraph)
                "\n",      # Single line breaks
                ". ",      # Sentences
                "! ",      # Exclamations
                "? ",      # Questions
                ";",       # Semicolons
                ":",       # Colons
                ",",       # Commas
                " ",       # Spaces (word boundaries)
                ""         # Characters
            ]
        )
        
        # Process chunks to ensure they maintain context
        chunks = text_splitter.split_documents(documents)
        
        # Add metadata to help with retrieval
        for i, chunk in enumerate(chunks):
            # Add chunk index and total chunks to metadata
            chunk.metadata["chunk_index"] = i
            chunk.metadata["total_chunks"] = len(chunks)
            
            # Add first 50 characters as a preview
            if chunk.page_content:
                chunk.metadata["preview"] = chunk.page_content[:50].replace("\n", " ")
        
        logging.info(f"Created {len(chunks)} chunks with enhanced metadata")
        return chunks
        
    def create_vectorstore(self, chunks):
        """Create and persist the vector store."""
        logging.info(f"Creating vector store with {len(chunks)} chunks")
        
        try:
            vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=self.embedding,
                persist_directory=self.persist_dir
            )
            
            vectorstore.persist()
            logging.info(f"Vector store created and persisted to {self.persist_dir}")
            return vectorstore
            
        except Exception as e:
            logging.error(f"Error creating vector store: {e}")
            raise
            
    def process(self):
        """Process the documents: load, split, embed, and store."""
        logging.info("Starting document processing pipeline")
        
        # Load documents
        documents = self.load_documents()
        
        # Split into chunks
        chunks = self.split_documents(documents)
        
        # Create vector store
        vectorstore = self.create_vectorstore(chunks)
        
        logging.info("Document processing complete")
        return vectorstore

if __name__ == "__main__":
    # Example usage with improved settings
    processor = EnhancedEmbedding(
        input_file="website_content.json",  # Use JSON output from enhanced crawler
        persist_dir="./chroma_db",
        chunk_size=800,  # Larger chunks for more context
        chunk_overlap=150,  # More overlap to maintain context between chunks
        embedding_model="text-embedding-3-small"  # Keep using small model for compatibility
    )
    
    print("🚀 Starting enhanced embedding process...")
    print("📊 Configuration:")
    print(f"  - Input file: {processor.input_file}")
    print(f"  - Persist directory: {processor.persist_dir}")
    print(f"  - Chunk size: {processor.chunk_size}")
    print(f"  - Chunk overlap: {processor.chunk_overlap}")
    print(f"  - Embedding model: {processor.embedding_model}")
    
    processor.process()