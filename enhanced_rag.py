import os
import logging
import time
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import Chroma
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_openai.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
# Remove StreamingStdOutCallbackHandler import
from langchain.schema import Document
import tiktoken

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("rag.log"),
        logging.StreamHandler()
    ]
)

class EnhancedRAG:
    def __init__(self,
                 persist_dir="./chroma_db",
                 model_name="gpt-4",  # Use GPT-4 for better quality if available
                 temperature=0.0,
                 streaming=True,
                 embedding_model="text-embedding-3-small",  # Keep using small model for compatibility
                 k_documents=6,  # Retrieve more documents for better context
                 search_type="mmr",  # Use Maximum Marginal Relevance for diversity
                 similarity_top_k=10,  # Consider more candidates for MMR
                 fetch_k=20):  # Fetch more documents initially
        """
        Initialize the enhanced RAG system.
        
        Args:
            persist_dir (str): Directory where the vector store is persisted
            model_name (str): OpenAI model to use (e.g., "gpt-3.5-turbo", "gpt-4")
            temperature (float): Temperature for the model
            streaming (bool): Whether to stream responses
            embedding_model (str): OpenAI embedding model to use
            k_documents (int): Number of documents to retrieve
            search_type (str): Type of search to use (e.g., "similarity", "mmr")
            similarity_top_k (int): Number of documents to consider for MMR
            fetch_k (int): Number of documents to fetch initially
        """
        # Load environment variables
        load_dotenv()
        
        self.persist_dir = persist_dir
        self.model_name = model_name
        self.temperature = temperature
        self.streaming = streaming
        self.embedding_model = embedding_model
        self.k_documents = k_documents
        self.search_type = search_type
        self.similarity_top_k = similarity_top_k
        self.fetch_k = fetch_k
        
        # Check if OpenAI API key is available
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY environment variable is not set")
            
        # Initialize components
        self._init_embeddings()
        self._init_vectorstore()
        self._init_llm()
        self._init_memory()
        self._init_prompts()
        self._init_chain()
        
        # Cache for responses
        self.cache = {}
        
    def _init_embeddings(self):
        """Initialize the embeddings model."""
        logging.info(f"Initializing embeddings model: {self.embedding_model}")
        self.embeddings = OpenAIEmbeddings(
            model=self.embedding_model,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        
    def _init_vectorstore(self):
        """Initialize the vector store."""
        logging.info(f"Loading vector store from {self.persist_dir}")
        try:
            self.vectorstore = Chroma(
                persist_directory=self.persist_dir,
                embedding_function=self.embeddings
            )
            
            # Configure advanced retrieval options
            self.retriever = self.vectorstore.as_retriever(
                search_type=self.search_type,
                search_kwargs={
                    "k": self.k_documents,
                    "fetch_k": self.fetch_k,
                    "lambda_mult": 0.5,  # Control diversity in MMR (0.0-1.0)
                }
            )
            
            logging.info(f"Configured retriever with search_type={self.search_type}, k={self.k_documents}, fetch_k={self.fetch_k}")
        except Exception as e:
            logging.error(f"Error loading vector store: {e}")
            raise
            
    def _init_llm(self):
        """Initialize the language model."""
        logging.info(f"Initializing LLM: {self.model_name}")
        
        # Initialize the LLM without streaming callbacks
        # This is compatible with newer LangChain versions
        self.llm = ChatOpenAI(
            model_name=self.model_name,
            temperature=self.temperature,
            streaming=self.streaming,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        
    def _init_memory(self):
        """Initialize conversation memory."""
        logging.info("Initializing conversation memory")
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
        
    def _init_prompts(self):
        """Initialize prompt templates."""
        logging.info("Initializing prompt templates")
        
        # System prompt for the chatbot
        self.system_template = """
        You are an AI assistant for Neutrino Tech Systems, a technology company.
        Your goal is to provide accurate, helpful, and concise information based on the provided context.
        
        CRITICAL INSTRUCTIONS:
        1. CAREFULLY READ and ANALYZE ALL the context information below. The answer to the user's question is contained within it.
        2. Provide ACCURATE answers based ONLY on the context provided. If the context doesn't contain the answer, say you don't know.
        3. Be THOROUGH in your analysis of the context - look for specific details that answer the question.
        4. Keep responses CONCISE - typically 2-3 sentences unless more detail is needed.
        5. Be DIRECT and SPECIFIC - focus on answering exactly what was asked.
        6. NEVER mention sources in your response - the system handles citations separately.
        7. Avoid phrases like "based on the context" or "according to the information" - just give the answer directly.
        8. If multiple pieces of context contain relevant information, SYNTHESIZE them into a coherent answer.
        9. If the context contains technical information relevant to the question, include it in your answer.
        
        Context information (ANALYZE CAREFULLY):
        {context}
        
        Chat History:
        {chat_history}
        
        Question: {question}
        
        Answer (be accurate, specific, and concise):
        """
        
        self.qa_prompt = PromptTemplate(
            input_variables=["context", "chat_history", "question"],
            template=self.system_template
        )
        
    def _init_chain(self):
        """Initialize the conversational retrieval chain."""
        logging.info("Initializing conversational retrieval chain")
        
        # Use a more advanced chain configuration
        self.chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.retriever,
            memory=self.memory,
            return_source_documents=True,
            get_chat_history=lambda h: h,  # Pass chat history directly
            combine_docs_chain_kwargs={"prompt": self.qa_prompt},
            verbose=True
        )
        
    def _get_cache_key(self, query: str, chat_history: List) -> str:
        """Generate a cache key from the query and chat history."""
        history_str = ""
        if chat_history:
            history_str = "".join([f"{msg.content}" for msg in chat_history])
        return f"{query}_{hash(history_str)}"
        
    def _format_source_documents(self, source_documents: List[Document]) -> str:
        """Format source documents for citation."""
        # Don't add any source information to the answer
        # as per user request to completely hide sources
        return ""
        
    def query(self, query: str, callbacks=None) -> Dict[str, Any]:
        """
        Process a query through the RAG system.
        
        Args:
            query (str): The user's query
            callbacks (list, optional): List of callback handlers for streaming
            
        Returns:
            Dict: Response containing answer and source documents
        """
        start_time = time.time()
        logging.info(f"Processing query: {query}")
        
        # Check cache
        cache_key = self._get_cache_key(query, self.memory.chat_memory.messages)
        if cache_key in self.cache and not self.streaming:
            logging.info("Cache hit")
            return self.cache[cache_key]
            
        try:
            # Process query
            if self.streaming and callbacks:
                # If streaming with callbacks, use them
                self.llm.callbacks = callbacks
            
            # Process the query
            response = self.chain({"question": query})
            
            # Format response
            answer = response.get("answer", "")
            source_documents = response.get("source_documents", [])
            
            # Add source citations if not streaming
            if not self.streaming and source_documents:
                sources_text = self._format_source_documents(source_documents)
                answer += sources_text
                
            result = {
                "answer": answer,
                "source_documents": source_documents,
                "processing_time": time.time() - start_time
            }
            
            # Cache result (only if not streaming)
            if not self.streaming:
                self.cache[cache_key] = result
            
            logging.info(f"Query processed in {result['processing_time']:.2f} seconds")
            return result
            
        except Exception as e:
            logging.error(f"Error processing query: {e}")
            return {
                "answer": "I'm sorry, I encountered an error while processing your query. Please try again.",
                "error": str(e),
                "processing_time": time.time() - start_time
            }
            
    def reset_conversation(self):
        """Reset the conversation history."""
        logging.info("Resetting conversation history")
        self.memory.clear()
        
    def get_token_count(self, text: str) -> int:
        """Count the number of tokens in a text."""
        try:
            encoding = tiktoken.encoding_for_model(self.model_name)
            return len(encoding.encode(text))
        except Exception:
            # Fallback to approximate counting
            return len(text.split()) * 1.3

if __name__ == "__main__":
    # Example usage
    rag = EnhancedRAG(
        persist_dir="./chroma_db",
        model_name="gpt-3.5-turbo",  # Use gpt-4 for better quality if available
        temperature=0.0,
        streaming=True,
        embedding_model="text-embedding-3-small",
        k_documents=4
    )
    
    print("\n🤖 Neutrino Tech Systems AI Assistant")
    print("Type 'exit' to quit, 'reset' to start a new conversation\n")
    
    while True:
        query = input("\n💬 You: ")
        
        if query.lower() == "exit":
            break
            
        if query.lower() == "reset":
            rag.reset_conversation()
            print("Conversation reset.")
            continue
            
        print("\n🤖 Assistant: ", end="")
        if not rag.streaming:
            result = rag.query(query)
            print(result["answer"])
        else:
            # When streaming, the output is handled by the callback
            result = rag.query(query)