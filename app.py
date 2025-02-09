from datetime import datetime
import streamlit as st
import logging
import os
from typing import List
from dataclasses import dataclass
from pathlib import Path
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from dotenv import load_dotenv

# Load environment variables and configure logging
load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

@dataclass
class ChatMessage:
    timestamp: str
    question: str
    answer: str
    context: str

def reset_all_states():
    """Reset all session states and widgets"""
    # Clear file uploader by resetting its key
    st.session_state["file_uploader_key"] = f"file_{datetime.now().timestamp()}"
    
    # Clear text input by resetting its key
    st.session_state["query_input_key"] = f"query_{datetime.now().timestamp()}"
    
    # Clear all session state variables
    for key in list(st.session_state.keys()):
        if key not in ["file_uploader_key", "query_input_key"]:
            del st.session_state[key]

class DocumentQASystem:
    TEMP_DIR = Path("temp")
    
    def __init__(self):
        self._load_api_keys()
        self._initialize_components()
        self._initialize_session_state()
        
    def _load_api_keys(self):
        """Load and validate API keys."""
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        self.google_api_key = os.getenv("GOOGLE_API_KEY")
        
        if not self.groq_api_key or not self.google_api_key:
            raise ValueError("Missing required API keys. Please check your .env file.")

    def _initialize_components(self):
        """Initialize LLM components with error handling."""
        try:
            self.embeddings = GoogleGenerativeAIEmbeddings(
                model="models/text-embedding-004",
                google_api_key=self.google_api_key
            )
            
            self.llm = ChatGroq(
                temperature=0.1,
                model_name="llama3-8b-8192",
                groq_api_key=self.groq_api_key,
                max_tokens=1024
            )
            
            self.output_parser = StrOutputParser()
            self.vector_store = None
            
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len,
                separators=["\n\n", "\n", " ", ""]
            )
            
        except Exception as e:
            logger.error(f"Failed to initialize components: {str(e)}")
            raise

    def _initialize_session_state(self):
        """Initialize Streamlit session state variables."""
        # Initialize widget keys if they don't exist
        if "file_uploader_key" not in st.session_state:
            st.session_state["file_uploader_key"] = "file_uploader_initial"
        if "query_input_key" not in st.session_state:
            st.session_state["query_input_key"] = "query_input_initial"
            
        # Initialize other session state variables
        if 'uploaded_files' not in st.session_state:
            st.session_state.uploaded_files = []
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        if 'vector_store' not in st.session_state:
            st.session_state.vector_store = None

    def upload_and_process_document(self, uploaded_file) -> bool:
        """Process uploaded PDF with improved error handling and cleanup."""
        temp_file_path = self.TEMP_DIR / uploaded_file.name
        
        try:
            self.TEMP_DIR.mkdir(exist_ok=True)
            temp_file_path.write_bytes(uploaded_file.getvalue())
            
            docs = PDFPlumberLoader(str(temp_file_path)).load()
            chunks = self.text_splitter.split_documents(docs)
            
            # Store vector_store in session state for persistence
            if st.session_state.vector_store is None:
                st.session_state.vector_store = FAISS.from_documents(chunks, self.embeddings)
            else:
                st.session_state.vector_store.add_documents(chunks)
            
            # Update instance variable
            self.vector_store = st.session_state.vector_store
            return True
            
        except Exception as e:
            logger.error(f"Document processing failed: {str(e)}")
            st.error(f"Failed to process {uploaded_file.name}: {str(e)}")
            return False
            
        finally:
            if temp_file_path.exists():
                temp_file_path.unlink()

    def similarity_search(self, query: str, top_k: int = 5) -> List[Document]:
        """Perform similarity search with validation and error handling."""
        if not query.strip():
            raise ValueError("Query cannot be empty")
            
        if st.session_state.vector_store is None:
            logger.warning("No documents loaded for search")
            return []
            
        try:
            results = st.session_state.vector_store.similarity_search(
                query,
                k=top_k,
                fetch_k=20
            )
            return results
            
        except Exception as e:
            logger.error(f"Similarity search failed: {str(e)}")
            return []

    def generate_answer(self, user_query: str, context_docs: List[Document]) -> str:
        """Generate answer with improved prompt and error handling."""
        if not context_docs:
            return "Please upload and process some documents before asking questions."
            
        try:
            PROMPT_TEMPLATE = """You are an expert research assistant. Answer the question based on the provided context.
            Guidelines:
            - Be concise and factual (max 5 sentences)
            - Cite specific parts of the documents when possible
            - Express uncertainty when the context doesn't fully answer the question
            - Focus only on information present in the context
            
            Context: {document_context}
            
            Question: {user_query}
            
            Answer: """
            
            context_text = "\n\n".join([
                f"Document {i+1}: {doc.page_content}"
                for i, doc in enumerate(context_docs)
            ])
            
            conversation_prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
            chain = conversation_prompt | self.llm | self.output_parser
            
            return chain.invoke({
                "user_query": user_query,
                "document_context": context_text
            })
            
        except Exception as e:
            logger.error(f"Answer generation failed: {str(e)}")
            return "I encountered an error while generating the answer. Please try again."

def setup_streamlit_ui():
    """Configure Streamlit UI with improved styling."""
    st.set_page_config(
        page_title="PDF Question Answering System",
        page_icon="📚",
        layout="wide"
    )
    
    st.markdown("""
        <style>
        .stApp {
            max-width: 1800px;
            margin: 0 auto;
        }
        .chat-message {
            padding: 1.5rem;
            border-radius: 0.75rem;
            margin-bottom: 1.5rem;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .chat-message.question {
            background-color: #1e3a8a;
            color: white;
        }
        .chat-message.answer {
            background-color: #000000;
            border: 1px solid #e2e8f0;
        }
        .chat-context {
            font-size: 0.875rem;
            color: #64748b;
            margin-top: 0.75rem;
            padding-top: 0.75rem;
            border-top: 1px solid #e2e8f0;
        }
        .timestamp {
            color: #94a3b8;
            font-size: 0.75rem;
        }
        .stTextInput > label {
            display: none;
        }
        </style>
    """, unsafe_allow_html=True)


def main():
    setup_streamlit_ui()
    qa_system = DocumentQASystem()

    # Sidebar
    with st.sidebar:
        st.title("📑 Document Management")
        
        # Use dynamic key for file uploader to reset it
        uploaded_files = st.file_uploader(
            "Upload PDF Documents",
            type=['pdf'],
            accept_multiple_files=True,
            help="Upload one or more PDF documents to analyze",
            label_visibility="visible",
            key=st.session_state["file_uploader_key"]
        )
        
        if st.button("Process Documents", type="primary"):
            if uploaded_files:
                progress_bar = st.progress(0)
                for i, uploaded_file in enumerate(uploaded_files):
                    if uploaded_file.name not in st.session_state.uploaded_files:
                        with st.spinner(f"Processing {uploaded_file.name}..."):
                            if qa_system.upload_and_process_document(uploaded_file):
                                st.session_state.uploaded_files.append(uploaded_file.name)
                                st.success(f"✅ {uploaded_file.name}")
                        progress_bar.progress((i + 1) / len(uploaded_files))
                progress_bar.empty()
                # Reset file uploader after processing
                st.session_state["file_uploader_key"] = str(datetime.now())
            else:
                st.warning("Please upload documents first.")
        
        if st.session_state.uploaded_files:
            st.markdown("### 📚 Processed Documents")
            for file in st.session_state.uploaded_files:
                st.text(f"• {file}")
        
        if st.button("Reset Session", type="secondary"):
            reset_all_states()
            st.success("Session reset successfully!")
            st.rerun()

    # Main chat interface
    st.title("💬 Document Q&A Chat")
    
    def process_query():
        """Callback to process the query and clear input"""
        query = st.session_state[st.session_state["query_input_key"]]
        if not query:
            return
            
        with st.spinner("Searching documents and generating answer..."):
            context_docs = qa_system.similarity_search(query)
            if context_docs:
                answer = qa_system.generate_answer(query, context_docs)
                context_summary = "\n".join([doc.page_content[:200] + "..." for doc in context_docs[:2]])
                
                st.session_state.chat_history.append(
                    ChatMessage(
                        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        question=query,
                        answer=answer,
                        context=context_summary
                    )
                )
            else:
                st.warning("Please upload and process documents before asking questions.")
        
        # Clear the input by changing its key
        st.session_state["query_input_key"] = f"query_input_{datetime.now().timestamp()}"


    # Chat input with dynamic key to reset it
    query = st.text_input(
        label="Question Input",
        placeholder="Ask a question about your documents...",
        key=st.session_state["query_input_key"],
        label_visibility="collapsed",
        on_change=process_query
    )

    # Display chat history
    if st.session_state.chat_history:
        for chat in reversed(st.session_state.chat_history):
            st.markdown(
                f"""<div class="chat-message question">
                    <strong>Question:</strong> {chat.question}
                    <div class="timestamp">{chat.timestamp}</div>
                </div>
                <div class="chat-message answer">
                    <strong>Answer:</strong> {chat.answer}
                    <div class="chat-context">
                        <strong>Related Context:</strong><br>{chat.context}
                    </div>
                </div>""",
                unsafe_allow_html=True
            )
    else:
        st.info("Upload documents and start asking questions to begin the conversation!")

    # Footer
    st.markdown("---")
    st.markdown("*Powered by Groq, Google AI, and LangChain*")

if __name__ == "__main__":
    main()