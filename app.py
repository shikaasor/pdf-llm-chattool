from datetime import datetime
import streamlit as st
import logging
import os
import google.generativeai as genai
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
from langchain_community.document_loaders import PDFPlumberLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from dotenv import load_dotenv
from PIL import Image
import io
import base64
import docx
import pdfplumber
import uuid
import time

# Load environment variables and configure logging
load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

@dataclass
class ChatMessage:
    timestamp: str
    question: str
    answer: str
    context_text: str
    context_images: List[Dict[str, Any]] = field(default_factory=list)
    processing_time: float = 0.0

def reset_all_states():
    """Reset all session states and widgets"""
    st.session_state["file_uploader_key"] = f"file_{datetime.now().timestamp()}"
    st.session_state["query_input_key"] = f"query_{datetime.now().timestamp()}"
    keys_to_keep = ["file_uploader_key", "query_input_key"]
    for key in list(st.session_state.keys()):
        if key not in keys_to_keep:
            del st.session_state[key]
    # Re-initialize necessary states
    st.session_state.uploaded_files_info = {}
    st.session_state.chat_history = []
    st.session_state.vector_store = None
    st.session_state.image_store = {}
    st.session_state.doc_image_map = {}
    st.session_state.processing_stats = {"total_chunks": 0, "total_images": 0}

class DocumentQASystem:
    TEMP_DIR = Path("temp")
    MAX_IMAGE_SIZE = (800, 600)  # Resize large images to save memory
    SUPPORTED_IMAGE_FORMATS = {"JPEG", "PNG", "GIF", "BMP", "TIFF"}

    def __init__(self):
        self._load_api_keys()
        self._initialize_components()
        self._initialize_session_state()
        self.TEMP_DIR.mkdir(exist_ok=True)

    def _load_api_keys(self):
        self.google_api_key = os.getenv("GOOGLE_API_KEY")
        if not self.google_api_key:
            st.error("❌ Missing GOOGLE_API_KEY. Please check your .env file.")
            st.info("💡 Create a .env file with: GOOGLE_API_KEY=your_api_key_here")
            raise ValueError("Missing GOOGLE_API_KEY")

    def _initialize_components(self):
        try:
            self.embeddings = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001",
                google_api_key=self.google_api_key
            )
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-flash-latest",
                google_api_key=self.google_api_key,
                temperature=0.3,
                convert_system_message_to_human=True
            )
            
            # Initialize Gemini for image analysis
            genai.configure(api_key=self.google_api_key)
            self.vision_model = genai.GenerativeModel('gemini-1.5-flash-latest')
            
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1200,
                chunk_overlap=200,
                length_function=len,
                separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""]
            )
            logger.info("✅ Google AI components initialized successfully")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Google AI components: {str(e)}")
            st.error(f"Failed to initialize Google AI components: {str(e)}")
            raise

    def _initialize_session_state(self):
        defaults = {
            "file_uploader_key": "file_uploader_initial",
            "query_input_key": "query_input_initial",
            "uploaded_files_info": {},
            "chat_history": [],
            "vector_store": None,
            "image_store": {},
            "doc_image_map": {},
            "processing_stats": {"total_chunks": 0, "total_images": 0}
        }
        
        for key, default_value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = default_value

    def _resize_image(self, img: Image.Image) -> Image.Image:
        """Resize image if it's too large to save memory"""
        if img.size[0] > self.MAX_IMAGE_SIZE[0] or img.size[1] > self.MAX_IMAGE_SIZE[1]:
            img.thumbnail(self.MAX_IMAGE_SIZE, Image.Resampling.LANCZOS)
            logger.info(f"Resized image to {img.size}")
        return img
    
    def _analyze_image_with_gemini(self, image_data: str, source: str, page: int = None) -> dict:
        """Analyze image using Gemini Vision for description and OCR"""
        try:
            # Convert base64 to PIL Image
            img_bytes = base64.b64decode(image_data)
            img = Image.open(io.BytesIO(img_bytes))
            
            # Prepare the prompt for comprehensive analysis
            prompt = """
            Analyze this image comprehensively and provide:
            1. A detailed description of what you see in the image
            2. Extract and transcribe ALL text visible in the image (OCR)
            3. Identify any charts, graphs, tables, or structured data
            4. Note any important visual elements that could be relevant for document search
            
            Format your response as:
            DESCRIPTION: [detailed description]
            OCR_TEXT: [all extracted text]
            VISUAL_ELEMENTS: [charts, tables, diagrams, etc.]
            CONTEXT: [how this relates to document content]
            """
            
            # Generate content using Gemini Vision
            response = self.vision_model.generate_content([prompt, img])
            analysis_text = response.text
            
            # Parse the response to extract different components
            analysis_parts = {
                'description': '',
                'ocr_text': '',
                'visual_elements': '',
                'context': '',
                'full_analysis': analysis_text
            }
            
            # Simple parsing of the structured response
            lines = analysis_text.split('\n')
            current_section = None
            
            for line in lines:
                line = line.strip()
                if line.startswith('DESCRIPTION:'):
                    current_section = 'description'
                    analysis_parts['description'] = line.replace('DESCRIPTION:', '').strip()
                elif line.startswith('OCR_TEXT:'):
                    current_section = 'ocr_text'
                    analysis_parts['ocr_text'] = line.replace('OCR_TEXT:', '').strip()
                elif line.startswith('VISUAL_ELEMENTS:'):
                    current_section = 'visual_elements'
                    analysis_parts['visual_elements'] = line.replace('VISUAL_ELEMENTS:', '').strip()
                elif line.startswith('CONTEXT:'):
                    current_section = 'context'
                    analysis_parts['context'] = line.replace('CONTEXT:', '').strip()
                elif current_section and line:
                    analysis_parts[current_section] += ' ' + line
            
            logger.info(f"Successfully analyzed image from {source}" + (f" page {page}" if page else ""))
            return analysis_parts
            
        except Exception as e:
            logger.error(f"Failed to analyze image with Gemini: {str(e)}")
            return {
                'description': f'Image analysis failed: {str(e)}',
                'ocr_text': '',
                'visual_elements': '',
                'context': '',
                'full_analysis': f'Error analyzing image: {str(e)}'
            }

    def _extract_images_pdf(self, pdf_path: Path, file_name: str) -> int:
        """Extract images from PDF, analyze them, and store with analysis. Returns count of extracted images."""
        extracted_count = 0
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for i, page in enumerate(pdf.pages):
                    page_num = i + 1
                    try:
                        if hasattr(page, 'images') and page.images:
                            for j, img_obj in enumerate(page.images):
                                try:
                                    if 'stream' in img_obj and hasattr(img_obj['stream'], 'get_data'):
                                        img_bytes = img_obj['stream'].get_data()
                                    else:
                                        continue
                                    
                                    img = Image.open(io.BytesIO(img_bytes))
                                    
                                    if img.size[0] < 50 or img.size[1] < 50:
                                        continue
                                    
                                    if img.mode not in ["RGB", "RGBA"]:
                                        img = img.convert("RGB")
                                    
                                    img = self._resize_image(img)
                                    image_id = f"{file_name}_p{page_num}_img{j}_{uuid.uuid4().hex[:8]}"
                                    
                                    # Convert to base64
                                    buffered = io.BytesIO()
                                    img_format = "PNG"
                                    img.save(buffered, format=img_format, optimize=True)
                                    img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

                                    # Analyze image with Gemini
                                    analysis = self._analyze_image_with_gemini(img_base64, file_name, page_num)
                                    
                                    st.session_state.image_store[image_id] = {
                                        "data": img_base64,
                                        "source": file_name,
                                        "page": page_num,
                                        "type": "png",
                                        "size": img.size,
                                        "analysis": analysis  # Add analysis results
                                    }
                                    
                                    if file_name not in st.session_state.doc_image_map:
                                        st.session_state.doc_image_map[file_name] = []
                                    st.session_state.doc_image_map[file_name].append(image_id)
                                    
                                    extracted_count += 1
                                    logger.info(f"Extracted and analyzed image {image_id} from page {page_num}")
                                    
                                except Exception as img_ex:
                                    logger.warning(f"Failed to process image {j} on page {page_num}: {img_ex}")
                                    continue
                    except Exception as page_ex:
                        logger.warning(f"Error processing page {page_num}: {page_ex}")
                        continue
                        
        except Exception as e:
            logger.error(f"Error extracting images from PDF {file_name}: {e}")
        
        logger.info(f"Extracted and analyzed {extracted_count} images from {file_name}")
        return extracted_count
    
    def _extract_images_docx(self, docx_path: Path, file_name: str) -> int:
        """Extract images from DOCX, analyze them, and store with analysis. Returns count of extracted images."""
        extracted_count = 0
        try:
            document = docx.Document(docx_path)
            
            for rel_id, rel in document.part.rels.items():
                if "image" in rel.target_ref:
                    try:
                        img_part = rel.target_part
                        img_bytes = img_part.blob
                        img = Image.open(io.BytesIO(img_bytes))
                        
                        if img.size[0] < 50 or img.size[1] < 50:
                            continue
                        
                        if img.mode not in ["RGB", "RGBA"]:
                            img = img.convert("RGB")
                        
                        img = self._resize_image(img)
                        image_id = f"{file_name}_img{extracted_count}_{uuid.uuid4().hex[:8]}"

                        # Convert to base64
                        buffered = io.BytesIO()
                        img_format = "PNG"
                        img.save(buffered, format=img_format, optimize=True)
                        img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

                        # Analyze image with Gemini
                        analysis = self._analyze_image_with_gemini(img_base64, file_name)

                        st.session_state.image_store[image_id] = {
                            "data": img_base64,
                            "source": file_name,
                            "page": None,
                            "type": "png",
                            "size": img.size,
                            "analysis": analysis  # Add analysis results
                        }
                        
                        if file_name not in st.session_state.doc_image_map:
                            st.session_state.doc_image_map[file_name] = []
                        st.session_state.doc_image_map[file_name].append(image_id)
                        
                        extracted_count += 1
                        logger.info(f"Extracted and analyzed image {image_id} from DOCX")
                        
                    except Exception as img_ex:
                        logger.warning(f"Failed to process image with rel_id {rel_id}: {img_ex}")
                        continue
                        
        except Exception as e:
            logger.error(f"Error extracting images from DOCX {file_name}: {e}")
        
        logger.info(f"Extracted and analyzed {extracted_count} images from {file_name}")
        return extracted_count
    
    def upload_and_process_document(self, uploaded_file) -> Tuple[bool, Dict[str, Any]]:
        """Process uploaded document and return success status with stats"""
        temp_file_path = self.TEMP_DIR / uploaded_file.name
        file_type = uploaded_file.type
        file_name = uploaded_file.name
        
        stats = {"text_chunks": 0, "images": 0, "file_size": len(uploaded_file.getvalue())}
        
        try:
            temp_file_path.write_bytes(uploaded_file.getvalue())
            logger.info(f"Processing file: {file_name} ({stats['file_size']} bytes)")

            # Load text content
            docs = []
            if file_type == "application/pdf":
                loader = PDFPlumberLoader(str(temp_file_path))
                docs = loader.load()
                stats["images"] = self._extract_images_pdf(temp_file_path, file_name)
            elif file_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                loader = Docx2txtLoader(str(temp_file_path))
                docs = loader.load()
                stats["images"] = self._extract_images_docx(temp_file_path, file_name)
            else:
                return False, {"error": f"Unsupported file type: {file_type}"}

            # Process text content and create image-based documents
            all_documents = []
            
            if docs:
                # Add metadata to text documents
                for doc in docs:
                    doc.metadata["source"] = file_name
                    doc.metadata["content_type"] = "text"
                    doc.metadata["image_ids"] = st.session_state.doc_image_map.get(file_name, [])
                all_documents.extend(docs)

            # Create documents from image analysis
            if file_name in st.session_state.doc_image_map:
                for img_id in st.session_state.doc_image_map[file_name]:
                    if img_id in st.session_state.image_store:
                        img_info = st.session_state.image_store[img_id]
                        analysis = img_info.get('analysis', {})
                        
                        # Create comprehensive content from image analysis
                        image_content_parts = []
                        
                        if analysis.get('description'):
                            image_content_parts.append(f"Image Description: {analysis['description']}")
                        
                        if analysis.get('ocr_text'):
                            image_content_parts.append(f"Extracted Text (OCR): {analysis['ocr_text']}")
                        
                        if analysis.get('visual_elements'):
                            image_content_parts.append(f"Visual Elements: {analysis['visual_elements']}")
                        
                        if analysis.get('context'):
                            image_content_parts.append(f"Context: {analysis['context']}")
                        
                        if image_content_parts:
                            image_content = "\n\n".join(image_content_parts)
                            
                            # Create a document for the image analysis
                            img_doc = Document(
                                page_content=image_content,
                                metadata={
                                    "source": file_name,
                                    "content_type": "image_analysis",
                                    "image_id": img_id,
                                    "page": img_info.get("page"),
                                    "image_size": img_info.get("size")
                                }
                            )
                            all_documents.append(img_doc)

            # Split documents into chunks
            if all_documents:
                chunks = self.text_splitter.split_documents(all_documents)
                stats["text_chunks"] = len(chunks)
                
                # Fix for the FAISS error - ensure embeddings are 2D
                if chunks:
                    try:
                        # Update vector store with error handling
                        if st.session_state.vector_store is None:
                            st.session_state.vector_store = FAISS.from_documents(chunks, self.embeddings)
                            logger.info("Created new vector store")
                        else:
                            # Add documents one by one to handle potential embedding issues
                            texts = [doc.page_content for doc in chunks]
                            metadatas = [doc.metadata for doc in chunks]
                            
                            # Generate embeddings first to check dimensionality
                            embeddings_list = self.embeddings.embed_documents(texts)
                            
                            # Ensure embeddings are properly formatted
                            if embeddings_list and len(embeddings_list) > 0:
                                st.session_state.vector_store.add_texts(texts, metadatas=metadatas)
                                logger.info("Added documents to existing vector store")
                            else:
                                logger.error("Failed to generate embeddings for documents")
                                return False, {"error": "Failed to generate embeddings"}
                                
                    except Exception as embed_error:
                        logger.error(f"Embedding error: {str(embed_error)}")
                        return False, {"error": f"Embedding failed: {str(embed_error)}"}
                
                self.vector_store = st.session_state.vector_store

            # Update processing stats
            st.session_state.processing_stats["total_chunks"] += stats["text_chunks"]
            st.session_state.processing_stats["total_images"] += stats["images"]
            
            logger.info(f"Successfully processed {file_name}: {stats['text_chunks']} chunks, {stats['images']} images with analysis")
            return True, stats

        except Exception as e:
            logger.error(f"Document processing failed for {file_name}: {str(e)}", exc_info=True)
            return False, {"error": str(e)}
        finally:
            if temp_file_path.exists():
                try:
                    temp_file_path.unlink()
                except OSError as e:
                    logger.error(f"Error removing temp file: {e}")

    def similarity_search(self, query: str, top_k: int = 5) -> List[Document]:
        """Search for relevant documents"""
        if not query.strip():
            return []
        
        if st.session_state.vector_store is None:
            logger.warning("No documents loaded for search")
            return []
        
        try:
            results = st.session_state.vector_store.similarity_search(
                query, k=top_k, fetch_k=min(20, top_k * 4)
            )
            logger.info(f"Found {len(results)} relevant chunks for query")
            return results
        except Exception as e:
            logger.error(f"Similarity search failed: {str(e)}")
            return []

    def generate_answer(self, user_query: str, context_docs: List[Document]) -> Tuple[str, List[Dict[str, Any]], float]:
        """Generate answer using context. Returns answer, context images, and processing time."""
        start_time = time.time()
        
        if not self.llm:
            return "LLM not initialized. Cannot generate answer.", [], 0.0

        try:
            # Prepare context text including image analysis
            context_parts = []
            image_analysis_parts = []
            
            for doc in context_docs:
                source = doc.metadata.get('source', 'Unknown')
                content_type = doc.metadata.get('content_type', 'text')
                
                if content_type == 'image_analysis':
                    image_analysis_parts.append(f"Image Analysis from {source}: {doc.page_content}")
                else:
                    context_parts.append(f"Source: {source}\n{doc.page_content}")
            
            # Combine text and image analysis
            context_text = "\n\n".join(context_parts)
            if image_analysis_parts:
                context_text += "\n\n--- IMAGE ANALYSIS ---\n\n" + "\n\n".join(image_analysis_parts)
            
            # Get relevant images with their analysis
            relevant_image_ids = set()
            for doc in context_docs:
                if doc.metadata.get('content_type') == 'image_analysis':
                    relevant_image_ids.add(doc.metadata.get('image_id'))
                else:
                    relevant_image_ids.update(doc.metadata.get("image_ids", []))
            
            context_images_info = []
            image_context_text = ""
            
            if relevant_image_ids:
                image_context_text = f"\n\nAdditionally, there are {len(relevant_image_ids)} analyzed images from the documents with extracted text and visual descriptions."
                for img_id in list(relevant_image_ids)[:5]:  # Limit to 5 images
                    if img_id in st.session_state.image_store:
                        img_info = st.session_state.image_store[img_id]
                        context_images_info.append({
                            "id": img_id,
                            "source": img_info["source"],
                            "page": img_info.get("page"),
                            "data": img_info["data"],
                            "type": img_info["type"],
                            "size": img_info["size"],
                            "analysis": img_info.get("analysis", {})  # Include analysis
                        })

            # Create enhanced prompt
            prompt_template = ChatPromptTemplate.from_template("""
    You are a helpful assistant analyzing documents with both text and visual content. Answer the user's question based ONLY on the provided context, which includes both text content and detailed image analysis with OCR results.

    Context:
    {context}
    {image_context}

    Important Instructions:
    - Use information from both text content and image analysis (descriptions, OCR text, visual elements)
    - If images contain text or data relevant to the question, incorporate that information
    - Be specific and cite sources when possible
    - If the context doesn't contain enough information to answer the question, say so clearly
    - When referencing image content, mention that it comes from image analysis

    Question: {question}

    Answer:""")

            # Generate response
            chain = prompt_template | self.llm | StrOutputParser()
            
            answer = chain.invoke({
                "context": context_text,
                "image_context": image_context_text,
                "question": user_query
            })
            
            processing_time = time.time() - start_time
            logger.info(f"Generated answer with image analysis in {processing_time:.2f}s")
            
            return answer, context_images_info, processing_time

        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Answer generation failed: {str(e)}", exc_info=True)
            return f"I encountered an error while generating the answer: {str(e)}", [], processing_time
    
def setup_streamlit_ui():
    """Configure Streamlit UI with enhanced styling"""
    st.set_page_config(
        page_title="📄 Enhanced Document Chat",
        page_icon="📄",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.markdown("""
        <style>
        .stApp {
            max-width: 1400px;
            margin: 0 auto;
        }
        .main-header {
            text-align: center;
            padding: 1rem 0;
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 10px;
            margin-bottom: 2rem;
        }
        .stats-container {
            display: flex;
            justify-content: space-around;
            margin: 1rem 0;
        }
        .stat-box {
            text-align: center;
            padding: 1rem;
            background: #f8f9fa;
            border-radius: 8px;
            border-left: 4px solid #667eea;
        }
        .stat-number {
            font-size: 2rem;
            font-weight: bold;
            color: #667eea;
        }
        .stat-label {
            font-size: 0.9rem;
            color: #6c757d;
        }
        .chat-message {
            padding: 1rem;
            border-radius: 10px;
            margin-bottom: 1rem;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .chat-message.user {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }
        .chat-message.assistant {
            background: #f8f9fa;
            border: 1px solid #dee2e6;
        }
        .context-section {
            background: #f1f3f4;
            padding: 1rem;
            border-radius: 8px;
            margin-top: 1rem;
        }
        .image-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(120px, 1fr));
            gap: 10px;
            margin-top: 10px;
        }
        .context-image {
            width: 100%;
            height: 80px;
            object-fit: cover;
            border-radius: 4px;
            border: 2px solid #dee2e6;
            cursor: pointer;
            transition: border-color 0.2s;
        }
        .context-image:hover {
            border-color: #667eea;
        }
        .processing-time {
            font-size: 0.8rem;
            color: #6c757d;
            font-style: italic;
        }
        .sidebar-section {
            background: #f8f9fa;
            padding: 1rem;
            border-radius: 8px;
            margin-bottom: 1rem;
        }
        </style>
    """, unsafe_allow_html=True)

def main():
    setup_streamlit_ui()
    
    # Header
    st.markdown("""
        <div class="main-header">
            <h1>📄 Document Chat Assistant</h1>
            <p>Upload PDF or DOCX files with images and chat with your documents using AI</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Initialize system
    try:
        qa_system = DocumentQASystem()
    except ValueError:
        st.stop()
    except Exception as e:
        st.error(f"Critical error during initialization: {e}")
        st.stop()

    # Sidebar
    with st.sidebar:
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.markdown("### 📁 Document Manager")
        st.markdown("Upload PDF or Word documents to analyze their content and images.")
        
        uploaded_files = st.file_uploader(
            "Choose files",
            type=["pdf", "docx"],
            accept_multiple_files=True,
            help="Upload PDF or DOCX files (max 200MB each)",
            key=st.session_state["file_uploader_key"]
        )
        
        col1, col2 = st.columns(2)
        with col1:
            process_btn = st.button("🚀 Process Files", type="primary", use_container_width=True)
        with col2:
            reset_btn = st.button("🗑️ Reset All", use_container_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Process files
        if process_btn and uploaded_files:
            files_to_process = [f for f in uploaded_files if f.name not in st.session_state.uploaded_files_info]
            
            if not files_to_process:
                st.info("ℹ️ No new files to process.")
            else:
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                success_count = 0
                for i, uploaded_file in enumerate(files_to_process):
                    status_text.text(f"Processing {uploaded_file.name}...")
                    progress_bar.progress((i) / len(files_to_process))
                    
                    success, stats = qa_system.upload_and_process_document(uploaded_file)
                    
                    if success:
                        st.session_state.uploaded_files_info[uploaded_file.name] = {
                            "type": uploaded_file.type,
                            "stats": stats
                        }
                        success_count += 1
                
                progress_bar.progress(1.0)
                status_text.text(f"✅ Processed {success_count}/{len(files_to_process)} files")
                
                if success_count > 0:
                    st.success(f"Successfully processed {success_count} document(s)!")
                    time.sleep(1)
                    st.rerun()

        # Reset functionality
        if reset_btn:
            reset_all_states()
            st.success("🔄 Session reset successfully!")
            time.sleep(1)
            st.rerun()

        # Show processed documents
        if st.session_state.uploaded_files_info:
            st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
            st.markdown("### 📚 Processed Documents")
            
            for name, info in st.session_state.uploaded_files_info.items():
                with st.expander(f"📄 {name}"):
                    file_type = info["type"].split("/")[-1].upper()
                    st.write(f"**Type:** {file_type}")
                    
                    if "stats" in info:
                        stats = info["stats"]
                        st.write(f"**Text chunks:** {stats.get('text_chunks', 0)}")
                        st.write(f"**Images found:** {stats.get('images', 0)}")
                        
                        if stats.get('file_size'):
                            size_mb = stats['file_size'] / 1024 / 1024
                            st.write(f"**Size:** {size_mb:.1f} MB")
            
            st.markdown('</div>', unsafe_allow_html=True)

        # # Show statistics
        # if st.session_state.processing_stats["total_chunks"] > 0:
        #     st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        #     st.markdown("### 📊 Collection Stats")
            
        #     stats = st.session_state.processing_stats
        #     col1, col2 = st.columns(2)
            
        #     with col1:
        #         st.metric("Text Chunks", stats["total_chunks"])
        #     with col2:
        #         st.metric("Images", stats["total_images"])
            
        #     st.markdown('</div>', unsafe_allow_html=True)

    # Main chat interface
    if not st.session_state.uploaded_files_info:
        st.info("👆 Please upload documents using the sidebar to get started!")
    else:
        # Display chat history
        for i, chat in enumerate(st.session_state.chat_history):
            # User message
            with st.chat_message("user"):
                st.markdown(chat.question)
                st.markdown(f'<div class="processing-time">Asked at {chat.timestamp}</div>', 
                           unsafe_allow_html=True)
            
            # Assistant message
            with st.chat_message("assistant"):
                st.markdown(chat.answer)
                
                if chat.processing_time > 0:
                    st.markdown(f'<div class="processing-time">Processed in {chat.processing_time:.1f}s</div>', 
                            unsafe_allow_html=True)
                
                # Enhanced context expander with image analysis
                with st.expander("📋 View Context & Sources"):
                    if chat.context_text:
                        st.markdown("**📝 Text Context:**")
                        st.markdown(chat.context_text)
                    
                    if chat.context_images:
                        st.markdown("**🖼️ Related Images with Analysis:**")
                        
                        # Display images with their analysis
                        for idx, img_info in enumerate(chat.context_images):
                            col1, col2 = st.columns([1, 2])
                            
                            with col1:
                                img_data = f"data:image/{img_info['type']};base64,{img_info['data']}"
                                st.image(
                                    img_data,
                                    caption=f"From {img_info['source']}" + 
                                        (f" (Page {img_info['page']})" if img_info['page'] else ""),
                                    width=300
                                )
                            
                            with col2:
                                # Display image analysis if available
                                if 'analysis' in img_info:
                                    analysis = img_info['analysis']
                                    
                                    if analysis.get('description'):
                                        st.markdown("**🔍 Description:**")
                                        st.markdown(analysis['description'])
                                    
                                    if analysis.get('ocr_text'):
                                        st.markdown("**📝 Extracted Text:**")
                                        st.markdown(f"`{analysis['ocr_text']}`")
                                    
                                    if analysis.get('visual_elements'):
                                        st.markdown("**📊 Visual Elements:**")
                                        st.markdown(analysis['visual_elements'])
                                    
                                    if analysis.get('context'):
                                        st.markdown("**🎯 Context:**")
                                        st.markdown(analysis['context'])
                                else:
                                    st.markdown("*Image analysis not available*")
                            
                            if idx < len(chat.context_images) - 1:
                                st.divider()


        # Chat input
        if prompt := st.chat_input("💬 Ask a question about your documents..."):
            if not st.session_state.vector_store and not st.session_state.image_store:
                st.warning("⚠️ Please upload and process documents before asking questions.")
            else:
                # Show user message immediately
                with st.chat_message("user"):
                    st.markdown(prompt)
                
                # Generate response
                with st.chat_message("assistant"):
                    with st.spinner("🤔 Analyzing documents..."):
                        # Search for relevant content
                        context_docs = qa_system.similarity_search(prompt, top_k=5)
                        
                        # Generate answer
                        answer, context_images, processing_time = qa_system.generate_answer(prompt, context_docs)
                        
                        # Display answer
                        st.markdown(answer)
                        
                        # Prepare context summary
                        context_summary = ""
                        if context_docs:
                            context_summary = "\n\n".join([
                                f"**{doc.metadata.get('source', 'Unknown')}:** {doc.page_content[:200]}..."
                                for doc in context_docs[:3]
                            ])
                        else:
                            context_summary = "No relevant text context found."
                        
                        # Save to chat history
                        st.session_state.chat_history.append(
                            ChatMessage(
                                timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                question=prompt,
                                answer=answer,
                                context_text=context_summary,
                                context_images=context_images,
                                processing_time=processing_time
                            )
                        )
                
                # Rerun to show the conversation properly
                st.rerun()

    # Footer
    st.markdown("---")
    st.markdown(
        "*💓 Built with love. *"
    )

if __name__ == "__main__":
    main()