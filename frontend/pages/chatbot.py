import streamlit as st
import threading
import time
import requests
import json
import os
import uuid
from datetime import datetime
# from llama_index.readers.docling import DoclingReader
from docling.document_converter import DocumentConverter
import tempfile
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
from llama_index.core import PromptTemplate
from llama_index.core import VectorStoreIndex, Document
# from langchain_experimental.text_splitter import SemanticSplitterNodeParser
from langchain_experimental.text_splitter import SemanticChunker
from bs4 import BeautifulSoup
import html
import re
import logging
# Retrievers
from llama_index.core.retrievers import (
    BaseRetriever,
    VectorIndexRetriever,
    KeywordTableSimpleRetriever,
)
# class CustomRetriever(BaseRetriever):
#     """Custom retriever that performs both semantic search and hybrid search."""

#     def __init__(
#         self,
#         vector_retriever: VectorIndexRetriever,
#         keyword_retriever: KeywordTableSimpleRetriever,
#         mode: str = "AND",
#     ) -> None:
#         """Init params."""

#         self._vector_retriever = vector_retriever
#         self._keyword_retriever = keyword_retriever
#         if mode not in ("AND", "OR"):
#             raise ValueError("Invalid mode.")
#         self._mode = mode
#         super().__init__()

#     def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
#         """Retrieve nodes given query."""

#         vector_nodes = self._vector_retriever.retrieve(query_bundle)
#         keyword_nodes = self._keyword_retriever.retrieve(query_bundle)

#         vector_ids = {n.node.node_id for n in vector_nodes}
#         keyword_ids = {n.node.node_id for n in keyword_nodes}

#         combined_dict = {n.node.node_id: n for n in vector_nodes}
#         combined_dict.update({n.node.node_id: n for n in keyword_nodes})

#         if self._mode == "AND":
#             retrieve_ids = vector_ids.intersection(keyword_ids)
#         else:
#             retrieve_ids = vector_ids.union(keyword_ids)

#         retrieve_nodes = [combined_dict[rid] for rid in retrieve_ids]
#         return retrieve_nodes
# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s', filename='chat_debug.log', filemode='w')
# Định nghĩa class xử lý tài liệu
# groq_api_key = "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"
# api_key = "AIzaSyBmy0AA0EVuTioK2gs0F1CI84HphjBFbRE" 
@st.cache_resource
def load_embedding_model():
    """Loads the embedding model once and caches it."""
    return HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

class DocummentProcessor():
    def __init__(self,uploaded_file):
        self.uploaded_file = uploaded_file
        self.file_path = self.mkdir()

    def mkdir(self):
        # Define a fixed temporary directory
        TEMP_DIR = "./temp_uploads"
        if not os.path.exists(TEMP_DIR):
            # Ensure the directory exists
            os.makedirs(TEMP_DIR, exist_ok=True)
        else:
            # Tạo đường dẫn đến file tạm thời
            file_path = os.path.join(TEMP_DIR, self.uploaded_file.name)

            # Lưu file vào thư mục tạm thời
            # with open(file_path, "wb") as f:
            #     f.write(self.uploaded_file.getvalue())
            return file_path

    def convert_document(self):
        """Function to convert a document."""
        converter = DocumentConverter()
        try:
            logging.debug(f"Uploaded Source file: {self.file_path}")
            result = converter.convert(self.file_path)
            markdown_text = result.document.export_to_markdown()
            logging.debug(f"Markdown text length: {len(markdown_text)}")
            return markdown_text
        except Exception as e:
            logging.error(f"Error processing file {self.file_path}: {e}")
            return None   

# def clean_output(raw_text):
#     # Giải mã chuỗi Unicode bị lỗi
#     fixed_text = raw_text.encode('utf-8', 'ignore').decode('utf-8')

#     # Xóa HTML bằng BeautifulSoup
#     soup = BeautifulSoup(fixed_text, "html.parser")
#     cleaned_text = soup.get_text(separator=" ", strip=True)

#     # Giải mã các ký tự escape (nếu có)
#     cleaned_text = html.unescape(cleaned_text)

#     return cleaned_text

def clean_output(raw_text):
    # Fix incorrectly encoded Unicode characters
    fixed_text = bytes(raw_text, "utf-8").decode("utf-8", "ignore")

    # Remove HTML tags
    soup = BeautifulSoup(fixed_text, "html.parser")
    cleaned_text = soup.get_text(separator=" ", strip=True)

    # Decode escaped HTML entities
    cleaned_text = html.unescape(cleaned_text)

    # Remove any lingering Unicode control characters
    cleaned_text = re.sub(r'[\u200b-\u200f\u202a-\u202e]', '', cleaned_text)

    # Normalize spaces: remove excessive spaces, newlines, and fix non-breaking spaces
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()

    return cleaned_text
           
class ModelManagement():
    """
    List of support models:
    https://docs.llamaindex.ai/en/stable/api_reference/llms/
    """
    def __init__(self, model_name,api_key):
        self.api_key = api_key
        self.model_name = model_name
        self.llm = self.load_model()

    def load_model(self):
        if self.model_name.startswith("gemini"):
            from llama_index.llms.gemini import Gemini
            model = Gemini(model_name="models/gemini-1.5-flash", api_key=self.api_key)
            return model
        if self.model_name.startswith("llama"):
            from llama_index.llms.groq import Groq
            # Initialize Groq LLM
            model = Groq(model="llama3-8b-8192", api_key=self.api_key)
            return model
            
        
    def run_model(self, user_input):
        response = self.llm.complete(user_input)
        return response

class Ragmodule_v1():
    def __init__(self):
        self.model_name = "sentence-transformers/all-MiniLM-L6-v2"

    def build_rag_index(self,llm,document_text:str):
        """Create a RAG index from text."""
        from llama_index.core import VectorStoreIndex, Document
        # from llama_index.core.node_parser import SimpleNodeParser
        try:
            from llama_index.node_parser import SemanticSplitterNodeParser
        except ImportError:
            from llama_index.core.node_parser import SemanticSplitterNodeParser

        logging.debug(f"Document text type: {type(document_text)}")
        # Split document into paragraphs
        docs = [Document(text=paragraph) for paragraph in document_text.split("\n\n")]
        # Use Sentence Splitting (More Granular/ Single Document List/ Paragraph Split/ Sentence Tokenization/SentenceSplitter (LlamaIndex)
        
        # docs = Document(text=document_text)
        embed_model = HuggingFaceEmbedding(model_name=self.model_name)

        # Creating an index over loaded data
        Settings.embed_model = embed_model

        # Use a node parser for **semantic chunking**
        # node_parser = SimpleNodeParser.from_defaults(chunk_size=512, chunk_overlap=50)
        # 2️⃣ **Initialize Semantic Chunking Parser**
        # You can use OpenAI or Hugging Face models for chunking
        node_parser = SemanticSplitterNodeParser(
            # llm=llm,  # Can replace with Hugging Face or another LLM
            embed_model=embed_model,
        )
        
        # Use **Hugging Face embeddings**
        # 3️⃣ **Parse documents into nodes**
        # node_parser = SimpleNodeParser()
        nodes = node_parser.get_nodes_from_documents(docs)
        logging.debug(f"==============================Add embedding start==============================")
        # 4️⃣ **Create a VectorStoreIndex using the embedding model**
        index = VectorStoreIndex.from_documents(
            documents=docs, 
            transformations=[node_parser], 
            embed_model=embed_model,  # 🔥 Use the Hugging Face embedding model here
            show_progress=True
        )
        logging.debug(f"==============================Add embedding end==============================")

        # Save index to avoid re-embedding later
        # index.storage_context.persist(persist_dir="saved_index")

        return index# pip install llama-index-llms-openai
    
    def query_rag(self,index,query):
        from llama_index.core.query_engine import RetrieverQueryEngine
        from llama_index.core.retrievers import VectorIndexRetriever
        from llama_index.core import VectorStoreIndex, get_response_synthesizer
        from llama_index.llms.openai import OpenAI
        # https://docs.llamaindex.ai/en/stable/module_guides/deploying/query_engine/usage_pattern/
        """Retrieve relevant info and generate answers."""
        # retriever = index.as_retriever(similarity_top_k=3)
        # configure retriever
        retriever = VectorIndexRetriever(
            index=index,
            similarity_top_k=3,
        )
        # configure response synthesizer
        response_synthesizer = get_response_synthesizer(
            response_mode="tree_summarize",
        )
        # 5️⃣ **Create a RetrieverQueryEngine**
        query_engine = RetrieverQueryEngine(retriever=retriever,
                                            response_synthesizer=response_synthesizer,)
        # response = query_engine.query(query)
        return response

class Ragmodule_v2():
    def __init__(self):
        self.embed_model = load_embedding_model()  # ✅ Cached model

    def build_rag_index(self, llm, document_text: str):
        """Create a RAG index from text."""
        from llama_index.core import VectorStoreIndex, Document
        try:
            from llama_index.node_parser import SemanticSplitterNodeParser
        except ImportError:
            from llama_index.core.node_parser import SemanticSplitterNodeParser

        logging.debug(f"Document text type: {type(document_text)}")

        # Split document into paragraphs
        docs = [Document(text=paragraph) for paragraph in document_text.split("\n\n")]

        # Check if embeddings are already computed
        if "cached_embeddings" not in st.session_state:
            st.session_state.cached_embeddings = {}

        logging.debug(f"cache embedding:{st.session_state.cached_embeddings}=============================")
        # Compute embeddings only for new documents
        for doc in docs:
            if doc.text not in st.session_state.cached_embeddings:
                st.session_state.cached_embeddings[doc.text] = self.embed_model.get_text_embedding(doc.text)

        # Use cached embeddings
        nodes = []
        for doc in docs:
            nodes.append(st.session_state.cached_embeddings[doc.text])

        # Use precomputed embeddings instead of recomputing
        index = VectorStoreIndex.from_documents(
            documents=docs,
            embed_model=self.embed_model,
            show_progress=True
        )

        logging.debug(f"==============================Add embedding end==============================")
        return index

class TextSplitterModule():
    def __init__(self):
        # self.embed_model = load_embedding_model()
        pass

    def split_document(self,document_text, method="semantic", chunk_size=1000, chunk_overlap=100,debug=True):
        """
        Experiment with different splitting methods.
        :param document_text: Raw document text.
        :param method: "recursive" (RecursiveCharacterTextSplitter), "token" (TokenTextSplitter), or "custom".
        :param chunk_size: Size of each chunk.
        :param chunk_overlap: Overlap between chunks.
        :return: List of split text chunks.
        """
        from langchain.text_splitter import RecursiveCharacterTextSplitter, TokenTextSplitter
        if method == "recursive":
            splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        elif method == "token":
            splitter = TokenTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        elif method == "paragraph":
            chunk_length = len([chunk for chunk in self.custom_split(document_text)])
            return self.custom_split(document_text)
        elif method == "semantic":
            from langchain.embeddings.huggingface import HuggingFaceEmbeddings
            # Initialize Hugging Face Embeddings
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            # Use Semantic Splitter
            splitter = SemanticChunker(embeddings)#SemanticSplitterNodeParser()
        elif method == "spacing":
            # def chunk_text(text, chunk_size, overlap, split_on_whitespace_only=True):
            # reference: https://github.com/tomasonjo/kg-rag/blob/main/notebooks/utils.py
            chunks = []
            split_on_whitespace_only=True
            index = 0
            overlap = chunk_overlap
            text = document_text
            while index < len(text):
                if split_on_whitespace_only:
                    prev_whitespace = 0
                    left_index = index - overlap
                    while left_index >= 0:
                        if text[left_index] == " ":
                            prev_whitespace = left_index
                            break
                        left_index -= 1
                    next_whitespace = text.find(" ", index + chunk_size)
                    if next_whitespace == -1:
                        next_whitespace = len(text)
                    chunk = text[prev_whitespace:next_whitespace].strip()
                    chunks.append(chunk)
                    index = next_whitespace + 1
                else:
                    start = max(0, index - overlap + 1)
                    end = min(index + chunk_size + overlap, len(text))
                    chunk = text[start:end].strip()
                    chunks.append(chunk)
                    index += chunk_size
            return chunks
        else:
            raise ValueError("Invalid method. Choose 'recursive', 'token', or 'custom'.")
        if debug:
            logging.info(f"==============================Split document==============================")
            if method == "paragraph":
                logging.debug(f"Split method: {method} | Chunk size: {chunk_length}")
            else:
                logging.debug(f"Split method: {method} | Chunk size: {chunk_size} | Overlap: {chunk_overlap}")

        return splitter.split_text(document_text)

    def custom_split(self, document_text):
        paragraph = document_text.split("\n\n")
        return paragraph

class Ragmodule():
    def __init__(self,text_splitter_module:TextSplitterModule):
        self.embed_model = load_embedding_model()  # ✅ Cached model
        self.text_splitter = text_splitter_module

    def build_rag_index(self, llm, document_text: str):
        """Create a RAG index from text."""
        from llama_index.core import VectorStoreIndex, Document
        try:
            from llama_index.node_parser import SemanticSplitterNodeParser
        except ImportError:
            from llama_index.core.node_parser import SemanticSplitterNodeParser


        # Split document into paragraphs
        # docs = [Document(text=paragraph) for paragraph in chunk]

        chunks = self.text_splitter.split_document(document_text)
        docs = [Document(text=chunk) for chunk in chunks]
        
        # Chia text theo từng đoạn
        # docs = [Document(text=paragraph) for paragraph in chunk]

        # Check if embeddings are already computed
        if "cached_embeddings" not in st.session_state:
            st.session_state.cached_embeddings = {}

        # logging.debug(f"cache embedding:{st.session_state.cached_embeddings}=============================")

        # Compute embeddings only for new documents
        for doc in docs:
            if doc.text not in st.session_state.cached_embeddings:
                st.session_state.cached_embeddings[doc.text] = self.embed_model.get_text_embedding(doc.text)

        # Use cached embeddings
        nodes = []
        for doc in docs:
            nodes.append(st.session_state.cached_embeddings[doc.text])

        # Use precomputed embeddings instead of recomputing
        index = VectorStoreIndex.from_documents(
            documents=docs,
            embed_model=self.embed_model,
            show_progress=True
        )

        # logging.debug(f"==============================Add embedding end==============================")
        return index

def extract_chat_history(chat_session,session_name):
    chat_history_html = ""
    # Icons for user and bot
    USER_ICON = "👤"
    BOT_ICON = "🤖"

    # Mới mở chat không có cuộc trò chuyện
    if (len(chat_session[session_name])) ==0:
        return "Hiện tại chưa có cuộc trò chuyện mới"
    logging.debug(f"Full message: {chat_session[session_name]} ")
    # Duyệt qua từng tin nhắn trong cuộc trò chuyện
    for message in chat_session[session_name]:
        logging.debug(f"Check extracted message: {message} ")
        user_msg, bot_msg = message[0], message[1]

        # Hậu xử lý trả lời của bot
        bot_msg = clean_output(bot_msg)

        logging.debug(f"User/Bot message: {user_msg}, {bot_msg} ")

        # HTML & CSS for right-aligning user messages
        user_msg_html = f"""
        <div style='text-align: right; margin-bottom: 10px;'>
            <span style='background-color: #4F46E5; padding: 8px 12px; border-radius: 10px; color: white;'>
                {USER_ICON}{user_msg}
            </span>
        </div>
        """
        # Thêm khoảng trắng giữa tin nhắn của người dùng và bot
        space_html = "<div style='height: 15px;'></div>"

        # HTML & CSS for left-aligning bot messages
        bot_msg_html = f"""
        <div style='text-align: left; margin-bottom: 10px;'>
            <span style='background-color: #374151; padding: 8px 12px; border-radius: 10px; color: white;'>
                {BOT_ICON}{bot_msg}
            </span>
        </div>
        """
        chat_history_html += user_msg_html + space_html + bot_msg_html + space_html
        logging.debug(f"Chat history HTML: {chat_history_html}")
    return chat_history_html


 
def render_chat_UI():
    # Streamlit App
    st.title("Chatbot with Streamlit")
    st.sidebar.header("Load mô hình và API Key")
    api_key = st.sidebar.text_input("Nhập/Dán API Key của mô hình", type="password")
    model_name = st.sidebar.selectbox("Chọn mô hình", ["gemini-1.5-flash", "llama3-8b-8192","llama-3.3-70b-versatile"])
    markdown_text  = " "

    # Định nghĩa biến markdown text global
    if "markdown_text" not in st.session_state:
        st.session_state.markdown_text = ""

    if "session_name" not in st.session_state:
        st.session_state.session_name = " "

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if "session_list" not in st.session_state:
        st.session_state.session_list = {'default': []}

    if "available_session_list" not in st.session_state:
        st.session_state.available_session_list = []

    st.sidebar.header("Phiên trò chuyện")
    # Khi nhấn nút tạo phiên trò chuyện mới 
    if st.sidebar.button("Tạo phiên trò chuyện mới"):
        # Tạo ra tên của phiên trò chuyện
        session_name = f"Chat_{len(st.session_state.available_session_list)}"
        # Kiểm tra xem tên phiên trò chuyện này có trong danh sách phiên chat không
        if session_name in st.session_state.available_session_list:
            st.error(f"Session name '{session_name}' already exists!")
        else:
            st.session_state.available_session_list.append(session_name)

    # Chuyển danh sách phiên trò chuyện đang tồn tại
    session_list = list(st.session_state.available_session_list)

    # Chọn phiên trò chuyện
    session_name = st.sidebar.selectbox("Lịch sử trò chuyện", session_list)

    # Kiểm tra phiên chat đang được chọn
    if session_name != st.session_state.session_name:
        st.session_state.session_name = session_name

    if session_name not in st.session_state.session_list:
        st.session_state.session_list[session_name] = []
    
    # st.text_area("Chat History", chat_history_text, height=300, disabled=False)
    # ==========================================Luồng 2: Xử lý tài liệu upload lên=========================================================#
    # reference https://github.com/patchy631/ai-engineering-hub/blob/main/rag-with-dockling/app.py
    # Tạo form nhập tin nhắn và tải tệp lên
    # with st.expander("Chat With file"):
    with st.sidebar:
        st.header("Tải tệp lên")
        uploaded_file = st.file_uploader("Chọn một file để tải lên", type=["txt", "json", "csv",'pdf','docx','xlsx'])
        if uploaded_file:
            if "processed_file" in st.session_state and st.session_state.processed_file == uploaded_file.name:
                st.write("Tài liệu đã được tải trước đó, không cần xử lý lại.")
            else:
                st.session_state.processed_file = uploaded_file.name
                # st.write("Chờ một lát để nhúng file vào mô hình")
                # logging.debug(f"==============================Uploaded file==============================")
                # Khởi taọ đối tượng xử lý tài liệu
                document_processor = DocummentProcessor(uploaded_file)

                # Chuyển đổi tài liệu thành markdown
                markdown_text = document_processor.convert_document()

                # Khởi tạo đối tượng quản lý mô hình
                llm_modules = ModelManagement(model_name,api_key)

                # Khởi tạo mô hình
                model = llm_modules.load_model()

                # Cài đặt mặc định cho mô hình vừa tạo
                Settings.llm = model

                # Khởi tạo mô dule RAG
                # rag_modules = Ragmodule()

                # 28/02/2025: Update textsplitter module
                text_splitter = TextSplitterModule()
                rag_modules = Ragmodule(text_splitter)

                # Tạo index cho tài liệu
                index = rag_modules.build_rag_index(llm=model,document_text=markdown_text)
                # logging.debug(f"Index: {index}")
                # logging.debug(f"==============================Done index==============================")

                # Tạo query engine, sử dụng cohere reranker trên các node đã lấy được
                # ✅ Create query engine (Check for None)
                if index:
                    st.session_state.query_engine = index.as_query_engine() # ✅ Save query engine
                else:
                    st.error("Error: Index could not be created.")

    # ====================================== Luồng 1: Xử lý tin nhắn ======================================#
    # Hiển thị tin nhắn và lịch sử trong form
    with st.form("chat_form", clear_on_submit=True):
        # Khi người dùng nhập tin nhắn
        user_input = st.text_input("Nhập tin nhắn của bạn:")
        # Tạo nút gửi trong form
        submitted = st.form_submit_button("Gửi")
        # markdown_text = st.session_state.markdown_text
        # Nếu sự kiện gửi được kích hoạt
        chat_history_html = extract_chat_history(st.session_state.session_list,session_name) 
        st.markdown(chat_history_html, unsafe_allow_html=True)

        if submitted: # 
            if "query_engine" not in st.session_state:
                st.error("Query engine is not initialized. Please upload a document first.")

            query_engine = st.session_state.query_engine  # Retrieve from session state
            # logging.debug(f"Query engine: {query_engine}")
            # ====== Tạo promp template ======#
            qa_prompt_tmpl_str = (
            # "Dưới đây là ngữ cảnh.\n"
            # "---------------------\n"
            # "{context_str}\n"
            # "---------------------\n"
            # "Dựa trên ngữ cảnh trên hãy đọc query dưới đây suy nghĩ và trả lời bằng tiếng việt. Câu nào không biết thì trả lời tôi không biết, đừng bịa.\n"
            # """ Trường hợp người dùng hỏi những câu không liên quan đến ngữ cảnh và bạn không biết, hãy dẫn dắt họ hỏi về những câu liên quan trong nội dung trong ngữ cảnh\n"""
            # "Query: {query_str}\n"
            # "Answer: "
            "Dưới đây là ngữ cảnh.\n"  
            "---------------------  "
            "{context_str}" 
            "--------------------- " 
            """ Dựa trên ngữ cảnh trên, hãy đọc query dưới đây, suy nghĩ và trả lời bằng tiếng Việt. 
            - Nếu câu hỏi thuộc ngữ cảnh, hãy trả lời chính xác dựa trên thông tin có sẵn.  
            - Nếu không biết câu trả lời, hãy nói "Tôi không biết" và không bịa đặt.  
            - Nếu câu hỏi không liên quan đến ngữ cảnh, hãy hướng người dùng đặt câu hỏi liên quan đến nội dung ngữ cảnh đã cho.
            """   
            "Query: {query_str}" 
            "Answer:"
            )
            qa_prompt_tmpl = PromptTemplate(qa_prompt_tmpl_str)
            # Cập nhật prompt cho query engine
            query_engine.update_prompts(
                        {"response_synthesizer:text_qa_template": qa_prompt_tmpl}
                    )
            # logging.debug(f"After update query engine: {query_engine}")
            logging.debug(f"Prompt template: {qa_prompt_tmpl}")
            
            # Chat bot phản hồi lại người dùng
            bot_response = query_engine.query(user_input)

            logging.debug(f"Bot response: {bot_response}")

            # ========================================== DISPLAY CURRENT INPUT ==========================================#
            USER_ICON = "👤"
            BOT_ICON = "🤖"
            user_msg_html = f"""
            <div style='text-align: right; margin-bottom: 10px;'>
                <span style='background-color: #4F46E5; padding: 8px 12px; border-radius: 10px; color: white;'>
                    {USER_ICON}{user_input}
                </span>
            </div>
            """
            # Thêm khoảng trắng giữa tin nhắn của người dùng và bot
            space_html = "<div style='height: 15px;'></div>"
            current_message = user_msg_html + space_html
            st.markdown(current_message, unsafe_allow_html=True)
            # ========================================== ENSURE BOT RESPONSE IS STRING ==========================================#
            # Ensure bot_response is a string (extract text if needed)
            if hasattr(bot_response, 'text'):
                bot_msg = bot_response.text  # Extract text if it's an object with a `.text` attribute
                bot_msg = clean_output(bot_msg)
            elif isinstance(bot_response, str):
                pass  # Already a string, do nothing
            else:
                bot_msg = str(bot_response)  # Convert to string as fallback
                bot_msg = clean_output(bot_msg)
            # ========================================== DISPLAY BOT RESPONSE ==========================================#
            # Stream bot response
            bot_placeholder = st.empty()
            streamed_text = ""
            logging.debug(f"Bot response: {bot_msg}")
            for char in bot_msg:
                streamed_text += char
                bot_placeholder.markdown(
                    f"""
                    <div style='text-align: left; margin-bottom: 10px;'>
                        <span style='background-color: #374151; padding: 8px 12px; border-radius: 10px; color: white;'>
                            {BOT_ICON}{streamed_text}
                        </span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                time.sleep(0.02)  # Simulate typing speed

            # Thêm câu hỏi của người dùng và phản hồi của chatbot vào phiên chat hiện tại
            st.session_state.session_list[session_name].append((f"{user_input}", f"{bot_response}"))
            
                


   



       




