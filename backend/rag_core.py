"""backend/rag_core.py - RAG 核心模块"""
import os
import numpy as np
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredWordDocumentLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# 加载环境变量（从 backend 目录向上找父目录的 .env）
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

# ============================================
# 配置
# ============================================
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
if not DASHSCOPE_API_KEY:
    raise ValueError("请在项目根目录的 .env 文件中设置 DASHSCOPE_API_KEY")

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
RETRIEVAL_TOP_K = 4

# ============================================
# 文档加载
# ============================================
def load_documents_from_folder(folder_path=None):
    """加载文件夹内的所有 PDF、TXT 和 Word 文件"""
    if folder_path is None:
        # 默认使用 backend/docs 目录
        folder_path = os.path.join(os.path.dirname(__file__), "docs")
    
    all_documents = []
    
    if not os.path.exists(folder_path):
        os.makedirs(folder_path, exist_ok=True)
        print(f"📁 已创建 {folder_path} 文件夹，请放入你的文档")
        return []
    
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        
        if filename.endswith('.pdf'):
            try:
                loader = PyPDFLoader(file_path)
                all_documents.extend(loader.load())
                print(f"📄 加载 PDF: {filename}")
            except Exception as e:
                print(f"❌ 加载 PDF 失败 {filename}: {e}")
                
        elif filename.endswith('.txt'):
            try:
                for encoding in ['utf-8', 'gbk', 'gb2312']:
                    try:
                        loader = TextLoader(file_path, encoding=encoding)
                        all_documents.extend(loader.load())
                        print(f"📝 加载文本: {filename}")
                        break
                    except UnicodeDecodeError:
                        continue
            except Exception as e:
                print(f"❌ 加载文本失败 {filename}: {e}")
        
        elif filename.endswith('.docx') or filename.endswith('.doc'):
            try:
                loader = UnstructuredWordDocumentLoader(file_path, mode="single")
                all_documents.extend(loader.load())
                print(f"📝 加载 Word: {filename}")
            except Exception as e:
                print(f"❌ 加载 Word 失败 {filename}: {e}")
    
    return all_documents


def split_documents(documents):
    """将文档切分成小块"""
    if not documents:
        return []
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]
    )
    chunks = splitter.split_documents(documents)
    print(f"✂️ 切分成 {len(chunks)} 个文本块")
    return chunks


# ============================================
# TF-IDF 检索器
# ============================================
class TFIDFRetriever:
    def __init__(self):
        self.vectorizer = None
        self.tfidf_matrix = None
        self.chunks = []
    
    def build_index(self, chunks):
        self.chunks = chunks
        if not chunks:
            self.vectorizer = None
            self.tfidf_matrix = None
            print("⚠️ 没有文档块可索引")
            return
        
        texts = [chunk.page_content for chunk in chunks]
        self.vectorizer = TfidfVectorizer(max_features=5000)
        self.tfidf_matrix = self.vectorizer.fit_transform(texts)
        print(f"✅ 索引构建完成，共 {len(chunks)} 个文档块")
    
    def retrieve(self, query, top_k=RETRIEVAL_TOP_K):
        if self.vectorizer is None or self.tfidf_matrix is None:
            return []
        if self.tfidf_matrix.shape[0] == 0:
            return []
        
        query_vec = self.vectorizer.transform([query])
        similarities = np.array(cosine_similarity(query_vec, self.tfidf_matrix)).flatten()
        n_results = min(top_k, len(similarities))
        if n_results == 0:
            return []
        top_indices = similarities.argsort()[-n_results:][::-1]
        return [self.chunks[i] for i in top_indices]


# ============================================
# RAG 问答
# ============================================
def create_rag_chain(retriever):
    """创建 RAG 问答函数"""
    llm = ChatOpenAI(
        model="qwen-plus",
        temperature=0.1,
        openai_api_key=DASHSCOPE_API_KEY,
        openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    
    def ask(question):
        # 1. 检索相关文档
        docs = retriever.retrieve(question)
        
        # 2. 构建上下文
        if docs:
            context = "\n\n---\n\n".join([doc.page_content for doc in docs])
        else:
            context = "（没有找到相关文档内容）"
        
        # 3. 手动构建 prompt（避免占位符问题）
        system_prompt = f"""你是一个专业的文档问答助手。你必须严格基于下面的【文档内容】来回答用户的问题。

【文档内容】
{context}

重要规则：
1. 如果文档中有相关信息，直接基于文档内容回答
2. 如果文档中没有相关信息，直接说"根据现有文档，我没有找到相关信息"
3. 不要编造任何内容
4. 回答要简洁准确"""

        user_prompt = question
        
        # 4. 调用大模型
        response = llm.invoke([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ])
        
        return response.content
    
    return ask