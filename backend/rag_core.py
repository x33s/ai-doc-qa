"""backend/rag_core.py - RAG 核心模块"""
import os
import re
import numpy as np
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredWordDocumentLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 加载环境变量
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
# RAG 问答（带来源标注）
# ============================================
# ============================================
# RAG 问答（终极修复版：严禁编造来源）
# ============================================
 # ============================================
# RAG 问答（修复版：基于文档回答，不拒绝）
# ============================================
def create_rag_chain(retriever):
    """创建 RAG 问答函数"""
    from openai import OpenAI
    
    client = OpenAI(
        api_key=DASHSCOPE_API_KEY,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    
    def ask(question):
        # 1. 检索相关文档
        docs = retriever.retrieve(question)
        
        # 2. 构建带编号的上下文
        if docs:
            context_parts = []
            for i, doc in enumerate(docs):
                source_file = os.path.basename(doc.metadata.get('source', 'unknown'))
                content = doc.page_content
                context_parts.append(f"【文档{i+1}】（来自 {source_file}）\n{content}")
            context = "\n\n---\n\n".join(context_parts)
        else:
            context = "（没有找到相关文档）"
        
        # 3. 系统提示词：不拒绝任何问题，基于文档回答
        system_prompt = f"""你是一个专业的文档问答助手。请基于下面的【文档内容】回答用户的问题。

【文档内容】
{context}

规则：
1. 如果文档中有相关信息，直接回答，并在末尾标注【来源：文档X】（X是文档编号）
2. 如果文档中没有相关信息，直接说"根据现有文档，我没有找到相关信息"，不要编造
3. 回答要简洁准确

请回答："""

        response = client.chat.completions.create(
            model="qwen-plus",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question}
            ],
            temperature=0.1
        )
        
        return response.choices[0].message.content
    
    return ask


def extract_sources_from_answer(answer, retrieved_docs):
    """从回答中提取来源编号，并返回对应的文档内容"""
    # 匹配 【来源：X】 或 【来源：X,Y】 格式
    pattern = r'【来源：([\d,]+)】'
    match = re.search(pattern, answer)
    
    # 如果没有找到来源标注，返回原始回答和空列表
    if not match:
        # 清理可能的不完整标注
        clean_answer = re.sub(r'\s*【来源：[\d,]*】?\s*', '', answer)
        return clean_answer, []
    
    # 提取编号
    source_indices = [int(idx.strip()) - 1 for idx in match.group(1).split(',')]
    
    # 根据编号获取对应的文档内容
    sources = []
    for idx in source_indices:
        if 0 <= idx < len(retrieved_docs):
            content = retrieved_docs[idx].page_content
            # 截取前 200 字符作为预览
            preview = content[:200] + "..." if len(content) > 200 else content
            sources.append(preview)
    
    # 从回答中移除来源标注
    clean_answer = re.sub(r'\s*【来源：[\d,]+】\s*', '', answer)
    
    return clean_answer, sources