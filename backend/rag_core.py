"""backend/rag_core.py - RAG 核心模块（向量化检索版）"""
import os
import re
import time
import numpy as np
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredWordDocumentLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import faiss

# 加载环境变量
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

# ============================================
# 配置
# ============================================
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
if not DASHSCOPE_API_KEY:
    raise ValueError("请在项目根目录的 .env 文件中设置 DASHSCOPE_API_KEY")

# 文档分块配置 - 优化分块大小以适应中文语义
CHUNK_SIZE = 500  # 增大块大小以包含更多上下文
CHUNK_OVERLAP = 80  # 增加重叠以保持上下文连贯
RETRIEVAL_TOP_K = 5  # 检索更多相关文档以提高覆盖率

# ============================================
# 向量化检索器类
# ============================================
class VectorRetriever:
    """基于向量化检索的检索器"""

    def __init__(self):
        self.index = None
        self.chunks = []
        self.embedding_dim = 1536  # DashScope text-embedding-v2 维度
        self.max_tokens = 1800  # 设置安全上限，留出余量

    def truncate_text(self, text, max_length=1800):
        """截断文本以符合 API 限制"""
        if len(text) <= max_length:
            return text
        # 按字符截断
        return text[:max_length]

    def get_embedding(self, text):
        """调用 DashScope API 获取文本向量"""
        from openai import OpenAI

        client = OpenAI(
            api_key=DASHSCOPE_API_KEY,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )

        # 截断文本以符合 API 限制
        text = self.truncate_text(text, self.max_tokens)

        try:
            response = client.embeddings.create(
                model="text-embedding-v2",
                input=text
            )
            embedding = response.data[0].embedding
            return np.array(embedding, dtype=np.float32)
        except Exception as e:
            print(f"❌ 获取向量失败: {e}")
            return None

    def build_index(self, chunks):
        """构建 FAISS 索引"""
        self.chunks = chunks

        if not chunks:
            self.index = None
            print("⚠️ 没有文档块可索引")
            return

        print(f"🔄 正在生成 {len(chunks)} 个文档块的向量...")

        # 逐个生成向量（更稳定）
        embeddings = []
        failed_count = 0

        for i, chunk in enumerate(chunks):
            embedding = self.get_embedding(chunk.page_content)
            if embedding is not None:
                embeddings.append(embedding)
            else:
                failed_count += 1
                # 使用零向量作为占位
                embeddings.append(np.zeros(self.embedding_dim, dtype=np.float32))

            if (i + 1) % 10 == 0:
                print(f"  已处理 {i + 1}/{len(chunks)} 个文档块...")

        if failed_count > 0:
            print(f"⚠️ 有 {failed_count} 个文档块生成向量失败")

        if not embeddings:
            print("❌ 无法生成任何向量")
            return

        # 构建 FAISS 索引
        embeddings_matrix = np.array(embeddings)
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        self.index.add(embeddings_matrix)

        print(f"✅ 向量索引构建完成，共 {len(chunks)} 个文档块")

    def retrieve(self, query, top_k=RETRIEVAL_TOP_K):
        """检索最相关的文档块"""
        if self.index is None or not self.chunks:
            return []

        # 获取查询向量
        query_embedding = self.get_embedding(query)
        if query_embedding is None:
            return []

        query_embedding = query_embedding.reshape(1, -1)

        # 检索
        n_results = min(top_k, len(self.chunks))
        distances, indices = self.index.search(query_embedding, n_results)

        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.chunks):
                results.append(self.chunks[idx])

        return results


# ============================================
# 文档加载
# ============================================
def load_documents_from_folder(folder_path=None):
    """递归加载文件夹内的所有 PDF、TXT 和 Word 文件"""
    if folder_path is None:
        folder_path = os.path.join(os.path.dirname(__file__), "docs")

    all_documents = []

    if not os.path.exists(folder_path):
        os.makedirs(folder_path, exist_ok=True)
        print(f"📁 已创建 {folder_path} 文件夹，请放入你的文档")
        return []

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        if os.path.isdir(file_path):
            all_documents.extend(load_documents_from_folder(file_path))
            continue

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

        elif filename.endswith('.xlsx') or filename.endswith('.xls'):
            try:
                from langchain_community.document_loaders import UnstructuredExcelLoader
                loader = UnstructuredExcelLoader(file_path, mode="elements")
                all_documents.extend(loader.load())
                print(f"📊 加载 Excel: {filename}")
            except Exception as e:
                print(f"❌ 加载 Excel 失败 {filename}: {e}")

    return all_documents


def split_documents(documents):
    """将文档切分成小块 - 优化中文分块策略"""
    if not documents:
        return []

    # 优化分块策略：更好地保持语义完整性
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        # 中文分块优先按段落、句子切分
        separators=[
            "\n\n",  # 段落分隔
            "\n",    # 换行分隔
            "。", "！", "？", "；",  # 句子分隔
            "，", "、",  # 短句分隔
            " ", ""
        ],
        length_function=len
    )

    chunks = splitter.split_documents(documents)

    # 过滤过短的块
    chunks = [chunk for chunk in chunks if len(chunk.page_content) >= 50]

    print(f"✂️ 切分成 {len(chunks)} 个文本块（过滤了过短块后）")
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
        
        # 3. 系统提示词：充分利用检索到的文档内容
        system_prompt = f"""你是一个专业的文档问答助手。你已经通过检索系统找到了与用户问题相关的文档片段。请仔细阅读这些文档内容，并基于它们回答问题。

【检索到的相关文档内容】
{context}

【回答要求】
1. 必须基于上述文档内容回答，不要说"根据现有文档，我没有找到相关信息"，除非文档内容完全不相关
2. 如果文档中有部分相关信息，在末尾标注【来源：文档X】
3. 如果文档内容与问题相关，直接引用文档内容进行回答 
4. 回答要准确，不要编造文档中没有的信息

用户问题：{question}

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


def create_streaming_rag_chain(retriever):
    """创建流式 RAG 问答函数"""
    from openai import OpenAI
    
    client = OpenAI(
        api_key=DASHSCOPE_API_KEY,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    
    def ask_streaming(question):
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
        
        # 3. 系统提示词：充分利用检索到的文档内容
        system_prompt = f"""你是一个专业的文档问答助手。你已经通过检索系统找到了与用户问题相关的文档片段。请仔细阅读这些文档内容，并基于它们回答问题。

【检索到的相关文档内容】
{context}

【回答要求】
1. 必须基于上述文档内容回答，不要说"根据现有文档，我没有找到相关信息"，除非文档内容完全不相关
2. 如果文档中有部分相关信息， 在末尾标注【来源：文档X】
3. 如果文档内容与问题相关，直接引用文档内容进行回答
4. 回答要准确，不要编造文档中没有的信息

用户问题：{question}

请回答："""

        # 4. 流式生成回答
        stream = client.chat.completions.create(
            model="qwen-plus",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question}
            ],
            temperature=0.1,
            stream=True
        )
        
        # 5. 逐块返回
        for chunk in stream:
            content = chunk.choices[0].delta.content
            if content:
                yield content
    
    return ask_streaming


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