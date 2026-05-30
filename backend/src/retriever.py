"""检索模块 - 基于 TF-IDF 的本地检索"""
from typing import List
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from langchain_core.documents import Document

from src.config import Config


class SimpleRetriever:
    """简单检索器 - 使用 TF-IDF 进行文档检索"""
    
    def __init__(self):
        self.vectorizer = None
        self.tfidf_matrix = None
        self.chunks = []
        self.chunk_texts = []
    
    def build_index(self, chunks: List[Document]):
        """构建索引"""
        if not chunks:
            print("⚠️ 没有文档块可索引")
            return
        
        self.chunks = chunks
        self.chunk_texts = [chunk.page_content for chunk in chunks]
        
        # 构建 TF-IDF 向量
        print("🔍 正在构建检索索引...")
        self.vectorizer = TfidfVectorizer(max_features=5000)
        self.tfidf_matrix = self.vectorizer.fit_transform(self.chunk_texts)
        print(f"✅ 索引构建完成，共 {len(chunks)} 个文档块")
    
    def retrieve(self, query: str) -> List[Document]:
        """检索与问题最相关的文档块"""
        if self.vectorizer is None or self.tfidf_matrix is None:
            return []
        
        if self.tfidf_matrix.shape[0] == 0:
            return []
        
        # 将问题转换成向量
        query_vec = self.vectorizer.transform([query])
        
        # 计算相似度（确保返回的是 1D 数组）
        similarity_matrix = cosine_similarity(query_vec, self.tfidf_matrix)
        similarities = np.array(similarity_matrix).flatten()
        
        # 获取相似度最高的 K 个
        n_results = min(Config.RETRIEVAL_TOP_K, len(similarities))
        if n_results == 0:
            return []
        
        top_indices = similarities.argsort()[-n_results:][::-1]
        
        return [self.chunks[i] for i in top_indices]
    
    def search_and_show(self, query: str):
        """调试用：显示检索结果"""
        docs = self.retrieve(query)
        print(f"\n🔍 检索到 {len(docs)} 个相关文本块：")
        for i, doc in enumerate(docs):
            content = doc.page_content
            print(f"\n--- 结果 {i+1} ---")
            if len(content) > 200:
                print(content[:200] + "...")
            else:
                print(content)