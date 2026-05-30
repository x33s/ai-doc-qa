"""文档加载模块 - 支持 PDF、TXT、DOC、DOCX 等格式"""
import os
from typing import List

from langchain_community.document_loaders import (
    PyPDFLoader,      # PDF 加载器
    TextLoader,       # TXT 加载器
    UnstructuredWordDocumentLoader,  # Word 加载器
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from src.config import Config


class DocumentLoader:
    """文档加载器 - 负责读取和切分文档"""
    
    # 支持的文件格式及对应的加载器
    SUPPORTED_FORMATS = {
        '.pdf': PyPDFLoader,
        '.txt': TextLoader,
        '.docx': UnstructuredWordDocumentLoader,
        '.doc': UnstructuredWordDocumentLoader,
    }
    
    def __init__(self, folder_path: str = "./docs"):
        self.folder_path = folder_path
        self.chunks: List[Document] = []
    
    def load_all(self) -> List[Document]:
        """加载文件夹内所有支持的文档"""
        all_documents = []
        
        if not os.path.exists(self.folder_path):
            os.makedirs(self.folder_path)
            print(f"📁 已创建 {self.folder_path} 文件夹，请放入你的文档")
            return []
        
        for filename in os.listdir(self.folder_path):
            file_path = os.path.join(self.folder_path, filename)
            ext = os.path.splitext(filename)[1].lower()
            
            if ext in self.SUPPORTED_FORMATS:
                doc = self._load_single_file(file_path, ext, filename)
                if doc:
                    all_documents.extend(doc)
        
        print(f"✅ 共加载 {len(all_documents)} 页/段")
        
        # 切分文档
        self.chunks = self._split_documents(all_documents)
        return self.chunks
    
    def _load_single_file(self, file_path: str, ext: str, filename: str):
        """加载单个文件"""
        try:
            if ext == '.pdf':
                loader = self.SUPPORTED_FORMATS[ext](file_path)
                print(f"📄 加载 PDF: {filename}")
                return loader.load()
            
            elif ext == '.txt':
                return self._load_text_file(file_path, filename)
            
            elif ext in ['.docx', '.doc']:
                loader = self.SUPPORTED_FORMATS[ext](file_path, mode="single")
                print(f"📝 加载 Word: {filename}")
                return loader.load()
                
        except Exception as e:
            print(f"❌ 加载失败 {filename}: {e}")
            return None
    
    def _load_text_file(self, file_path: str, filename: str):
        """加载文本文件（尝试多种编码）"""
        for encoding in ['utf-8', 'gbk', 'gb2312']:
            try:
                loader = TextLoader(file_path, encoding=encoding)
                print(f"📝 加载文本: {filename} (编码: {encoding})")
                return loader.load()
            except UnicodeDecodeError:
                continue
        print(f"❌ 加载文本失败 {filename}: 无法识别编码")
        return None
    
    def _split_documents(self, documents: List[Document]) -> List[Document]:
        """将文档切分成小块"""
        if not documents:
            return []
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP,
            separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]
        )
        chunks = splitter.split_documents(documents)
        print(f"✂️ 切分成 {len(chunks)} 个文本块")
        return chunks
    
    def get_document_info(self) -> dict:
        """获取文档信息（用于显示）"""
        return {
            "total_chunks": len(self.chunks),
            "supported_formats": list(self.SUPPORTED_FORMATS.keys()),
        }