"""backend/app.py - FastAPI 服务"""
import os
import sys
from contextlib import asynccontextmanager

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(__file__))

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import shutil

# 导入 RAG 核心模块
from rag_core import (
    load_documents_from_folder,
    split_documents,
    TFIDFRetriever,
    create_rag_chain
)

# ============================================
# 全局变量
# ============================================
rag_chain = None
retriever = None
docs_folder = os.path.join(os.path.dirname(__file__), "docs")


# ============================================
# Lifespan 事件处理（替代旧的 on_event）
# ============================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """处理应用启动和关闭时的逻辑"""
    global rag_chain, retriever, docs_folder
    
    # --- 启动逻辑 (原 startup 事件) ---
    print("🚀 启动智能文档问答 API 服务...")
    print(f"📁 文档目录: {docs_folder}")
    
    documents = load_documents_from_folder(docs_folder)
    
    if documents:
        chunks = split_documents(documents)
        retriever = TFIDFRetriever()
        retriever.build_index(chunks)
        rag_chain = create_rag_chain(retriever)
        print(f"✅ RAG 系统初始化完成，已加载 {len(chunks)} 个文本块")
    else:
        print("⚠️ 没有找到文档，请通过 /upload 接口上传文件")
    
    # yield 将应用逻辑分隔为启动和关闭两部分
    yield
    
    # --- 关闭逻辑 (原 shutdown 事件) ---
    print("🛑 应用正在关闭，清理资源...")


# ============================================
# 创建 FastAPI 应用
# ============================================
app = FastAPI(
    title="智能文档问答 API",
    version="1.0",
    lifespan=lifespan  # 使用 lifespan 替代 on_event
)


# ============================================
# 跨域中间件配置
# ============================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:5174", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================
# 请求/响应模型
# ============================================
class ChatRequest(BaseModel):
    question: str
    history: Optional[List[dict]] = []


class ChatResponse(BaseModel):
    answer: str
    sources: List[str] = []


# ============================================
# API 端点
# ============================================
@app.get("/")
async def root():
    return {"message": "智能文档问答 API", "status": "running"}


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """上传文档"""
    global rag_chain, retriever
    
    # 确保 docs 目录存在
    os.makedirs(docs_folder, exist_ok=True)
    
    file_path = os.path.join(docs_folder, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # 重新加载文档
    documents = load_documents_from_folder(docs_folder)
    if documents:
        chunks = split_documents(documents)
        retriever = TFIDFRetriever()
        retriever.build_index(chunks)
        rag_chain = create_rag_chain(retriever)
        print(f"✅ 已重新加载，共 {len(chunks)} 个文本块")
    
    return {"message": f"文件 {file.filename} 上传成功", "filename": file.filename}


@app.get("/documents")
async def list_documents():
    """列出所有已上传的文档"""
    if not os.path.exists(docs_folder):
        return {"documents": []}
    
    files = [f for f in os.listdir(docs_folder) if os.path.isfile(os.path.join(docs_folder, f))]
    return {"documents": files}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """对话接口"""
    global rag_chain, retriever
    
    if rag_chain is None:
        raise HTTPException(status_code=503, detail="RAG 系统未初始化，请先上传文档")
    
    try:
        answer = rag_chain(request.question)
        
        # 获取检索来源
        sources = []
        if retriever:
            docs = retriever.retrieve(request.question)
            sources = [doc.page_content[:150] + "..." for doc in docs[:2]]
        
        return ChatResponse(answer=answer, sources=sources)
    except Exception as e:
        print(f"错误: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/documents/{filename}")
async def delete_document(filename: str):
    """删除文档"""
    global rag_chain, retriever
    
    file_path = os.path.join(docs_folder, filename)
    if os.path.exists(file_path):
        os.remove(file_path)
        
        # 重新加载
        documents = load_documents_from_folder(docs_folder)
        if documents:
            chunks = split_documents(documents)
            retriever = TFIDFRetriever()
            retriever.build_index(chunks)
            rag_chain = create_rag_chain(retriever)
        else:
            retriever = TFIDFRetriever()
            retriever.build_index([])
            rag_chain = None
        
        return {"message": f"已删除 {filename}"}
    else:
        raise HTTPException(status_code=404, detail="文件不存在")


# ============================================
# 启动入口
# ============================================
if __name__ == "__main__":
    import uvicorn
    # 注意：这里去掉了 reload=True，避免警告
    uvicorn.run(app, host="0.0.0.0", port=8000)