# 📚 AI 智能文档问答系统

基于 RAG（检索增强生成）技术的文档问答助手，支持上传 PDF/TXT 文档，基于文档内容进行智能问答。

## 技术栈
- **前端**：命令行交互（后续可升级为 React）
- **后端**：FastAPI + LangChain
- **检索**：TF-IDF 向量化 + 余弦相似度
- **大模型**：阿里云百炼（qwen-plus）

## 功能特点
- 支持 PDF/TXT 文档上传和加载
- 基于文档内容的精准问答
- 检索结果可追溯
- 支持多轮对话

## 快速开始
```bash
pip install -r requirements.txt
python rag_demo.py



## 项目结构

ai-doc-qa/
├── docs/                    # 放你的文档
├── src/
│   ├── __init__.py          # 空文件
│   ├── document_loader.py   # 文档加载模块（支持多种格式）
│   ├── retriever.py         # 检索模块（TF-IDF）
│   ├── rag_chain.py         # RAG 问答链
│   └── config.py            # 配置管理
├── rag_demo.py              # 主程序（简洁）
├── .env                     # API Key
└── requirements.txt         # 依赖列表

