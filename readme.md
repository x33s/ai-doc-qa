## 性能优化
替换 TF-IDF 为 Embedding 模型（提高检索准确率） ok

添加多轮对话记忆 -

流式输出回答 ok

支持更多文档格式（PPT、Markdown） ok
 
每次重启，都要重新生成向量索引，非常慢，解决办法：
- 1. 项目启动时，先检查是否有缓存的向量索引，如果有，直接加载。
- 2. 如果没有缓存，再生成新的向量索引。
- 3. 项目关闭时，将当前向量索引缓存起来，方便下次启动。



# 📚 智能文档问答系统

一个基于 RAG（检索增强生成）技术的智能文档问答系统，支持上传 PDF、TXT、Word 文档，并基于文档内容进行智能问答。

## ✨ 功能特点

- 📄 **多格式支持**：支持 PDF、TXT、DOC、DOCX 等常见文档格式
- 🔍 **智能检索**：基于 TF-IDF 算法的本地检索，无需联网
- 🤖 **AI 问答**：集成阿里云百炼（通义千问）大模型，基于文档内容精准回答
- 📌 **引用来源**：回答时自动标注信息来源，确保可追溯
- 💬 **对话界面**：Vue 3 构建的现代化聊天界面，支持拖拽上传
- 🚀 **开箱即用**：完整的后端 API + 前端界面，一键启动

## 🛠️ 技术栈

### 后端
- **FastAPI**：高性能 Web 框架
- **LangChain**：RAG 流程编排
- **TF-IDF + 余弦相似度**：本地文档检索
- **阿里云百炼（通义千问）**：大语言模型 API

### 前端
- **Vue 3**：渐进式 JavaScript 框架
- **Vite**：极速构建工具
- **Axios**：HTTP 请求库

## 📁 项目结构
ai-doc-qa/
├── backend/ # 后端服务
│ ├── app.py # FastAPI 主程序
│ ├── rag_core.py # RAG 核心逻辑
│ ├── docs/ # 上传的文档目录
│ └── requirements.txt # Python 依赖
├── frontend/ # 前端项目
│ ├── src/
│ │ ├── App.vue # 主界面
│ │ ├── components/ # 组件
│ │ └── main.js
│ └── package.json
├── .env # 环境变量（API Key）
└── README.md



1. 启动后
## 🚀 快速开始

### 环境要求

- Python 3.9+
- Node.js 16+
- 阿里云百炼 API Key（[免费领取](https://bailian.console.aliyun.com/)）

### 1. 克隆项目

```bash
git clone https://github.com/yourname/ai-doc-qa.git
cd ai-doc-qa


DASHSCOPE_API_KEY=sk-your-api-key

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或 venv\Scripts\activate  # Windows

# 安装依赖
pip install -r backend/requirements.txt

# 启动服务
python backend/app.py

### 2. 启动前端

cd frontend
npm install
npm run dev


### RAG（检索增强生成）流程

用户问题 → 文档检索 → 构建上下文 → AI 生成 → 返回答案 + 来源


