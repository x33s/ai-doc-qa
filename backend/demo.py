import os
from dotenv import load_dotenv

# 加载 .env 中的 API Key
load_dotenv()

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.embeddings import HuggingFaceEmbeddings

# ============================================
# 配置
# ============================================
# 百炼 API Key（和你的 hello.py 同一个）
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
if not DASHSCOPE_API_KEY:
    # 如果没有环境变量，可以直接写在这里（注意：不要提交到 GitHub）
    DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")  # 替换成你自己的

# 使用本地 Embedding 模型（不需要 API Key，完全免费，离线可用）
print("🔄 正在加载本地 Embedding 模型（首次运行会下载，约 500MB，请稍候）...")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)
print("✅ Embedding 模型加载完成")

# 使用百炼对话模型（和你的 hello.py 一样的配置）
llm = ChatOpenAI(
    model="qwen-plus",  # 或 qwen-max / qwen-turbo
    temperature=0.1,
    openai_api_key=DASHSCOPE_API_KEY,
    openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# ============================================
# 1. 加载文档
# ============================================
def load_documents_from_folder(folder_path="./docs"):
    """加载文件夹内的所有 PDF 和 TXT 文件"""
    all_documents = []
    
    # 如果 docs 文件夹不存在，创建它
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
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
                # 尝试不同编码
                for encoding in ['utf-8', 'gbk', 'gb2312']:
                    try:
                        loader = TextLoader(file_path, encoding=encoding)
                        all_documents.extend(loader.load())
                        print(f"📝 加载文本: {filename} (编码: {encoding})")
                        break
                    except UnicodeDecodeError:
                        continue
            except Exception as e:
                print(f"❌ 加载文本失败 {filename}: {e}")
        else:
            continue
    
    return all_documents

# 加载文档
print("\n📂 开始加载文档...")
documents = load_documents_from_folder()
print(f"✅ 共加载 {len(documents)} 页/段")

if len(documents) == 0:
    print("\n⚠️ 没有找到文档！")
    print("请在项目目录下创建 docs 文件夹，并放入 PDF 或 TXT 文件")
    print("例如：创建一个 test.txt 文件，写入一些测试内容")
    exit(1)

# ============================================
# 2. 文本切分
# ============================================
print("\n✂️ 正在切分文本...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]
)

chunks = text_splitter.split_documents(documents)
print(f"✅ 切分成 {len(chunks)} 个文本块")

# ============================================
# 3. 向量化并存储
# ============================================
print("\n🔢 正在生成向量并构建索引（首次运行可能需要几分钟）...")
vectorstore = FAISS.from_documents(chunks, embeddings)
print("✅ 向量库构建完成")

# ============================================
# 4. 创建检索器
# ============================================
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 4}
)

# ============================================
# 5. 构建 RAG 提示词
# ============================================
prompt_template = ChatPromptTemplate.from_messages([
    ("system", """你是一个专业的文档问答助手。你的任务是基于提供的文档内容回答用户问题。

重要规则：
1. 必须严格基于【文档内容】回答，不要编造任何信息
2. 如果文档中没有相关信息，直接说"根据现有文档，我没有找到相关信息"
3. 回答要简洁、准确，直接回答问题
4. 如果引用文档中的原文，请标注出来

【文档内容】
{context}"""),
    ("human", "问题：{question}"),
])

# ============================================
# 6. 构建 RAG 链
# ============================================
def format_docs(docs):
    """格式化检索到的文档"""
    return "\n\n---\n\n".join([doc.page_content for doc in docs])

rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough(),
    }
    | prompt_template
    | llm
    | StrOutputParser()
)

# ============================================
# 7. 命令行交互
# ============================================
def search_documents(question):
    """测试检索效果（不调用 LLM，只看检索到了什么）"""
    docs = retriever.invoke(question)
    print(f"\n🔍 检索到 {len(docs)} 个相关文本块：")
    for i, doc in enumerate(docs):
        print(f"\n--- 结果 {i+1} ---")
        content = doc.page_content
        if len(content) > 200:
            print(content[:200] + "...")
        else:
            print(content)

def show_document_info():
    """显示已加载的文档信息"""
    print("\n" + "="*50)
    print("📚 当前文档库信息")
    print("="*50)
    print(f"文本块数量: {len(chunks)}")
    print("\n文档来源:")
    for filename in os.listdir("./docs"):
        print(f"  - {filename}")
    print("="*50)

if __name__ == "__main__":
    print("\n" + "="*50)
    print("🤖 RAG 智能文档问答系统")
    print("="*50)
    show_document_info()
    
    while True:
        print("\n" + "-*"*30)
        question = input("📝 请输入问题（输入 'q' 退出，输入 's:' 查看检索效果，输入 'info' 查看文档信息）：")
        
        if question.lower() == 'q':
            print("👋 再见！")
            break
        
        if question.lower() == 'info':
            show_document_info()
            continue
        
        # 调试模式：只看检索结果
        if question.startswith('s:'):
            search_documents(question[2:])
            continue
        
        print("🤔 思考中...")
        try:
            answer = rag_chain.invoke(question)
            print(f"\n💡 回答：{answer}")
        except Exception as e:
            print(f"❌ 出错了：{e}")
            print("请检查网络连接和 API Key 是否正确")