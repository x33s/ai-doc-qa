"""RAG 问答链 - 整合检索和生成"""
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from src.config import Config
from src.retriever import SimpleRetriever


# 提示词模板
PROMPT_TEMPLATE = ChatPromptTemplate.from_messages([
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


def format_docs(docs):
    """将检索到的文档块格式化成字符串"""
    if not docs:
        return "（没有找到相关文档内容）"
    return "\n\n---\n\n".join([doc.page_content for doc in docs])


class RAGChain:
    """RAG 问答链"""
    
    def __init__(self, retriever: SimpleRetriever):
        self.retriever = retriever
        
        # 初始化大模型
        self.llm = ChatOpenAI(
            model=Config.LLM_MODEL,
            temperature=Config.LLM_TEMPERATURE,
            openai_api_key=Config.DASHSCOPE_API_KEY,
            openai_api_base=Config.LLM_BASE_URL,
        )
    
    def ask(self, question: str) -> str:
        """提问并获取回答"""
        # 1. 检索相关文档块
        relevant_docs = self.retriever.retrieve(question)
        
        # 2. 构建 Prompt
        context = format_docs(relevant_docs)
        prompt = PROMPT_TEMPLATE.format_messages(context=context, question=question)
        
        # 3. 调用大模型生成回答
        response = self.llm.invoke(prompt)
        
        return response.content