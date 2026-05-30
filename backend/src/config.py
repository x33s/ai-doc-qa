"""配置管理模块"""
import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    """所有配置项"""
    
    # 百炼 API 配置
    DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
    
    # 如果没有环境变量，可以在这里直接填写（注意：不要提交到 Git）
    if not DASHSCOPE_API_KEY:
        DASHSCOPE_API_KEY = "sk-your-key-here"  # 替换成你的 Key
    
    # 模型配置
    LLM_MODEL = "qwen-plus"
    LLM_TEMPERATURE = 0.1
    LLM_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    
    # 文档切分配置
    CHUNK_SIZE = 500      # 每块字符数
    CHUNK_OVERLAP = 50    # 重叠字符数
    
    # 检索配置
    RETRIEVAL_TOP_K = 4   # 每次检索返回的块数