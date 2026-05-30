"""RAG 智能文档问答系统 - 主程序"""
from src.document_loader import DocumentLoader
from src.retriever import SimpleRetriever
from src.rag_chain import RAGChain


def main():
    """主函数"""
    print("\n" + "=" * 50)
    print("🤖 RAG 智能文档问答系统")
    print("=" * 50)
    
    # 1. 加载文档
    print("\n📂 加载文档中...")
    loader = DocumentLoader("./docs")
    chunks = loader.load_all()
    
    if not chunks:
        print("❌ 没有找到文档，请在 docs 文件夹下放入 PDF、TXT 或 Word 文件")
        return
    
    # 2. 构建检索索引
    retriever = SimpleRetriever()
    retriever.build_index(chunks)
    
    # 3. 创建问答链
    rag = RAGChain(retriever)
    
    # 4. 显示文档信息
    info = loader.get_document_info()
    print(f"\n📚 已加载 {info['total_chunks']} 个文本块")
    print(f"📁 支持的格式: {', '.join(info['supported_formats'])}")
    
    # 5. 交互问答
    print("\n" + "-" * 30)
    print("💡 提示：输入 'q' 退出，输入 's:问题' 查看检索效果")
    print("-" * 30)
    
    while True:
        question = input("\n📝 请输入问题：")
        
        if question.lower() == 'q':
            print("👋 再见！")
            break
        
        # 调试模式
        if question.startswith('s:'):
            retriever.search_and_show(question[2:])
            continue
        
        # 正常问答
        print("🤔 思考中...")
        try:
            answer = rag.ask(question)
            print(f"\n💡 回答：{answer}")
        except Exception as e:
            print(f"❌ 出错了：{e}")


if __name__ == "__main__":
    main()