# test_load.py
from src.document_loader import DocumentLoader

loader = DocumentLoader()
docs = loader.load_all()
print(f"加载了 {len(docs)} 个文档块")