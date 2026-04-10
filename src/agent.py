from typing import Callable, Any
from .store import EmbeddingStore 

class KnowledgeBaseAgent:
    def __init__(self, store: EmbeddingStore, llm_fn: Callable[[str], str]) -> None:
        self.store = store
        self.llm_fn = llm_fn

    def answer(self, question: str, top_k: int = 3) -> str:
        results = self.store.search(question, top_k=top_k)
        
        if not results:
            return "Xin lỗi, tôi không tìm thấy tài liệu nào khớp với câu hỏi của bạn."

        # LƯU Ý: Lấy dữ liệu từ key "content"
        context_chunks = [record["content"] for record in results]
        context_str = "\n\n---\n\n".join(context_chunks)
        
        prompt = (
            "Bạn là một trợ lý AI hữu ích, làm việc dựa trên tài liệu nội bộ.\n"
            "Dưới đây là các thông tin ngữ cảnh được trích xuất từ cơ sở dữ liệu:\n"
            "====================\n"
            f"{context_str}\n"
            "====================\n"
            "Nhiệm vụ của bạn:\n"
            "1. Chỉ sử dụng thông tin từ ngữ cảnh được cung cấp ở trên để trả lời.\n"
            "2. Nếu thông tin KHÔNG có, hãy dũng cảm nói không biết.\n\n"
            f"Câu hỏi: {question}\n"
            "Trả lời:"
        )

        answer_text = self.llm_fn(prompt)
        
        return answer_text, results