<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "2efc4c2aba5ed06c780c05539c492ae3",
  "translation_date": "2025-09-05T20:28:06+00:00",
  "source_file": "6-NLP/2-Tasks/assignment.md",
  "language_code": "vi"
}
-->
# Làm cho Bot phản hồi

## Hướng dẫn

Trong những bài học trước, bạn đã lập trình một bot cơ bản để trò chuyện. Bot này đưa ra các câu trả lời ngẫu nhiên cho đến khi bạn nói 'bye'. Bạn có thể làm cho các câu trả lời ít ngẫu nhiên hơn và kích hoạt các câu trả lời khi bạn nói những từ cụ thể như 'tại sao' hoặc 'như thế nào' không? Hãy suy nghĩ một chút về cách học máy có thể làm cho loại công việc này bớt thủ công hơn khi bạn mở rộng bot của mình. Bạn có thể sử dụng thư viện NLTK hoặc TextBlob để làm cho các nhiệm vụ của mình dễ dàng hơn.

## Tiêu chí đánh giá

| Tiêu chí  | Xuất sắc                                      | Đạt yêu cầu                                      | Cần cải thiện           |
| --------- | --------------------------------------------- | ------------------------------------------------ | ----------------------- |
|           | Một tệp bot.py mới được trình bày và có tài liệu | Một tệp bot mới được trình bày nhưng có lỗi      | Không có tệp nào được trình bày |

---

**Tuyên bố miễn trừ trách nhiệm**:  
Tài liệu này đã được dịch bằng dịch vụ dịch thuật AI [Co-op Translator](https://github.com/Azure/co-op-translator). Mặc dù chúng tôi cố gắng đảm bảo độ chính xác, xin lưu ý rằng các bản dịch tự động có thể chứa lỗi hoặc không chính xác. Tài liệu gốc bằng ngôn ngữ bản địa nên được coi là nguồn thông tin chính thức. Đối với các thông tin quan trọng, nên sử dụng dịch vụ dịch thuật chuyên nghiệp từ con người. Chúng tôi không chịu trách nhiệm cho bất kỳ sự hiểu lầm hoặc diễn giải sai nào phát sinh từ việc sử dụng bản dịch này.