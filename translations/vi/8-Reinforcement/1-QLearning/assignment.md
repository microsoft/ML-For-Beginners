<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "68394b2102d3503882e5e914bd0ff5c1",
  "translation_date": "2025-09-05T20:17:41+00:00",
  "source_file": "8-Reinforcement/1-QLearning/assignment.md",
  "language_code": "vi"
}
-->
# Một Thế Giới Thực Tế Hơn

Trong tình huống của chúng ta, Peter có thể di chuyển gần như không cảm thấy mệt mỏi hay đói. Trong một thế giới thực tế hơn, anh ấy cần phải ngồi xuống nghỉ ngơi thỉnh thoảng, và cũng cần ăn uống để duy trì sức khỏe. Hãy làm cho thế giới của chúng ta thực tế hơn bằng cách áp dụng các quy tắc sau:

1. Khi di chuyển từ nơi này sang nơi khác, Peter mất **năng lượng** và tăng thêm **mệt mỏi**.
2. Peter có thể tăng năng lượng bằng cách ăn táo.
3. Peter có thể giảm mệt mỏi bằng cách nghỉ ngơi dưới gốc cây hoặc trên cỏ (tức là đi vào vị trí trên bảng có cây hoặc cỏ - ô màu xanh lá).
4. Peter cần tìm và tiêu diệt con sói.
5. Để tiêu diệt con sói, Peter cần đạt mức năng lượng và mệt mỏi nhất định, nếu không anh ấy sẽ thua trong trận chiến.

## Hướng Dẫn

Sử dụng [notebook.ipynb](../../../../8-Reinforcement/1-QLearning/notebook.ipynb) gốc làm điểm bắt đầu cho giải pháp của bạn.

Chỉnh sửa hàm thưởng theo các quy tắc của trò chơi, chạy thuật toán học tăng cường để tìm chiến lược tốt nhất để chiến thắng trò chơi, và so sánh kết quả của việc đi ngẫu nhiên với thuật toán của bạn về số lượng trò chơi thắng và thua.

> **Note**: Trong thế giới mới của bạn, trạng thái phức tạp hơn, và ngoài vị trí của con người còn bao gồm mức độ mệt mỏi và năng lượng. Bạn có thể chọn biểu diễn trạng thái dưới dạng một tuple (Board,energy,fatigue), hoặc định nghĩa một lớp cho trạng thái (bạn cũng có thể muốn kế thừa từ `Board`), hoặc thậm chí chỉnh sửa lớp `Board` gốc trong [rlboard.py](../../../../8-Reinforcement/1-QLearning/rlboard.py).

Trong giải pháp của bạn, hãy giữ lại đoạn mã chịu trách nhiệm cho chiến lược đi ngẫu nhiên, và so sánh kết quả của thuật toán của bạn với chiến lược đi ngẫu nhiên ở cuối.

> **Note**: Bạn có thể cần điều chỉnh các siêu tham số để làm cho nó hoạt động, đặc biệt là số lượng epochs. Vì thành công của trò chơi (đánh bại con sói) là một sự kiện hiếm gặp, bạn có thể mong đợi thời gian huấn luyện lâu hơn.

## Tiêu Chí Đánh Giá

| Tiêu chí | Xuất Sắc                                                                                                                                                                                             | Đạt Yêu Cầu                                                                                                                                                                                | Cần Cải Thiện                                                                                                                          |
| -------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
|          | Notebook được trình bày với định nghĩa các quy tắc mới của thế giới, thuật toán Q-Learning và một số giải thích bằng văn bản. Q-Learning có thể cải thiện đáng kể kết quả so với đi ngẫu nhiên. | Notebook được trình bày, Q-Learning được triển khai và cải thiện kết quả so với đi ngẫu nhiên, nhưng không đáng kể; hoặc notebook được tài liệu hóa kém và mã không được cấu trúc tốt | Có một số cố gắng để tái định nghĩa các quy tắc của thế giới, nhưng thuật toán Q-Learning không hoạt động, hoặc hàm thưởng không được định nghĩa đầy đủ |

---

**Tuyên bố miễn trừ trách nhiệm**:  
Tài liệu này đã được dịch bằng dịch vụ dịch thuật AI [Co-op Translator](https://github.com/Azure/co-op-translator). Mặc dù chúng tôi cố gắng đảm bảo độ chính xác, xin lưu ý rằng các bản dịch tự động có thể chứa lỗi hoặc không chính xác. Tài liệu gốc bằng ngôn ngữ bản địa nên được coi là nguồn thông tin chính thức. Đối với các thông tin quan trọng, nên sử dụng dịch vụ dịch thuật chuyên nghiệp bởi con người. Chúng tôi không chịu trách nhiệm cho bất kỳ sự hiểu lầm hoặc diễn giải sai nào phát sinh từ việc sử dụng bản dịch này.