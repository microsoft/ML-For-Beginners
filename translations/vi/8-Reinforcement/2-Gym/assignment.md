<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "1f2b7441745eb52e25745423b247016b",
  "translation_date": "2025-09-05T20:23:06+00:00",
  "source_file": "8-Reinforcement/2-Gym/assignment.md",
  "language_code": "vi"
}
-->
# Huấn luyện Mountain Car

[OpenAI Gym](http://gym.openai.com) được thiết kế sao cho tất cả các môi trường đều cung cấp cùng một API - tức là các phương thức `reset`, `step` và `render`, cùng các khái niệm về **action space** và **observation space**. Do đó, có thể áp dụng cùng một thuật toán học tăng cường cho các môi trường khác nhau với ít thay đổi mã nguồn.

## Môi trường Mountain Car

[Môi trường Mountain Car](https://gym.openai.com/envs/MountainCar-v0/) chứa một chiếc xe bị mắc kẹt trong một thung lũng:

Mục tiêu là thoát khỏi thung lũng và bắt được lá cờ, bằng cách thực hiện một trong các hành động sau tại mỗi bước:

| Giá trị | Ý nghĩa |
|---|---|
| 0 | Tăng tốc về bên trái |
| 1 | Không tăng tốc |
| 2 | Tăng tốc về bên phải |

Điểm mấu chốt của vấn đề này là động cơ của xe không đủ mạnh để leo lên núi trong một lần. Vì vậy, cách duy nhất để thành công là lái xe qua lại để tạo đà.

Không gian quan sát chỉ bao gồm hai giá trị:

| Num | Quan sát       | Min   | Max   |
|-----|----------------|-------|-------|
|  0  | Vị trí của xe  | -1.2  | 0.6   |
|  1  | Vận tốc của xe | -0.07 | 0.07  |

Hệ thống thưởng trong Mountain Car khá phức tạp:

 * Thưởng 0 được trao nếu agent đạt được lá cờ (vị trí = 0.5) trên đỉnh núi.
 * Thưởng -1 được trao nếu vị trí của agent nhỏ hơn 0.5.

Tập sẽ kết thúc nếu vị trí của xe lớn hơn 0.5, hoặc độ dài tập vượt quá 200.
## Hướng dẫn

Điều chỉnh thuật toán học tăng cường của chúng ta để giải quyết vấn đề Mountain Car. Bắt đầu với mã nguồn hiện có trong [notebook.ipynb](../../../../8-Reinforcement/2-Gym/notebook.ipynb), thay thế môi trường mới, thay đổi các hàm phân chia trạng thái, và cố gắng làm cho thuật toán hiện có huấn luyện với ít thay đổi mã nguồn nhất. Tối ưu hóa kết quả bằng cách điều chỉnh các siêu tham số.

> **Note**: Có khả năng cần điều chỉnh siêu tham số để làm cho thuật toán hội tụ.
## Tiêu chí đánh giá

| Tiêu chí | Xuất sắc | Đạt yêu cầu | Cần cải thiện |
| -------- | -------- | ----------- | ------------- |
|          | Thuật toán Q-Learning được điều chỉnh thành công từ ví dụ CartPole, với ít thay đổi mã nguồn, có thể giải quyết vấn đề bắt được lá cờ trong dưới 200 bước. | Một thuật toán Q-Learning mới được áp dụng từ Internet, nhưng được tài liệu hóa tốt; hoặc thuật toán hiện có được áp dụng nhưng không đạt kết quả mong muốn. | Học viên không thể áp dụng thành công bất kỳ thuật toán nào, nhưng đã có những bước tiến đáng kể hướng tới giải pháp (đã triển khai phân chia trạng thái, cấu trúc dữ liệu Q-Table, v.v.) |

---

**Tuyên bố miễn trừ trách nhiệm**:  
Tài liệu này đã được dịch bằng dịch vụ dịch thuật AI [Co-op Translator](https://github.com/Azure/co-op-translator). Mặc dù chúng tôi cố gắng đảm bảo độ chính xác, xin lưu ý rằng các bản dịch tự động có thể chứa lỗi hoặc không chính xác. Tài liệu gốc bằng ngôn ngữ bản địa nên được coi là nguồn thông tin chính thức. Đối với các thông tin quan trọng, khuyến nghị sử dụng dịch vụ dịch thuật chuyên nghiệp bởi con người. Chúng tôi không chịu trách nhiệm cho bất kỳ sự hiểu lầm hoặc diễn giải sai nào phát sinh từ việc sử dụng bản dịch này.