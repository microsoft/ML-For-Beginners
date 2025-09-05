<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "20ca019012b1725de956681d036d8b18",
  "translation_date": "2025-09-05T20:09:51+00:00",
  "source_file": "8-Reinforcement/README.md",
  "language_code": "vi"
}
-->
# Giới thiệu về học tăng cường

Học tăng cường, hay RL, được xem là một trong những mô hình học máy cơ bản, bên cạnh học có giám sát và học không giám sát. RL xoay quanh việc đưa ra quyết định: đưa ra quyết định đúng hoặc ít nhất là học hỏi từ những quyết định đã đưa ra.

Hãy tưởng tượng bạn có một môi trường mô phỏng như thị trường chứng khoán. Điều gì sẽ xảy ra nếu bạn áp dụng một quy định nhất định? Nó có tác động tích cực hay tiêu cực? Nếu có điều gì tiêu cực xảy ra, bạn cần tiếp nhận _tăng cường tiêu cực_, học hỏi từ đó và thay đổi hướng đi. Nếu kết quả là tích cực, bạn cần xây dựng dựa trên _tăng cường tích cực_ đó.

![peter và con sói](../../../8-Reinforcement/images/peter.png)

> Peter và bạn bè của cậu ấy cần thoát khỏi con sói đói! Hình ảnh bởi [Jen Looper](https://twitter.com/jenlooper)

## Chủ đề khu vực: Peter và con sói (Nga)

[Peter và con sói](https://en.wikipedia.org/wiki/Peter_and_the_Wolf) là một câu chuyện cổ tích âm nhạc được viết bởi nhà soạn nhạc người Nga [Sergei Prokofiev](https://en.wikipedia.org/wiki/Sergei_Prokofiev). Đây là câu chuyện về cậu bé tiên phong Peter, người dũng cảm rời khỏi nhà để đến khu rừng và đuổi theo con sói. Trong phần này, chúng ta sẽ huấn luyện các thuật toán học máy để giúp Peter:

- **Khám phá** khu vực xung quanh và xây dựng bản đồ điều hướng tối ưu
- **Học** cách sử dụng ván trượt và giữ thăng bằng trên đó để di chuyển nhanh hơn.

[![Peter và con sói](https://img.youtube.com/vi/Fmi5zHg4QSM/0.jpg)](https://www.youtube.com/watch?v=Fmi5zHg4QSM)

> 🎥 Nhấp vào hình ảnh trên để nghe Peter và con sói của Prokofiev

## Học tăng cường

Trong các phần trước, bạn đã thấy hai ví dụ về các vấn đề học máy:

- **Có giám sát**, nơi chúng ta có các tập dữ liệu gợi ý các giải pháp mẫu cho vấn đề mà chúng ta muốn giải quyết. [Phân loại](../4-Classification/README.md) và [hồi quy](../2-Regression/README.md) là các nhiệm vụ học có giám sát.
- **Không giám sát**, trong đó chúng ta không có dữ liệu huấn luyện được gắn nhãn. Ví dụ chính của học không giám sát là [Phân cụm](../5-Clustering/README.md).

Trong phần này, chúng ta sẽ giới thiệu một loại vấn đề học mới không yêu cầu dữ liệu huấn luyện được gắn nhãn. Có một số loại vấn đề như vậy:

- **[Học bán giám sát](https://wikipedia.org/wiki/Semi-supervised_learning)**, nơi chúng ta có rất nhiều dữ liệu không gắn nhãn có thể được sử dụng để tiền huấn luyện mô hình.
- **[Học tăng cường](https://wikipedia.org/wiki/Reinforcement_learning)**, trong đó một tác nhân học cách hành xử bằng cách thực hiện các thí nghiệm trong một môi trường mô phỏng.

### Ví dụ - trò chơi máy tính

Giả sử bạn muốn dạy máy tính chơi một trò chơi, chẳng hạn như cờ vua, hoặc [Super Mario](https://wikipedia.org/wiki/Super_Mario). Để máy tính chơi trò chơi, chúng ta cần nó dự đoán nước đi nào cần thực hiện trong mỗi trạng thái của trò chơi. Mặc dù điều này có vẻ giống như một vấn đề phân loại, nhưng thực tế không phải - vì chúng ta không có tập dữ liệu với các trạng thái và hành động tương ứng. Mặc dù chúng ta có thể có một số dữ liệu như các trận đấu cờ vua hiện có hoặc các bản ghi của người chơi chơi Super Mario, nhưng có khả năng dữ liệu đó sẽ không đủ để bao phủ một số lượng lớn các trạng thái có thể xảy ra.

Thay vì tìm kiếm dữ liệu trò chơi hiện có, **Học tăng cường** (RL) dựa trên ý tưởng *cho máy tính chơi* nhiều lần và quan sát kết quả. Do đó, để áp dụng Học tăng cường, chúng ta cần hai điều:

- **Một môi trường** và **một trình mô phỏng** cho phép chúng ta chơi trò chơi nhiều lần. Trình mô phỏng này sẽ định nghĩa tất cả các quy tắc trò chơi cũng như các trạng thái và hành động có thể xảy ra.

- **Một hàm thưởng**, sẽ cho chúng ta biết chúng ta đã làm tốt như thế nào trong mỗi nước đi hoặc trò chơi.

Sự khác biệt chính giữa các loại học máy khác và RL là trong RL chúng ta thường không biết liệu chúng ta thắng hay thua cho đến khi kết thúc trò chơi. Do đó, chúng ta không thể nói liệu một nước đi cụ thể có tốt hay không - chúng ta chỉ nhận được phần thưởng vào cuối trò chơi. Và mục tiêu của chúng ta là thiết kế các thuật toán cho phép chúng ta huấn luyện một mô hình trong điều kiện không chắc chắn. Chúng ta sẽ tìm hiểu về một thuật toán RL gọi là **Q-learning**.

## Các bài học

1. [Giới thiệu về học tăng cường và Q-Learning](1-QLearning/README.md)
2. [Sử dụng môi trường mô phỏng gym](2-Gym/README.md)

## Tín dụng

"Giới thiệu về Học Tăng Cường" được viết với ♥️ bởi [Dmitry Soshnikov](http://soshnikov.com)

---

**Tuyên bố miễn trừ trách nhiệm**:  
Tài liệu này đã được dịch bằng dịch vụ dịch thuật AI [Co-op Translator](https://github.com/Azure/co-op-translator). Mặc dù chúng tôi cố gắng đảm bảo độ chính xác, xin lưu ý rằng các bản dịch tự động có thể chứa lỗi hoặc không chính xác. Tài liệu gốc bằng ngôn ngữ bản địa nên được coi là nguồn thông tin chính thức. Đối với các thông tin quan trọng, khuyến nghị sử dụng dịch vụ dịch thuật chuyên nghiệp bởi con người. Chúng tôi không chịu trách nhiệm cho bất kỳ sự hiểu lầm hoặc diễn giải sai nào phát sinh từ việc sử dụng bản dịch này.