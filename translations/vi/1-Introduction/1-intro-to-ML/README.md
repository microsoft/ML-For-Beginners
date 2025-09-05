<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "69389392fa6346e0dfa30f664b7b6fec",
  "translation_date": "2025-09-05T19:39:02+00:00",
  "source_file": "1-Introduction/1-intro-to-ML/README.md",
  "language_code": "vi"
}
-->
# Giới thiệu về học máy

## [Câu hỏi trước bài giảng](https://ff-quizzes.netlify.app/en/ml/)

---

[![Học máy cho người mới bắt đầu - Giới thiệu về học máy cho người mới bắt đầu](https://img.youtube.com/vi/6mSx_KJxcHI/0.jpg)](https://youtu.be/6mSx_KJxcHI "Học máy cho người mới bắt đầu - Giới thiệu về học máy cho người mới bắt đầu")

> 🎥 Nhấp vào hình ảnh trên để xem video ngắn về bài học này.

Chào mừng bạn đến với khóa học về học máy cổ điển dành cho người mới bắt đầu! Dù bạn hoàn toàn mới với chủ đề này hay là một người thực hành học máy có kinh nghiệm muốn ôn lại một lĩnh vực, chúng tôi rất vui khi bạn tham gia cùng chúng tôi! Chúng tôi muốn tạo một điểm khởi đầu thân thiện cho việc học học máy của bạn và rất vui được đánh giá, phản hồi và tích hợp [ý kiến đóng góp của bạn](https://github.com/microsoft/ML-For-Beginners/discussions).

[![Giới thiệu về học máy](https://img.youtube.com/vi/h0e2HAPTGF4/0.jpg)](https://youtu.be/h0e2HAPTGF4 "Giới thiệu về học máy")

> 🎥 Nhấp vào hình ảnh trên để xem video: John Guttag từ MIT giới thiệu về học máy

---
## Bắt đầu với học máy

Trước khi bắt đầu với chương trình học này, bạn cần chuẩn bị máy tính của mình để chạy các notebook cục bộ.

- **Cấu hình máy của bạn với các video này**. Sử dụng các liên kết sau để học [cách cài đặt Python](https://youtu.be/CXZYvNRIAKM) trên hệ thống của bạn và [cài đặt trình soạn thảo văn bản](https://youtu.be/EU8eayHWoZg) để phát triển.
- **Học Python**. Bạn cũng nên có hiểu biết cơ bản về [Python](https://docs.microsoft.com/learn/paths/python-language/?WT.mc_id=academic-77952-leestott), một ngôn ngữ lập trình hữu ích cho các nhà khoa học dữ liệu mà chúng tôi sử dụng trong khóa học này.
- **Học Node.js và JavaScript**. Chúng tôi cũng sử dụng JavaScript vài lần trong khóa học này khi xây dựng ứng dụng web, vì vậy bạn cần cài đặt [node](https://nodejs.org) và [npm](https://www.npmjs.com/), cũng như [Visual Studio Code](https://code.visualstudio.com/) để phát triển cả Python và JavaScript.
- **Tạo tài khoản GitHub**. Vì bạn đã tìm thấy chúng tôi trên [GitHub](https://github.com), có thể bạn đã có tài khoản, nhưng nếu chưa, hãy tạo một tài khoản và sau đó fork chương trình học này để sử dụng riêng. (Đừng ngại cho chúng tôi một ngôi sao nhé 😊)
- **Khám phá Scikit-learn**. Làm quen với [Scikit-learn](https://scikit-learn.org/stable/user_guide.html), một bộ thư viện học máy mà chúng tôi tham khảo trong các bài học này.

---
## Học máy là gì?

Thuật ngữ 'học máy' là một trong những thuật ngữ phổ biến và được sử dụng nhiều nhất hiện nay. Có khả năng cao là bạn đã nghe thuật ngữ này ít nhất một lần nếu bạn có chút quen thuộc với công nghệ, bất kể lĩnh vực bạn làm việc. Tuy nhiên, cơ chế của học máy vẫn là một bí ẩn đối với hầu hết mọi người. Đối với người mới bắt đầu học máy, chủ đề này đôi khi có thể cảm thấy quá tải. Vì vậy, điều quan trọng là phải hiểu học máy thực sự là gì và học về nó từng bước, thông qua các ví dụ thực tế.

---
## Đường cong cường điệu

![ml hype curve](../../../../1-Introduction/1-intro-to-ML/images/hype.png)

> Google Trends cho thấy 'đường cong cường điệu' gần đây của thuật ngữ 'học máy'

---
## Một vũ trụ bí ẩn

Chúng ta sống trong một vũ trụ đầy những bí ẩn hấp dẫn. Những nhà khoa học vĩ đại như Stephen Hawking, Albert Einstein, và nhiều người khác đã dành cả cuộc đời để tìm kiếm thông tin có ý nghĩa nhằm khám phá những bí ẩn của thế giới xung quanh chúng ta. Đây là bản chất học hỏi của con người: một đứa trẻ học những điều mới và khám phá cấu trúc của thế giới xung quanh nó từng năm khi nó lớn lên.

---
## Bộ não của trẻ em

Bộ não và các giác quan của trẻ em nhận thức các sự kiện xung quanh và dần dần học các mẫu ẩn của cuộc sống, giúp trẻ tạo ra các quy tắc logic để nhận diện các mẫu đã học. Quá trình học hỏi của bộ não con người khiến con người trở thành sinh vật sống tinh vi nhất trên thế giới này. Việc học liên tục bằng cách khám phá các mẫu ẩn và sau đó đổi mới trên các mẫu đó cho phép chúng ta cải thiện bản thân ngày càng tốt hơn trong suốt cuộc đời. Khả năng học hỏi và phát triển này liên quan đến một khái niệm gọi là [tính dẻo của não](https://www.simplypsychology.org/brain-plasticity.html). Một cách bề mặt, chúng ta có thể rút ra một số điểm tương đồng mang tính động lực giữa quá trình học hỏi của bộ não con người và các khái niệm của học máy.

---
## Bộ não con người

[Bộ não con người](https://www.livescience.com/29365-human-brain.html) nhận thức các sự kiện từ thế giới thực, xử lý thông tin nhận thức, đưa ra quyết định hợp lý và thực hiện các hành động nhất định dựa trên hoàn cảnh. Đây là điều chúng ta gọi là hành xử thông minh. Khi chúng ta lập trình một bản sao của quá trình hành xử thông minh vào một máy móc, nó được gọi là trí tuệ nhân tạo (AI).

---
## Một số thuật ngữ

Mặc dù các thuật ngữ có thể gây nhầm lẫn, học máy (ML) là một phần quan trọng của trí tuệ nhân tạo. **ML liên quan đến việc sử dụng các thuật toán chuyên biệt để khám phá thông tin có ý nghĩa và tìm các mẫu ẩn từ dữ liệu nhận thức nhằm hỗ trợ quá trình ra quyết định hợp lý**.

---
## AI, ML, Học sâu

![AI, ML, deep learning, data science](../../../../1-Introduction/1-intro-to-ML/images/ai-ml-ds.png)

> Một sơ đồ hiển thị mối quan hệ giữa AI, ML, học sâu và khoa học dữ liệu. Đồ họa thông tin bởi [Jen Looper](https://twitter.com/jenlooper) lấy cảm hứng từ [đồ họa này](https://softwareengineering.stackexchange.com/questions/366996/distinction-between-ai-ml-neural-networks-deep-learning-and-data-mining)

---
## Các khái niệm sẽ được đề cập

Trong chương trình học này, chúng ta sẽ chỉ đề cập đến các khái niệm cốt lõi của học máy mà người mới bắt đầu cần biết. Chúng ta sẽ tập trung vào cái gọi là 'học máy cổ điển', chủ yếu sử dụng Scikit-learn, một thư viện xuất sắc mà nhiều sinh viên sử dụng để học các kiến thức cơ bản. Để hiểu các khái niệm rộng hơn về trí tuệ nhân tạo hoặc học sâu, một nền tảng kiến thức vững chắc về học máy là không thể thiếu, và chúng tôi muốn cung cấp điều đó ở đây.

---
## Trong khóa học này bạn sẽ học:

- các khái niệm cốt lõi của học máy
- lịch sử của ML
- ML và sự công bằng
- các kỹ thuật ML hồi quy
- các kỹ thuật ML phân loại
- các kỹ thuật ML phân cụm
- các kỹ thuật ML xử lý ngôn ngữ tự nhiên
- các kỹ thuật ML dự đoán chuỗi thời gian
- học tăng cường
- các ứng dụng thực tế của ML

---
## Những gì chúng ta sẽ không đề cập

- học sâu
- mạng nơ-ron
- AI

Để tạo trải nghiệm học tập tốt hơn, chúng ta sẽ tránh các phức tạp của mạng nơ-ron, 'học sâu' - xây dựng mô hình nhiều lớp bằng mạng nơ-ron - và AI, mà chúng ta sẽ thảo luận trong một chương trình học khác. Chúng tôi cũng sẽ cung cấp một chương trình học khoa học dữ liệu sắp tới để tập trung vào khía cạnh đó của lĩnh vực lớn hơn này.

---
## Tại sao học máy lại quan trọng?

Học máy, từ góc độ hệ thống, được định nghĩa là việc tạo ra các hệ thống tự động có thể học các mẫu ẩn từ dữ liệu để hỗ trợ đưa ra các quyết định thông minh.

Động lực này được lấy cảm hứng một cách lỏng lẻo từ cách bộ não con người học một số điều dựa trên dữ liệu mà nó nhận thức từ thế giới bên ngoài.

✅ Hãy nghĩ trong một phút tại sao một doanh nghiệp lại muốn sử dụng các chiến lược học máy thay vì tạo một hệ thống dựa trên các quy tắc được mã hóa cứng.

---
## Ứng dụng của học máy

Ứng dụng của học máy hiện nay gần như ở khắp mọi nơi, và phổ biến như dữ liệu đang lưu chuyển xung quanh xã hội của chúng ta, được tạo ra bởi điện thoại thông minh, các thiết bị kết nối, và các hệ thống khác. Xét đến tiềm năng to lớn của các thuật toán học máy tiên tiến, các nhà nghiên cứu đã khám phá khả năng của chúng để giải quyết các vấn đề thực tế đa chiều và đa ngành với kết quả tích cực lớn.

---
## Ví dụ về học máy ứng dụng

**Bạn có thể sử dụng học máy theo nhiều cách**:

- Dự đoán khả năng mắc bệnh từ lịch sử y tế hoặc báo cáo của bệnh nhân.
- Sử dụng dữ liệu thời tiết để dự đoán các sự kiện thời tiết.
- Hiểu cảm xúc của một văn bản.
- Phát hiện tin tức giả để ngăn chặn sự lan truyền của tuyên truyền.

Tài chính, kinh tế, khoa học trái đất, khám phá không gian, kỹ thuật y sinh, khoa học nhận thức, và thậm chí các lĩnh vực trong nhân văn đã thích nghi với học máy để giải quyết các vấn đề nặng về xử lý dữ liệu trong lĩnh vực của họ.

---
## Kết luận

Học máy tự động hóa quá trình khám phá mẫu bằng cách tìm kiếm các thông tin có ý nghĩa từ dữ liệu thực tế hoặc dữ liệu được tạo ra. Nó đã chứng minh giá trị của mình trong các ứng dụng kinh doanh, y tế, và tài chính, cùng nhiều lĩnh vực khác.

Trong tương lai gần, việc hiểu các kiến thức cơ bản về học máy sẽ trở thành một yêu cầu cần thiết cho mọi người từ bất kỳ lĩnh vực nào do sự phổ biến rộng rãi của nó.

---
# 🚀 Thử thách

Phác thảo, trên giấy hoặc sử dụng một ứng dụng trực tuyến như [Excalidraw](https://excalidraw.com/), sự hiểu biết của bạn về sự khác biệt giữa AI, ML, học sâu, và khoa học dữ liệu. Thêm một số ý tưởng về các vấn đề mà mỗi kỹ thuật này có thể giải quyết tốt.

# [Câu hỏi sau bài giảng](https://ff-quizzes.netlify.app/en/ml/)

---
# Ôn tập & Tự học

Để tìm hiểu thêm về cách bạn có thể làm việc với các thuật toán ML trên đám mây, hãy theo dõi [Lộ trình học tập](https://docs.microsoft.com/learn/paths/create-no-code-predictive-models-azure-machine-learning/?WT.mc_id=academic-77952-leestott).

Tham gia một [Lộ trình học tập](https://docs.microsoft.com/learn/modules/introduction-to-machine-learning/?WT.mc_id=academic-77952-leestott) về các kiến thức cơ bản của ML.

---
# Bài tập

[Khởi động và chạy](assignment.md)

---

**Tuyên bố miễn trừ trách nhiệm**:  
Tài liệu này đã được dịch bằng dịch vụ dịch thuật AI [Co-op Translator](https://github.com/Azure/co-op-translator). Mặc dù chúng tôi cố gắng đảm bảo độ chính xác, xin lưu ý rằng các bản dịch tự động có thể chứa lỗi hoặc không chính xác. Tài liệu gốc bằng ngôn ngữ bản địa nên được coi là nguồn thông tin chính thức. Đối với các thông tin quan trọng, khuyến nghị sử dụng dịch vụ dịch thuật chuyên nghiệp bởi con người. Chúng tôi không chịu trách nhiệm cho bất kỳ sự hiểu lầm hoặc diễn giải sai nào phát sinh từ việc sử dụng bản dịch này.