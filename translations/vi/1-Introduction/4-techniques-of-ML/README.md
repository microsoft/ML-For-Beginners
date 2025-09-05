<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "9d91f3af3758fdd4569fb410575995ef",
  "translation_date": "2025-09-05T19:35:41+00:00",
  "source_file": "1-Introduction/4-techniques-of-ML/README.md",
  "language_code": "vi"
}
-->
# Kỹ thuật Học Máy

Quy trình xây dựng, sử dụng và duy trì các mô hình học máy cùng dữ liệu mà chúng sử dụng là một quy trình rất khác biệt so với nhiều quy trình phát triển khác. Trong bài học này, chúng ta sẽ làm rõ quy trình này và phác thảo các kỹ thuật chính mà bạn cần biết. Bạn sẽ:

- Hiểu các quy trình nền tảng của học máy ở mức độ cao.
- Khám phá các khái niệm cơ bản như 'mô hình', 'dự đoán', và 'dữ liệu huấn luyện'.

## [Câu hỏi trước bài giảng](https://ff-quizzes.netlify.app/en/ml/)

[![Học máy cho người mới bắt đầu - Kỹ thuật Học Máy](https://img.youtube.com/vi/4NGM0U2ZSHU/0.jpg)](https://youtu.be/4NGM0U2ZSHU "Học máy cho người mới bắt đầu - Kỹ thuật Học Máy")

> 🎥 Nhấp vào hình ảnh trên để xem video ngắn về bài học này.

## Giới thiệu

Ở mức độ cao, việc tạo ra các quy trình học máy (ML) bao gồm một số bước:

1. **Xác định câu hỏi**. Hầu hết các quy trình ML bắt đầu bằng việc đặt ra một câu hỏi mà không thể trả lời bằng một chương trình điều kiện đơn giản hoặc một hệ thống dựa trên quy tắc. Những câu hỏi này thường xoay quanh việc dự đoán dựa trên một tập hợp dữ liệu.
2. **Thu thập và chuẩn bị dữ liệu**. Để có thể trả lời câu hỏi của bạn, bạn cần dữ liệu. Chất lượng và, đôi khi, số lượng dữ liệu của bạn sẽ quyết định mức độ bạn có thể trả lời câu hỏi ban đầu. Việc trực quan hóa dữ liệu là một khía cạnh quan trọng của giai đoạn này. Giai đoạn này cũng bao gồm việc chia dữ liệu thành nhóm huấn luyện và kiểm tra để xây dựng mô hình.
3. **Chọn phương pháp huấn luyện**. Tùy thuộc vào câu hỏi của bạn và bản chất của dữ liệu, bạn cần chọn cách huấn luyện mô hình để phản ánh tốt nhất dữ liệu và đưa ra dự đoán chính xác. Đây là phần của quy trình ML yêu cầu chuyên môn cụ thể và thường cần một lượng lớn thử nghiệm.
4. **Huấn luyện mô hình**. Sử dụng dữ liệu huấn luyện của bạn, bạn sẽ sử dụng các thuật toán khác nhau để huấn luyện mô hình nhận diện các mẫu trong dữ liệu. Mô hình có thể sử dụng các trọng số nội bộ có thể được điều chỉnh để ưu tiên một số phần của dữ liệu hơn các phần khác nhằm xây dựng mô hình tốt hơn.
5. **Đánh giá mô hình**. Bạn sử dụng dữ liệu chưa từng thấy trước đây (dữ liệu kiểm tra của bạn) từ tập dữ liệu đã thu thập để xem mô hình hoạt động như thế nào.
6. **Điều chỉnh tham số**. Dựa trên hiệu suất của mô hình, bạn có thể thực hiện lại quy trình bằng cách sử dụng các tham số hoặc biến khác nhau để kiểm soát hành vi của các thuật toán được sử dụng để huấn luyện mô hình.
7. **Dự đoán**. Sử dụng các đầu vào mới để kiểm tra độ chính xác của mô hình.

## Câu hỏi cần đặt ra

Máy tính đặc biệt giỏi trong việc khám phá các mẫu ẩn trong dữ liệu. Tiện ích này rất hữu ích cho các nhà nghiên cứu có câu hỏi về một lĩnh vực nhất định mà không thể dễ dàng trả lời bằng cách tạo một hệ thống dựa trên quy tắc điều kiện. Ví dụ, trong một nhiệm vụ tính toán bảo hiểm, một nhà khoa học dữ liệu có thể xây dựng các quy tắc thủ công về tỷ lệ tử vong của người hút thuốc so với người không hút thuốc.

Tuy nhiên, khi nhiều biến khác được đưa vào phương trình, một mô hình ML có thể chứng minh hiệu quả hơn trong việc dự đoán tỷ lệ tử vong trong tương lai dựa trên lịch sử sức khỏe trước đây. Một ví dụ vui vẻ hơn có thể là dự đoán thời tiết cho tháng Tư tại một địa điểm cụ thể dựa trên dữ liệu bao gồm vĩ độ, kinh độ, biến đổi khí hậu, khoảng cách đến đại dương, các mẫu luồng khí, và nhiều yếu tố khác.

✅ Bộ [slide này](https://www2.cisl.ucar.edu/sites/default/files/2021-10/0900%20June%2024%20Haupt_0.pdf) về các mô hình thời tiết cung cấp một góc nhìn lịch sử về việc sử dụng ML trong phân tích thời tiết.

## Nhiệm vụ trước khi xây dựng

Trước khi bắt đầu xây dựng mô hình của bạn, có một số nhiệm vụ bạn cần hoàn thành. Để kiểm tra câu hỏi của bạn và hình thành giả thuyết dựa trên dự đoán của mô hình, bạn cần xác định và cấu hình một số yếu tố.

### Dữ liệu

Để có thể trả lời câu hỏi của bạn với bất kỳ mức độ chắc chắn nào, bạn cần một lượng dữ liệu tốt và đúng loại. Có hai điều bạn cần làm tại thời điểm này:

- **Thu thập dữ liệu**. Ghi nhớ bài học trước về tính công bằng trong phân tích dữ liệu, hãy thu thập dữ liệu của bạn một cách cẩn thận. Hãy nhận thức về nguồn gốc của dữ liệu này, bất kỳ thiên kiến nào mà nó có thể mang theo, và ghi lại nguồn gốc của nó.
- **Chuẩn bị dữ liệu**. Có một số bước trong quy trình chuẩn bị dữ liệu. Bạn có thể cần tổng hợp dữ liệu và chuẩn hóa nó nếu nó đến từ các nguồn khác nhau. Bạn có thể cải thiện chất lượng và số lượng dữ liệu thông qua các phương pháp khác nhau như chuyển đổi chuỗi thành số (như chúng ta làm trong [Phân cụm](../../5-Clustering/1-Visualize/README.md)). Bạn cũng có thể tạo dữ liệu mới dựa trên dữ liệu gốc (như chúng ta làm trong [Phân loại](../../4-Classification/1-Introduction/README.md)). Bạn có thể làm sạch và chỉnh sửa dữ liệu (như chúng ta sẽ làm trước bài học [Ứng dụng Web](../../3-Web-App/README.md)). Cuối cùng, bạn cũng có thể cần ngẫu nhiên hóa và xáo trộn dữ liệu, tùy thuộc vào kỹ thuật huấn luyện của bạn.

✅ Sau khi thu thập và xử lý dữ liệu của bạn, hãy dành một chút thời gian để xem liệu hình dạng của nó có cho phép bạn giải quyết câu hỏi dự định hay không. Có thể dữ liệu sẽ không hoạt động tốt trong nhiệm vụ của bạn, như chúng ta phát hiện trong các bài học [Phân cụm](../../5-Clustering/1-Visualize/README.md)!

### Đặc trưng và Mục tiêu

Một [đặc trưng](https://www.datasciencecentral.com/profiles/blogs/an-introduction-to-variable-and-feature-selection) là một thuộc tính có thể đo lường của dữ liệu. Trong nhiều tập dữ liệu, nó được biểu diễn dưới dạng tiêu đề cột như 'ngày', 'kích thước' hoặc 'màu sắc'. Biến đặc trưng của bạn, thường được biểu diễn là `X` trong mã, đại diện cho biến đầu vào sẽ được sử dụng để huấn luyện mô hình.

Mục tiêu là điều bạn đang cố gắng dự đoán. Mục tiêu, thường được biểu diễn là `y` trong mã, đại diện cho câu trả lời cho câu hỏi bạn đang cố gắng hỏi từ dữ liệu: vào tháng 12, **màu sắc** của bí ngô nào sẽ rẻ nhất? ở San Francisco, khu vực nào sẽ có **giá** bất động sản tốt nhất? Đôi khi mục tiêu cũng được gọi là thuộc tính nhãn.

### Chọn biến đặc trưng của bạn

🎓 **Lựa chọn đặc trưng và Trích xuất đặc trưng** Làm thế nào để bạn biết biến nào cần chọn khi xây dựng mô hình? Bạn có thể sẽ trải qua một quy trình lựa chọn đặc trưng hoặc trích xuất đặc trưng để chọn các biến phù hợp nhất cho mô hình hiệu quả nhất. Tuy nhiên, chúng không giống nhau: "Trích xuất đặc trưng tạo ra các đặc trưng mới từ các hàm của các đặc trưng gốc, trong khi lựa chọn đặc trưng trả về một tập hợp con của các đặc trưng." ([nguồn](https://wikipedia.org/wiki/Feature_selection))

### Trực quan hóa dữ liệu của bạn

Một khía cạnh quan trọng trong bộ công cụ của nhà khoa học dữ liệu là khả năng trực quan hóa dữ liệu bằng cách sử dụng một số thư viện xuất sắc như Seaborn hoặc MatPlotLib. Việc biểu diễn dữ liệu của bạn một cách trực quan có thể cho phép bạn khám phá các mối tương quan ẩn mà bạn có thể tận dụng. Các biểu đồ trực quan của bạn cũng có thể giúp bạn phát hiện thiên kiến hoặc dữ liệu không cân bằng (như chúng ta phát hiện trong [Phân loại](../../4-Classification/2-Classifiers-1/README.md)).

### Chia tập dữ liệu của bạn

Trước khi huấn luyện, bạn cần chia tập dữ liệu của mình thành hai hoặc nhiều phần có kích thước không bằng nhau nhưng vẫn đại diện tốt cho dữ liệu.

- **Huấn luyện**. Phần này của tập dữ liệu được sử dụng để huấn luyện mô hình của bạn. Tập này chiếm phần lớn của tập dữ liệu gốc.
- **Kiểm tra**. Tập dữ liệu kiểm tra là một nhóm dữ liệu độc lập, thường được thu thập từ dữ liệu gốc, mà bạn sử dụng để xác nhận hiệu suất của mô hình đã xây dựng.
- **Xác thực**. Tập xác thực là một nhóm nhỏ các ví dụ độc lập mà bạn sử dụng để điều chỉnh các siêu tham số hoặc kiến trúc của mô hình nhằm cải thiện mô hình. Tùy thuộc vào kích thước dữ liệu của bạn và câu hỏi bạn đang hỏi, bạn có thể không cần xây dựng tập thứ ba này (như chúng ta lưu ý trong [Dự báo chuỗi thời gian](../../7-TimeSeries/1-Introduction/README.md)).

## Xây dựng mô hình

Sử dụng dữ liệu huấn luyện của bạn, mục tiêu của bạn là xây dựng một mô hình, hoặc một biểu diễn thống kê của dữ liệu, bằng cách sử dụng các thuật toán khác nhau để **huấn luyện** nó. Việc huấn luyện mô hình cho phép nó tiếp xúc với dữ liệu và đưa ra các giả định về các mẫu mà nó phát hiện, xác nhận, và chấp nhận hoặc từ chối.

### Quyết định phương pháp huấn luyện

Tùy thuộc vào câu hỏi của bạn và bản chất của dữ liệu, bạn sẽ chọn một phương pháp để huấn luyện nó. Khi xem qua [tài liệu của Scikit-learn](https://scikit-learn.org/stable/user_guide.html) - mà chúng ta sử dụng trong khóa học này - bạn có thể khám phá nhiều cách để huấn luyện mô hình. Tùy thuộc vào kinh nghiệm của bạn, bạn có thể phải thử nhiều phương pháp khác nhau để xây dựng mô hình tốt nhất. Bạn có khả năng trải qua một quy trình mà các nhà khoa học dữ liệu đánh giá hiệu suất của mô hình bằng cách cung cấp cho nó dữ liệu chưa từng thấy, kiểm tra độ chính xác, thiên kiến, và các vấn đề làm giảm chất lượng khác, và chọn phương pháp huấn luyện phù hợp nhất cho nhiệm vụ hiện tại.

### Huấn luyện mô hình

Với dữ liệu huấn luyện của bạn, bạn đã sẵn sàng 'fit' nó để tạo ra một mô hình. Bạn sẽ nhận thấy rằng trong nhiều thư viện ML, bạn sẽ thấy mã 'model.fit' - đây là lúc bạn gửi biến đặc trưng của mình dưới dạng một mảng giá trị (thường là 'X') và một biến mục tiêu (thường là 'y').

### Đánh giá mô hình

Khi quá trình huấn luyện hoàn tất (nó có thể mất nhiều lần lặp lại, hoặc 'epochs', để huấn luyện một mô hình lớn), bạn sẽ có thể đánh giá chất lượng của mô hình bằng cách sử dụng dữ liệu kiểm tra để đo lường hiệu suất của nó. Dữ liệu này là một tập hợp con của dữ liệu gốc mà mô hình chưa từng phân tích trước đó. Bạn có thể in ra một bảng các chỉ số về chất lượng của mô hình.

🎓 **Fit mô hình**

Trong bối cảnh học máy, fit mô hình đề cập đến độ chính xác của hàm cơ bản của mô hình khi nó cố gắng phân tích dữ liệu mà nó không quen thuộc.

🎓 **Underfitting** và **overfitting** là các vấn đề phổ biến làm giảm chất lượng của mô hình, khi mô hình fit không đủ tốt hoặc quá tốt. Điều này khiến mô hình đưa ra dự đoán quá sát hoặc quá lỏng lẻo với dữ liệu huấn luyện của nó. Một mô hình overfit dự đoán dữ liệu huấn luyện quá tốt vì nó đã học quá kỹ các chi tiết và nhiễu của dữ liệu. Một mô hình underfit không chính xác vì nó không thể phân tích chính xác dữ liệu huấn luyện của nó hoặc dữ liệu mà nó chưa 'thấy'.

![mô hình overfitting](../../../../1-Introduction/4-techniques-of-ML/images/overfitting.png)
> Đồ họa thông tin bởi [Jen Looper](https://twitter.com/jenlooper)

## Điều chỉnh tham số

Khi quá trình huấn luyện ban đầu hoàn tất, hãy quan sát chất lượng của mô hình và cân nhắc cải thiện nó bằng cách điều chỉnh các 'siêu tham số' của nó. Đọc thêm về quy trình này [trong tài liệu](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-tune-hyperparameters?WT.mc_id=academic-77952-leestott).

## Dự đoán

Đây là thời điểm bạn có thể sử dụng dữ liệu hoàn toàn mới để kiểm tra độ chính xác của mô hình. Trong một môi trường ML 'ứng dụng', nơi bạn đang xây dựng các tài sản web để sử dụng mô hình trong sản xuất, quy trình này có thể bao gồm việc thu thập đầu vào từ người dùng (ví dụ, một lần nhấn nút) để đặt một biến và gửi nó đến mô hình để suy luận hoặc đánh giá.

Trong các bài học này, bạn sẽ khám phá cách sử dụng các bước này để chuẩn bị, xây dựng, kiểm tra, đánh giá, và dự đoán - tất cả các thao tác của một nhà khoa học dữ liệu và hơn thế nữa, khi bạn tiến bộ trong hành trình trở thành một kỹ sư ML 'full stack'.

---

## 🚀Thử thách

Vẽ một biểu đồ luồng phản ánh các bước của một nhà thực hành ML. Bạn thấy mình đang ở đâu trong quy trình này? Bạn dự đoán sẽ gặp khó khăn ở đâu? Điều gì có vẻ dễ dàng đối với bạn?

## [Câu hỏi sau bài giảng](https://ff-quizzes.netlify.app/en/ml/)

## Ôn tập & Tự học

Tìm kiếm trực tuyến các cuộc phỏng vấn với các nhà khoa học dữ liệu thảo luận về công việc hàng ngày của họ. Đây là [một cuộc phỏng vấn](https://www.youtube.com/watch?v=Z3IjgbbCEfs).

## Bài tập

[Phỏng vấn một nhà khoa học dữ liệu](assignment.md)

---

**Tuyên bố miễn trừ trách nhiệm**:  
Tài liệu này đã được dịch bằng dịch vụ dịch thuật AI [Co-op Translator](https://github.com/Azure/co-op-translator). Mặc dù chúng tôi cố gắng đảm bảo độ chính xác, xin lưu ý rằng các bản dịch tự động có thể chứa lỗi hoặc không chính xác. Tài liệu gốc bằng ngôn ngữ bản địa nên được coi là nguồn thông tin chính thức. Đối với các thông tin quan trọng, khuyến nghị sử dụng dịch vụ dịch thuật chuyên nghiệp bởi con người. Chúng tôi không chịu trách nhiệm cho bất kỳ sự hiểu lầm hoặc diễn giải sai nào phát sinh từ việc sử dụng bản dịch này.