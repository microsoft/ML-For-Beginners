<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "1c2ec40cf55c98a028a359c27ef7e45a",
  "translation_date": "2025-09-05T20:36:18+00:00",
  "source_file": "6-NLP/1-Introduction-to-NLP/README.md",
  "language_code": "vi"
}
-->
# Giới thiệu về xử lý ngôn ngữ tự nhiên

Bài học này bao gồm lịch sử ngắn gọn và các khái niệm quan trọng của *xử lý ngôn ngữ tự nhiên*, một lĩnh vực con của *ngôn ngữ học tính toán*.

## [Câu hỏi trước bài giảng](https://ff-quizzes.netlify.app/en/ml/)

## Giới thiệu

NLP, như thường được gọi, là một trong những lĩnh vực nổi tiếng nhất nơi học máy đã được áp dụng và sử dụng trong phần mềm sản xuất.

✅ Bạn có thể nghĩ đến phần mềm nào mà bạn sử dụng hàng ngày có thể tích hợp một số NLP không? Còn các chương trình xử lý văn bản hoặc ứng dụng di động mà bạn thường xuyên sử dụng thì sao?

Bạn sẽ học về:

- **Ý tưởng về ngôn ngữ**. Cách ngôn ngữ phát triển và các lĩnh vực nghiên cứu chính đã được thực hiện.
- **Định nghĩa và khái niệm**. Bạn cũng sẽ học các định nghĩa và khái niệm về cách máy tính xử lý văn bản, bao gồm phân tích cú pháp, ngữ pháp, và xác định danh từ và động từ. Có một số nhiệm vụ lập trình trong bài học này, và một số khái niệm quan trọng sẽ được giới thiệu mà bạn sẽ học cách lập trình trong các bài học tiếp theo.

## Ngôn ngữ học tính toán

Ngôn ngữ học tính toán là một lĩnh vực nghiên cứu và phát triển qua nhiều thập kỷ, nghiên cứu cách máy tính có thể làm việc với, thậm chí hiểu, dịch và giao tiếp bằng ngôn ngữ. Xử lý ngôn ngữ tự nhiên (NLP) là một lĩnh vực liên quan tập trung vào cách máy tính có thể xử lý ngôn ngữ 'tự nhiên', hay ngôn ngữ của con người.

### Ví dụ - nhập liệu bằng giọng nói trên điện thoại

Nếu bạn từng nhập liệu bằng giọng nói thay vì gõ hoặc hỏi một trợ lý ảo một câu hỏi, giọng nói của bạn đã được chuyển đổi thành dạng văn bản và sau đó được xử lý hoặc *phân tích cú pháp* từ ngôn ngữ bạn nói. Các từ khóa được phát hiện sau đó được xử lý thành một định dạng mà điện thoại hoặc trợ lý có thể hiểu và thực hiện.

![comprehension](../../../../6-NLP/1-Introduction-to-NLP/images/comprehension.png)
> Hiểu ngôn ngữ thực sự rất khó! Hình ảnh bởi [Jen Looper](https://twitter.com/jenlooper)

### Công nghệ này được thực hiện như thế nào?

Điều này có thể thực hiện được vì ai đó đã viết một chương trình máy tính để làm điều này. Vài thập kỷ trước, một số nhà văn khoa học viễn tưởng đã dự đoán rằng con người sẽ chủ yếu nói chuyện với máy tính của họ, và máy tính sẽ luôn hiểu chính xác ý nghĩa của họ. Đáng tiếc, vấn đề này hóa ra khó hơn nhiều so với tưởng tượng, và mặc dù ngày nay vấn đề này đã được hiểu rõ hơn, vẫn còn nhiều thách thức lớn trong việc đạt được xử lý ngôn ngữ tự nhiên 'hoàn hảo' khi nói đến việc hiểu ý nghĩa của một câu. Đây là một vấn đề đặc biệt khó khi nói đến việc hiểu sự hài hước hoặc phát hiện cảm xúc như sự mỉa mai trong một câu.

Lúc này, bạn có thể nhớ lại các lớp học ở trường nơi giáo viên dạy về các phần ngữ pháp trong một câu. Ở một số quốc gia, học sinh được dạy ngữ pháp và ngôn ngữ học như một môn học riêng biệt, nhưng ở nhiều nơi, các chủ đề này được bao gồm như một phần của việc học ngôn ngữ: hoặc ngôn ngữ đầu tiên của bạn ở trường tiểu học (học đọc và viết) và có thể là ngôn ngữ thứ hai ở cấp trung học. Đừng lo lắng nếu bạn không phải là chuyên gia trong việc phân biệt danh từ với động từ hoặc trạng từ với tính từ!

Nếu bạn gặp khó khăn với sự khác biệt giữa *hiện tại đơn* và *hiện tại tiếp diễn*, bạn không đơn độc. Đây là một điều thách thức đối với nhiều người, ngay cả những người nói ngôn ngữ đó như tiếng mẹ đẻ. Tin tốt là máy tính thực sự rất giỏi trong việc áp dụng các quy tắc chính thức, và bạn sẽ học cách viết mã để *phân tích cú pháp* một câu tốt như con người. Thách thức lớn hơn mà bạn sẽ khám phá sau này là hiểu *ý nghĩa* và *cảm xúc* của một câu.

## Yêu cầu trước

Đối với bài học này, yêu cầu chính là có thể đọc và hiểu ngôn ngữ của bài học này. Không có bài toán hoặc phương trình nào cần giải. Mặc dù tác giả ban đầu viết bài học này bằng tiếng Anh, nó cũng được dịch sang các ngôn ngữ khác, vì vậy bạn có thể đang đọc một bản dịch. Có những ví dụ nơi một số ngôn ngữ khác nhau được sử dụng (để so sánh các quy tắc ngữ pháp khác nhau của các ngôn ngữ). Những ví dụ này *không* được dịch, nhưng văn bản giải thích thì có, vì vậy ý nghĩa sẽ rõ ràng.

Đối với các nhiệm vụ lập trình, bạn sẽ sử dụng Python và các ví dụ sử dụng Python 3.8.

Trong phần này, bạn sẽ cần và sử dụng:

- **Hiểu Python 3**. Hiểu ngôn ngữ lập trình Python 3, bài học này sử dụng đầu vào, vòng lặp, đọc tệp, mảng.
- **Visual Studio Code + tiện ích mở rộng**. Chúng ta sẽ sử dụng Visual Studio Code và tiện ích mở rộng Python của nó. Bạn cũng có thể sử dụng một IDE Python mà bạn chọn.
- **TextBlob**. [TextBlob](https://github.com/sloria/TextBlob) là một thư viện xử lý văn bản đơn giản hóa cho Python. Làm theo hướng dẫn trên trang TextBlob để cài đặt nó trên hệ thống của bạn (cài đặt cả corpora như được hiển thị dưới đây):

   ```bash
   pip install -U textblob
   python -m textblob.download_corpora
   ```

> 💡 Mẹo: Bạn có thể chạy Python trực tiếp trong môi trường VS Code. Kiểm tra [tài liệu](https://code.visualstudio.com/docs/languages/python?WT.mc_id=academic-77952-leestott) để biết thêm thông tin.

## Nói chuyện với máy móc

Lịch sử cố gắng làm cho máy tính hiểu ngôn ngữ của con người đã kéo dài hàng thập kỷ, và một trong những nhà khoa học đầu tiên xem xét xử lý ngôn ngữ tự nhiên là *Alan Turing*.

### 'Kiểm tra Turing'

Khi Turing nghiên cứu *trí tuệ nhân tạo* vào những năm 1950, ông đã xem xét liệu một bài kiểm tra hội thoại có thể được thực hiện giữa một con người và máy tính (thông qua trao đổi văn bản) nơi con người trong cuộc hội thoại không chắc chắn liệu họ đang nói chuyện với một con người khác hay một máy tính.

Nếu, sau một khoảng thời gian hội thoại nhất định, con người không thể xác định rằng các câu trả lời là từ máy tính hay không, thì liệu máy tính có thể được coi là *đang suy nghĩ*?

### Cảm hứng - 'trò chơi bắt chước'

Ý tưởng này xuất phát từ một trò chơi tiệc tùng gọi là *Trò chơi Bắt chước* nơi một người thẩm vấn ở một phòng riêng và được giao nhiệm vụ xác định ai trong hai người (ở phòng khác) là nam và nữ tương ứng. Người thẩm vấn có thể gửi ghi chú, và phải cố gắng nghĩ ra các câu hỏi mà câu trả lời bằng văn bản tiết lộ giới tính của người bí ẩn. Tất nhiên, những người chơi ở phòng khác đang cố gắng đánh lừa người thẩm vấn bằng cách trả lời các câu hỏi theo cách gây nhầm lẫn hoặc làm người thẩm vấn hiểu sai, đồng thời cũng tạo ra vẻ ngoài trả lời một cách trung thực.

### Phát triển Eliza

Vào những năm 1960, một nhà khoa học MIT tên là *Joseph Weizenbaum* đã phát triển [*Eliza*](https://wikipedia.org/wiki/ELIZA), một 'nhà trị liệu' máy tính sẽ hỏi con người các câu hỏi và tạo ra vẻ ngoài hiểu câu trả lời của họ. Tuy nhiên, mặc dù Eliza có thể phân tích cú pháp một câu và xác định một số cấu trúc ngữ pháp và từ khóa nhất định để đưa ra câu trả lời hợp lý, nó không thể được coi là *hiểu* câu. Nếu Eliza được đưa ra một câu theo định dạng "**Tôi đang** <u>buồn</u>", nó có thể sắp xếp lại và thay thế các từ trong câu để tạo thành câu trả lời "Bạn đã **buồn** <u>bao lâu</u>?".

Điều này tạo ra ấn tượng rằng Eliza hiểu tuyên bố và đang hỏi một câu hỏi tiếp theo, trong khi thực tế, nó chỉ thay đổi thì và thêm một số từ. Nếu Eliza không thể xác định một từ khóa mà nó có câu trả lời, nó sẽ thay vào đó đưa ra một câu trả lời ngẫu nhiên có thể áp dụng cho nhiều tuyên bố khác nhau. Eliza có thể dễ dàng bị đánh lừa, ví dụ nếu người dùng viết "**Bạn là** một <u>chiếc xe đạp</u>", nó có thể trả lời "Tôi đã **là** một <u>chiếc xe đạp</u> bao lâu?", thay vì một câu trả lời hợp lý hơn.

[![Trò chuyện với Eliza](https://img.youtube.com/vi/RMK9AphfLco/0.jpg)](https://youtu.be/RMK9AphfLco "Trò chuyện với Eliza")

> 🎥 Nhấp vào hình ảnh trên để xem video về chương trình ELIZA gốc

> Lưu ý: Bạn có thể đọc mô tả gốc về [Eliza](https://cacm.acm.org/magazines/1966/1/13317-elizaa-computer-program-for-the-study-of-natural-language-communication-between-man-and-machine/abstract) được xuất bản năm 1966 nếu bạn có tài khoản ACM. Ngoài ra, hãy đọc về Eliza trên [wikipedia](https://wikipedia.org/wiki/ELIZA)

## Bài tập - lập trình một bot hội thoại cơ bản

Một bot hội thoại, giống như Eliza, là một chương trình thu hút đầu vào của người dùng và dường như hiểu và phản hồi một cách thông minh. Không giống như Eliza, bot của chúng ta sẽ không có nhiều quy tắc tạo ra vẻ ngoài của một cuộc hội thoại thông minh. Thay vào đó, bot của chúng ta sẽ chỉ có một khả năng duy nhất, đó là tiếp tục cuộc hội thoại với các câu trả lời ngẫu nhiên có thể phù hợp trong hầu hết các cuộc hội thoại đơn giản.

### Kế hoạch

Các bước của bạn khi xây dựng một bot hội thoại:

1. In hướng dẫn cho người dùng cách tương tác với bot
2. Bắt đầu một vòng lặp
   1. Nhận đầu vào từ người dùng
   2. Nếu người dùng yêu cầu thoát, thì thoát
   3. Xử lý đầu vào của người dùng và xác định câu trả lời (trong trường hợp này, câu trả lời là một lựa chọn ngẫu nhiên từ danh sách các câu trả lời chung có thể)
   4. In câu trả lời
3. Quay lại bước 2

### Xây dựng bot

Hãy tạo bot ngay bây giờ. Chúng ta sẽ bắt đầu bằng cách định nghĩa một số câu nói.

1. Tự tạo bot này trong Python với các câu trả lời ngẫu nhiên sau:

    ```python
    random_responses = ["That is quite interesting, please tell me more.",
                        "I see. Do go on.",
                        "Why do you say that?",
                        "Funny weather we've been having, isn't it?",
                        "Let's change the subject.",
                        "Did you catch the game last night?"]
    ```

    Đây là một số đầu ra mẫu để hướng dẫn bạn (đầu vào của người dùng nằm trên các dòng bắt đầu bằng `>`):

    ```output
    Hello, I am Marvin, the simple robot.
    You can end this conversation at any time by typing 'bye'
    After typing each answer, press 'enter'
    How are you today?
    > I am good thanks
    That is quite interesting, please tell me more.
    > today I went for a walk     
    Did you catch the game last night?
    > I did, but my team lost
    Funny weather we've been having, isn't it?
    > yes but I hope next week is better
    Let's change the subject.
    > ok, lets talk about music
    Why do you say that?
    > because I like music!
    Why do you say that?
    > bye
    It was nice talking to you, goodbye!
    ```

    Một giải pháp khả thi cho nhiệm vụ này có thể được tìm thấy [ở đây](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/1-Introduction-to-NLP/solution/bot.py)

    ✅ Dừng lại và suy nghĩ

    1. Bạn có nghĩ rằng các câu trả lời ngẫu nhiên có thể 'đánh lừa' ai đó nghĩ rằng bot thực sự hiểu họ không?
    2. Bot cần có những tính năng gì để hiệu quả hơn?
    3. Nếu một bot thực sự có thể 'hiểu' ý nghĩa của một câu, liệu nó có cần 'nhớ' ý nghĩa của các câu trước đó trong một cuộc hội thoại không?

---

## 🚀Thử thách

Chọn một trong các yếu tố "dừng lại và suy nghĩ" ở trên và thử triển khai nó trong mã hoặc viết một giải pháp trên giấy bằng mã giả.

Trong bài học tiếp theo, bạn sẽ học về một số cách tiếp cận khác để phân tích ngôn ngữ tự nhiên và học máy.

## [Câu hỏi sau bài giảng](https://ff-quizzes.netlify.app/en/ml/)

## Ôn tập & Tự học

Hãy xem các tài liệu tham khảo dưới đây như cơ hội đọc thêm.

### Tài liệu tham khảo

1. Schubert, Lenhart, "Ngôn ngữ học tính toán", *The Stanford Encyclopedia of Philosophy* (Phiên bản Mùa Xuân 2020), Edward N. Zalta (ed.), URL = <https://plato.stanford.edu/archives/spr2020/entries/computational-linguistics/>.
2. Đại học Princeton "Giới thiệu về WordNet." [WordNet](https://wordnet.princeton.edu/). Đại học Princeton. 2010. 

## Bài tập 

[Tìm kiếm một bot](assignment.md)

---

**Tuyên bố miễn trừ trách nhiệm**:  
Tài liệu này đã được dịch bằng dịch vụ dịch thuật AI [Co-op Translator](https://github.com/Azure/co-op-translator). Mặc dù chúng tôi cố gắng đảm bảo độ chính xác, xin lưu ý rằng các bản dịch tự động có thể chứa lỗi hoặc không chính xác. Tài liệu gốc bằng ngôn ngữ bản địa nên được coi là nguồn thông tin chính thức. Đối với các thông tin quan trọng, khuyến nghị sử dụng dịch vụ dịch thuật chuyên nghiệp bởi con người. Chúng tôi không chịu trách nhiệm cho bất kỳ sự hiểu lầm hoặc diễn giải sai nào phát sinh từ việc sử dụng bản dịch này.