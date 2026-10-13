# Xây dựng các giải pháp Machine Learning với AI có trách nhiệm
 
![Tóm tắt AI có trách nhiệm trong Machine Learning dưới dạng sketchnote](../../../../translated_images/vi/ml-fairness.ef296ebec6afc98a.webp)
> Sketchnote bởi [Tomomi Imura](https://www.twitter.com/girlie_mac)

## [Trắc nghiệm trước bài học](https://ff-quizzes.netlify.app/en/ml/)
 
## Giới thiệu

Trong chương trình này, bạn sẽ bắt đầu khám phá cách mà machine learning có thể và đang ảnh hưởng đến cuộc sống hàng ngày của chúng ta. Ngay cả bây giờ, các hệ thống và mô hình đã tham gia vào các nhiệm vụ ra quyết định hàng ngày, như chẩn đoán y tế, phê duyệt khoản vay hoặc phát hiện gian lận. Vì vậy, việc các mô hình này hoạt động tốt để cung cấp kết quả đáng tin cậy là rất quan trọng. Giống như bất kỳ ứng dụng phần mềm nào, hệ thống AI cũng có thể không đáp ứng được kỳ vọng hoặc có kết quả không mong muốn. Đó là lý do tại sao điều quan trọng là phải có khả năng hiểu và giải thích hành vi của một mô hình AI.

Hãy tưởng tượng điều gì có thể xảy ra khi dữ liệu bạn sử dụng để xây dựng các mô hình này thiếu một số nhóm dân số nhất định, chẳng hạn như chủng tộc, giới tính, quan điểm chính trị, tôn giáo, hoặc đại diện không cân đối các nhóm dân số đó. Sẽ ra sao khi kết quả của mô hình bị diễn giải để ưu tiên một nhóm dân số nào đó? Hậu quả đối với ứng dụng là gì? Thêm vào đó, chuyện gì xảy ra khi mô hình có kết quả tiêu cực và gây hại cho con người? Ai sẽ chịu trách nhiệm về hành vi của hệ thống AI? Đây là một số câu hỏi mà chúng ta sẽ khám phá trong chương trình này.

Trong bài học này, bạn sẽ:

- Nâng cao nhận thức về tầm quan trọng của sự công bằng trong machine learning và các tác hại liên quan đến công bằng.
- Làm quen với việc khám phá các ngoại lệ và các kịch bản bất thường để đảm bảo độ tin cậy và an toàn.
- Hiểu về nhu cầu trao quyền cho mọi người bằng cách thiết kế các hệ thống bao gồm.
- Khám phá tầm quan trọng của việc bảo vệ quyền riêng tư và an ninh của dữ liệu và con người.
- Thấy được tầm quan trọng của cách tiếp cận hộp kính để giải thích hành vi của các mô hình AI.
- Nhận thức về tầm quan trọng của trách nhiệm giải trình để xây dựng niềm tin trong các hệ thống AI.

## Yêu cầu trước

Là yêu cầu trước, vui lòng hoàn thành "Nguyên tắc AI có trách nhiệm" trong Hành trình Học tập và xem video dưới đây về chủ đề này:

Tìm hiểu thêm về AI có trách nhiệm bằng cách theo dõi [Hành trình Học tập](https://docs.microsoft.com/learn/modules/responsible-ai-principles/?WT.mc_id=academic-77952-leestott)

[![Cách tiếp cận AI có trách nhiệm của Microsoft](https://img.youtube.com/vi/dnC8-uUZXSc/0.jpg)](https://youtu.be/dnC8-uUZXSc "Cách tiếp cận AI có trách nhiệm của Microsoft")

> 🎥 Nhấn vào hình trên để xem video: Cách tiếp cận AI có trách nhiệm của Microsoft

## Công bằng

Hệ thống AI nên đối xử công bằng với mọi người và tránh ảnh hưởng đến các nhóm người tương tự theo các cách khác nhau. Ví dụ, khi hệ thống AI cung cấp hướng dẫn về điều trị y tế, hồ sơ vay vốn hoặc tuyển dụng, chúng nên đưa ra các khuyến nghị giống nhau cho tất cả mọi người có các triệu chứng, hoàn cảnh tài chính hoặc trình độ chuyên môn tương tự. Mỗi chúng ta, với tư cách là con người, mang trong mình những thiên kiến thừa kế ảnh hưởng đến quyết định và hành động của mình. Những thiên kiến này có thể hiển nhiên trong dữ liệu mà chúng ta sử dụng để huấn luyện các hệ thống AI. Việc thao túng như vậy đôi khi xảy ra một cách vô ý. Thường thì rất khó để nhận biết một cách có ý thức khi bạn đang thêm thiên kiến vào dữ liệu.

**“Sự không công bằng”** bao gồm các tác động tiêu cực, hay “tác hại”, đối với một nhóm người, chẳng hạn như những người được xác định theo chủng tộc, giới tính, tuổi tác hoặc tình trạng khuyết tật. Các tác hại liên quan đến công bằng chính có thể được phân loại như:

- **Phân bổ**, nếu một giới tính hoặc dân tộc nào đó được ưu tiên hơn những người khác.
- **Chất lượng dịch vụ**. Nếu bạn huấn luyện dữ liệu cho một kịch bản cụ thể nhưng thực tế phức tạp hơn nhiều, dẫn đến dịch vụ hoạt động kém. Ví dụ, một vòi xà phòng tự động không thể nhận biết được những người có làn da tối màu. [Tham khảo](https://gizmodo.com/why-cant-this-soap-dispenser-identify-dark-skin-1797931773)
- **Gièm pha**. Phê bình và gán nhãn không công bằng cho điều gì đó hoặc ai đó. Ví dụ, công nghệ gán nhãn hình ảnh đã bị chỉ trích khi gán nhãn sai các hình ảnh người da đen thành khỉ đột.
- **Đại diện quá mức hoặc thiếu đại diện**. Ý tưởng là một nhóm người nào đó không được nhìn thấy trong một nghề nghiệp nhất định, và bất kỳ dịch vụ hoặc chức năng nào cứ tiếp tục thúc đẩy điều này đều góp phần gây hại.
- **Định kiến**. Liên kết một nhóm người với các thuộc tính được gán sẵn. Ví dụ, một hệ thống dịch ngôn ngữ giữa tiếng Anh và tiếng Thổ Nhĩ Kỳ có thể có sự không chính xác do các từ có liên quan đến các định kiến giới tính.

![dịch sang tiếng Thổ Nhĩ Kỳ](../../../../translated_images/vi/gender-bias-translate-en-tr.f185fd8822c2d437.webp)
> dịch sang tiếng Thổ Nhĩ Kỳ

![dịch ngược lại tiếng Anh](../../../../translated_images/vi/gender-bias-translate-tr-en.4eee7e3cecb8c70e.webp)
> dịch ngược lại tiếng Anh

Khi thiết kế và thử nghiệm các hệ thống AI, ta cần đảm bảo AI công bằng và không bị lập trình để đưa ra các quyết định thiên vị hoặc phân biệt đối xử, điều mà con người cũng bị cấm làm. Đảm bảo công bằng trong AI và machine learning vẫn là một thách thức xã hội-kỹ thuật phức tạp.

### Độ tin cậy và an toàn

Để xây dựng lòng tin, các hệ thống AI cần đáng tin cậy, an toàn và nhất quán dưới các điều kiện bình thường và bất ngờ. Việc biết hệ thống AI sẽ hoạt động như thế nào trong nhiều tình huống khác nhau, đặc biệt là khi có ngoại lệ, rất quan trọng. Khi xây dựng các giải pháp AI, cần tập trung lớn vào cách xử lý đa dạng các hoàn cảnh mà giải pháp AI có thể gặp phải. Ví dụ, một chiếc xe tự lái cần đặt an toàn của người dân lên hàng đầu. Do đó, AI điều khiển xe cần cân nhắc tất cả các kịch bản có thể xảy ra như ban đêm, giông bão hoặc bão tuyết, trẻ con chạy qua đường, thú cưng, công trường xây dựng v.v. Mức độ tốt của hệ thống AI trong việc xử lý nhiều điều kiện đa dạng một cách tin cậy và an toàn phản ánh mức độ tiên liệu mà nhà khoa học dữ liệu hay nhà phát triển AI đã cân nhắc trong quá trình thiết kế hoặc thử nghiệm hệ thống.

> [🎥 Nhấn vào đây để xem video: ](https://www.microsoft.com/videoplayer/embed/RE4vvIl)

### Tính bao gồm

Hệ thống AI nên được thiết kế để thu hút và trao quyền cho mọi người. Khi thiết kế và triển khai hệ thống AI, các nhà khoa học dữ liệu và nhà phát triển AI xác định và giải quyết các rào cản tiềm ẩn trong hệ thống có thể vô tình loại trừ mọi người. Ví dụ, trên thế giới có 1 tỷ người khuyết tật. Với sự tiến bộ của AI, họ có thể tiếp cận nhiều thông tin và cơ hội hơn trong cuộc sống hàng ngày dễ dàng hơn. Bằng cách xử lý các rào cản này, nó tạo ra cơ hội đổi mới và phát triển các sản phẩm AI với trải nghiệm tốt hơn mang lại lợi ích cho tất cả mọi người.

> [🎥 Nhấn vào đây để xem video: tính bao gồm trong AI](https://www.microsoft.com/videoplayer/embed/RE4vl9v)

### Bảo mật và quyền riêng tư

Hệ thống AI cần an toàn và tôn trọng quyền riêng tư của con người. Mọi người ít tin tưởng vào những hệ thống mà đặt quyền riêng tư, thông tin hoặc cuộc sống của họ vào nguy cơ. Khi huấn luyện các mô hình machine learning, ta dựa vào dữ liệu để tạo ra kết quả tốt nhất. Do đó, nguồn gốc dữ liệu và tính toàn vẹn phải được xem xét. Ví dụ, dữ liệu có phải do người dùng cung cấp hay là dữ liệu công khai? Tiếp theo, khi làm việc với dữ liệu, việc phát triển các hệ thống AI có thể bảo vệ thông tin bảo mật và chống lại các cuộc tấn công là rất quan trọng. Khi AI trở nên phổ biến hơn, việc bảo vệ quyền riêng tư và bảo mật thông tin cá nhân và kinh doanh quan trọng càng trở nên phức tạp hơn. Các vấn đề về quyền riêng tư và bảo mật dữ liệu đòi hỏi sự chú ý đặc biệt đối với AI vì việc truy cập dữ liệu là cần thiết để các hệ thống AI đưa ra các dự đoán và quyết định chính xác, có căn cứ về con người.

> [🎥 Nhấn vào đây để xem video: bảo mật trong AI](https://www.microsoft.com/videoplayer/embed/RE4voJF)

- Là một ngành, chúng ta đã đạt được tiến bộ quan trọng trong quyền riêng tư & bảo mật, được thúc đẩy đáng kể bởi các quy định như GDPR (Quy định Bảo vệ Dữ liệu Chung).
- Tuy nhiên với các hệ thống AI, chúng ta phải thừa nhận sự căng thẳng giữa nhu cầu có nhiều dữ liệu cá nhân hơn để làm hệ thống cá nhân hóa và hiệu quả hơn – và quyền riêng tư.
- Cũng giống như với sự ra đời của các máy tính kết nối internet, chúng ta cũng đang thấy sự gia tăng lớn các vấn đề bảo mật liên quan đến AI.
- Đồng thời, AI cũng được dùng để cải thiện bảo mật. Ví dụ, hầu hết các phần mềm quét virus hiện đại ngày nay được điều khiển bằng các thuật toán AI.
- Chúng ta cần đảm bảo quy trình Khoa học Dữ liệu của mình kết hợp hài hòa với các thực hành bảo mật và quyền riêng tư mới nhất.


### Minh bạch
Các hệ thống AI cần dễ hiểu. Một phần thiết yếu của minh bạch là giải thích hành vi của các hệ thống AI và các thành phần của chúng. Cải thiện sự hiểu biết về các hệ thống AI yêu cầu các bên liên quan nhận thức được cách và lý do hoạt động của chúng để có thể xác định các vấn đề tiềm ẩn về hiệu suất, an toàn và quyền riêng tư, thiên kiến, các thực hành loại trừ, hoặc kết quả không mong muốn. Chúng tôi cũng tin rằng những người sử dụng hệ thống AI nên trung thực và minh bạch về khi nào, tại sao và cách họ chọn triển khai chúng, cũng như các giới hạn của hệ thống họ sử dụng. Ví dụ, nếu một ngân hàng sử dụng hệ thống AI để hỗ trợ quyết định cho vay tiêu dùng, thì việc xem xét kết quả và hiểu biết dữ liệu nào ảnh hưởng đến các khuyến nghị của hệ thống là rất quan trọng. Chính phủ đang bắt đầu điều chỉnh AI trên nhiều ngành, vì vậy các nhà khoa học dữ liệu và tổ chức phải giải thích liệu hệ thống AI có đáp ứng các yêu cầu về quy định hay không, đặc biệt khi có kết quả không mong muốn.

> [🎥 Nhấn vào đây để xem video: minh bạch trong AI](https://www.microsoft.com/videoplayer/embed/RE4voJF)

- Vì các hệ thống AI rất phức tạp, nên rất khó hiểu cách thức chúng hoạt động và giải thích kết quả.
- Thiếu hiểu biết này ảnh hưởng đến cách các hệ thống được quản lý, vận hành và ghi chép.
- Thiếu hiểu biết này, quan trọng hơn, ảnh hưởng đến các quyết định được đưa ra dựa trên kết quả mà các hệ thống này tạo ra.

### Trách nhiệm giải trình
 
Những người thiết kế và triển khai hệ thống AI phải chịu trách nhiệm về cách hệ thống của họ vận hành. Nhu cầu về trách nhiệm giải trình là đặc biệt quan trọng với các công nghệ sử dụng nhạy cảm như nhận diện khuôn mặt. Gần đây, có sự gia tăng nhu cầu về công nghệ nhận diện khuôn mặt, đặc biệt từ các tổ chức thực thi pháp luật, những người thấy tiềm năng của công nghệ trong các ứng dụng như tìm kiếm trẻ em mất tích. Tuy nhiên, công nghệ này có thể bị chính phủ dùng để đe dọa các quyền tự do cơ bản của công dân, ví dụ như cho phép giám sát liên tục những cá nhân cụ thể. Do đó, các nhà khoa học dữ liệu và tổ chức cần chịu trách nhiệm về cách hệ thống AI của họ ảnh hưởng đến cá nhân hoặc xã hội.

[![Nhà nghiên cứu AI hàng đầu cảnh báo về giám sát hàng loạt qua nhận diện khuôn mặt](../../../../translated_images/vi/accountability.41d8c0f4b85b6231.webp)](https://www.youtube.com/watch?v=Wldt8P5V6D0 "Cách tiếp cận AI có trách nhiệm của Microsoft")

> 🎥 Nhấn vào hình trên để xem video: Cảnh báo về giám sát hàng loạt qua nhận diện khuôn mặt

Cuối cùng, một trong những câu hỏi lớn nhất đối với thế hệ chúng ta, là thế hệ đầu tiên đưa AI vào xã hội, là làm thế nào để chắc chắn rằng máy tính sẽ luôn chịu trách nhiệm với con người và làm thế nào để đảm bảo những người thiết kế máy tính luôn chịu trách nhiệm với tất cả mọi người.

## Đánh giá tác động

Trước khi huấn luyện một mô hình machine learning, việc tiến hành đánh giá tác động là rất quan trọng để hiểu mục đích của hệ thống AI; mục đích sử dụng dự định; nơi nó sẽ được triển khai; và ai sẽ tương tác với hệ thống. Điều này giúp người đánh giá hoặc kiểm thử hiểu các yếu tố cần cân nhắc khi xác định các rủi ro tiềm ẩn và hậu quả mong đợi.

Các lĩnh vực cần tập trung khi tiến hành đánh giá tác động bao gồm:

* **Tác động bất lợi đến cá nhân**. Nhận thức về bất kỳ hạn chế hoặc yêu cầu nào, việc sử dụng không được hỗ trợ hoặc bất kỳ giới hạn nào biết đến cản trở hiệu suất hệ thống là rất quan trọng để đảm bảo hệ thống không được sử dụng theo cách có thể gây hại cho cá nhân.
* **Yêu cầu dữ liệu**. Hiểu cách và nơi hệ thống sẽ sử dụng dữ liệu giúp người đánh giá khám phá bất kỳ yêu cầu dữ liệu nào cần lưu ý (ví dụ, các quy định dữ liệu GDPR hoặc HIPAA). Thêm vào đó, xem xét nguồn gốc hoặc số lượng dữ liệu có đủ lớn để đào tạo không.
* **Tóm tắt tác động**. Tập hợp danh sách các tổn hại tiềm ẩn có thể phát sinh từ việc sử dụng hệ thống. Trong suốt vòng đời ML, kiểm tra xem các vấn đề đã được giảm thiểu hoặc xử lý chưa.
* **Mục tiêu áp dụng** cho mỗi sáu nguyên tắc cốt lõi. Đánh giá xem các mục tiêu của từng nguyên tắc có được đáp ứng không và có khoảng trống nào không.


## Gỡ lỗi với AI có trách nhiệm

Tương tự như gỡ lỗi một ứng dụng phần mềm, gỡ lỗi hệ thống AI là quá trình cần thiết để xác định và giải quyết các vấn đề trong hệ thống. Có nhiều yếu tố ảnh hưởng đến việc một mô hình không hoạt động như mong đợi hoặc không có trách nhiệm. Hầu hết các chỉ số hiệu suất mô hình truyền thống là tổng hợp định lượng hiệu suất mô hình, không đủ để phân tích cách mô hình vi phạm các nguyên tắc AI có trách nhiệm. Hơn nữa, một mô hình machine learning là một hộp đen khiến khó hiểu điều gì thúc đẩy kết quả hoặc cung cấp lời giải thích khi nó mắc sai lầm. Trong khóa học này, chúng ta sẽ học cách sử dụng bảng điều khiển Responsible AI để giúp gỡ lỗi các hệ thống AI. Bảng điều khiển cung cấp công cụ toàn diện cho các nhà khoa học dữ liệu và nhà phát triển AI thực hiện:

* **Phân tích lỗi**. Để xác định phân bố lỗi của mô hình có thể ảnh hưởng đến sự công bằng hoặc độ tin cậy của hệ thống.
* **Tổng quan mô hình**. Để phát hiện nơi có sự chênh lệch về hiệu suất của mô hình trên các nhóm dữ liệu.
* **Phân tích dữ liệu**. Để hiểu phân bố dữ liệu và xác định bất kỳ thiên kiến tiềm ẩn nào trong dữ liệu có thể dẫn đến các vấn đề về công bằng, bao gồm, và độ tin cậy.
* **Khả năng giải thích mô hình**. Để hiểu điều gì ảnh hưởng hoặc tác động đến dự đoán của mô hình. Điều này giúp giải thích hành vi của mô hình, rất quan trọng cho tính minh bạch và trách nhiệm giải trình.


## 🚀 Thử thách
 
Để ngăn chặn các tác hại được tạo ra ngay từ đầu, chúng ta nên:

- có sự đa dạng về nền tảng và quan điểm trong những người làm việc trên các hệ thống
- đầu tư vào các bộ dữ liệu phản ánh sự đa dạng của xã hội chúng ta
- phát triển các phương pháp tốt hơn trong suốt vòng đời machine learning để phát hiện và sửa chữa AI có trách nhiệm khi xảy ra

Hãy nghĩ về các kịch bản thực tế nơi sự không đáng tin cậy của một mô hình trở nên rõ ràng trong việc xây dựng và sử dụng mô hình. Chúng ta còn nên cân nhắc điều gì khác?

## [Trắc nghiệm sau bài học](https://ff-quizzes.netlify.app/en/ml/)

## Ôn tập & Tự học
 

Trong bài học này, bạn đã học được một số kiến thức cơ bản về các khái niệm công bằng và không công bằng trong học máy.  
 
Xem hội thảo này để tìm hiểu sâu hơn về các chủ đề: 

- Theo đuổi AI có trách nhiệm: Đem các nguyên tắc vào thực tiễn bởi Besmira Nushi, Mehrnoosh Sameki và Amit Sharma

[![Responsible AI Toolbox: An open-source framework for building responsible AI](https://img.youtube.com/vi/tGgJCrA-MZU/0.jpg)](https://www.youtube.com/watch?v=tGgJCrA-MZU "RAI Toolbox: An open-source framework for building responsible AI")

> 🎥 Nhấp vào hình ảnh trên để xem video: RAI Toolbox: Một khung làm việc mã nguồn mở để xây dựng AI có trách nhiệm bởi Besmira Nushi, Mehrnoosh Sameki và Amit Sharma

Ngoài ra, hãy đọc: 

- Trung tâm tài nguyên RAI của Microsoft: [Responsible AI Resources – Microsoft AI](https://www.microsoft.com/ai/responsible-ai-resources?activetab=pivot1%3aprimaryr4) 

- Nhóm nghiên cứu FATE của Microsoft: [FATE: Fairness, Accountability, Transparency, and Ethics in AI - Microsoft Research](https://www.microsoft.com/research/theme/fate/) 

Hộp công cụ RAI: 

- [Kho GitHub Responsible AI Toolbox](https://github.com/microsoft/responsible-ai-toolbox)

Đọc về các công cụ của Azure Machine Learning để đảm bảo công bằng:

- [Azure Machine Learning](https://docs.microsoft.com/azure/machine-learning/concept-fairness-ml?WT.mc_id=academic-77952-leestott) 

## Bài tập

[Khám phá Hộp công cụ RAI](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**Tuyên bố miễn trừ trách nhiệm**:
Tài liệu này đã được dịch bằng dịch vụ dịch thuật AI [Co-op Translator](https://github.com/Azure/co-op-translator). Mặc dù chúng tôi cố gắng đảm bảo độ chính xác, xin lưu ý rằng bản dịch tự động có thể chứa lỗi hoặc sai sót. Tài liệu gốc bằng ngôn ngữ gốc nên được coi là nguồn tin chính thức. Đối với thông tin quan trọng, nên sử dụng dịch vụ dịch thuật chuyên nghiệp bởi con người. Chúng tôi không chịu trách nhiệm về bất kỳ hiểu lầm hoặc giải thích sai nào phát sinh từ việc sử dụng bản dịch này.
<!-- CO-OP TRANSLATOR DISCLAIMER END -->