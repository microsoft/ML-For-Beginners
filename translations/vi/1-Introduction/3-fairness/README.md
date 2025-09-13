<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "9a6b702d1437c0467e3c5c28d763dac2",
  "translation_date": "2025-09-05T19:31:55+00:00",
  "source_file": "1-Introduction/3-fairness/README.md",
  "language_code": "vi"
}
-->
# Xây dựng giải pháp Machine Learning với AI có trách nhiệm

![Tóm tắt về AI có trách nhiệm trong Machine Learning qua sketchnote](../../../../sketchnotes/ml-fairness.png)
> Sketchnote bởi [Tomomi Imura](https://www.twitter.com/girlie_mac)

## [Câu hỏi trước bài giảng](https://ff-quizzes.netlify.app/en/ml/)

## Giới thiệu

Trong chương trình học này, bạn sẽ bắt đầu khám phá cách mà machine learning có thể và đang ảnh hưởng đến cuộc sống hàng ngày của chúng ta. Ngay cả hiện tại, các hệ thống và mô hình đã tham gia vào các nhiệm vụ ra quyết định hàng ngày, chẳng hạn như chẩn đoán y tế, phê duyệt khoản vay hoặc phát hiện gian lận. Vì vậy, điều quan trọng là các mô hình này phải hoạt động tốt để cung cấp kết quả đáng tin cậy. Giống như bất kỳ ứng dụng phần mềm nào, các hệ thống AI cũng có thể không đáp ứng được kỳ vọng hoặc tạo ra kết quả không mong muốn. Đó là lý do tại sao việc hiểu và giải thích hành vi của một mô hình AI là rất cần thiết.

Hãy tưởng tượng điều gì có thể xảy ra khi dữ liệu bạn sử dụng để xây dựng các mô hình này thiếu các nhóm nhân khẩu học nhất định, chẳng hạn như chủng tộc, giới tính, quan điểm chính trị, tôn giáo, hoặc đại diện không cân đối cho các nhóm nhân khẩu học đó. Điều gì xảy ra khi đầu ra của mô hình được diễn giải để ưu tiên một số nhóm nhân khẩu học? Hậu quả đối với ứng dụng là gì? Ngoài ra, điều gì xảy ra khi mô hình có kết quả bất lợi và gây hại cho con người? Ai sẽ chịu trách nhiệm cho hành vi của hệ thống AI? Đây là một số câu hỏi mà chúng ta sẽ khám phá trong chương trình học này.

Trong bài học này, bạn sẽ:

- Nâng cao nhận thức về tầm quan trọng của sự công bằng trong machine learning và các tác hại liên quan đến sự không công bằng.
- Làm quen với việc khám phá các trường hợp ngoại lệ và tình huống bất thường để đảm bảo độ tin cậy và an toàn.
- Hiểu rõ về nhu cầu trao quyền cho mọi người bằng cách thiết kế các hệ thống toàn diện.
- Khám phá tầm quan trọng của việc bảo vệ quyền riêng tư và an ninh của dữ liệu và con người.
- Thấy được sự cần thiết của cách tiếp cận "hộp kính" để giải thích hành vi của các mô hình AI.
- Nhận thức rằng trách nhiệm là yếu tố thiết yếu để xây dựng niềm tin vào các hệ thống AI.

## Điều kiện tiên quyết

Trước khi bắt đầu, hãy tham gia lộ trình học "Nguyên tắc AI có trách nhiệm" và xem video dưới đây về chủ đề này:

Tìm hiểu thêm về AI có trách nhiệm qua [Lộ trình học](https://docs.microsoft.com/learn/modules/responsible-ai-principles/?WT.mc_id=academic-77952-leestott)

[![Cách tiếp cận của Microsoft đối với AI có trách nhiệm](https://img.youtube.com/vi/dnC8-uUZXSc/0.jpg)](https://youtu.be/dnC8-uUZXSc "Cách tiếp cận của Microsoft đối với AI có trách nhiệm")

> 🎥 Nhấp vào hình ảnh trên để xem video: Cách tiếp cận của Microsoft đối với AI có trách nhiệm

## Công bằng

Các hệ thống AI nên đối xử công bằng với mọi người và tránh ảnh hưởng đến các nhóm tương tự theo cách khác nhau. Ví dụ, khi các hệ thống AI cung cấp hướng dẫn về điều trị y tế, đơn xin vay vốn, hoặc việc làm, chúng nên đưa ra các khuyến nghị giống nhau cho mọi người có triệu chứng, hoàn cảnh tài chính, hoặc trình độ chuyên môn tương tự. Mỗi chúng ta, với tư cách là con người, đều mang theo những định kiến di truyền ảnh hưởng đến quyết định và hành động của mình. Những định kiến này có thể xuất hiện trong dữ liệu mà chúng ta sử dụng để huấn luyện các hệ thống AI. Đôi khi, sự thao túng này xảy ra một cách vô tình. Thường rất khó để nhận thức rõ ràng khi bạn đang đưa định kiến vào dữ liệu.

**“Sự không công bằng”** bao gồm các tác động tiêu cực, hay “tác hại”, đối với một nhóm người, chẳng hạn như những người được định nghĩa theo chủng tộc, giới tính, tuổi tác, hoặc tình trạng khuyết tật. Các tác hại chính liên quan đến sự không công bằng có thể được phân loại như sau:

- **Phân bổ**, nếu một giới tính hoặc dân tộc, chẳng hạn, được ưu tiên hơn nhóm khác.
- **Chất lượng dịch vụ**. Nếu bạn huấn luyện dữ liệu cho một kịch bản cụ thể nhưng thực tế phức tạp hơn nhiều, điều này dẫn đến dịch vụ hoạt động kém. Ví dụ, một máy phân phối xà phòng không thể nhận diện người có làn da tối màu. [Tham khảo](https://gizmodo.com/why-cant-this-soap-dispenser-identify-dark-skin-1797931773)
- **Phỉ báng**. Chỉ trích và gán nhãn không công bằng cho một thứ hoặc một người. Ví dụ, công nghệ gán nhãn hình ảnh từng gán nhãn sai hình ảnh của người da tối màu là khỉ đột.
- **Đại diện quá mức hoặc thiếu đại diện**. Ý tưởng rằng một nhóm nhất định không được nhìn thấy trong một nghề nghiệp nào đó, và bất kỳ dịch vụ hoặc chức năng nào tiếp tục thúc đẩy điều đó đều góp phần gây hại.
- **Định kiến**. Gắn một nhóm nhất định với các thuộc tính được gán trước. Ví dụ, một hệ thống dịch ngôn ngữ giữa tiếng Anh và tiếng Thổ Nhĩ Kỳ có thể gặp sai sót do các từ có liên kết định kiến với giới tính.

![dịch sang tiếng Thổ Nhĩ Kỳ](../../../../1-Introduction/3-fairness/images/gender-bias-translate-en-tr.png)
> dịch sang tiếng Thổ Nhĩ Kỳ

![dịch lại sang tiếng Anh](../../../../1-Introduction/3-fairness/images/gender-bias-translate-tr-en.png)
> dịch lại sang tiếng Anh

Khi thiết kế và kiểm tra các hệ thống AI, chúng ta cần đảm bảo rằng AI công bằng và không được lập trình để đưa ra các quyết định thiên vị hoặc phân biệt đối xử, điều mà con người cũng bị cấm thực hiện. Đảm bảo sự công bằng trong AI và machine learning vẫn là một thách thức xã hội-kỹ thuật phức tạp.

### Độ tin cậy và an toàn

Để xây dựng niềm tin, các hệ thống AI cần phải đáng tin cậy, an toàn, và nhất quán trong điều kiện bình thường và bất ngờ. Điều quan trọng là phải biết các hệ thống AI sẽ hoạt động như thế nào trong nhiều tình huống khác nhau, đặc biệt là khi chúng gặp các trường hợp ngoại lệ. Khi xây dựng các giải pháp AI, cần tập trung đáng kể vào cách xử lý một loạt các tình huống mà các giải pháp AI có thể gặp phải. Ví dụ, một chiếc xe tự lái cần đặt sự an toàn của con người lên hàng đầu. Do đó, AI điều khiển xe cần xem xét tất cả các kịch bản có thể xảy ra như ban đêm, giông bão, bão tuyết, trẻ em chạy qua đường, thú cưng, công trình đường bộ, v.v. Mức độ mà một hệ thống AI có thể xử lý một loạt các điều kiện một cách đáng tin cậy và an toàn phản ánh mức độ dự đoán mà nhà khoa học dữ liệu hoặc nhà phát triển AI đã xem xét trong quá trình thiết kế hoặc kiểm tra hệ thống.

> [🎥 Nhấp vào đây để xem video: ](https://www.microsoft.com/videoplayer/embed/RE4vvIl)

### Tính toàn diện

Các hệ thống AI nên được thiết kế để thu hút và trao quyền cho mọi người. Khi thiết kế và triển khai các hệ thống AI, các nhà khoa học dữ liệu và nhà phát triển AI cần xác định và giải quyết các rào cản tiềm năng trong hệ thống có thể vô tình loại trừ một số người. Ví dụ, có 1 tỷ người khuyết tật trên toàn thế giới. Với sự phát triển của AI, họ có thể dễ dàng tiếp cận một loạt thông tin và cơ hội trong cuộc sống hàng ngày. Bằng cách giải quyết các rào cản, điều này tạo ra cơ hội đổi mới và phát triển các sản phẩm AI với trải nghiệm tốt hơn, mang lại lợi ích cho tất cả mọi người.

> [🎥 Nhấp vào đây để xem video: tính toàn diện trong AI](https://www.microsoft.com/videoplayer/embed/RE4vl9v)

### An ninh và quyền riêng tư

Các hệ thống AI nên an toàn và tôn trọng quyền riêng tư của mọi người. Mọi người ít tin tưởng vào các hệ thống đặt quyền riêng tư, thông tin, hoặc cuộc sống của họ vào rủi ro. Khi huấn luyện các mô hình machine learning, chúng ta dựa vào dữ liệu để tạo ra kết quả tốt nhất. Trong quá trình này, nguồn gốc và tính toàn vẹn của dữ liệu phải được xem xét. Ví dụ, dữ liệu có được người dùng cung cấp hay công khai? Tiếp theo, trong khi làm việc với dữ liệu, điều quan trọng là phải phát triển các hệ thống AI có thể bảo vệ thông tin bí mật và chống lại các cuộc tấn công. Khi AI trở nên phổ biến hơn, việc bảo vệ quyền riêng tư và đảm bảo an ninh cho thông tin cá nhân và doanh nghiệp quan trọng ngày càng trở nên cấp thiết và phức tạp. Các vấn đề về quyền riêng tư và bảo mật dữ liệu đòi hỏi sự chú ý đặc biệt đối với AI vì việc truy cập dữ liệu là rất cần thiết để các hệ thống AI đưa ra dự đoán và quyết định chính xác, có thông tin về con người.

> [🎥 Nhấp vào đây để xem video: an ninh trong AI](https://www.microsoft.com/videoplayer/embed/RE4voJF)

- Ngành công nghiệp đã đạt được những tiến bộ đáng kể trong quyền riêng tư và bảo mật, được thúc đẩy đáng kể bởi các quy định như GDPR (Quy định chung về bảo vệ dữ liệu).
- Tuy nhiên, với các hệ thống AI, chúng ta phải thừa nhận sự căng thẳng giữa nhu cầu về dữ liệu cá nhân để làm cho các hệ thống trở nên cá nhân hóa và hiệu quả hơn – và quyền riêng tư.
- Giống như sự ra đời của máy tính kết nối với internet, chúng ta cũng đang chứng kiến sự gia tăng lớn về số lượng các vấn đề bảo mật liên quan đến AI.
- Đồng thời, chúng ta đã thấy AI được sử dụng để cải thiện an ninh. Ví dụ, hầu hết các trình quét virus hiện đại đều được điều khiển bởi các thuật toán AI.
- Chúng ta cần đảm bảo rằng các quy trình khoa học dữ liệu của mình hòa hợp với các thực tiễn bảo mật và quyền riêng tư mới nhất.

### Tính minh bạch

Các hệ thống AI nên dễ hiểu. Một phần quan trọng của tính minh bạch là giải thích hành vi của các hệ thống AI và các thành phần của chúng. Việc cải thiện sự hiểu biết về các hệ thống AI đòi hỏi các bên liên quan phải hiểu cách thức và lý do chúng hoạt động để có thể xác định các vấn đề về hiệu suất tiềm năng, lo ngại về an toàn và quyền riêng tư, định kiến, các thực tiễn loại trừ, hoặc kết quả không mong muốn. Chúng tôi cũng tin rằng những người sử dụng các hệ thống AI nên trung thực và cởi mở về thời điểm, lý do, và cách họ chọn triển khai chúng, cũng như những hạn chế của các hệ thống mà họ sử dụng. Ví dụ, nếu một ngân hàng sử dụng hệ thống AI để hỗ trợ các quyết định cho vay tiêu dùng, điều quan trọng là phải kiểm tra kết quả và hiểu dữ liệu nào ảnh hưởng đến các khuyến nghị của hệ thống. Các chính phủ đang bắt đầu điều chỉnh AI trong các ngành công nghiệp, vì vậy các nhà khoa học dữ liệu và tổ chức phải giải thích liệu hệ thống AI có đáp ứng các yêu cầu quy định hay không, đặc biệt là khi có kết quả không mong muốn.

> [🎥 Nhấp vào đây để xem video: tính minh bạch trong AI](https://www.microsoft.com/videoplayer/embed/RE4voJF)

- Vì các hệ thống AI rất phức tạp, nên rất khó để hiểu cách chúng hoạt động và diễn giải kết quả.
- Sự thiếu hiểu biết này ảnh hưởng đến cách các hệ thống này được quản lý, vận hành, và ghi chép.
- Quan trọng hơn, sự thiếu hiểu biết này ảnh hưởng đến các quyết định được đưa ra dựa trên kết quả mà các hệ thống này tạo ra.

### Trách nhiệm

Những người thiết kế và triển khai các hệ thống AI phải chịu trách nhiệm về cách các hệ thống của họ hoạt động. Nhu cầu về trách nhiệm đặc biệt quan trọng với các công nghệ nhạy cảm như nhận diện khuôn mặt. Gần đây, đã có nhu cầu ngày càng tăng đối với công nghệ nhận diện khuôn mặt, đặc biệt từ các tổ chức thực thi pháp luật, những người thấy tiềm năng của công nghệ này trong các ứng dụng như tìm kiếm trẻ em mất tích. Tuy nhiên, các công nghệ này có thể được sử dụng bởi một chính phủ để đặt các quyền tự do cơ bản của công dân vào rủi ro, chẳng hạn như cho phép giám sát liên tục các cá nhân cụ thể. Do đó, các nhà khoa học dữ liệu và tổ chức cần chịu trách nhiệm về cách hệ thống AI của họ ảnh hưởng đến cá nhân hoặc xã hội.

[![Nhà nghiên cứu AI hàng đầu cảnh báo về giám sát hàng loạt qua nhận diện khuôn mặt](../../../../1-Introduction/3-fairness/images/accountability.png)](https://www.youtube.com/watch?v=Wldt8P5V6D0 "Cách tiếp cận của Microsoft đối với AI có trách nhiệm")

> 🎥 Nhấp vào hình ảnh trên để xem video: Cảnh báo về giám sát hàng loạt qua nhận diện khuôn mặt

Cuối cùng, một trong những câu hỏi lớn nhất cho thế hệ của chúng ta, với tư cách là thế hệ đầu tiên đưa AI vào xã hội, là làm thế nào để đảm bảo rằng máy tính sẽ luôn chịu trách nhiệm trước con người và làm thế nào để đảm bảo rằng những người thiết kế máy tính chịu trách nhiệm trước tất cả mọi người.

## Đánh giá tác động

Trước khi huấn luyện một mô hình machine learning, điều quan trọng là phải thực hiện đánh giá tác động để hiểu mục đích của hệ thống AI; mục đích sử dụng dự kiến; nơi nó sẽ được triển khai; và ai sẽ tương tác với hệ thống. Những điều này rất hữu ích cho người đánh giá hoặc kiểm tra hệ thống để biết các yếu tố cần xem xét khi xác định các rủi ro tiềm năng và hậu quả dự kiến.

Các lĩnh vực cần tập trung khi thực hiện đánh giá tác động bao gồm:

* **Tác động bất lợi đối với cá nhân**. Nhận thức về bất kỳ hạn chế hoặc yêu cầu nào, việc sử dụng không được hỗ trợ hoặc bất kỳ giới hạn nào đã biết cản trở hiệu suất của hệ thống là rất quan trọng để đảm bảo rằng hệ thống không được sử dụng theo cách có thể gây hại cho cá nhân.
* **Yêu cầu dữ liệu**. Hiểu cách và nơi hệ thống sẽ sử dụng dữ liệu cho phép người đánh giá khám phá bất kỳ yêu cầu dữ liệu nào cần lưu ý (ví dụ: các quy định về dữ liệu GDPR hoặc HIPPA). Ngoài ra, kiểm tra xem nguồn hoặc số lượng dữ liệu có đủ để huấn luyện hay không.
* **Tóm tắt tác động**. Thu thập danh sách các tác hại tiềm năng có thể phát sinh từ việc sử dụng hệ thống. Trong suốt vòng đời ML, xem xét liệu các vấn đề đã xác định có được giảm thiểu hoặc giải quyết hay không.
* **Mục tiêu áp dụng** cho từng nguyên tắc cốt lõi. Đánh giá xem các mục tiêu từ mỗi nguyên tắc có được đáp ứng hay không và liệu có bất kỳ khoảng trống nào.

## Gỡ lỗi với AI có trách nhiệm

Tương tự như việc gỡ lỗi một ứng dụng phần mềm, gỡ lỗi một hệ thống AI là một quá trình cần thiết để xác định và giải quyết các vấn đề trong hệ thống. Có nhiều yếu tố có thể ảnh hưởng đến việc một mô hình không hoạt động như mong đợi hoặc không có trách nhiệm. Hầu hết các chỉ số hiệu suất mô hình truyền thống là các tổng hợp định lượng về hiệu suất của mô hình, không đủ để phân tích cách một mô hình vi phạm các nguyên tắc AI có trách nhiệm. Hơn nữa, một mô hình machine learning là một hộp đen, khiến việc hiểu điều gì thúc đẩy kết quả của nó hoặc cung cấp lời giải thích khi nó mắc lỗi trở nên khó khăn. Sau này trong khóa học, chúng ta sẽ học cách sử dụng bảng điều khiển AI có trách nhiệm để giúp gỡ lỗi các hệ thống AI. Bảng điều khiển cung cấp một công cụ toàn diện cho các nhà khoa học dữ liệu và nhà phát triển AI để thực hiện:

* **Phân tích lỗi**. Để xác định phân bố lỗi của mô hình có thể ảnh hưởng đến sự công bằng hoặc độ tin cậy của hệ thống.
* **Tổng quan về mô hình**. Để khám phá nơi có sự chênh lệch trong hiệu suất của mô hình trên các nhóm dữ liệu.
* **Phân tích dữ liệu**. Để hiểu phân bố dữ liệu và xác định bất kỳ định kiến tiềm năng nào trong dữ liệu có thể dẫn đến các vấn đề về công bằng, tính toàn diện, và độ tin cậy.
* **Giải thích mô hình**. Để hiểu điều gì ảnh hưởng hoặc tác động đến các dự đoán của mô hình. Điều này giúp giải thích hành vi của mô hình, điều quan trọng đối với tính minh bạch và trách nhiệm.

## 🚀 Thử thách

Để ngăn chặn các tác hại được đưa vào ngay từ đầu, chúng ta nên:

- có sự đa dạng về nền tảng và quan điểm giữa những người làm việc trên các hệ thống
- đầu tư vào các tập dữ liệu phản ánh sự đa dạng của xã hội chúng ta
- phát triển các phương pháp tốt hơn trong suốt vòng đời machine learning để phát hiện và sửa chữa AI có trách nhiệm khi nó xảy ra

Hãy nghĩ về các tình huống thực tế nơi sự không đáng tin cậy của mô hình trở nên rõ ràng trong việc xây dựng và sử dụng mô hình. Chúng ta còn cần xem xét điều gì nữa?

## [Câu hỏi sau bài giảng](https://ff-quizzes.netlify.app/en/ml/)


Xem hội thảo này để tìm hiểu sâu hơn về các chủ đề:

- Theo đuổi AI có trách nhiệm: Đưa các nguyên tắc vào thực tiễn bởi Besmira Nushi, Mehrnoosh Sameki và Amit Sharma

[![Responsible AI Toolbox: Một khung nguồn mở để xây dựng AI có trách nhiệm](https://img.youtube.com/vi/tGgJCrA-MZU/0.jpg)](https://www.youtube.com/watch?v=tGgJCrA-MZU "RAI Toolbox: Một khung nguồn mở để xây dựng AI có trách nhiệm")

> 🎥 Nhấp vào hình ảnh trên để xem video: RAI Toolbox: Một khung nguồn mở để xây dựng AI có trách nhiệm bởi Besmira Nushi, Mehrnoosh Sameki và Amit Sharma

Ngoài ra, hãy đọc:

- Trung tâm tài nguyên RAI của Microsoft: [Responsible AI Resources – Microsoft AI](https://www.microsoft.com/ai/responsible-ai-resources?activetab=pivot1%3aprimaryr4)

- Nhóm nghiên cứu FATE của Microsoft: [FATE: Fairness, Accountability, Transparency, and Ethics in AI - Microsoft Research](https://www.microsoft.com/research/theme/fate/)

RAI Toolbox:

- [Kho lưu trữ GitHub của Responsible AI Toolbox](https://github.com/microsoft/responsible-ai-toolbox)

Tìm hiểu về các công cụ của Azure Machine Learning để đảm bảo tính công bằng:

- [Azure Machine Learning](https://docs.microsoft.com/azure/machine-learning/concept-fairness-ml?WT.mc_id=academic-77952-leestott)

## Bài tập

[Khám phá RAI Toolbox](assignment.md)

---

**Tuyên bố miễn trừ trách nhiệm**:  
Tài liệu này đã được dịch bằng dịch vụ dịch thuật AI [Co-op Translator](https://github.com/Azure/co-op-translator). Mặc dù chúng tôi cố gắng đảm bảo độ chính xác, xin lưu ý rằng các bản dịch tự động có thể chứa lỗi hoặc không chính xác. Tài liệu gốc bằng ngôn ngữ bản địa nên được coi là nguồn thông tin chính thức. Đối với các thông tin quan trọng, khuyến nghị sử dụng dịch vụ dịch thuật chuyên nghiệp bởi con người. Chúng tôi không chịu trách nhiệm cho bất kỳ sự hiểu lầm hoặc diễn giải sai nào phát sinh từ việc sử dụng bản dịch này.