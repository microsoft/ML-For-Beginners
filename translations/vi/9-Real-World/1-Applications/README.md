<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "83320d6b6994909e35d830cebf214039",
  "translation_date": "2025-09-05T19:22:37+00:00",
  "source_file": "9-Real-World/1-Applications/README.md",
  "language_code": "vi"
}
-->
# Tái bút: Học máy trong thế giới thực

![Tóm tắt về học máy trong thế giới thực qua sketchnote](../../../../sketchnotes/ml-realworld.png)
> Sketchnote bởi [Tomomi Imura](https://www.twitter.com/girlie_mac)

Trong chương trình học này, bạn đã học nhiều cách để chuẩn bị dữ liệu cho việc huấn luyện và tạo ra các mô hình học máy. Bạn đã xây dựng một loạt các mô hình kinh điển như hồi quy, phân cụm, phân loại, xử lý ngôn ngữ tự nhiên và chuỗi thời gian. Chúc mừng bạn! Bây giờ, bạn có thể đang tự hỏi tất cả những điều này để làm gì... ứng dụng thực tế của các mô hình này là gì?

Mặc dù AI, thường sử dụng học sâu, đã thu hút rất nhiều sự quan tâm trong ngành công nghiệp, nhưng các mô hình học máy cổ điển vẫn có những ứng dụng giá trị. Bạn thậm chí có thể đang sử dụng một số ứng dụng này ngay hôm nay! Trong bài học này, bạn sẽ khám phá cách tám ngành công nghiệp và lĩnh vực chuyên môn khác nhau sử dụng các loại mô hình này để làm cho ứng dụng của họ trở nên hiệu quả hơn, đáng tin cậy hơn, thông minh hơn và có giá trị hơn đối với người dùng.

## [Câu hỏi trước bài giảng](https://ff-quizzes.netlify.app/en/ml/)

## 💰 Tài chính

Ngành tài chính mang lại nhiều cơ hội cho học máy. Nhiều vấn đề trong lĩnh vực này có thể được mô hình hóa và giải quyết bằng cách sử dụng học máy.

### Phát hiện gian lận thẻ tín dụng

Chúng ta đã học về [phân cụm k-means](../../5-Clustering/2-K-Means/README.md) trước đó trong khóa học, nhưng làm thế nào nó có thể được sử dụng để giải quyết các vấn đề liên quan đến gian lận thẻ tín dụng?

Phân cụm k-means rất hữu ích trong một kỹ thuật phát hiện gian lận thẻ tín dụng gọi là **phát hiện điểm ngoại lai**. Các điểm ngoại lai, hoặc sự sai lệch trong các quan sát về một tập dữ liệu, có thể cho chúng ta biết liệu một thẻ tín dụng đang được sử dụng bình thường hay có điều gì bất thường đang xảy ra. Như được trình bày trong bài báo liên kết dưới đây, bạn có thể phân loại dữ liệu thẻ tín dụng bằng thuật toán phân cụm k-means và gán mỗi giao dịch vào một cụm dựa trên mức độ ngoại lai của nó. Sau đó, bạn có thể đánh giá các cụm rủi ro nhất để phân biệt giao dịch gian lận và hợp pháp.  
[Tham khảo](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.680.1195&rep=rep1&type=pdf)

### Quản lý tài sản

Trong quản lý tài sản, một cá nhân hoặc công ty xử lý các khoản đầu tư thay mặt cho khách hàng của họ. Công việc của họ là duy trì và tăng trưởng tài sản trong dài hạn, vì vậy việc chọn các khoản đầu tư có hiệu quả là rất quan trọng.

Một cách để đánh giá hiệu quả của một khoản đầu tư cụ thể là thông qua hồi quy thống kê. [Hồi quy tuyến tính](../../2-Regression/1-Tools/README.md) là một công cụ giá trị để hiểu cách một quỹ hoạt động so với một chuẩn mực nào đó. Chúng ta cũng có thể suy luận liệu kết quả của hồi quy có ý nghĩa thống kê hay không, hoặc mức độ ảnh hưởng của nó đến các khoản đầu tư của khách hàng. Bạn thậm chí có thể mở rộng phân tích của mình bằng cách sử dụng hồi quy đa biến, nơi các yếu tố rủi ro bổ sung có thể được tính đến. Để biết ví dụ về cách điều này hoạt động đối với một quỹ cụ thể, hãy xem bài báo dưới đây về việc đánh giá hiệu suất quỹ bằng hồi quy.  
[Tham khảo](http://www.brightwoodventures.com/evaluating-fund-performance-using-regression/)

## 🎓 Giáo dục

Ngành giáo dục cũng là một lĩnh vực rất thú vị nơi học máy có thể được áp dụng. Có những vấn đề thú vị cần giải quyết như phát hiện gian lận trong bài kiểm tra hoặc bài luận, hoặc quản lý sự thiên vị, dù vô tình hay không, trong quá trình chấm điểm.

### Dự đoán hành vi của học sinh

[Coursera](https://coursera.com), một nhà cung cấp khóa học trực tuyến mở, có một blog công nghệ tuyệt vời nơi họ thảo luận về nhiều quyết định kỹ thuật. Trong nghiên cứu trường hợp này, họ đã vẽ một đường hồi quy để cố gắng khám phá bất kỳ mối tương quan nào giữa điểm NPS (Net Promoter Score) thấp và việc giữ chân hoặc bỏ học khóa học.  
[Tham khảo](https://medium.com/coursera-engineering/controlled-regression-quantifying-the-impact-of-course-quality-on-learner-retention-31f956bd592a)

### Giảm thiểu sự thiên vị

[Grammarly](https://grammarly.com), một trợ lý viết giúp kiểm tra lỗi chính tả và ngữ pháp, sử dụng các [hệ thống xử lý ngôn ngữ tự nhiên](../../6-NLP/README.md) tinh vi trong các sản phẩm của mình. Họ đã xuất bản một nghiên cứu trường hợp thú vị trên blog công nghệ của mình về cách họ xử lý sự thiên vị giới tính trong học máy, điều mà bạn đã học trong [bài học về công bằng](../../1-Introduction/3-fairness/README.md).  
[Tham khảo](https://www.grammarly.com/blog/engineering/mitigating-gender-bias-in-autocorrect/)

## 👜 Bán lẻ

Ngành bán lẻ chắc chắn có thể hưởng lợi từ việc sử dụng học máy, từ việc tạo ra hành trình khách hàng tốt hơn đến việc quản lý hàng tồn kho một cách tối ưu.

### Cá nhân hóa hành trình khách hàng

Tại Wayfair, một công ty bán đồ gia dụng như nội thất, việc giúp khách hàng tìm thấy sản phẩm phù hợp với sở thích và nhu cầu của họ là điều tối quan trọng. Trong bài viết này, các kỹ sư của công ty mô tả cách họ sử dụng học máy và NLP để "hiển thị kết quả phù hợp cho khách hàng". Đáng chú ý, Công cụ Ý định Tìm kiếm của họ đã được xây dựng để sử dụng trích xuất thực thể, huấn luyện bộ phân loại, trích xuất tài sản và ý kiến, và gắn thẻ cảm xúc trên các đánh giá của khách hàng. Đây là một trường hợp sử dụng kinh điển của cách NLP hoạt động trong bán lẻ trực tuyến.  
[Tham khảo](https://www.aboutwayfair.com/tech-innovation/how-we-use-machine-learning-and-natural-language-processing-to-empower-search)

### Quản lý hàng tồn kho

Các công ty sáng tạo, linh hoạt như [StitchFix](https://stitchfix.com), một dịch vụ hộp gửi quần áo đến người tiêu dùng, dựa rất nhiều vào học máy để đưa ra gợi ý và quản lý hàng tồn kho. Các nhóm tạo kiểu của họ làm việc cùng với các nhóm hàng hóa của họ, thực tế: "một trong những nhà khoa học dữ liệu của chúng tôi đã thử nghiệm với một thuật toán di truyền và áp dụng nó vào lĩnh vực thời trang để dự đoán một món đồ quần áo thành công mà hiện tại chưa tồn tại. Chúng tôi đã mang điều đó đến nhóm hàng hóa và bây giờ họ có thể sử dụng nó như một công cụ."  
[Tham khảo](https://www.zdnet.com/article/how-stitch-fix-uses-machine-learning-to-master-the-science-of-styling/)

## 🏥 Chăm sóc sức khỏe

Ngành chăm sóc sức khỏe có thể tận dụng học máy để tối ưu hóa các nhiệm vụ nghiên cứu và các vấn đề hậu cần như tái nhập viện hoặc ngăn chặn sự lây lan của bệnh.

### Quản lý thử nghiệm lâm sàng

Độc tính trong các thử nghiệm lâm sàng là một mối quan tâm lớn đối với các nhà sản xuất thuốc. Bao nhiêu độc tính là có thể chấp nhận được? Trong nghiên cứu này, việc phân tích các phương pháp thử nghiệm lâm sàng khác nhau đã dẫn đến việc phát triển một cách tiếp cận mới để dự đoán khả năng kết quả của thử nghiệm lâm sàng. Cụ thể, họ đã sử dụng random forest để tạo ra một [bộ phân loại](../../4-Classification/README.md) có khả năng phân biệt giữa các nhóm thuốc.  
[Tham khảo](https://www.sciencedirect.com/science/article/pii/S2451945616302914)

### Quản lý tái nhập viện

Chăm sóc bệnh viện rất tốn kém, đặc biệt khi bệnh nhân phải tái nhập viện. Bài báo này thảo luận về một công ty sử dụng học máy để dự đoán khả năng tái nhập viện bằng cách sử dụng các thuật toán [phân cụm](../../5-Clustering/README.md). Các cụm này giúp các nhà phân tích "phát hiện các nhóm tái nhập viện có thể chia sẻ một nguyên nhân chung".  
[Tham khảo](https://healthmanagement.org/c/healthmanagement/issuearticle/hospital-readmissions-and-machine-learning)

### Quản lý bệnh dịch

Đại dịch gần đây đã làm nổi bật cách học máy có thể hỗ trợ ngăn chặn sự lây lan của bệnh. Trong bài viết này, bạn sẽ nhận ra việc sử dụng ARIMA, logistic curves, hồi quy tuyến tính và SARIMA. "Công việc này là một nỗ lực để tính toán tốc độ lây lan của virus này và do đó dự đoán số ca tử vong, hồi phục và ca nhiễm, để giúp chúng ta chuẩn bị tốt hơn và sống sót."  
[Tham khảo](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7979218/)

## 🌲 Sinh thái và Công nghệ xanh

Thiên nhiên và sinh thái bao gồm nhiều hệ thống nhạy cảm nơi sự tương tác giữa động vật và thiên nhiên được chú trọng. Việc đo lường chính xác các hệ thống này và hành động phù hợp nếu có điều gì xảy ra, như cháy rừng hoặc sự suy giảm số lượng động vật, là rất quan trọng.

### Quản lý rừng

Bạn đã học về [Học tăng cường](../../8-Reinforcement/README.md) trong các bài học trước. Nó có thể rất hữu ích khi cố gắng dự đoán các mô hình trong tự nhiên. Đặc biệt, nó có thể được sử dụng để theo dõi các vấn đề sinh thái như cháy rừng và sự lây lan của các loài xâm lấn. Ở Canada, một nhóm các nhà nghiên cứu đã sử dụng Học tăng cường để xây dựng các mô hình động lực cháy rừng từ hình ảnh vệ tinh. Sử dụng một quy trình "lan truyền không gian (SSP)" sáng tạo, họ hình dung một đám cháy rừng như "tác nhân tại bất kỳ ô nào trong cảnh quan." "Tập hợp các hành động mà đám cháy có thể thực hiện từ một vị trí tại bất kỳ thời điểm nào bao gồm lan truyền về phía bắc, nam, đông, hoặc tây hoặc không lan truyền.

Cách tiếp cận này đảo ngược thiết lập RL thông thường vì động lực của Quy trình Quyết định Markov (MDP) tương ứng là một hàm đã biết đối với sự lan truyền ngay lập tức của cháy rừng." Đọc thêm về các thuật toán kinh điển được nhóm này sử dụng tại liên kết dưới đây.  
[Tham khảo](https://www.frontiersin.org/articles/10.3389/fict.2018.00006/full)

### Cảm biến chuyển động của động vật

Mặc dù học sâu đã tạo ra một cuộc cách mạng trong việc theo dõi chuyển động của động vật bằng hình ảnh (bạn có thể tự xây dựng [trình theo dõi gấu Bắc Cực](https://docs.microsoft.com/learn/modules/build-ml-model-with-azure-stream-analytics/?WT.mc_id=academic-77952-leestott) tại đây), học máy cổ điển vẫn có chỗ đứng trong nhiệm vụ này.

Các cảm biến để theo dõi chuyển động của động vật nuôi và IoT sử dụng loại xử lý hình ảnh này, nhưng các kỹ thuật học máy cơ bản hơn lại hữu ích để tiền xử lý dữ liệu. Ví dụ, trong bài báo này, tư thế của cừu đã được giám sát và phân tích bằng các thuật toán phân loại khác nhau. Bạn có thể nhận ra đường cong ROC ở trang 335.  
[Tham khảo](https://druckhaus-hofmann.de/gallery/31-wj-feb-2020.pdf)

### ⚡️ Quản lý năng lượng

Trong các bài học về [dự báo chuỗi thời gian](../../7-TimeSeries/README.md), chúng ta đã đề cập đến khái niệm đồng hồ đỗ xe thông minh để tạo doanh thu cho một thị trấn dựa trên việc hiểu cung và cầu. Bài viết này thảo luận chi tiết cách phân cụm, hồi quy và dự báo chuỗi thời gian kết hợp để giúp dự đoán mức sử dụng năng lượng trong tương lai ở Ireland, dựa trên đồng hồ đo thông minh.  
[Tham khảo](https://www-cdn.knime.com/sites/default/files/inline-images/knime_bigdata_energy_timeseries_whitepaper.pdf)

## 💼 Bảo hiểm

Ngành bảo hiểm là một lĩnh vực khác sử dụng học máy để xây dựng và tối ưu hóa các mô hình tài chính và tính toán khả thi.

### Quản lý biến động

MetLife, một nhà cung cấp bảo hiểm nhân thọ, rất cởi mở về cách họ phân tích và giảm thiểu biến động trong các mô hình tài chính của mình. Trong bài viết này, bạn sẽ thấy các hình ảnh trực quan về phân loại nhị phân và thứ tự. Bạn cũng sẽ khám phá các hình ảnh trực quan về dự báo.  
[Tham khảo](https://investments.metlife.com/content/dam/metlifecom/us/investments/insights/research-topics/macro-strategy/pdf/MetLifeInvestmentManagement_MachineLearnedRanking_070920.pdf)

## 🎨 Nghệ thuật, Văn hóa và Văn học

Trong lĩnh vực nghệ thuật, ví dụ như báo chí, có nhiều vấn đề thú vị. Phát hiện tin giả là một vấn đề lớn vì nó đã được chứng minh là ảnh hưởng đến ý kiến của mọi người và thậm chí làm lung lay các nền dân chủ. Các bảo tàng cũng có thể hưởng lợi từ việc sử dụng học máy trong mọi thứ từ tìm kiếm liên kết giữa các hiện vật đến lập kế hoạch tài nguyên.

### Phát hiện tin giả

Phát hiện tin giả đã trở thành một trò chơi mèo vờn chuột trong truyền thông ngày nay. Trong bài viết này, các nhà nghiên cứu đề xuất rằng một hệ thống kết hợp một số kỹ thuật học máy mà chúng ta đã nghiên cứu có thể được thử nghiệm và mô hình tốt nhất được triển khai: "Hệ thống này dựa trên xử lý ngôn ngữ tự nhiên để trích xuất các đặc điểm từ dữ liệu và sau đó các đặc điểm này được sử dụng để huấn luyện các bộ phân loại học máy như Naive Bayes, Support Vector Machine (SVM), Random Forest (RF), Stochastic Gradient Descent (SGD), và Logistic Regression (LR)."  
[Tham khảo](https://www.irjet.net/archives/V7/i6/IRJET-V7I6688.pdf)

Bài viết này cho thấy cách kết hợp các lĩnh vực học máy khác nhau có thể tạo ra kết quả thú vị giúp ngăn chặn tin giả lan truyền và gây ra thiệt hại thực sự; trong trường hợp này, động lực là sự lan truyền tin đồn về các phương pháp điều trị COVID đã kích động bạo lực đám đông.

### Học máy trong bảo tàng

Các bảo tàng đang ở ngưỡng cửa của một cuộc cách mạng AI, nơi việc lập danh mục và số hóa các bộ sưu tập và tìm kiếm liên kết giữa các hiện vật trở nên dễ dàng hơn khi công nghệ tiến bộ. Các dự án như [In Codice Ratio](https://www.sciencedirect.com/science/article/abs/pii/S0306457321001035#:~:text=1.,studies%20over%20large%20historical%20sources.) đang giúp mở khóa những bí ẩn của các bộ sưu tập khó tiếp cận như Lưu trữ Vatican. Nhưng, khía cạnh kinh doanh của các bảo tàng cũng hưởng lợi từ các mô hình học máy.

Ví dụ, Viện Nghệ thuật Chicago đã xây dựng các mô hình để dự đoán những gì khán giả quan tâm và khi nào họ sẽ tham dự triển lãm. Mục tiêu là tạo ra trải nghiệm khách tham quan cá nhân hóa và tối ưu hóa mỗi lần người dùng ghé thăm bảo tàng. "Trong năm tài chính 2017, mô hình đã dự đoán số lượng khách tham quan và doanh thu vé với độ chính xác trong vòng 1%, theo Andrew Simnick, phó chủ tịch cấp cao tại Viện Nghệ thuật."  
[Tham khảo](https://www.chicagobusiness.com/article/20180518/ISSUE01/180519840/art-institute-of-chicago-uses-data-to-make-exhibit-choices)

## 🏷 Tiếp thị

### Phân khúc khách hàng

Các chiến lược tiếp thị hiệu quả nhất nhắm mục tiêu khách hàng theo những cách khác nhau dựa trên các nhóm khác nhau. Trong bài viết này, các ứng dụng của các thuật toán phân cụm được thảo luận để hỗ trợ tiếp thị phân biệt. Tiếp thị phân biệt giúp các công ty cải thiện nhận diện thương hiệu, tiếp cận nhiều khách hàng hơn và kiếm được nhiều tiền hơn.  
[Tham khảo](https://ai.inqline.com/machine-learning-for-marketing-customer-segmentation/)

## 🚀 Thử thách

Xác định một lĩnh vực khác hưởng lợi từ một số kỹ thuật bạn đã học trong chương trình học này và khám phá cách nó sử dụng học máy.
## [Câu hỏi kiểm tra sau bài giảng](https://ff-quizzes.netlify.app/en/ml/)

## Ôn tập & Tự học

Đội ngũ khoa học dữ liệu của Wayfair có một số video thú vị về cách họ sử dụng ML tại công ty của mình. Đáng để [xem qua](https://www.youtube.com/channel/UCe2PjkQXqOuwkW1gw6Ameuw/videos)!

## Bài tập

[Một cuộc săn tìm ML](assignment.md)

---

**Tuyên bố miễn trừ trách nhiệm**:  
Tài liệu này đã được dịch bằng dịch vụ dịch thuật AI [Co-op Translator](https://github.com/Azure/co-op-translator). Mặc dù chúng tôi cố gắng đảm bảo độ chính xác, xin lưu ý rằng các bản dịch tự động có thể chứa lỗi hoặc không chính xác. Tài liệu gốc bằng ngôn ngữ bản địa nên được coi là nguồn thông tin chính thức. Đối với các thông tin quan trọng, khuyến nghị sử dụng dịch vụ dịch thuật chuyên nghiệp từ con người. Chúng tôi không chịu trách nhiệm cho bất kỳ sự hiểu lầm hoặc diễn giải sai nào phát sinh từ việc sử dụng bản dịch này.