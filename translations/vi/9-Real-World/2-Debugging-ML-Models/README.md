# Tái bút: Gỡ lỗi mô hình trong Machine Learning bằng các thành phần của bảng điều khiển AI có trách nhiệm

## [Câu hỏi trước bài giảng](https://ff-quizzes.netlify.app/en/ml/)

## Giới thiệu

Machine learning ảnh hưởng đến cuộc sống hàng ngày của chúng ta. AI đang dần xuất hiện trong một số hệ thống quan trọng nhất ảnh hưởng đến chúng ta với tư cách cá nhân cũng như xã hội, từ chăm sóc sức khỏe, tài chính, giáo dục, đến việc làm. Ví dụ, các hệ thống và mô hình được sử dụng trong các nhiệm vụ ra quyết định hàng ngày như chẩn đoán y tế hoặc phát hiện gian lận. Do đó, sự tiến bộ của AI cùng với việc áp dụng nhanh chóng đang đối mặt với những kỳ vọng xã hội đang thay đổi và các quy định ngày càng tăng. Chúng ta thường thấy những trường hợp AI không đáp ứng được kỳ vọng, làm lộ ra những thách thức mới, và các chính phủ bắt đầu điều chỉnh các giải pháp AI. Vì vậy, điều quan trọng là các mô hình này phải được phân tích để đảm bảo kết quả công bằng, đáng tin cậy, bao trùm, minh bạch và có trách nhiệm cho tất cả mọi người.

Trong chương trình học này, chúng ta sẽ tìm hiểu các công cụ thực tiễn có thể được sử dụng để đánh giá xem một mô hình có vấn đề về AI có trách nhiệm hay không. Các kỹ thuật gỡ lỗi truyền thống trong machine learning thường dựa trên các tính toán định lượng như độ chính xác tổng hợp hoặc lỗi trung bình. Hãy tưởng tượng điều gì sẽ xảy ra khi dữ liệu bạn sử dụng để xây dựng các mô hình này thiếu một số nhóm nhân khẩu học, chẳng hạn như chủng tộc, giới tính, quan điểm chính trị, tôn giáo, hoặc đại diện không cân đối các nhóm này. Hoặc khi đầu ra của mô hình được diễn giải để ưu tiên một số nhóm nhân khẩu học. Điều này có thể dẫn đến sự đại diện quá mức hoặc thiếu mức của các nhóm đặc điểm nhạy cảm, gây ra các vấn đề về công bằng, bao trùm hoặc độ tin cậy từ mô hình. Một yếu tố khác là các mô hình machine learning thường được coi là "hộp đen", khiến việc hiểu và giải thích điều gì thúc đẩy dự đoán của mô hình trở nên khó khăn. Tất cả những điều này là thách thức mà các nhà khoa học dữ liệu và nhà phát triển AI phải đối mặt khi họ không có đủ công cụ để gỡ lỗi và đánh giá tính công bằng hoặc độ tin cậy của mô hình.

Trong bài học này, bạn sẽ học cách gỡ lỗi mô hình của mình bằng cách sử dụng:

- **Phân tích lỗi**: xác định nơi trong phân phối dữ liệu mà mô hình có tỷ lệ lỗi cao.
- **Tổng quan mô hình**: thực hiện phân tích so sánh giữa các nhóm dữ liệu khác nhau để khám phá sự chênh lệch trong các chỉ số hiệu suất của mô hình.
- **Phân tích dữ liệu**: điều tra nơi có thể xảy ra sự đại diện quá mức hoặc thiếu mức trong dữ liệu của bạn, điều này có thể làm lệch mô hình để ưu tiên một nhóm nhân khẩu học hơn nhóm khác.
- **Tầm quan trọng của đặc điểm**: hiểu các đặc điểm nào đang thúc đẩy dự đoán của mô hình ở cấp độ toàn cầu hoặc cấp độ cục bộ.

## Điều kiện tiên quyết

Trước khi bắt đầu, vui lòng xem lại [Các công cụ AI có trách nhiệm dành cho nhà phát triển](https://www.microsoft.com/ai/ai-lab-responsible-ai-dashboard)

> ![Gif về các công cụ AI có trách nhiệm](../../../../9-Real-World/2-Debugging-ML-Models/images/rai-overview.gif)

## Phân tích lỗi

Các chỉ số hiệu suất mô hình truyền thống được sử dụng để đo lường độ chính xác chủ yếu là các tính toán dựa trên dự đoán đúng và sai. Ví dụ, xác định rằng một mô hình chính xác 89% thời gian với lỗi mất mát là 0.001 có thể được coi là hiệu suất tốt. Tuy nhiên, lỗi thường không được phân phối đồng đều trong tập dữ liệu cơ bản của bạn. Bạn có thể đạt được điểm độ chính xác mô hình 89% nhưng phát hiện ra rằng có những vùng dữ liệu khác nhau mà mô hình thất bại 42% thời gian. Hậu quả của các mẫu lỗi này với một số nhóm dữ liệu nhất định có thể dẫn đến các vấn đề về công bằng hoặc độ tin cậy. Điều cần thiết là phải hiểu các khu vực mà mô hình hoạt động tốt hoặc không. Các vùng dữ liệu có số lượng lỗi cao trong mô hình của bạn có thể hóa ra là một nhóm nhân khẩu học quan trọng.

![Phân tích và gỡ lỗi lỗi mô hình](../../../../9-Real-World/2-Debugging-ML-Models/images/ea-error-distribution.png)

Thành phần Phân tích Lỗi trên bảng điều khiển RAI minh họa cách lỗi mô hình được phân phối qua các nhóm khác nhau bằng cách sử dụng hình ảnh cây. Điều này hữu ích trong việc xác định các đặc điểm hoặc khu vực có tỷ lệ lỗi cao trong tập dữ liệu của bạn. Bằng cách thấy nơi hầu hết các lỗi của mô hình xuất hiện, bạn có thể bắt đầu điều tra nguyên nhân gốc rễ. Bạn cũng có thể tạo các nhóm dữ liệu để thực hiện phân tích. Các nhóm dữ liệu này hỗ trợ trong quá trình gỡ lỗi để xác định lý do tại sao hiệu suất mô hình tốt ở một nhóm nhưng lại sai ở nhóm khác.

![Phân tích lỗi](../../../../9-Real-World/2-Debugging-ML-Models/images/ea-error-cohort.png)

Các chỉ báo trực quan trên bản đồ cây giúp xác định các khu vực vấn đề nhanh hơn. Ví dụ, màu đỏ đậm hơn của một nút cây cho thấy tỷ lệ lỗi cao hơn.

Bản đồ nhiệt là một chức năng hình ảnh khác mà người dùng có thể sử dụng để điều tra tỷ lệ lỗi bằng cách sử dụng một hoặc hai đặc điểm để tìm yếu tố góp phần vào lỗi mô hình trên toàn bộ tập dữ liệu hoặc các nhóm.

![Bản đồ nhiệt phân tích lỗi](../../../../9-Real-World/2-Debugging-ML-Models/images/ea-heatmap.png)

Sử dụng phân tích lỗi khi bạn cần:

* Hiểu sâu về cách lỗi mô hình được phân phối qua tập dữ liệu và qua nhiều đầu vào và đặc điểm.
* Phân tích các chỉ số hiệu suất tổng hợp để tự động khám phá các nhóm lỗi nhằm thông báo các bước giảm thiểu mục tiêu của bạn.

## Tổng quan mô hình

Đánh giá hiệu suất của một mô hình machine learning yêu cầu hiểu toàn diện về hành vi của nó. Điều này có thể đạt được bằng cách xem xét nhiều chỉ số như tỷ lệ lỗi, độ chính xác, recall, precision, hoặc MAE (Mean Absolute Error) để tìm sự chênh lệch giữa các chỉ số hiệu suất. Một chỉ số hiệu suất có thể trông tuyệt vời, nhưng các lỗi có thể được lộ ra ở một chỉ số khác. Ngoài ra, so sánh các chỉ số để tìm sự chênh lệch trên toàn bộ tập dữ liệu hoặc các nhóm giúp làm sáng tỏ nơi mô hình hoạt động tốt hoặc không. Điều này đặc biệt quan trọng trong việc xem hiệu suất của mô hình giữa các đặc điểm nhạy cảm và không nhạy cảm (ví dụ: chủng tộc, giới tính, hoặc tuổi của bệnh nhân) để phát hiện sự không công bằng tiềm ẩn mà mô hình có thể có. Ví dụ, phát hiện rằng mô hình có nhiều lỗi hơn ở một nhóm có các đặc điểm nhạy cảm có thể tiết lộ sự không công bằng tiềm ẩn.

Thành phần Tổng quan Mô hình của bảng điều khiển RAI không chỉ giúp phân tích các chỉ số hiệu suất của sự đại diện dữ liệu trong một nhóm, mà còn cung cấp cho người dùng khả năng so sánh hành vi của mô hình giữa các nhóm khác nhau.

![Nhóm dữ liệu - tổng quan mô hình trong bảng điều khiển RAI](../../../../9-Real-World/2-Debugging-ML-Models/images/model-overview-dataset-cohorts.png)

Chức năng phân tích dựa trên đặc điểm của thành phần này cho phép người dùng thu hẹp các nhóm dữ liệu con trong một đặc điểm cụ thể để xác định các bất thường ở mức độ chi tiết. Ví dụ, bảng điều khiển có trí tuệ tích hợp để tự động tạo các nhóm cho một đặc điểm do người dùng chọn (ví dụ: *"time_in_hospital < 3"* hoặc *"time_in_hospital >= 7"*). Điều này cho phép người dùng cô lập một đặc điểm cụ thể từ một nhóm dữ liệu lớn hơn để xem liệu nó có phải là yếu tố ảnh hưởng chính đến kết quả sai của mô hình hay không.

![Nhóm đặc điểm - tổng quan mô hình trong bảng điều khiển RAI](../../../../9-Real-World/2-Debugging-ML-Models/images/model-overview-feature-cohorts.png)

Thành phần Tổng quan Mô hình hỗ trợ hai loại chỉ số chênh lệch:

**Chênh lệch trong hiệu suất mô hình**: Các chỉ số này tính toán sự chênh lệch (khác biệt) trong giá trị của chỉ số hiệu suất được chọn giữa các nhóm dữ liệu. Một vài ví dụ:

* Chênh lệch trong tỷ lệ chính xác
* Chênh lệch trong tỷ lệ lỗi
* Chênh lệch trong precision
* Chênh lệch trong recall
* Chênh lệch trong lỗi tuyệt đối trung bình (MAE)

**Chênh lệch trong tỷ lệ lựa chọn**: Chỉ số này chứa sự khác biệt trong tỷ lệ lựa chọn (dự đoán thuận lợi) giữa các nhóm. Một ví dụ về điều này là chênh lệch trong tỷ lệ phê duyệt khoản vay. Tỷ lệ lựa chọn nghĩa là phần trăm điểm dữ liệu trong mỗi lớp được phân loại là 1 (trong phân loại nhị phân) hoặc phân phối giá trị dự đoán (trong hồi quy).

## Phân tích dữ liệu

> "Nếu bạn tra tấn dữ liệu đủ lâu, nó sẽ thú nhận bất cứ điều gì" - Ronald Coase

Câu nói này nghe có vẻ cực đoan, nhưng đúng là dữ liệu có thể bị thao túng để hỗ trợ bất kỳ kết luận nào. Sự thao túng này đôi khi xảy ra một cách vô tình. Là con người, chúng ta đều có thành kiến, và thường khó nhận thức được khi nào chúng ta đang đưa thành kiến vào dữ liệu. Đảm bảo tính công bằng trong AI và machine learning vẫn là một thách thức phức tạp.

Dữ liệu là một điểm mù lớn đối với các chỉ số hiệu suất mô hình truyền thống. Bạn có thể có điểm độ chính xác cao, nhưng điều này không phải lúc nào cũng phản ánh sự thiên vị dữ liệu cơ bản có thể tồn tại trong tập dữ liệu của bạn. Ví dụ, nếu một tập dữ liệu về nhân viên có 27% phụ nữ ở vị trí điều hành trong một công ty và 73% nam giới ở cùng cấp độ, một mô hình AI quảng cáo việc làm được đào tạo trên dữ liệu này có thể chủ yếu nhắm mục tiêu đến nam giới cho các vị trí cấp cao. Sự mất cân bằng này trong dữ liệu đã làm lệch dự đoán của mô hình để ưu tiên một giới tính. Điều này cho thấy vấn đề công bằng, nơi có sự thiên vị giới tính trong mô hình AI.

Thành phần Phân tích Dữ liệu trên bảng điều khiển RAI giúp xác định các khu vực có sự đại diện quá mức hoặc thiếu mức trong tập dữ liệu. Nó giúp người dùng chẩn đoán nguyên nhân gốc rễ của các lỗi và vấn đề công bằng được tạo ra từ sự mất cân bằng dữ liệu hoặc thiếu sự đại diện của một nhóm dữ liệu cụ thể. Điều này cung cấp cho người dùng khả năng hình dung tập dữ liệu dựa trên kết quả dự đoán và thực tế, nhóm lỗi, và các đặc điểm cụ thể. Đôi khi việc phát hiện một nhóm dữ liệu thiếu đại diện cũng có thể tiết lộ rằng mô hình không học tốt, dẫn đến tỷ lệ lỗi cao. Có một mô hình có sự thiên vị dữ liệu không chỉ là vấn đề công bằng mà còn cho thấy mô hình không bao trùm hoặc đáng tin cậy.

![Thành phần Phân tích Dữ liệu trên bảng điều khiển RAI](../../../../9-Real-World/2-Debugging-ML-Models/images/dataanalysis-cover.png)

Sử dụng phân tích dữ liệu khi bạn cần:

* Khám phá thống kê tập dữ liệu của bạn bằng cách chọn các bộ lọc khác nhau để phân chia dữ liệu của bạn thành các chiều khác nhau (còn được gọi là nhóm).
* Hiểu sự phân phối của tập dữ liệu của bạn qua các nhóm và nhóm đặc điểm khác nhau.
* Xác định liệu các phát hiện của bạn liên quan đến công bằng, phân tích lỗi, và nhân quả (được lấy từ các thành phần khác của bảng điều khiển) có phải là kết quả của sự phân phối tập dữ liệu của bạn hay không.
* Quyết định khu vực nào cần thu thập thêm dữ liệu để giảm thiểu các lỗi xuất phát từ vấn đề đại diện, nhiễu nhãn, nhiễu đặc điểm, thiên vị nhãn, và các yếu tố tương tự.

## Giải thích mô hình

Các mô hình machine learning thường được coi là "hộp đen". Hiểu các đặc điểm dữ liệu chính nào thúc đẩy dự đoán của mô hình có thể là một thách thức. Điều quan trọng là phải cung cấp sự minh bạch về lý do tại sao một mô hình đưa ra một dự đoán nhất định. Ví dụ, nếu một hệ thống AI dự đoán rằng một bệnh nhân tiểu đường có nguy cơ nhập viện lại trong vòng chưa đầy 30 ngày, nó nên cung cấp dữ liệu hỗ trợ dẫn đến dự đoán của nó. Có các chỉ số dữ liệu hỗ trợ mang lại sự minh bạch để giúp các bác sĩ hoặc bệnh viện đưa ra quyết định sáng suốt. Ngoài ra, khả năng giải thích lý do tại sao một mô hình đưa ra dự đoán cho một bệnh nhân cụ thể cho phép trách nhiệm với các quy định y tế. Khi bạn sử dụng các mô hình machine learning theo cách ảnh hưởng đến cuộc sống của con người, điều quan trọng là phải hiểu và giải thích điều gì ảnh hưởng đến hành vi của mô hình. Giải thích và hiểu mô hình giúp trả lời các câu hỏi trong các tình huống như:

* Gỡ lỗi mô hình: Tại sao mô hình của tôi lại mắc lỗi này? Làm thế nào tôi có thể cải thiện mô hình của mình?
* Hợp tác giữa con người và AI: Làm thế nào tôi có thể hiểu và tin tưởng các quyết định của mô hình?
* Tuân thủ quy định: Mô hình của tôi có đáp ứng các yêu cầu pháp lý không?

Thành phần Tầm quan trọng của Đặc điểm trên bảng điều khiển RAI giúp bạn gỡ lỗi và có cái nhìn toàn diện về cách một mô hình đưa ra dự đoán. Đây cũng là một công cụ hữu ích cho các chuyên gia machine learning và những người ra quyết định để giải thích và cung cấp bằng chứng về các đặc điểm ảnh hưởng đến hành vi của mô hình nhằm tuân thủ quy định. Tiếp theo, người dùng có thể khám phá cả giải thích toàn cầu và cục bộ để xác nhận các đặc điểm nào thúc đẩy dự đoán của mô hình. Giải thích toàn cầu liệt kê các đặc điểm hàng đầu ảnh hưởng đến dự đoán tổng thể của mô hình. Giải thích cục bộ hiển thị các đặc điểm dẫn đến dự đoán của mô hình cho một trường hợp cụ thể. Khả năng đánh giá các giải thích cục bộ cũng hữu ích trong việc gỡ lỗi hoặc kiểm tra một trường hợp cụ thể để hiểu rõ hơn và giải thích lý do tại sao mô hình đưa ra dự đoán chính xác hoặc không chính xác.

![Thành phần Tầm quan trọng của Đặc điểm trên bảng điều khiển RAI](../../../../9-Real-World/2-Debugging-ML-Models/images/9-feature-importance.png)

* Giải thích toàn cầu: Ví dụ, các đặc điểm nào ảnh hưởng đến hành vi tổng thể của mô hình nhập viện lại của bệnh nhân tiểu đường?
* Giải thích cục bộ: Ví dụ, tại sao một bệnh nhân tiểu đường trên 60 tuổi với các lần nhập viện trước đó lại được dự đoán sẽ nhập viện lại hoặc không nhập viện lại trong vòng 30 ngày?

Trong quá trình gỡ lỗi để kiểm tra hiệu suất của mô hình qua các nhóm khác nhau, Tầm quan trọng của Đặc điểm cho thấy mức độ ảnh hưởng của một đặc điểm qua các nhóm. Nó giúp tiết lộ các bất thường khi so sánh mức độ ảnh hưởng của đặc điểm trong việc thúc đẩy các dự đoán sai của mô hình. Thành phần Tầm quan trọng của Đặc điểm có thể hiển thị các giá trị trong một đặc điểm ảnh hưởng tích cực hoặc tiêu cực đến kết quả của mô hình. Ví dụ, nếu một mô hình đưa ra dự đoán không chính xác, thành phần này cung cấp khả năng khoan sâu và xác định đặc điểm hoặc giá trị đặc điểm nào đã thúc đẩy dự đoán. Mức độ chi tiết này không chỉ giúp trong việc gỡ lỗi mà còn cung cấp sự minh bạch và trách nhiệm trong các tình huống kiểm tra. Cuối cùng, thành phần này có thể giúp bạn xác định các vấn đề về công bằng. Để minh họa, nếu một đặc điểm nhạy cảm như dân tộc hoặc giới tính có ảnh hưởng lớn trong việc thúc đẩy dự đoán của mô hình, điều này có thể là dấu hiệu của sự thiên vị về chủng tộc hoặc giới tính trong mô hình.

![Tầm quan trọng của đặc điểm](../../../../9-Real-World/2-Debugging-ML-Models/images/9-features-influence.png)

Sử dụng khả năng giải thích khi bạn cần:

* Xác định mức độ đáng tin cậy của các dự đoán của hệ thống AI của bạn bằng cách hiểu các đặc điểm nào quan trọng nhất đối với các dự đoán.
* Tiếp cận việc gỡ lỗi mô hình của bạn bằng cách hiểu nó trước và xác định liệu mô hình có đang sử dụng các đặc điểm lành mạnh hay chỉ là các mối tương quan sai lầm.
* Phát hiện các nguồn gốc tiềm năng của sự không công bằng bằng cách hiểu liệu mô hình có đang dựa vào các đặc điểm nhạy cảm hoặc các đặc điểm có mối tương quan cao với chúng hay không.
* Xây dựng lòng tin của người dùng vào các quyết định của mô hình của bạn bằng cách tạo ra các giải thích cục bộ để minh họa kết quả của chúng.
* Hoàn thành kiểm tra quy định của một hệ thống AI để xác nhận các mô hình và giám sát tác động của các quyết định mô hình đối với con người.

## Kết luận

Tất cả các thành phần của bảng điều khiển RAI đều là các công cụ thực tiễn giúp bạn xây dựng các mô hình machine learning ít gây hại hơn và đáng tin cậy hơn đối với xã hội. Nó cải thiện việc ngăn chặn các mối đe dọa đối với quyền con người; phân biệt hoặc loại trừ một số nhóm khỏi các cơ hội sống; và giảm thiểu nguy cơ tổn thương thể chất hoặc tâm lý. Nó cũng giúp xây dựng lòng tin vào các quyết định của mô
- **Đại diện quá mức hoặc quá ít**. Ý tưởng ở đây là một nhóm nhất định không được nhìn thấy trong một nghề nghiệp nào đó, và bất kỳ dịch vụ hoặc chức năng nào tiếp tục thúc đẩy điều này đều góp phần gây hại.

### Bảng điều khiển Azure RAI

[Bảng điều khiển Azure RAI](https://learn.microsoft.com/en-us/azure/machine-learning/concept-responsible-ai-dashboard?WT.mc_id=aiml-90525-ruyakubu) được xây dựng dựa trên các công cụ mã nguồn mở do các tổ chức và học viện hàng đầu, bao gồm Microsoft, phát triển. Đây là công cụ quan trọng giúp các nhà khoa học dữ liệu và nhà phát triển AI hiểu rõ hơn về hành vi của mô hình, phát hiện và giảm thiểu các vấn đề không mong muốn từ các mô hình AI.

- Tìm hiểu cách sử dụng các thành phần khác nhau bằng cách xem tài liệu về bảng điều khiển RAI [docs.](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-responsible-ai-dashboard?WT.mc_id=aiml-90525-ruyakubu)

- Xem một số [notebook mẫu](https://github.com/Azure/RAI-vNext-Preview/tree/main/examples/notebooks) của bảng điều khiển RAI để gỡ lỗi các kịch bản AI có trách nhiệm hơn trong Azure Machine Learning.

---
## 🚀 Thử thách

Để ngăn chặn sự thiên vị thống kê hoặc dữ liệu ngay từ đầu, chúng ta nên:

- có sự đa dạng về nền tảng và quan điểm giữa những người làm việc trên các hệ thống
- đầu tư vào các tập dữ liệu phản ánh sự đa dạng của xã hội chúng ta
- phát triển các phương pháp tốt hơn để phát hiện và sửa chữa sự thiên vị khi nó xảy ra

Hãy suy nghĩ về các tình huống thực tế nơi sự không công bằng rõ ràng trong việc xây dựng và sử dụng mô hình. Chúng ta còn cần cân nhắc điều gì khác?

## [Câu hỏi sau bài giảng](https://ff-quizzes.netlify.app/en/ml/)
## Ôn tập & Tự học

Trong bài học này, bạn đã học một số công cụ thực tiễn để tích hợp AI có trách nhiệm vào học máy.

Xem hội thảo này để tìm hiểu sâu hơn về các chủ đề:

- Bảng điều khiển AI có trách nhiệm: Nền tảng toàn diện để thực hiện RAI trong thực tế bởi Besmira Nushi và Mehrnoosh Sameki

[![Bảng điều khiển AI có trách nhiệm: Nền tảng toàn diện để thực hiện RAI trong thực tế](https://img.youtube.com/vi/f1oaDNl3djg/0.jpg)](https://www.youtube.com/watch?v=f1oaDNl3djg "Bảng điều khiển AI có trách nhiệm: Nền tảng toàn diện để thực hiện RAI trong thực tế")

> 🎥 Nhấp vào hình ảnh trên để xem video: Bảng điều khiển AI có trách nhiệm: Nền tảng toàn diện để thực hiện RAI trong thực tế bởi Besmira Nushi và Mehrnoosh Sameki

Tham khảo các tài liệu sau để tìm hiểu thêm về AI có trách nhiệm và cách xây dựng các mô hình đáng tin cậy hơn:

- Công cụ bảng điều khiển RAI của Microsoft để gỡ lỗi mô hình ML: [Tài nguyên công cụ AI có trách nhiệm](https://aka.ms/rai-dashboard)

- Khám phá bộ công cụ AI có trách nhiệm: [Github](https://github.com/microsoft/responsible-ai-toolbox)

- Trung tâm tài nguyên RAI của Microsoft: [Tài nguyên AI có trách nhiệm – Microsoft AI](https://www.microsoft.com/en-us/ai/tools-practices)

- Nhóm nghiên cứu FATE của Microsoft: [FATE: Công bằng, Trách nhiệm, Minh bạch và Đạo đức trong AI - Microsoft Research](https://www.microsoft.com/research/theme/fate/)

## Bài tập

[Khám phá bảng điều khiển RAI](assignment.md)

---

**Tuyên bố miễn trừ trách nhiệm**:  
Tài liệu này đã được dịch bằng dịch vụ dịch thuật AI [Co-op Translator](https://github.com/Azure/co-op-translator). Mặc dù chúng tôi cố gắng đảm bảo độ chính xác, xin lưu ý rằng các bản dịch tự động có thể chứa lỗi hoặc không chính xác. Tài liệu gốc bằng ngôn ngữ bản địa nên được coi là nguồn thông tin chính thức. Đối với các thông tin quan trọng, nên sử dụng dịch vụ dịch thuật chuyên nghiệp từ con người. Chúng tôi không chịu trách nhiệm cho bất kỳ sự hiểu lầm hoặc diễn giải sai nào phát sinh từ việc sử dụng bản dịch này.