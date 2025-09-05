<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "8d32dadeda93c6fb5c43619854882ab1",
  "translation_date": "2025-09-05T20:30:33+00:00",
  "source_file": "6-NLP/4-Hotel-Reviews-1/README.md",
  "language_code": "vi"
}
-->
# Phân tích cảm xúc với đánh giá khách sạn - xử lý dữ liệu

Trong phần này, bạn sẽ sử dụng các kỹ thuật đã học ở các bài trước để thực hiện phân tích dữ liệu khám phá trên một tập dữ liệu lớn. Sau khi hiểu rõ về tính hữu ích của các cột khác nhau, bạn sẽ học:

- cách loại bỏ các cột không cần thiết
- cách tính toán dữ liệu mới dựa trên các cột hiện có
- cách lưu tập dữ liệu kết quả để sử dụng trong thử thách cuối cùng

## [Câu hỏi trước bài giảng](https://ff-quizzes.netlify.app/en/ml/)

### Giới thiệu

Cho đến nay, bạn đã học về cách dữ liệu văn bản khác biệt hoàn toàn so với dữ liệu dạng số. Nếu đó là văn bản được viết hoặc nói bởi con người, nó có thể được phân tích để tìm ra các mẫu, tần suất, cảm xúc và ý nghĩa. Bài học này sẽ đưa bạn vào một tập dữ liệu thực tế với một thử thách thực tế: **[515K Đánh giá Khách sạn ở Châu Âu](https://www.kaggle.com/jiashenliu/515k-hotel-reviews-data-in-europe)** và bao gồm giấy phép [CC0: Public Domain license](https://creativecommons.org/publicdomain/zero/1.0/). Tập dữ liệu này được thu thập từ Booking.com từ các nguồn công khai. Người tạo tập dữ liệu là Jiashen Liu.

### Chuẩn bị

Bạn sẽ cần:

* Khả năng chạy các notebook .ipynb bằng Python 3
* pandas
* NLTK, [cài đặt tại đây](https://www.nltk.org/install.html)
* Tập dữ liệu có sẵn trên Kaggle [515K Đánh giá Khách sạn ở Châu Âu](https://www.kaggle.com/jiashenliu/515k-hotel-reviews-data-in-europe). Tập dữ liệu này có dung lượng khoảng 230 MB sau khi giải nén. Tải xuống và lưu vào thư mục gốc `/data` liên quan đến các bài học NLP này.

## Phân tích dữ liệu khám phá

Thử thách này giả định rằng bạn đang xây dựng một bot gợi ý khách sạn sử dụng phân tích cảm xúc và điểm đánh giá của khách. Tập dữ liệu bạn sẽ sử dụng bao gồm các đánh giá của 1493 khách sạn khác nhau tại 6 thành phố.

Sử dụng Python, tập dữ liệu đánh giá khách sạn, và phân tích cảm xúc của NLTK, bạn có thể tìm ra:

* Những từ và cụm từ nào được sử dụng thường xuyên nhất trong các đánh giá?
* Các *thẻ* chính thức mô tả khách sạn có liên quan đến điểm đánh giá không (ví dụ: liệu các đánh giá tiêu cực hơn có xuất hiện nhiều hơn đối với một khách sạn dành cho *Gia đình có trẻ nhỏ* so với *Khách du lịch một mình*, có thể cho thấy khách sạn phù hợp hơn với *Khách du lịch một mình*)?
* Điểm cảm xúc của NLTK có 'đồng ý' với điểm số đánh giá của khách không?

#### Tập dữ liệu

Hãy khám phá tập dữ liệu mà bạn đã tải xuống và lưu cục bộ. Mở tệp trong một trình soạn thảo như VS Code hoặc thậm chí Excel.

Các tiêu đề trong tập dữ liệu như sau:

*Hotel_Address, Additional_Number_of_Scoring, Review_Date, Average_Score, Hotel_Name, Reviewer_Nationality, Negative_Review, Review_Total_Negative_Word_Counts, Total_Number_of_Reviews, Positive_Review, Review_Total_Positive_Word_Counts, Total_Number_of_Reviews_Reviewer_Has_Given, Reviewer_Score, Tags, days_since_review, lat, lng*

Dưới đây là cách nhóm các cột để dễ dàng kiểm tra hơn:
##### Các cột về khách sạn

* `Hotel_Name`, `Hotel_Address`, `lat` (vĩ độ), `lng` (kinh độ)
  * Sử dụng *lat* và *lng* bạn có thể vẽ bản đồ bằng Python hiển thị vị trí khách sạn (có thể mã hóa màu cho các đánh giá tiêu cực và tích cực)
  * Hotel_Address không rõ ràng là hữu ích với chúng ta, và có thể sẽ được thay thế bằng quốc gia để dễ dàng sắp xếp và tìm kiếm

**Các cột meta-review của khách sạn**

* `Average_Score`
  * Theo người tạo tập dữ liệu, cột này là *Điểm trung bình của khách sạn, được tính dựa trên nhận xét mới nhất trong năm qua*. Đây có vẻ là một cách tính điểm không bình thường, nhưng vì đây là dữ liệu được thu thập nên chúng ta có thể tạm chấp nhận.

  ✅ Dựa trên các cột khác trong dữ liệu này, bạn có thể nghĩ ra cách nào khác để tính điểm trung bình không?

* `Total_Number_of_Reviews`
  * Tổng số đánh giá mà khách sạn này đã nhận được - không rõ (nếu không viết mã) liệu điều này có đề cập đến các đánh giá trong tập dữ liệu hay không.
* `Additional_Number_of_Scoring`
  * Điều này có nghĩa là một điểm số đánh giá đã được đưa ra nhưng không có đánh giá tích cực hoặc tiêu cực nào được viết bởi người đánh giá.

**Các cột đánh giá**

- `Reviewer_Score`
  - Đây là giá trị số với tối đa 1 chữ số thập phân giữa giá trị tối thiểu và tối đa là 2.5 và 10
  - Không được giải thích tại sao 2.5 là điểm thấp nhất có thể
- `Negative_Review`
  - Nếu người đánh giá không viết gì, trường này sẽ có "**No Negative**"
  - Lưu ý rằng người đánh giá có thể viết một đánh giá tích cực trong cột Negative review (ví dụ: "không có gì xấu về khách sạn này")
- `Review_Total_Negative_Word_Counts`
  - Số lượng từ tiêu cực cao hơn cho thấy điểm số thấp hơn (mà không kiểm tra cảm xúc)
- `Positive_Review`
  - Nếu người đánh giá không viết gì, trường này sẽ có "**No Positive**"
  - Lưu ý rằng người đánh giá có thể viết một đánh giá tiêu cực trong cột Positive review (ví dụ: "không có gì tốt về khách sạn này cả")
- `Review_Total_Positive_Word_Counts`
  - Số lượng từ tích cực cao hơn cho thấy điểm số cao hơn (mà không kiểm tra cảm xúc)
- `Review_Date` và `days_since_review`
  - Có thể áp dụng một thước đo độ mới hoặc cũ cho một đánh giá (các đánh giá cũ có thể không chính xác bằng các đánh giá mới vì quản lý khách sạn đã thay đổi, hoặc đã được cải tạo, hoặc đã thêm một hồ bơi, v.v.)
- `Tags`
  - Đây là các mô tả ngắn mà người đánh giá có thể chọn để mô tả loại khách mà họ là (ví dụ: đi một mình hoặc gia đình), loại phòng họ đã ở, thời gian lưu trú và cách đánh giá được gửi.
  - Thật không may, việc sử dụng các thẻ này gặp vấn đề, hãy xem phần bên dưới thảo luận về tính hữu ích của chúng.

**Các cột về người đánh giá**

- `Total_Number_of_Reviews_Reviewer_Has_Given`
  - Điều này có thể là một yếu tố trong mô hình gợi ý, ví dụ, nếu bạn có thể xác định rằng những người đánh giá thường xuyên với hàng trăm đánh giá có xu hướng tiêu cực hơn là tích cực. Tuy nhiên, người đánh giá của bất kỳ đánh giá cụ thể nào không được xác định bằng một mã duy nhất, và do đó không thể liên kết với một tập hợp các đánh giá. Có 30 người đánh giá với 100 hoặc nhiều đánh giá hơn, nhưng khó thấy điều này có thể hỗ trợ mô hình gợi ý như thế nào.
- `Reviewer_Nationality`
  - Một số người có thể nghĩ rằng một số quốc tịch có xu hướng đưa ra đánh giá tích cực hoặc tiêu cực hơn vì một khuynh hướng quốc gia. Hãy cẩn thận khi xây dựng những quan điểm giai thoại như vậy vào các mô hình của bạn. Đây là những khuôn mẫu quốc gia (và đôi khi là chủng tộc), và mỗi người đánh giá là một cá nhân đã viết một đánh giá dựa trên trải nghiệm của họ. Nó có thể đã được lọc qua nhiều lăng kính như các lần lưu trú khách sạn trước đó, khoảng cách đã đi, và tính cách cá nhân của họ. Việc nghĩ rằng quốc tịch của họ là lý do cho điểm số đánh giá là khó biện minh.

##### Ví dụ

| Average  Score | Total Number   Reviews | Reviewer   Score | Negative <br />Review                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  | Positive   Review                 | Tags                                                                                      |
| -------------- | ---------------------- | ---------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------- | ----------------------------------------------------------------------------------------- |
| 7.8            | 1945                   | 2.5              | Đây hiện tại không phải là một khách sạn mà là một công trường xây dựng Tôi bị làm phiền từ sáng sớm và cả ngày với tiếng ồn xây dựng không thể chấp nhận được trong khi nghỉ ngơi sau một chuyến đi dài và làm việc trong phòng Người ta làm việc cả ngày với máy khoan trong các phòng liền kề Tôi yêu cầu đổi phòng nhưng không có phòng yên tĩnh nào có sẵn Tệ hơn nữa, tôi bị tính phí quá mức Tôi trả phòng vào buổi tối vì tôi phải rời đi chuyến bay rất sớm và nhận được hóa đơn phù hợp Một ngày sau khách sạn đã thực hiện một khoản phí khác mà không có sự đồng ý của tôi vượt quá giá đã đặt Đây là một nơi khủng khiếp Đừng tự làm khổ mình bằng cách đặt phòng ở đây | Không có gì Nơi khủng khiếp Tránh xa | Chuyến công tác Cặp đôi Phòng đôi tiêu chuẩn Lưu trú 2 đêm |

Như bạn có thể thấy, vị khách này đã không có một kỳ nghỉ vui vẻ tại khách sạn này. Khách sạn có điểm trung bình tốt là 7.8 và 1945 đánh giá, nhưng người đánh giá này đã cho điểm 2.5 và viết 115 từ về việc kỳ nghỉ của họ tiêu cực như thế nào. Nếu họ không viết gì trong cột Positive_Review, bạn có thể suy luận rằng không có gì tích cực, nhưng họ đã viết 7 từ cảnh báo. Nếu chúng ta chỉ đếm từ thay vì ý nghĩa, hoặc cảm xúc của các từ, chúng ta có thể có một cái nhìn sai lệch về ý định của người đánh giá. Lạ thay, điểm số 2.5 của họ gây nhầm lẫn, bởi vì nếu kỳ nghỉ tại khách sạn đó tệ như vậy, tại sao lại cho bất kỳ điểm nào? Khi điều tra tập dữ liệu kỹ lưỡng, bạn sẽ thấy rằng điểm thấp nhất có thể là 2.5, không phải 0. Điểm cao nhất có thể là 10.

##### Tags

Như đã đề cập ở trên, thoạt nhìn, ý tưởng sử dụng `Tags` để phân loại dữ liệu có vẻ hợp lý. Tuy nhiên, các thẻ này không được chuẩn hóa, điều này có nghĩa là trong một khách sạn, các tùy chọn có thể là *Phòng đơn*, *Phòng đôi*, và *Phòng đôi tiêu chuẩn*, nhưng ở khách sạn tiếp theo, chúng là *Phòng đơn cao cấp*, *Phòng Queen cổ điển*, và *Phòng King điều hành*. Đây có thể là cùng một loại phòng, nhưng có quá nhiều biến thể khiến lựa chọn trở thành:

1. Cố gắng thay đổi tất cả các thuật ngữ thành một tiêu chuẩn duy nhất, điều này rất khó khăn, vì không rõ đường dẫn chuyển đổi sẽ là gì trong mỗi trường hợp (ví dụ: *Phòng đơn cổ điển* ánh xạ tới *Phòng đơn* nhưng *Phòng Queen cao cấp với sân vườn hoặc tầm nhìn thành phố* khó ánh xạ hơn)

1. Chúng ta có thể áp dụng cách tiếp cận NLP và đo tần suất của các thuật ngữ nhất định như *Đi một mình*, *Khách công tác*, hoặc *Gia đình có trẻ nhỏ* khi chúng áp dụng cho mỗi khách sạn, và đưa yếu tố này vào mô hình gợi ý.

Các thẻ thường (nhưng không phải luôn luôn) là một trường duy nhất chứa danh sách 5 đến 6 giá trị được phân tách bằng dấu phẩy tương ứng với *Loại chuyến đi*, *Loại khách*, *Loại phòng*, *Số đêm*, và *Loại thiết bị đánh giá được gửi*. Tuy nhiên, vì một số người đánh giá không điền vào mỗi trường (họ có thể để trống một trường), các giá trị không phải lúc nào cũng theo cùng một thứ tự.

Ví dụ, hãy lấy *Loại nhóm*. Có 1025 khả năng duy nhất trong trường này trong cột `Tags`, và không may chỉ một số trong số đó đề cập đến nhóm (một số là loại phòng, v.v.). Nếu bạn lọc chỉ những giá trị đề cập đến gia đình, kết quả chứa nhiều loại *Phòng gia đình*. Nếu bạn bao gồm thuật ngữ *với*, tức là đếm các giá trị *Gia đình với*, kết quả sẽ tốt hơn, với hơn 80,000 trong số 515,000 kết quả chứa cụm từ "Gia đình với trẻ nhỏ" hoặc "Gia đình với trẻ lớn".

Điều này có nghĩa là cột thẻ không hoàn toàn vô dụng với chúng ta, nhưng sẽ cần một số công việc để làm cho nó hữu ích.

##### Điểm trung bình của khách sạn

Có một số điểm kỳ lạ hoặc không nhất quán với tập dữ liệu mà tôi không thể giải thích, nhưng được minh họa ở đây để bạn nhận thức được khi xây dựng các mô hình của mình. Nếu bạn tìm ra, hãy cho chúng tôi biết trong phần thảo luận!

Tập dữ liệu có các cột sau liên quan đến điểm trung bình và số lượng đánh giá:

1. Hotel_Name
2. Additional_Number_of_Scoring
3. Average_Score
4. Total_Number_of_Reviews
5. Reviewer_Score  

Khách sạn duy nhất có số lượng đánh giá nhiều nhất trong tập dữ liệu này là *Britannia International Hotel Canary Wharf* với 4789 đánh giá trong số 515,000. Nhưng nếu chúng ta xem giá trị `Total_Number_of_Reviews` cho khách sạn này, nó là 9086. Bạn có thể suy luận rằng có nhiều điểm số hơn mà không có đánh giá, vì vậy có lẽ chúng ta nên thêm giá trị cột `Additional_Number_of_Scoring`. Giá trị đó là 2682, và cộng nó với 4789 được 7471, vẫn thiếu 1615 so với `Total_Number_of_Reviews`.

Nếu bạn lấy cột `Average_Score`, bạn có thể suy luận rằng đó là điểm trung bình của các đánh giá trong tập dữ liệu, nhưng mô tả từ Kaggle là "*Điểm trung bình của khách sạn, được tính dựa trên nhận xét mới nhất trong năm qua*". Điều này có vẻ không hữu ích lắm, nhưng chúng ta có thể tự tính điểm trung bình dựa trên điểm số đánh giá trong tập dữ liệu. Sử dụng cùng một khách sạn làm ví dụ, điểm trung bình của khách sạn được đưa ra là 7.1 nhưng điểm số tính toán (điểm trung bình của người đánh giá *trong* tập dữ liệu) là 6.8. Điều này gần đúng, nhưng không phải là giá trị giống nhau, và chúng ta chỉ có thể đoán rằng các điểm số được đưa ra trong các đánh giá `Additional_Number_of_Scoring` đã tăng điểm trung bình lên 7.1. Thật không may, không có cách nào để kiểm tra hoặc chứng minh khẳng định đó, rất khó để sử dụng hoặc tin tưởng `Average_Score`, `Additional_Number_of_Scoring` và `Total_Number_of_Reviews` khi chúng dựa trên, hoặc đề cập đến, dữ liệu mà chúng ta không có.

Để làm phức tạp thêm, khách sạn có số lượng đánh giá cao thứ hai có điểm trung bình tính toán là 8.12 và điểm trung bình trong tập dữ liệu là 8.1. Điều này có phải là điểm số chính xác hay là sự trùng hợp ngẫu nhiên hoặc khách sạn đầu tiên là một sự không nhất quán?

Với khả năng rằng các khách sạn này có thể là một ngoại lệ, và có thể hầu hết các giá trị khớp nhau (nhưng một số không vì lý do nào đó), chúng ta sẽ viết một chương trình ngắn tiếp theo để khám phá các giá trị trong tập dữ liệu và xác định cách sử dụng đúng (hoặc không sử dụng) các giá trị.
> 🚨 Một lưu ý quan trọng  
>  
> Khi làm việc với bộ dữ liệu này, bạn sẽ viết mã để tính toán điều gì đó từ văn bản mà không cần phải đọc hoặc phân tích văn bản trực tiếp. Đây chính là cốt lõi của NLP, diễn giải ý nghĩa hoặc cảm xúc mà không cần con người thực hiện. Tuy nhiên, có khả năng bạn sẽ đọc một số đánh giá tiêu cực. Tôi khuyên bạn không nên làm vậy, vì bạn không cần phải làm thế. Một số đánh giá tiêu cực có thể ngớ ngẩn hoặc không liên quan, chẳng hạn như "Thời tiết không tốt", điều này nằm ngoài khả năng kiểm soát của khách sạn, hoặc thực tế là bất kỳ ai. Nhưng cũng có mặt tối trong một số đánh giá. Đôi khi các đánh giá tiêu cực mang tính phân biệt chủng tộc, giới tính, hoặc tuổi tác. Điều này thật đáng tiếc nhưng không thể tránh khỏi trong một bộ dữ liệu được thu thập từ một trang web công cộng. Một số người viết đánh giá mà bạn có thể thấy khó chịu, không thoải mái, hoặc gây tổn thương. Tốt hơn là để mã đo lường cảm xúc thay vì tự mình đọc chúng và cảm thấy khó chịu. Dù vậy, chỉ có một số ít người viết những điều như vậy, nhưng chúng vẫn tồn tại.
## Bài tập - Khám phá dữ liệu
### Tải dữ liệu

Đủ rồi việc kiểm tra dữ liệu bằng mắt, bây giờ bạn sẽ viết một số đoạn mã để tìm câu trả lời! Phần này sử dụng thư viện pandas. Nhiệm vụ đầu tiên của bạn là đảm bảo rằng bạn có thể tải và đọc dữ liệu CSV. Thư viện pandas có một trình tải CSV nhanh, và kết quả được đặt trong một dataframe, giống như trong các bài học trước. Tệp CSV mà chúng ta đang tải có hơn nửa triệu dòng, nhưng chỉ có 17 cột. Pandas cung cấp nhiều cách mạnh mẽ để tương tác với dataframe, bao gồm khả năng thực hiện các thao tác trên từng dòng.

Từ đây trở đi trong bài học này, sẽ có các đoạn mã và một số giải thích về mã cũng như thảo luận về ý nghĩa của kết quả. Sử dụng tệp _notebook.ipynb_ đi kèm để viết mã của bạn.

Hãy bắt đầu bằng cách tải tệp dữ liệu mà bạn sẽ sử dụng:

```python
# Load the hotel reviews from CSV
import pandas as pd
import time
# importing time so the start and end time can be used to calculate file loading time
print("Loading data file now, this could take a while depending on file size")
start = time.time()
# df is 'DataFrame' - make sure you downloaded the file to the data folder
df = pd.read_csv('../../data/Hotel_Reviews.csv')
end = time.time()
print("Loading took " + str(round(end - start, 2)) + " seconds")
```

Bây giờ dữ liệu đã được tải, chúng ta có thể thực hiện một số thao tác trên nó. Giữ đoạn mã này ở đầu chương trình của bạn cho phần tiếp theo.

## Khám phá dữ liệu

Trong trường hợp này, dữ liệu đã được *làm sạch*, nghĩa là nó đã sẵn sàng để làm việc và không có các ký tự trong ngôn ngữ khác có thể gây lỗi cho các thuật toán chỉ mong đợi ký tự tiếng Anh.

✅ Bạn có thể phải làm việc với dữ liệu yêu cầu một số xử lý ban đầu để định dạng trước khi áp dụng các kỹ thuật NLP, nhưng không phải lần này. Nếu bạn phải làm, bạn sẽ xử lý các ký tự không phải tiếng Anh như thế nào?

Dành một chút thời gian để đảm bảo rằng sau khi dữ liệu được tải, bạn có thể khám phá nó bằng mã. Rất dễ bị thu hút vào các cột `Negative_Review` và `Positive_Review`. Chúng chứa văn bản tự nhiên để các thuật toán NLP của bạn xử lý. Nhưng khoan đã! Trước khi bạn bắt đầu với NLP và phân tích cảm xúc, bạn nên làm theo đoạn mã dưới đây để xác định xem các giá trị được cung cấp trong tập dữ liệu có khớp với các giá trị bạn tính toán bằng pandas hay không.

## Các thao tác trên dataframe

Nhiệm vụ đầu tiên trong bài học này là kiểm tra xem các khẳng định sau có đúng không bằng cách viết một số đoạn mã để kiểm tra dataframe (mà không thay đổi nó).

> Giống như nhiều nhiệm vụ lập trình, có nhiều cách để hoàn thành, nhưng lời khuyên tốt là làm theo cách đơn giản và dễ dàng nhất, đặc biệt nếu nó sẽ dễ hiểu hơn khi bạn quay lại đoạn mã này trong tương lai. Với dataframe, có một API toàn diện thường sẽ có cách để làm điều bạn muốn một cách hiệu quả.

Hãy coi các câu hỏi sau như các nhiệm vụ lập trình và cố gắng trả lời chúng mà không nhìn vào giải pháp.

1. In ra *shape* của dataframe mà bạn vừa tải (shape là số dòng và cột).
2. Tính tần suất xuất hiện của quốc tịch người đánh giá:
   1. Có bao nhiêu giá trị khác nhau cho cột `Reviewer_Nationality` và chúng là gì?
   2. Quốc tịch người đánh giá nào phổ biến nhất trong tập dữ liệu (in tên quốc gia và số lượng đánh giá)?
   3. 10 quốc tịch phổ biến tiếp theo và tần suất xuất hiện của chúng là gì?
3. Khách sạn nào được đánh giá nhiều nhất bởi mỗi quốc tịch trong top 10 quốc tịch người đánh giá?
4. Có bao nhiêu đánh giá cho mỗi khách sạn (tần suất xuất hiện của khách sạn) trong tập dữ liệu?
5. Mặc dù có cột `Average_Score` cho mỗi khách sạn trong tập dữ liệu, bạn cũng có thể tính điểm trung bình (lấy trung bình tất cả điểm đánh giá của người đánh giá trong tập dữ liệu cho mỗi khách sạn). Thêm một cột mới vào dataframe của bạn với tiêu đề cột `Calc_Average_Score` chứa điểm trung bình đã tính toán.
6. Có khách sạn nào có giá trị `Average_Score` và `Calc_Average_Score` giống nhau (làm tròn đến 1 chữ số thập phân) không?
   1. Thử viết một hàm Python nhận một Series (dòng) làm tham số và so sánh các giá trị, in ra thông báo khi các giá trị không bằng nhau. Sau đó sử dụng phương thức `.apply()` để xử lý từng dòng với hàm này.
7. Tính và in ra có bao nhiêu dòng có giá trị cột `Negative_Review` là "No Negative".
8. Tính và in ra có bao nhiêu dòng có giá trị cột `Positive_Review` là "No Positive".
9. Tính và in ra có bao nhiêu dòng có giá trị cột `Positive_Review` là "No Positive" **và** giá trị cột `Negative_Review` là "No Negative".

### Đáp án bằng mã

1. In ra *shape* của dataframe mà bạn vừa tải (shape là số dòng và cột).

   ```python
   print("The shape of the data (rows, cols) is " + str(df.shape))
   > The shape of the data (rows, cols) is (515738, 17)
   ```

2. Tính tần suất xuất hiện của quốc tịch người đánh giá:

   1. Có bao nhiêu giá trị khác nhau cho cột `Reviewer_Nationality` và chúng là gì?
   2. Quốc tịch người đánh giá nào phổ biến nhất trong tập dữ liệu (in tên quốc gia và số lượng đánh giá)?

   ```python
   # value_counts() creates a Series object that has index and values in this case, the country and the frequency they occur in reviewer nationality
   nationality_freq = df["Reviewer_Nationality"].value_counts()
   print("There are " + str(nationality_freq.size) + " different nationalities")
   # print first and last rows of the Series. Change to nationality_freq.to_string() to print all of the data
   print(nationality_freq) 
   
   There are 227 different nationalities
    United Kingdom               245246
    United States of America      35437
    Australia                     21686
    Ireland                       14827
    United Arab Emirates          10235
                                  ...  
    Comoros                           1
    Palau                             1
    Northern Mariana Islands          1
    Cape Verde                        1
    Guinea                            1
   Name: Reviewer_Nationality, Length: 227, dtype: int64
   ```

   3. 10 quốc tịch phổ biến tiếp theo và tần suất xuất hiện của chúng là gì?

      ```python
      print("The highest frequency reviewer nationality is " + str(nationality_freq.index[0]).strip() + " with " + str(nationality_freq[0]) + " reviews.")
      # Notice there is a leading space on the values, strip() removes that for printing
      # What is the top 10 most common nationalities and their frequencies?
      print("The next 10 highest frequency reviewer nationalities are:")
      print(nationality_freq[1:11].to_string())
      
      The highest frequency reviewer nationality is United Kingdom with 245246 reviews.
      The next 10 highest frequency reviewer nationalities are:
       United States of America     35437
       Australia                    21686
       Ireland                      14827
       United Arab Emirates         10235
       Saudi Arabia                  8951
       Netherlands                   8772
       Switzerland                   8678
       Germany                       7941
       Canada                        7894
       France                        7296
      ```

3. Khách sạn nào được đánh giá nhiều nhất bởi mỗi quốc tịch trong top 10 quốc tịch người đánh giá?

   ```python
   # What was the most frequently reviewed hotel for the top 10 nationalities
   # Normally with pandas you will avoid an explicit loop, but wanted to show creating a new dataframe using criteria (don't do this with large amounts of data because it could be very slow)
   for nat in nationality_freq[:10].index:
      # First, extract all the rows that match the criteria into a new dataframe
      nat_df = df[df["Reviewer_Nationality"] == nat]   
      # Now get the hotel freq
      freq = nat_df["Hotel_Name"].value_counts()
      print("The most reviewed hotel for " + str(nat).strip() + " was " + str(freq.index[0]) + " with " + str(freq[0]) + " reviews.") 
      
   The most reviewed hotel for United Kingdom was Britannia International Hotel Canary Wharf with 3833 reviews.
   The most reviewed hotel for United States of America was Hotel Esther a with 423 reviews.
   The most reviewed hotel for Australia was Park Plaza Westminster Bridge London with 167 reviews.
   The most reviewed hotel for Ireland was Copthorne Tara Hotel London Kensington with 239 reviews.
   The most reviewed hotel for United Arab Emirates was Millennium Hotel London Knightsbridge with 129 reviews.
   The most reviewed hotel for Saudi Arabia was The Cumberland A Guoman Hotel with 142 reviews.
   The most reviewed hotel for Netherlands was Jaz Amsterdam with 97 reviews.
   The most reviewed hotel for Switzerland was Hotel Da Vinci with 97 reviews.
   The most reviewed hotel for Germany was Hotel Da Vinci with 86 reviews.
   The most reviewed hotel for Canada was St James Court A Taj Hotel London with 61 reviews.
   ```

4. Có bao nhiêu đánh giá cho mỗi khách sạn (tần suất xuất hiện của khách sạn) trong tập dữ liệu?

   ```python
   # First create a new dataframe based on the old one, removing the uneeded columns
   hotel_freq_df = df.drop(["Hotel_Address", "Additional_Number_of_Scoring", "Review_Date", "Average_Score", "Reviewer_Nationality", "Negative_Review", "Review_Total_Negative_Word_Counts", "Positive_Review", "Review_Total_Positive_Word_Counts", "Total_Number_of_Reviews_Reviewer_Has_Given", "Reviewer_Score", "Tags", "days_since_review", "lat", "lng"], axis = 1)
   
   # Group the rows by Hotel_Name, count them and put the result in a new column Total_Reviews_Found
   hotel_freq_df['Total_Reviews_Found'] = hotel_freq_df.groupby('Hotel_Name').transform('count')
   
   # Get rid of all the duplicated rows
   hotel_freq_df = hotel_freq_df.drop_duplicates(subset = ["Hotel_Name"])
   display(hotel_freq_df) 
   ```
   |                 Hotel_Name                 | Total_Number_of_Reviews | Total_Reviews_Found |
   | :----------------------------------------: | :---------------------: | :-----------------: |
   | Britannia International Hotel Canary Wharf |          9086           |        4789         |
   |    Park Plaza Westminster Bridge London    |          12158          |        4169         |
   |   Copthorne Tara Hotel London Kensington   |          7105           |        3578         |
   |                    ...                     |           ...           |         ...         |
   |       Mercure Paris Porte d Orleans        |           110           |         10          |
   |                Hotel Wagner                |           135           |         10          |
   |            Hotel Gallitzinberg             |           173           |          8          |
   
   Bạn có thể nhận thấy rằng kết quả *được đếm trong tập dữ liệu* không khớp với giá trị trong `Total_Number_of_Reviews`. Không rõ liệu giá trị này trong tập dữ liệu đại diện cho tổng số đánh giá mà khách sạn có, nhưng không phải tất cả đều được thu thập, hay một tính toán nào khác. `Total_Number_of_Reviews` không được sử dụng trong mô hình vì sự không rõ ràng này.

5. Mặc dù có cột `Average_Score` cho mỗi khách sạn trong tập dữ liệu, bạn cũng có thể tính điểm trung bình (lấy trung bình tất cả điểm đánh giá của người đánh giá trong tập dữ liệu cho mỗi khách sạn). Thêm một cột mới vào dataframe của bạn với tiêu đề cột `Calc_Average_Score` chứa điểm trung bình đã tính toán. In ra các cột `Hotel_Name`, `Average_Score`, và `Calc_Average_Score`.

   ```python
   # define a function that takes a row and performs some calculation with it
   def get_difference_review_avg(row):
     return row["Average_Score"] - row["Calc_Average_Score"]
   
   # 'mean' is mathematical word for 'average'
   df['Calc_Average_Score'] = round(df.groupby('Hotel_Name').Reviewer_Score.transform('mean'), 1)
   
   # Add a new column with the difference between the two average scores
   df["Average_Score_Difference"] = df.apply(get_difference_review_avg, axis = 1)
   
   # Create a df without all the duplicates of Hotel_Name (so only 1 row per hotel)
   review_scores_df = df.drop_duplicates(subset = ["Hotel_Name"])
   
   # Sort the dataframe to find the lowest and highest average score difference
   review_scores_df = review_scores_df.sort_values(by=["Average_Score_Difference"])
   
   display(review_scores_df[["Average_Score_Difference", "Average_Score", "Calc_Average_Score", "Hotel_Name"]])
   ```

   Bạn cũng có thể thắc mắc về giá trị `Average_Score` và tại sao đôi khi nó khác với điểm trung bình đã tính toán. Vì chúng ta không thể biết tại sao một số giá trị khớp, nhưng những giá trị khác lại có sự khác biệt, tốt nhất trong trường hợp này là sử dụng điểm đánh giá mà chúng ta có để tự tính toán điểm trung bình. Tuy nhiên, sự khác biệt thường rất nhỏ, đây là các khách sạn có độ lệch lớn nhất giữa điểm trung bình trong tập dữ liệu và điểm trung bình đã tính toán:

   | Average_Score_Difference | Average_Score | Calc_Average_Score |                                  Hotel_Name |
   | :----------------------: | :-----------: | :----------------: | ------------------------------------------: |
   |           -0.8           |      7.7      |        8.5         |                  Best Western Hotel Astoria |
   |           -0.7           |      8.8      |        9.5         | Hotel Stendhal Place Vend me Paris MGallery |
   |           -0.7           |      7.5      |        8.2         |               Mercure Paris Porte d Orleans |
   |           -0.7           |      7.9      |        8.6         |             Renaissance Paris Vendome Hotel |
   |           -0.5           |      7.0      |        7.5         |                         Hotel Royal Elys es |
   |           ...            |      ...      |        ...         |                                         ... |
   |           0.7            |      7.5      |        6.8         |     Mercure Paris Op ra Faubourg Montmartre |
   |           0.8            |      7.1      |        6.3         |      Holiday Inn Paris Montparnasse Pasteur |
   |           0.9            |      6.8      |        5.9         |                               Villa Eugenie |
   |           0.9            |      8.6      |        7.7         |   MARQUIS Faubourg St Honor Relais Ch teaux |
   |           1.3            |      7.2      |        5.9         |                          Kube Hotel Ice Bar |

   Với chỉ 1 khách sạn có sự khác biệt về điểm lớn hơn 1, điều này có nghĩa là chúng ta có thể bỏ qua sự khác biệt và sử dụng điểm trung bình đã tính toán.

6. Tính và in ra có bao nhiêu dòng có giá trị cột `Negative_Review` là "No Negative".

7. Tính và in ra có bao nhiêu dòng có giá trị cột `Positive_Review` là "No Positive".

8. Tính và in ra có bao nhiêu dòng có giá trị cột `Positive_Review` là "No Positive" **và** giá trị cột `Negative_Review` là "No Negative".

   ```python
   # with lambdas:
   start = time.time()
   no_negative_reviews = df.apply(lambda x: True if x['Negative_Review'] == "No Negative" else False , axis=1)
   print("Number of No Negative reviews: " + str(len(no_negative_reviews[no_negative_reviews == True].index)))
   
   no_positive_reviews = df.apply(lambda x: True if x['Positive_Review'] == "No Positive" else False , axis=1)
   print("Number of No Positive reviews: " + str(len(no_positive_reviews[no_positive_reviews == True].index)))
   
   both_no_reviews = df.apply(lambda x: True if x['Negative_Review'] == "No Negative" and x['Positive_Review'] == "No Positive" else False , axis=1)
   print("Number of both No Negative and No Positive reviews: " + str(len(both_no_reviews[both_no_reviews == True].index)))
   end = time.time()
   print("Lambdas took " + str(round(end - start, 2)) + " seconds")
   
   Number of No Negative reviews: 127890
   Number of No Positive reviews: 35946
   Number of both No Negative and No Positive reviews: 127
   Lambdas took 9.64 seconds
   ```

## Một cách khác

Một cách khác để đếm các mục mà không cần Lambdas, và sử dụng sum để đếm các dòng:

   ```python
   # without lambdas (using a mixture of notations to show you can use both)
   start = time.time()
   no_negative_reviews = sum(df.Negative_Review == "No Negative")
   print("Number of No Negative reviews: " + str(no_negative_reviews))
   
   no_positive_reviews = sum(df["Positive_Review"] == "No Positive")
   print("Number of No Positive reviews: " + str(no_positive_reviews))
   
   both_no_reviews = sum((df.Negative_Review == "No Negative") & (df.Positive_Review == "No Positive"))
   print("Number of both No Negative and No Positive reviews: " + str(both_no_reviews))
   
   end = time.time()
   print("Sum took " + str(round(end - start, 2)) + " seconds")
   
   Number of No Negative reviews: 127890
   Number of No Positive reviews: 35946
   Number of both No Negative and No Positive reviews: 127
   Sum took 0.19 seconds
   ```

   Bạn có thể nhận thấy rằng có 127 dòng có cả giá trị "No Negative" và "No Positive" cho các cột `Negative_Review` và `Positive_Review` tương ứng. Điều này có nghĩa là người đánh giá đã cho khách sạn một điểm số, nhưng từ chối viết cả đánh giá tích cực hoặc tiêu cực. May mắn thay, đây là một lượng nhỏ dòng (127 trong số 515738, hoặc 0.02%), vì vậy nó có lẽ sẽ không làm lệch mô hình hoặc kết quả của chúng ta theo bất kỳ hướng nào, nhưng bạn có thể không mong đợi một tập dữ liệu đánh giá lại có các dòng không có đánh giá, vì vậy việc khám phá dữ liệu để phát hiện các dòng như thế này là rất đáng giá.

Bây giờ bạn đã khám phá tập dữ liệu, trong bài học tiếp theo bạn sẽ lọc dữ liệu và thêm một số phân tích cảm xúc.

---
## 🚀Thử thách

Bài học này minh họa, như chúng ta đã thấy trong các bài học trước, tầm quan trọng cực kỳ của việc hiểu dữ liệu và những điểm bất thường của nó trước khi thực hiện các thao tác trên đó. Dữ liệu dựa trên văn bản, đặc biệt, cần được kiểm tra cẩn thận. Khám phá các tập dữ liệu nặng về văn bản khác nhau và xem liệu bạn có thể phát hiện ra các khu vực có thể giới thiệu sự thiên vị hoặc cảm xúc lệch lạc vào một mô hình.

## [Câu hỏi sau bài giảng](https://ff-quizzes.netlify.app/en/ml/)

## Ôn tập & Tự học

Tham gia [Lộ trình học về NLP này](https://docs.microsoft.com/learn/paths/explore-natural-language-processing/?WT.mc_id=academic-77952-leestott) để khám phá các công cụ thử nghiệm khi xây dựng các mô hình nặng về giọng nói và văn bản.

## Bài tập

[NLTK](assignment.md)

---

**Tuyên bố miễn trừ trách nhiệm**:  
Tài liệu này đã được dịch bằng dịch vụ dịch thuật AI [Co-op Translator](https://github.com/Azure/co-op-translator). Mặc dù chúng tôi cố gắng đảm bảo độ chính xác, xin lưu ý rằng các bản dịch tự động có thể chứa lỗi hoặc không chính xác. Tài liệu gốc bằng ngôn ngữ bản địa nên được coi là nguồn thông tin chính thức. Đối với các thông tin quan trọng, khuyến nghị sử dụng dịch vụ dịch thuật chuyên nghiệp bởi con người. Chúng tôi không chịu trách nhiệm cho bất kỳ sự hiểu lầm hoặc diễn giải sai nào phát sinh từ việc sử dụng bản dịch này.