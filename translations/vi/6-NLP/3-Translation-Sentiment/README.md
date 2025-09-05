<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "be03c8182982b87ced155e4e9d1438e8",
  "translation_date": "2025-09-05T20:39:35+00:00",
  "source_file": "6-NLP/3-Translation-Sentiment/README.md",
  "language_code": "vi"
}
-->
# Dịch thuật và phân tích cảm xúc với ML

Trong các bài học trước, bạn đã học cách xây dựng một bot cơ bản sử dụng `TextBlob`, một thư viện tích hợp ML phía sau để thực hiện các nhiệm vụ NLP cơ bản như trích xuất cụm danh từ. Một thách thức quan trọng khác trong ngôn ngữ học máy tính là việc dịch _chính xác_ một câu từ một ngôn ngữ nói hoặc viết sang một ngôn ngữ khác.

## [Câu hỏi trước bài giảng](https://ff-quizzes.netlify.app/en/ml/)

Dịch thuật là một vấn đề rất khó khăn do có hàng ngàn ngôn ngữ và mỗi ngôn ngữ có thể có các quy tắc ngữ pháp rất khác nhau. Một cách tiếp cận là chuyển đổi các quy tắc ngữ pháp chính thức của một ngôn ngữ, chẳng hạn như tiếng Anh, thành một cấu trúc không phụ thuộc vào ngôn ngữ, sau đó dịch bằng cách chuyển đổi lại sang ngôn ngữ khác. Cách tiếp cận này có nghĩa là bạn sẽ thực hiện các bước sau:

1. **Xác định**. Xác định hoặc gắn thẻ các từ trong ngôn ngữ đầu vào thành danh từ, động từ, v.v.
2. **Tạo bản dịch**. Tạo bản dịch trực tiếp của từng từ theo định dạng ngôn ngữ đích.

### Ví dụ câu, từ tiếng Anh sang tiếng Ireland

Trong tiếng 'Anh', câu _I feel happy_ gồm ba từ theo thứ tự:

- **chủ ngữ** (I)
- **động từ** (feel)
- **tính từ** (happy)

Tuy nhiên, trong ngôn ngữ 'Ireland', câu tương tự có cấu trúc ngữ pháp rất khác - cảm xúc như "*happy*" hoặc "*sad*" được diễn đạt như là *đang ở trên bạn*.

Cụm từ tiếng Anh `I feel happy` trong tiếng Ireland sẽ là `Tá athas orm`. Một bản dịch *theo nghĩa đen* sẽ là `Happy is upon me`.

Một người nói tiếng Ireland dịch sang tiếng Anh sẽ nói `I feel happy`, không phải `Happy is upon me`, bởi vì họ hiểu ý nghĩa của câu, ngay cả khi từ ngữ và cấu trúc câu khác nhau.

Thứ tự chính thức của câu trong tiếng Ireland là:

- **động từ** (Tá hoặc is)
- **tính từ** (athas, hoặc happy)
- **chủ ngữ** (orm, hoặc upon me)

## Dịch thuật

Một chương trình dịch thuật đơn giản có thể chỉ dịch từ mà bỏ qua cấu trúc câu.

✅ Nếu bạn đã học một ngôn ngữ thứ hai (hoặc thứ ba hoặc nhiều hơn) khi trưởng thành, bạn có thể đã bắt đầu bằng cách suy nghĩ bằng ngôn ngữ mẹ đẻ, dịch một khái niệm từng từ trong đầu sang ngôn ngữ thứ hai, và sau đó nói ra bản dịch của mình. Điều này tương tự như những gì các chương trình dịch thuật máy tính đơn giản đang làm. Điều quan trọng là phải vượt qua giai đoạn này để đạt được sự lưu loát!

Dịch thuật đơn giản dẫn đến các bản dịch sai (và đôi khi hài hước): `I feel happy` dịch theo nghĩa đen thành `Mise bhraitheann athas` trong tiếng Ireland. Điều đó có nghĩa (theo nghĩa đen) là `me feel happy` và không phải là một câu hợp lệ trong tiếng Ireland. Mặc dù tiếng Anh và tiếng Ireland là các ngôn ngữ được nói trên hai hòn đảo láng giềng gần nhau, chúng là những ngôn ngữ rất khác nhau với cấu trúc ngữ pháp khác nhau.

> Bạn có thể xem một số video về truyền thống ngôn ngữ Ireland như [video này](https://www.youtube.com/watch?v=mRIaLSdRMMs)

### Các phương pháp học máy

Cho đến nay, bạn đã học về cách tiếp cận quy tắc chính thức đối với xử lý ngôn ngữ tự nhiên. Một cách tiếp cận khác là bỏ qua ý nghĩa của các từ, và _thay vào đó sử dụng học máy để phát hiện các mẫu_. Điều này có thể hoạt động trong dịch thuật nếu bạn có nhiều văn bản (một *corpus*) hoặc các văn bản (*corpora*) trong cả ngôn ngữ gốc và ngôn ngữ đích.

Ví dụ, hãy xem xét trường hợp của *Pride and Prejudice*, một tiểu thuyết tiếng Anh nổi tiếng được viết bởi Jane Austen vào năm 1813. Nếu bạn tham khảo cuốn sách bằng tiếng Anh và một bản dịch của con người sang tiếng *Pháp*, bạn có thể phát hiện các cụm từ trong một ngôn ngữ được dịch _theo cách diễn đạt_ sang ngôn ngữ kia. Bạn sẽ làm điều đó trong một phút nữa.

Ví dụ, khi một cụm từ tiếng Anh như `I have no money` được dịch theo nghĩa đen sang tiếng Pháp, nó có thể trở thành `Je n'ai pas de monnaie`. "Monnaie" là một từ tiếng Pháp dễ gây nhầm lẫn, vì 'money' và 'monnaie' không đồng nghĩa. Một bản dịch tốt hơn mà một người nói tiếng Pháp có thể làm sẽ là `Je n'ai pas d'argent`, vì nó truyền tải ý nghĩa rằng bạn không có tiền (thay vì 'tiền lẻ' là ý nghĩa của 'monnaie').

![monnaie](../../../../6-NLP/3-Translation-Sentiment/images/monnaie.png)

> Hình ảnh bởi [Jen Looper](https://twitter.com/jenlooper)

Nếu một mô hình ML có đủ các bản dịch của con người để xây dựng một mô hình, nó có thể cải thiện độ chính xác của các bản dịch bằng cách xác định các mẫu phổ biến trong các văn bản đã được dịch trước đó bởi các chuyên gia nói cả hai ngôn ngữ.

### Bài tập - dịch thuật

Bạn có thể sử dụng `TextBlob` để dịch các câu. Hãy thử câu nổi tiếng đầu tiên của **Pride and Prejudice**:

```python
from textblob import TextBlob

blob = TextBlob(
    "It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want of a wife!"
)
print(blob.translate(to="fr"))

```

`TextBlob` làm khá tốt việc dịch: "C'est une vérité universellement reconnue, qu'un homme célibataire en possession d'une bonne fortune doit avoir besoin d'une femme!". 

Có thể cho rằng bản dịch của TextBlob thực sự chính xác hơn so với bản dịch tiếng Pháp năm 1932 của cuốn sách bởi V. Leconte và Ch. Pressoir:

"C'est une vérité universelle qu'un célibataire pourvu d'une belle fortune doit avoir envie de se marier, et, si peu que l'on sache de son sentiment à cet egard, lorsqu'il arrive dans une nouvelle résidence, cette idée est si bien fixée dans l'esprit de ses voisins qu'ils le considèrent sur-le-champ comme la propriété légitime de l'une ou l'autre de leurs filles."

Trong trường hợp này, bản dịch được hỗ trợ bởi ML làm tốt hơn so với người dịch, người đã thêm các từ không cần thiết vào lời của tác giả gốc để làm rõ.

> Điều gì đang xảy ra ở đây? Và tại sao TextBlob lại tốt trong việc dịch thuật? Thực tế, phía sau nó đang sử dụng Google Translate, một AI tinh vi có khả năng phân tích hàng triệu cụm từ để dự đoán các chuỗi tốt nhất cho nhiệm vụ. Không có gì thủ công diễn ra ở đây và bạn cần kết nối internet để sử dụng `blob.translate`.

✅ Hãy thử một số câu khác. Cái nào tốt hơn, ML hay bản dịch của con người? Trong những trường hợp nào?

## Phân tích cảm xúc

Một lĩnh vực khác mà học máy có thể hoạt động rất tốt là phân tích cảm xúc. Một cách tiếp cận không sử dụng ML để phân tích cảm xúc là xác định các từ và cụm từ 'tích cực' và 'tiêu cực'. Sau đó, với một đoạn văn bản mới, tính toán tổng giá trị của các từ tích cực, tiêu cực và trung lập để xác định cảm xúc tổng thể.

Cách tiếp cận này dễ bị đánh lừa như bạn có thể đã thấy trong nhiệm vụ Marvin - câu `Great, that was a wonderful waste of time, I'm glad we are lost on this dark road` là một câu cảm xúc tiêu cực mang tính châm biếm, nhưng thuật toán đơn giản phát hiện 'great', 'wonderful', 'glad' là tích cực và 'waste', 'lost' và 'dark' là tiêu cực. Cảm xúc tổng thể bị ảnh hưởng bởi những từ mâu thuẫn này.

✅ Dừng lại một chút và nghĩ về cách chúng ta truyền tải sự châm biếm khi nói. Ngữ điệu đóng vai trò lớn. Hãy thử nói câu "Well, that film was awesome" theo nhiều cách khác nhau để khám phá cách giọng nói của bạn truyền tải ý nghĩa.

### Các phương pháp ML

Cách tiếp cận ML sẽ là thu thập thủ công các đoạn văn bản tiêu cực và tích cực - tweet, hoặc đánh giá phim, hoặc bất cứ điều gì mà con người đã đưa ra điểm số *và* ý kiến bằng văn bản. Sau đó, các kỹ thuật NLP có thể được áp dụng để phân tích ý kiến và điểm số, để các mẫu xuất hiện (ví dụ: các đánh giá phim tích cực có xu hướng chứa cụm từ 'Oscar worthy' nhiều hơn các đánh giá phim tiêu cực, hoặc các đánh giá nhà hàng tích cực nói 'gourmet' nhiều hơn 'disgusting').

> ⚖️ **Ví dụ**: Nếu bạn làm việc trong văn phòng của một chính trị gia và có một luật mới đang được tranh luận, các cử tri có thể viết email đến văn phòng để ủng hộ hoặc phản đối luật mới đó. Giả sử bạn được giao nhiệm vụ đọc email và phân loại chúng thành 2 nhóm, *ủng hộ* và *phản đối*. Nếu có rất nhiều email, bạn có thể bị quá tải khi cố gắng đọc tất cả. Sẽ thật tuyệt nếu một bot có thể đọc tất cả cho bạn, hiểu chúng và cho bạn biết mỗi email thuộc nhóm nào? 
> 
> Một cách để đạt được điều đó là sử dụng Học Máy. Bạn sẽ huấn luyện mô hình với một phần email *phản đối* và một phần email *ủng hộ*. Mô hình sẽ có xu hướng liên kết các cụm từ và từ với nhóm phản đối và nhóm ủng hộ, *nhưng nó sẽ không hiểu bất kỳ nội dung nào*, chỉ là các từ và mẫu nhất định có khả năng xuất hiện nhiều hơn trong email *phản đối* hoặc *ủng hộ*. Bạn có thể kiểm tra nó với một số email mà bạn chưa sử dụng để huấn luyện mô hình, và xem liệu nó có đưa ra kết luận giống như bạn không. Sau đó, khi bạn hài lòng với độ chính xác của mô hình, bạn có thể xử lý các email trong tương lai mà không cần phải đọc từng cái.

✅ Quy trình này có giống với các quy trình bạn đã sử dụng trong các bài học trước không?

## Bài tập - các câu cảm xúc

Cảm xúc được đo bằng *độ phân cực* từ -1 đến 1, nghĩa là -1 là cảm xúc tiêu cực nhất, và 1 là cảm xúc tích cực nhất. Cảm xúc cũng được đo bằng điểm từ 0 - 1 cho tính khách quan (0) và tính chủ quan (1).

Hãy xem lại *Pride and Prejudice* của Jane Austen. Văn bản có sẵn tại [Project Gutenberg](https://www.gutenberg.org/files/1342/1342-h/1342-h.htm). Mẫu dưới đây hiển thị một chương trình ngắn phân tích cảm xúc của câu đầu tiên và câu cuối cùng từ cuốn sách và hiển thị độ phân cực cảm xúc và điểm số khách quan/chủ quan của nó.

Bạn nên sử dụng thư viện `TextBlob` (được mô tả ở trên) để xác định `sentiment` (bạn không cần phải tự viết trình tính toán cảm xúc) trong nhiệm vụ sau.

```python
from textblob import TextBlob

quote1 = """It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want of a wife."""

quote2 = """Darcy, as well as Elizabeth, really loved them; and they were both ever sensible of the warmest gratitude towards the persons who, by bringing her into Derbyshire, had been the means of uniting them."""

sentiment1 = TextBlob(quote1).sentiment
sentiment2 = TextBlob(quote2).sentiment

print(quote1 + " has a sentiment of " + str(sentiment1))
print(quote2 + " has a sentiment of " + str(sentiment2))
```

Bạn sẽ thấy đầu ra sau:

```output
It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want # of a wife. has a sentiment of Sentiment(polarity=0.20952380952380953, subjectivity=0.27142857142857146)

Darcy, as well as Elizabeth, really loved them; and they were
     both ever sensible of the warmest gratitude towards the persons
      who, by bringing her into Derbyshire, had been the means of
      uniting them. has a sentiment of Sentiment(polarity=0.7, subjectivity=0.8)
```

## Thử thách - kiểm tra độ phân cực cảm xúc

Nhiệm vụ của bạn là xác định, sử dụng độ phân cực cảm xúc, liệu *Pride and Prejudice* có nhiều câu hoàn toàn tích cực hơn câu hoàn toàn tiêu cực hay không. Đối với nhiệm vụ này, bạn có thể giả định rằng điểm độ phân cực là 1 hoặc -1 tương ứng với cảm xúc hoàn toàn tích cực hoặc tiêu cực.

**Các bước:**

1. Tải xuống một [bản sao của Pride and Prejudice](https://www.gutenberg.org/files/1342/1342-h/1342-h.htm) từ Project Gutenberg dưới dạng tệp .txt. Loại bỏ siêu dữ liệu ở đầu và cuối tệp, chỉ để lại văn bản gốc
2. Mở tệp trong Python và trích xuất nội dung dưới dạng chuỗi
3. Tạo một TextBlob bằng chuỗi của cuốn sách
4. Phân tích từng câu trong cuốn sách trong một vòng lặp
   1. Nếu độ phân cực là 1 hoặc -1, lưu câu vào một mảng hoặc danh sách các thông điệp tích cực hoặc tiêu cực
5. Cuối cùng, in ra tất cả các câu tích cực và tiêu cực (riêng biệt) và số lượng của mỗi loại.

Đây là một [giải pháp mẫu](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/3-Translation-Sentiment/solution/notebook.ipynb).

✅ Kiểm tra kiến thức

1. Cảm xúc dựa trên các từ được sử dụng trong câu, nhưng liệu mã có *hiểu* các từ không?
2. Bạn có nghĩ rằng độ phân cực cảm xúc là chính xác không, hay nói cách khác, bạn có *đồng ý* với các điểm số không?
   1. Đặc biệt, bạn có đồng ý hay không đồng ý với độ phân cực **tích cực** tuyệt đối của các câu sau đây?
      * “What an excellent father you have, girls!” said she, when the door was shut.
      * “Your examination of Mr. Darcy is over, I presume,” said Miss Bingley; “and pray what is the result?” “I am perfectly convinced by it that Mr. Darcy has no defect.
      * How wonderfully these sort of things occur!
      * I have the greatest dislike in the world to that sort of thing.
      * Charlotte is an excellent manager, I dare say.
      * “This is delightful indeed!
      * I am so happy!
      * Your idea of the ponies is delightful.
   2. Ba câu tiếp theo được đánh giá với cảm xúc tích cực tuyệt đối, nhưng khi đọc kỹ, chúng không phải là câu tích cực. Tại sao phân tích cảm xúc lại nghĩ rằng chúng là câu tích cực?
      * Happy shall I be, when his stay at Netherfield is over!” “I wish I could say anything to comfort you,” replied Elizabeth; “but it is wholly out of my power.
      * If I could but see you as happy!
      * Our distress, my dear Lizzy, is very great.
   3. Bạn có đồng ý hay không đồng ý với độ phân cực **tiêu cực** tuyệt đối của các câu sau đây?
      - Everybody is disgusted with his pride.
      - “I should like to know how he behaves among strangers.” “You shall hear then—but prepare yourself for something very dreadful.
      - The pause was to Elizabeth’s feelings dreadful.
      - It would be dreadful!

✅ Bất kỳ người yêu thích Jane Austen nào cũng sẽ hiểu rằng bà thường sử dụng các cuốn sách của mình để phê phán các khía cạnh lố bịch hơn của xã hội Anh thời Regency. Elizabeth Bennett, nhân vật chính trong *Pride and Prejudice*, là một nhà quan sát xã hội sắc sảo (như tác giả) và ngôn ngữ của cô thường rất tinh tế. Ngay cả Mr. Darcy (người yêu trong câu chuyện) cũng nhận xét về cách sử dụng ngôn ngữ vui tươi và trêu chọc của Elizabeth: "Tôi đã có niềm vui được quen biết bạn đủ lâu để biết rằng bạn rất thích thú khi thỉnh thoảng bày tỏ những ý kiến mà thực tế không phải của bạn."

---

## 🚀Thử thách

Bạn có thể làm cho Marvin tốt hơn bằng cách trích xuất các đặc điểm khác từ đầu vào của người dùng không?

## [Câu hỏi sau bài giảng](https://ff-quizzes.netlify.app/en/ml/)

## Ôn tập & Tự học
Có nhiều cách để trích xuất cảm xúc từ văn bản. Hãy nghĩ về các ứng dụng kinh doanh có thể sử dụng kỹ thuật này. Hãy nghĩ về cách nó có thể gặp sai sót. Đọc thêm về các hệ thống phân tích cảm xúc tiên tiến, sẵn sàng cho doanh nghiệp như [Azure Text Analysis](https://docs.microsoft.com/azure/cognitive-services/Text-Analytics/how-tos/text-analytics-how-to-sentiment-analysis?tabs=version-3-1?WT.mc_id=academic-77952-leestott). Thử nghiệm một số câu từ Pride and Prejudice ở trên và xem liệu nó có thể phát hiện được sắc thái hay không.

## Bài tập

[Giấy phép sáng tạo](assignment.md)

---

**Tuyên bố miễn trừ trách nhiệm**:  
Tài liệu này đã được dịch bằng dịch vụ dịch thuật AI [Co-op Translator](https://github.com/Azure/co-op-translator). Mặc dù chúng tôi cố gắng đảm bảo độ chính xác, xin lưu ý rằng các bản dịch tự động có thể chứa lỗi hoặc không chính xác. Tài liệu gốc bằng ngôn ngữ bản địa nên được coi là nguồn thông tin chính thức. Đối với các thông tin quan trọng, khuyến nghị sử dụng dịch vụ dịch thuật chuyên nghiệp bởi con người. Chúng tôi không chịu trách nhiệm cho bất kỳ sự hiểu lầm hoặc diễn giải sai nào phát sinh từ việc sử dụng bản dịch này.