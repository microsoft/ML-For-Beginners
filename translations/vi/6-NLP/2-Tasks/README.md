<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "5f3cb462e3122e1afe7ab0050ccf2bd3",
  "translation_date": "2025-09-05T20:26:06+00:00",
  "source_file": "6-NLP/2-Tasks/README.md",
  "language_code": "vi"
}
-->
# Các nhiệm vụ và kỹ thuật phổ biến trong xử lý ngôn ngữ tự nhiên

Đối với hầu hết các nhiệm vụ *xử lý ngôn ngữ tự nhiên*, văn bản cần được xử lý phải được phân tích, kiểm tra, và kết quả được lưu trữ hoặc đối chiếu với các quy tắc và tập dữ liệu. Những nhiệm vụ này cho phép lập trình viên xác định _ý nghĩa_, _mục đích_, hoặc chỉ đơn giản là _tần suất_ của các thuật ngữ và từ trong văn bản.

## [Câu hỏi trước bài giảng](https://ff-quizzes.netlify.app/en/ml/)

Hãy khám phá các kỹ thuật phổ biến được sử dụng trong xử lý văn bản. Kết hợp với học máy, các kỹ thuật này giúp bạn phân tích lượng lớn văn bản một cách hiệu quả. Tuy nhiên, trước khi áp dụng học máy vào các nhiệm vụ này, hãy tìm hiểu các vấn đề mà một chuyên gia NLP thường gặp phải.

## Các nhiệm vụ phổ biến trong NLP

Có nhiều cách khác nhau để phân tích văn bản mà bạn đang làm việc. Có những nhiệm vụ bạn có thể thực hiện, và thông qua các nhiệm vụ này, bạn có thể hiểu được văn bản và rút ra kết luận. Thông thường, bạn thực hiện các nhiệm vụ này theo một trình tự.

### Phân tách từ (Tokenization)

Có lẽ điều đầu tiên mà hầu hết các thuật toán NLP phải làm là chia văn bản thành các token, hoặc từ. Mặc dù điều này nghe có vẻ đơn giản, việc phải xử lý dấu câu và các dấu phân cách từ và câu của các ngôn ngữ khác nhau có thể làm cho nó trở nên phức tạp. Bạn có thể phải sử dụng nhiều phương pháp để xác định các điểm phân cách.

![tokenization](../../../../6-NLP/2-Tasks/images/tokenization.png)
> Phân tách một câu từ **Pride and Prejudice**. Đồ họa thông tin bởi [Jen Looper](https://twitter.com/jenlooper)

### Biểu diễn từ (Embeddings)

[Biểu diễn từ](https://wikipedia.org/wiki/Word_embedding) là một cách để chuyển đổi dữ liệu văn bản của bạn thành dạng số. Biểu diễn được thực hiện sao cho các từ có ý nghĩa tương tự hoặc các từ thường được sử dụng cùng nhau sẽ được nhóm lại gần nhau.

![word embeddings](../../../../6-NLP/2-Tasks/images/embedding.png)
> "Tôi rất tôn trọng thần kinh của bạn, chúng là những người bạn cũ của tôi." - Biểu diễn từ cho một câu trong **Pride and Prejudice**. Đồ họa thông tin bởi [Jen Looper](https://twitter.com/jenlooper)

✅ Thử [công cụ thú vị này](https://projector.tensorflow.org/) để thực nghiệm với biểu diễn từ. Nhấp vào một từ sẽ hiển thị các nhóm từ tương tự: 'toy' được nhóm với 'disney', 'lego', 'playstation', và 'console'.

### Phân tích cú pháp & Gắn thẻ từ loại (Parsing & Part-of-speech Tagging)

Mỗi từ đã được phân tách có thể được gắn thẻ như một từ loại - danh từ, động từ, hoặc tính từ. Câu `the quick red fox jumped over the lazy brown dog` có thể được gắn thẻ từ loại như fox = danh từ, jumped = động từ.

![parsing](../../../../6-NLP/2-Tasks/images/parse.png)

> Phân tích cú pháp một câu từ **Pride and Prejudice**. Đồ họa thông tin bởi [Jen Looper](https://twitter.com/jenlooper)

Phân tích cú pháp là việc nhận biết các từ có liên quan với nhau trong một câu - ví dụ `the quick red fox jumped` là một chuỗi tính từ-danh từ-động từ tách biệt với chuỗi `lazy brown dog`.

### Tần suất từ và cụm từ

Một quy trình hữu ích khi phân tích một lượng lớn văn bản là xây dựng một từ điển của mọi từ hoặc cụm từ quan tâm và tần suất xuất hiện của chúng. Cụm từ `the quick red fox jumped over the lazy brown dog` có tần suất từ là 2 cho từ "the".

Hãy xem một ví dụ văn bản nơi chúng ta đếm tần suất từ. Bài thơ The Winners của Rudyard Kipling chứa đoạn sau:

```output
What the moral? Who rides may read.
When the night is thick and the tracks are blind
A friend at a pinch is a friend, indeed,
But a fool to wait for the laggard behind.
Down to Gehenna or up to the Throne,
He travels the fastest who travels alone.
```

Vì tần suất cụm từ có thể không phân biệt chữ hoa chữ thường hoặc phân biệt chữ hoa chữ thường tùy theo yêu cầu, cụm từ `a friend` có tần suất là 2 và `the` có tần suất là 6, và `travels` là 2.

### N-grams

Văn bản có thể được chia thành các chuỗi từ với độ dài cố định, một từ (unigram), hai từ (bigram), ba từ (trigram) hoặc bất kỳ số lượng từ nào (n-grams).

Ví dụ `the quick red fox jumped over the lazy brown dog` với điểm n-gram là 2 tạo ra các n-grams sau:

1. the quick 
2. quick red 
3. red fox
4. fox jumped 
5. jumped over 
6. over the 
7. the lazy 
8. lazy brown 
9. brown dog

Có thể dễ dàng hình dung nó như một hộp trượt qua câu. Đây là ví dụ cho n-grams gồm 3 từ, n-gram được in đậm trong mỗi câu:

1.   <u>**the quick red**</u> fox jumped over the lazy brown dog
2.   the **<u>quick red fox</u>** jumped over the lazy brown dog
3.   the quick **<u>red fox jumped</u>** over the lazy brown dog
4.   the quick red **<u>fox jumped over</u>** the lazy brown dog
5.   the quick red fox **<u>jumped over the</u>** lazy brown dog
6.   the quick red fox jumped **<u>over the lazy</u>** brown dog
7.   the quick red fox jumped over <u>**the lazy brown**</u> dog
8.   the quick red fox jumped over the **<u>lazy brown dog</u>**

![n-grams sliding window](../../../../6-NLP/2-Tasks/images/n-grams.gif)

> Giá trị N-gram là 3: Đồ họa thông tin bởi [Jen Looper](https://twitter.com/jenlooper)

### Trích xuất cụm danh từ

Trong hầu hết các câu, có một danh từ là chủ ngữ hoặc đối tượng của câu. Trong tiếng Anh, nó thường được nhận biết bằng cách có 'a', 'an', hoặc 'the' đứng trước. Xác định chủ ngữ hoặc đối tượng của một câu bằng cách 'trích xuất cụm danh từ' là một nhiệm vụ phổ biến trong NLP khi cố gắng hiểu ý nghĩa của câu.

✅ Trong câu "I cannot fix on the hour, or the spot, or the look or the words, which laid the foundation. It is too long ago. I was in the middle before I knew that I had begun.", bạn có thể xác định các cụm danh từ không?

Trong câu `the quick red fox jumped over the lazy brown dog` có 2 cụm danh từ: **quick red fox** và **lazy brown dog**.

### Phân tích cảm xúc

Một câu hoặc văn bản có thể được phân tích để xác định cảm xúc, hoặc mức độ *tích cực* hay *tiêu cực*. Cảm xúc được đo lường bằng *độ phân cực* và *khách quan/chủ quan*. Độ phân cực được đo từ -1.0 đến 1.0 (tiêu cực đến tích cực) và 0.0 đến 1.0 (khách quan nhất đến chủ quan nhất).

✅ Sau này bạn sẽ học rằng có nhiều cách khác nhau để xác định cảm xúc bằng cách sử dụng học máy, nhưng một cách là có một danh sách các từ và cụm từ được phân loại là tích cực hoặc tiêu cực bởi một chuyên gia con người và áp dụng mô hình đó vào văn bản để tính điểm phân cực. Bạn có thể thấy cách này hoạt động tốt trong một số trường hợp và không tốt trong các trường hợp khác?

### Biến đổi từ (Inflection)

Biến đổi từ cho phép bạn lấy một từ và tìm dạng số ít hoặc số nhiều của từ đó.

### Chuẩn hóa từ (Lemmatization)

Một *lemma* là gốc hoặc từ chính cho một tập hợp các từ, ví dụ *flew*, *flies*, *flying* có lemma là động từ *fly*.

Ngoài ra còn có các cơ sở dữ liệu hữu ích dành cho nhà nghiên cứu NLP, đáng chú ý là:

### WordNet

[WordNet](https://wordnet.princeton.edu/) là một cơ sở dữ liệu về từ, từ đồng nghĩa, từ trái nghĩa và nhiều chi tiết khác cho mỗi từ trong nhiều ngôn ngữ khác nhau. Nó cực kỳ hữu ích khi cố gắng xây dựng các công cụ dịch thuật, kiểm tra chính tả, hoặc các công cụ ngôn ngữ thuộc bất kỳ loại nào.

## Thư viện NLP

May mắn thay, bạn không cần phải tự xây dựng tất cả các kỹ thuật này, vì có các thư viện Python xuất sắc giúp việc này trở nên dễ tiếp cận hơn đối với các nhà phát triển không chuyên về xử lý ngôn ngữ tự nhiên hoặc học máy. Các bài học tiếp theo sẽ bao gồm nhiều ví dụ hơn về những thư viện này, nhưng ở đây bạn sẽ học một số ví dụ hữu ích để giúp bạn với nhiệm vụ tiếp theo.

### Bài tập - sử dụng thư viện `TextBlob`

Hãy sử dụng một thư viện gọi là TextBlob vì nó chứa các API hữu ích để giải quyết các loại nhiệm vụ này. TextBlob "được xây dựng trên nền tảng vững chắc của [NLTK](https://nltk.org) và [pattern](https://github.com/clips/pattern), và hoạt động tốt với cả hai." Nó có một lượng lớn học máy được tích hợp trong API của mình.

> Lưu ý: Một hướng dẫn [Quick Start](https://textblob.readthedocs.io/en/dev/quickstart.html#quickstart) hữu ích có sẵn cho TextBlob, được khuyến nghị cho các nhà phát triển Python có kinh nghiệm.

Khi cố gắng xác định *cụm danh từ*, TextBlob cung cấp một số tùy chọn trình trích xuất để tìm cụm danh từ.

1. Hãy xem `ConllExtractor`.

    ```python
    from textblob import TextBlob
    from textblob.np_extractors import ConllExtractor
    # import and create a Conll extractor to use later 
    extractor = ConllExtractor()
    
    # later when you need a noun phrase extractor:
    user_input = input("> ")
    user_input_blob = TextBlob(user_input, np_extractor=extractor)  # note non-default extractor specified
    np = user_input_blob.noun_phrases                                    
    ```

    > Điều gì đang diễn ra ở đây? [ConllExtractor](https://textblob.readthedocs.io/en/dev/api_reference.html?highlight=Conll#textblob.en.np_extractors.ConllExtractor) là "Một trình trích xuất cụm danh từ sử dụng phân tích cú pháp khối được huấn luyện với tập dữ liệu huấn luyện ConLL-2000." ConLL-2000 đề cập đến Hội nghị về Học Tự nhiên Ngôn ngữ Tính toán năm 2000. Mỗi năm hội nghị tổ chức một hội thảo để giải quyết một vấn đề khó khăn trong NLP, và năm 2000 là phân tích khối danh từ. Một mô hình đã được huấn luyện trên Wall Street Journal, với "các phần 15-18 làm dữ liệu huấn luyện (211727 token) và phần 20 làm dữ liệu kiểm tra (47377 token)". Bạn có thể xem các quy trình được sử dụng [tại đây](https://www.clips.uantwerpen.be/conll2000/chunking/) và [kết quả](https://ifarm.nl/erikt/research/np-chunking.html).

### Thử thách - cải thiện bot của bạn với NLP

Trong bài học trước, bạn đã xây dựng một bot Q&A rất đơn giản. Bây giờ, bạn sẽ làm cho Marvin trở nên đồng cảm hơn bằng cách phân tích đầu vào của bạn để xác định cảm xúc và in ra phản hồi phù hợp với cảm xúc đó. Bạn cũng cần xác định một `noun_phrase` và hỏi về nó.

Các bước của bạn khi xây dựng một bot trò chuyện tốt hơn:

1. In hướng dẫn khuyên người dùng cách tương tác với bot
2. Bắt đầu vòng lặp 
   1. Nhận đầu vào từ người dùng
   2. Nếu người dùng yêu cầu thoát, thì thoát
   3. Xử lý đầu vào của người dùng và xác định phản hồi cảm xúc phù hợp
   4. Nếu một cụm danh từ được phát hiện trong cảm xúc, chuyển nó sang dạng số nhiều và hỏi thêm về chủ đề đó
   5. In phản hồi
3. Quay lại bước 2

Đây là đoạn mã để xác định cảm xúc bằng TextBlob. Lưu ý rằng chỉ có bốn *mức độ* phản hồi cảm xúc (bạn có thể thêm nhiều hơn nếu muốn):

```python
if user_input_blob.polarity <= -0.5:
  response = "Oh dear, that sounds bad. "
elif user_input_blob.polarity <= 0:
  response = "Hmm, that's not great. "
elif user_input_blob.polarity <= 0.5:
  response = "Well, that sounds positive. "
elif user_input_blob.polarity <= 1:
  response = "Wow, that sounds great. "
```

Đây là một số đầu ra mẫu để hướng dẫn bạn (đầu vào của người dùng nằm trên các dòng bắt đầu bằng >):

```output
Hello, I am Marvin, the friendly robot.
You can end this conversation at any time by typing 'bye'
After typing each answer, press 'enter'
How are you today?
> I am ok
Well, that sounds positive. Can you tell me more?
> I went for a walk and saw a lovely cat
Well, that sounds positive. Can you tell me more about lovely cats?
> cats are the best. But I also have a cool dog
Wow, that sounds great. Can you tell me more about cool dogs?
> I have an old hounddog but he is sick
Hmm, that's not great. Can you tell me more about old hounddogs?
> bye
It was nice talking to you, goodbye!
```

Một giải pháp khả thi cho nhiệm vụ này có thể được tìm thấy [tại đây](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/2-Tasks/solution/bot.py)

✅ Kiểm tra kiến thức

1. Bạn có nghĩ rằng các phản hồi đồng cảm có thể 'lừa' ai đó nghĩ rằng bot thực sự hiểu họ không?
2. Việc xác định cụm danh từ có làm cho bot trở nên 'đáng tin' hơn không?
3. Tại sao việc trích xuất một 'cụm danh từ' từ một câu lại là điều hữu ích?

---

Hãy triển khai bot trong phần kiểm tra kiến thức trước đó và thử nghiệm nó với một người bạn. Nó có thể lừa họ không? Bạn có thể làm cho bot của mình trở nên 'đáng tin' hơn không?

## 🚀Thử thách

Hãy thực hiện một nhiệm vụ trong phần kiểm tra kiến thức trước đó và thử triển khai nó. Thử nghiệm bot với một người bạn. Nó có thể lừa họ không? Bạn có thể làm cho bot của mình trở nên 'đáng tin' hơn không?

## [Câu hỏi sau bài giảng](https://ff-quizzes.netlify.app/en/ml/)

## Ôn tập & Tự học

Trong các bài học tiếp theo, bạn sẽ học thêm về phân tích cảm xúc. Nghiên cứu kỹ thuật thú vị này trong các bài viết như bài viết trên [KDNuggets](https://www.kdnuggets.com/tag/nlp)

## Bài tập 

[Hãy làm cho bot phản hồi](assignment.md)

---

**Tuyên bố miễn trừ trách nhiệm**:  
Tài liệu này đã được dịch bằng dịch vụ dịch thuật AI [Co-op Translator](https://github.com/Azure/co-op-translator). Mặc dù chúng tôi cố gắng đảm bảo độ chính xác, xin lưu ý rằng các bản dịch tự động có thể chứa lỗi hoặc không chính xác. Tài liệu gốc bằng ngôn ngữ bản địa nên được coi là nguồn thông tin chính thức. Đối với các thông tin quan trọng, khuyến nghị sử dụng dịch vụ dịch thuật chuyên nghiệp bởi con người. Chúng tôi không chịu trách nhiệm cho bất kỳ sự hiểu lầm hoặc diễn giải sai nào phát sinh từ việc sử dụng bản dịch này.