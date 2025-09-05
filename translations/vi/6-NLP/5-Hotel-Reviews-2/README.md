<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "2c742993fe95d5bcbb2846eda3d442a1",
  "translation_date": "2025-09-05T20:44:42+00:00",
  "source_file": "6-NLP/5-Hotel-Reviews-2/README.md",
  "language_code": "vi"
}
-->
# Phân tích cảm xúc với đánh giá khách sạn

Bây giờ bạn đã khám phá chi tiết bộ dữ liệu, đã đến lúc lọc các cột và sử dụng các kỹ thuật NLP trên bộ dữ liệu để thu thập những thông tin mới về các khách sạn.

## [Câu hỏi trước bài giảng](https://ff-quizzes.netlify.app/en/ml/)

### Các thao tác lọc và phân tích cảm xúc

Như bạn có thể đã nhận thấy, bộ dữ liệu có một số vấn đề. Một số cột chứa thông tin không cần thiết, trong khi những cột khác dường như không chính xác. Nếu chúng chính xác, thì cũng không rõ cách chúng được tính toán, và bạn không thể tự mình xác minh kết quả.

## Bài tập: xử lý dữ liệu thêm một chút

Hãy làm sạch dữ liệu thêm một chút. Thêm các cột hữu ích cho các bước sau, thay đổi giá trị trong các cột khác, và loại bỏ hoàn toàn một số cột.

1. Xử lý cột ban đầu

   1. Loại bỏ `lat` và `lng`

   2. Thay thế giá trị `Hotel_Address` bằng các giá trị sau (nếu địa chỉ chứa tên thành phố và quốc gia, hãy thay đổi thành chỉ tên thành phố và quốc gia).

      Đây là các thành phố và quốc gia duy nhất trong bộ dữ liệu:

      Amsterdam, Netherlands

      Barcelona, Spain

      London, United Kingdom

      Milan, Italy

      Paris, France

      Vienna, Austria 

      ```python
      def replace_address(row):
          if "Netherlands" in row["Hotel_Address"]:
              return "Amsterdam, Netherlands"
          elif "Barcelona" in row["Hotel_Address"]:
              return "Barcelona, Spain"
          elif "United Kingdom" in row["Hotel_Address"]:
              return "London, United Kingdom"
          elif "Milan" in row["Hotel_Address"]:        
              return "Milan, Italy"
          elif "France" in row["Hotel_Address"]:
              return "Paris, France"
          elif "Vienna" in row["Hotel_Address"]:
              return "Vienna, Austria" 
      
      # Replace all the addresses with a shortened, more useful form
      df["Hotel_Address"] = df.apply(replace_address, axis = 1)
      # The sum of the value_counts() should add up to the total number of reviews
      print(df["Hotel_Address"].value_counts())
      ```

      Bây giờ bạn có thể truy vấn dữ liệu theo cấp quốc gia:

      ```python
      display(df.groupby("Hotel_Address").agg({"Hotel_Name": "nunique"}))
      ```

      | Hotel_Address          | Hotel_Name |
      | :--------------------- | :--------: |
      | Amsterdam, Netherlands |    105     |
      | Barcelona, Spain       |    211     |
      | London, United Kingdom |    400     |
      | Milan, Italy           |    162     |
      | Paris, France          |    458     |
      | Vienna, Austria        |    158     |

2. Xử lý các cột Meta-review của khách sạn

   1. Loại bỏ `Additional_Number_of_Scoring`

   2. Thay thế `Total_Number_of_Reviews` bằng tổng số đánh giá thực sự có trong bộ dữ liệu cho khách sạn đó

   3. Thay thế `Average_Score` bằng điểm số tự tính toán của chúng ta

   ```python
  # Drop `Additional_Number_of_Scoring`
  df.drop(["Additional_Number_of_Scoring"], axis = 1, inplace=True)
  # Replace `Total_Number_of_Reviews` and `Average_Score` with our own calculated values
  df.Total_Number_of_Reviews = df.groupby('Hotel_Name').transform('count')
  df.Average_Score = round(df.groupby('Hotel_Name').Reviewer_Score.transform('mean'), 1)
  ```

3. Xử lý các cột đánh giá

   1. Loại bỏ `Review_Total_Negative_Word_Counts`, `Review_Total_Positive_Word_Counts`, `Review_Date` và `days_since_review`

   2. Giữ nguyên `Reviewer_Score`, `Negative_Review`, và `Positive_Review`
     
   3. Giữ `Tags` tạm thời

     - Chúng ta sẽ thực hiện một số thao tác lọc bổ sung trên các thẻ trong phần tiếp theo và sau đó sẽ loại bỏ các thẻ

4. Xử lý các cột của người đánh giá

   1. Loại bỏ `Total_Number_of_Reviews_Reviewer_Has_Given`
  
   2. Giữ `Reviewer_Nationality`

### Cột thẻ (Tags)

Cột `Tag` gặp vấn đề vì nó là một danh sách (dạng văn bản) được lưu trong cột. Thật không may, thứ tự và số lượng các phần trong cột này không phải lúc nào cũng giống nhau. Rất khó để con người xác định các cụm từ chính xác cần quan tâm, vì có 515,000 hàng, 1427 khách sạn, và mỗi khách sạn có các tùy chọn hơi khác nhau mà người đánh giá có thể chọn. Đây là nơi NLP phát huy tác dụng. Bạn có thể quét văn bản và tìm các cụm từ phổ biến nhất, sau đó đếm chúng.

Thật không may, chúng ta không quan tâm đến các từ đơn lẻ, mà là các cụm từ nhiều từ (ví dụ: *Business trip*). Chạy một thuật toán phân phối tần suất cụm từ nhiều từ trên lượng dữ liệu lớn như vậy (6762646 từ) có thể mất rất nhiều thời gian, nhưng nếu không xem xét dữ liệu, có vẻ như đó là một chi phí cần thiết. Đây là lúc phân tích dữ liệu khám phá trở nên hữu ích, vì bạn đã thấy một mẫu của các thẻ như `[' Business trip  ', ' Solo traveler ', ' Single Room ', ' Stayed 5 nights ', ' Submitted from  a mobile device ']`, bạn có thể bắt đầu hỏi liệu có thể giảm đáng kể khối lượng xử lý cần thực hiện hay không. May mắn thay, điều đó là có thể - nhưng trước tiên bạn cần thực hiện một vài bước để xác định các thẻ quan tâm.

### Lọc thẻ

Hãy nhớ rằng mục tiêu của bộ dữ liệu là thêm cảm xúc và các cột giúp bạn chọn khách sạn tốt nhất (cho bản thân hoặc có thể là một khách hàng yêu cầu bạn tạo bot gợi ý khách sạn). Bạn cần tự hỏi liệu các thẻ có hữu ích hay không trong bộ dữ liệu cuối cùng. Đây là một cách diễn giải (nếu bạn cần bộ dữ liệu cho các mục đích khác, các thẻ khác có thể được giữ lại/loại bỏ):

1. Loại hình chuyến đi là quan trọng và nên giữ lại
2. Loại nhóm khách là quan trọng và nên giữ lại
3. Loại phòng, suite, hoặc studio mà khách đã ở không liên quan (tất cả các khách sạn về cơ bản đều có các phòng giống nhau)
4. Thiết bị mà đánh giá được gửi từ không liên quan
5. Số đêm mà người đánh giá đã ở *có thể* liên quan nếu bạn cho rằng thời gian lưu trú dài hơn đồng nghĩa với việc họ thích khách sạn hơn, nhưng điều này không chắc chắn và có lẽ không liên quan

Tóm lại, **giữ lại 2 loại thẻ và loại bỏ các thẻ khác**.

Đầu tiên, bạn không muốn đếm các thẻ cho đến khi chúng ở định dạng tốt hơn, điều đó có nghĩa là loại bỏ các dấu ngoặc vuông và dấu ngoặc kép. Bạn có thể làm điều này theo nhiều cách, nhưng bạn muốn cách nhanh nhất vì nó có thể mất nhiều thời gian để xử lý lượng dữ liệu lớn. May mắn thay, pandas có một cách dễ dàng để thực hiện từng bước này.

```Python
# Remove opening and closing brackets
df.Tags = df.Tags.str.strip("[']")
# remove all quotes too
df.Tags = df.Tags.str.replace(" ', '", ",", regex = False)
```

Mỗi thẻ trở thành dạng như: `Business trip, Solo traveler, Single Room, Stayed 5 nights, Submitted from a mobile device`. 

Tiếp theo, chúng ta gặp một vấn đề. Một số đánh giá, hoặc hàng, có 5 cột, một số có 3, một số có 6. Đây là kết quả của cách bộ dữ liệu được tạo ra, và khó sửa chữa. Bạn muốn có một số tần suất của mỗi cụm từ, nhưng chúng ở các thứ tự khác nhau trong mỗi đánh giá, vì vậy số lượng có thể bị sai, và một khách sạn có thể không nhận được thẻ mà nó xứng đáng.

Thay vào đó, bạn sẽ sử dụng thứ tự khác nhau này để làm lợi thế, vì mỗi thẻ là cụm từ nhiều từ nhưng cũng được phân tách bằng dấu phẩy! Cách đơn giản nhất để làm điều này là tạo 6 cột tạm thời với mỗi thẻ được chèn vào cột tương ứng với thứ tự của nó trong thẻ. Sau đó, bạn có thể gộp 6 cột này thành một cột lớn và chạy phương thức `value_counts()` trên cột kết quả. Khi in ra, bạn sẽ thấy có 2428 thẻ duy nhất. Đây là một mẫu nhỏ:

| Tag                            | Count  |
| ------------------------------ | ------ |
| Leisure trip                   | 417778 |
| Submitted from a mobile device | 307640 |
| Couple                         | 252294 |
| Stayed 1 night                 | 193645 |
| Stayed 2 nights                | 133937 |
| Solo traveler                  | 108545 |
| Stayed 3 nights                | 95821  |
| Business trip                  | 82939  |
| Group                          | 65392  |
| Family with young children     | 61015  |
| Stayed 4 nights                | 47817  |
| Double Room                    | 35207  |
| Standard Double Room           | 32248  |
| Superior Double Room           | 31393  |
| Family with older children     | 26349  |
| Deluxe Double Room             | 24823  |
| Double or Twin Room            | 22393  |
| Stayed 5 nights                | 20845  |
| Standard Double or Twin Room   | 17483  |
| Classic Double Room            | 16989  |
| Superior Double or Twin Room   | 13570  |
| 2 rooms                        | 12393  |

Một số thẻ phổ biến như `Submitted from a mobile device` không có ích đối với chúng ta, vì vậy có thể là một ý tưởng thông minh để loại bỏ chúng trước khi đếm số lần xuất hiện của cụm từ, nhưng đây là một thao tác nhanh nên bạn có thể để chúng lại và bỏ qua.

### Loại bỏ các thẻ về thời gian lưu trú

Loại bỏ các thẻ này là bước đầu tiên, nó giảm số lượng thẻ cần xem xét một chút. Lưu ý rằng bạn không loại bỏ chúng khỏi bộ dữ liệu, chỉ chọn loại bỏ chúng khỏi việc xem xét như các giá trị để đếm/giữ trong bộ dữ liệu đánh giá.

| Length of stay   | Count  |
| ---------------- | ------ |
| Stayed 1 night   | 193645 |
| Stayed  2 nights | 133937 |
| Stayed 3 nights  | 95821  |
| Stayed  4 nights | 47817  |
| Stayed 5 nights  | 20845  |
| Stayed  6 nights | 9776   |
| Stayed 7 nights  | 7399   |
| Stayed  8 nights | 2502   |
| Stayed 9 nights  | 1293   |
| ...              | ...    |

Có rất nhiều loại phòng, suite, studio, căn hộ và v.v. Tất cả chúng đều có ý nghĩa tương tự và không liên quan đến bạn, vì vậy hãy loại bỏ chúng khỏi việc xem xét.

| Type of room                  | Count |
| ----------------------------- | ----- |
| Double Room                   | 35207 |
| Standard  Double Room         | 32248 |
| Superior Double Room          | 31393 |
| Deluxe  Double Room           | 24823 |
| Double or Twin Room           | 22393 |
| Standard  Double or Twin Room | 17483 |
| Classic Double Room           | 16989 |
| Superior  Double or Twin Room | 13570 |

Cuối cùng, và điều này thật thú vị (vì không mất nhiều công xử lý), bạn sẽ còn lại các thẻ *hữu ích* sau:

| Tag                                           | Count  |
| --------------------------------------------- | ------ |
| Leisure trip                                  | 417778 |
| Couple                                        | 252294 |
| Solo  traveler                                | 108545 |
| Business trip                                 | 82939  |
| Group (combined with Travellers with friends) | 67535  |
| Family with young children                    | 61015  |
| Family  with older children                   | 26349  |
| With a  pet                                   | 1405   |

Bạn có thể cho rằng `Travellers with friends` giống như `Group` hơn hoặc kém, và điều đó sẽ hợp lý để gộp hai thẻ này lại như trên. Mã để xác định các thẻ chính xác nằm trong [Tags notebook](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/1-notebook.ipynb).

Bước cuối cùng là tạo các cột mới cho từng thẻ này. Sau đó, đối với mỗi hàng đánh giá, nếu cột `Tag` khớp với một trong các cột mới, thêm giá trị 1, nếu không, thêm giá trị 0. Kết quả cuối cùng sẽ là số lượng người đánh giá đã chọn khách sạn này (tổng hợp) cho, ví dụ, công việc so với giải trí, hoặc để mang theo thú cưng, và đây là thông tin hữu ích khi gợi ý khách sạn.

```python
# Process the Tags into new columns
# The file Hotel_Reviews_Tags.py, identifies the most important tags
# Leisure trip, Couple, Solo traveler, Business trip, Group combined with Travelers with friends, 
# Family with young children, Family with older children, With a pet
df["Leisure_trip"] = df.Tags.apply(lambda tag: 1 if "Leisure trip" in tag else 0)
df["Couple"] = df.Tags.apply(lambda tag: 1 if "Couple" in tag else 0)
df["Solo_traveler"] = df.Tags.apply(lambda tag: 1 if "Solo traveler" in tag else 0)
df["Business_trip"] = df.Tags.apply(lambda tag: 1 if "Business trip" in tag else 0)
df["Group"] = df.Tags.apply(lambda tag: 1 if "Group" in tag or "Travelers with friends" in tag else 0)
df["Family_with_young_children"] = df.Tags.apply(lambda tag: 1 if "Family with young children" in tag else 0)
df["Family_with_older_children"] = df.Tags.apply(lambda tag: 1 if "Family with older children" in tag else 0)
df["With_a_pet"] = df.Tags.apply(lambda tag: 1 if "With a pet" in tag else 0)

```

### Lưu tệp của bạn

Cuối cùng, lưu bộ dữ liệu như hiện tại với một tên mới.

```python
df.drop(["Review_Total_Negative_Word_Counts", "Review_Total_Positive_Word_Counts", "days_since_review", "Total_Number_of_Reviews_Reviewer_Has_Given"], axis = 1, inplace=True)

# Saving new data file with calculated columns
print("Saving results to Hotel_Reviews_Filtered.csv")
df.to_csv(r'../data/Hotel_Reviews_Filtered.csv', index = False)
```

## Các thao tác phân tích cảm xúc

Trong phần cuối này, bạn sẽ áp dụng phân tích cảm xúc cho các cột đánh giá và lưu kết quả vào bộ dữ liệu.

## Bài tập: tải và lưu dữ liệu đã lọc

Lưu ý rằng bây giờ bạn đang tải bộ dữ liệu đã lọc được lưu trong phần trước, **không phải** bộ dữ liệu gốc.

```python
import time
import pandas as pd
import nltk as nltk
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')

# Load the filtered hotel reviews from CSV
df = pd.read_csv('../../data/Hotel_Reviews_Filtered.csv')

# You code will be added here


# Finally remember to save the hotel reviews with new NLP data added
print("Saving results to Hotel_Reviews_NLP.csv")
df.to_csv(r'../data/Hotel_Reviews_NLP.csv', index = False)
```

### Loại bỏ các từ dừng

Nếu bạn chạy phân tích cảm xúc trên các cột đánh giá tiêu cực và tích cực, có thể mất rất nhiều thời gian. Được thử nghiệm trên một laptop mạnh với CPU nhanh, mất từ 12 - 14 phút tùy thuộc vào thư viện phân tích cảm xúc được sử dụng. Đây là khoảng thời gian (tương đối) dài, vì vậy đáng để điều tra xem liệu có thể tăng tốc hay không. 

Loại bỏ các từ dừng, hoặc các từ tiếng Anh phổ biến không làm thay đổi cảm xúc của câu, là bước đầu tiên. Bằng cách loại bỏ chúng, phân tích cảm xúc sẽ chạy nhanh hơn, nhưng không kém chính xác (vì các từ dừng không ảnh hưởng đến cảm xúc, nhưng chúng làm chậm quá trình phân tích). 

Đánh giá tiêu cực dài nhất là 395 từ, nhưng sau khi loại bỏ các từ dừng, chỉ còn 195 từ.

Loại bỏ các từ dừng cũng là một thao tác nhanh, loại bỏ các từ dừng khỏi 2 cột đánh giá trên 515,000 hàng mất 3.3 giây trên thiết bị thử nghiệm. Có thể mất ít hoặc nhiều thời gian hơn một chút tùy thuộc vào tốc độ CPU, RAM, việc bạn có SSD hay không, và một số yếu tố khác. Thời gian ngắn tương đối của thao tác này có nghĩa là nếu nó cải thiện thời gian phân tích cảm xúc, thì đáng để thực hiện.

```python
from nltk.corpus import stopwords

# Load the hotel reviews from CSV
df = pd.read_csv("../../data/Hotel_Reviews_Filtered.csv")

# Remove stop words - can be slow for a lot of text!
# Ryan Han (ryanxjhan on Kaggle) has a great post measuring performance of different stop words removal approaches
# https://www.kaggle.com/ryanxjhan/fast-stop-words-removal # using the approach that Ryan recommends
start = time.time()
cache = set(stopwords.words("english"))
def remove_stopwords(review):
    text = " ".join([word for word in review.split() if word not in cache])
    return text

# Remove the stop words from both columns
df.Negative_Review = df.Negative_Review.apply(remove_stopwords)   
df.Positive_Review = df.Positive_Review.apply(remove_stopwords)
```

### Thực hiện phân tích cảm xúc

Bây giờ bạn nên tính toán phân tích cảm xúc cho cả cột đánh giá tiêu cực và tích cực, và lưu kết quả vào 2 cột mới. Bài kiểm tra cảm xúc sẽ là so sánh nó với điểm số của người đánh giá cho cùng một đánh giá. Ví dụ, nếu phân tích cảm xúc cho rằng đánh giá tiêu cực có cảm xúc là 1 (cảm xúc cực kỳ tích cực) và cảm xúc của đánh giá tích cực là 1, nhưng người đánh giá cho khách sạn điểm thấp nhất có thể, thì hoặc văn bản đánh giá không khớp với điểm số, hoặc công cụ phân tích cảm xúc không thể nhận diện cảm xúc chính xác. Bạn nên mong đợi một số điểm cảm xúc hoàn toàn sai, và thường điều đó sẽ có thể giải thích được, ví dụ: đánh giá có thể cực kỳ mỉa mai "Tất nhiên tôi RẤT THÍCH ngủ trong một phòng không có sưởi" và công cụ phân tích cảm xúc nghĩ rằng đó là cảm xúc tích cực, mặc dù con người đọc sẽ biết đó là sự mỉa mai.
NLTK cung cấp nhiều công cụ phân tích cảm xúc khác nhau để bạn học hỏi, và bạn có thể thay thế chúng để xem liệu kết quả phân tích cảm xúc có chính xác hơn hay không. Phân tích cảm xúc VADER được sử dụng ở đây.

> Hutto, C.J. & Gilbert, E.E. (2014). VADER: A Parsimonious Rule-based Model for Sentiment Analysis of Social Media Text. Eighth International Conference on Weblogs and Social Media (ICWSM-14). Ann Arbor, MI, June 2014.

```python
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Create the vader sentiment analyser (there are others in NLTK you can try too)
vader_sentiment = SentimentIntensityAnalyzer()
# Hutto, C.J. & Gilbert, E.E. (2014). VADER: A Parsimonious Rule-based Model for Sentiment Analysis of Social Media Text. Eighth International Conference on Weblogs and Social Media (ICWSM-14). Ann Arbor, MI, June 2014.

# There are 3 possibilities of input for a review:
# It could be "No Negative", in which case, return 0
# It could be "No Positive", in which case, return 0
# It could be a review, in which case calculate the sentiment
def calc_sentiment(review):    
    if review == "No Negative" or review == "No Positive":
        return 0
    return vader_sentiment.polarity_scores(review)["compound"]    
```

Sau đó, trong chương trình của bạn, khi bạn sẵn sàng tính toán cảm xúc, bạn có thể áp dụng nó cho từng đánh giá như sau:

```python
# Add a negative sentiment and positive sentiment column
print("Calculating sentiment columns for both positive and negative reviews")
start = time.time()
df["Negative_Sentiment"] = df.Negative_Review.apply(calc_sentiment)
df["Positive_Sentiment"] = df.Positive_Review.apply(calc_sentiment)
end = time.time()
print("Calculating sentiment took " + str(round(end - start, 2)) + " seconds")
```

Quá trình này mất khoảng 120 giây trên máy tính của tôi, nhưng thời gian sẽ khác nhau trên mỗi máy tính. Nếu bạn muốn in kết quả và xem liệu cảm xúc có khớp với đánh giá hay không:

```python
df = df.sort_values(by=["Negative_Sentiment"], ascending=True)
print(df[["Negative_Review", "Negative_Sentiment"]])
df = df.sort_values(by=["Positive_Sentiment"], ascending=True)
print(df[["Positive_Review", "Positive_Sentiment"]])
```

Việc cuối cùng cần làm với tệp trước khi sử dụng nó trong thử thách là lưu lại! Bạn cũng nên cân nhắc sắp xếp lại tất cả các cột mới để chúng dễ làm việc hơn (đối với con người, đây là một thay đổi mang tính thẩm mỹ).

```python
# Reorder the columns (This is cosmetic, but to make it easier to explore the data later)
df = df.reindex(["Hotel_Name", "Hotel_Address", "Total_Number_of_Reviews", "Average_Score", "Reviewer_Score", "Negative_Sentiment", "Positive_Sentiment", "Reviewer_Nationality", "Leisure_trip", "Couple", "Solo_traveler", "Business_trip", "Group", "Family_with_young_children", "Family_with_older_children", "With_a_pet", "Negative_Review", "Positive_Review"], axis=1)

print("Saving results to Hotel_Reviews_NLP.csv")
df.to_csv(r"../data/Hotel_Reviews_NLP.csv", index = False)
```

Bạn nên chạy toàn bộ mã trong [notebook phân tích](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/3-notebook.ipynb) (sau khi bạn đã chạy [notebook lọc dữ liệu](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/1-notebook.ipynb) để tạo tệp Hotel_Reviews_Filtered.csv).

Để tổng kết, các bước thực hiện là:

1. Tệp dữ liệu gốc **Hotel_Reviews.csv** được khám phá trong bài học trước với [notebook khám phá](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/4-Hotel-Reviews-1/solution/notebook.ipynb)
2. Hotel_Reviews.csv được lọc bởi [notebook lọc dữ liệu](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/1-notebook.ipynb) tạo ra **Hotel_Reviews_Filtered.csv**
3. Hotel_Reviews_Filtered.csv được xử lý bởi [notebook phân tích cảm xúc](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/3-notebook.ipynb) tạo ra **Hotel_Reviews_NLP.csv**
4. Sử dụng Hotel_Reviews_NLP.csv trong Thử thách NLP dưới đây

### Kết luận

Khi bạn bắt đầu, bạn có một tập dữ liệu với các cột và dữ liệu nhưng không phải tất cả đều có thể được xác minh hoặc sử dụng. Bạn đã khám phá dữ liệu, lọc ra những gì không cần thiết, chuyển đổi các thẻ thành thứ gì đó hữu ích, tính toán trung bình của riêng bạn, thêm một số cột cảm xúc và hy vọng rằng bạn đã học được một số điều thú vị về xử lý văn bản tự nhiên.

## [Câu hỏi sau bài giảng](https://ff-quizzes.netlify.app/en/ml/)

## Thử thách

Bây giờ bạn đã phân tích cảm xúc cho tập dữ liệu của mình, hãy thử sử dụng các chiến lược bạn đã học trong chương trình học này (có thể là phân cụm?) để xác định các mẫu liên quan đến cảm xúc.

## Ôn tập & Tự học

Tham khảo [module học này](https://docs.microsoft.com/en-us/learn/modules/classify-user-feedback-with-the-text-analytics-api/?WT.mc_id=academic-77952-leestott) để tìm hiểu thêm và sử dụng các công cụ khác nhau để khám phá cảm xúc trong văn bản.

## Bài tập

[Thử một tập dữ liệu khác](assignment.md)

---

**Tuyên bố miễn trừ trách nhiệm**:  
Tài liệu này đã được dịch bằng dịch vụ dịch thuật AI [Co-op Translator](https://github.com/Azure/co-op-translator). Mặc dù chúng tôi cố gắng đảm bảo độ chính xác, xin lưu ý rằng các bản dịch tự động có thể chứa lỗi hoặc không chính xác. Tài liệu gốc bằng ngôn ngữ bản địa nên được coi là nguồn thông tin chính thức. Đối với các thông tin quan trọng, khuyến nghị sử dụng dịch vụ dịch thuật chuyên nghiệp bởi con người. Chúng tôi không chịu trách nhiệm cho bất kỳ sự hiểu lầm hoặc diễn giải sai nào phát sinh từ việc sử dụng bản dịch này.