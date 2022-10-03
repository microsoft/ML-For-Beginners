# 호텔 리뷰로 감정 분석하기

지금까지 자세히 데이터셋을 살펴보았으며, 열을 필터링하고 데이터셋으로 NLP 기술을 사용하여 호텔에 대한 새로운 시각을 얻게 될 시간입니다.

## [강의 전 퀴즈](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/39/)

### 필터링 & 감정 분석 작업

알고 있는 것처럼, 데이터셋에 약간의 이슈가 있었습니다. 일부 열은 필요없는 정보로 채워져있으며, 부정확해 보입니다. 만약 맞는 경우, 어떻게 계산되었는지 불투명하고, 답을 스스로 계산해서 독립적으로 확인할 수 없습니다.

## 연습: 조금 더 데이터 처리하기

조금 더 데이터를 정리합니다. 열을 추가하는 것은 나중에 유용하며, 다른 열에서 값을 변경하고, 특정한 열을 완전히 드랍하게 됩니다.

1. 초기 column 처리합니다

   1. `lat` 과 `lng`를 드랍합니다

   2. `Hotel_Address` 값을 다음 값으로 치환합니다 (만약 주소에 도시와 국가가 같다면, 도시와 국가만 변경합니다).

      데이터셋에서 도시와 국가만 있습니다:

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

      지금부터 국가 레벨 데이터로 쿼리할 수 있습니다:

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

2. 호텔 Meta-review 열을 처리합니다

  1. `Additional_Number_of_Scoring`을 드랍합니다

  1. `Total_Number_of_Reviews`를 데이터셋에 실제로 있는 총 호텔 리뷰로 치환합니다

  1. `Average_Score`를 계산해둔 점수로 치환합니다
 
  ```python
  # Drop `Additional_Number_of_Scoring`
  df.drop(["Additional_Number_of_Scoring"], axis = 1, inplace=True)
  # Replace `Total_Number_of_Reviews` and `Average_Score` with our own calculated values
  df.Total_Number_of_Reviews = df.groupby('Hotel_Name').transform('count')
  df.Average_Score = round(df.groupby('Hotel_Name').Reviewer_Score.transform('mean'), 1)
  ```

3. 리뷰 열을 처리합니다

   1. `Review_Total_Negative_Word_Counts`, `Review_Total_Positive_Word_Counts`, `Review_Date` 그리고 `days_since_review`를 드랍합니다

   2. `Reviewer_Score`, `Negative_Review`, 그리고 `Positive_Review` 를 두고,
     
   3. 당장 `Tags` 도 둡니다

     - 다음 섹션에서 태그에 추가적인 필터링 작업을 조금 진행하고 태그를 드랍하게 됩니다

4. 리뷰어 열을 처리합니다

  1. `Total_Number_of_Reviews_Reviewer_Has_Given`을 드랍합니다
  
  2. `Reviewer_Nationality`를 둡니다

### Tag 열

`Tag`열은 열에 저장된 (텍스트 폼의) 리스트라서 문제가 됩니다. 불행하게 순서와 열의 서브 섹션의 숫자는 항상 같지 않습니다.  515,000 행과 1427개 호텔이고, 각자 리뷰어가 선택할 수 있는 옵션은 조금씩 다르기 때문에, 사람에게 흥미로운 알맞은 문구를 가리기 힘듭니다. NLP가 빛나는 영역입니다. 텍스트를 스캔하고 가장 일반적인 문구를 찾으며, 셀 수 있습니다.

불행히도, 단일 단어는 아니지만, multi-word 구문은 (예시. *Business trip*) 흥미롭습니다. 많은 데이터에 (6762646 단어) multi-word frequency distribution 알고리즘을 실행하는 것은 특별히 오래 걸릴 수 있지만, 데이터를 보지 않아도, 필요한 비용으로 보일 것입니다. `[' Business trip  ', ' Solo traveler ', ' Single Room ', ' Stayed 5 nights ', ' Submitted from  a mobile device ']` 처럼 태그 샘플로 보면, 해야 하는 처리로 많이 줄일 수 있는지 물어볼 수 있기 때문에, 탐색적 데이터 분석은 유용합니다. 운이 좋게도, - 그러나 먼저 관심있는 태그를 확실히 하고자 다음 몇 단계가 필요합니다.

### tags 필터링

데이터셋의 목표는 좋은 호텔을 선택할 때 (호텔 추천 봇을 만들어 달라고 맡기는 클라이언트일 수 있습니다) 도움을 받고자 감정과 열을 추가하는 것이라고 되새깁니다. 태그가 최종 데이터셋에서 유용한지 스스로에게 물어볼 필요가 있습니다. 한 가지 해석이 있습니다 (만약 다른 사유로 데이터셋이 필요한 경우에 선택할 수 있거나 안하기도 합니다):

1. 여행 타입이 적절하고, 유지되어야 합니다
2. 게스트 그룹 타입은 중요하고, 유지되어야 합니다
3. 게스트가 지낸 룸 타입, 스위트, 또는 스튜디오 타입은 관련 없습니다 (모든 호텔은 기본적으로 같은 룸이 존재합니다) 
4. 리뷰를 작성한 디바이스는 관련 없습니다
5. 만약 리뷰어가 좋아하는 호텔을 더 오래 지낸다면, 리뷰어가 지낸 숙박 기간과 *관련이 있을* 수 있지만, 확대 해석이며, 아마 관련 없습니다

요약하면, **2가지 종류 태그를 유지하고 나머지를 제거합니다**.

먼저, 더 좋은 포맷이 될 때까지 태그를 안 세고 싶으므로, square brackets과 quotes를 제거해야 합니다. 여러 방식으로 할 수 있지만, 많은 데이터를 처리하며 오랜 시간이 걸리므로 빠르게 하길 원합니다. 운이 좋게도, pandas는 각 단계를 쉬운 방식으로 할 수 있습니다.

```Python
# Remove opening and closing brackets
df.Tags = df.Tags.str.strip("[']")
# remove all quotes too
df.Tags = df.Tags.str.replace(" ', '", ",", regex = False)
```

각자 태그는 이처럼 이루어집니다: `Business trip, Solo traveler, Single Room, Stayed 5 nights, Submitted from a mobile device`. 

다음으로 문제를 찾았습니다. 리뷰, 또는 행에 5개 열이 있고, 일부는 3개이거나, 6개입니다. 데이터셋이 어떻게 만들어졌는가에 따른 결과이며, 고치기 어렵습니다. 각 구문의 빈도 카운트를 얻고 싶지만, 각 리뷰의 순서가 다르므로, 카운트에 벗어날 수 있고, 호텔이 가치가 있는 태그로 할당받지 못할 수 있습니다.

각 태그는 multi-word 지만 쉼표로 구분되어 있기 때문에, 대신 다른 순서로 사용하며 가산점을 받습니다! 간단한 방식은 태그에서 순서와 일치하는 열에 넣은 각 태그로 6개 임시 열을 만드는 것입니다. 6개 열을 하나의 큰 열로 합치고 결과 열에 `value_counts()` 메소드를 실행할 수 있습니다. 출력하면, 2428개 유니크 태그를 보게 됩니다. 여기 작은 샘플이 있습니다:

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

`Submitted from a mobile device` 같은 일부 일반적인 태그는 사용하지 못해서, phrase occurrence를 카운트하기 전에 지우는 게 똑똑할 수 있지만, 빠르게 작업하려면 그냥 두고 무시할 수 있습니다.

### length of stay 태그 지우기

이 태그를 지우는 것은 1단계이며, 고려할 태그의 총 개수를 약간 줄이게 됩니다. 데이터셋에서 지우지 말고, 리뷰 데이터셋에 카운트/유지할 값으로 고려할 대상에서 지우게 선택합니다.

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

룸, 스위트, 스튜디오, 아파트 등 매우 다양합니다. 대부분 같은 것을 의미하고 관련 없으므로, 대상에서 지웁니다.

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

최종적으로, (많이 처리할 일이 없기 때문에) 즐겁게, 다음 *유용한* 태그만 남길  예정입니다:

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

`Travellers with friends`는 `Group`과 거의 같다고 주장할 수 있어서, 둘을 합치면 공평할 것입니다. 올바른 태그로 식별하기 위한 코드는 [the Tags notebook](../solution/1-notebook.ipynb)에 있습니다.

마지막 단계는 각 태그로 새로운 열을 만드는 것입니다. 그러면, 모든 리뷰 행에서, `Tag` 열이 하나의 새로운 열과 매치되면, 1을 추가하고, 아니면, 0을 추가합니다. 마지막 결과는 비지니스 vs 레저, 또는 애완동물 동반 언급하면서, 호텔 추천할 때 유용한 정보이므로, 얼마나 많은 리뷰어가 (총계) 호텔을 선택했는지 카운트합니다.

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

### 파일 저장하기

마지막으로, 새로운 이름으로 바로 데이터셋을 저장합니다.

```python
df.drop(["Review_Total_Negative_Word_Counts", "Review_Total_Positive_Word_Counts", "days_since_review", "Total_Number_of_Reviews_Reviewer_Has_Given"], axis = 1, inplace=True)

# Saving new data file with calculated columns
print("Saving results to Hotel_Reviews_Filtered.csv")
df.to_csv(r'../data/Hotel_Reviews_Filtered.csv', index = False)
```

## 감정 분석 작업

마지막 섹션에서, 리뷰 열에 감정 분석을 적용하고 데이터셋에 결과를 저장힙니다.

## 연습: 필터링된 데이터를 불러오고 저장하기

지금 원본 데이터셋 *말고*, 이전 색션에서 저장했던 필터링된 데이터셋을 불러오고 있습니다.

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

### stop word 제거하기

만약 긍정적이고 부정적인 리뷰 열에 감정 분석을 하는 경우, 오랜 시간이 걸릴 수 있습니다. 빠른 CPU를 가진 강력한 노트북으로 테스트하면, 사용한 감정 라이브러리에 따라서 12 - 14 분 정도 걸립니다. (상대적)으로 오래 걸려서, 빠르게 할 수 있는지 알아볼 가치가 있습니다.

문장의 감정을 바꾸지 않는 stop word나, 일반적인 영어 단어를 지우는 것은, 첫 단계입니다. 지우게 된다면, 감정 분석이 더 빠르게 되지만, 정확도가 낮아지지 않습니다 (stop word는 감정에 영향없지만, 분석이 느려집니다).

긴 부정적 리뷰는 395 단어로 었지만 , stop word를 지우면, 195 단어만 남습니다.

stop word를 지우는 것은 빨라서, 테스트 디바이스에서 515,000 행이 넘는 2개 리뷰 열에 stop word를 지우면 3.3초 걸립니다. 디바이스의 CPU 속도, 램, SSD 등에 따라 더 오래 걸리거나 빨리 끝날 수 있습니다. 작업이 상대적으로 빨라지고 감정 분석 시간도 향상시킬 수 있다면, 할 가치가 있음을 의미합니다.

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

### 감정 분석하기

지금부터 모든 부정적이고 긍정적인 리뷰 열에 대한 감정 분석을 계산하고, 2개 열에 결과를 저장해야 합니다. 감정 테스트는 같은 리뷰로 리뷰어의 점수를 비교할 예정입니다. 예시로, 만약 부정적인 리뷰가 1 (많이 긍정적인 감정) 감정이고 1 긍정적인 리뷰 감정이라고 감정을 내렸지만, 리뷰어가 낮은 점수로 호텔을 리뷰하면, 리뷰 텍스트가 점수와 어느 것도 매치되지 않거나, sentiment analyser가 감정을 똑바로 인식할 수 없습니다. 일부 감정 점수는 다 틀릴 수 있고, 그 이유를 자주 설명할 수 있습니다. 예시로. "Of course I LOVED sleeping in a room with no heating" 리뷰는 극도로 풍자적이고 sentiment analyser는 긍정적인 감정이라고 생각하지만, 사람이 읽으면 풍자라는 것을 알 수 있습니다.

NLTK는 학습하는 다양한 sentiment analyzer를 제공하고, 이를 대신헤서 감정이 얼마나 정확한지 볼 수 있습니다. VADER sentiment analysis를 여기에서 사용했습니다. 

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

이후에 프로그램에서 감정을 계산하려 준비할 때, 다음 각 리뷰에서 적용할 수 있습니다:

```python
# Add a negative sentiment and positive sentiment column
print("Calculating sentiment columns for both positive and negative reviews")
start = time.time()
df["Negative_Sentiment"] = df.Negative_Review.apply(calc_sentiment)
df["Positive_Sentiment"] = df.Positive_Review.apply(calc_sentiment)
end = time.time()
print("Calculating sentiment took " + str(round(end - start, 2)) + " seconds")
```

이 컴퓨터에서 120초 정도 걸리지만, 각자 컴퓨터마다 다릅니다. 만약 결과를 출력하고 감정이 리뷰와 매치되는지 보려면 아래와 같이 진행합니다:

```python
df = df.sort_values(by=["Negative_Sentiment"], ascending=True)
print(df[["Negative_Review", "Negative_Sentiment"]])
df = df.sort_values(by=["Positive_Sentiment"], ascending=True)
print(df[["Positive_Review", "Positive_Sentiment"]])
```

마지막으로 할 일은 도전에서 사용하기 전, 파일을 저장하는 것입니다! 쉽게 작업하도록 모든 새로운 열로 다시 정렬을 (사람인 경우, 외형 변경) 고려해야 합니다.

```python
# Reorder the columns (This is cosmetic, but to make it easier to explore the data later)
df = df.reindex(["Hotel_Name", "Hotel_Address", "Total_Number_of_Reviews", "Average_Score", "Reviewer_Score", "Negative_Sentiment", "Positive_Sentiment", "Reviewer_Nationality", "Leisure_trip", "Couple", "Solo_traveler", "Business_trip", "Group", "Family_with_young_children", "Family_with_older_children", "With_a_pet", "Negative_Review", "Positive_Review"], axis=1)

print("Saving results to Hotel_Reviews_NLP.csv")
df.to_csv(r"../data/Hotel_Reviews_NLP.csv", index = False)
```

(Hotel_Reviews_Filtered.csv 파일 만들어서 [your filtering notebook](../solution/1-notebook.ipynb) 실행한 후에) [the analysis notebook](../solution/3-notebook.ipynb)으로 전체 코드를 실행해야 합니다.

검토하는, 단계는 이렇습니다:

1. 원본 데이터셋 **Hotel_Reviews.csv** 파일은 이전 강의에서 [the explorer notebook](../../4-Hotel-Reviews-1/solution/notebook.ipynb)으로 살펴보았습니다
2. Hotel_Reviews.csv는 [the filtering notebook](../solution/1-notebook.ipynb)에서 필터링되고 **Hotel_Reviews_Filtered.csv**에 결과로 남습니다
3. Hotel_Reviews_Filtered.csv는 [the sentiment analysis notebook](../solution/3-notebook.ipynb)에서 처리되어 **Hotel_Reviews_NLP.csv**에 결과로 남습니다
4. 다음 NLP 도전에서 Hotel_Reviews_NLP.csv를 사용합니다

### 결론

시작했을 때, 열과 데이터로 이루어진 데이터셋이 었었지만 모두 다 확인되거나 사용되지 않았습니다. 데이터를 살펴보았으며, 필요없는 것은 필터링해서 지웠고, 유용하게 태그를 변환했고, 평균을 계산했으며, 일부 감정 열을 추가하고 기대하면서, 자연어 처리에 대한 일부 흥미로운 사실을 학습했습니다.

## [강의 후 퀴즈](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/40/)

## 도전

이제부터 감정을 분석해둔 데이터셋을 가지고 있으므로, 이 커리큘럼 (clustering, perhaps?)에서 배웠던 전략으로 감정 주변 패턴을 결정해봅니다.

## 검토 & 자기주도 학습

[this Learn module](https://docs.microsoft.com/en-us/learn/modules/classify-user-feedback-with-the-text-analytics-api/?WT.mc_id=academic-77952-leestott)로 더 배우고 다른 도구도 사용해서 텍스트에서 감정을 찾습니다.

## 과제 

[Try a different dataset](../assignment.md)
