<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "a2aa4e9b91b9640db2c15363c4299d8b",
  "translation_date": "2025-09-04T00:56:23+00:00",
  "source_file": "6-NLP/5-Hotel-Reviews-2/README.md",
  "language_code": "ko"
}
-->
# 호텔 리뷰를 활용한 감정 분석

데이터셋을 자세히 탐색한 후, 이제 열을 필터링하고 NLP 기술을 사용하여 호텔에 대한 새로운 통찰력을 얻을 때입니다.
## [강의 전 퀴즈](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/39/)

### 필터링 및 감정 분석 작업

아마도 눈치채셨겠지만, 데이터셋에는 몇 가지 문제가 있습니다. 일부 열은 쓸모없는 정보로 채워져 있고, 다른 열은 부정확해 보입니다. 만약 정확하다면, 어떻게 계산되었는지 불분명하며, 여러분의 계산으로 독립적으로 검증할 수 없습니다.

## 연습: 데이터 처리 조금 더 하기

데이터를 조금 더 정리하세요. 나중에 유용할 열을 추가하고, 다른 열의 값을 변경하며, 특정 열을 완전히 삭제하세요.

1. 초기 열 처리

   1. `lat`와 `lng` 삭제

   2. `Hotel_Address` 값을 다음 값으로 대체 (주소에 도시와 국가 이름이 포함되어 있다면, 도시와 국가 이름만 남기세요).

      데이터셋에 포함된 도시와 국가는 다음과 같습니다:

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

      이제 국가 수준 데이터를 쿼리할 수 있습니다:

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

2. 호텔 메타 리뷰 열 처리

  1. `Additional_Number_of_Scoring` 삭제

  1. `Total_Number_of_Reviews`를 데이터셋에 실제로 포함된 해당 호텔의 리뷰 총 수로 대체

  1. `Average_Score`를 직접 계산한 점수로 대체

  ```python
  # Drop `Additional_Number_of_Scoring`
  df.drop(["Additional_Number_of_Scoring"], axis = 1, inplace=True)
  # Replace `Total_Number_of_Reviews` and `Average_Score` with our own calculated values
  df.Total_Number_of_Reviews = df.groupby('Hotel_Name').transform('count')
  df.Average_Score = round(df.groupby('Hotel_Name').Reviewer_Score.transform('mean'), 1)
  ```

3. 리뷰 열 처리

   1. `Review_Total_Negative_Word_Counts`, `Review_Total_Positive_Word_Counts`, `Review_Date`, `days_since_review` 삭제

   2. `Reviewer_Score`, `Negative_Review`, `Positive_Review`는 그대로 유지
     
   3. `Tags`는 일단 유지

     - 다음 섹션에서 태그에 추가 필터링 작업을 수행한 후 태그를 삭제할 예정

4. 리뷰어 열 처리

  1. `Total_Number_of_Reviews_Reviewer_Has_Given` 삭제
  
  2. `Reviewer_Nationality`는 유지

### 태그 열

`Tag` 열은 텍스트 형식의 리스트로 저장되어 있어 문제가 됩니다. 불행히도 이 열의 하위 섹션 순서와 개수가 항상 동일하지 않습니다. 사람이 올바른 관심 태그를 식별하기 어렵습니다. 왜냐하면 515,000개의 행과 1427개의 호텔이 있고, 각 리뷰어가 선택할 수 있는 옵션이 약간씩 다르기 때문입니다. 이럴 때 NLP가 빛을 발합니다. 텍스트를 스캔하여 가장 일반적인 구문을 찾고 이를 계산할 수 있습니다.

불행히도 단어 하나보다는 다단어 구문(예: *Business trip*)에 관심이 있습니다. 이 많은 데이터(6762646 단어)에 대해 다단어 빈도 분포 알고리즘을 실행하면 엄청난 시간이 걸릴 수 있습니다. 하지만 데이터를 살펴보지 않고는 필요한 작업인지 판단하기 어렵습니다. 탐색적 데이터 분석이 유용한 이유입니다. 예를 들어 `[' Business trip  ', ' Solo traveler ', ' Single Room ', ' Stayed 5 nights ', ' Submitted from  a mobile device ']`와 같은 태그 샘플을 보면, 처리량을 크게 줄일 수 있는지 질문을 시작할 수 있습니다. 다행히도 가능합니다. 하지만 먼저 관심 태그를 확인하기 위해 몇 가지 단계를 따라야 합니다.

### 태그 필터링

데이터셋의 목표는 감정을 추가하고 최종적으로 최고의 호텔을 선택하는 데 도움이 되는 열을 추가하는 것입니다(자신을 위해서든, 호텔 추천 봇을 만들라는 고객의 요청을 위해서든). 태그가 최종 데이터셋에서 유용한지 아닌지를 스스로 물어봐야 합니다. 다음은 한 가지 해석입니다(다른 이유로 데이터셋이 필요하다면 다른 태그가 포함/제외될 수 있습니다):

1. 여행 유형은 관련이 있으므로 유지해야 합니다.
2. 게스트 그룹 유형은 중요하므로 유지해야 합니다.
3. 게스트가 머문 방, 스위트룸, 스튜디오 유형은 관련이 없습니다(모든 호텔은 기본적으로 동일한 방을 가지고 있음).
4. 리뷰가 제출된 기기는 관련이 없습니다.
5. 리뷰어가 머문 밤 수는 호텔을 더 좋아하는 것과 연관이 있을 수 있지만, 가능성이 낮고 아마도 관련이 없습니다.

요약하자면, **두 가지 유형의 태그만 유지하고 나머지는 제거하세요**.

먼저, 태그를 더 나은 형식으로 만들기 전까지는 태그를 계산하지 않으려 합니다. 즉, 대괄호와 따옴표를 제거해야 합니다. 여러 방법이 있지만, 가장 빠른 방법을 선택해야 합니다. 많은 데이터를 처리하는 데 시간이 오래 걸릴 수 있기 때문입니다. 다행히도 pandas는 각 단계를 쉽게 수행할 수 있는 방법을 제공합니다.

```Python
# Remove opening and closing brackets
df.Tags = df.Tags.str.strip("[']")
# remove all quotes too
df.Tags = df.Tags.str.replace(" ', '", ",", regex = False)
```

각 태그는 다음과 같이 변환됩니다: `Business trip, Solo traveler, Single Room, Stayed 5 nights, Submitted from a mobile device`. 

다음으로 문제가 발생합니다. 일부 리뷰 또는 행에는 5개의 열이 있고, 일부는 3개, 일부는 6개입니다. 이는 데이터셋 생성 방식의 결과이며 수정하기 어렵습니다. 각 구문의 빈도 수를 얻고 싶지만, 리뷰마다 순서가 다르기 때문에 카운트가 정확하지 않을 수 있으며, 호텔이 받을 자격이 있는 태그를 받지 못할 수도 있습니다.

대신 이 다른 순서를 활용할 것입니다. 각 태그는 다단어 구문이지만 쉼표로 구분되어 있습니다! 가장 간단한 방법은 임시로 6개의 열을 생성하여 각 태그를 해당 순서에 맞는 열에 삽입하는 것입니다. 그런 다음 6개의 열을 하나의 큰 열로 병합하고 결과 열에서 `value_counts()` 메서드를 실행합니다. 이를 출력하면 2428개의 고유 태그가 있음을 확인할 수 있습니다. 다음은 작은 샘플입니다:

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

`Submitted from a mobile device`와 같은 일반적인 태그는 우리에게 아무런 도움이 되지 않으므로, 구문 발생 횟수를 계산하기 전에 제거하는 것이 현명할 수 있습니다. 하지만 작업이 매우 빠르기 때문에 그대로 두고 무시할 수 있습니다.

### 숙박 기간 태그 제거

이 태그를 제거하는 것이 첫 번째 단계입니다. 고려해야 할 태그의 총 수를 약간 줄입니다. 데이터셋에서 태그를 제거하지는 않고, 리뷰 데이터셋에서 값을 계산/유지 대상으로 선택하지 않을 뿐입니다.

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

방, 스위트룸, 스튜디오, 아파트 등 다양한 유형이 있습니다. 이들은 모두 대체로 동일한 의미를 가지며 관련이 없으므로 고려 대상에서 제거하세요.

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

마지막으로, 기쁜 소식은 (거의 처리 없이) 다음과 같은 **유용한** 태그만 남게 된다는 것입니다:

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

`Travellers with friends`는 `Group`과 거의 동일하다고 볼 수 있으며, 두 태그를 위와 같이 결합하는 것이 공정할 것입니다. 올바른 태그를 식별하는 코드는 [Tags notebook](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/1-notebook.ipynb)에 있습니다.

최종 단계는 각 태그에 대해 새로운 열을 생성하는 것입니다. 그런 다음 각 리뷰 행에서 `Tag` 열이 새 열 중 하나와 일치하면 1을 추가하고, 그렇지 않으면 0을 추가합니다. 최종 결과는 비즈니스 vs 레저, 또는 애완동물 동반 여부 등으로 호텔을 선택한 리뷰어 수를 집계한 데이터가 됩니다.

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

### 파일 저장

마지막으로, 현재 상태의 데이터셋을 새 이름으로 저장하세요.

```python
df.drop(["Review_Total_Negative_Word_Counts", "Review_Total_Positive_Word_Counts", "days_since_review", "Total_Number_of_Reviews_Reviewer_Has_Given"], axis = 1, inplace=True)

# Saving new data file with calculated columns
print("Saving results to Hotel_Reviews_Filtered.csv")
df.to_csv(r'../data/Hotel_Reviews_Filtered.csv', index = False)
```

## 감정 분석 작업

이 마지막 섹션에서는 리뷰 열에 감정 분석을 적용하고 결과를 데이터셋에 저장합니다.

## 연습: 필터링된 데이터 로드 및 저장

이제 이전 섹션에서 저장한 필터링된 데이터셋을 로드하며, **원본 데이터셋이 아님**을 유의하세요.

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

### 불용어 제거

Negative 및 Positive 리뷰 열에서 감정 분석을 실행하면 시간이 오래 걸릴 수 있습니다. 빠른 CPU를 가진 강력한 테스트 노트북에서 테스트한 결과, 사용된 감정 라이브러리에 따라 12~14분이 걸렸습니다. 이는 (상대적으로) 긴 시간이므로 속도를 높일 수 있는지 조사할 가치가 있습니다.

불용어 제거는 첫 번째 단계입니다. 불용어는 문장의 감정에 영향을 주지 않는 일반적인 영어 단어입니다. 이를 제거하면 감정 분석이 더 빨리 실행되지만 정확도가 떨어지지 않습니다(불용어는 감정에 영향을 주지 않지만 분석 속도를 늦춥니다).

가장 긴 부정 리뷰는 395단어였지만, 불용어를 제거한 후에는 195단어로 줄어듭니다.

불용어 제거는 또한 빠른 작업입니다. 515,000개의 행에서 2개의 리뷰 열에서 불용어를 제거하는 데 테스트 장치에서 3.3초가 걸렸습니다. CPU 속도, RAM, SSD 여부 등 여러 요인에 따라 시간이 약간 더 걸리거나 덜 걸릴 수 있습니다. 작업이 상대적으로 짧기 때문에 감정 분석 시간을 개선한다면 수행할 가치가 있습니다.

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

### 감정 분석 수행
이제 부정적 리뷰와 긍정적 리뷰 열에 대한 감정 분석을 계산하고, 결과를 두 개의 새로운 열에 저장해야 합니다. 감정 분석의 테스트는 동일한 리뷰에 대해 리뷰어가 준 점수와 비교하는 것입니다. 예를 들어, 감정 분석 결과 부정적 리뷰의 감정 점수가 1(매우 긍정적 감정)이고 긍정적 리뷰의 감정 점수도 1로 나왔지만, 리뷰어가 호텔에 최저 점수를 준 경우, 이는 리뷰 텍스트가 점수와 일치하지 않거나 감정 분석기가 감정을 제대로 인식하지 못했음을 의미합니다. 일부 감정 점수가 완전히 잘못될 수도 있으며, 이는 종종 설명 가능합니다. 예를 들어, 리뷰가 매우 풍자적일 수 있습니다. "난방이 없는 방에서 자는 게 정말 좋았어요" 같은 문장은 감정 분석기가 긍정적 감정으로 인식할 수 있지만, 사람이라면 이 문장이 풍자임을 알 수 있습니다.

NLTK는 학습할 수 있는 다양한 감정 분석기를 제공합니다. 이를 대체하여 감정 분석이 더 정확한지 덜 정확한지 확인할 수 있습니다. 여기서는 VADER 감정 분석을 사용합니다.

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

프로그램에서 감정을 계산할 준비가 되었을 때, 각 리뷰에 다음과 같이 적용할 수 있습니다:

```python
# Add a negative sentiment and positive sentiment column
print("Calculating sentiment columns for both positive and negative reviews")
start = time.time()
df["Negative_Sentiment"] = df.Negative_Review.apply(calc_sentiment)
df["Positive_Sentiment"] = df.Positive_Review.apply(calc_sentiment)
end = time.time()
print("Calculating sentiment took " + str(round(end - start, 2)) + " seconds")
```

제 컴퓨터에서는 약 120초가 소요되지만, 컴퓨터마다 다를 수 있습니다. 결과를 출력하여 감정이 리뷰와 일치하는지 확인하려면:

```python
df = df.sort_values(by=["Negative_Sentiment"], ascending=True)
print(df[["Negative_Review", "Negative_Sentiment"]])
df = df.sort_values(by=["Positive_Sentiment"], ascending=True)
print(df[["Positive_Review", "Positive_Sentiment"]])
```

파일을 도전에 사용하기 전에 마지막으로 해야 할 일은 저장하는 것입니다! 또한, 새로 추가된 모든 열을 재정렬하여 작업하기 쉽게 만드는 것도 고려해야 합니다(사람이 보기 좋게 하기 위한 미적 변경입니다).

```python
# Reorder the columns (This is cosmetic, but to make it easier to explore the data later)
df = df.reindex(["Hotel_Name", "Hotel_Address", "Total_Number_of_Reviews", "Average_Score", "Reviewer_Score", "Negative_Sentiment", "Positive_Sentiment", "Reviewer_Nationality", "Leisure_trip", "Couple", "Solo_traveler", "Business_trip", "Group", "Family_with_young_children", "Family_with_older_children", "With_a_pet", "Negative_Review", "Positive_Review"], axis=1)

print("Saving results to Hotel_Reviews_NLP.csv")
df.to_csv(r"../data/Hotel_Reviews_NLP.csv", index = False)
```

[분석 노트북](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/3-notebook.ipynb)의 전체 코드를 실행해야 합니다(먼저 [필터링 노트북](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/1-notebook.ipynb)을 실행하여 Hotel_Reviews_Filtered.csv 파일을 생성한 후).

요약하자면, 단계는 다음과 같습니다:

1. 원본 데이터셋 파일 **Hotel_Reviews.csv**는 이전 강의에서 [탐색 노트북](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/4-Hotel-Reviews-1/solution/notebook.ipynb)을 사용해 탐색되었습니다.
2. **Hotel_Reviews.csv**는 [필터링 노트북](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/1-notebook.ipynb)을 통해 필터링되어 **Hotel_Reviews_Filtered.csv**가 생성되었습니다.
3. **Hotel_Reviews_Filtered.csv**는 [감정 분석 노트북](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/3-notebook.ipynb)을 통해 처리되어 **Hotel_Reviews_NLP.csv**가 생성되었습니다.
4. 아래 NLP 도전 과제에서 **Hotel_Reviews_NLP.csv**를 사용합니다.

### 결론

처음에는 열과 데이터가 있는 데이터셋을 가지고 있었지만, 모든 데이터를 확인하거나 사용할 수는 없었습니다. 데이터를 탐색하고, 필요 없는 데이터를 필터링하고, 태그를 유용한 것으로 변환하고, 자체 평균을 계산하고, 감정 열을 추가했으며, 자연어 텍스트를 처리하는 데 있어 흥미로운 점을 배웠기를 바랍니다.

## [강의 후 퀴즈](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/40/)

## 도전 과제

이제 데이터셋의 감정을 분석했으니, 이 커리큘럼에서 배운 전략(예: 클러스터링)을 사용하여 감정과 관련된 패턴을 찾아보세요.

## 복습 및 자기 학습

[이 학습 모듈](https://docs.microsoft.com/en-us/learn/modules/classify-user-feedback-with-the-text-analytics-api/?WT.mc_id=academic-77952-leestott)을 통해 텍스트에서 감정을 탐색하고 다양한 도구를 사용하는 방법을 더 배워보세요.

## 과제

[다른 데이터셋 시도하기](assignment.md)

---

**면책 조항**:  
이 문서는 AI 번역 서비스 [Co-op Translator](https://github.com/Azure/co-op-translator)를 사용하여 번역되었습니다. 정확성을 위해 최선을 다하고 있지만, 자동 번역에는 오류나 부정확성이 포함될 수 있습니다. 원본 문서를 해당 언어로 작성된 상태에서 권위 있는 자료로 간주해야 합니다. 중요한 정보의 경우, 전문 번역가에 의한 번역을 권장합니다. 이 번역 사용으로 인해 발생하는 오해나 잘못된 해석에 대해 당사는 책임을 지지 않습니다.  