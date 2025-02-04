# 호텔 리뷰를 통한 감정 분석

이제 데이터셋을 자세히 탐색했으니, 열을 필터링하고 NLP 기법을 사용하여 호텔에 대한 새로운 통찰을 얻을 때입니다.
## [강의 전 퀴즈](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/39/)

### 필터링 및 감정 분석 작업

아마도 데이터셋에 몇 가지 문제가 있다는 것을 눈치챘을 것입니다. 일부 열은 쓸모없는 정보로 채워져 있고, 다른 열은 잘못된 것처럼 보입니다. 만약 그 열이 맞다고 하더라도, 그것들이 어떻게 계산되었는지 명확하지 않으며, 자신의 계산으로 독립적으로 검증할 수 없습니다.

## 연습: 데이터 처리 조금 더 하기

데이터를 조금 더 정리하세요. 나중에 유용할 열을 추가하고, 다른 열의 값을 변경하고, 특정 열을 완전히 삭제하세요.

1. 초기 열 처리

   1. `lat` 및 `lng` 삭제

   2. `Hotel_Address` 값을 다음 값으로 대체하세요 (주소에 도시와 국가가 포함되어 있다면, 도시와 국가만 남기세요).

      데이터셋에 있는 유일한 도시와 국가는 다음과 같습니다:

      암스테르담, 네덜란드

      바르셀로나, 스페인

      런던, 영국

      밀라노, 이탈리아

      파리, 프랑스

      비엔나, 오스트리아 

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

      이제 국가 수준의 데이터를 쿼리할 수 있습니다:

      ```python
      display(df.groupby("Hotel_Address").agg({"Hotel_Name": "nunique"}))
      ```

      | 호텔 주소               | 호텔 이름 |
      | :--------------------- | :--------: |
      | 암스테르담, 네덜란드   |    105     |
      | 바르셀로나, 스페인     |    211     |
      | 런던, 영국             |    400     |
      | 밀라노, 이탈리아       |    162     |
      | 파리, 프랑스           |    458     |
      | 비엔나, 오스트리아     |    158     |

2. 호텔 메타 리뷰 열 처리

  1. `Additional_Number_of_Scoring`

  1. Replace `Total_Number_of_Reviews` with the total number of reviews for that hotel that are actually in the dataset 

  1. Replace `Average_Score` 삭제하고 직접 계산한 점수로 대체

  ```python
  # Drop `Additional_Number_of_Scoring`
  df.drop(["Additional_Number_of_Scoring"], axis = 1, inplace=True)
  # Replace `Total_Number_of_Reviews` and `Average_Score` with our own calculated values
  df.Total_Number_of_Reviews = df.groupby('Hotel_Name').transform('count')
  df.Average_Score = round(df.groupby('Hotel_Name').Reviewer_Score.transform('mean'), 1)
  ```

3. 리뷰 열 처리

   1. `Review_Total_Negative_Word_Counts`, `Review_Total_Positive_Word_Counts`, `Review_Date` and `days_since_review`

   2. Keep `Reviewer_Score`, `Negative_Review`, and `Positive_Review` as they are,
     
   3. Keep `Tags` for now

     - We'll be doing some additional filtering operations on the tags in the next section and then tags will be dropped

4. Process reviewer columns

  1. Drop `Total_Number_of_Reviews_Reviewer_Has_Given`
  
  2. Keep `Reviewer_Nationality`

### Tag columns

The `Tag` column is problematic as it is a list (in text form) stored in the column. Unfortunately the order and number of sub sections in this column are not always the same. It's hard for a human to identify the correct phrases to be interested in, because there are 515,000 rows, and 1427 hotels, and each has slightly different options a reviewer could choose. This is where NLP shines. You can scan the text and find the most common phrases, and count them.

Unfortunately, we are not interested in single words, but multi-word phrases (e.g. *Business trip*). Running a multi-word frequency distribution algorithm on that much data (6762646 words) could take an extraordinary amount of time, but without looking at the data, it would seem that is a necessary expense. This is where exploratory data analysis comes in useful, because you've seen a sample of the tags such as `[' Business trip  ', ' Solo traveler ', ' Single Room ', ' Stayed 5 nights ', ' Submitted from  a mobile device ']` 삭제하고, 관심 있는 태그를 확인하기 위해 몇 가지 단계를 따르세요.

### 태그 필터링

데이터셋의 목표는 감정을 추가하고 최종 데이터셋에서 유용한 열을 추가하여 최고의 호텔을 선택하는 데 도움을 주는 것입니다 (자신을 위해서나 호텔 추천 봇을 만들기 위해 클라이언트가 요청한 경우). 태그가 최종 데이터셋에서 유용한지 아닌지 스스로에게 물어봐야 합니다. 다음은 한 가지 해석입니다 (다른 이유로 데이터셋이 필요하다면 다른 태그가 선택에 남거나 제외될 수 있습니다):

1. 여행 유형은 관련이 있으며, 유지해야 합니다.
2. 게스트 그룹 유형은 중요하며, 유지해야 합니다.
3. 게스트가 머문 방, 스위트룸, 스튜디오 유형은 무관합니다 (모든 호텔에 기본적으로 동일한 방이 있습니다).
4. 리뷰가 제출된 장치는 무관합니다.
5. 리뷰어가 머문 밤 수는 *관련*이 있을 수 있습니다. 더 긴 숙박이 호텔을 더 좋아하는 것과 관련이 있다고 가정할 수 있지만, 이는 다소 무관할 수 있습니다.

요약하자면, **두 가지 종류의 태그를 유지하고 나머지는 제거하세요**.

먼저, 태그가 더 나은 형식으로 변환될 때까지 태그를 계산하지 않으려면 대괄호와 따옴표를 제거해야 합니다. 여러 가지 방법이 있지만, 데이터 처리 시간이 오래 걸릴 수 있으므로 가장 빠른 방법을 원합니다. 다행히도, 판다는 이러한 각 단계를 쉽게 수행할 수 있는 방법을 제공합니다.

```Python
# Remove opening and closing brackets
df.Tags = df.Tags.str.strip("[']")
# remove all quotes too
df.Tags = df.Tags.str.replace(" ', '", ",", regex = False)
```

각 태그는 다음과 같이 됩니다: `Business trip, Solo traveler, Single Room, Stayed 5 nights, Submitted from a mobile device`. 

Next we find a problem. Some reviews, or rows, have 5 columns, some 3, some 6. This is a result of how the dataset was created, and hard to fix. You want to get a frequency count of each phrase, but they are in different order in each review, so the count might be off, and a hotel might not get a tag assigned to it that it deserved.

Instead you will use the different order to our advantage, because each tag is multi-word but also separated by a comma! The simplest way to do this is to create 6 temporary columns with each tag inserted in to the column corresponding to its order in the tag. You can then merge the 6 columns into one big column and run the `value_counts()` method on the resulting column. Printing that out, you'll see there was 2428 unique tags. Here is a small sample:

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

Some of the common tags like `Submitted from a mobile device` are of no use to us, so it might be a smart thing to remove them before counting phrase occurrence, but it is such a fast operation you can leave them in and ignore them.

### Removing the length of stay tags

Removing these tags is step 1, it reduces the total number of tags to be considered slightly. Note you do not remove them from the dataset, just choose to remove them from consideration as values to  count/keep in the reviews dataset.

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

There are a huge variety of rooms, suites, studios, apartments and so on. They all mean roughly the same thing and not relevant to you, so remove them from consideration.

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

Finally, and this is delightful (because it didn't take much processing at all), you will be left with the following *useful* tags:

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

You could argue that `Travellers with friends` is the same as `Group` more or less, and that would be fair to combine the two as above. The code for identifying the correct tags is [the Tags notebook](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/1-notebook.ipynb).

The final step is to create new columns for each of these tags. Then, for every review row, if the `Tag` 열이 새 열 중 하나와 일치하면 1을 추가하고, 그렇지 않으면 0을 추가합니다. 최종 결과는 비즈니스 vs 레저 또는 애완동물을 데리고 오는 등의 이유로 이 호텔을 선택한 리뷰어의 수를 집계한 것입니다. 이는 호텔을 추천할 때 유용한 정보입니다.

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

이제 이전 섹션에서 저장한 필터링된 데이터셋을 로드하고, **원본 데이터셋이 아님**을 주의하세요.

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

부정적 리뷰와 긍정적 리뷰 열에서 감정 분석을 수행하려면 시간이 오래 걸릴 수 있습니다. 빠른 CPU를 가진 강력한 테스트 노트북에서 테스트한 결과, 사용된 감정 라이브러리에 따라 12-14분이 걸렸습니다. 이는 (상대적으로) 긴 시간이므로, 이를 빠르게 할 수 있는지 조사할 가치가 있습니다.

불용어, 즉 문장의 감정을 바꾸지 않는 일반적인 영어 단어를 제거하는 것이 첫 번째 단계입니다. 불용어를 제거하면 감정 분석이 더 빨리 실행되지만, 정확도가 떨어지지 않습니다 (불용어는 감정에 영향을 미치지 않지만, 분석을 느리게 만듭니다).

가장 긴 부정적 리뷰는 395단어였지만, 불용어를 제거한 후에는 195단어입니다.

불용어를 제거하는 작업도 빠른 작업이며, 515,000개의 행에서 2개의 리뷰 열에서 불용어를 제거하는 데 테스트 장치에서 3.3초가 걸렸습니다. 장치의 CPU 속도, RAM, SSD 여부 등 여러 요인에 따라 시간이 약간 더 걸리거나 덜 걸릴 수 있습니다. 작업이 상대적으로 짧으므로, 감정 분석 시간을 개선할 수 있다면 수행할 가치가 있습니다.

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

이제 부정적 리뷰와 긍정적 리뷰 열에 대해 감정 분석을 계산하고 결과를 두 개의 새로운 열에 저장해야 합니다. 감정의 테스트는 같은 리뷰에 대해 리뷰어의 점수와 비교하는 것입니다. 예를 들어, 감정 분석이 부정적 리뷰에서 1 (매우 긍정적인 감정)을 나타내고 긍정적 리뷰에서 1을 나타내지만, 리뷰어가 호텔에 가능한 최저 점수를 준다면, 리뷰 텍스트가 점수와 일치하지 않거나 감정 분석기가 감정을 올바르게 인식하지 못했음을 나타냅니다. 일부 감정 점수가 완전히 잘못될 것으로 예상해야 하며, 종종 설명할 수 있습니다. 예를 들어, 리뷰가 매우 비꼬는 "물론 난 난방이 없는 방에서 자는 걸 정말 좋아했어요"와 같이 작성되었고, 감정 분석기가 이를 긍정적인 감정으로 인식할 수 있지만, 사람이 읽으면 비꼬는 것임을 알 수 있습니다.

NLTK는 학습을 위한 다양한 감정 분석기를 제공하며, 이를 대체하여 감정이 더 정확한지 덜 정확한지 확인할 수 있습니다. 여기서는 VADER 감정 분석이 사용되었습니다.

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

프로그램에서 감정을 계산할 준비가 되었을 때, 다음과 같이 각 리뷰에 적용할 수 있습니다:

```python
# Add a negative sentiment and positive sentiment column
print("Calculating sentiment columns for both positive and negative reviews")
start = time.time()
df["Negative_Sentiment"] = df.Negative_Review.apply(calc_sentiment)
df["Positive_Sentiment"] = df.Positive_Review.apply(calc_sentiment)
end = time.time()
print("Calculating sentiment took " + str(round(end - start, 2)) + " seconds")
```

이 작업은 제 컴퓨터에서 약 120초가 걸리지만, 각 컴퓨터마다 다를 수 있습니다. 결과를 출력하여 감정이 리뷰와 일치하는지 확인하려면:

```python
df = df.sort_values(by=["Negative_Sentiment"], ascending=True)
print(df[["Negative_Review", "Negative_Sentiment"]])
df = df.sort_values(by=["Positive_Sentiment"], ascending=True)
print(df[["Positive_Review", "Positive_Sentiment"]])
```

파일을 사용하기 전에 마지막으로 해야 할 일은 저장하는 것입니다! 또한 모든 새로운 열을 다시 정렬하여 작업하기 쉽게 만드세요 (사람에게는 미용적 변경입니다).

```python
# Reorder the columns (This is cosmetic, but to make it easier to explore the data later)
df = df.reindex(["Hotel_Name", "Hotel_Address", "Total_Number_of_Reviews", "Average_Score", "Reviewer_Score", "Negative_Sentiment", "Positive_Sentiment", "Reviewer_Nationality", "Leisure_trip", "Couple", "Solo_traveler", "Business_trip", "Group", "Family_with_young_children", "Family_with_older_children", "With_a_pet", "Negative_Review", "Positive_Review"], axis=1)

print("Saving results to Hotel_Reviews_NLP.csv")
df.to_csv(r"../data/Hotel_Reviews_NLP.csv", index = False)
```

전체 코드를 실행하여 [분석 노트북](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/3-notebook.ipynb)을 실행하세요 (필터링 노트북을 실행하여 Hotel_Reviews_Filtered.csv 파일을 생성한 후).

복습을 위해, 단계는 다음과 같습니다:

1. 원본 데이터셋 파일 **Hotel_Reviews.csv**는 이전 강의에서 [탐색 노트북](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/4-Hotel-Reviews-1/solution/notebook.ipynb)으로 탐색되었습니다.
2. Hotel_Reviews.csv는 [필터링 노트북](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/1-notebook.ipynb)에 의해 필터링되어 **Hotel_Reviews_Filtered.csv**가 생성되었습니다.
3. Hotel_Reviews_Filtered.csv는 [감정 분석 노트북](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/3-notebook.ipynb)에 의해 처리되어 **Hotel_Reviews_NLP.csv**가 생성되었습니다.
4. 아래의 NLP 챌린지에서 Hotel_Reviews_NLP.csv를 사용하세요.

### 결론

처음 시작할 때, 검증하거나 사용할 수 없는 열과 데이터가 포함된 데이터셋이 있었습니다. 데이터를 탐색하고, 필요 없는 것을 필터링하고, 태그를 유용한 것으로 변환하고, 자신의 평균을 계산하고, 감정 열을 추가하여 자연어 텍스트 처리에 대해 흥미로운 것을 배웠기를 바랍니다.

## [강의 후 퀴즈](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/40/)

## 챌린지

이제 데이터셋의 감정 분석이 완료되었으니, 이 커리큘럼에서 배운 전략(예를 들어 클러스터링)을 사용하여 감정 주위의 패턴을 결정할 수 있는지 확인하세요.

## 복습 및 자습

[이 Learn 모듈](https://docs.microsoft.com/en-us/learn/modules/classify-user-feedback-with-the-text-analytics-api/?WT.mc_id=academic-77952-leestott)을 통해 더 많은 것을 배우고, 텍스트에서 감정을 탐색하는 데 다양한 도구를 사용해 보세요.
## 과제 

[다른 데이터셋 시도해보기](assignment.md)

**면책 조항**:
이 문서는 기계 기반 AI 번역 서비스를 사용하여 번역되었습니다. 정확성을 위해 노력하고 있지만, 자동 번역에는 오류나 부정확성이 있을 수 있습니다. 원어로 작성된 원본 문서를 권위 있는 자료로 간주해야 합니다. 중요한 정보의 경우, 전문 인간 번역을 권장합니다. 이 번역 사용으로 인해 발생하는 오해나 잘못된 해석에 대해 당사는 책임을 지지 않습니다.