# 호텔 리뷰를 통한 감정 분석 - 데이터 처리

이 섹션에서는 이전 강의에서 배운 기술들을 사용하여 대규모 데이터셋을 탐색적으로 분석할 것입니다. 다양한 열의 유용성을 잘 이해한 후에는 다음을 배우게 됩니다:

- 불필요한 열을 제거하는 방법
- 기존 열을 기반으로 새로운 데이터를 계산하는 방법
- 최종 도전에 사용할 결과 데이터셋을 저장하는 방법

## [강의 전 퀴즈](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/37/)

### 소개

지금까지 텍스트 데이터가 숫자 데이터와는 상당히 다르다는 것을 배웠습니다. 사람이 작성하거나 말한 텍스트는 패턴과 빈도, 감정 및 의미를 분석할 수 있습니다. 이번 강의에서는 실제 데이터셋과 실제 도전을 다루게 됩니다: **[유럽의 515K 호텔 리뷰 데이터](https://www.kaggle.com/jiashenliu/515k-hotel-reviews-data-in-europe)** 및 [CC0: Public Domain 라이선스](https://creativecommons.org/publicdomain/zero/1.0/)가 포함되어 있습니다. 이 데이터는 Booking.com에서 공개 소스를 통해 수집되었습니다. 데이터셋의 작성자는 Jiashen Liu입니다.

### 준비

다음이 필요합니다:

* Python 3을 사용하여 .ipynb 노트북을 실행할 수 있는 능력
* pandas
* NLTK, [로컬에 설치해야 합니다](https://www.nltk.org/install.html)
* Kaggle에서 사용할 수 있는 데이터셋 [유럽의 515K 호텔 리뷰 데이터](https://www.kaggle.com/jiashenliu/515k-hotel-reviews-data-in-europe). 압축을 풀면 약 230 MB입니다. 이를 NLP 강의와 관련된 루트 `/data` 폴더에 다운로드하십시오.

## 탐색적 데이터 분석

이 도전 과제는 감정 분석과 손님 리뷰 점수를 사용하여 호텔 추천 봇을 구축하는 것을 가정합니다. 사용할 데이터셋에는 6개 도시의 1493개 호텔에 대한 리뷰가 포함되어 있습니다.

Python, 호텔 리뷰 데이터셋 및 NLTK의 감정 분석을 사용하여 다음을 알아낼 수 있습니다:

* 리뷰에서 가장 자주 사용되는 단어와 구는 무엇인가?
* 호텔을 설명하는 공식 *태그*가 리뷰 점수와 상관관계가 있는가? (예: 특정 호텔에 대한 더 부정적인 리뷰가 *어린 자녀를 둔 가족*보다 *솔로 여행자*에게 더 많은가? 이는 *솔로 여행자*에게 더 적합하다는 것을 나타낼 수 있습니다.)
* NLTK 감정 점수가 호텔 리뷰어의 숫자 점수와 '일치'하는가?

#### 데이터셋

다운로드하여 로컬에 저장한 데이터셋을 탐색해 봅시다. VS Code나 Excel과 같은 편집기에서 파일을 열어보세요.

데이터셋의 헤더는 다음과 같습니다:

*Hotel_Address, Additional_Number_of_Scoring, Review_Date, Average_Score, Hotel_Name, Reviewer_Nationality, Negative_Review, Review_Total_Negative_Word_Counts, Total_Number_of_Reviews, Positive_Review, Review_Total_Positive_Word_Counts, Total_Number_of_Reviews_Reviewer_Has_Given, Reviewer_Score, Tags, days_since_review, lat, lng*

여기에는 더 쉽게 검토할 수 있도록 그룹화되어 있습니다: 
##### 호텔 열

* `Hotel_Name`, `Hotel_Address`, `lat` (위도), `lng` (경도)
  * *lat*과 *lng*를 사용하여 Python으로 호텔 위치를 보여주는 지도를 그릴 수 있습니다 (부정적 및 긍정적 리뷰에 따라 색상을 구분할 수 있음)
  * Hotel_Address는 우리에게 명확히 유용하지 않으며, 더 쉬운 정렬 및 검색을 위해 국가로 대체할 가능성이 큽니다.

**호텔 메타 리뷰 열**

* `Average_Score`
  * 데이터셋 작성자에 따르면, 이 열은 *지난 1년 동안의 최신 댓글을 기반으로 계산된 호텔의 평균 점수*입니다. 이는 점수를 계산하는 독특한 방법처럼 보이지만, 지금은 데이터를 액면 그대로 받아들여야 합니다.
  
  ✅ 이 데이터의 다른 열을 기반으로 평균 점수를 계산할 다른 방법을 생각해 볼 수 있나요?

* `Total_Number_of_Reviews`
  * 이 호텔이 받은 총 리뷰 수 - 코드 작성 없이 이 데이터셋의 리뷰를 의미하는지 명확하지 않습니다.
* `Additional_Number_of_Scoring`
  * 이는 리뷰어가 긍정적 또는 부정적 리뷰를 작성하지 않았지만 점수를 부여한 것을 의미합니다.

**리뷰 열**

- `Reviewer_Score`
  - 이는 최소 1자리 소수점으로 된 숫자 값이며, 최소값과 최대값은 2.5와 10 사이입니다.
  - 2.5가 가능한 최저 점수인 이유는 설명되지 않았습니다.
- `Negative_Review`
  - 리뷰어가 아무것도 작성하지 않았다면, 이 필드는 "**No Negative**"로 표시됩니다.
  - 리뷰어가 부정적 리뷰 칸에 긍정적 리뷰를 작성할 수도 있습니다 (예: "이 호텔에 나쁜 점이 없습니다").
- `Review_Total_Negative_Word_Counts`
  - 부정적 단어 수가 많을수록 점수가 낮아집니다 (감정을 확인하지 않은 경우).
- `Positive_Review`
  - 리뷰어가 아무것도 작성하지 않았다면, 이 필드는 "**No Positive**"로 표시됩니다.
  - 리뷰어가 긍정적 리뷰 칸에 부정적 리뷰를 작성할 수도 있습니다 (예: "이 호텔에는 좋은 점이 전혀 없습니다").
- `Review_Total_Positive_Word_Counts`
  - 긍정적 단어 수가 많을수록 점수가 높아집니다 (감정을 확인하지 않은 경우).
- `Review_Date` 및 `days_since_review`
  - 신선도 또는 오래된 리뷰에 대한 측정을 적용할 수 있습니다 (호텔 관리가 변경되었거나, 리모델링이 이루어졌거나, 수영장이 추가되었기 때문에 오래된 리뷰는 정확하지 않을 수 있음).
- `Tags`
  - 이는 리뷰어가 자신이 어떤 유형의 손님이었는지, 어떤 유형의 방을 가졌는지, 체류 기간, 리뷰를 제출한 방법 등을 설명하기 위해 선택할 수 있는 짧은 설명입니다.
  - 불행히도, 이러한 태그를 사용하는 것은 문제가 있으며, 아래 섹션에서 그 유용성에 대해 논의합니다.

**리뷰어 열**

- `Total_Number_of_Reviews_Reviewer_Has_Given`
  - 이는 추천 모델에서 요인이 될 수 있습니다. 예를 들어, 수백 개의 리뷰를 작성한 더 많은 리뷰어가 부정적 리뷰를 남길 가능성이 더 높다는 것을 알 수 있다면 유용할 수 있습니다. 그러나 특정 리뷰의 리뷰어는 고유 코드로 식별되지 않으므로 리뷰 세트와 연결할 수 없습니다. 100개 이상의 리뷰를 작성한 리뷰어가 30명 있지만, 이것이 추천 모델에 어떻게 도움이 되는지 보기 어렵습니다.
- `Reviewer_Nationality`
  - 일부 사람들은 특정 국적이 국가적 성향 때문에 긍정적 또는 부정적 리뷰를 남길 가능성이 더 높다고 생각할 수 있습니다. 이러한 일화적 견해를 모델에 포함하는 데 주의해야 합니다. 이는 국가적(때로는 인종적) 고정관념이며, 각 리뷰어는 자신의 경험을 바탕으로 리뷰를 작성한 개인입니다. 그들의 국적이 리뷰 점수의 이유였다고 생각하는 것은 정당화하기 어렵습니다.

##### 예시

| 평균 점수 | 총 리뷰 수 | 리뷰어 점수 | 부정적 리뷰                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  | 긍정적 리뷰                 | 태그                                                                                      |
| -------------- | ---------------------- | ---------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------- | ----------------------------------------------------------------------------------------- |
| 7.8            | 1945                   | 2.5              | This is  currently not a hotel but a construction site I was terrorized from early  morning and all day with unacceptable building noise while resting after a  long trip and working in the room People were working all day i e with  jackhammers in the adjacent rooms I asked for a room change but no silent  room was available To make things worse I was overcharged I checked out in  the evening since I had to leave very early flight and received an appropriate  bill A day later the hotel made another charge without my consent in excess  of booked price It's a terrible place Don't punish yourself by booking  here | Nothing  Terrible place Stay away | Business trip                                Couple Standard Double  Room Stayed 2 nights |

보시다시피, 이 손님은 이 호텔에서 행복한 체류를 하지 않았습니다. 이 호텔은 7.8의 좋은 평균 점수와 1945개의 리뷰를 가지고 있지만, 이 리뷰어는 2.5를 주고 그들의 체류가 얼마나 부정적이었는지에 대해 115단어를 작성했습니다. 긍정적 리뷰 칸에 아무것도 작성하지 않았다면 긍정적인 것이 없다고 추측할 수 있지만, 그들은 7단어로 경고를 작성했습니다. 단어의 의미나 감정을 고려하지 않고 단어 수만 센다면, 리뷰어의 의도를 왜곡할 수 있습니다. 이상하게도, 그들의 점수 2.5는 혼란스럽습니다. 호텔 체류가 그렇게 나빴다면 왜 점수를 주었을까요? 데이터셋을 자세히 조사하면 가능한 최저 점수가 2.5이며, 0이 아니라는 것을 알 수 있습니다. 가능한 최고 점수는 10입니다.

##### 태그

위에서 언급했듯이, 처음에는 `Tags`을 사용하여 데이터를 분류하는 아이디어가 타당해 보입니다. 불행히도 이러한 태그는 표준화되어 있지 않아서, 특정 호텔에서는 *싱글룸*, *트윈룸*, *더블룸* 옵션이 있을 수 있지만, 다음 호텔에서는 *디럭스 싱글룸*, *클래식 퀸룸*, *이그제큐티브 킹룸* 옵션이 있습니다. 이는 동일한 것일 수 있지만, 너무 많은 변형이 있어 선택이 어려워집니다:

1. 모든 용어를 단일 표준으로 변경하려고 시도하는 것, 이는 각 경우에 변환 경로가 명확하지 않기 때문에 매우 어렵습니다 (예: *클래식 싱글룸*을 *싱글룸*으로 매핑하지만 *코트야드 가든 또는 시티 뷰가 있는 슈페리어 퀸룸*은 매핑하기가 훨씬 어렵습니다).

1. NLP 접근 방식을 사용하여 각 호텔에 적용되는 *솔로*, *비즈니스 여행객*, 또는 *어린 아이가 있는 가족*과 같은 특정 용어의 빈도를 측정하고 이를 추천에 반영할 수 있습니다.

태그는 일반적으로 (항상 그렇지는 않지만) *여행 유형*, *손님 유형*, *방 유형*, *숙박 기간*, 및 *리뷰를 제출한 장치 유형*에 맞추어진 5~6개의 쉼표로 구분된 값을 포함하는 단일 필드입니다. 그러나 일부 리뷰어가 각 필드를 채우지 않는 경우 (하나를 비워둘 수 있음), 값은 항상 같은 순서로 제공되지 않습니다.

예를 들어 *그룹 유형*을 보세요. `Tags` 열에서 이 필드에는 1025개의 고유 가능성이 있으며, 불행히도 그 중 일부만 그룹을 참조합니다 (일부는 방 유형 등입니다). 가족을 언급하는 것만 필터링하면 많은 *가족 방* 유형의 결과가 포함됩니다. *with* 용어를 포함하면, 즉 *가족과 함께* 값을 계산하면, "어린 자녀를 둔 가족" 또는 "나이 든 자녀를 둔 가족" 문구가 포함된 80,000개 이상의 515,000개 결과가 나옵니다.

이것은 태그 열이 완전히 쓸모없지는 않지만, 유용하게 만들기 위해서는 약간의 작업이 필요함을 의미합니다.

##### 호텔 평균 점수

데이터셋에는 이해하기 어렵지만 모델을 구축할 때 인식해야 하는 몇 가지 이상 현상 또는 불일치가 있습니다. 이해하게 되면 토론 섹션에서 알려주세요!

데이터셋에는 평균 점수 및 리뷰 수와 관련된 다음 열이 있습니다:

1. Hotel_Name
2. Additional_Number_of_Scoring
3. Average_Score
4. Total_Number_of_Reviews
5. Reviewer_Score  

이 데이터셋에서 가장 많은 리뷰를 가진 단일 호텔은 *Britannia International Hotel Canary Wharf*로 515,000개의 리뷰 중 4789개의 리뷰를 가지고 있습니다. 그러나 `Total_Number_of_Reviews` 값이 9086인 이 호텔을 보면, 리뷰 없이 점수만 있는 경우가 많이 있을 수 있다고 추측할 수 있습니다. `Additional_Number_of_Scoring` 열 값을 추가해야 할 수도 있습니다. 그 값은 2682이며, 4789에 추가하면 7471이 되어 `Total_Number_of_Reviews`에서 여전히 1615 부족합니다.

`Average_Score` 열을 보면, 데이터셋의 리뷰 평균일 수 있다고 추측할 수 있지만, Kaggle의 설명은 "*지난 1년 동안의 최신 댓글을 기반으로 계산된 호텔의 평균 점수*"입니다. 이는 그다지 유용해 보이지 않지만, 데이터셋의 리뷰 점수를 기반으로 자체 평균을 계산할 수 있습니다. 동일한 호텔을 예로 들면, 호텔 평균 점수는 7.1로 주어졌지만, 데이터셋에서 계산된 점수 (리뷰어 점수의 평균)는 6.8입니다. 이는 비슷하지만 동일한 값은 아니며, `Additional_Number_of_Scoring` 리뷰에서 제공된 점수가 평균을 7.1로 올렸다고 추측할 수 있습니다. 그러나 이를 테스트하거나 증명할 방법이 없으므로, `Average_Score`, `Additional_Number_of_Scoring` 및 `Total_Number_of_Reviews`를 사용하는 것이 어렵습니다.

더 복잡하게도, 두 번째로 많은 리뷰를 가진 호텔의 계산된 평균 점수는 8.12이고, 데이터셋 `Average_Score`는 8.1입니다. 이 정확한 점수는 우연의 일치인가요, 아니면 첫 번째 호텔의 불일치인가요?

이 호텔이 특이치일 가능성을 고려하여, 대부분의 값이 일치할 수도 있지만 (어떤 이유로 일부는 그렇지 않음) 다음으로 데이터셋의 값을 탐색하고 값의 올바른 사용 (또는 비사용)을 결정하는 짧은 프로그램을 작성할 것입니다.

> 🚨 주의 사항
>
> 이 데이터셋을 작업할 때 텍스트를 읽거나 분석하지 않고도 텍스트에서 무언가를 계산하는 코드를 작성하게 될 것입니다. 이는 NLP의 본질로, 사람이 하지 않고도 의미나 감정을 해석하는 것입니다. 그러나 부정적 리뷰를 읽을 가능성이 있습니다. 그렇게 하지 않기를 권장합니다. 일부는 어리석거나, 호텔의 통제 밖의 것들에 대한 부적절한 부정적 리뷰일 수 있습니다. 예를 들어 "날씨가 좋지 않았다"는 호텔이나 그 누구도 통제할 수 없는 것에 대한 리뷰입니다. 그러나 일부 리뷰는 인종차별적, 성차별적, 연령차별적일 수 있습니다. 이는 불행하지만 공개 웹사이트에서 수집된 데이터셋에서 예상할 수 있는 일입니다. 일부 리뷰어는 불쾌하거나 불편하거나 화나게 할 수 있는 리뷰를 남깁니다. 코드를 통해 감정을 측정하는 것이 좋습니다. 다만, 소수의 사람들이 그러한 리뷰를 남기지만, 여전히 존재합니다.

## 연습 - 데이터 탐색
### 데이터 로드

데이터를 시각적으로 충분히 살펴보았으니 이제 코드를 작성하여 몇 가지 답을 얻어봅시다! 이 섹션에서는 pandas 라이브러리를 사용합니다. 첫 번째 작업은 CSV 데이터를 로드하고 읽을 수 있는지 확인하는 것입니다. pandas 라이브러리는 빠른 CSV 로더를 가지고 있으며, 결과는 이전 강의에서와 같이 데이터프레임에 배치됩니다. 로드할 CSV는 50만 개 이상의 행이 있지만, 열은 17개뿐입니다. pandas는 데이터프레임과 상호 작용할 수 있는 강력한 방법을 많이 제공합니다. 여기에는 모든 행에 대해 연산을 수행하는 기능도 포함됩니다.

이 강의의 이후 부분에서는 코드 스니펫과 코드 설명, 결과에 대한 논의가 포함됩니다. 코드 작성에는 _notebook.ipynb_를 사용하세요.

다음은 사용할 데이터 파일을 로드하는 코드입니다:

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

이제 데이터가 로드되었으니 몇 가지 연산을 수행할 수 있습니다. 이 코드를 프로그램의 상단에 유지하세요.

## 데이터 탐색

이 경우 데이터는 이미 *깨끗*합니다. 즉, 작업할 준비가 되어 있으며, 알고리즘이 영어 문자만 기대하는 다른 언어의 문자가 포함되어 있지 않습니다.

✅ NLP 기술을 적용하기 전에 데이터를 형식화하기 위한 초기 처리가 필요한 데이터를 다룰 수도 있습니다. 그렇다면 비영어 문자를 어떻게 처리하시겠습니까?

데이터가 로드된 후 코드를 사용하여 탐색할 수 있는지 확인하세요. `Negative_Review` 및 `Positive_Review` 열에 집중하고 싶어질 수 있습니다. 이 열들은 NLP 알고리즘이 처리할 자연어 텍스트로 가득 차 있습니다. 그러나 기다리세요! NLP와 감정 분석에 뛰어들기 전에, 아래 코드를 따라 데이터셋에 주어진 값이 pandas로 계산한 값과 일치하는지 확인하세요.

## 데이터프레임 연산

이 강의의 첫 번째 작업은 데이터 프레임을 변경하지 않고 데이터를 검사하는 코드를 작성하여 다음 주장이 올바른지 확인하는 것입니다.

> 많은 프로그래밍 작업과 마찬가지로, 이를 완료하는 여러 가지 방법이 있지만, 가장 간단하고 쉬운 방법으로 하는 것이 좋습니다. 특히 나중에 이 코드를
rows have column `Positive_Review` values of "No Positive" 9. Calculate and print out how many rows have column `Positive_Review` values of "No Positive" **and** `Negative_Review` values of "No Negative" ### Code answers 1. Print out the *shape* of the data frame you have just loaded (the shape is the number of rows and columns) ```python
   print("The shape of the data (rows, cols) is " + str(df.shape))
   > The shape of the data (rows, cols) is (515738, 17)
   ``` 2. Calculate the frequency count for reviewer nationalities: 1. How many distinct values are there for the column `Reviewer_Nationality` and what are they? 2. What reviewer nationality is the most common in the dataset (print country and number of reviews)? ```python
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
   ``` 3. What are the next top 10 most frequently found nationalities, and their frequency count? ```python
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
      ``` 3. What was the most frequently reviewed hotel for each of the top 10 most reviewer nationalities? ```python
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
   ``` 4. How many reviews are there per hotel (frequency count of hotel) in the dataset? ```python
   # First create a new dataframe based on the old one, removing the uneeded columns
   hotel_freq_df = df.drop(["Hotel_Address", "Additional_Number_of_Scoring", "Review_Date", "Average_Score", "Reviewer_Nationality", "Negative_Review", "Review_Total_Negative_Word_Counts", "Positive_Review", "Review_Total_Positive_Word_Counts", "Total_Number_of_Reviews_Reviewer_Has_Given", "Reviewer_Score", "Tags", "days_since_review", "lat", "lng"], axis = 1)
   
   # Group the rows by Hotel_Name, count them and put the result in a new column Total_Reviews_Found
   hotel_freq_df['Total_Reviews_Found'] = hotel_freq_df.groupby('Hotel_Name').transform('count')
   
   # Get rid of all the duplicated rows
   hotel_freq_df = hotel_freq_df.drop_duplicates(subset = ["Hotel_Name"])
   display(hotel_freq_df) 
   ``` | Hotel_Name | Total_Number_of_Reviews | Total_Reviews_Found | | :----------------------------------------: | :---------------------: | :-----------------: | | Britannia International Hotel Canary Wharf | 9086 | 4789 | | Park Plaza Westminster Bridge London | 12158 | 4169 | | Copthorne Tara Hotel London Kensington | 7105 | 3578 | | ... | ... | ... | | Mercure Paris Porte d Orleans | 110 | 10 | | Hotel Wagner | 135 | 10 | | Hotel Gallitzinberg | 173 | 8 | You may notice that the *counted in the dataset* results do not match the value in `Total_Number_of_Reviews`. It is unclear if this value in the dataset represented the total number of reviews the hotel had, but not all were scraped, or some other calculation. `Total_Number_of_Reviews` is not used in the model because of this unclarity. 5. While there is an `Average_Score` column for each hotel in the dataset, you can also calculate an average score (getting the average of all reviewer scores in the dataset for each hotel). Add a new column to your dataframe with the column header `Calc_Average_Score` that contains that calculated average. Print out the columns `Hotel_Name`, `Average_Score`, and `Calc_Average_Score`. ```python
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
   ``` You may also wonder about the `Average_Score` value and why it is sometimes different from the calculated average score. As we can't know why some of the values match, but others have a difference, it's safest in this case to use the review scores that we have to calculate the average ourselves. That said, the differences are usually very small, here are the hotels with the greatest deviation from the dataset average and the calculated average: | Average_Score_Difference | Average_Score | Calc_Average_Score | Hotel_Name | | :----------------------: | :-----------: | :----------------: | ------------------------------------------: | | -0.8 | 7.7 | 8.5 | Best Western Hotel Astoria | | -0.7 | 8.8 | 9.5 | Hotel Stendhal Place Vend me Paris MGallery | | -0.7 | 7.5 | 8.2 | Mercure Paris Porte d Orleans | | -0.7 | 7.9 | 8.6 | Renaissance Paris Vendome Hotel | | -0.5 | 7.0 | 7.5 | Hotel Royal Elys es | | ... | ... | ... | ... | | 0.7 | 7.5 | 6.8 | Mercure Paris Op ra Faubourg Montmartre | | 0.8 | 7.1 | 6.3 | Holiday Inn Paris Montparnasse Pasteur | | 0.9 | 6.8 | 5.9 | Villa Eugenie | | 0.9 | 8.6 | 7.7 | MARQUIS Faubourg St Honor Relais Ch teaux | | 1.3 | 7.2 | 5.9 | Kube Hotel Ice Bar | With only 1 hotel having a difference of score greater than 1, it means we can probably ignore the difference and use the calculated average score. 6. Calculate and print out how many rows have column `Negative_Review` values of "No Negative" 7. Calculate and print out how many rows have column `Positive_Review` values of "No Positive" 8. Calculate and print out how many rows have column `Positive_Review` values of "No Positive" **and** `Negative_Review` values of "No Negative" ```python
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
   ``` ## Another way Another way count items without Lambdas, and use sum to count the rows: ```python
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
   ``` You may have noticed that there are 127 rows that have both "No Negative" and "No Positive" values for the columns `Negative_Review` and `Positive_Review` respectively. That means that the reviewer gave the hotel a numerical score, but declined to write either a positive or negative review. Luckily this is a small amount of rows (127 out of 515738, or 0.02%), so it probably won't skew our model or results in any particular direction, but you might not have expected a data set of reviews to have rows with no reviews, so it's worth exploring the data to discover rows like this. Now that you have explored the dataset, in the next lesson you will filter the data and add some sentiment analysis. --- ## 🚀Challenge This lesson demonstrates, as we saw in previous lessons, how critically important it is to understand your data and its foibles before performing operations on it. Text-based data, in particular, bears careful scrutiny. Dig through various text-heavy datasets and see if you can discover areas that could introduce bias or skewed sentiment into a model. ## [Post-lecture quiz](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/38/) ## Review & Self Study Take [this Learning Path on NLP](https://docs.microsoft.com/learn/paths/explore-natural-language-processing/?WT.mc_id=academic-77952-leestott) to discover tools to try when building speech and text-heavy models. ## Assignment [NLTK](assignment.md)

**면책 조항**:
이 문서는 기계 기반 AI 번역 서비스를 사용하여 번역되었습니다. 정확성을 위해 노력하고 있지만, 자동 번역에는 오류나 부정확성이 있을 수 있음을 유의하시기 바랍니다. 원어로 작성된 원본 문서를 권위 있는 출처로 간주해야 합니다. 중요한 정보의 경우, 전문적인 인간 번역을 권장합니다. 이 번역 사용으로 인해 발생하는 오해나 잘못된 해석에 대해서는 책임지지 않습니다.