# 호텔 리뷰로 감정 분석하기 - 데이터 처리

이 섹션에서는 큰 데이터셋의 탐색적으로 데이터를 분석하는 이전 강의의 기술을 사용할 예정입니다. 다양한 열의 유용성을 잘 이해하면, 배울 수 있습니다:

- 필요하지 않은 열을 제거하는 방식
- 이미 존재하는 열을 기반으로 일부 새로운 데이터를 계산하는 방식
- 최종 도전에서 사용하고자 결과 데이터셋을 저장하는 방식

## [강의 전 퀴즈](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/37/)

### 소개

지금까지 텍스트 데이터가 숫자 데이터 타입과 꽤 다르다는 것을 배웠습니다.  만약 사람이 텍스트를 쓰거나 읽는다면, 패턴과 빈도, 감정과 의미를 찾기 위해서 분석할 수 있습니다. 이 강의에서 실전인 진짜 데이터로 진행합니다:  **[515K Hotel Reviews Data in Europe](https://www.kaggle.com/jiashenliu/515k-hotel-reviews-data-in-europe)** 이며 [CC0: Public Domain license](https://creativecommons.org/publicdomain/zero/1.0/)를 포함합니다. 퍼블릭 소스의 from Booking.com에서 스크랩했습니다. Jiashen Liu가 데이터셋을 생성했습니다.

### 준비

다음 내용이 필요할 예정입니다:

* Python 3로 .ipynb 노트북을 실행할 능력
* pandas
* NLTK, [which you should install locally](https://www.nltk.org/install.html)
* Kaggle [515K Hotel Reviews Data in Europe](https://www.kaggle.com/jiashenliu/515k-hotel-reviews-data-in-europe)에서 사용할 데이터셋. 압축 풀면 230 MB 입니다. NLP 강의와 관련있는 상단 `/data` 폴더에 내려받습니다.

## 탐색적 데이터 분석

이 도전은 감정 분석과 게스트 리뷰 점수로 호텔 추천 봇을 만든다고 가정합니다. 데이터셋은 6개 도시에 있는 1493개 다른 호텔의 리뷰를 포함해서 사용합니다.

호델 리뷰 데이터셋을 Python으로, NLTK 감정 분석을 확인할 수 있습니다:

* 리뷰에서 가장 빈번하게 사용된 단어와 구문은 무엇인가요?
* 호텔을 설명하는 공식 *tags*는 리뷰 점수와 관계가 있나요? (예시로, *Solo traveller*보다 *Family with young children*으로 왔을 때 호텔 리뷰가 더 부정적이고, 어쩌면 *Solo travellers*가 더 괜찮나요?)
* NLTK 감정 점수가 호텔 리뷰어의 숫자로만 이루어진 점수를 '동의'할 수 있나요?

#### 데이터셋

내려받아서 로컬에 저장한 데이터셋을 알아보겠습니다. VS Code나 엑셀 같은 에디터로 파일을 엽니다.

데이터셋의 헤더는 다음과 같습니다:

*Hotel_Address, Additional_Number_of_Scoring, Review_Date, Average_Score, Hotel_Name, Reviewer_Nationality, Negative_Review, Review_Total_Negative_Word_Counts, Total_Number_of_Reviews, Positive_Review, Review_Total_Positive_Word_Counts, Total_Number_of_Reviews_Reviewer_Has_Given, Reviewer_Score, Tags, days_since_review, lat, lng*

여기에 쉽게 풀 수 있는 방식으로 그룹을 묶습니다:

##### Hotel 열

* `Hotel_Name`, `Hotel_Address`, `lat` (latitude), `lng` (longitude)
  * *lat* 과 *lng*으로 Python에서 호텔 위치를 보일 맵을 plot합니다 (아마 부정적이고 긍정적인 리뷰에 대한 색상)
  * Hotel_Address는 분명 우리에게 유용하지 않으므로, 쉽게 분류하고 검색할 수 있게 국가로 대체할 예정입니다

**Hotel Meta-review 열**

* `Average_Score`
  * 데이터셋을 만든 사람에 의하면, 이 열은 *지난 연도에 작성된 최신 코멘트를 기반으로 계산한, 호텔의 평균 점수*입니다. 점수를 계산하는 특이한 방식처럼 보이지만, 현재 액면가로 보일 수 있도록 스크랩한 데이터입니다.

  ✅ 데이터에서 다른 열을 기반으로, 평균 점수를 계산하는 방식을 생각할 수 있나요?

* `Total_Number_of_Reviews`
  * 호텔이 받은 리뷰의 총 개수 - 만약 데이터셋의 리뷰를 나타내는 것이면 (일부 코드 작성없이) 명확하지 않습니다 
* `Additional_Number_of_Scoring`
  * 리뷰 점수가 부여되었지만 리뷰어가 작성한 긍정적이나 부정적인 리뷰가 없다는 점을 의미합니다

**Review columns**

- `Reviewer_Score`
  - 숫자 값은 최소 2.5 최대 값 10 사이의 최대 소수점 1자리 입니다.
  - 2.5가 낮은 숫자인 이유는 없습니다
- `Negative_Review`
  - 만약 리뷰어가 아무것도 작성하지 않았다면, 필드는 "**No Negative**"로 채워집니다
  - 리뷰어가 부정적인 리뷰 열에 긍정적인 리뷰를 남길 수 있다는 점을 참고합니다 (e.g. "there is nothing bad about this hotel")
- `Review_Total_Negative_Word_Counts`
  - 많은 부정적인 단어들은 점수가 낮습니다 (감정을 확인하지 않습니다)
- `Positive_Review`
  - 만약 리뷰어가 아무것도 작성하지 않았다면, 필드는 "**No Positive**"로 채워집니다
  - 리뷰어가 긍정적인 리뷰 열에 부정적인 리뷰를 남길 수 있다는 점을 참고합니다 (e.g. "there is nothing good about this hotel at all")
- `Review_Total_Positive_Word_Counts`
  - 많은 긍정적인 단어들은 점수가 높습니다 (감정을 확인하지 않습니다)
- `Review_Date` and `days_since_review`
  - 최근이나 과거에 작성된 리뷰가 적용될 수 있습니다 (과거에 작성된 리뷰는 호텔 관리가 바뀌었거나, 리노베이션이 끝났거나, 수영장이 생기는 등 최근보다 정확하지 않을 수 있습니다.) 
- `Tags`
  - 리뷰어가 게스트 타입 (예시. 솔로 혹은 패밀리), 지낸 룸 타입, 지낸 기간처럼 작성한 리뷰 방식을 설명하고자 선택할 수 있는 짧은 설명문입니다. 
  - 불행하게도, 이 태그를 사용하는 것은 문제가 있으므로, 아래 usefulness를 설명하는 섹션으로 확인합니다

**리뷰어 열**

- `Total_Number_of_Reviews_Reviewer_Has_Given`
  - 예시로, 수백 개 리뷰를 작성한 다량의 리뷰어가 긍정적보다 부정적으로 리뷰를 남긴다고 판단할 수 있다면, recommendation 모델에서 한 인자가 될 수 있습니다. 그러나, 특정 리뷰를 작성한 리뷰어는 유니크 코드로 식별하지 않으므로, 리뷰 셋에 링크할 수 없습니다. 100개보다 더 작성한 30명 리뷰어가 있지만, recommendation 모델에 어떤 방식으로 도울 수 있는지 보기 힘듭니다.
  
- `Reviewer_Nationality`
  - 일부 사람들은 국가마다 성향이 다르므로 긍정적이나 부정적인 리뷰를 줄 수 있다고 생각합니다. 모델에 사례 증거를 신중히 만듭니다. 국가(인종)에 대한 고정 관념이며, 각 리뷰어는 경험을 기반해서 개인적으로 리뷰를 작성했습니다. 이전에 지낸 호텔, 이동한 거리, 그리고 개인 체질 같은 요인으로 다르게 필터링되었을 수 있습니다. 국적이 리뷰 점수에 대한 이유가 되는 것은 합리적이지 않습니다.

##### 예제

| Average  Score | Total Number   Reviews | Reviewer   Score | Negative <br />Review                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  | Positive   Review                 | Tags                                                                                      |
| -------------- | ---------------------- | ---------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------- | ----------------------------------------------------------------------------------------- |
| 7.8            | 1945                   | 2.5              | This is  currently not a hotel but a construction site I was terroized from early  morning and all day with unacceptable building noise while resting after a  long trip and working in the room People were working all day i e with  jackhammers in the adjacent rooms I asked for a room change but no silent  room was available To make thinks worse I was overcharged I checked out in  the evening since I had to leave very early flight and received an appropiate  bill A day later the hotel made another charge without my concent in excess  of booked price It s a terrible place Don t punish yourself by booking  here | Nothing  Terrible place Stay away | Business trip                                Couple Standard Double  Room Stayed 2 nights |

본 내용처럼, 게스트는 호텔에 즐겁게 지내지 못했습니다. 호텔은 1945개 리뷰 중 7.8 점으로 좋은 평균이지만, 리뷰어는 2.5점을 주며 지냈던 내용을 115개 단어를 부정적으로 남겼습니다. 만약 Positive_Review 열이 깨끗하면, 긍정이 없다고 추측하겠지만, 7개 경고문을 작성했습니다. 만약 단어 의미나 단어 감정 대신 단어 자체를 센다면, 리뷰어 의도를 왜곡해서 볼 수 있습니다. 이상하게, 2.5 점은 혼란스러우며, 만약 호탤애 지낸 게 불쾌하다고, 모든 포인트를 다 주나요? 데이터셋을 가까이 조사하면, 낮을 수 있는 점수가 0점이 아니라, 2.5점이라고 볼 수 있습니다. 가능한 높은 점수는 10점 입니다.

##### Tags

언급된 것처럼, 한 눈에, `Tags`로 데이터를 분류하는 아이디어는 의미있습니다. 불행히, 주어진 호텔에서는, 옵션이  *Single room*, *Twin room*, 그리고 *Double room* 옵션일 수 있지만, 다음 호텔에서, *Deluxe Single Room*, *Classic Queen Room*, 그리고 *Executive King Room*처럼 태그는 표준화되지 못했습니다. 같은 것을 의미할 수 있지만, 너무 많은 옵션이 있으므로 다음을 선택합니다:

1. 각자 케이스에 맞게 바꿀 방식이 명확하지 않으므로, 모든 단어를 단일 표준으로 변경하려는 시도는 매우 어렵습니다. (예시. *Classic single room* 을 *Single room*으로 맵핑하는 것은 가능하지만 *Superior Queen Room with Courtyard Garden or City View*를 맵핑하려면 매우 어렵습니다)

1. NLP 접근 방식을 가지고 각자 호텔에 적용되는 *Solo*, *Business Traveller*, 또는 *Family with young kids* 같은 특정 용어의 빈도를 측정하며, 추천 모델에 인자로 반영합니다 

태그는 일반적으로 (항상 그렇지 않지만) *Type of trip*, *Type of guests*, *Type of room*, *Number of nights*, 그리고 *Type of device review was submitted on* 에 따라서 5에서 6개 컴마로 구분된 값 리스트를 포함한 단일 필드입니다. 그러나, 일부 리뷰어는 각 필드를 채우지 않기 때문에 (하나정도 공백으로 남길 수 있습니다), 값을 동일한 순서로 항상 유지할 수 없습니다.

예시는, *Type of group*으로 합니다. `Tags` 열에 1025개 필드가 서로 안 겹칠 가능성이 있지만, 불행히도 일부만 그룹으로 (일부 방 타입처럼) 참조합니다. 만약 패밀리를 언급한 것만 필터링하면, 많은 *Family room* 타입 결과를 포함해서 냅니다. 만약 *with* 용어를 포함하면, 즉 *Family with* 값을 센다면, 결과는 더 좋게, "Family with young children" 또는 "Family with older children" 문구가 포함된 515,000개 중 80,000개 넘는 결과가 나옵니다.

태그 열을 쓰기에는 완벽하지 않다는 것을 의미하지만, 약간 작업하면 유용해질 수 있습니다.

##### 평균 호텔 점수

이해할 수 없는 데이터셋으로 이상하거나 모순되는게 많지만, 모델을 만들 때 알 수 있는 예시가 있습니다.  만약 알게 된다면, 토론 섹션에 알려주세요!

데이터셋의 평균 점수와 리뷰 수에 관련된 열은 다음과 같습니다:

1. Hotel_Name
2. Additional_Number_of_Scoring
3. Average_Score
4. Total_Number_of_Reviews
5. Reviewer_Score  

이 데이터셋에서 가장 리뷰가 많은 싱글 호텔은 515,000개 중 4789개 리뷰를 받은 *Britannia International Hotel Canary Wharf*입니다. 하지만 만약 호텔의 `Total_Number_of_Reviews` 값을 보면, 9086 입니다. 리뷰없는 점수가 많을 것이라고 추정할 수 있어서, `Additional_Number_of_Scoring` 열 값을 추가할 필요가 있습니다. 이 값은 2682이며, `Total_Number_of_Reviews`의 1615가 부족해져서 4789를 더하면 7,471이 됩니다.

만약 `Average_Score` 열로, 데이터셋에서 리뷰의 평균을 추축할 수 있지만, Kaggle 설명은 "*Average Score of the hotel, calculated based on the latest comment in the last year*"라고 합니다. 유용해보이지 않지만, 데이터셋의 리뷰 점수를 기반해서 평균을 계산할 수 있습니다. 예시로 동일한 호텔을 사용해서, 평균 호텔 점수가 7.1로 주어졌지만 계산된 점수는(데이터셋의 평균 리뷰어 점수) 6.8입니다. 비슷하지만, 같은 값은 아니고, `Additional_Number_of_Scoring` 리뷰에서 주어진 점수가 평균을 7.1 올렸다고 추측할 수 있습니다. 불행히 테스트하거나 확인할 방식은 없으므로, 가지고 있지 않는 것을 기반으로 하거나 참조할 때 `Average_Score`, `Additional_Number_of_Scoring` 과 `Total_Number_of_Reviews`를 쓰거나 신뢰하기 어렵습니다.

더 복잡해지면, 두 번째로 리뷰 수가 높은 호텔은 계산된 평균 점수 8.12이며 데이터셋 `Average_Score`는 8.1입니다. 우연히 점수를 맞추었거나 첫 번째 호텔이 모순되었나요?

이 호텔이 아웃라이어일 수 있고, 아마도 대부분 값을 센다는 가정에 (하지만 일부 이유로 안 합니다) 데이터셋 값을 찾고 값을 올바르게 사용하거나 (또는 사용하지 않게) 결정하는 짧은 프로그램을 작성할 예정입니다.

> 🚨 주의할 사항
>
> 데이터셋으로 작업할 때 텍스트를 수동으로 읽거나 분석하지 않고 텍스트에서 무언가를 계산하는 코드를 작성하게 됩니다. 사람을 거치지 않고 의미나 감정을 분석하는 것은, NLP의 정수입니다. 그러나, 알부 부정적인 리뷰를 읽을 수 있습니다. 직접 할 필요가 없으므로, 하지말라고 권유하고 싶습니다. 일부는 "The weather wasn't great"와 같이, 멍청하거나, 관련없는 부정적 호텔 리뷰이거나, 호텔에서 어떤 컨트롤도 할 수 없는 내용입니다. 하지만 이것도 일부 어두운 측면의 리뷰입니다. 때로 부정적인 리뷰는 racist, sexist, 또는 ageist적 입니다. 불행하지만 퍼블릭 웹사이트에서 긁어온 데이터에서 예상하게 됩니다. 일부 리뷰어는 싫어하거나, 불편하거나, 화낼 수 있는 리뷰를 남깁니다. 읽고 속상한 것보다 코드로 감정을 측정하는 것이 더 좋습니다. 즉, 소수가 작성하지만, 모두 똑같이 있습니다.

## 연습 -  데이터 탐색

### 데이터 불러오기

데이터를 시각적으로 확인하는 것도 충분하므로, 이제 살짝 코드를 작성하고 답변을 얻겠습니다! 이 섹션에서 pandas 라이브러리를 사용합니다. 첫 작업은 CSV 데이터를 불러오고 읽어올 수 있는지 확인하는 작업입니다. pandas 라이브러리는 빠른 CSV 리더이며, 이전 강의에서 했던, dataframe에 결과를 둡니다. 불러올 CSV는 50만 행이 넘지만, 17 열만 있습니다. Pandas는 모든 행에서 작업할 기능을 포함해서, 데이터프레임과 상호 작용하는 강력한 방식을 많이 제공합니다.

이 강의의 이 곳에서, 코드 스니펫과 일부 코드 설명과 일부 결과에 의미를 둔 토론을 하게 됩니다.  코드에 있는 _notebook.ipynb_ 를 사용합니다.

사용할 데이터 파일을 불러오기 위해서 시작합니다:

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

이제 데이터가 불러와지면, 일부 작업을 할 수 있습니다. 다음 파트를 위해서 프로그램의 상단에 코드를 둡니다.

## 데이터 탐색하기

이 케이스에서, 데이터는 이미 *깨끗하므로*, 작업할 준비가 되었고, 영어만 대상으로한 알고리즘에 걸릴 수 있는 다른 언어가 없다는 점을 의미합니다.

✅ NLP 기술을 적용하기 전 포맷에 일부 초기 처리가 필요한 데이터로 작성할 수 있지만, 이번에는 아닙니다. 만약 한다면, 비-영어권 언어를 핸들링 할 수 있나요?

데이터를 불러오는 순간, 코드로 탐색할 수 있는지 시간을 내어 확인합니다. `Negative_Review` 와 `Positive_Review` 열에 집중하기 윈하면 매우 쉽습니다. NLP 알고리즘이 처리할 자연어 텍스트로 채워졌습니다. 하지만 잠깐 기다려봅니다! NLP와 감정으로 들어가기 전, 다음 코드로 만약 데이터셋에 주어진 값이 pandas로 계산한 값과 매치되는지 확실히 봅시다.

## 데이터프레임 작업

이 강의에서 첫 작업은 데이터프레임을 (바꾸지 않고) 점검하는 코드로 작성해서 다음 가정이 올바른지 체크하는 것입니다.

> 많은 프로그래밍 작업처럼, 마무리할 여러 방식이 있지만, 특히 미래에 다시 코드를 작성할 때 이해하기 쉬우려면, 간단하고, 쉽게 할 수 있도록 좋은 조언을 하게 됩니다. 데이터프레임에는, 효율적인 방식으로 자주 할 수 있는 포괄적인 API가 있습니다.

코딩 작업으로 다음 질문을 설명하며 솔루션에 기대지 않고 답변할 수 있게 시도합니다

1. 직전에 불러온 데이터프레임의 *shape*를 출력합니다 (shape는 행과 열의 개수입니다)
2. 리뷰어 국적 대상으로 빈도 카운트를 계산합니다:
   1. `Reviewer_Nationality` 열에 얼마나 많은 별개 값이 있으며 어떤가요?
   2. 데이터셋에서 가장 일반적인 리뷰어 국적은 어디인가요 (국가와 리뷰 개수 출력)?
   3. 다음으로 더 자주 발견되는 국적은 어디고, 빈도를 카운트하나요?
3. 상위 10에 드는 리뷰어 국적에서 가장 자주 리뷰된 호텔은 어디인가요?
4. 데이터셋에서 호텔 별 (호텔의 빈도 개수) 리뷰가 얼마나 있나요?
5. 데이터셋에서 각 호텔마다 `Average_Score` 열이 있지만, 평균 점수도 계산할 수 있습니다 (각 호텔의 데이터셋에서 모든 리뷰어 점수의 평균을 얻습니다). 계산된 평균을 포함한 `Calc_Average_Score` 열 헤더로 데이터프레임에 새로운 열을 추가합니다. 
6. 호텔에서 동일한 (반올림한 소수점 1자리) `Average_Score` 와 `Calc_Average_Score`를 가지고 있나요?
   1. Series (row)로 값을 비교하며, 값이 같지 않을 때 메시지를 출력하는 Python 함수를 작성합니다. 그러면 `.apply()` 메소드를 사용해서 처리하면 함수로 모든 행을 처리할 수 있습니다.
7. "No Negative"의 `Negative_Review` 열 값에 있는 행이 얼마나 있는지 계산하고 출력합니다
8. "No Positive"의 `Positive_Review` 열 값에 있는 행이 얼마나 있는지 계산하고 출력합니다
9. "No Positive"의 `Negative_Review` 열 값**과** "No Negative"의 `Negative_Review` 열 값에 있는 행이 얼마나 있는지 계산하고 출력합니다

### 코드 답변

1. 직전에 불러온 데이터프레임의 *shape*를 출력합니다 (shape는 행과 열의 개수입니다)

   ```python
   print("The shape of the data (rows, cols) is " + str(df.shape))
   > The shape of the data (rows, cols) is (515738, 17)
   ```

2. 리뷰어 국적 대상으로 빈도 카운트를 계산합니다:

   1. `Reviewer_Nationality` 열에 얼마나 많은 별개 값이 있으며 어떤가요?
   2. 데이터셋에서 가장 일반적인 리뷰어 국적은 어디인가요 (국가와 리뷰 개수 출력)?

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

   3. 다음으로 더 자주 발견되는 국적은 어디고, 빈도를 카운트하나요?

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

3. 상위 10에 드는 리뷰어 국적에서 가장 자주 리뷰된 호텔은 어디인가요?

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

4. 데이터셋에서 호텔 별 (호텔의 빈도 개수) 리뷰가 얼마나 있나요?

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
   
   *counted in the dataset* 결과로 `Total_Number_of_Reviews`에서 값이 매치되지 않는 점을 알 것입니다. 데이터셋의 값이 호텔의 모든 리뷰 개수를 나타낸다고 하지만, 모두 스크랩하지 못했거나, 계산이 다릅니다. `Total_Number_of_Reviews`는 불명확성으로 인하여 모델에서 사용하지 않습니다.

5. 데이터셋에서 각 호텔마다 `Average_Score` 열이 있지만, 평균 점수를 계산할 수 있습니다 (각 호텔의 데이터셋에서 모든 리뷰어 점수의 평균을 얻습니다). 계산된 평균을 포함한 `Calc_Average_Score` 열 헤더로 데이터프레임에 새로운 열을 추가합니다. `Hotel_Name`, `Average_Score`, 그리고 `Calc_Average_Score` 열을 출력합니다.

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

   또는 `Average_Score` 값과 계산된 평균 점수가 가끔 다른 이유를 궁금할 수 있습니다. 값의 일부는 매칭되지만, 다른지 알 수 없으므로, 이 케이스에서 스스로 평균을 계산하는 리뷰 점수로 사용해야 안전합니다. 말하기를, 차이가 일반적으로 매우 작지만, 여기는 데이터셋 평균과 계산된 평균에 가장 큰 차이를 보이는 호텔입니다:

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

   점수 차이가 1보다 큰 호텔이 오직 1개이므로, 아마도 차이를 무시하고 계산된 평균 점수를 사용할 수 있을 것입니다.

6. "No Negative"의 `Negative_Review` 열 값에 있는 행이 얼마나 있는지 계산하고 출력합니다

7. "No Positive"의 `Positive_Review` 열 값에 있는 행이 얼마나 있는지 계산하고 출력합니다

8. "No Positive"의 `Negative_Review` 열 값**과** "No Negative"의 `Negative_Review` 열 값에 있는 행이 얼마나 있는지 계산하고 출력합니다

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

## 다른 방식

다른 방식으로 Lambda 없이 아이템을 카운트하고, sum으로 행을 카운트합니다:

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

   둘 다 `Negative_Review` 와  `Positive_Review` 열에 대하여 각자 "No Negative"와 "No Positive" 값이 127개 행을 가지고 있다는 점을 알 수 있습니다. 리뷰어가 숫자로 이루어진 점수를 호텔에 주었지만, 어느 긍정적이나 부정적인 리뷰도 남기지 않았다는 점을 의미합니다. 운이 좋아서 적은 행 개수로(515738 중 127, 0.02%), 모델 또는 결과가 어느 방향으로 치우치지 않앗지만, 리뷰 데이터셋에 리뷰가 없으므로, 이와 같이 행을 찾기 위해서 데이터를 찾을 보람이 있습니다.

지금 데이터셋을 찾아서, 다음 강의에서 데이터를 필터링하고 감정 분석을 추가해봅니다.

---
## 🚀 도전

이전 강의에서 본 것처럼, 이 강의에서 작업하기 전 데이터와 약점을 이해하는 것이 얼마나 치명적이게 중요한지 보여줍니다. 특별히, 텍스트-기반 데이터는, 조심히 조사해야 합니다. 다양한 text-heavy 데이터셋을 파보고 모델에서 치우치거나 편향된 감정으로 끼워놓은 영역을 찾을 수 있는지 확인합니다.

## [강의 후 퀴즈](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/38/)

## 검토 & 자기주도 학습

[this Learning Path on NLP](https://docs.microsoft.com/learn/paths/explore-natural-language-processing/?WT.mc_id=academic-77952-leestott)로 음성과 text-heavy 모델을 만들 때 시도하는 도구를 찾아봅니다.

## 과제 

[NLTK](../assignment.md)
