<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "8d32dadeda93c6fb5c43619854882ab1",
  "translation_date": "2025-09-05T10:55:43+00:00",
  "source_file": "6-NLP/4-Hotel-Reviews-1/README.md",
  "language_code": "ko"
}
-->
# 호텔 리뷰를 활용한 감정 분석 - 데이터 처리

이 섹션에서는 이전 강의에서 배운 기술을 활용하여 대규모 데이터셋에 대한 탐색적 데이터 분석을 수행합니다. 다양한 열의 유용성을 충분히 이해한 후, 다음을 배우게 됩니다:

- 불필요한 열 제거 방법
- 기존 열을 기반으로 새로운 데이터를 계산하는 방법
- 최종 과제에서 사용할 결과 데이터셋 저장 방법

## [강의 전 퀴즈](https://ff-quizzes.netlify.app/en/ml/)

### 소개

지금까지 텍스트 데이터가 숫자형 데이터와는 매우 다르다는 것을 배웠습니다. 사람이 작성하거나 말한 텍스트는 패턴, 빈도, 감정 및 의미를 분석할 수 있습니다. 이번 강의에서는 실제 데이터셋과 실제 과제를 다룹니다: **[유럽의 515K 호텔 리뷰 데이터](https://www.kaggle.com/jiashenliu/515k-hotel-reviews-data-in-europe)**. 이 데이터는 [CC0: 퍼블릭 도메인 라이선스](https://creativecommons.org/publicdomain/zero/1.0/)를 포함하며, Booking.com에서 공개 소스를 통해 수집되었습니다. 데이터셋의 제작자는 Jiashen Liu입니다.

### 준비

필요한 사항:

* Python 3을 사용하여 .ipynb 노트북 실행 가능
* pandas
* NLTK, [로컬 설치 필요](https://www.nltk.org/install.html)
* Kaggle에서 제공되는 데이터셋 [유럽의 515K 호텔 리뷰 데이터](https://www.kaggle.com/jiashenliu/515k-hotel-reviews-data-in-europe). 압축 해제 시 약 230MB. 이 NLP 강의와 관련된 루트 `/data` 폴더에 다운로드하세요.

## 탐색적 데이터 분석

이 과제는 감정 분석과 고객 리뷰 점수를 활용하여 호텔 추천 봇을 구축하는 것을 목표로 합니다. 사용하게 될 데이터셋은 6개 도시의 1493개 호텔에 대한 리뷰를 포함하고 있습니다.

Python, 호텔 리뷰 데이터셋, 그리고 NLTK의 감정 분석을 사용하여 다음을 알아낼 수 있습니다:

* 리뷰에서 가장 자주 사용되는 단어와 구는 무엇인가요?
* 호텔을 설명하는 공식 *태그*가 리뷰 점수와 상관관계가 있나요? (예: 특정 호텔에 대해 *어린 자녀를 동반한 가족*이 *혼자 여행하는 사람*보다 더 부정적인 리뷰를 남겼다면, 이 호텔이 *혼자 여행하는 사람*에게 더 적합하다는 것을 나타낼 수 있을까요?)
* NLTK 감정 점수가 호텔 리뷰어의 숫자 점수와 '일치'하나요?

#### 데이터셋

다운로드한 데이터셋을 로컬에 저장한 후 탐색해 봅시다. VS Code나 Excel 같은 편집기에서 파일을 열어보세요.

데이터셋의 헤더는 다음과 같습니다:

*Hotel_Address, Additional_Number_of_Scoring, Review_Date, Average_Score, Hotel_Name, Reviewer_Nationality, Negative_Review, Review_Total_Negative_Word_Counts, Total_Number_of_Reviews, Positive_Review, Review_Total_Positive_Word_Counts, Total_Number_of_Reviews_Reviewer_Has_Given, Reviewer_Score, Tags, days_since_review, lat, lng*

다음과 같이 그룹화하면 더 쉽게 살펴볼 수 있습니다:
##### 호텔 관련 열

* `Hotel_Name`, `Hotel_Address`, `lat` (위도), `lng` (경도)
  * *lat*와 *lng*를 사용하여 Python으로 호텔 위치를 표시하는 지도를 그릴 수 있습니다 (부정적 및 긍정적 리뷰에 따라 색상을 구분할 수도 있음)
  * Hotel_Address는 명확히 유용하지 않으므로, 더 쉽게 정렬 및 검색할 수 있도록 국가로 대체할 가능성이 있습니다.

**호텔 메타 리뷰 열**

* `Average_Score`
  * 데이터셋 제작자에 따르면, 이 열은 *지난 1년 동안의 최신 댓글을 기반으로 계산된 호텔의 평균 점수*를 나타냅니다. 이는 다소 특이한 계산 방식이지만, 현재로서는 데이터를 그대로 받아들여야 할 것 같습니다.
  
  ✅ 이 데이터의 다른 열을 기반으로 평균 점수를 계산할 다른 방법을 생각할 수 있나요?

* `Total_Number_of_Reviews`
  * 이 호텔이 받은 총 리뷰 수 - 데이터셋의 리뷰를 참조하는지 여부는 코드 작성 없이 명확하지 않습니다.
* `Additional_Number_of_Scoring`
  * 리뷰어가 긍정적 또는 부정적 리뷰를 작성하지 않고 점수만 남긴 경우를 의미합니다.

**리뷰 관련 열**

- `Reviewer_Score`
  - 소수점 한 자리까지 표시되는 숫자 값으로, 최소값 2.5에서 최대값 10 사이입니다.
  - 왜 2.5가 가능한 최저 점수인지 설명되어 있지 않습니다.
- `Negative_Review`
  - 리뷰어가 아무것도 작성하지 않은 경우, 이 필드는 "**No Negative**"로 표시됩니다.
  - 리뷰어가 부정적 리뷰 열에 긍정적 리뷰를 작성할 수도 있습니다 (예: "이 호텔에 나쁜 점은 없습니다").
- `Review_Total_Negative_Word_Counts`
  - 부정적 단어 수가 많을수록 점수가 낮아집니다 (감정 분석을 확인하지 않은 경우).
- `Positive_Review`
  - 리뷰어가 아무것도 작성하지 않은 경우, 이 필드는 "**No Positive**"로 표시됩니다.
  - 리뷰어가 긍정적 리뷰 열에 부정적 리뷰를 작성할 수도 있습니다 (예: "이 호텔에는 좋은 점이 전혀 없습니다").
- `Review_Total_Positive_Word_Counts`
  - 긍정적 단어 수가 많을수록 점수가 높아집니다 (감정 분석을 확인하지 않은 경우).
- `Review_Date` 및 `days_since_review`
  - 리뷰의 신선도 또는 오래됨을 측정할 수 있습니다 (오래된 리뷰는 호텔 관리 변경, 리노베이션, 수영장 추가 등으로 인해 최신 리뷰만큼 정확하지 않을 수 있음).
- `Tags`
  - 리뷰어가 선택할 수 있는 짧은 설명으로, 게스트 유형 (예: 혼자 또는 가족), 객실 유형, 숙박 기간, 리뷰 제출 방법 등을 나타냅니다.
  - 불행히도, 이러한 태그를 사용하는 데는 문제가 있습니다. 아래 섹션에서 유용성에 대해 논의합니다.

**리뷰어 관련 열**

- `Total_Number_of_Reviews_Reviewer_Has_Given`
  - 예를 들어, 수백 개의 리뷰를 작성한 리뷰어가 긍정적 리뷰보다 부정적 리뷰를 남길 가능성이 더 높다는 것을 알 수 있다면, 추천 모델에서 이 요소를 고려할 수 있습니다. 그러나 특정 리뷰의 리뷰어는 고유 코드로 식별되지 않으므로 리뷰 세트와 연결할 수 없습니다. 100개 이상의 리뷰를 작성한 리뷰어는 30명 있지만, 추천 모델에 어떻게 도움이 될지 판단하기 어렵습니다.
- `Reviewer_Nationality`
  - 일부 사람들은 특정 국적이 긍정적 또는 부정적 리뷰를 남길 가능성이 더 높다고 생각할 수 있습니다. 하지만 이러한 견해를 모델에 포함할 때 주의해야 합니다. 이는 국가적 (때로는 인종적) 고정관념이며, 각 리뷰어는 자신의 경험을 바탕으로 리뷰를 작성한 개인입니다. 이전 호텔 숙박 경험, 여행 거리, 개인 성격 등 여러 관점에서 필터링되었을 수 있습니다. 리뷰 점수가 국적 때문이라고 생각하는 것은 정당화하기 어렵습니다.

##### 예시

| Average  Score | Total Number   Reviews | Reviewer   Score | Negative <br />Review                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  | Positive   Review                 | Tags                                                                                      |
| -------------- | ---------------------- | ---------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------- | ----------------------------------------------------------------------------------------- |
| 7.8            | 1945                   | 2.5              | 현재 이곳은 호텔이 아니라 공사 현장입니다. 긴 여행 후 휴식을 취하며 방에서 작업하는 동안 아침 일찍부터 하루 종일 용납할 수 없는 공사 소음에 시달렸습니다. 인접한 방에서 사람들이 하루 종일 작업했습니다. 즉, 잭해머를 사용했습니다. 방 변경을 요청했지만 조용한 방은 없었습니다. 상황을 더 악화시킨 것은 과도한 요금이 부과된 것입니다. 이른 아침 비행기로 떠나야 했기 때문에 저녁에 체크아웃했고 적절한 청구서를 받았습니다. 하루 후 호텔은 예약 가격을 초과하여 제 동의 없이 추가 요금을 부과했습니다. 끔찍한 곳입니다. 여기 예약하지 마세요. | 아무것도 없음 끔찍한 곳 멀리하세요 | 비즈니스 여행                                커플 스탠다드 더블룸 2박 숙박 |

보시다시피, 이 손님은 호텔에서 행복한 숙박을 하지 못했습니다. 호텔은 7.8의 좋은 평균 점수와 1945개의 리뷰를 가지고 있지만, 이 리뷰어는 2.5를 주고 숙박의 부정적인 점에 대해 115단어를 작성했습니다. 긍정적 리뷰 열에 아무것도 작성하지 않았다면 긍정적인 점이 없다고 추측할 수 있지만, 경고의 7단어를 작성했습니다. 단어 수만 계산하고 단어의 의미나 감정을 고려하지 않으면 리뷰어의 의도를 왜곡할 수 있습니다. 이상하게도, 2.5라는 점수는 혼란스럽습니다. 호텔 숙박이 그렇게 나빴다면 왜 점수를 주었을까요? 데이터셋을 자세히 조사하면 가능한 최저 점수가 2.5이고 0이 아님을 알 수 있습니다. 가능한 최고 점수는 10입니다.

##### 태그

위에서 언급했듯이, 처음에는 `Tags`를 사용하여 데이터를 분류하는 아이디어가 합리적으로 보입니다. 그러나 이 태그는 표준화되어 있지 않아, 특정 호텔에서는 *싱글룸*, *트윈룸*, *더블룸* 옵션이 있을 수 있지만, 다른 호텔에서는 *디럭스 싱글룸*, *클래식 퀸룸*, *이그제큐티브 킹룸* 옵션이 있을 수 있습니다. 이것들이 동일한 것일 수도 있지만, 변형이 너무 많아 선택지가 다음과 같이 됩니다:

1. 모든 용어를 단일 표준으로 변경하려고 시도하지만, 각 경우에 변환 경로가 명확하지 않아 매우 어렵습니다 (예: *클래식 싱글룸*은 *싱글룸*으로 매핑되지만 *정원 또는 도시 전망이 있는 슈페리어 퀸룸*은 매핑하기 훨씬 어렵습니다).

1. NLP 접근 방식을 사용하여 *혼자*, *비즈니스 여행객*, *어린 자녀를 동반한 가족*과 같은 특정 용어의 빈도를 측정하고 이를 추천에 반영합니다.

태그는 일반적으로 (항상 그런 것은 아니지만) *여행 유형*, *게스트 유형*, *객실 유형*, *숙박 기간*, *리뷰 제출 장치 유형*에 맞춰 5~6개의 쉼표로 구분된 값을 포함하는 단일 필드입니다. 그러나 일부 리뷰어가 각 필드를 채우지 않는 경우 (하나를 비워둘 수 있음), 값이 항상 동일한 순서로 나타나지는 않습니다.

예를 들어, *그룹 유형*을 살펴보면, `Tags` 열에서 이 필드에 1025개의 고유 가능성이 있으며, 불행히도 그중 일부만 그룹을 참조합니다 (일부는 객실 유형 등입니다). 가족을 언급하는 태그만 필터링하면 결과에 많은 *가족룸* 유형 결과가 포함됩니다. *with*라는 용어를 포함하여 *가족과 함께* 값을 계산하면 결과가 더 좋아지며, "어린 자녀를 동반한 가족" 또는 "나이 든 자녀를 동반한 가족"이라는 문구가 포함된 515,000개의 결과 중 80,000개 이상을 얻을 수 있습니다.

따라서 태그 열이 완전히 쓸모없는 것은 아니지만, 유용하게 만들기 위해 약간의 작업이 필요합니다.

##### 호텔 평균 점수

데이터셋과 관련하여 몇 가지 이상하거나 불일치한 점이 있지만, 이를 명확히 하기 위해 여기서 설명합니다. 모델을 구축할 때 이를 인지하고 해결 방법을 찾으면 토론 섹션에서 알려주세요!

데이터셋에는 평균 점수와 리뷰 수와 관련된 다음 열이 포함되어 있습니다:

1. Hotel_Name
2. Additional_Number_of_Scoring
3. Average_Score
4. Total_Number_of_Reviews
5. Reviewer_Score  

이 데이터셋에서 가장 많은 리뷰를 가진 호텔은 *Britannia International Hotel Canary Wharf*로, 515,000개 중 4789개의 리뷰를 가지고 있습니다. 그러나 이 호텔의 `Total_Number_of_Reviews` 값은 9086입니다. 많은 점수가 리뷰 없이 존재한다고 추측할 수 있으므로 `Additional_Number_of_Scoring` 열 값을 추가해야 할 것 같습니다. 해당 값은 2682이며, 이를 4789에 더하면 7471이 되지만 여전히 `Total_Number_of_Reviews` 값보다 1615 부족합니다.

`Average_Score` 열을 보면 데이터셋의 리뷰 평균이라고 추측할 수 있지만, Kaggle 설명에 따르면 "*지난 1년 동안의 최신 댓글을 기반으로 계산된 호텔의 평균 점수*"라고 합니다. 이는 그다지 유용하지 않아 보이지만, 데이터셋의 리뷰 점수를 기반으로 자체 평균을 계산할 수 있습니다. 동일한 호텔을 예로 들면, 호텔의 평균 점수는 7.1로 제공되지만 데이터셋 내 리뷰어 점수의 계산 평균은 6.8입니다. 이는 근접하지만 동일한 값은 아니며, `Additional_Number_of_Scoring` 리뷰에서 제공된 점수가 평균을 7.1로 증가시켰다고 추측할 수 있습니다. 그러나 이를 테스트하거나 증명할 방법이 없으므로 `Average_Score`, `Additional_Number_of_Scoring`, `Total_Number_of_Reviews`를 사용할 수 없거나 신뢰하기 어렵습니다.

더 복잡한 점은 리뷰 수가 두 번째로 많은 호텔의 계산 평균 점수가 8.12이고 데이터셋의 `Average_Score`는 8.1이라는 점입니다. 첫 번째 호텔이 불일치인지, 두 번째 호텔이 올바른 점수인지 우연인지 알 수 없습니다.

이 호텔이 이상치일 가능성이 있으며, 대부분의 값이 일치하지만 일부는 어떤 이유로 일치하지 않을 수 있다고 가정하고, 데이터셋의 값을 탐색하고 올바른 사용법 (또는 사용하지 않는 방법)을 결정하기 위해 다음에 짧은 프로그램을 작성할 것입니다.
> 🚨 주의 사항  
>  
> 이 데이터셋을 다룰 때, 텍스트를 직접 읽거나 분석하지 않고 텍스트에서 무언가를 계산하는 코드를 작성하게 됩니다. 이것이 바로 NLP의 본질로, 인간이 직접 하지 않아도 의미나 감정을 해석하는 것입니다. 하지만 부정적인 리뷰를 읽게 될 가능성도 있습니다. 저는 여러분이 이를 읽지 않기를 권합니다. 왜냐하면 그럴 필요가 없기 때문입니다. 일부 리뷰는 터무니없거나 관련성이 없는 부정적인 호텔 리뷰일 수 있습니다. 예를 들어, "날씨가 좋지 않았다"는 호텔이나 누구도 통제할 수 없는 문제입니다. 하지만 리뷰에는 어두운 면도 존재합니다. 때로는 부정적인 리뷰가 인종차별적, 성차별적, 혹은 연령차별적일 수 있습니다. 이는 공공 웹사이트에서 수집된 데이터셋에서 예상할 수 있는 불행한 현실입니다. 일부 리뷰어는 불쾌하거나 불편하며 마음을 상하게 할 수 있는 리뷰를 남기기도 합니다. 리뷰를 직접 읽고 마음이 상하기보다는 코드가 감정을 측정하도록 하는 것이 더 나을 것입니다. 그렇다고 해서 이런 리뷰를 작성하는 사람이 다수는 아니지만, 소수라도 존재한다는 사실은 변하지 않습니다.
## 연습 - 데이터 탐색
### 데이터 불러오기

이제 데이터를 시각적으로 살펴보는 것은 충분하니, 코드를 작성하여 답을 찾아봅시다! 이 섹션에서는 pandas 라이브러리를 사용합니다. 첫 번째 과제는 CSV 데이터를 불러오고 읽을 수 있는지 확인하는 것입니다. pandas 라이브러리는 빠른 CSV 로더를 제공하며, 결과는 이전 강의에서처럼 데이터프레임에 저장됩니다. 우리가 불러올 CSV 파일은 50만 개 이상의 행과 17개의 열로 구성되어 있습니다. pandas는 데이터프레임과 상호작용할 수 있는 강력한 기능을 많이 제공하며, 각 행에 대해 연산을 수행할 수도 있습니다.

이제부터 이 강의에서는 코드 스니펫과 코드 설명, 그리고 결과에 대한 논의가 포함될 것입니다. 제공된 _notebook.ipynb_ 파일을 사용하여 코드를 작성하세요.

우선 사용할 데이터 파일을 불러오는 것부터 시작해봅시다:

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

데이터를 불러왔으니, 이제 몇 가지 연산을 수행할 수 있습니다. 이 코드를 프로그램 상단에 유지하면서 다음 단계를 진행하세요.

## 데이터 탐색

이번 경우, 데이터는 이미 *정리(clean)* 되어 있습니다. 즉, 작업할 준비가 되어 있으며, 영어 문자만을 기대하는 알고리즘을 방해할 수 있는 다른 언어의 문자가 포함되어 있지 않습니다.

✅ 데이터를 처리하기 전에 형식을 맞추는 초기 처리가 필요한 경우도 있지만, 이번에는 그렇지 않습니다. 만약 비영어 문자를 처리해야 한다면, 어떻게 처리할지 생각해보세요.

데이터를 불러온 후, 코드를 사용하여 데이터를 탐색할 수 있는지 확인하세요. `Negative_Review`와 `Positive_Review` 열에 초점을 맞추고 싶을 수 있습니다. 이 열들은 NLP 알고리즘이 처리할 자연어 텍스트로 채워져 있습니다. 하지만 잠깐! NLP와 감정 분석에 뛰어들기 전에, 아래 코드를 따라 데이터셋에 주어진 값이 pandas로 계산한 값과 일치하는지 확인하세요.

## 데이터프레임 연산

이번 강의의 첫 번째 과제는 데이터프레임을 변경하지 않고 데이터를 확인하는 코드를 작성하여 다음 주장이 맞는지 확인하는 것입니다.

> 많은 프로그래밍 작업과 마찬가지로, 여러 가지 방법으로 작업을 완료할 수 있습니다. 하지만 가장 간단하고 쉬운 방법으로 작업하는 것이 좋습니다. 특히 나중에 이 코드를 다시 볼 때 이해하기 쉬운 방법이라면 더욱 그렇습니다. 데이터프레임의 경우, 종합적인 API가 제공되므로 원하는 작업을 효율적으로 수행할 방법이 있을 가능성이 높습니다.

다음 질문들을 코딩 과제로 간주하고, 솔루션을 보기 전에 스스로 답을 찾아보세요.

1. 방금 불러온 데이터프레임의 *shape* (행과 열의 수)을 출력하세요.
2. 리뷰어 국적에 대한 빈도수를 계산하세요:
   1. `Reviewer_Nationality` 열에 몇 가지 고유 값이 있는지, 그리고 그 값들이 무엇인지 확인하세요.
   2. 데이터셋에서 가장 흔한 리뷰어 국적은 무엇인지 (국가와 리뷰 수를 출력) 확인하세요.
   3. 다음으로 많이 등장하는 10개의 국적과 그 빈도수를 확인하세요.
3. 가장 많이 리뷰된 호텔을 각 상위 10개 리뷰어 국적별로 확인하세요.
4. 데이터셋에서 호텔별 리뷰 수(호텔의 빈도수)를 계산하세요.
5. 데이터셋에 각 호텔의 `Average_Score` 열이 있지만, 각 호텔에 대한 모든 리뷰어 점수의 평균을 계산하여 새로운 열 `Calc_Average_Score`를 추가하세요.
6. `Average_Score`와 `Calc_Average_Score`가 동일한 호텔이 있는지 확인하세요 (소수점 첫째 자리까지 반올림하여 비교).
   1. Series(행)를 인자로 받아 값을 비교하고, 값이 같지 않을 때 메시지를 출력하는 Python 함수를 작성하세요. 그런 다음 `.apply()` 메서드를 사용하여 모든 행을 처리하세요.
7. `Negative_Review` 열 값이 "No Negative"인 행의 수를 계산하고 출력하세요.
8. `Positive_Review` 열 값이 "No Positive"인 행의 수를 계산하고 출력하세요.
9. `Positive_Review` 열 값이 "No Positive"이고 `Negative_Review` 열 값이 "No Negative"인 행의 수를 계산하고 출력하세요.

### 코드 답안

1. 방금 불러온 데이터프레임의 *shape* (행과 열의 수)을 출력하세요.

   ```python
   print("The shape of the data (rows, cols) is " + str(df.shape))
   > The shape of the data (rows, cols) is (515738, 17)
   ```

2. 리뷰어 국적에 대한 빈도수를 계산하세요:

   1. `Reviewer_Nationality` 열에 몇 가지 고유 값이 있는지, 그리고 그 값들이 무엇인지 확인하세요.
   2. 데이터셋에서 가장 흔한 리뷰어 국적은 무엇인지 (국가와 리뷰 수를 출력) 확인하세요.

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

   3. 다음으로 많이 등장하는 10개의 국적과 그 빈도수를 확인하세요.

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

3. 가장 많이 리뷰된 호텔을 각 상위 10개 리뷰어 국적별로 확인하세요.

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

4. 데이터셋에서 호텔별 리뷰 수(호텔의 빈도수)를 계산하세요.

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

   데이터셋에서 계산된 결과와 `Total_Number_of_Reviews` 값이 일치하지 않는 것을 알 수 있습니다. 이 값이 호텔의 총 리뷰 수를 나타내지만, 모든 리뷰가 스크랩되지 않았거나 다른 계산이 적용된 것일 수 있습니다. 이러한 불확실성 때문에 `Total_Number_of_Reviews`는 모델에서 사용되지 않습니다.

5. 데이터셋에 각 호텔의 `Average_Score` 열이 있지만, 각 호텔에 대한 모든 리뷰어 점수의 평균을 계산하여 새로운 열 `Calc_Average_Score`를 추가하세요. `Hotel_Name`, `Average_Score`, `Calc_Average_Score` 열을 출력하세요.

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

   데이터셋 평균과 계산된 평균이 다른 이유가 궁금할 수 있습니다. 일부 값이 일치하지만, 다른 값은 차이가 있는 이유를 알 수 없으므로, 이 경우 우리가 가진 리뷰 점수를 사용하여 평균을 직접 계산하는 것이 가장 안전합니다. 차이가 매우 작기 때문에, 계산된 평균 점수를 사용하는 것이 적절합니다.

6. `Negative_Review` 열 값이 "No Negative"인 행의 수를 계산하고 출력하세요.

7. `Positive_Review` 열 값이 "No Positive"인 행의 수를 계산하고 출력하세요.

8. `Positive_Review` 열 값이 "No Positive"이고 `Negative_Review` 열 값이 "No Negative"인 행의 수를 계산하고 출력하세요.

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

## 또 다른 방법

람다를 사용하지 않고 항목을 세는 또 다른 방법은 sum을 사용하는 것입니다:

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

   `Negative_Review`와 `Positive_Review` 열 값이 각각 "No Negative"와 "No Positive"인 행이 127개 있다는 것을 알 수 있습니다. 이는 리뷰어가 호텔에 점수를 부여했지만, 긍정적이거나 부정적인 리뷰를 작성하지 않았다는 것을 의미합니다. 다행히도 이러한 행은 515,738개 중 127개(0.02%)로 매우 적기 때문에, 모델이나 결과에 큰 영향을 미치지 않을 것입니다. 하지만 리뷰 데이터셋에 리뷰가 없는 행이 포함되어 있다는 점은 예상치 못했을 수 있으므로, 데이터를 탐색하여 이러한 행을 발견하는 것이 중요합니다.

이제 데이터셋을 탐색했으니, 다음 강의에서는 데이터를 필터링하고 감정 분석을 추가할 것입니다.

---
## 🚀도전 과제

이번 강의는 이전 강의에서 본 것처럼, 데이터를 이해하고 그 특성을 파악하는 것이 얼마나 중요한지 보여줍니다. 특히 텍스트 기반 데이터는 신중히 검토해야 합니다. 다양한 텍스트 중심 데이터셋을 탐색하며, 모델에 편향이나 왜곡된 감정을 도입할 수 있는 영역을 발견해보세요.

## [강의 후 퀴즈](https://ff-quizzes.netlify.app/en/ml/)

## 복습 및 자기 학습

[NLP 학습 경로](https://docs.microsoft.com/learn/paths/explore-natural-language-processing/?WT.mc_id=academic-77952-leestott)를 통해 음성 및 텍스트 중심 모델을 구축할 때 사용할 수 있는 도구를 알아보세요.

## 과제 

[NLTK](assignment.md)

---

**면책 조항**:  
이 문서는 AI 번역 서비스 [Co-op Translator](https://github.com/Azure/co-op-translator)를 사용하여 번역되었습니다. 정확성을 위해 최선을 다하고 있으나, 자동 번역에는 오류나 부정확성이 포함될 수 있습니다. 원본 문서를 해당 언어로 작성된 상태에서 권위 있는 자료로 간주해야 합니다. 중요한 정보의 경우, 전문적인 인간 번역을 권장합니다. 이 번역 사용으로 인해 발생하는 오해나 잘못된 해석에 대해 당사는 책임을 지지 않습니다.  