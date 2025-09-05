<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "7c077988328ebfe33b24d07945f16eca",
  "translation_date": "2025-09-05T10:41:52+00:00",
  "source_file": "2-Regression/2-Data/README.md",
  "language_code": "ko"
}
-->
# Scikit-learn을 사용한 회귀 모델 구축: 데이터 준비 및 시각화

![데이터 시각화 인포그래픽](../../../../2-Regression/2-Data/images/data-visualization.png)

인포그래픽 제작: [Dasani Madipalli](https://twitter.com/dasani_decoded)

## [강의 전 퀴즈](https://ff-quizzes.netlify.app/en/ml/)

> ### [이 강의는 R에서도 제공됩니다!](../../../../2-Regression/2-Data/solution/R/lesson_2.html)

## 소개

Scikit-learn을 사용하여 머신러닝 모델을 구축하기 위한 도구를 준비했으니, 이제 데이터를 분석하고 질문을 던질 준비가 되었습니다. 데이터를 다루고 ML 솔루션을 적용할 때, 올바른 질문을 던지는 것이 데이터의 잠재력을 제대로 활용하는 데 매우 중요합니다.

이 강의에서 배우게 될 내용:

- 모델 구축을 위한 데이터 준비 방법
- Matplotlib을 사용한 데이터 시각화 방법

## 데이터에 올바른 질문 던지기

답을 얻고자 하는 질문은 어떤 유형의 ML 알고리즘을 사용할지 결정합니다. 그리고 얻는 답변의 품질은 데이터의 특성에 크게 좌우됩니다.

이 강의에서 제공된 [데이터](https://github.com/microsoft/ML-For-Beginners/blob/main/2-Regression/data/US-pumpkins.csv)를 살펴보세요. 이 .csv 파일을 VS Code에서 열어보면, 빈칸과 문자열 및 숫자 데이터가 혼합되어 있음을 바로 확인할 수 있습니다. 또한 'Package'라는 이상한 열이 있는데, 데이터가 'sacks', 'bins' 등 다양한 값으로 구성되어 있습니다. 사실, 이 데이터는 약간 엉망입니다.

[![ML 초보자를 위한 데이터셋 분석 및 정리 방법](https://img.youtube.com/vi/5qGjczWTrDQ/0.jpg)](https://youtu.be/5qGjczWTrDQ "ML 초보자를 위한 데이터셋 분석 및 정리 방법")

> 🎥 위 이미지를 클릭하면 이 강의를 위한 데이터를 준비하는 과정을 다룬 짧은 영상을 볼 수 있습니다.

사실, ML 모델을 바로 사용할 수 있도록 완벽히 준비된 데이터셋을 받는 경우는 드뭅니다. 이 강의에서는 표준 Python 라이브러리를 사용하여 원시 데이터를 준비하는 방법을 배우게 됩니다. 또한 데이터를 시각화하는 다양한 기술도 배울 것입니다.

## 사례 연구: '호박 시장'

이 폴더의 루트 `data` 폴더에는 [US-pumpkins.csv](https://github.com/microsoft/ML-For-Beginners/blob/main/2-Regression/data/US-pumpkins.csv)라는 .csv 파일이 포함되어 있으며, 이는 도시별로 그룹화된 호박 시장에 대한 1757개의 데이터 라인을 포함하고 있습니다. 이 데이터는 미국 농무부(USDA)가 배포한 [특수 작물 터미널 시장 표준 보고서](https://www.marketnews.usda.gov/mnp/fv-report-config-step1?type=termPrice)에서 추출한 원시 데이터입니다.

### 데이터 준비

이 데이터는 공공 도메인에 속합니다. USDA 웹사이트에서 도시별로 여러 개의 파일로 다운로드할 수 있습니다. 너무 많은 파일을 방지하기 위해 모든 도시 데이터를 하나의 스프레드시트로 결합했으므로, 이미 데이터를 약간 _준비_한 상태입니다. 이제 데이터를 좀 더 자세히 살펴보겠습니다.

### 호박 데이터 - 초기 결론

이 데이터에서 무엇을 발견할 수 있나요? 이미 문자열, 숫자, 빈칸 및 이상한 값이 혼합되어 있다는 것을 확인했습니다.

회귀 기법을 사용하여 이 데이터에 어떤 질문을 던질 수 있을까요? 예를 들어, "특정 월에 판매되는 호박의 가격을 예측하라"는 질문은 어떨까요? 데이터를 다시 살펴보면, 이 작업에 필요한 데이터 구조를 만들기 위해 몇 가지 변경이 필요하다는 것을 알 수 있습니다.

## 실습 - 호박 데이터 분석

[판다스(Pandas)](https://pandas.pydata.org/)를 사용해 봅시다. 판다스는 데이터를 분석하고 준비하는 데 매우 유용한 도구입니다.

### 첫 번째 단계: 누락된 날짜 확인

먼저 누락된 날짜를 확인하기 위한 단계를 수행해야 합니다.

1. 날짜를 월 형식으로 변환합니다(미국 날짜 형식은 `MM/DD/YYYY`입니다).
2. 월을 새 열로 추출합니다.

Visual Studio Code에서 _notebook.ipynb_ 파일을 열고 스프레드시트를 새 판다스 데이터프레임으로 가져옵니다.

1. `head()` 함수를 사용하여 처음 다섯 줄을 확인합니다.

    ```python
    import pandas as pd
    pumpkins = pd.read_csv('../data/US-pumpkins.csv')
    pumpkins.head()
    ```

    ✅ 마지막 다섯 줄을 확인하려면 어떤 함수를 사용해야 할까요?

1. 현재 데이터프레임에 누락된 데이터가 있는지 확인합니다.

    ```python
    pumpkins.isnull().sum()
    ```

    누락된 데이터가 있지만, 현재 작업에는 문제가 없을 수도 있습니다.

1. 데이터프레임을 더 쉽게 작업할 수 있도록 `loc` 함수를 사용하여 필요한 열만 선택합니다. `loc` 함수는 원래 데이터프레임에서 행(첫 번째 매개변수)과 열(두 번째 매개변수)을 추출합니다. 아래의 경우 `:`는 "모든 행"을 의미합니다.

    ```python
    columns_to_select = ['Package', 'Low Price', 'High Price', 'Date']
    pumpkins = pumpkins.loc[:, columns_to_select]
    ```

### 두 번째 단계: 호박의 평균 가격 결정

특정 월에 호박의 평균 가격을 결정하는 방법을 생각해 보세요. 이 작업에 어떤 열이 필요할까요? 힌트: 3개의 열이 필요합니다.

해결 방법: `Low Price`와 `High Price` 열의 평균을 계산하여 새 Price 열을 채우고, Date 열을 월만 표시하도록 변환합니다. 다행히 위의 확인에 따르면 날짜와 가격에 누락된 데이터는 없습니다.

1. 평균을 계산하려면 다음 코드를 추가하세요.

    ```python
    price = (pumpkins['Low Price'] + pumpkins['High Price']) / 2

    month = pd.DatetimeIndex(pumpkins['Date']).month

    ```

   ✅ `print(month)`를 사용하여 데이터를 확인할 수 있습니다.

2. 변환된 데이터를 새 판다스 데이터프레임에 복사합니다.

    ```python
    new_pumpkins = pd.DataFrame({'Month': month, 'Package': pumpkins['Package'], 'Low Price': pumpkins['Low Price'],'High Price': pumpkins['High Price'], 'Price': price})
    ```

    데이터프레임을 출력하면 새 회귀 모델을 구축할 수 있는 깨끗하고 정돈된 데이터셋을 확인할 수 있습니다.

### 그런데! 이상한 점이 있습니다

`Package` 열을 보면, 호박은 다양한 구성으로 판매됩니다. 일부는 '1 1/9 bushel' 단위로, 일부는 '1/2 bushel' 단위로, 일부는 호박 개별 단위로, 일부는 파운드 단위로, 그리고 일부는 다양한 크기의 큰 상자로 판매됩니다.

> 호박은 일관되게 무게를 측정하기가 매우 어렵습니다.

원래 데이터를 살펴보면, `Unit of Sale`이 'EACH' 또는 'PER BIN'인 경우 `Package` 유형이 인치 단위, 빈 단위 또는 'each'로 표시됩니다. 호박은 일관되게 무게를 측정하기가 매우 어려운 것 같습니다. 따라서 `Package` 열에 'bushel' 문자열이 포함된 호박만 선택하여 필터링해 봅시다.

1. 초기 .csv 가져오기 아래에 필터를 추가하세요.

    ```python
    pumpkins = pumpkins[pumpkins['Package'].str.contains('bushel', case=True, regex=True)]
    ```

    데이터를 출력하면 이제 'bushel'이 포함된 약 415개의 데이터 행만 가져오는 것을 확인할 수 있습니다.

### 그런데! 해야 할 일이 하나 더 있습니다

버셸 양이 행마다 다르다는 것을 눈치챘나요? 가격을 표준화하여 버셸 단위로 가격을 표시하도록 수학적 계산을 수행해야 합니다.

1. 새 데이터프레임을 생성하는 블록 아래에 다음 줄을 추가하세요.

    ```python
    new_pumpkins.loc[new_pumpkins['Package'].str.contains('1 1/9'), 'Price'] = price/(1 + 1/9)

    new_pumpkins.loc[new_pumpkins['Package'].str.contains('1/2'), 'Price'] = price/(1/2)
    ```

✅ [The Spruce Eats](https://www.thespruceeats.com/how-much-is-a-bushel-1389308)에 따르면, 버셸의 무게는 농산물의 종류에 따라 다릅니다. 이는 부피 측정 단위입니다. "예를 들어, 토마토 한 버셸은 56파운드로 간주됩니다... 잎과 채소는 더 많은 공간을 차지하지만 무게는 적기 때문에 시금치 한 버셸은 20파운드에 불과합니다." 이는 매우 복잡합니다! 버셸을 파운드로 변환하는 작업은 생략하고 대신 버셸 단위로 가격을 표시합시다. 그러나 호박의 버셸을 연구하면서 데이터의 특성을 이해하는 것이 얼마나 중요한지 알게 됩니다!

이제 버셸 측정 단위에 따라 가격을 분석할 수 있습니다. 데이터를 한 번 더 출력하면 표준화된 데이터를 확인할 수 있습니다.

✅ 반 버셸로 판매되는 호박이 매우 비싸다는 것을 눈치챘나요? 왜 그런지 알아낼 수 있나요? 힌트: 작은 호박은 큰 호박보다 훨씬 비쌉니다. 이는 큰 속이 빈 파이 호박 하나가 차지하는 공간 때문에 버셸당 훨씬 더 많은 작은 호박이 포함되기 때문입니다.

## 시각화 전략

데이터 과학자의 역할 중 하나는 자신이 작업하는 데이터의 품질과 특성을 보여주는 것입니다. 이를 위해 데이터의 다양한 측면을 보여주는 흥미로운 시각화, 즉 플롯, 그래프, 차트를 생성합니다. 이를 통해 관계와 격차를 시각적으로 보여줄 수 있으며, 이는 그렇지 않으면 발견하기 어려운 경우가 많습니다.

[![ML 초보자를 위한 Matplotlib을 사용한 데이터 시각화 방법](https://img.youtube.com/vi/SbUkxH6IJo0/0.jpg)](https://youtu.be/SbUkxH6IJo0 "ML 초보자를 위한 Matplotlib을 사용한 데이터 시각화 방법")

> 🎥 위 이미지를 클릭하면 이 강의를 위한 데이터를 시각화하는 과정을 다룬 짧은 영상을 볼 수 있습니다.

시각화는 데이터에 가장 적합한 머신러닝 기법을 결정하는 데도 도움이 됩니다. 예를 들어, 선을 따르는 것처럼 보이는 산점도는 데이터가 선형 회귀 연습에 적합하다는 것을 나타냅니다.

Jupyter 노트북에서 잘 작동하는 데이터 시각화 라이브러리 중 하나는 [Matplotlib](https://matplotlib.org/)입니다(이전 강의에서도 보았습니다).

> 데이터 시각화에 대한 경험을 더 쌓고 싶다면 [이 튜토리얼](https://docs.microsoft.com/learn/modules/explore-analyze-data-with-python?WT.mc_id=academic-77952-leestott)을 확인하세요.

## 실습 - Matplotlib 실험하기

방금 생성한 새 데이터프레임을 표시하기 위해 기본 플롯을 만들어 보세요. 기본 선형 플롯은 무엇을 보여줄까요?

1. 파일 상단의 Pandas 가져오기 아래에 Matplotlib을 가져옵니다.

    ```python
    import matplotlib.pyplot as plt
    ```

1. 노트북 전체를 다시 실행하여 새로 고칩니다.
1. 노트북 하단에 데이터를 박스로 표시하는 셀을 추가합니다.

    ```python
    price = new_pumpkins.Price
    month = new_pumpkins.Month
    plt.scatter(price, month)
    plt.show()
    ```

    ![가격과 월 관계를 보여주는 산점도](../../../../2-Regression/2-Data/images/scatterplot.png)

    이 플롯이 유용한가요? 놀라운 점이 있나요?

    이 플롯은 특정 월에 데이터가 퍼져 있는 점으로만 표시되므로 특별히 유용하지 않습니다.

### 유용하게 만들기

유용한 데이터를 표시하려면 데이터를 그룹화해야 합니다. y축에 월을 표시하고 데이터가 분포를 나타내는 플롯을 만들어 봅시다.

1. 그룹화된 막대 차트를 생성하는 셀을 추가합니다.

    ```python
    new_pumpkins.groupby(['Month'])['Price'].mean().plot(kind='bar')
    plt.ylabel("Pumpkin Price")
    ```

    ![가격과 월 관계를 보여주는 막대 차트](../../../../2-Regression/2-Data/images/barchart.png)

    이 시각화는 더 유용합니다! 호박의 최고 가격이 9월과 10월에 발생한다는 것을 나타내는 것 같습니다. 이것이 예상과 일치하나요? 왜 그런지 생각해 보세요.

---

## 🚀도전 과제

Matplotlib이 제공하는 다양한 시각화 유형을 탐색해 보세요. 회귀 문제에 가장 적합한 유형은 무엇인가요?

## [강의 후 퀴즈](https://ff-quizzes.netlify.app/en/ml/)

## 복습 및 자기 학습

데이터를 시각화하는 다양한 방법을 살펴보세요. 사용 가능한 라이브러리 목록을 작성하고, 2D 시각화와 3D 시각화와 같은 특정 작업에 가장 적합한 라이브러리를 기록하세요. 무엇을 발견했나요?

## 과제

[시각화 탐구](assignment.md)

---

**면책 조항**:  
이 문서는 AI 번역 서비스 [Co-op Translator](https://github.com/Azure/co-op-translator)를 사용하여 번역되었습니다. 정확성을 위해 최선을 다하고 있지만, 자동 번역에는 오류나 부정확성이 포함될 수 있습니다. 원본 문서의 원어 버전을 권위 있는 출처로 간주해야 합니다. 중요한 정보의 경우, 전문적인 인간 번역을 권장합니다. 이 번역 사용으로 인해 발생하는 오해나 잘못된 해석에 대해 책임을 지지 않습니다.