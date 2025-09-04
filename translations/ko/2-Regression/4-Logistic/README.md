<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "72b5bae0879baddf6aafc82bb07b8776",
  "translation_date": "2025-09-03T22:26:52+00:00",
  "source_file": "2-Regression/4-Logistic/README.md",
  "language_code": "ko"
}
-->
# 카테고리 예측을 위한 로지스틱 회귀

![로지스틱 vs. 선형 회귀 인포그래픽](../../../../translated_images/linear-vs-logistic.ba180bf95e7ee66721ba10ebf2dac2666acbd64a88b003c83928712433a13c7d.ko.png)

## [강의 전 퀴즈](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/15/)

> ### [이 강의는 R에서도 제공됩니다!](../../../../2-Regression/4-Logistic/solution/R/lesson_4.html)

## 소개

이번 회귀에 대한 마지막 강의에서는 기본적인 _고전적인_ 머신러닝 기법 중 하나인 로지스틱 회귀를 살펴보겠습니다. 이 기법은 패턴을 발견하여 이진 카테고리를 예측하는 데 사용됩니다. 이 사탕이 초콜릿인가 아닌가? 이 질병이 전염성인가 아닌가? 이 고객이 이 제품을 선택할 것인가 아닌가?

이번 강의에서 배우게 될 내용:

- 데이터 시각화를 위한 새로운 라이브러리
- 로지스틱 회귀 기법

✅ 이 [학습 모듈](https://docs.microsoft.com/learn/modules/train-evaluate-classification-models?WT.mc_id=academic-77952-leestott)을 통해 이 회귀 유형에 대한 이해를 심화하세요.

## 사전 준비

호박 데이터를 다루면서, 이제 `Color`라는 이진 카테고리를 작업할 수 있다는 것을 충분히 이해하게 되었습니다.

이제 로지스틱 회귀 모델을 구축하여 몇 가지 변수에 따라 _주어진 호박의 색상이 무엇일 가능성이 높은지_ 예측해 봅시다 (주황색 🎃 또는 흰색 👻).

> 왜 회귀에 대한 강의에서 이진 분류를 다루고 있을까요? 언어적 편의를 위해서입니다. 로지스틱 회귀는 [사실 분류 방법](https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression)이며, 선형 기반이긴 하지만요. 다음 강의 그룹에서 데이터를 분류하는 다른 방법에 대해 알아보세요.

## 질문 정의하기

우리의 목적을 위해 이를 이진으로 표현하겠습니다: '흰색' 또는 '흰색이 아님'. 데이터셋에 '줄무늬' 카테고리도 있지만, 해당 사례가 적기 때문에 사용하지 않을 것입니다. 어쨌든 데이터셋에서 null 값을 제거하면 사라집니다.

> 🎃 재미있는 사실: 우리는 때때로 흰색 호박을 '유령' 호박이라고 부릅니다. 조각하기가 쉽지 않아서 주황색 호박만큼 인기가 많지는 않지만, 멋지게 생겼습니다! 따라서 질문을 이렇게 다시 표현할 수도 있습니다: '유령' 또는 '유령이 아님'. 👻

## 로지스틱 회귀에 대하여

로지스틱 회귀는 이전에 배운 선형 회귀와 몇 가지 중요한 점에서 다릅니다.

[![초보자를 위한 머신러닝 - 로지스틱 회귀를 통한 데이터 분류 이해하기](https://img.youtube.com/vi/KpeCT6nEpBY/0.jpg)](https://youtu.be/KpeCT6nEpBY "초보자를 위한 머신러닝 - 로지스틱 회귀를 통한 데이터 분류 이해하기")

> 🎥 위 이미지를 클릭하여 로지스틱 회귀에 대한 간단한 비디오 개요를 확인하세요.

### 이진 분류

로지스틱 회귀는 선형 회귀와 같은 기능을 제공하지 않습니다. 전자는 이진 카테고리("흰색 또는 흰색이 아님")에 대한 예측을 제공하는 반면, 후자는 연속적인 값을 예측할 수 있습니다. 예를 들어, 호박의 원산지와 수확 시기를 기준으로 _가격이 얼마나 오를지_ 예측할 수 있습니다.

![호박 분류 모델](../../../../translated_images/pumpkin-classifier.562771f104ad5436b87d1c67bca02a42a17841133556559325c0a0e348e5b774.ko.png)
> 인포그래픽: [Dasani Madipalli](https://twitter.com/dasani_decoded)

### 기타 분류

로지스틱 회귀에는 다항 및 순서형을 포함한 다른 유형도 있습니다:

- **다항**: 여러 카테고리가 있는 경우 - "주황색, 흰색, 줄무늬".
- **순서형**: 카테고리가 논리적으로 순서가 있는 경우, 예를 들어 호박의 크기가 제한된 수의 크기(미니, 소, 중, 대, 특대, 초특대)로 정렬된 경우.

![다항 vs 순서형 회귀](../../../../translated_images/multinomial-vs-ordinal.36701b4850e37d86c9dd49f7bef93a2f94dbdb8fe03443eb68f0542f97f28f29.ko.png)

### 변수는 반드시 상관관계가 있을 필요 없음

선형 회귀가 더 많은 상관관계가 있는 변수와 잘 작동했던 것을 기억하시나요? 로지스틱 회귀는 반대입니다 - 변수들이 반드시 일치할 필요가 없습니다. 이는 상관관계가 약한 이 데이터에 적합합니다.

### 많은 양의 깨끗한 데이터가 필요함

로지스틱 회귀는 더 많은 데이터를 사용할수록 더 정확한 결과를 제공합니다. 우리의 작은 데이터셋은 이 작업에 최적이 아니므로 이를 염두에 두세요.

[![초보자를 위한 머신러닝 - 로지스틱 회귀를 위한 데이터 분석 및 준비](https://img.youtube.com/vi/B2X4H9vcXTs/0.jpg)](https://youtu.be/B2X4H9vcXTs "초보자를 위한 머신러닝 - 로지스틱 회귀를 위한 데이터 분석 및 준비")

> 🎥 위 이미지를 클릭하여 선형 회귀를 위한 데이터 준비에 대한 간단한 비디오 개요를 확인하세요.

✅ 로지스틱 회귀에 적합한 데이터 유형에 대해 생각해 보세요.

## 실습 - 데이터 정리하기

먼저 null 값을 제거하고 몇 가지 열만 선택하여 데이터를 약간 정리합니다:

1. 다음 코드를 추가하세요:

    ```python
  
    columns_to_select = ['City Name','Package','Variety', 'Origin','Item Size', 'Color']
    pumpkins = full_pumpkins.loc[:, columns_to_select]

    pumpkins.dropna(inplace=True)
    ```

    새 데이터프레임을 확인할 수도 있습니다:

    ```python
    pumpkins.info
    ```

### 시각화 - 카테고리 플롯

이제 [시작 노트북](./notebook.ipynb)에 호박 데이터를 다시 로드하고 몇 가지 변수, 특히 `Color`를 포함하는 데이터셋을 유지하도록 정리했습니다. 이번에는 다른 라이브러리를 사용하여 데이터프레임을 노트북에서 시각화해 봅시다: [Seaborn](https://seaborn.pydata.org/index.html). 이는 이전에 사용한 Matplotlib을 기반으로 구축되었습니다.

Seaborn은 데이터를 시각화하는 멋진 방법을 제공합니다. 예를 들어, `Variety`와 `Color`의 데이터 분포를 카테고리 플롯에서 비교할 수 있습니다.

1. 호박 데이터 `pumpkins`를 사용하여 `catplot` 함수를 사용하고 각 호박 카테고리(주황색 또는 흰색)에 대한 색상 매핑을 지정하여 플롯을 생성하세요:

    ```python
    import seaborn as sns
    
    palette = {
    'ORANGE': 'orange',
    'WHITE': 'wheat',
    }

    sns.catplot(
    data=pumpkins, y="Variety", hue="Color", kind="count",
    palette=palette, 
    )
    ```

    ![시각화된 데이터의 그리드](../../../../translated_images/pumpkins_catplot_1.c55c409b71fea2ecc01921e64b91970542101f90bcccfa4aa3a205db8936f48b.ko.png)

    데이터를 관찰함으로써 Color 데이터가 Variety와 어떻게 관련되어 있는지 확인할 수 있습니다.

    ✅ 이 카테고리 플롯을 보고 어떤 흥미로운 탐구를 상상할 수 있나요?

### 데이터 전처리: 특징 및 라벨 인코딩
우리의 호박 데이터셋은 모든 열에 문자열 값을 포함하고 있습니다. 카테고리 데이터를 다루는 것은 인간에게는 직관적이지만 기계에게는 그렇지 않습니다. 머신러닝 알고리즘은 숫자와 잘 작동합니다. 따라서 인코딩은 데이터 전처리 단계에서 매우 중요한 단계입니다. 이는 카테고리 데이터를 숫자 데이터로 변환할 수 있게 해주며, 정보를 잃지 않습니다. 좋은 인코딩은 좋은 모델을 구축하는 데 도움이 됩니다.

특징 인코딩에는 두 가지 주요 유형의 인코더가 있습니다:

1. 순서형 인코더: 순서형 변수에 적합합니다. 순서형 변수는 데이터가 논리적 순서를 따르는 카테고리 변수입니다. 예를 들어, 데이터셋의 `Item Size` 열이 이에 해당합니다. 각 카테고리가 열의 순서에 따라 숫자로 표현되도록 매핑을 생성합니다.

    ```python
    from sklearn.preprocessing import OrdinalEncoder

    item_size_categories = [['sml', 'med', 'med-lge', 'lge', 'xlge', 'jbo', 'exjbo']]
    ordinal_features = ['Item Size']
    ordinal_encoder = OrdinalEncoder(categories=item_size_categories)
    ```

2. 카테고리 인코더: 명목 변수에 적합합니다. 명목 변수는 데이터가 논리적 순서를 따르지 않는 카테고리 변수입니다. 데이터셋에서 `Item Size`를 제외한 모든 특징이 이에 해당합니다. 이는 원-핫 인코딩으로, 각 카테고리가 이진 열로 표현됩니다: 인코딩된 변수는 호박이 해당 Variety에 속하면 1이고 그렇지 않으면 0입니다.

    ```python
    from sklearn.preprocessing import OneHotEncoder

    categorical_features = ['City Name', 'Package', 'Variety', 'Origin']
    categorical_encoder = OneHotEncoder(sparse_output=False)
    ```
그런 다음, `ColumnTransformer`를 사용하여 여러 인코더를 단일 단계로 결합하고 적절한 열에 적용합니다.

```python
    from sklearn.compose import ColumnTransformer
    
    ct = ColumnTransformer(transformers=[
        ('ord', ordinal_encoder, ordinal_features),
        ('cat', categorical_encoder, categorical_features)
        ])
    
    ct.set_output(transform='pandas')
    encoded_features = ct.fit_transform(pumpkins)
```
한편, 라벨을 인코딩하기 위해 scikit-learn의 `LabelEncoder` 클래스를 사용합니다. 이는 라벨을 정규화하여 0에서 n_classes-1(여기서는 0과 1) 사이의 값만 포함하도록 돕는 유틸리티 클래스입니다.

```python
    from sklearn.preprocessing import LabelEncoder

    label_encoder = LabelEncoder()
    encoded_label = label_encoder.fit_transform(pumpkins['Color'])
```
특징과 라벨을 인코딩한 후, 이를 새로운 데이터프레임 `encoded_pumpkins`로 병합할 수 있습니다.

```python
    encoded_pumpkins = encoded_features.assign(Color=encoded_label)
```
✅ `Item Size` 열에 순서형 인코더를 사용하는 장점은 무엇인가요?

### 변수 간 관계 분석

이제 데이터를 전처리했으니, 특징과 라벨 간의 관계를 분석하여 모델이 특징을 기반으로 라벨을 얼마나 잘 예측할 수 있을지에 대한 아이디어를 얻을 수 있습니다.
이러한 분석을 수행하는 가장 좋은 방법은 데이터를 시각화하는 것입니다. Seaborn의 `catplot` 함수를 다시 사용하여 `Item Size`, `Variety`, `Color` 간의 관계를 카테고리 플롯에서 시각화합니다. 데이터를 더 잘 시각화하기 위해 인코딩된 `Item Size` 열과 인코딩되지 않은 `Variety` 열을 사용합니다.

```python
    palette = {
    'ORANGE': 'orange',
    'WHITE': 'wheat',
    }
    pumpkins['Item Size'] = encoded_pumpkins['ord__Item Size']

    g = sns.catplot(
        data=pumpkins,
        x="Item Size", y="Color", row='Variety',
        kind="box", orient="h",
        sharex=False, margin_titles=True,
        height=1.8, aspect=4, palette=palette,
    )
    g.set(xlabel="Item Size", ylabel="").set(xlim=(0,6))
    g.set_titles(row_template="{row_name}")
```
![시각화된 데이터의 카테고리 플롯](../../../../translated_images/pumpkins_catplot_2.87a354447880b3889278155957f8f60dd63db4598de5a6d0fda91c334d31f9f1.ko.png)

### 스웜 플롯 사용하기

Color는 이진 카테고리(흰색 또는 흰색이 아님)이므로 '시각화를 위한 [특별한 접근법](https://seaborn.pydata.org/tutorial/categorical.html?highlight=bar)'이 필요합니다. 이 카테고리가 다른 변수와의 관계를 시각화하는 다른 방법도 있습니다.

변수를 나란히 시각화할 수 있는 Seaborn 플롯을 사용할 수 있습니다.

1. 값의 분포를 보여주는 '스웜' 플롯을 시도해 보세요:

    ```python
    palette = {
    0: 'orange',
    1: 'wheat'
    }
    sns.swarmplot(x="Color", y="ord__Item Size", data=encoded_pumpkins, palette=palette)
    ```

    ![시각화된 데이터의 스웜](../../../../translated_images/swarm_2.efeacfca536c2b577dc7b5f8891f28926663fbf62d893ab5e1278ae734ca104e.ko.png)

**주의**: 위 코드는 경고를 생성할 수 있습니다. Seaborn이 많은 데이터 포인트를 스웜 플롯에 표현하지 못하기 때문입니다. 가능한 해결책은 'size' 매개변수를 사용하여 마커 크기를 줄이는 것입니다. 그러나 이는 플롯의 가독성에 영향을 미칠 수 있다는 점을 유의하세요.

> **🧮 수학을 보여주세요**
>
> 로지스틱 회귀는 [시그모이드 함수](https://wikipedia.org/wiki/Sigmoid_function)를 사용하여 '최대 가능도' 개념에 의존합니다. 플롯에서 '시그모이드 함수'는 'S' 모양으로 보입니다. 값 하나를 받아 0과 1 사이의 값으로 매핑합니다. 이 곡선은 '로지스틱 곡선'이라고도 불립니다. 공식은 다음과 같습니다:
>
> ![로지스틱 함수](../../../../translated_images/sigmoid.8b7ba9d095c789cf72780675d0d1d44980c3736617329abfc392dfc859799704.ko.png)
>
> 여기서 시그모이드의 중간점은 x의 0 지점에 위치하며, L은 곡선의 최대값, k는 곡선의 가파름을 나타냅니다. 함수 결과가 0.5보다 크면 해당 라벨은 이진 선택의 '1' 클래스에 할당됩니다. 그렇지 않으면 '0'으로 분류됩니다.

## 모델 구축하기

Scikit-learn에서 이진 분류를 찾는 모델을 구축하는 것은 놀라울 정도로 간단합니다.

[![초보자를 위한 머신러닝 - 데이터 분류를 위한 로지스틱 회귀](https://img.youtube.com/vi/MmZS2otPrQ8/0.jpg)](https://youtu.be/MmZS2otPrQ8 "초보자를 위한 머신러닝 - 데이터 분류를 위한 로지스틱 회귀")

> 🎥 위 이미지를 클릭하여 선형 회귀 모델 구축에 대한 간단한 비디오 개요를 확인하세요.

1. 분류 모델에서 사용할 변수를 선택하고 `train_test_split()`을 호출하여 학습 및 테스트 세트를 분리하세요:

    ```python
    from sklearn.model_selection import train_test_split
    
    X = encoded_pumpkins[encoded_pumpkins.columns.difference(['Color'])]
    y = encoded_pumpkins['Color']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    
    ```

2. 이제 학습 데이터를 사용하여 `fit()`을 호출하여 모델을 학습시키고 결과를 출력할 수 있습니다:

    ```python
    from sklearn.metrics import f1_score, classification_report 
    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    print(classification_report(y_test, predictions))
    print('Predicted labels: ', predictions)
    print('F1-score: ', f1_score(y_test, predictions))
    ```

    모델의 점수표를 확인하세요. 데이터가 약 1000행밖에 없다는 점을 고려하면 나쁘지 않습니다:

    ```output
                       precision    recall  f1-score   support
    
                    0       0.94      0.98      0.96       166
                    1       0.85      0.67      0.75        33
    
        accuracy                                0.92       199
        macro avg           0.89      0.82      0.85       199
        weighted avg        0.92      0.92      0.92       199
    
        Predicted labels:  [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 1 0 0 0 0 0 0 0 0 1 0 0 0 0
        0 0 0 0 0 1 0 1 0 0 1 0 0 0 0 0 1 0 1 0 1 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0
        1 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 1 0 1 0 0 0 0 0 0 0 1 0
        0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 1 0 0 0 1 1 0
        0 0 0 0 1 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1
        0 0 0 1 0 0 0 0 0 0 0 0 1 1]
        F1-score:  0.7457627118644068
    ```

## 혼동 행렬을 통한 더 나은 이해

위에서 항목을 출력하여 [용어](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html?highlight=classification_report#sklearn.metrics.classification_report)를 보고 점수표를 확인할 수 있지만, 모델을 더 쉽게 이해하려면 [혼동 행렬](https://scikit-learn.org/stable/modules/model_evaluation.html#confusion-matrix)을 사용하여 모델 성능을 이해할 수 있습니다.

> 🎓 '[혼동 행렬](https://wikipedia.org/wiki/Confusion_matrix)'(또는 '오류 행렬')은 모델의 실제 vs. 예측된 긍정 및 부정을 표현하는 표로, 예측 정확도를 측정합니다.

1. 혼동 행렬을 사용하려면 `confusion_matrix()`를 호출하세요:

    ```python
    from sklearn.metrics import confusion_matrix
    confusion_matrix(y_test, predictions)
    ```

    모델의 혼동 행렬을 확인하세요:

    ```output
    array([[162,   4],
           [ 11,  22]])
    ```

Scikit-learn에서 혼동 행렬의 행(축 0)은 실제 라벨이고 열(축 1)은 예측된 라벨입니다.

|       |   0   |   1   |
| :---: | :---: | :---: |
|   0   |  TN   |  FP   |
|   1   |  FN   |  TP   |

여기서 무슨 일이 일어나고 있을까요? 모델이 호박을 두 가지 이진 카테고리, '흰색'과 '흰색이 아님'으로 분류하도록 요청받았다고 가정해 봅시다.

- 모델이 호박을 흰색이 아니라고 예측했는데 실제로 '흰색이 아님' 카테고리에 속한다면 이를 참 부정(true negative)이라고 하며, 왼쪽 상단 숫자로 표시됩니다.
- 모델이 호박을 흰색이라고 예측했는데 실제로 '흰색이 아님' 카테고리에 속한다면 이를 거짓 부정(false negative)이라고 하며, 왼쪽 하단 숫자로 표시됩니다.
- 모델이 호박을 흰색이 아니라고 예측했는데 실제로 '흰색' 카테고리에 속한다면 이를 거짓 긍정(false positive)이라고 하며, 오른쪽 상단 숫자로 표시됩니다.
- 모델이 호박을 흰색이라고 예측했는데 실제로 '흰색' 카테고리에 속한다면 이를 참 긍정(true positive)이라고 하며, 오른쪽 하단 숫자로 표시됩니다.

예상하셨겠지만, 참 긍정과 참 부정의 숫자가 많고 거짓 긍정과 거짓 부정의 숫자가 적을수록 모델 성능이 더 좋다는 것을 의미합니다.
혼동 행렬은 정밀도와 재현율과 어떻게 관련이 있을까요? 위에서 출력된 분류 보고서는 정밀도(0.85)와 재현율(0.67)을 보여줍니다.

정밀도 = tp / (tp + fp) = 22 / (22 + 4) = 0.8461538461538461

재현율 = tp / (tp + fn) = 22 / (22 + 11) = 0.6666666666666666

✅ Q: 혼동 행렬에 따르면 모델의 성능은 어땠나요?  
A: 나쁘지 않습니다. 참 음성(true negatives)이 꽤 많지만, 몇 개의 거짓 음성(false negatives)도 있습니다.

혼동 행렬의 TP/TN 및 FP/FN 매핑을 통해 이전에 본 용어들을 다시 살펴봅시다:

🎓 정밀도(Precision): TP/(TP + FP)  
검색된 인스턴스 중에서 관련 있는 인스턴스의 비율 (예: 잘 분류된 레이블)

🎓 재현율(Recall): TP/(TP + FN)  
검색된 인스턴스 중에서 관련 있는 인스턴스의 비율, 잘 분류되었는지 여부와 관계없이

🎓 f1-score: (2 * 정밀도 * 재현율)/(정밀도 + 재현율)  
정밀도와 재현율의 가중 평균, 최고는 1, 최악은 0

🎓 지원(Support): 검색된 각 레이블의 발생 횟수

🎓 정확도(Accuracy): (TP + TN)/(TP + TN + FP + FN)  
샘플에 대해 정확히 예측된 레이블의 비율

🎓 매크로 평균(Macro Avg):  
레이블 불균형을 고려하지 않고 각 레이블에 대한 비중 없는 평균 지표 계산

🎓 가중 평균(Weighted Avg):  
레이블 불균형을 고려하여 각 레이블에 대한 평균 지표를 지원(각 레이블의 실제 인스턴스 수)으로 가중 계산

✅ 거짓 음성(false negatives)의 수를 줄이고 싶다면 어떤 지표를 주의 깊게 봐야 할지 생각해볼 수 있나요?

## 이 모델의 ROC 곡선 시각화

[![초보자를 위한 머신러닝 - ROC 곡선을 사용한 로지스틱 회귀 성능 분석](https://img.youtube.com/vi/GApO575jTA0/0.jpg)](https://youtu.be/GApO575jTA0 "초보자를 위한 머신러닝 - ROC 곡선을 사용한 로지스틱 회귀 성능 분석")

> 🎥 위 이미지를 클릭하면 ROC 곡선에 대한 짧은 영상 개요를 볼 수 있습니다.

이른바 'ROC' 곡선을 보기 위해 한 번 더 시각화를 진행해 봅시다:

```python
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib
import matplotlib.pyplot as plt
%matplotlib inline

y_scores = model.predict_proba(X_test)
fpr, tpr, thresholds = roc_curve(y_test, y_scores[:,1])

fig = plt.figure(figsize=(6, 6))
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()
```

Matplotlib을 사용하여 모델의 [수신 운영 특성(ROC)](https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html?highlight=roc) 또는 ROC를 그립니다. ROC 곡선은 분류기의 출력(참 양성 대 거짓 양성)을 확인하는 데 자주 사용됩니다. "ROC 곡선은 일반적으로 Y축에 참 양성 비율, X축에 거짓 양성 비율을 표시합니다." 따라서 곡선의 가파름과 중간선과 곡선 사이의 공간이 중요합니다. 곡선이 빠르게 위로 올라가고 선을 넘어가는 것이 이상적입니다. 우리의 경우, 처음에는 거짓 양성이 있지만 이후 곡선이 제대로 위로 올라가고 넘어갑니다:

![ROC](../../../../translated_images/ROC_2.777f20cdfc4988ca683ade6850ac832cb70c96c12f1b910d294f270ef36e1a1c.ko.png)

마지막으로 Scikit-learn의 [`roc_auc_score` API](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html?highlight=roc_auc#sklearn.metrics.roc_auc_score)를 사용하여 실제 '곡선 아래 면적(AUC)'을 계산합니다:

```python
auc = roc_auc_score(y_test,y_scores[:,1])
print(auc)
```  
결과는 `0.9749908725812341`입니다. AUC는 0에서 1 사이의 값을 가지며, 예측이 100% 정확한 모델은 AUC가 1이 됩니다. 이 경우, 모델은 _꽤 괜찮습니다_.

향후 분류에 대한 수업에서는 모델의 점수를 개선하기 위해 반복하는 방법을 배우게 될 것입니다. 하지만 지금은 축하합니다! 이 회귀 수업을 완료했습니다!

---
## 🚀도전 과제

로지스틱 회귀에 대해 더 알아볼 것이 많습니다! 하지만 배우는 가장 좋은 방법은 실험하는 것입니다. 이 유형의 분석에 적합한 데이터셋을 찾아 모델을 만들어 보세요. 무엇을 배우게 될까요? 팁: [Kaggle](https://www.kaggle.com/search?q=logistic+regression+datasets)에서 흥미로운 데이터셋을 찾아보세요.

## [강의 후 퀴즈](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/16/)

## 복습 및 자기 학습

[스탠포드의 이 논문](https://web.stanford.edu/~jurafsky/slp3/5.pdf)의 첫 몇 페이지를 읽고 로지스틱 회귀의 실용적인 사용 사례를 살펴보세요. 지금까지 공부한 회귀 유형 중 어떤 작업에 더 적합할지 생각해 보세요. 어떤 것이 가장 잘 작동할까요?

## 과제

[이 회귀 다시 시도하기](assignment.md)

---

**면책 조항**:  
이 문서는 AI 번역 서비스 [Co-op Translator](https://github.com/Azure/co-op-translator)를 사용하여 번역되었습니다. 정확성을 위해 최선을 다하고 있지만, 자동 번역에는 오류나 부정확성이 포함될 수 있습니다. 원본 문서를 해당 언어로 작성된 상태에서 권위 있는 자료로 간주해야 합니다. 중요한 정보의 경우, 전문적인 인간 번역을 권장합니다. 이 번역 사용으로 인해 발생하는 오해나 잘못된 해석에 대해 책임을 지지 않습니다.