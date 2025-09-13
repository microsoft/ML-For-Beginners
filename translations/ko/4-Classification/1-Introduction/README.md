<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "aaf391d922bd6de5efba871d514c6d47",
  "translation_date": "2025-09-05T10:52:28+00:00",
  "source_file": "4-Classification/1-Introduction/README.md",
  "language_code": "ko"
}
-->
# 분류 소개

이 네 가지 강의에서는 고전적인 머신 러닝의 핵심 주제인 _분류_를 탐구합니다. 아시아와 인도의 다양한 요리 데이터를 사용하여 여러 분류 알고리즘을 살펴볼 것입니다. 배가 고프셨다면 잘 오셨습니다!

![한 꼬집만!](../../../../4-Classification/1-Introduction/images/pinch.png)

> 이 강의에서 범아시아 요리를 축하하세요! 이미지 제공: [Jen Looper](https://twitter.com/jenlooper)

분류는 [지도 학습](https://wikipedia.org/wiki/Supervised_learning)의 한 형태로, 회귀 기법과 많은 공통점을 가지고 있습니다. 머신 러닝이 데이터셋을 사용하여 값이나 이름을 예측하는 것이라면, 분류는 일반적으로 두 가지 그룹으로 나뉩니다: _이진 분류_와 _다중 클래스 분류_.

[![분류 소개](https://img.youtube.com/vi/eg8DJYwdMyg/0.jpg)](https://youtu.be/eg8DJYwdMyg "분류 소개")

> 🎥 위 이미지를 클릭하면 MIT의 John Guttag가 분류를 소개하는 영상을 볼 수 있습니다.

기억하세요:

- **선형 회귀**는 변수 간의 관계를 예측하고 새로운 데이터 포인트가 그 선과 어떤 관계에 있는지 정확히 예측할 수 있도록 도와줍니다. 예를 들어, _9월과 12월에 호박 가격이 얼마일지_ 예측할 수 있습니다.
- **로지스틱 회귀**는 "이진 카테고리"를 발견하는 데 도움을 줍니다: 특정 가격대에서, _이 호박이 주황색인지 아닌지_를 예측할 수 있습니다.

분류는 다양한 알고리즘을 사용하여 데이터 포인트의 레이블이나 클래스를 결정하는 방법을 제공합니다. 이 요리 데이터를 사용하여 재료 그룹을 관찰함으로써 해당 요리의 출처를 결정할 수 있는지 살펴보겠습니다.

## [강의 전 퀴즈](https://ff-quizzes.netlify.app/en/ml/)

> ### [이 강의는 R에서도 제공됩니다!](../../../../4-Classification/1-Introduction/solution/R/lesson_10.html)

### 소개

분류는 머신 러닝 연구자와 데이터 과학자의 기본 활동 중 하나입니다. 이진 값의 기본 분류("이 이메일이 스팸인지 아닌지")부터 컴퓨터 비전을 사용한 복잡한 이미지 분류 및 세분화까지, 데이터를 클래스별로 정렬하고 질문을 던질 수 있는 능력은 항상 유용합니다.

과정을 좀 더 과학적으로 설명하자면, 분류 방법은 입력 변수와 출력 변수 간의 관계를 매핑할 수 있는 예측 모델을 생성합니다.

![이진 vs. 다중 클래스 분류](../../../../4-Classification/1-Introduction/images/binary-multiclass.png)

> 분류 알고리즘이 처리해야 할 이진 문제와 다중 클래스 문제. 인포그래픽 제공: [Jen Looper](https://twitter.com/jenlooper)

데이터를 정리하고 시각화하며 머신 러닝 작업을 준비하기 전에, 머신 러닝이 데이터를 분류하는 다양한 방법에 대해 조금 배워봅시다.

[통계학](https://wikipedia.org/wiki/Statistical_classification)에서 유래된 고전적인 머신 러닝을 사용한 분류는 `smoker`, `weight`, `age`와 같은 특징을 사용하여 _X 질병에 걸릴 가능성_을 결정합니다. 이전에 수행한 회귀 연습과 유사한 지도 학습 기법으로, 데이터는 레이블이 지정되어 있으며 머신 러닝 알고리즘은 이러한 레이블을 사용하여 데이터셋의 클래스(또는 '특징')를 분류하고 그룹이나 결과에 할당합니다.

✅ 요리에 대한 데이터셋을 상상해 보세요. 다중 클래스 모델은 어떤 질문에 답할 수 있을까요? 이진 모델은 어떤 질문에 답할 수 있을까요? 특정 요리가 페넬그릭을 사용할 가능성을 결정하고 싶다면 어떻게 할까요? 아니면 스타 아니스, 아티초크, 콜리플라워, 고추냉이가 가득 든 장바구니를 선물로 받았을 때, 전형적인 인도 요리를 만들 수 있을지 알고 싶다면 어떻게 할까요?

[![미스터리 바구니](https://img.youtube.com/vi/GuTeDbaNoEU/0.jpg)](https://youtu.be/GuTeDbaNoEU "미스터리 바구니")

> 🎥 위 이미지를 클릭하면 영상을 볼 수 있습니다. 'Chopped'라는 쇼의 전체 전제는 '미스터리 바구니'로, 셰프들이 무작위로 선택된 재료로 요리를 만들어야 합니다. 머신 러닝 모델이 도움이 되었을지도 모르겠네요!

## 안녕하세요, '분류기'

이 요리 데이터셋에서 우리가 묻고 싶은 질문은 사실 **다중 클래스 질문**입니다. 여러 국가 요리 중에서, 주어진 재료 그룹이 어떤 클래스에 속할지 결정해야 합니다.

Scikit-learn은 문제 유형에 따라 데이터를 분류하는 데 사용할 수 있는 여러 알고리즘을 제공합니다. 다음 두 강의에서는 이러한 알고리즘 중 몇 가지를 배울 것입니다.

## 실습 - 데이터 정리 및 균형 맞추기

이 프로젝트를 시작하기 전에 첫 번째 작업은 데이터를 정리하고 **균형을 맞추는 것**입니다. 더 나은 결과를 얻기 위해 데이터를 정리하세요. 이 폴더의 루트에 있는 빈 _notebook.ipynb_ 파일에서 시작하세요.

첫 번째로 설치해야 할 것은 [imblearn](https://imbalanced-learn.org/stable/)입니다. 이는 데이터를 더 잘 균형 맞출 수 있도록 도와주는 Scikit-learn 패키지입니다(이 작업에 대해 곧 더 배우게 될 것입니다).

1. `imblearn`을 설치하려면 `pip install`을 실행하세요:

    ```python
    pip install imblearn
    ```

1. 데이터를 가져오고 시각화하기 위해 필요한 패키지를 가져오세요. 또한 `imblearn`에서 `SMOTE`를 가져오세요.

    ```python
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    import numpy as np
    from imblearn.over_sampling import SMOTE
    ```

    이제 데이터를 가져올 준비가 되었습니다.

1. 다음 작업은 데이터를 가져오는 것입니다:

    ```python
    df  = pd.read_csv('../data/cuisines.csv')
    ```

   `read_csv()`를 사용하면 _cusines.csv_ 파일의 내용을 읽고 변수 `df`에 저장합니다.

1. 데이터의 형태를 확인하세요:

    ```python
    df.head()
    ```

   처음 다섯 행은 다음과 같습니다:

    ```output
    |     | Unnamed: 0 | cuisine | almond | angelica | anise | anise_seed | apple | apple_brandy | apricot | armagnac | ... | whiskey | white_bread | white_wine | whole_grain_wheat_flour | wine | wood | yam | yeast | yogurt | zucchini |
    | --- | ---------- | ------- | ------ | -------- | ----- | ---------- | ----- | ------------ | ------- | -------- | --- | ------- | ----------- | ---------- | ----------------------- | ---- | ---- | --- | ----- | ------ | -------- |
    | 0   | 65         | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
    | 1   | 66         | indian  | 1      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
    | 2   | 67         | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
    | 3   | 68         | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
    | 4   | 69         | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 1      | 0        |
    ```

1. `info()`를 호출하여 데이터에 대한 정보를 얻으세요:

    ```python
    df.info()
    ```

    출력은 다음과 같습니다:

    ```output
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 2448 entries, 0 to 2447
    Columns: 385 entries, Unnamed: 0 to zucchini
    dtypes: int64(384), object(1)
    memory usage: 7.2+ MB
    ```

## 실습 - 요리에 대해 배우기

이제 작업이 더 흥미로워지기 시작합니다. 각 요리별 데이터 분포를 알아봅시다.

1. `barh()`를 호출하여 데이터를 막대로 시각화하세요:

    ```python
    df.cuisine.value_counts().plot.barh()
    ```

    ![요리 데이터 분포](../../../../4-Classification/1-Introduction/images/cuisine-dist.png)

    요리의 수는 유한하지만 데이터 분포는 고르지 않습니다. 이를 수정할 수 있습니다! 수정하기 전에 조금 더 탐색해 보세요.

1. 요리별로 사용 가능한 데이터 양을 확인하고 출력하세요:

    ```python
    thai_df = df[(df.cuisine == "thai")]
    japanese_df = df[(df.cuisine == "japanese")]
    chinese_df = df[(df.cuisine == "chinese")]
    indian_df = df[(df.cuisine == "indian")]
    korean_df = df[(df.cuisine == "korean")]
    
    print(f'thai df: {thai_df.shape}')
    print(f'japanese df: {japanese_df.shape}')
    print(f'chinese df: {chinese_df.shape}')
    print(f'indian df: {indian_df.shape}')
    print(f'korean df: {korean_df.shape}')
    ```

    출력은 다음과 같습니다:

    ```output
    thai df: (289, 385)
    japanese df: (320, 385)
    chinese df: (442, 385)
    indian df: (598, 385)
    korean df: (799, 385)
    ```

## 재료 발견하기

이제 데이터를 더 깊이 탐구하여 각 요리별로 일반적인 재료를 알아볼 수 있습니다. 혼란을 초래하는 반복 데이터를 제거해야 하므로 이 문제에 대해 배워봅시다.

1. Python에서 `create_ingredient()`라는 함수를 만들어 재료 데이터프레임을 생성하세요. 이 함수는 도움이 되지 않는 열을 제거하고 재료를 개수별로 정렬합니다:

    ```python
    def create_ingredient_df(df):
        ingredient_df = df.T.drop(['cuisine','Unnamed: 0']).sum(axis=1).to_frame('value')
        ingredient_df = ingredient_df[(ingredient_df.T != 0).any()]
        ingredient_df = ingredient_df.sort_values(by='value', ascending=False,
        inplace=False)
        return ingredient_df
    ```

   이제 이 함수를 사용하여 요리별로 가장 인기 있는 상위 10개 재료를 파악할 수 있습니다.

1. `create_ingredient()`를 호출하고 `barh()`를 호출하여 시각화하세요:

    ```python
    thai_ingredient_df = create_ingredient_df(thai_df)
    thai_ingredient_df.head(10).plot.barh()
    ```

    ![태국 요리](../../../../4-Classification/1-Introduction/images/thai.png)

1. 일본 데이터를 동일한 방식으로 처리하세요:

    ```python
    japanese_ingredient_df = create_ingredient_df(japanese_df)
    japanese_ingredient_df.head(10).plot.barh()
    ```

    ![일본 요리](../../../../4-Classification/1-Introduction/images/japanese.png)

1. 중국 요리 재료를 시각화하세요:

    ```python
    chinese_ingredient_df = create_ingredient_df(chinese_df)
    chinese_ingredient_df.head(10).plot.barh()
    ```

    ![중국 요리](../../../../4-Classification/1-Introduction/images/chinese.png)

1. 인도 요리 재료를 시각화하세요:

    ```python
    indian_ingredient_df = create_ingredient_df(indian_df)
    indian_ingredient_df.head(10).plot.barh()
    ```

    ![인도 요리](../../../../4-Classification/1-Introduction/images/indian.png)

1. 마지막으로 한국 요리 재료를 시각화하세요:

    ```python
    korean_ingredient_df = create_ingredient_df(korean_df)
    korean_ingredient_df.head(10).plot.barh()
    ```

    ![한국 요리](../../../../4-Classification/1-Introduction/images/korean.png)

1. 이제 `drop()`을 호출하여 서로 다른 요리 간 혼란을 초래하는 가장 일반적인 재료를 제거하세요:

   모두가 쌀, 마늘, 생강을 좋아합니다!

    ```python
    feature_df= df.drop(['cuisine','Unnamed: 0','rice','garlic','ginger'], axis=1)
    labels_df = df.cuisine #.unique()
    feature_df.head()
    ```

## 데이터셋 균형 맞추기

데이터를 정리한 후, [SMOTE](https://imbalanced-learn.org/dev/references/generated/imblearn.over_sampling.SMOTE.html) - "Synthetic Minority Over-sampling Technique"를 사용하여 데이터를 균형 맞추세요.

1. `fit_resample()`을 호출하세요. 이 전략은 보간법을 통해 새로운 샘플을 생성합니다.

    ```python
    oversample = SMOTE()
    transformed_feature_df, transformed_label_df = oversample.fit_resample(feature_df, labels_df)
    ```

    데이터를 균형 맞춤으로써 분류 시 더 나은 결과를 얻을 수 있습니다. 이진 분류를 생각해 보세요. 대부분의 데이터가 한 클래스에 속한다면, 머신 러닝 모델은 단순히 데이터가 더 많기 때문에 해당 클래스를 더 자주 예측할 것입니다. 데이터를 균형 맞추면 이러한 불균형을 제거하는 데 도움이 됩니다.

1. 이제 재료별 레이블 수를 확인할 수 있습니다:

    ```python
    print(f'new label count: {transformed_label_df.value_counts()}')
    print(f'old label count: {df.cuisine.value_counts()}')
    ```

    출력은 다음과 같습니다:

    ```output
    new label count: korean      799
    chinese     799
    indian      799
    japanese    799
    thai        799
    Name: cuisine, dtype: int64
    old label count: korean      799
    indian      598
    chinese     442
    japanese    320
    thai        289
    Name: cuisine, dtype: int64
    ```

    데이터가 깔끔하고 균형 잡혔으며 매우 맛있어졌습니다!

1. 마지막 단계는 레이블과 특징을 포함한 균형 잡힌 데이터를 새 데이터프레임에 저장하여 파일로 내보내는 것입니다:

    ```python
    transformed_df = pd.concat([transformed_label_df,transformed_feature_df],axis=1, join='outer')
    ```

1. `transformed_df.head()`와 `transformed_df.info()`를 사용하여 데이터를 한 번 더 확인하세요. 이 데이터를 저장하여 향후 강의에서 사용하세요:

    ```python
    transformed_df.head()
    transformed_df.info()
    transformed_df.to_csv("../data/cleaned_cuisines.csv")
    ```

    이 새 CSV는 이제 루트 데이터 폴더에서 찾을 수 있습니다.

---

## 🚀도전 과제

이 커리큘럼에는 여러 흥미로운 데이터셋이 포함되어 있습니다. `data` 폴더를 탐색하여 이진 또는 다중 클래스 분류에 적합한 데이터셋이 있는지 확인하세요. 이 데이터셋에 대해 어떤 질문을 던질 수 있을까요?

## [강의 후 퀴즈](https://ff-quizzes.netlify.app/en/ml/)

## 복습 및 자기 학습

SMOTE의 API를 탐색하세요. 어떤 사용 사례에 가장 적합하며, 어떤 문제를 해결할 수 있을까요?

## 과제 

[분류 방법 탐구하기](assignment.md)

---

**면책 조항**:  
이 문서는 AI 번역 서비스 [Co-op Translator](https://github.com/Azure/co-op-translator)를 사용하여 번역되었습니다. 정확성을 위해 최선을 다하고 있지만, 자동 번역에는 오류나 부정확성이 포함될 수 있습니다. 원본 문서를 해당 언어로 작성된 상태에서 권위 있는 자료로 간주해야 합니다. 중요한 정보의 경우, 전문적인 인간 번역을 권장합니다. 이 번역 사용으로 인해 발생하는 오해나 잘못된 해석에 대해 당사는 책임을 지지 않습니다.  