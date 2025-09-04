<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "2f88fbc741d792890ff2f1430fe0dae0",
  "translation_date": "2025-09-03T22:18:04+00:00",
  "source_file": "2-Regression/3-Linear/README.md",
  "language_code": "ko"
}
-->
# Scikit-learn을 사용하여 회귀 모델 구축: 네 가지 방법으로 회귀 분석하기

![선형 회귀 vs 다항 회귀 인포그래픽](../../../../translated_images/linear-polynomial.5523c7cb6576ccab0fecbd0e3505986eb2d191d9378e785f82befcf3a578a6e7.ko.png)
> 인포그래픽 제작: [Dasani Madipalli](https://twitter.com/dasani_decoded)
## [강의 전 퀴즈](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/13/)

> ### [이 강의는 R에서도 제공됩니다!](../../../../2-Regression/3-Linear/solution/R/lesson_3.html)
### 소개

지금까지 호박 가격 데이터셋을 사용하여 회귀 분석이 무엇인지 탐구하고, Matplotlib을 사용하여 시각화하는 방법을 배웠습니다.

이제 머신러닝을 위한 회귀 분석을 더 깊이 탐구할 준비가 되었습니다. 시각화를 통해 데이터를 이해할 수 있지만, 머신러닝의 진정한 힘은 _모델 훈련_에서 나옵니다. 모델은 과거 데이터를 기반으로 데이터 간의 의존성을 자동으로 포착하며, 이전에 본 적 없는 새로운 데이터에 대한 결과를 예측할 수 있습니다.

이번 강의에서는 두 가지 유형의 회귀 분석, 즉 _기본 선형 회귀_와 _다항 회귀_에 대해 배우고, 이러한 기술의 수학적 기초를 살펴볼 것입니다. 이러한 모델을 사용하여 다양한 입력 데이터를 기반으로 호박 가격을 예측할 수 있습니다.

[![초보자를 위한 머신러닝 - 선형 회귀 이해하기](https://img.youtube.com/vi/CRxFT8oTDMg/0.jpg)](https://youtu.be/CRxFT8oTDMg "초보자를 위한 머신러닝 - 선형 회귀 이해하기")

> 🎥 위 이미지를 클릭하면 선형 회귀에 대한 짧은 영상 개요를 볼 수 있습니다.

> 이 커리큘럼에서는 수학에 대한 최소한의 지식을 가정하며, 다른 분야에서 온 학생들이 이해할 수 있도록 접근성을 높이기 위해 노트, 🧮 수학 팁, 다이어그램 및 기타 학습 도구를 제공합니다.

### 사전 요구사항

이제 우리가 분석할 호박 데이터의 구조에 익숙해졌을 것입니다. 이 강의의 _notebook.ipynb_ 파일에서 미리 로드되고 정리된 데이터를 찾을 수 있습니다. 이 파일에서는 호박 가격이 새로운 데이터 프레임에서 부셸당 표시됩니다. Visual Studio Code의 커널에서 이 노트북을 실행할 수 있는지 확인하세요.

### 준비

데이터를 로드하는 이유는 질문을 던지기 위함입니다.

- 호박을 구매하기 가장 좋은 시기는 언제인가요?
- 미니어처 호박 한 상자의 가격은 얼마일까요?
- 반 부셸 바구니로 구매하는 것이 좋을까요, 아니면 1 1/9 부셸 상자로 구매하는 것이 좋을까요?
이 데이터를 계속 탐구해 봅시다.

이전 강의에서는 Pandas 데이터 프레임을 생성하고 원본 데이터셋의 일부를 표준화하여 부셸당 가격을 계산했습니다. 그러나 그렇게 함으로써 약 400개의 데이터 포인트만 얻을 수 있었고, 가을철 데이터만 포함되었습니다.

이번 강의의 노트북에 미리 로드된 데이터를 살펴보세요. 데이터는 미리 로드되어 있으며, 초기 산점도가 월별 데이터를 보여주도록 차트화되어 있습니다. 데이터를 더 정리하면 데이터의 특성을 조금 더 자세히 알 수 있을지도 모릅니다.

## 선형 회귀선

1강에서 배운 것처럼, 선형 회귀 분석의 목표는 다음을 가능하게 하는 선을 그리는 것입니다:

- **변수 관계를 보여주기**. 변수 간의 관계를 보여줍니다.
- **예측하기**. 새로운 데이터 포인트가 이 선과의 관계에서 어디에 위치할지 정확히 예측합니다.

**최소 제곱 회귀**는 이러한 유형의 선을 그리는 데 일반적으로 사용됩니다. '최소 제곱'이라는 용어는 회귀선 주변의 모든 데이터 포인트를 제곱한 다음 합산한다는 것을 의미합니다. 이상적으로는 최종 합계가 가능한 한 작아야 합니다. 이는 오류가 적다는 것을 의미하며, 즉 `최소 제곱`을 원합니다.

우리는 데이터 포인트와 선 사이의 누적 거리가 가장 적은 선을 모델링하고자 합니다. 또한 방향보다는 크기에 관심이 있기 때문에 항목을 제곱한 후 합산합니다.

> **🧮 수학을 보여주세요**
>
> 이 선은 _최적의 적합선_이라고 하며, [다음 방정식](https://en.wikipedia.org/wiki/Simple_linear_regression)으로 표현할 수 있습니다:
>
> ```
> Y = a + bX
> ```
>
> `X`는 '설명 변수'이고, `Y`는 '종속 변수'입니다. 선의 기울기는 `b`이고, `a`는 y절편으로, 이는 `X = 0`일 때 `Y`의 값을 나타냅니다.
>
>![기울기 계산](../../../../translated_images/slope.f3c9d5910ddbfcf9096eb5564254ba22c9a32d7acd7694cab905d29ad8261db3.ko.png)
>
> 먼저 기울기 `b`를 계산합니다. 인포그래픽 제작: [Jen Looper](https://twitter.com/jenlooper)
>
> 즉, 호박 데이터의 원래 질문인 "월별로 부셸당 호박 가격을 예측한다"를 참조하면, `X`는 가격을 나타내고 `Y`는 판매 월을 나타냅니다.
>
>![방정식 완성](../../../../translated_images/calculation.a209813050a1ddb141cdc4bc56f3af31e67157ed499e16a2ecf9837542704c94.ko.png)
>
> `Y` 값을 계산합니다. 약 $4를 지불하고 있다면, 아마도 4월일 것입니다! 인포그래픽 제작: [Jen Looper](https://twitter.com/jenlooper)
>
> 선의 기울기를 계산하는 수학은 y절편, 즉 `X = 0`일 때 `Y`의 위치에 따라 달라집니다.
>
> 이러한 값의 계산 방법은 [Math is Fun](https://www.mathsisfun.com/data/least-squares-regression.html) 웹사이트에서 확인할 수 있습니다. 또한 [최소 제곱 계산기](https://www.mathsisfun.com/data/least-squares-calculator.html)를 방문하여 숫자 값이 선에 어떤 영향을 미치는지 확인하세요.

## 상관관계

이해해야 할 또 다른 용어는 주어진 X와 Y 변수 간의 **상관 계수**입니다. 산점도를 사용하면 이 계수를 빠르게 시각화할 수 있습니다. 데이터 포인트가 깔끔한 선으로 흩어져 있는 플롯은 높은 상관관계를 가지며, 데이터 포인트가 X와 Y 사이에 무작위로 흩어져 있는 플롯은 낮은 상관관계를 가집니다.

좋은 선형 회귀 모델은 높은 상관 계수(0보다 1에 가까운 값)를 가지며, 최소 제곱 회귀 방법과 회귀선을 사용합니다.

✅ 이번 강의의 노트북을 실행하고 월별 가격 산점도를 확인하세요. 호박 판매의 월별 가격 데이터는 산점도를 시각적으로 해석했을 때 높은 상관관계를 가지는 것처럼 보이나요? `Month` 대신 더 세분화된 측정값(예: *연도의 일수*, 즉 연초부터의 일수)을 사용하면 결과가 달라지나요?

아래 코드에서는 데이터를 정리하고 `new_pumpkins`라는 데이터 프레임을 얻었다고 가정합니다. 데이터는 다음과 비슷합니다:

ID | Month | DayOfYear | Variety | City | Package | Low Price | High Price | Price
---|-------|-----------|---------|------|---------|-----------|------------|-------
70 | 9 | 267 | PIE TYPE | BALTIMORE | 1 1/9 bushel cartons | 15.0 | 15.0 | 13.636364
71 | 9 | 267 | PIE TYPE | BALTIMORE | 1 1/9 bushel cartons | 18.0 | 18.0 | 16.363636
72 | 10 | 274 | PIE TYPE | BALTIMORE | 1 1/9 bushel cartons | 18.0 | 18.0 | 16.363636
73 | 10 | 274 | PIE TYPE | BALTIMORE | 1 1/9 bushel cartons | 17.0 | 17.0 | 15.454545
74 | 10 | 281 | PIE TYPE | BALTIMORE | 1 1/9 bushel cartons | 15.0 | 15.0 | 13.636364

> 데이터를 정리하는 코드는 [`notebook.ipynb`](notebook.ipynb)에 있습니다. 이전 강의와 동일한 정리 단계를 수행했으며, 다음 표현식을 사용하여 `DayOfYear` 열을 계산했습니다:

```python
day_of_year = pd.to_datetime(pumpkins['Date']).apply(lambda dt: (dt-datetime(dt.year,1,1)).days)
```

이제 선형 회귀의 수학적 기초를 이해했으니, 호박 패키지 중 가장 좋은 가격을 가진 패키지를 예측할 수 있는 회귀 모델을 만들어 봅시다. 휴일 호박 패치를 위해 호박을 구매하려는 사람은 패치에 최적화된 호박 패키지를 구매하기 위해 이 정보를 원할 수 있습니다.

## 상관관계 찾기

[![초보자를 위한 머신러닝 - 상관관계 찾기: 선형 회귀의 핵심](https://img.youtube.com/vi/uoRq-lW2eQo/0.jpg)](https://youtu.be/uoRq-lW2eQo "초보자를 위한 머신러닝 - 상관관계 찾기: 선형 회귀의 핵심")

> 🎥 위 이미지를 클릭하면 상관관계에 대한 짧은 영상 개요를 볼 수 있습니다.

이전 강의에서 월별 평균 가격이 다음과 같다는 것을 보았을 것입니다:

<img alt="월별 평균 가격" src="../2-Data/images/barchart.png" width="50%"/>

이는 상관관계가 있을 가능성을 시사하며, `Month`와 `Price` 또는 `DayOfYear`와 `Price` 간의 관계를 예측하기 위해 선형 회귀 모델을 훈련시킬 수 있습니다. 아래는 후자의 관계를 보여주는 산점도입니다:

<img alt="가격 vs 연도의 일수 산점도" src="images/scatter-dayofyear.png" width="50%" /> 

`corr` 함수를 사용하여 상관관계를 확인해 봅시다:

```python
print(new_pumpkins['Month'].corr(new_pumpkins['Price']))
print(new_pumpkins['DayOfYear'].corr(new_pumpkins['Price']))
```

상관관계는 `Month` 기준으로 -0.15, `DayOfYear` 기준으로 -0.17로 비교적 낮아 보이지만, 다른 중요한 관계가 있을 수 있습니다. 가격이 호박 품종에 따라 다른 클러스터로 나뉘는 것처럼 보입니다. 이 가설을 확인하기 위해 각 호박 카테고리를 다른 색상으로 표시해 봅시다. `scatter` 플롯 함수에 `ax` 매개변수를 전달하여 모든 포인트를 동일한 그래프에 표시할 수 있습니다:

```python
ax=None
colors = ['red','blue','green','yellow']
for i,var in enumerate(new_pumpkins['Variety'].unique()):
    df = new_pumpkins[new_pumpkins['Variety']==var]
    ax = df.plot.scatter('DayOfYear','Price',ax=ax,c=colors[i],label=var)
```

<img alt="가격 vs 연도의 일수 산점도" src="images/scatter-dayofyear-color.png" width="50%" /> 

우리의 조사 결과는 판매 날짜보다 품종이 전체 가격에 더 큰 영향을 미친다는 것을 시사합니다. 이를 막대 그래프로 확인할 수 있습니다:

```python
new_pumpkins.groupby('Variety')['Price'].mean().plot(kind='bar')
```

<img alt="품종별 가격 막대 그래프" src="images/price-by-variety.png" width="50%" /> 

잠시 동안 '파이 타입'이라는 하나의 호박 품종에만 집중하여 날짜가 가격에 미치는 영향을 살펴봅시다:

```python
pie_pumpkins = new_pumpkins[new_pumpkins['Variety']=='PIE TYPE']
pie_pumpkins.plot.scatter('DayOfYear','Price') 
```
<img alt="가격 vs 연도의 일수 산점도" src="images/pie-pumpkins-scatter.png" width="50%" /> 

이제 `corr` 함수를 사용하여 `Price`와 `DayOfYear` 간의 상관관계를 계산하면 약 `-0.27` 정도가 나옵니다. 이는 예측 모델을 훈련시키는 것이 의미가 있다는 것을 나타냅니다.

> 선형 회귀 모델을 훈련시키기 전에 데이터가 깨끗한지 확인하는 것이 중요합니다. 선형 회귀는 결측값과 잘 작동하지 않으므로 모든 빈 셀을 제거하는 것이 좋습니다:

```python
pie_pumpkins.dropna(inplace=True)
pie_pumpkins.info()
```

또 다른 접근법은 빈 값을 해당 열의 평균 값으로 채우는 것입니다.

## 간단한 선형 회귀

[![초보자를 위한 머신러닝 - Scikit-learn을 사용한 선형 및 다항 회귀](https://img.youtube.com/vi/e4c_UP2fSjg/0.jpg)](https://youtu.be/e4c_UP2fSjg "초보자를 위한 머신러닝 - Scikit-learn을 사용한 선형 및 다항 회귀")

> 🎥 위 이미지를 클릭하면 선형 및 다항 회귀에 대한 짧은 영상 개요를 볼 수 있습니다.

선형 회귀 모델을 훈련시키기 위해 **Scikit-learn** 라이브러리를 사용할 것입니다.

```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
```

먼저 입력 값(특징)과 예상 출력(레이블)을 별도의 numpy 배열로 분리합니다:

```python
X = pie_pumpkins['DayOfYear'].to_numpy().reshape(-1,1)
y = pie_pumpkins['Price']
```

> 입력 데이터를 Linear Regression 패키지가 올바르게 이해할 수 있도록 `reshape`를 수행해야 했다는 점에 유의하세요. 선형 회귀는 2D 배열을 입력으로 기대하며, 배열의 각 행은 입력 특징 벡터에 해당합니다. 우리의 경우 입력이 하나뿐이므로 데이터셋 크기 N×1의 배열이 필요합니다.

그런 다음 데이터를 훈련 및 테스트 데이터셋으로 분리하여 훈련 후 모델을 검증할 수 있도록 합니다:

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
```

마지막으로 실제 선형 회귀 모델을 훈련시키는 데는 코드 두 줄만 필요합니다. `LinearRegression` 객체를 정의하고 `fit` 메서드를 사용하여 데이터를 적용합니다:

```python
lin_reg = LinearRegression()
lin_reg.fit(X_train,y_train)
```

`fit`이 완료된 후 `LinearRegression` 객체는 회귀의 모든 계수를 포함하며, `.coef_` 속성을 사용하여 접근할 수 있습니다. 우리의 경우 계수는 하나뿐이며, 약 `-0.017` 정도일 것입니다. 이는 시간이 지남에 따라 가격이 약간 하락하는 경향이 있음을 나타내며, 하루에 약 2센트 정도입니다. 회귀가 Y축과 교차하는 지점을 `lin_reg.intercept_`를 사용하여 접근할 수 있으며, 우리의 경우 약 `21` 정도로, 연초의 가격을 나타냅니다.

모델의 정확성을 확인하려면 테스트 데이터셋에서 가격을 예측한 다음, 예측 값과 예상 값이 얼마나 가까운지 측정할 수 있습니다. 이는 평균 제곱 오차(MSE) 메트릭을 사용하여 수행할 수 있으며, 이는 예상 값과 예측 값 간의 모든 제곱 차이의 평균입니다.

```python
pred = lin_reg.predict(X_test)

mse = np.sqrt(mean_squared_error(y_test,pred))
print(f'Mean error: {mse:3.3} ({mse/np.mean(pred)*100:3.3}%)')
```
우리의 오류는 약 2개의 포인트에서 발생하는 것으로 보이며, 이는 약 17%입니다. 그다지 좋지 않은 결과입니다. 모델 품질의 또 다른 지표는 **결정 계수**로, 다음과 같이 얻을 수 있습니다:

```python
score = lin_reg.score(X_train,y_train)
print('Model determination: ', score)
```  
값이 0이면 모델이 입력 데이터를 고려하지 않고, 단순히 결과의 평균값을 예측하는 *최악의 선형 예측기*로 작동한다는 것을 의미합니다. 값이 1이면 모든 예상 출력값을 완벽하게 예측할 수 있음을 나타냅니다. 우리의 경우, 결정 계수는 약 0.06으로 매우 낮습니다.

또한 테스트 데이터를 회귀선과 함께 플롯하여 우리의 경우 회귀가 어떻게 작동하는지 더 잘 볼 수 있습니다:

```python
plt.scatter(X_test,y_test)
plt.plot(X_test,pred)
```  

<img alt="선형 회귀" src="images/linear-results.png" width="50%" />

## 다항 회귀

선형 회귀의 또 다른 유형은 다항 회귀입니다. 때로는 변수 간에 선형 관계가 있을 수 있습니다. 예를 들어, 호박의 부피가 클수록 가격이 높아지는 경우가 있습니다. 하지만 이러한 관계가 평면이나 직선으로 표현될 수 없는 경우도 있습니다.

✅ [여기](https://online.stat.psu.edu/stat501/lesson/9/9.8)에서 다항 회귀를 사용할 수 있는 데이터의 몇 가지 예를 확인할 수 있습니다.

날짜와 가격 간의 관계를 다시 살펴보세요. 이 산점도가 반드시 직선으로 분석되어야 할 것처럼 보이나요? 가격은 변동할 수 있지 않을까요? 이 경우, 다항 회귀를 시도해볼 수 있습니다.

✅ 다항식은 하나 이상의 변수와 계수를 포함할 수 있는 수학적 표현입니다.

다항 회귀는 곡선을 생성하여 비선형 데이터를 더 잘 맞춥니다. 우리의 경우, 입력 데이터에 `DayOfYear` 변수의 제곱을 포함하면, 연중 특정 지점에서 최소값을 가지는 포물선 곡선으로 데이터를 맞출 수 있습니다.

Scikit-learn에는 데이터 처리의 여러 단계를 결합할 수 있는 유용한 [파이프라인 API](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.make_pipeline.html?highlight=pipeline#sklearn.pipeline.make_pipeline)가 포함되어 있습니다. **파이프라인**은 **추정기**의 체인입니다. 우리의 경우, 먼저 모델에 다항식 특성을 추가한 다음 회귀를 학습시키는 파이프라인을 생성할 것입니다:

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

pipeline = make_pipeline(PolynomialFeatures(2), LinearRegression())

pipeline.fit(X_train,y_train)
```  

`PolynomialFeatures(2)`를 사용하면 입력 데이터에서 모든 2차 다항식을 포함하게 됩니다. 우리의 경우, 이는 단순히 `DayOfYear`<sup>2</sup>를 의미하지만, 두 개의 입력 변수 X와 Y가 주어지면 X<sup>2</sup>, XY, Y<sup>2</sup>를 추가합니다. 더 높은 차수의 다항식을 사용하고 싶다면 사용할 수도 있습니다.

파이프라인은 원래의 `LinearRegression` 객체와 동일한 방식으로 사용할 수 있습니다. 즉, 파이프라인을 `fit`하고, `predict`를 사용하여 예측 결과를 얻을 수 있습니다. 아래는 테스트 데이터와 근사 곡선을 보여주는 그래프입니다:

<img alt="다항 회귀" src="images/poly-results.png" width="50%" />

다항 회귀를 사용하면 약간 낮은 MSE와 약간 높은 결정 계수를 얻을 수 있지만, 큰 차이는 없습니다. 다른 특성을 고려해야 합니다!

> 최소 호박 가격이 할로윈 즈음에 관찰되는 것을 볼 수 있습니다. 이를 어떻게 설명할 수 있을까요?

🎃 축하합니다! 파이 호박의 가격을 예측하는 데 도움이 되는 모델을 만들었습니다. 아마도 모든 호박 종류에 대해 동일한 절차를 반복할 수 있겠지만, 이는 번거로울 것입니다. 이제 모델에서 호박 종류를 고려하는 방법을 배워봅시다!

## 범주형 특성

이상적인 상황에서는 동일한 모델을 사용하여 다양한 호박 종류의 가격을 예측할 수 있기를 원합니다. 그러나 `Variety` 열은 `Month`와 같은 열과는 다릅니다. 이 열은 숫자가 아닌 값을 포함하고 있습니다. 이러한 열을 **범주형**이라고 합니다.

[![초보자를 위한 ML - 선형 회귀를 사용한 범주형 특성 예측](https://img.youtube.com/vi/DYGliioIAE0/0.jpg)](https://youtu.be/DYGliioIAE0 "초보자를 위한 ML - 선형 회귀를 사용한 범주형 특성 예측")

> 🎥 위 이미지를 클릭하면 범주형 특성을 사용하는 짧은 비디오 개요를 볼 수 있습니다.

여기에서 평균 가격이 종류에 따라 어떻게 달라지는지 확인할 수 있습니다:

<img alt="종류별 평균 가격" src="images/price-by-variety.png" width="50%" />

종류를 고려하려면 먼저 이를 숫자 형태로 변환하거나 **인코딩**해야 합니다. 이를 수행하는 방법은 여러 가지가 있습니다:

* 간단한 **숫자 인코딩**은 다양한 종류의 테이블을 생성한 다음, 종류 이름을 해당 테이블의 인덱스로 대체합니다. 이는 선형 회귀에 가장 좋은 방법은 아닙니다. 선형 회귀는 인덱스의 실제 숫자 값을 가져와 결과에 추가하고 특정 계수를 곱하기 때문입니다. 우리의 경우, 인덱스 번호와 가격 간의 관계는 명확히 비선형적이며, 특정 방식으로 인덱스를 정렬하더라도 마찬가지입니다.
* **원-핫 인코딩**은 `Variety` 열을 각 종류에 대해 4개의 다른 열로 대체합니다. 각 열은 해당 행이 특정 종류인 경우 `1`을 포함하고, 그렇지 않은 경우 `0`을 포함합니다. 이는 선형 회귀에서 각 호박 종류에 대해 "시작 가격"(혹은 "추가 가격")을 담당하는 4개의 계수를 생성합니다.

아래 코드는 종류를 원-핫 인코딩하는 방법을 보여줍니다:

```python
pd.get_dummies(new_pumpkins['Variety'])
```  

 ID | FAIRYTALE | MINIATURE | MIXED HEIRLOOM VARIETIES | PIE TYPE  
----|-----------|-----------|--------------------------|----------  
70 | 0 | 0 | 0 | 1  
71 | 0 | 0 | 0 | 1  
... | ... | ... | ... | ...  
1738 | 0 | 1 | 0 | 0  
1739 | 0 | 1 | 0 | 0  
1740 | 0 | 1 | 0 | 0  
1741 | 0 | 1 | 0 | 0  
1742 | 0 | 1 | 0 | 0  

원-핫 인코딩된 종류를 입력으로 사용하여 선형 회귀를 학습시키려면, `X`와 `y` 데이터를 올바르게 초기화하기만 하면 됩니다:

```python
X = pd.get_dummies(new_pumpkins['Variety'])
y = new_pumpkins['Price']
```  

나머지 코드는 위에서 선형 회귀를 학습시키는 데 사용한 코드와 동일합니다. 이를 시도해보면 평균 제곱 오차는 거의 동일하지만, 결정 계수가 훨씬 높아진다는 것을 알 수 있습니다(~77%). 더 정확한 예측을 위해서는 `Month`나 `DayOfYear`와 같은 숫자 특성을 포함하여 더 많은 범주형 특성을 고려할 수 있습니다. 하나의 큰 특성 배열을 얻으려면 `join`을 사용할 수 있습니다:

```python
X = pd.get_dummies(new_pumpkins['Variety']) \
        .join(new_pumpkins['Month']) \
        .join(pd.get_dummies(new_pumpkins['City'])) \
        .join(pd.get_dummies(new_pumpkins['Package']))
y = new_pumpkins['Price']
```  

여기에서는 `City`와 `Package` 유형도 고려하여 MSE 2.84(10%)와 결정 계수 0.94를 얻습니다!

## 모든 것을 종합하기

최상의 모델을 만들기 위해 위의 예에서 다항 회귀와 함께 결합된 (원-핫 인코딩된 범주형 + 숫자형) 데이터를 사용할 수 있습니다. 아래는 편의를 위해 전체 코드입니다:

```python
# set up training data
X = pd.get_dummies(new_pumpkins['Variety']) \
        .join(new_pumpkins['Month']) \
        .join(pd.get_dummies(new_pumpkins['City'])) \
        .join(pd.get_dummies(new_pumpkins['Package']))
y = new_pumpkins['Price']

# make train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# setup and train the pipeline
pipeline = make_pipeline(PolynomialFeatures(2), LinearRegression())
pipeline.fit(X_train,y_train)

# predict results for test data
pred = pipeline.predict(X_test)

# calculate MSE and determination
mse = np.sqrt(mean_squared_error(y_test,pred))
print(f'Mean error: {mse:3.3} ({mse/np.mean(pred)*100:3.3}%)')

score = pipeline.score(X_train,y_train)
print('Model determination: ', score)
```  

이를 통해 거의 97%의 결정 계수와 MSE=2.23(~8% 예측 오류)을 얻을 수 있습니다.

| 모델 | MSE | 결정 계수 |  
|------|-----|-----------|  
| `DayOfYear` 선형 | 2.77 (17.2%) | 0.07 |  
| `DayOfYear` 다항 | 2.73 (17.0%) | 0.08 |  
| `Variety` 선형 | 5.24 (19.7%) | 0.77 |  
| 모든 특성 선형 | 2.84 (10.5%) | 0.94 |  
| 모든 특성 다항 | 2.23 (8.25%) | 0.97 |  

🏆 잘하셨습니다! 한 수업에서 네 가지 회귀 모델을 만들고 모델 품질을 97%로 개선했습니다. 회귀에 대한 마지막 섹션에서는 범주를 결정하기 위한 로지스틱 회귀에 대해 배울 것입니다.

---

## 🚀도전 과제

이 노트북에서 여러 변수를 테스트하여 상관관계가 모델 정확도와 어떻게 연결되는지 확인하세요.

## [강의 후 퀴즈](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/14/)

## 복습 및 자기 학습

이 수업에서는 선형 회귀에 대해 배웠습니다. 다른 중요한 회귀 유형도 있습니다. Stepwise, Ridge, Lasso 및 Elasticnet 기법에 대해 읽어보세요. 더 많은 것을 배우기 위해 공부할 수 있는 좋은 과정은 [Stanford Statistical Learning course](https://online.stanford.edu/courses/sohs-ystatslearning-statistical-learning)입니다.

## 과제

[모델 만들기](assignment.md)  

---

**면책 조항**:  
이 문서는 AI 번역 서비스 [Co-op Translator](https://github.com/Azure/co-op-translator)를 사용하여 번역되었습니다. 정확성을 위해 최선을 다하고 있으나, 자동 번역에는 오류나 부정확성이 포함될 수 있습니다. 원본 문서의 원어 버전을 권위 있는 출처로 간주해야 합니다. 중요한 정보의 경우, 전문적인 인간 번역을 권장합니다. 이 번역 사용으로 인해 발생하는 오해나 잘못된 해석에 대해 책임을 지지 않습니다.