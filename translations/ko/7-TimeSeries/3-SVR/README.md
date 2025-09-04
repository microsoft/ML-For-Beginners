<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "f80e513b3279869e7661e3190cc83076",
  "translation_date": "2025-09-03T22:52:53+00:00",
  "source_file": "7-TimeSeries/3-SVR/README.md",
  "language_code": "ko"
}
-->
# 지원 벡터 회귀를 활용한 시계열 예측

이전 강의에서는 ARIMA 모델을 사용하여 시계열 예측을 수행하는 방법을 배웠습니다. 이번에는 연속 데이터를 예측하는 데 사용되는 회귀 모델인 지원 벡터 회귀(Support Vector Regressor) 모델을 살펴보겠습니다.

## [강의 전 퀴즈](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/51/) 

## 소개

이번 강의에서는 회귀를 위한 [**SVM**: **S**upport **V**ector **M**achine](https://en.wikipedia.org/wiki/Support-vector_machine) 모델, 즉 **SVR: Support Vector Regressor**를 구축하는 방법을 배웁니다.

### 시계열에서의 SVR [^1]

시계열 예측에서 SVR의 중요성을 이해하기 전에 알아야 할 몇 가지 중요한 개념이 있습니다:

- **회귀:** 주어진 입력 세트에서 연속 값을 예측하는 지도 학습 기법입니다. 이 기법은 특징 공간에서 최대한 많은 데이터 포인트를 포함하는 곡선(또는 선)을 맞추는 것을 목표로 합니다. [여기](https://en.wikipedia.org/wiki/Regression_analysis)를 클릭하여 더 많은 정보를 확인하세요.
- **지원 벡터 머신(SVM):** 분류, 회귀 및 이상치 탐지에 사용되는 지도 학습 모델의 한 유형입니다. 이 모델은 특징 공간에서 초평면으로 작동하며, 분류의 경우 경계로, 회귀의 경우 최적의 선으로 작동합니다. SVM에서는 일반적으로 커널 함수를 사용하여 데이터셋을 더 높은 차원의 공간으로 변환하여 쉽게 분리할 수 있도록 합니다. SVM에 대한 더 많은 정보는 [여기](https://en.wikipedia.org/wiki/Support-vector_machine)를 클릭하세요.
- **지원 벡터 회귀(SVR):** SVM의 한 유형으로, 최대한 많은 데이터 포인트를 포함하는 최적의 선(회귀의 경우 초평면)을 찾습니다.

### 왜 SVR인가? [^1]

지난 강의에서는 시계열 데이터를 예측하는 데 매우 성공적인 통계적 선형 방법인 ARIMA에 대해 배웠습니다. 하지만 많은 경우 시계열 데이터는 *비선형성*을 포함하고 있으며, 이는 선형 모델로는 매핑할 수 없습니다. 이러한 경우, 회귀 작업에서 데이터의 비선형성을 고려할 수 있는 SVM의 능력은 시계열 예측에서 SVR을 성공적으로 만듭니다.

## 실습 - SVR 모델 구축하기

데이터 준비를 위한 초기 단계는 [ARIMA](https://github.com/microsoft/ML-For-Beginners/tree/main/7-TimeSeries/2-ARIMA) 강의와 동일합니다.

이 강의의 [_/working_](https://github.com/microsoft/ML-For-Beginners/tree/main/7-TimeSeries/3-SVR/working) 폴더를 열고 [_notebook.ipynb_](https://github.com/microsoft/ML-For-Beginners/blob/main/7-TimeSeries/3-SVR/working/notebook.ipynb) 파일을 찾으세요. [^2]

1. 노트북을 실행하고 필요한 라이브러리를 가져옵니다: [^2]

   ```python
   import sys
   sys.path.append('../../')
   ```

   ```python
   import os
   import warnings
   import matplotlib.pyplot as plt
   import numpy as np
   import pandas as pd
   import datetime as dt
   import math
   
   from sklearn.svm import SVR
   from sklearn.preprocessing import MinMaxScaler
   from common.utils import load_data, mape
   ```

2. `/data/energy.csv` 파일에서 데이터를 Pandas 데이터프레임으로 로드하고 확인합니다: [^2]

   ```python
   energy = load_data('../../data')[['load']]
   ```

3. 2012년 1월부터 2014년 12월까지의 모든 에너지 데이터를 시각화합니다: [^2]

   ```python
   energy.plot(y='load', subplots=True, figsize=(15, 8), fontsize=12)
   plt.xlabel('timestamp', fontsize=12)
   plt.ylabel('load', fontsize=12)
   plt.show()
   ```

   ![전체 데이터](../../../../translated_images/full-data.a82ec9957e580e976f651a4fc38f280b9229c6efdbe3cfe7c60abaa9486d2cbe.ko.png)

   이제 SVR 모델을 구축해봅시다.

### 학습 및 테스트 데이터셋 생성

데이터가 로드되었으니 이를 학습 및 테스트 세트로 분리합니다. 그런 다음 SVR에 필요한 시계열 기반 데이터셋을 생성하기 위해 데이터를 재구성합니다. 학습 세트에서 모델을 학습시킨 후, 학습 세트, 테스트 세트, 전체 데이터셋에서 모델의 정확도를 평가하여 전체 성능을 확인합니다. 테스트 세트가 학습 세트보다 이후의 기간을 포함하도록 설정하여 모델이 미래 시점의 정보를 얻지 않도록 해야 합니다 [^2] (이를 *과적합*이라고 합니다).

1. 2014년 9월 1일부터 10월 31일까지의 두 달 기간을 학습 세트로 할당합니다. 테스트 세트는 2014년 11월 1일부터 12월 31일까지의 두 달 기간을 포함합니다: [^2]

   ```python
   train_start_dt = '2014-11-01 00:00:00'
   test_start_dt = '2014-12-30 00:00:00'
   ```

2. 차이를 시각화합니다: [^2]

   ```python
   energy[(energy.index < test_start_dt) & (energy.index >= train_start_dt)][['load']].rename(columns={'load':'train'}) \
       .join(energy[test_start_dt:][['load']].rename(columns={'load':'test'}), how='outer') \
       .plot(y=['train', 'test'], figsize=(15, 8), fontsize=12)
   plt.xlabel('timestamp', fontsize=12)
   plt.ylabel('load', fontsize=12)
   plt.show()
   ```

   ![학습 및 테스트 데이터](../../../../translated_images/train-test.ead0cecbfc341921d4875eccf25fed5eefbb860cdbb69cabcc2276c49e4b33e5.ko.png)

### 학습을 위한 데이터 준비

이제 데이터를 필터링하고 스케일링하여 학습을 위한 준비를 합니다. 데이터셋을 필요한 기간과 열만 포함하도록 필터링하고, 데이터를 0과 1 사이의 범위로 투영하여 스케일링합니다.

1. 원본 데이터셋을 필터링하여 앞서 언급한 기간별 세트와 필요한 'load' 열 및 날짜만 포함합니다: [^2]

   ```python
   train = energy.copy()[(energy.index >= train_start_dt) & (energy.index < test_start_dt)][['load']]
   test = energy.copy()[energy.index >= test_start_dt][['load']]
   
   print('Training data shape: ', train.shape)
   print('Test data shape: ', test.shape)
   ```

   ```output
   Training data shape:  (1416, 1)
   Test data shape:  (48, 1)
   ```
   
2. 학습 데이터를 (0, 1) 범위로 스케일링합니다: [^2]

   ```python
   scaler = MinMaxScaler()
   train['load'] = scaler.fit_transform(train)
   ```
   
4. 이제 테스트 데이터를 스케일링합니다: [^2]

   ```python
   test['load'] = scaler.transform(test)
   ```

### 시계열 데이터 생성 [^1]

SVR을 위해 입력 데이터를 `[batch, timesteps]` 형태로 변환합니다. 따라서 기존 `train_data`와 `test_data`를 재구성하여 새로운 차원을 추가합니다. 이 차원은 시계열을 나타냅니다.

```python
# Converting to numpy arrays
train_data = train.values
test_data = test.values
```

이 예제에서는 `timesteps = 5`로 설정합니다. 따라서 모델의 입력은 첫 4개의 시계열 데이터이고, 출력은 5번째 시계열 데이터가 됩니다.

```python
timesteps=5
```

중첩 리스트 컴프리헨션을 사용하여 학습 데이터를 2D 텐서로 변환:

```python
train_data_timesteps=np.array([[j for j in train_data[i:i+timesteps]] for i in range(0,len(train_data)-timesteps+1)])[:,:,0]
train_data_timesteps.shape
```

```output
(1412, 5)
```

테스트 데이터를 2D 텐서로 변환:

```python
test_data_timesteps=np.array([[j for j in test_data[i:i+timesteps]] for i in range(0,len(test_data)-timesteps+1)])[:,:,0]
test_data_timesteps.shape
```

```output
(44, 5)
```

학습 및 테스트 데이터에서 입력과 출력을 선택:

```python
x_train, y_train = train_data_timesteps[:,:timesteps-1],train_data_timesteps[:,[timesteps-1]]
x_test, y_test = test_data_timesteps[:,:timesteps-1],test_data_timesteps[:,[timesteps-1]]

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)
```

```output
(1412, 4) (1412, 1)
(44, 4) (44, 1)
```

### SVR 구현 [^1]

이제 SVR을 구현할 시간입니다. 이 구현에 대해 더 읽고 싶다면 [이 문서](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html)를 참조하세요. 구현 단계는 다음과 같습니다:

1. `SVR()`를 호출하고 커널, 감마, C, epsilon과 같은 하이퍼파라미터를 전달하여 모델 정의
2. `fit()` 함수를 호출하여 학습 데이터를 준비
3. `predict()` 함수를 호출하여 예측 수행

이제 SVR 모델을 생성합니다. 여기서는 [RBF 커널](https://scikit-learn.org/stable/modules/svm.html#parameters-of-the-rbf-kernel)을 사용하며, 하이퍼파라미터 감마, C, epsilon을 각각 0.5, 10, 0.05로 설정합니다.

```python
model = SVR(kernel='rbf',gamma=0.5, C=10, epsilon = 0.05)
```

#### 학습 데이터에 모델 적합 [^1]

```python
model.fit(x_train, y_train[:,0])
```

```output
SVR(C=10, cache_size=200, coef0=0.0, degree=3, epsilon=0.05, gamma=0.5,
    kernel='rbf', max_iter=-1, shrinking=True, tol=0.001, verbose=False)
```

#### 모델 예측 수행 [^1]

```python
y_train_pred = model.predict(x_train).reshape(-1,1)
y_test_pred = model.predict(x_test).reshape(-1,1)

print(y_train_pred.shape, y_test_pred.shape)
```

```output
(1412, 1) (44, 1)
```

SVR을 구축했습니다! 이제 이를 평가해야 합니다.

### 모델 평가 [^1]

평가를 위해 먼저 데이터를 원래 스케일로 되돌립니다. 그런 다음 성능을 확인하기 위해 원본 및 예측 시계열 플롯을 그리고 MAPE 결과를 출력합니다.

예측 및 원본 출력 스케일 조정:

```python
# Scaling the predictions
y_train_pred = scaler.inverse_transform(y_train_pred)
y_test_pred = scaler.inverse_transform(y_test_pred)

print(len(y_train_pred), len(y_test_pred))
```

```python
# Scaling the original values
y_train = scaler.inverse_transform(y_train)
y_test = scaler.inverse_transform(y_test)

print(len(y_train), len(y_test))
```

#### 학습 및 테스트 데이터에서 모델 성능 확인 [^1]

데이터셋에서 타임스탬프를 추출하여 플롯의 x축에 표시합니다. 첫 ```timesteps-1``` 값을 첫 번째 출력의 입력으로 사용하므로 출력의 타임스탬프는 그 이후부터 시작됩니다.

```python
train_timestamps = energy[(energy.index < test_start_dt) & (energy.index >= train_start_dt)].index[timesteps-1:]
test_timestamps = energy[test_start_dt:].index[timesteps-1:]

print(len(train_timestamps), len(test_timestamps))
```

```output
1412 44
```

학습 데이터 예측 플롯:

```python
plt.figure(figsize=(25,6))
plt.plot(train_timestamps, y_train, color = 'red', linewidth=2.0, alpha = 0.6)
plt.plot(train_timestamps, y_train_pred, color = 'blue', linewidth=0.8)
plt.legend(['Actual','Predicted'])
plt.xlabel('Timestamp')
plt.title("Training data prediction")
plt.show()
```

![학습 데이터 예측](../../../../translated_images/train-data-predict.3c4ef4e78553104ffdd53d47a4c06414007947ea328e9261ddf48d3eafdefbbf.ko.png)

학습 데이터의 MAPE 출력:

```python
print('MAPE for training data: ', mape(y_train_pred, y_train)*100, '%')
```

```output
MAPE for training data: 1.7195710200875551 %
```

테스트 데이터 예측 플롯:

```python
plt.figure(figsize=(10,3))
plt.plot(test_timestamps, y_test, color = 'red', linewidth=2.0, alpha = 0.6)
plt.plot(test_timestamps, y_test_pred, color = 'blue', linewidth=0.8)
plt.legend(['Actual','Predicted'])
plt.xlabel('Timestamp')
plt.show()
```

![테스트 데이터 예측](../../../../translated_images/test-data-predict.8afc47ee7e52874f514ebdda4a798647e9ecf44a97cc927c535246fcf7a28aa9.ko.png)

테스트 데이터의 MAPE 출력:

```python
print('MAPE for testing data: ', mape(y_test_pred, y_test)*100, '%')
```

```output
MAPE for testing data:  1.2623790187854018 %
```

🏆 테스트 데이터셋에서 매우 좋은 결과를 얻었습니다!

### 전체 데이터셋에서 모델 성능 확인 [^1]

```python
# Extracting load values as numpy array
data = energy.copy().values

# Scaling
data = scaler.transform(data)

# Transforming to 2D tensor as per model input requirement
data_timesteps=np.array([[j for j in data[i:i+timesteps]] for i in range(0,len(data)-timesteps+1)])[:,:,0]
print("Tensor shape: ", data_timesteps.shape)

# Selecting inputs and outputs from data
X, Y = data_timesteps[:,:timesteps-1],data_timesteps[:,[timesteps-1]]
print("X shape: ", X.shape,"\nY shape: ", Y.shape)
```

```output
Tensor shape:  (26300, 5)
X shape:  (26300, 4) 
Y shape:  (26300, 1)
```

```python
# Make model predictions
Y_pred = model.predict(X).reshape(-1,1)

# Inverse scale and reshape
Y_pred = scaler.inverse_transform(Y_pred)
Y = scaler.inverse_transform(Y)
```

```python
plt.figure(figsize=(30,8))
plt.plot(Y, color = 'red', linewidth=2.0, alpha = 0.6)
plt.plot(Y_pred, color = 'blue', linewidth=0.8)
plt.legend(['Actual','Predicted'])
plt.xlabel('Timestamp')
plt.show()
```

![전체 데이터 예측](../../../../translated_images/full-data-predict.4f0fed16a131c8f3bcc57a3060039dc7f2f714a05b07b68c513e0fe7fb3d8964.ko.png)

```python
print('MAPE: ', mape(Y_pred, Y)*100, '%')
```

```output
MAPE:  2.0572089029888656 %
```

🏆 매우 훌륭한 플롯으로, 높은 정확도를 가진 모델을 보여줍니다. 잘하셨습니다!

---

## 🚀도전 과제

- 모델을 생성할 때 하이퍼파라미터(감마, C, epsilon)를 조정하고 데이터를 평가하여 테스트 데이터에서 가장 좋은 결과를 제공하는 하이퍼파라미터 세트를 찾아보세요. 이러한 하이퍼파라미터에 대해 더 알고 싶다면 [여기](https://scikit-learn.org/stable/modules/svm.html#parameters-of-the-rbf-kernel)를 참조하세요.
- 모델에 대해 다른 커널 함수를 사용하고 데이터셋에서 성능을 분석해보세요. 도움이 되는 문서는 [여기](https://scikit-learn.org/stable/modules/svm.html#kernel-functions)에서 찾을 수 있습니다.
- 모델이 예측을 위해 되돌아볼 수 있도록 `timesteps`의 다른 값을 사용해보세요.

## [강의 후 퀴즈](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/52/)

## 복습 및 자기 학습

이번 강의는 시계열 예측을 위한 SVR의 적용을 소개하기 위한 것이었습니다. SVR에 대해 더 읽고 싶다면 [이 블로그](https://www.analyticsvidhya.com/blog/2020/03/support-vector-regression-tutorial-for-machine-learning/)를 참조하세요. 이 [scikit-learn 문서](https://scikit-learn.org/stable/modules/svm.html)는 일반적인 SVM, [SVR](https://scikit-learn.org/stable/modules/svm.html#regression) 및 다른 구현 세부사항(예: 사용할 수 있는 다양한 [커널 함수](https://scikit-learn.org/stable/modules/svm.html#kernel-functions)와 그 매개변수)에 대한 더 포괄적인 설명을 제공합니다.

## 과제

[새로운 SVR 모델](assignment.md)

## 출처

[^1]: 이 섹션의 텍스트, 코드 및 출력은 [@AnirbanMukherjeeXD](https://github.com/AnirbanMukherjeeXD)가 기여했습니다.
[^2]: 이 섹션의 텍스트, 코드 및 출력은 [ARIMA](https://github.com/microsoft/ML-For-Beginners/tree/main/7-TimeSeries/2-ARIMA)에서 가져왔습니다.

---

**면책 조항**:  
이 문서는 AI 번역 서비스 [Co-op Translator](https://github.com/Azure/co-op-translator)를 사용하여 번역되었습니다. 정확성을 위해 최선을 다하고 있으나, 자동 번역에는 오류나 부정확성이 포함될 수 있습니다. 원본 문서를 해당 언어로 작성된 상태에서 권위 있는 자료로 간주해야 합니다. 중요한 정보의 경우, 전문적인 인간 번역을 권장합니다. 이 번역 사용으로 인해 발생하는 오해나 잘못된 해석에 대해 당사는 책임을 지지 않습니다.  