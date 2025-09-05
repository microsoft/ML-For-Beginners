<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "fa81d226c71d5af7a2cade31c1c92b88",
  "translation_date": "2025-09-05T10:41:04+00:00",
  "source_file": "2-Regression/1-Tools/README.md",
  "language_code": "ko"
}
-->
# Python과 Scikit-learn으로 회귀 모델 시작하기

![스케치노트로 요약된 회귀](../../../../sketchnotes/ml-regression.png)

> 스케치노트: [Tomomi Imura](https://www.twitter.com/girlie_mac)

## [강의 전 퀴즈](https://ff-quizzes.netlify.app/en/ml/)

> ### [이 강의는 R에서도 제공됩니다!](../../../../2-Regression/1-Tools/solution/R/lesson_1.html)

## 소개

이 네 가지 강의에서는 회귀 모델을 구축하는 방법을 배웁니다. 회귀 모델이 무엇을 위한 것인지 곧 논의할 것입니다. 하지만 그 전에, 과정을 시작하기 위해 적절한 도구가 준비되어 있는지 확인하세요!

이 강의에서 배우게 될 내용:

- 로컬 머신 러닝 작업을 위한 컴퓨터 설정
- Jupyter 노트북 사용법
- Scikit-learn 설치 및 사용법
- 실습을 통해 선형 회귀 탐구

## 설치 및 설정

[![초보자를 위한 머신 러닝 - 머신 러닝 모델 구축을 위한 도구 설정](https://img.youtube.com/vi/-DfeD2k2Kj0/0.jpg)](https://youtu.be/-DfeD2k2Kj0 "초보자를 위한 머신 러닝 - 머신 러닝 모델 구축을 위한 도구 설정")

> 🎥 위 이미지를 클릭하면 머신 러닝을 위한 컴퓨터 설정 과정을 다룬 짧은 영상을 볼 수 있습니다.

1. **Python 설치**. [Python](https://www.python.org/downloads/)이 컴퓨터에 설치되어 있는지 확인하세요. Python은 데이터 과학 및 머신 러닝 작업에 많이 사용됩니다. 대부분의 컴퓨터 시스템에는 이미 Python이 설치되어 있습니다. 일부 사용자에게는 설정을 쉽게 해주는 [Python Coding Packs](https://code.visualstudio.com/learn/educators/installers?WT.mc_id=academic-77952-leestott)도 유용할 수 있습니다.

   하지만 Python의 일부 사용 사례는 특정 버전을 요구할 수 있습니다. 따라서 [가상 환경](https://docs.python.org/3/library/venv.html)에서 작업하는 것이 유용합니다.

2. **Visual Studio Code 설치**. 컴퓨터에 Visual Studio Code가 설치되어 있는지 확인하세요. [Visual Studio Code 설치](https://code.visualstudio.com/)에 대한 지침을 따라 기본 설치를 완료하세요. 이 강의에서는 Visual Studio Code에서 Python을 사용할 예정이므로, [Visual Studio Code를 Python 개발에 맞게 설정](https://docs.microsoft.com/learn/modules/python-install-vscode?WT.mc_id=academic-77952-leestott)하는 방법을 익혀두는 것이 좋습니다.

   > Python에 익숙해지려면 이 [Learn 모듈 모음](https://docs.microsoft.com/users/jenlooper-2911/collections/mp1pagggd5qrq7?WT.mc_id=academic-77952-leestott)을 살펴보세요.
   >
   > [![Visual Studio Code로 Python 설정](https://img.youtube.com/vi/yyQM70vi7V8/0.jpg)](https://youtu.be/yyQM70vi7V8 "Visual Studio Code로 Python 설정")
   >
   > 🎥 위 이미지를 클릭하면 VS Code에서 Python을 사용하는 방법에 대한 영상을 볼 수 있습니다.

3. **Scikit-learn 설치**. [이 지침](https://scikit-learn.org/stable/install.html)을 따라 Scikit-learn을 설치하세요. Python 3을 사용해야 하므로 가상 환경을 사용하는 것이 권장됩니다. M1 Mac에 이 라이브러리를 설치하는 경우, 위 링크에 특별 지침이 나와 있습니다.

4. **Jupyter Notebook 설치**. [Jupyter 패키지 설치](https://pypi.org/project/jupyter/)가 필요합니다.

## 머신 러닝 작성 환경

Python 코드 개발 및 머신 러닝 모델 생성을 위해 **노트북**을 사용할 것입니다. 이 파일 형식은 데이터 과학자들이 자주 사용하는 도구로, `.ipynb` 확장자로 식별됩니다.

노트북은 개발자가 코드를 작성하고, 코드에 대한 주석을 추가하며, 문서를 작성할 수 있는 대화형 환경을 제공합니다. 이는 실험적이거나 연구 지향적인 프로젝트에 매우 유용합니다.

[![초보자를 위한 머신 러닝 - 회귀 모델 구축을 위한 Jupyter 노트북 설정](https://img.youtube.com/vi/7E-jC8FLA2E/0.jpg)](https://youtu.be/7E-jC8FLA2E "초보자를 위한 머신 러닝 - 회귀 모델 구축을 위한 Jupyter 노트북 설정")

> 🎥 위 이미지를 클릭하면 이 연습 과정을 다룬 짧은 영상을 볼 수 있습니다.

### 연습 - 노트북 작업하기

이 폴더에서 _notebook.ipynb_ 파일을 찾을 수 있습니다.

1. Visual Studio Code에서 _notebook.ipynb_를 엽니다.

   Jupyter 서버가 Python 3+와 함께 시작됩니다. 노트북의 특정 영역에서 `실행(run)`할 수 있는 코드 블록을 찾을 수 있습니다. 코드 블록은 재생 버튼 모양의 아이콘을 선택하여 실행할 수 있습니다.

2. `md` 아이콘을 선택하고, 다음 텍스트를 추가하여 약간의 마크다운을 작성합니다: **# Welcome to your notebook**.

   다음으로, Python 코드를 추가합니다.

3. 코드 블록에 **print('hello notebook')**을 입력합니다.
4. 화살표를 선택하여 코드를 실행합니다.

   출력 결과로 다음과 같은 문구가 표시됩니다:

    ```output
    hello notebook
    ```

![노트북이 열려 있는 VS Code](../../../../2-Regression/1-Tools/images/notebook.jpg)

코드와 함께 주석을 추가하여 노트북을 스스로 문서화할 수 있습니다.

✅ 웹 개발자의 작업 환경과 데이터 과학자의 작업 환경이 얼마나 다른지 잠시 생각해 보세요.

## Scikit-learn 시작하기

이제 로컬 환경에 Python이 설정되었고, Jupyter 노트북에 익숙해졌으니, Scikit-learn에도 익숙해져 봅시다. Scikit-learn은 머신 러닝 작업을 수행하는 데 도움을 주는 [광범위한 API](https://scikit-learn.org/stable/modules/classes.html#api-ref)를 제공합니다.

Scikit-learn의 [웹사이트](https://scikit-learn.org/stable/getting_started.html)에 따르면, "Scikit-learn은 지도 학습 및 비지도 학습을 지원하는 오픈 소스 머신 러닝 라이브러리입니다. 또한 모델 피팅, 데이터 전처리, 모델 선택 및 평가, 기타 다양한 유틸리티를 제공합니다."

이 강의에서는 Scikit-learn 및 기타 도구를 사용하여 '전통적인 머신 러닝' 작업을 수행하는 머신 러닝 모델을 구축할 것입니다. 신경망과 딥러닝은 다루지 않으며, 이는 곧 제공될 'AI for Beginners' 커리큘럼에서 다룰 예정입니다.

Scikit-learn은 모델을 구축하고 평가하는 과정을 간단하게 만들어 줍니다. 주로 숫자 데이터를 사용하는 데 초점이 맞춰져 있으며, 학습 도구로 사용할 수 있는 여러 가지 사전 제작된 데이터셋을 포함하고 있습니다. 또한 학생들이 시도해볼 수 있는 사전 제작된 모델도 포함되어 있습니다. 이제 사전 패키지 데이터와 기본 데이터를 사용하여 첫 번째 머신 러닝 모델을 구축하는 과정을 살펴봅시다.

## 연습 - 첫 번째 Scikit-learn 노트북

> 이 튜토리얼은 Scikit-learn 웹사이트의 [선형 회귀 예제](https://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html#sphx-glr-auto-examples-linear-model-plot-ols-py)에서 영감을 받았습니다.

[![초보자를 위한 머신 러닝 - Python에서 첫 번째 선형 회귀 프로젝트](https://img.youtube.com/vi/2xkXL5EUpS0/0.jpg)](https://youtu.be/2xkXL5EUpS0 "초보자를 위한 머신 러닝 - Python에서 첫 번째 선형 회귀 프로젝트")

> 🎥 위 이미지를 클릭하면 이 연습 과정을 다룬 짧은 영상을 볼 수 있습니다.

이 강의와 관련된 _notebook.ipynb_ 파일에서 모든 셀을 '휴지통' 아이콘을 눌러 비웁니다.

이 섹션에서는 학습 목적으로 Scikit-learn에 내장된 당뇨병 관련 소규모 데이터셋을 사용합니다. 당뇨병 환자에 대한 치료를 테스트하려 한다고 가정해 봅시다. 머신 러닝 모델은 변수 조합을 기반으로 어떤 환자가 치료에 더 잘 반응할지 결정하는 데 도움을 줄 수 있습니다. 시각화된 매우 기본적인 회귀 모델조차도 이론적 임상 시험을 조직하는 데 도움이 되는 변수에 대한 정보를 보여줄 수 있습니다.

✅ 회귀 방법에는 여러 가지가 있으며, 어떤 방법을 선택할지는 찾고자 하는 답에 따라 다릅니다. 예를 들어, 특정 나이에 대한 예상 키를 예측하려면 선형 회귀를 사용합니다. 이는 **숫자 값**을 찾고 있기 때문입니다. 반면, 특정 요리가 비건인지 아닌지를 알아내고 싶다면 **범주 할당**을 찾고 있는 것이므로 로지스틱 회귀를 사용합니다. 이후에 로지스틱 회귀에 대해 더 배우게 될 것입니다. 데이터를 통해 어떤 질문을 할 수 있을지, 그리고 어떤 방법이 더 적합할지 잠시 생각해 보세요.

이제 시작해 봅시다.

### 라이브러리 가져오기

이 작업을 위해 몇 가지 라이브러리를 가져옵니다:

- **matplotlib**. [그래프 도구](https://matplotlib.org/)로 유용하며, 선 그래프를 생성하는 데 사용할 것입니다.
- **numpy**. [numpy](https://numpy.org/doc/stable/user/whatisnumpy.html)는 Python에서 숫자 데이터를 처리하는 데 유용한 라이브러리입니다.
- **sklearn**. 이는 [Scikit-learn](https://scikit-learn.org/stable/user_guide.html) 라이브러리입니다.

작업에 필요한 라이브러리를 가져옵니다.

1. 다음 코드를 입력하여 라이브러리를 가져옵니다:

   ```python
   import matplotlib.pyplot as plt
   import numpy as np
   from sklearn import datasets, linear_model, model_selection
   ```

   위 코드에서는 `matplotlib`, `numpy`를 가져오고, `sklearn`에서 `datasets`, `linear_model`, `model_selection`을 가져옵니다. `model_selection`은 데이터를 학습 및 테스트 세트로 나누는 데 사용됩니다.

### 당뇨병 데이터셋

내장된 [당뇨병 데이터셋](https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset)에는 당뇨병과 관련된 442개의 샘플 데이터와 10개의 특징 변수가 포함되어 있습니다. 일부 변수는 다음과 같습니다:

- age: 나이 (연도 단위)
- bmi: 체질량지수
- bp: 평균 혈압
- s1 tc: T-세포 (백혈구의 한 종류)

✅ 이 데이터셋에는 당뇨병 연구에서 중요한 특징 변수로 '성별' 개념이 포함되어 있습니다. 많은 의료 데이터셋에는 이러한 이진 분류가 포함되어 있습니다. 이러한 분류가 인구의 특정 부분을 치료에서 배제할 가능성에 대해 잠시 생각해 보세요.

이제 X와 y 데이터를 로드합니다.

> 🎓 이는 지도 학습(supervised learning)이며, 명명된 'y' 타겟이 필요합니다.

새로운 코드 셀에서 `load_diabetes()`를 호출하여 당뇨병 데이터셋을 로드합니다. 입력값 `return_X_y=True`는 `X`가 데이터 행렬이 되고, `y`가 회귀 타겟이 됨을 나타냅니다.

1. 데이터 행렬의 모양과 첫 번째 요소를 표시하는 몇 가지 print 명령을 추가합니다:

    ```python
    X, y = datasets.load_diabetes(return_X_y=True)
    print(X.shape)
    print(X[0])
    ```

    반환값은 튜플입니다. 튜플의 첫 번째 두 값을 각각 `X`와 `y`에 할당합니다. [튜플에 대해 더 알아보기](https://wikipedia.org/wiki/Tuple).

    이 데이터는 442개의 항목으로 구성되어 있으며, 각 항목은 10개의 요소로 이루어진 배열로 되어 있음을 확인할 수 있습니다:

    ```text
    (442, 10)
    [ 0.03807591  0.05068012  0.06169621  0.02187235 -0.0442235  -0.03482076
    -0.04340085 -0.00259226  0.01990842 -0.01764613]
    ```

    ✅ 데이터와 회귀 타겟 간의 관계에 대해 잠시 생각해 보세요. 선형 회귀는 특징 X와 타겟 변수 y 간의 관계를 예측합니다. 당뇨병 데이터셋의 [타겟](https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset)을 문서에서 찾을 수 있나요? 이 데이터셋은 무엇을 보여주고 있나요?

2. 다음으로, 데이터셋의 3번째 열을 선택하여 플롯합니다. `:` 연산자를 사용하여 모든 행을 선택한 다음, 인덱스(2)를 사용하여 3번째 열을 선택할 수 있습니다. 또한 `reshape(n_rows, n_columns)`를 사용하여 데이터를 2D 배열로 재구성할 수 있습니다. 매개변수 중 하나가 -1이면 해당 차원이 자동으로 계산됩니다.

   ```python
   X = X[:, 2]
   X = X.reshape((-1,1))
   ```

   ✅ 데이터의 모양을 확인하려면 언제든지 데이터를 출력하세요.

3. 이제 데이터를 플롯할 준비가 되었으니, 머신이 이 데이터셋의 숫자 간 논리적 분리를 결정할 수 있는지 확인할 수 있습니다. 이를 위해 데이터(X)와 타겟(y)을 테스트 및 학습 세트로 나눌 필요가 있습니다. Scikit-learn은 이를 간단히 수행할 수 있는 방법을 제공합니다. 특정 지점에서 테스트 데이터를 나눌 수 있습니다.

   ```python
   X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.33)
   ```

4. 이제 모델을 학습시킬 준비가 되었습니다! 선형 회귀 모델을 로드하고, `model.fit()`을 사용하여 X 및 y 학습 세트로 학습시킵니다:

    ```python
    model = linear_model.LinearRegression()
    model.fit(X_train, y_train)
    ```

    ✅ `model.fit()`은 TensorFlow와 같은 많은 머신 러닝 라이브러리에서 볼 수 있는 함수입니다.

5. 그런 다음, `predict()` 함수를 사용하여 테스트 데이터를 기반으로 예측을 생성합니다. 이는 데이터 그룹 간의 선을 그리는 데 사용됩니다.

    ```python
    y_pred = model.predict(X_test)
    ```

6. 이제 데이터를 플롯에 표시할 시간입니다. Matplotlib은 이 작업에 매우 유용한 도구입니다. 모든 X 및 y 테스트 데이터의 산점도를 생성하고, 예측을 사용하여 데이터 그룹 간 가장 적절한 위치에 선을 그립니다.

    ```python
    plt.scatter(X_test, y_test,  color='black')
    plt.plot(X_test, y_pred, color='blue', linewidth=3)
    plt.xlabel('Scaled BMIs')
    plt.ylabel('Disease Progression')
    plt.title('A Graph Plot Showing Diabetes Progression Against BMI')
    plt.show()
    ```

   ![당뇨병 데이터를 나타내는 산점도](../../../../2-Regression/1-Tools/images/scatterplot.png)
✅ 여기서 무슨 일이 일어나고 있는지 잠시 생각해 보세요. 많은 작은 데이터 점들을 관통하는 직선이 있습니다. 하지만 이 직선이 정확히 무엇을 하고 있는 걸까요? 이 직선을 사용하여 새로운, 보지 못한 데이터 포인트가 플롯의 y축과 어떤 관계를 맺어야 하는지 예측할 수 있다는 것을 이해할 수 있나요? 이 모델의 실질적인 사용 방법을 말로 표현해 보세요.

축하합니다! 첫 번째 선형 회귀 모델을 구축하고, 이를 사용해 예측을 생성한 뒤 플롯에 표시했습니다!

---
## 🚀도전 과제

이 데이터셋에서 다른 변수를 플롯해 보세요. 힌트: 이 줄을 수정하세요: `X = X[:,2]`. 이 데이터셋의 목표를 고려했을 때, 당뇨병이 질병으로서 어떻게 진행되는지에 대해 무엇을 발견할 수 있나요?

## [강의 후 퀴즈](https://ff-quizzes.netlify.app/en/ml/)

## 복습 및 자기 학습

이 튜토리얼에서는 단변량 또는 다변량 선형 회귀가 아닌 간단한 선형 회귀를 사용했습니다. 이러한 방법들 간의 차이에 대해 조금 읽어보거나 [이 비디오](https://www.coursera.org/lecture/quantifying-relationships-regression-models/linear-vs-nonlinear-categorical-variables-ai2Ef)를 확인해 보세요.

회귀의 개념에 대해 더 읽어보고 이 기술로 어떤 종류의 질문에 답할 수 있는지 생각해 보세요. 이 [튜토리얼](https://docs.microsoft.com/learn/modules/train-evaluate-regression-models?WT.mc_id=academic-77952-leestott)을 통해 이해를 심화해 보세요.

## 과제

[다른 데이터셋](assignment.md)

---

**면책 조항**:  
이 문서는 AI 번역 서비스 [Co-op Translator](https://github.com/Azure/co-op-translator)를 사용하여 번역되었습니다. 정확성을 위해 최선을 다하고 있으나, 자동 번역에는 오류나 부정확성이 포함될 수 있습니다. 원본 문서를 해당 언어로 작성된 상태에서 권위 있는 자료로 간주해야 합니다. 중요한 정보의 경우, 전문적인 인간 번역을 권장합니다. 이 번역 사용으로 인해 발생하는 오해나 잘못된 해석에 대해 당사는 책임을 지지 않습니다.  