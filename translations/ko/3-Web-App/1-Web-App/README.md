# 웹 앱을 만들어 ML 모델 사용하기

이번 강의에서는 _지난 한 세기 동안의 UFO 목격_ 데이터 세트를 사용하여 ML 모델을 훈련할 것입니다. 이 데이터는 NUFORC의 데이터베이스에서 가져왔습니다.

배울 내용:

- 훈련된 모델을 '피클'하는 방법
- Flask 앱에서 그 모델을 사용하는 방법

노트북을 사용하여 데이터를 정리하고 모델을 훈련하는 방법을 계속 배우겠지만, 이를 한 단계 더 나아가 웹 앱에서 모델을 사용하는 방법도 탐구할 것입니다.

이를 위해 Flask를 사용하여 웹 앱을 구축해야 합니다.

## [강의 전 퀴즈](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/17/)

## 앱 구축하기

머신 러닝 모델을 사용하는 웹 앱을 구축하는 방법에는 여러 가지가 있습니다. 웹 아키텍처는 모델 훈련 방식에 영향을 미칠 수 있습니다. 데이터 과학 그룹이 훈련한 모델을 앱에서 사용해야 하는 상황을 상상해 보세요.

### 고려 사항

다음과 같은 여러 질문을 해야 합니다:

- **웹 앱인가 모바일 앱인가?** 모바일 앱을 구축하거나 IoT 환경에서 모델을 사용해야 하는 경우, [TensorFlow Lite](https://www.tensorflow.org/lite/)를 사용하여 Android 또는 iOS 앱에서 모델을 사용할 수 있습니다.
- **모델은 어디에 위치할 것인가?** 클라우드에 있을 것인가, 로컬에 있을 것인가?
- **오프라인 지원.** 앱이 오프라인에서도 작동해야 하는가?
- **모델을 훈련하는 데 사용된 기술은 무엇인가?** 선택한 기술이 사용해야 할 도구에 영향을 미칠 수 있습니다.
    - **TensorFlow 사용.** 예를 들어 TensorFlow를 사용하여 모델을 훈련하는 경우, [TensorFlow.js](https://www.tensorflow.org/js/)를 사용하여 웹 앱에서 사용할 수 있도록 TensorFlow 모델을 변환할 수 있습니다.
    - **PyTorch 사용.** [PyTorch](https://pytorch.org/)와 같은 라이브러리를 사용하여 모델을 구축하는 경우, [ONNX](https://onnx.ai/) (Open Neural Network Exchange) 형식으로 내보내 JavaScript 웹 앱에서 사용할 수 있는 [Onnx Runtime](https://www.onnxruntime.ai/)을 사용할 수 있습니다. 이 옵션은 Scikit-learn으로 훈련된 모델에 대해 추후 강의에서 탐구할 것입니다.
    - **Lobe.ai 또는 Azure Custom Vision 사용.** [Lobe.ai](https://lobe.ai/) 또는 [Azure Custom Vision](https://azure.microsoft.com/services/cognitive-services/custom-vision-service/?WT.mc_id=academic-77952-leestott)과 같은 ML SaaS(Software as a Service) 시스템을 사용하여 모델을 훈련하는 경우, 이 소프트웨어는 모델을 다양한 플랫폼에 내보내는 방법을 제공합니다. 여기에는 온라인 애플리케이션에서 클라우드에서 쿼리할 수 있는 맞춤형 API를 구축하는 것도 포함됩니다.

또한 웹 브라우저에서 직접 모델을 훈련할 수 있는 전체 Flask 웹 앱을 구축할 수도 있습니다. 이는 JavaScript 환경에서 TensorFlow.js를 사용하여 수행할 수 있습니다.

우리의 목적을 위해, Python 기반 노트북을 사용하고 있으므로, 훈련된 모델을 이러한 노트북에서 Python으로 구축된 웹 앱에서 읽을 수 있는 형식으로 내보내는 단계를 탐구해 보겠습니다.

## 도구

이 작업을 위해 Flask와 Pickle 두 가지 도구가 필요합니다. 둘 다 Python에서 실행됩니다.

✅ [Flask](https://palletsprojects.com/p/flask/)란? Flask는 그 창시자들에 의해 '마이크로 프레임워크'로 정의되며, Python을 사용하여 웹 페이지를 구축하는 템플릿 엔진을 포함한 웹 프레임워크의 기본 기능을 제공합니다. Flask로 구축하는 연습을 위해 [이 학습 모듈](https://docs.microsoft.com/learn/modules/python-flask-build-ai-web-app?WT.mc_id=academic-77952-leestott)을 살펴보세요.

✅ [Pickle](https://docs.python.org/3/library/pickle.html)이란? Pickle 🥒은 Python 객체 구조를 직렬화하고 역직렬화하는 Python 모듈입니다. 모델을 '피클'할 때, 구조를 웹에서 사용할 수 있도록 직렬화하거나 평탄화합니다. 주의하세요: 피클은 본질적으로 안전하지 않으므로 파일을 '언피클'하라는 메시지가 표시되면 주의해야 합니다. 피클된 파일의 접미사는 `.pkl`입니다.

## 연습 - 데이터 정리하기

이번 강의에서는 [NUFORC](https://nuforc.org) (National UFO Reporting Center)에서 수집한 80,000건의 UFO 목격 데이터를 사용합니다. 이 데이터에는 흥미로운 UFO 목격 설명이 포함되어 있습니다. 예를 들어:

- **긴 설명 예시.** "한 남자가 밤에 풀밭에 빛나는 빔에서 나와 텍사스 인스트루먼트 주차장으로 달려갑니다".
- **짧은 설명 예시.** "불빛이 우리를 쫓아왔습니다".

[ufos.csv](../../../../3-Web-App/1-Web-App/data/ufos.csv) 스프레드시트에는 목격이 발생한 `city`, `state`, `country`에 대한 열과 객체의 `shape`, `latitude`, `longitude`가 포함되어 있습니다.

이번 강의에 포함된 빈 [노트북](../../../../3-Web-App/1-Web-App/notebook.ipynb)에서:

1. 이전 강의에서 했던 것처럼 `pandas`, `matplotlib`, `numpy`을 가져오고 ufos 스프레드시트를 가져옵니다. 샘플 데이터 세트를 확인할 수 있습니다:

    ```python
    import pandas as pd
    import numpy as np
    
    ufos = pd.read_csv('./data/ufos.csv')
    ufos.head()
    ```

1. ufos 데이터를 새 제목으로 작은 데이터프레임으로 변환합니다. `Country` 필드의 고유 값을 확인합니다.

    ```python
    ufos = pd.DataFrame({'Seconds': ufos['duration (seconds)'], 'Country': ufos['country'],'Latitude': ufos['latitude'],'Longitude': ufos['longitude']})
    
    ufos.Country.unique()
    ```

1. 이제 필요한 데이터 양을 줄이기 위해 null 값을 삭제하고 1-60초 사이의 목격만 가져옵니다:

    ```python
    ufos.dropna(inplace=True)
    
    ufos = ufos[(ufos['Seconds'] >= 1) & (ufos['Seconds'] <= 60)]
    
    ufos.info()
    ```

1. Scikit-learn의 `LabelEncoder` 라이브러리를 가져와 국가의 텍스트 값을 숫자로 변환합니다:

    ✅ LabelEncoder는 데이터를 알파벳 순서로 인코딩합니다

    ```python
    from sklearn.preprocessing import LabelEncoder
    
    ufos['Country'] = LabelEncoder().fit_transform(ufos['Country'])
    
    ufos.head()
    ```

    데이터는 다음과 같아야 합니다:

    ```output
    	Seconds	Country	Latitude	Longitude
    2	20.0	3		53.200000	-2.916667
    3	20.0	4		28.978333	-96.645833
    14	30.0	4		35.823889	-80.253611
    23	60.0	4		45.582778	-122.352222
    24	3.0		3		51.783333	-0.783333
    ```

## 연습 - 모델 구축하기

이제 데이터를 훈련 및 테스트 그룹으로 나누어 모델을 훈련할 준비를 할 수 있습니다.

1. 훈련할 세 가지 특징을 X 벡터로 선택하고, y 벡터는 `Country`. You want to be able to input `Seconds`, `Latitude` and `Longitude`를 선택하여 국가 ID를 반환합니다.

    ```python
    from sklearn.model_selection import train_test_split
    
    Selected_features = ['Seconds','Latitude','Longitude']
    
    X = ufos[Selected_features]
    y = ufos['Country']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    ```

1. 로지스틱 회귀를 사용하여 모델을 훈련합니다:

    ```python
    from sklearn.metrics import accuracy_score, classification_report
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    
    print(classification_report(y_test, predictions))
    print('Predicted labels: ', predictions)
    print('Accuracy: ', accuracy_score(y_test, predictions))
    ```

정확도는 **약 95%**로 나쁘지 않습니다. 이는 `Country` and `Latitude/Longitude` correlate.

The model you created isn't very revolutionary as you should be able to infer a `Country` from its `Latitude` and `Longitude` 때문입니다. 하지만, 정리한 원시 데이터에서 훈련하고, 내보낸 후 이 모델을 웹 앱에서 사용하는 연습을 하는 것은 좋은 경험입니다.

## 연습 - 모델 '피클'하기

이제 모델을 _피클_할 시간입니다! 몇 줄의 코드로 이를 수행할 수 있습니다. _피클_된 후에는 피클된 모델을 로드하고 초, 위도, 경도 값을 포함하는 샘플 데이터 배열에 대해 테스트합니다.

```python
import pickle
model_filename = 'ufo-model.pkl'
pickle.dump(model, open(model_filename,'wb'))

model = pickle.load(open('ufo-model.pkl','rb'))
print(model.predict([[50,44,-12]]))
```

모델은 **'3'**을 반환합니다. 이는 영국의 국가 코드입니다. 놀랍군요! 👽

## 연습 - Flask 앱 구축하기

이제 모델을 호출하고 유사한 결과를 반환하는 Flask 앱을 구축할 수 있습니다. 더 시각적으로 보기 좋게 만들 것입니다.

1. _notebook.ipynb_ 파일이 있는 위치에 **web-app** 폴더를 만듭니다. 여기에는 _ufo-model.pkl_ 파일이 있습니다.

1. 그 폴더에 세 개의 폴더를 더 만듭니다: **static** 폴더와 그 안에 **css** 폴더, 그리고 **templates** 폴더. 이제 다음과 같은 파일과 디렉토리가 있어야 합니다:

    ```output
    web-app/
      static/
        css/
      templates/
    notebook.ipynb
    ufo-model.pkl
    ```

    ✅ 완성된 앱의 모습을 보려면 솔루션 폴더를 참조하세요

1. _web-app_ 폴더에서 첫 번째로 생성할 파일은 **requirements.txt** 파일입니다. JavaScript 앱의 _package.json_과 같이 이 파일은 앱에 필요한 종속성을 나열합니다. **requirements.txt**에 다음 줄을 추가합니다:

    ```text
    scikit-learn
    pandas
    numpy
    flask
    ```

1. 이제 _web-app_으로 이동하여 이 파일을 실행합니다:

    ```bash
    cd web-app
    ```

1. 터미널에서 `pip install`을 입력하여 _requirements.txt_에 나열된 라이브러리를 설치합니다:

    ```bash
    pip install -r requirements.txt
    ```

1. 이제 앱을 완성하기 위해 세 개의 파일을 더 생성할 준비가 되었습니다:

    1. 루트에 **app.py**를 생성합니다.
    2. _templates_ 디렉토리에 **index.html**을 생성합니다.
    3. _static/css_ 디렉토리에 **styles.css**를 생성합니다.

1. _styles.css_ 파일에 몇 가지 스타일을 추가합니다:

    ```css
    body {
    	width: 100%;
    	height: 100%;
    	font-family: 'Helvetica';
    	background: black;
    	color: #fff;
    	text-align: center;
    	letter-spacing: 1.4px;
    	font-size: 30px;
    }
    
    input {
    	min-width: 150px;
    }
    
    .grid {
    	width: 300px;
    	border: 1px solid #2d2d2d;
    	display: grid;
    	justify-content: center;
    	margin: 20px auto;
    }
    
    .box {
    	color: #fff;
    	background: #2d2d2d;
    	padding: 12px;
    	display: inline-block;
    }
    ```

1. 다음으로 _index.html_ 파일을 작성합니다:

    ```html
    <!DOCTYPE html>
    <html>
      <head>
        <meta charset="UTF-8">
        <title>🛸 UFO Appearance Prediction! 👽</title>
        <link rel="stylesheet" href="{{ url_for('static', filename='css/styles.css') }}">
      </head>
    
      <body>
        <div class="grid">
    
          <div class="box">
    
            <p>According to the number of seconds, latitude and longitude, which country is likely to have reported seeing a UFO?</p>
    
            <form action="{{ url_for('predict')}}" method="post">
              <input type="number" name="seconds" placeholder="Seconds" required="required" min="0" max="60" />
              <input type="text" name="latitude" placeholder="Latitude" required="required" />
              <input type="text" name="longitude" placeholder="Longitude" required="required" />
              <button type="submit" class="btn">Predict country where the UFO is seen</button>
            </form>
    
            <p>{{ prediction_text }}</p>
    
          </div>
    
        </div>
    
      </body>
    </html>
    ```

    이 파일의 템플릿 구문을 살펴보세요. 예측 텍스트와 같은 변수가 제공될 때 `{{}}`. There's also a form that posts a prediction to the `/predict` route.

    Finally, you're ready to build the python file that drives the consumption of the model and the display of predictions:

1. In `app.py`에 추가합니다:

    ```python
    import numpy as np
    from flask import Flask, request, render_template
    import pickle
    
    app = Flask(__name__)
    
    model = pickle.load(open("./ufo-model.pkl", "rb"))
    
    
    @app.route("/")
    def home():
        return render_template("index.html")
    
    
    @app.route("/predict", methods=["POST"])
    def predict():
    
        int_features = [int(x) for x in request.form.values()]
        final_features = [np.array(int_features)]
        prediction = model.predict(final_features)
    
        output = prediction[0]
    
        countries = ["Australia", "Canada", "Germany", "UK", "US"]
    
        return render_template(
            "index.html", prediction_text="Likely country: {}".format(countries[output])
        )
    
    
    if __name__ == "__main__":
        app.run(debug=True)
    ```

    > 💡 팁: [`debug=True`](https://www.askpython.com/python-modules/flask/flask-debug-mode) while running the web app using Flask, any changes you make to your application will be reflected immediately without the need to restart the server. Beware! Don't enable this mode in a production app.

If you run `python app.py` or `python3 app.py` - your web server starts up, locally, and you can fill out a short form to get an answer to your burning question about where UFOs have been sighted!

Before doing that, take a look at the parts of `app.py`:

1. First, dependencies are loaded and the app starts.
1. Then, the model is imported.
1. Then, index.html is rendered on the home route.

On the `/predict` route, several things happen when the form is posted:

1. The form variables are gathered and converted to a numpy array. They are then sent to the model and a prediction is returned.
2. The Countries that we want displayed are re-rendered as readable text from their predicted country code, and that value is sent back to index.html to be rendered in the template.

Using a model this way, with Flask and a pickled model, is relatively straightforward. The hardest thing is to understand what shape the data is that must be sent to the model to get a prediction. That all depends on how the model was trained. This one has three data points to be input in order to get a prediction.

In a professional setting, you can see how good communication is necessary between the folks who train the model and those who consume it in a web or mobile app. In our case, it's only one person, you!

---

## 🚀 Challenge

Instead of working in a notebook and importing the model to the Flask app, you could train the model right within the Flask app! Try converting your Python code in the notebook, perhaps after your data is cleaned, to train the model from within the app on a route called `train`. 이 방법을 추구할 때의 장단점은 무엇인가요?

## [강의 후 퀴즈](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/18/)

## 복습 및 자습

ML 모델을 사용하는 웹 앱을 구축하는 방법은 여러 가지가 있습니다. JavaScript 또는 Python을 사용하여 머신 러닝을 활용하는 웹 앱을 구축할 수 있는 방법을 목록으로 만들어 보세요. 아키텍처를 고려해 보세요: 모델이 앱에 있어야 하나요, 아니면 클라우드에 있어야 하나요? 후자의 경우, 어떻게 접근할 수 있나요? 적용된 ML 웹 솔루션의 아키텍처 모델을 그려보세요.

## 과제

[다른 모델 시도해보기](assignment.md)

**면책 조항**:
이 문서는 기계 기반 AI 번역 서비스를 사용하여 번역되었습니다. 정확성을 위해 노력하고 있지만, 자동 번역에는 오류나 부정확성이 있을 수 있습니다. 원어로 작성된 원본 문서를 권위 있는 자료로 간주해야 합니다. 중요한 정보의 경우, 전문적인 인간 번역을 권장합니다. 이 번역의 사용으로 인해 발생하는 오해나 잘못된 해석에 대해 당사는 책임을 지지 않습니다.