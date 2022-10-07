# ML 모델 사용하여 Web App 만들기

이 강의에서, 이 세상에 없었던 데이터셋에 대하여 ML 모델을 훈련할 예정입니다: _UFO sightings over the past century_, sourced from NUFORC's database.

다음을 배우게 됩니다:

- 훈련된 모델을 'pickle'하는 방식
- Flask 앱에서 모델을 사용하는 방식

계속 노트북으로 데이터를 정리하고 모델을 훈련하지만, 웹 앱에서 'in the wild' 모델을 사용하면 단계를 넘어서 발전할 수 있습니다. 

이러면, Flask로 웹 앱을 만들어야 합니다.

## [강의 전 퀴즈](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/17/)

## 앱 만들기

머신러닝 모델로 웹 앱을 만드는 여러 방식이 존재합니다. 웹 구조는 모델을 훈련하는 방식에 영향을 줄 수 있습니다. 데이터 사이언스 그룹이 앱에서 사용하고 싶은 훈련된 모델을 가지고 비지니스에서 일한다고 상상해봅니다.

### 고려할 사항

많은 질문들을 물어볼 필요가 있습니다:

- **웹 앱 혹은 모바일 앱인가요?** 만약 모바일 앱을 만들거나 IoT 컨텍스트에서 모델을 사용해야 되는 경우, [TensorFlow Lite](https://www.tensorflow.org/lite/)로 Android 또는 iOS 앱에서 모델을 사용할 수 있습니다.
- **모델은 어디에 있나요?** 클라우드 또는 로컬 중 어디인가요?
- **오프라인 지원합니다.** 앱이 오프라인으로 동작하나요?
- **모델을 훈련시킬 때 사용하는 기술은 무엇인가요?** 선택된 기술은 사용할 도구에 영향을 줄 수 있습니다.
   - **Tensor flow 사용합니다.** 만약 TensorFlow로 모델을 훈련한다면, 예시로, 에코 시스템은 [TensorFlow.js](https://www.tensorflow.org/js/)로 웹 앱에서 사용할 TensorFlow 모델을 변환해주는 기능을 제공합니다.
   - **PyTorch 사용합니다.** 만약 [PyTorch](https://pytorch.org/) 같은 라이브러리로 모델을 만들면, [Onnx Runtime](https://www.onnxruntime.ai/)으로 할 수 있는 JavaScript 웹 앱에서 사용하기 위한 [ONNX](https://onnx.ai/) (Open Neural Network Exchange) 포맷으로 내보낼 옵션이 존재합니다. 이 옵션은 Scikit-learn-trained 모델로 이후 강의에서 알아볼 예정입니다.
   - **Lobe.ai 또는 Azure Custom vision 사용합니다.** 만약 [Lobe.ai](https://lobe.ai/) 또는 [Azure Custom Vision](https://azure.microsoft.com/services/cognitive-services/custom-vision-service/?WT.mc_id=academic-77952-leestott) 같은 ML SaaS (Software as a Service) 시스템으로 모델을 훈련하게 된다면, 이 소프트웨어 타입은 온라인 애플리케이션이 클라우드에서 쿼리된 bespoke API를 만드는 것도 포함해서 많은 플랫폼의 모델들을 내보낼 방식을 제공합니다.

또 웹 브라우저에서 모델로만 훈련할 수 있는 모든 Flask 웹 앱을 만들 수 있습니다. JavaScript 컨텍스트에서 TensorFlow.js로 마무리 지을 수 있습니다.

목적을 위해서, Python-기반의 노트북으로 작성했기 때문에, 노트북에서 훈련된 모델을 Python-제작한 웹 앱에서 읽을 수 있는 포맷으로 내보낼 때 필요한 단계를 알아봅니다.

## 도구

작업에서, 2가지 도구가 필요합니다: Flask 와 Pickle은, 둘 다 Python에서 작동합니다.

✅ [Flask](https://palletsprojects.com/p/flask/)는 무엇일까요? 작성자가 'micro-framework'로 정의한, Flask는 Python으로 웹 프레임워크의 기본적인 기능과 웹 페이지를 만드는 템플릿 엔진을 제공합니다. [this Learn module](https://docs.microsoft.com/learn/modules/python-flask-build-ai-web-app?WT.mc_id=academic-77952-leestott)을 보고 Flask로 만드는 것을 연습합니다.

✅ [Pickle](https://docs.python.org/3/library/pickle.html)은 무엇일까요? Pickle 🥒은 Python 객체 구조를 serializes와 de-serializes하는 Python 모듈입니다. 모델을 'pickle'하게 되면, 웹에서 쓰기 위해서 serialize 또는 flatten합니다. 주의합시다: pickle은 원래 안전하지 않아서, 파일을 'un-pickle'한다고 나오면 조심합니다. pickled 파일은 접미사 `.pkl`로 있습니다. 

## 연습 - 데이터 정리하기

[NUFORC](https://nuforc.org) (The National UFO Reporting Center)에서 모아둔, 80,000 UFO 목격 데이터를 이 강의에서 사용합니다. 데이터에 UFO 목격 관련한 몇 흥미로운 설명이 있습니다, 예시로 들어봅니다:

- **긴 예시를 설명합니다.** "A man emerges from a beam of light that shines on a grassy field at night and he runs towards the Texas Instruments parking lot".
- **짧은 예시를 설명합니다.** "the lights chased us".

[ufos.csv](.././data/ufos.csv) 스프레드시트에는 목격된 `city`, `state` 와 `country`, 오브젝트의 `shape` 와 `latitude` 및 `longitude` 열이 포함되어 있습니다.

강의에 있는 빈 [notebook](../notebook.ipynb)에서 진행합니다:

1. 이전 강의에서 했던 것처럼 `pandas`, `matplotlib`, 와 `numpy`를 import하고 ufos 스프레드시트도 import합니다. 샘플 데이터셋을 볼 수 있습니다:

    ```python
    import pandas as pd
    import numpy as np
    
    ufos = pd.read_csv('./data/ufos.csv')
    ufos.head()
    ```

1. ufos 데이터를 새로운 제목의 작은 데이터프레임으로 변환합니다. `Country` 필드가 유니크 값인지 확인합니다.

    ```python
    ufos = pd.DataFrame({'Seconds': ufos['duration (seconds)'], 'Country': ufos['country'],'Latitude': ufos['latitude'],'Longitude': ufos['longitude']})
    
    ufos.Country.unique()
    ```

1. 지금부터, 모든 null 값을 드랍하고 1-60초 사이 목격만 가져와서 처리할 데이터의 수량을 줄일 수 있습니다:

    ```python
    ufos.dropna(inplace=True)
    
    ufos = ufos[(ufos['Seconds'] >= 1) & (ufos['Seconds'] <= 60)]
    
    ufos.info()
    ```

1. Scikit-learn의 `LabelEncoder` 라이브러리를 Import해서 국가의 텍스트 값을 숫자로 변환합니다:

   ✅ LabelEncoder는 데이터를 알파벳 순서로 인코드합니다.

    ```python
    from sklearn.preprocessing import LabelEncoder
    
    ufos['Country'] = LabelEncoder().fit_transform(ufos['Country'])
    
    ufos.head()
    ```

    데이터는 이렇게 보일 것입니다:

    ```output
    	Seconds	Country	Latitude	Longitude
    2	20.0	3	    53.200000	-2.916667
    3	20.0	4	    28.978333	-96.645833
    14	30.0	4	    35.823889	-80.253611
    23	60.0	4	    45.582778	-122.352222
    24	3.0	    3	    51.783333	-0.783333
    ```

## 연습 - 모델 만들기

지금부터 데이터를 훈련하고 테스트할 그룹으로 나누어서 모델을 훈련할 준비가 되었습니다.

1. X 백터로 훈련할 3가지 features를 선택하면, y 백터는 `Country`로 됩니다. `Seconds`, `Latitude` 와 `Longitude`를 입력하면 국가 id로 반환되기를 원합니다.

    ```python
    from sklearn.model_selection import train_test_split
    
    Selected_features = ['Seconds','Latitude','Longitude']
    
    X = ufos[Selected_features]
    y = ufos['Country']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    ```

1. logistic regression을 사용해서 모델을 훈련합니다:

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

당연하게, `Country` 와 `Latitude/Longitude`가 상관 관계있어서, 정확도 **(around 95%)** 가 나쁘지 않습니다.

만든 모델은 `Latitude` 와 `Longitude`에서 `Country`를 알 수 있어야 하므로 매우 혁신적이지 않지만, 정리하면서, 뽑은 원본 데이터에서 훈련을 해보고 웹 앱에서 모델을 쓰기에 좋은 연습입니다.

## 연습 - 모델 'pickle'하기

모델을 _pickle_ 할 시간이 되었습니다! 코드 몇 줄로 할 수 있습니다. _pickled_ 되면, pickled 모델을 불러와서 초, 위도와 경도 값이 포함된 샘플 데이터 배열을 대상으로 테스트합니다.

```python
import pickle
model_filename = 'ufo-model.pkl'
pickle.dump(model, open(model_filename,'wb'))

model = pickle.load(open('ufo-model.pkl','rb'))
print(model.predict([[50,44,-12]]))
```

모델은 영국 국가 코드인, **'3'** 이 반환됩니다. Wild! 👽

## 연습 - Flask 앱 만들기

지금부터 Flask 앱을 만들어서 모델을 부르고 비슷한 결과를 반환하지만, 시각적으로 만족할 방식으로도 가능합니다.

1. _ufo-model.pkl_ 파일과 _notebook.ipynb_ 파일 옆에 **web-app** 이라고 불리는 폴더를 만들면서 시작합니다.

1. 폴더에서 3가지 폴더를 만듭니다: **static**, 내부에 **css** 폴더가 있으며, **templates`** 도 있습니다. 지금부터 다음 파일과 디렉토리들이 있어야 합니다:

    ```output
    web-app/
      static/
        css/
        templates/
    notebook.ipynb
    ufo-model.pkl
    ``` 

   ✅ 완성된 앱을 보려면 solution 폴더를 참조합니다

1. _web-app_ 폴더에서 만들 첫 파일은 **requirements.txt** 파일입니다. JavaScript 앱의 _package.json_ 처럼, 앱에 필요한 의존성을 리스트한 파일입니다. **requirements.txt** 에 해당 라인을 추가합니다:

    ```text
    scikit-learn
    pandas
    numpy
    flask
    ```

1. 지금부터, _web-app_ 으로 이동해서 파일을 실행합니다:

   ```bash
   cd web-app
   ```

1. 터미널에서 `pip install`을 타이핑해서, _requirements.txt_ 에 나열된 라이브러리를 설치합니다:

   ```bash
   pip install -r requirements.txt
   ```

1. 지금부터, 앱을 완성하기 위해서 3가지 파일을 더 만들 준비를 했습니다:

    1. 최상단에  **app.py**를 만듭니다.
    2. _templates_ 디렉토리에 **index.html**을 만듭니다.
    3. _static/css_ 디렉토리에 **styles.css**를 만듭니다.

1. 몇 스타일로 _styles.css_ 파일을 만듭니다:

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

1. 다음으로 _index.html_ 파일을 만듭니다:

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

   파일의 템플릿을 봅니다. 예측 텍스트: `{{}}`처럼, 앱에서 제공할 수 있는 변수 주위, 'mustache' 구문을 확인해봅니다. `/predict` 라우터에 예측을 보낼 폼도 있습니다.

    마지막으로, 모델을 써서 예측으로 보여줄 python 파일을 만들 준비가 되었습니다:

1. `app.py` 에 추가합니다:

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

   > 💡 팁: [`debug=True`](https://www.askpython.com/python-modules/flask/flask-debug-mode)를 추가하면 Flask 사용해서 웹 앱을 실행하는 도중에, 서버를 다시 시작할 필요없이 애플리케이션에 변경점이 바로 반영됩니다. 조심하세요! 프로덕션 앱에서 이 모드를 활성화하지 맙시다. 

만약 `python app.py` 또는 `python3 app.py`를 실행하면 - 웹 서버가 로컬에서 시작하고, 짧은 폼을 작성하면 UFOs가 목격된 장소에 대해 주목받을 질문의 답을 얻을 수 있습니다!

하기 전, `app.py`의 일부분을 봅니다:

1. 먼저, 의존성을 불러오고 앱이 시작합니다.
1. 그 다음, 모델을 가져옵니다.
1. 그 다음, index.html을 홈 라우터에 랜더링합니다.

`/predict` 라우터에서, 폼이 보내질 때 몇가지 해프닝이 생깁니다:

1. 폼 변수를 모아서 numpy 배열로 변환합니다. 그러면 모델로 보내지고 예측이 반환됩니다.
2. 국가를 보여줄 때는 예상된 국가 코드에서 읽을 수 있는 텍스트로 다시 랜더링하고, 이 값을 템플릿에서 랜더링할 수 있게 index.html로 보냅니다.

Flask와 pickled 모델과 같이, 모델을 사용하는 이 방식은, 비교적으로 간단합니다. 어려운 것은 예측을 받기 위해서 모델에 줄 데이터의 모양을 이해해야 한다는 것입니다. 모든 모델이 어떻게 훈련받았는 지에 따릅니다. 예측을 받기 위해서 3개 데이터 포인트를 넣어야 합니다.

전문 세팅에서, 모델을 훈련하는 사람과 웹 또는 모바일 앱에서 사용하는 사람 사이 얼마나 좋은 소통이 필요한 지 알 수 있습니다. 이 케이스는, 오직 한 사람, 당신입니다!

---

## 🚀 도전

노트북에서 작성하고 Flask 앱에서 모델을 가져오는 대신, Flask 앱에서 바로 모델을 훈련할 수 있습니다!  어쩌면 데이터를 정리하고, 노트북에서 Python 코드로 변환해서, `train`이라고 불리는 라우터로 앱에서 모델을 훈련합니다. 이러한 방식을 추구했을 때 장점과 단점은 무엇인가요?


## [강의 후 퀴즈](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/18/)

## 검토 & 자기주도 학습

ML 모델로 웹 앱을 만드는 방식은 많습니다. JavaScript 또는 Python으로 머신러닝을 활용하는 웹 앱의 제작 방식에 대한 목록을 만듭니다. 구조를 고려합니다: 모델이 앱이나 클라우드에 있나요? 만약 후자의 경우, 어떻게 접근하나요? 적용한 ML 웹 솔루션에 대해 아키텍쳐 모델을 그립니다.

## 과제 

[Try a different model](../assignment.md)


