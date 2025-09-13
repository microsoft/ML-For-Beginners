<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "e0b75f73e4a90d45181dc5581fe2ef5c",
  "translation_date": "2025-09-06T06:15:23+00:00",
  "source_file": "3-Web-App/1-Web-App/README.md",
  "language_code": "mr"
}
-->
# वेब अ‍ॅप तयार करा ज्यामध्ये ML मॉडेल वापरता येईल

या धड्यात, तुम्ही एका अनोख्या डेटासेटवर ML मॉडेल ट्रेन कराल: _गेल्या शतकातील UFO दिसण्याच्या घटना_, ज्याचे स्रोत NUFORC च्या डेटाबेसमधून घेतले आहे.

तुम्ही शिकाल:

- प्रशिक्षित मॉडेल 'pickle' कसे करायचे
- Flask अ‍ॅपमध्ये ते मॉडेल कसे वापरायचे

आम्ही डेटा साफ करण्यासाठी आणि आमचे मॉडेल ट्रेन करण्यासाठी नोटबुक्सचा वापर सुरू ठेवू, परंतु तुम्ही प्रक्रियेला पुढे नेऊन 'जंगली' वापरासाठी मॉडेल वापरण्याचा शोध घेऊ शकता: एका वेब अ‍ॅपमध्ये.

हे करण्यासाठी, तुम्हाला Flask वापरून वेब अ‍ॅप तयार करावे लागेल.

## [पूर्व-व्याख्यान क्विझ](https://ff-quizzes.netlify.app/en/ml/)

## अ‍ॅप तयार करणे

मशीन लर्निंग मॉडेल्स वापरण्यासाठी वेब अ‍ॅप्स तयार करण्याचे अनेक मार्ग आहेत. तुमची वेब आर्किटेक्चर तुमच्या मॉडेलच्या ट्रेनिंग पद्धतीवर परिणाम करू शकते. कल्पना करा की तुम्ही अशा व्यवसायात काम करत आहात जिथे डेटा सायन्स ग्रुपने एक मॉडेल तयार केले आहे जे ते तुम्हाला अ‍ॅपमध्ये वापरण्यास सांगत आहेत.

### विचार करण्यासारखे मुद्दे

तुम्हाला अनेक प्रश्न विचारावे लागतील:

- **वेब अ‍ॅप आहे की मोबाइल अ‍ॅप?** जर तुम्ही मोबाइल अ‍ॅप तयार करत असाल किंवा IoT संदर्भात मॉडेल वापरण्याची गरज असेल, तर तुम्ही [TensorFlow Lite](https://www.tensorflow.org/lite/) वापरू शकता आणि मॉडेल Android किंवा iOS अ‍ॅपमध्ये वापरू शकता.
- **मॉडेल कुठे ठेवले जाईल?** क्लाउडमध्ये की स्थानिक पातळीवर?
- **ऑफलाइन समर्थन.** अ‍ॅपला ऑफलाइन काम करावे लागेल का?
- **मॉडेल ट्रेन करण्यासाठी कोणती तंत्रज्ञान वापरली गेली?** निवडलेले तंत्रज्ञान तुम्हाला वापरायचे टूल्स प्रभावित करू शकते.
    - **TensorFlow वापरणे.** उदाहरणार्थ, जर तुम्ही TensorFlow वापरून मॉडेल ट्रेन करत असाल, तर त्या इकोसिस्टममध्ये [TensorFlow.js](https://www.tensorflow.org/js/) वापरून वेब अ‍ॅपसाठी TensorFlow मॉडेल रूपांतरित करण्याची क्षमता आहे.
    - **PyTorch वापरणे.** जर तुम्ही [PyTorch](https://pytorch.org/) सारख्या लायब्ररीचा वापर करून मॉडेल तयार करत असाल, तर तुम्ही ते [ONNX](https://onnx.ai/) (Open Neural Network Exchange) फॉरमॅटमध्ये निर्यात करू शकता, जे JavaScript वेब अ‍ॅप्ससाठी वापरले जाऊ शकते जे [Onnx Runtime](https://www.onnxruntime.ai/) वापरतात. हा पर्याय भविष्यातील धड्यात Scikit-learn-ट्रेन केलेल्या मॉडेलसाठी शोधला जाईल.
    - **Lobe.ai किंवा Azure Custom Vision वापरणे.** जर तुम्ही [Lobe.ai](https://lobe.ai/) किंवा [Azure Custom Vision](https://azure.microsoft.com/services/cognitive-services/custom-vision-service/?WT.mc_id=academic-77952-leestott) सारख्या ML SaaS (Software as a Service) प्रणालीचा वापर करून मॉडेल ट्रेन करत असाल, तर या प्रकारचे सॉफ्टवेअर अनेक प्लॅटफॉर्मसाठी मॉडेल निर्यात करण्याचे मार्ग प्रदान करते, ज्यामध्ये क्लाउडमध्ये तुमच्या ऑनलाइन अ‍ॅपद्वारे क्वेरी करण्यासाठी एक API तयार करणे समाविष्ट आहे.

तुम्हाला संपूर्ण Flask वेब अ‍ॅप तयार करण्याची संधी देखील आहे जे वेब ब्राउझरमध्ये स्वतःच मॉडेल ट्रेन करू शकेल. हे TensorFlow.js वापरून JavaScript संदर्भात देखील करता येते.

आमच्या उद्देशासाठी, कारण आम्ही Python-आधारित नोटबुक्ससह काम करत आहोत, चला अशा नोटबुकमधून Python-निर्मित वेब अ‍ॅपद्वारे वाचण्यायोग्य स्वरूपात प्रशिक्षित मॉडेल निर्यात करण्यासाठी आवश्यक असलेल्या चरणांचा शोध घेऊ.

## टूल

या कार्यासाठी तुम्हाला दोन टूल्सची आवश्यकता आहे: Flask आणि Pickle, जे दोन्ही Python वर चालतात.

✅ [Flask](https://palletsprojects.com/p/flask/) म्हणजे काय? त्याच्या निर्मात्यांनी 'मायक्रो-फ्रेमवर्क' म्हणून परिभाषित केलेले, Flask Python वापरून वेब फ्रेमवर्कची मूलभूत वैशिष्ट्ये आणि वेब पृष्ठे तयार करण्यासाठी टेम्पलेटिंग इंजिन प्रदान करते. Flask वापरून तयार करण्याचा सराव करण्यासाठी [हा Learn मॉड्यूल](https://docs.microsoft.com/learn/modules/python-flask-build-ai-web-app?WT.mc_id=academic-77952-leestott) पहा.

✅ [Pickle](https://docs.python.org/3/library/pickle.html) म्हणजे काय? Pickle 🥒 हा एक Python मॉड्यूल आहे जो Python ऑब्जेक्ट स्ट्रक्चरला सिरीयलाइज आणि डी-सिरीयलाइज करतो. जेव्हा तुम्ही मॉडेल 'pickle' करता, तेव्हा तुम्ही त्याचे स्ट्रक्चर वेबवर वापरण्यासाठी सिरीयलाइज किंवा फ्लॅटन करता. काळजी घ्या: pickle स्वाभाविकपणे सुरक्षित नाही, त्यामुळे जर तुम्हाला एखादी फाइल 'un-pickle' करण्यास सांगितले गेले तर काळजी घ्या. Pickled फाइलला `.pkl` हा प्रत्यय असतो.

## व्यायाम - तुमचा डेटा साफ करा

या धड्यात तुम्ही [NUFORC](https://nuforc.org) (The National UFO Reporting Center) कडून गोळा केलेल्या 80,000 UFO दिसण्याच्या घटनांचा डेटा वापराल. या डेटामध्ये UFO दिसण्याच्या काही मनोरंजक वर्णनांचा समावेश आहे, उदाहरणार्थ:

- **लांब उदाहरण वर्णन.** "एका प्रकाशाच्या किरणातून एक माणूस बाहेर पडतो जो रात्री गवताच्या मैदानावर चमकतो आणि तो Texas Instruments पार्किंग लॉटकडे धावतो".
- **लहान उदाहरण वर्णन.** "प्रकाश आमच्यामागे लागला".

[ufos.csv](../../../../3-Web-App/1-Web-App/data/ufos.csv) स्प्रेडशीटमध्ये `city`, `state` आणि `country` जिथे दिसण्याची घटना घडली, ऑब्जेक्टचा `shape` आणि त्याचा `latitude` आणि `longitude` याबद्दलचे कॉलम समाविष्ट आहेत.

या धड्यात समाविष्ट केलेल्या रिक्त [notebook](../../../../3-Web-App/1-Web-App/notebook.ipynb) मध्ये:

1. `pandas`, `matplotlib`, आणि `numpy` आयात करा जसे तुम्ही मागील धड्यांमध्ये केले आणि ufos स्प्रेडशीट आयात करा. तुम्ही नमुना डेटासेट पाहू शकता:

    ```python
    import pandas as pd
    import numpy as np
    
    ufos = pd.read_csv('./data/ufos.csv')
    ufos.head()
    ```

1. ufos डेटा नवीन शीर्षकांसह एका छोट्या डेटाफ्रेममध्ये रूपांतरित करा. `Country` फील्डमधील युनिक व्हॅल्यूज तपासा.

    ```python
    ufos = pd.DataFrame({'Seconds': ufos['duration (seconds)'], 'Country': ufos['country'],'Latitude': ufos['latitude'],'Longitude': ufos['longitude']})
    
    ufos.Country.unique()
    ```

1. आता, तुम्ही आवश्यक डेटा कमी करून null व्हॅल्यूज काढून टाकू शकता आणि फक्त 1-60 सेकंदांच्या दरम्यानच्या दिसण्याच्या घटनांचा डेटा आयात करू शकता:

    ```python
    ufos.dropna(inplace=True)
    
    ufos = ufos[(ufos['Seconds'] >= 1) & (ufos['Seconds'] <= 60)]
    
    ufos.info()
    ```

1. Scikit-learn च्या `LabelEncoder` लायब्ररी आयात करा जेणेकरून देशांसाठी टेक्स्ट व्हॅल्यूजला नंबरमध्ये रूपांतरित करता येईल:

    ✅ LabelEncoder डेटा वर्णानुक्रमाने एन्कोड करते

    ```python
    from sklearn.preprocessing import LabelEncoder
    
    ufos['Country'] = LabelEncoder().fit_transform(ufos['Country'])
    
    ufos.head()
    ```

    तुमचा डेटा असा दिसायला हवा:

    ```output
    	Seconds	Country	Latitude	Longitude
    2	20.0	3		53.200000	-2.916667
    3	20.0	4		28.978333	-96.645833
    14	30.0	4		35.823889	-80.253611
    23	60.0	4		45.582778	-122.352222
    24	3.0		3		51.783333	-0.783333
    ```

## व्यायाम - तुमचे मॉडेल तयार करा

आता तुम्ही डेटा ट्रेनिंग आणि टेस्टिंग गटात विभागून मॉडेल ट्रेन करण्यासाठी तयार होऊ शकता.

1. तुमच्या X व्हेक्टरसाठी ट्रेन करण्यासाठी तीन फीचर्स निवडा, आणि y व्हेक्टर `Country` असेल. तुम्हाला `Seconds`, `Latitude` आणि `Longitude` इनपुट करायचे आहे आणि देशाचा आयडी परत मिळवायचा आहे.

    ```python
    from sklearn.model_selection import train_test_split
    
    Selected_features = ['Seconds','Latitude','Longitude']
    
    X = ufos[Selected_features]
    y = ufos['Country']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    ```

1. लॉजिस्टिक रिग्रेशन वापरून तुमचे मॉडेल ट्रेन करा:

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

अचूकता वाईट नाही **(सुमारे 95%)**, आश्चर्यकारक नाही, कारण `Country` आणि `Latitude/Longitude` यांचा परस्पर संबंध आहे.

तुम्ही तयार केलेले मॉडेल फारसे क्रांतिकारी नाही कारण तुम्ही `Latitude` आणि `Longitude` वरून `Country` काढू शकता, परंतु कच्च्या डेटावरून ट्रेनिंग करण्याचा, डेटा साफ करण्याचा, निर्यात करण्याचा आणि नंतर वेब अ‍ॅपमध्ये मॉडेल वापरण्याचा सराव करण्यासाठी हे एक चांगले व्यायाम आहे.

## व्यायाम - तुमचे मॉडेल 'pickle' करा

आता, तुमचे मॉडेल _pickle_ करण्याची वेळ आली आहे! तुम्ही ते काही ओळींच्या कोडमध्ये करू शकता. एकदा ते _pickled_ झाले की, तुमचे pickled मॉडेल लोड करा आणि सेकंद, latitude आणि longitude साठी व्हॅल्यूज असलेल्या नमुना डेटा अ‍ॅरेवर ते टेस्ट करा,

```python
import pickle
model_filename = 'ufo-model.pkl'
pickle.dump(model, open(model_filename,'wb'))

model = pickle.load(open('ufo-model.pkl','rb'))
print(model.predict([[50,44,-12]]))
```

मॉडेल **'3'** परत करते, जे UK साठी देश कोड आहे. आश्चर्यकारक! 👽

## व्यायाम - Flask अ‍ॅप तयार करा

आता तुम्ही Flask अ‍ॅप तयार करू शकता जे तुमच्या मॉडेलला कॉल करेल आणि त्याच परिणामांना अधिक आकर्षक पद्धतीने परत करेल.

1. _notebook.ipynb_ फाइलच्या शेजारी **web-app** नावाचा फोल्डर तयार करा जिथे तुमची _ufo-model.pkl_ फाइल आहे.

1. त्या फोल्डरमध्ये आणखी तीन फोल्डर्स तयार करा: **static**, ज्यामध्ये **css** नावाचा फोल्डर आहे, आणि **templates**. आता तुमच्याकडे खालील फाइल्स आणि डिरेक्टरीज असाव्यात:

    ```output
    web-app/
      static/
        css/
      templates/
    notebook.ipynb
    ufo-model.pkl
    ```

    ✅ तयार अ‍ॅपचा आढावा घेण्यासाठी सोल्यूशन फोल्डरचा संदर्भ घ्या

1. _web-app_ फोल्डरमध्ये तयार करायची पहिली फाइल म्हणजे **requirements.txt** फाइल. JavaScript अ‍ॅपमधील _package.json_ प्रमाणे, ही फाइल अ‍ॅपसाठी आवश्यक असलेल्या डिपेंडन्सींची यादी देते. **requirements.txt** मध्ये खालील ओळी जोडा:

    ```text
    scikit-learn
    pandas
    numpy
    flask
    ```

1. आता, _web-app_ मध्ये नेव्हिगेट करून ही फाइल चालवा:

    ```bash
    cd web-app
    ```

1. तुमच्या टर्मिनलमध्ये `pip install` टाइप करा, जेणेकरून _requirements.txt_ मध्ये सूचीबद्ध लायब्ररी इंस्टॉल होतील:

    ```bash
    pip install -r requirements.txt
    ```

1. आता, अ‍ॅप पूर्ण करण्यासाठी आणखी तीन फाइल्स तयार करण्यासाठी तुम्ही तयार आहात:

    1. **app.py** रूटमध्ये तयार करा.
    2. _templates_ डिरेक्टरीमध्ये **index.html** तयार करा.
    3. _static/css_ डिरेक्टरीमध्ये **styles.css** तयार करा.

1. _styles.css_ फाइल काही स्टाइल्ससह तयार करा:

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

1. पुढे, _index.html_ फाइल तयार करा:

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

    या फाइलमधील टेम्पलेटिंगकडे लक्ष द्या. अ‍ॅपद्वारे प्रदान केलेल्या व्हेरिएबल्सभोवती 'mustache' सिंटॅक्स आहे, जसे की प्रेडिक्शन टेक्स्ट: `{{}}`. येथे एक फॉर्म देखील आहे जो `/predict` रूटवर प्रेडिक्शन पोस्ट करतो.

    शेवटी, तुम्ही मॉडेलचा वापर आणि प्रेडिक्शन प्रदर्शित करण्यासाठी Python फाइल तयार करण्यासाठी तयार आहात:

1. `app.py` मध्ये जोडा:

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

    > 💡 टिप: Flask वापरून वेब अ‍ॅप चालवताना [`debug=True`](https://www.askpython.com/python-modules/flask/flask-debug-mode) जोडल्यास, तुम्ही तुमच्या अ‍ॅप्लिकेशनमध्ये केलेले कोणतेही बदल त्वरित प्रतिबिंबित होतील, सर्व्हर पुन्हा सुरू करण्याची गरज नाही. सावध रहा! उत्पादन अ‍ॅपमध्ये हा मोड सक्षम करू नका.

जर तुम्ही `python app.py` किंवा `python3 app.py` चालवले - तुमचा वेब सर्व्हर स्थानिक पातळीवर सुरू होतो, आणि तुम्ही एक लहान फॉर्म भरून UFO दिसण्याच्या घटनांबद्दल तुमच्या जिज्ञासेचे उत्तर मिळवू शकता!

ते करण्यापूर्वी, `app.py` च्या भागांकडे लक्ष द्या:

1. प्रथम, डिपेंडन्सी लोड केल्या जातात आणि अ‍ॅप सुरू होते.
1. नंतर, मॉडेल आयात केले जाते.
1. नंतर, होम रूटवर index.html रेंडर केले जाते.

`/predict` रूटवर, फॉर्म पोस्ट केल्यावर अनेक गोष्टी घडतात:

1. फॉर्म व्हेरिएबल्स गोळा करून numpy अ‍ॅरेमध्ये रूपांतरित केल्या जातात. त्यानंतर ते मॉडेलला पाठवले जातात आणि प्रेडिक्शन परत केले जाते.
2. आम्हाला प्रदर्शित करायचे असलेले देश त्यांच्या प्रेडिक्टेड देश कोडमधून वाचण्यायोग्य टेक्स्ट म्हणून पुन्हा रेंडर केले जातात, आणि ती व्हॅल्यू index.html मध्ये परत पाठवली जाते जेणेकरून ती टेम्पलेटमध्ये रेंडर केली जाईल.

Flask आणि pickled मॉडेलसह मॉडेल वापरणे तुलनेने सोपे आहे. सर्वात कठीण गोष्ट म्हणजे मॉडेलला प्रेडिक्शन मिळवण्यासाठी पाठवले जाणारे डेटा कोणत्या स्वरूपात असले पाहिजे हे समजून घेणे. ते सर्व मॉडेल कसे ट्रेन केले गेले यावर अवलंबून असते. या मॉडेलसाठी तीन डेटा पॉइंट्स इनपुट करणे आवश्यक आहे जेणेकरून प्रेडिक्शन मिळेल.

व्यावसायिक सेटिंगमध्ये, तुम्ही पाहू शकता की मॉडेल ट्रेन करणाऱ्या लोकांमध्ये आणि वेब किंवा मोबाइल अ‍ॅपमध्ये त्याचा वापर करणाऱ्या लोकांमध्ये चांगल्या संवादाची गरज किती महत्त्वाची आहे. आमच्या बाबतीत, तो फक्त एक व्यक्ती आहे, तुम्ही!

---

## 🚀 आव्हान

नोटबुकमध्ये काम करून Flask अ‍ॅपमध्ये मॉडेल आयात करण्याऐवजी, तुम्ही Flask अ‍ॅपमध्येच मॉडेल ट्रेन करू शकता! तुमच्या नोटबुकमधील Python कोड, कदाचित तुमचा डेटा साफ केल्यानंतर, अ‍ॅपमध्ये `train` नावाच्या रूटवर मॉडेल ट्रेन करण्यासाठी रूपांतरित करण्याचा प्रयत्न करा. ही पद्धत स्वीकारण्याचे फायदे आणि तोटे काय आहेत?

## [व्याख्यानानंतरचा क्विझ](https://ff-quizzes.netlify.app/en/ml/)

## पुनरावलोकन आणि स्व-अभ्यास

ML मॉडेल्स वापरण्यासाठी वेब अ‍ॅप तयार करण्याचे अनेक मार्ग आहेत. JavaScript किंवा Python वापरून मशीन लर्निंगचा लाभ घेण्यासाठी वेब अ‍ॅप तयार करण्याचे मार्गांची यादी तयार करा. आर्किटेक्चर विचार करा: मॉडेल अ‍ॅपमध्ये राहावे की क्लाउडमध्ये? जर क्लाउडमध्ये असेल, तर तुम्ही ते कसे ऍक्सेस कराल? लागू ML वेब सोल्यूशनसाठी आर्किटेक्चरल मॉडेल काढा.

## असाइनमेंट

[वेगळे मॉडेल वापरून पहा](assignment.md)

---

**अस्वीकरण**:  
हा दस्तऐवज AI भाषांतर सेवा [Co-op Translator](https://github.com/Azure/co-op-translator) चा वापर करून भाषांतरित करण्यात आला आहे. आम्ही अचूकतेसाठी प्रयत्नशील असलो तरी, कृपया लक्षात घ्या की स्वयंचलित भाषांतरांमध्ये त्रुटी किंवा अचूकतेचा अभाव असू शकतो. मूळ भाषेतील मूळ दस्तऐवज हा अधिकृत स्रोत मानला जावा. महत्त्वाच्या माहितीसाठी व्यावसायिक मानवी भाषांतराची शिफारस केली जाते. या भाषांतराचा वापर केल्यामुळे उद्भवणाऱ्या कोणत्याही गैरसमज किंवा चुकीच्या अर्थासाठी आम्ही जबाबदार राहणार नाही.