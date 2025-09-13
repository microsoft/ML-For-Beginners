<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "e0b75f73e4a90d45181dc5581fe2ef5c",
  "translation_date": "2025-09-06T06:35:56+00:00",
  "source_file": "3-Web-App/1-Web-App/README.md",
  "language_code": "ne"
}
-->
# वेब एप निर्माण गरेर ML मोडेल प्रयोग गर्नुहोस्

यस पाठमा, तपाईं एक डेटा सेटमा ML मोडेललाई प्रशिक्षण दिनेछौं जुन अद्भुत छ: _पछिल्लो शताब्दीको UFO देखाइहरू_, जुन NUFORC को डाटाबेसबाट लिइएको छ।

तपाईं सिक्नुहुनेछ:

- कसरी 'pickle' गरेर प्रशिक्षित मोडेललाई सुरक्षित गर्ने
- कसरी त्यो मोडेललाई Flask एपमा प्रयोग गर्ने

हामी नोटबुकहरू प्रयोग गरेर डेटा सफा गर्ने र मोडेललाई प्रशिक्षण दिने प्रक्रिया जारी राख्नेछौं, तर तपाईं यस प्रक्रियालाई अर्को चरणमा लैजान सक्नुहुन्छ, जस्तै वेब एपमा मोडेललाई प्रयोग गर्ने।

यसका लागि, तपाईंलाई Flask प्रयोग गरेर वेब एप निर्माण गर्न आवश्यक छ।

## [पाठ अघि क्विज](https://ff-quizzes.netlify.app/en/ml/)

## एप निर्माण

मेसिन लर्निङ मोडेललाई उपभोग गर्न वेब एप निर्माण गर्ने विभिन्न तरिकाहरू छन्। तपाईंको वेब आर्किटेक्चरले तपाईंको मोडेल प्रशिक्षण गर्ने तरिकालाई प्रभाव पार्न सक्छ। कल्पना गर्नुहोस् कि तपाईं एक व्यवसायमा काम गर्दै हुनुहुन्छ जहाँ डेटा विज्ञान समूहले मोडेल प्रशिक्षण गरेको छ जुन उनीहरूले तपाईंलाई एपमा प्रयोग गर्न चाहन्छन्।

### विचारहरू

तपाईंले धेरै प्रश्नहरू सोध्न आवश्यक छ:

- **यो वेब एप हो कि मोबाइल एप?** यदि तपाईं मोबाइल एप निर्माण गर्दै हुनुहुन्छ वा IoT सन्दर्भमा मोडेल प्रयोग गर्न आवश्यक छ भने, तपाईं [TensorFlow Lite](https://www.tensorflow.org/lite/) प्रयोग गर्न सक्नुहुन्छ र मोडेललाई Android वा iOS एपमा प्रयोग गर्न सक्नुहुन्छ।
- **मोडेल कहाँ रहनेछ?** क्लाउडमा कि स्थानीय रूपमा?
- **अफलाइन समर्थन।** के एपले अफलाइन काम गर्नुपर्छ?
- **मोडेल प्रशिक्षण गर्न कुन प्रविधि प्रयोग गरिएको थियो?** चयन गरिएको प्रविधिले तपाईंले प्रयोग गर्नुपर्ने उपकरणलाई प्रभाव पार्न सक्छ।
    - **TensorFlow प्रयोग गर्दै।** यदि तपाईं TensorFlow प्रयोग गरेर मोडेल प्रशिक्षण गर्दै हुनुहुन्छ भने, उदाहरणका लागि, त्यो इकोसिस्टमले [TensorFlow.js](https://www.tensorflow.org/js/) प्रयोग गरेर वेब एपमा प्रयोग गर्न मोडेललाई रूपान्तरण गर्ने क्षमता प्रदान गर्दछ।
    - **PyTorch प्रयोग गर्दै।** यदि तपाईं [PyTorch](https://pytorch.org/) जस्तो लाइब्रेरी प्रयोग गरेर मोडेल निर्माण गर्दै हुनुहुन्छ भने, तपाईंले यसलाई [ONNX](https://onnx.ai/) (Open Neural Network Exchange) फर्म्याटमा निर्यात गर्न सक्नुहुन्छ जसले [Onnx Runtime](https://www.onnxruntime.ai/) प्रयोग गर्ने JavaScript वेब एपहरूमा प्रयोग गर्न सकिन्छ। यो विकल्प भविष्यको पाठमा Scikit-learn-प्रशिक्षित मोडेलको लागि अन्वेषण गरिनेछ।
    - **Lobe.ai वा Azure Custom Vision प्रयोग गर्दै।** यदि तपाईं [Lobe.ai](https://lobe.ai/) वा [Azure Custom Vision](https://azure.microsoft.com/services/cognitive-services/custom-vision-service/?WT.mc_id=academic-77952-leestott) जस्ता ML SaaS (Software as a Service) प्रणाली प्रयोग गरेर मोडेल प्रशिक्षण गर्दै हुनुहुन्छ भने, यस प्रकारको सफ्टवेयरले धेरै प्लेटफर्महरूको लागि मोडेल निर्यात गर्ने तरिकाहरू प्रदान गर्दछ, जसमा क्लाउडमा तपाईंको अनलाइन एपद्वारा सोध्न सकिने API निर्माण गर्ने विकल्प पनि समावेश छ।

तपाईंले एक सम्पूर्ण Flask वेब एप निर्माण गर्ने अवसर पनि पाउन सक्नुहुन्छ जसले वेब ब्राउजरमै मोडेललाई प्रशिक्षण गर्न सक्दछ। यो TensorFlow.js प्रयोग गरेर JavaScript सन्दर्भमा पनि गर्न सकिन्छ।

हाम्रो उद्देश्यका लागि, किनकि हामी Python-आधारित नोटबुकहरूसँग काम गर्दैछौं, नोटबुकबाट Python-निर्मित वेब एपद्वारा पढ्न सकिने फर्म्याटमा प्रशिक्षित मोडेल निर्यात गर्न आवश्यक चरणहरू अन्वेषण गरौं।

## उपकरण

यस कार्यका लागि, तपाईंलाई दुई उपकरणहरू आवश्यक छ: Flask र Pickle, दुवै Pythonमा चल्छन्।

✅ [Flask](https://palletsprojects.com/p/flask/) के हो? यसको निर्माताहरूले 'माइक्रो-फ्रेमवर्क' भनेर परिभाषित गरेका Flaskले Python प्रयोग गरेर वेब फ्रेमवर्कहरूको आधारभूत सुविधाहरू प्रदान गर्दछ र वेब पृष्ठहरू निर्माण गर्न टेम्प्लेटिङ इन्जिन प्रयोग गर्दछ। [यो Learn मोड्युल](https://docs.microsoft.com/learn/modules/python-flask-build-ai-web-app?WT.mc_id=academic-77952-leestott) हेर्नुहोस् Flaskसँग निर्माण अभ्यास गर्न।

✅ [Pickle](https://docs.python.org/3/library/pickle.html) के हो? Pickle 🥒 एक Python मोड्युल हो जसले Python वस्तु संरचनालाई सिरियलाइज र डी-सिरियलाइज गर्दछ। जब तपाईं मोडेललाई 'pickle' गर्नुहुन्छ, तपाईं यसको संरचनालाई वेबमा प्रयोग गर्न सिरियलाइज वा फ्ल्याट गर्नुहुन्छ। सावधान रहनुहोस्: pickle स्वाभाविक रूपमा सुरक्षित छैन, त्यसैले यदि तपाईंलाई फाइल 'un-pickle' गर्न सोधिएको छ भने सावधान रहनुहोस्। एक pickled फाइलको suffix `.pkl` हुन्छ।

## अभ्यास - आफ्नो डेटा सफा गर्नुहोस्

यस पाठमा तपाईं 80,000 UFO देखाइहरूको डेटा प्रयोग गर्नुहुनेछ, [NUFORC](https://nuforc.org) (The National UFO Reporting Center) द्वारा संकलित। यस डेटामा UFO देखाइहरूको केही रोचक विवरणहरू छन्, उदाहरणका लागि:

- **लामो विवरण उदाहरण।** "एक व्यक्ति रातको समयमा घाँसे मैदानमा चम्किरहेको प्रकाशको किरणबाट बाहिर निस्कन्छ र टेक्सास इन्स्ट्रुमेन्ट्सको पार्किङ क्षेत्रमा दौडन्छ।"
- **छोटो विवरण उदाहरण।** "बत्तीहरूले हामीलाई पछ्यायो।"

[ufos.csv](../../../../3-Web-App/1-Web-App/data/ufos.csv) स्प्रेडशीटमा `city`, `state` र `country` जहाँ देखाइ भएको थियो, वस्तुको `shape` र यसको `latitude` र `longitude` बारेका स्तम्भहरू समावेश छन्।

यस पाठमा समावेश गरिएको खाली [notebook](../../../../3-Web-App/1-Web-App/notebook.ipynb) मा:

1. `pandas`, `matplotlib`, र `numpy` लाई आयात गर्नुहोस् जस्तै तपाईंले अघिल्लो पाठहरूमा गरेको थियो र ufos स्प्रेडशीट आयात गर्नुहोस्। तपाईं डेटा सेटको नमूना हेर्न सक्नुहुन्छ:

    ```python
    import pandas as pd
    import numpy as np
    
    ufos = pd.read_csv('./data/ufos.csv')
    ufos.head()
    ```

1. ufos डेटालाई नयाँ शीर्षकहरू सहित सानो डेटा फ्रेममा रूपान्तरण गर्नुहोस्। `Country` फिल्डमा अद्वितीय मानहरू जाँच गर्नुहोस्।

    ```python
    ufos = pd.DataFrame({'Seconds': ufos['duration (seconds)'], 'Country': ufos['country'],'Latitude': ufos['latitude'],'Longitude': ufos['longitude']})
    
    ufos.Country.unique()
    ```

1. अब, तपाईंले आवश्यक डेटा घटाउन सक्नुहुन्छ null मानहरू हटाएर र 1-60 सेकेन्डको बीचमा देखाइहरू मात्र आयात गरेर:

    ```python
    ufos.dropna(inplace=True)
    
    ufos = ufos[(ufos['Seconds'] >= 1) & (ufos['Seconds'] <= 60)]
    
    ufos.info()
    ```

1. Scikit-learn को `LabelEncoder` लाइब्रेरी आयात गर्नुहोस् ताकि देशहरूको पाठ मानलाई नम्बरमा रूपान्तरण गर्न सकियोस्:

    ✅ LabelEncoder डेटा वर्णानुक्रममा एन्कोड गर्दछ

    ```python
    from sklearn.preprocessing import LabelEncoder
    
    ufos['Country'] = LabelEncoder().fit_transform(ufos['Country'])
    
    ufos.head()
    ```

    तपाईंको डेटा यस प्रकार देखिनुपर्छ:

    ```output
    	Seconds	Country	Latitude	Longitude
    2	20.0	3		53.200000	-2.916667
    3	20.0	4		28.978333	-96.645833
    14	30.0	4		35.823889	-80.253611
    23	60.0	4		45.582778	-122.352222
    24	3.0		3		51.783333	-0.783333
    ```

## अभ्यास - आफ्नो मोडेल निर्माण गर्नुहोस्

अब तपाईं आफ्नो डेटा प्रशिक्षण र परीक्षण समूहमा विभाजन गरेर मोडेल प्रशिक्षण गर्न तयार हुनुहुन्छ।

1. तपाईंले प्रशिक्षण गर्न चाहनुभएको तीन विशेषताहरूलाई आफ्नो X भेक्टरको रूपमा चयन गर्नुहोस्, र y भेक्टर `Country` हुनेछ। तपाईंले `Seconds`, `Latitude` र `Longitude` इनपुट गर्न चाहनुहुन्छ र देशको id प्राप्त गर्न चाहनुहुन्छ।

    ```python
    from sklearn.model_selection import train_test_split
    
    Selected_features = ['Seconds','Latitude','Longitude']
    
    X = ufos[Selected_features]
    y = ufos['Country']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    ```

1. आफ्नो मोडेललाई logistic regression प्रयोग गरेर प्रशिक्षण दिनुहोस्:

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

सटीकता खराब छैन **(लगभग 95%)**, आश्चर्यजनक रूपमा, किनकि `Country` र `Latitude/Longitude` सम्बन्धित छन्।

तपाईंले निर्माण गरेको मोडेल धेरै क्रान्तिकारी छैन किनकि तपाईंले `Latitude` र `Longitude` बाट `Country` अनुमान गर्न सक्नुहुन्छ, तर यो कच्चा डेटा सफा गरेर, निर्यात गरेर, र त्यसपछि वेब एपमा प्रयोग गर्ने मोडेल प्रशिक्षण गर्ने राम्रो अभ्यास हो।

## अभ्यास - आफ्नो मोडेललाई 'pickle' गर्नुहोस्

अब, आफ्नो मोडेललाई _pickle_ गर्ने समय हो! तपाईंले यसलाई केही लाइनको कोडमा गर्न सक्नुहुन्छ। एक पटक _pickled_ भएपछि, आफ्नो pickled मोडेललाई लोड गर्नुहोस् र सेकेन्ड, latitude र longitude को मानहरू समावेश गर्ने नमूना डेटा एरे विरुद्ध परीक्षण गर्नुहोस्,

```python
import pickle
model_filename = 'ufo-model.pkl'
pickle.dump(model, open(model_filename,'wb'))

model = pickle.load(open('ufo-model.pkl','rb'))
print(model.predict([[50,44,-12]]))
```

मोडेलले **'3'** फिर्ता गर्छ, जुन UK को देश कोड हो। अद्भुत! 👽

## अभ्यास - Flask एप निर्माण गर्नुहोस्

अब तपाईं आफ्नो मोडेललाई कल गर्न र समान परिणामहरू फिर्ता गर्न, तर अधिक दृश्यात्मक रूपमा आकर्षक तरिकामा, एक Flask एप निर्माण गर्न सक्नुहुन्छ।

1. _notebook.ipynb_ फाइलको छेउमा **web-app** नामक फोल्डर बनाउनुहोस् जहाँ तपाईंको _ufo-model.pkl_ फाइल रहेको छ।

1. त्यस फोल्डरमा तीन थप फोल्डरहरू बनाउनुहोस्: **static**, जसको भित्र **css** फोल्डर छ, र **templates**। तपाईंले अब निम्न फाइलहरू र डाइरेक्टरीहरू पाउनुपर्छ:

    ```output
    web-app/
      static/
        css/
      templates/
    notebook.ipynb
    ufo-model.pkl
    ```

    ✅ समाप्त एपको दृश्यको लागि समाधान फोल्डरलाई हेर्नुहोस्

1. _web-app_ फोल्डरमा पहिलो फाइल **requirements.txt** बनाउनुहोस्। जस्तै _package.json_ JavaScript एपमा, यो फाइलले एपले आवश्यक पर्ने निर्भरता सूचीबद्ध गर्दछ। **requirements.txt** मा निम्न लाइनहरू थप्नुहोस्:

    ```text
    scikit-learn
    pandas
    numpy
    flask
    ```

1. अब, _web-app_ मा नेभिगेट गरेर यो फाइल चलाउनुहोस्:

    ```bash
    cd web-app
    ```

1. आफ्नो टर्मिनलमा `pip install` टाइप गर्नुहोस्, _requirements.txt_ मा सूचीबद्ध लाइब्रेरीहरू स्थापना गर्न:

    ```bash
    pip install -r requirements.txt
    ```

1. अब, तपाईं एप समाप्त गर्न तीन थप फाइलहरू बनाउन तयार हुनुहुन्छ:

    1. **app.py** जडमा बनाउनुहोस्।
    2. _templates_ डाइरेक्टरीमा **index.html** बनाउनुहोस्।
    3. _static/css_ डाइरेक्टरीमा **styles.css** बनाउनुहोस्।

1. _styles.css_ फाइललाई केही शैलीहरू सहित निर्माण गर्नुहोस्:

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

1. त्यसपछि, _index.html_ फाइल निर्माण गर्नुहोस्:

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

    यस फाइलमा टेम्प्लेटिङलाई हेर्नुहोस्। ध्यान दिनुहोस् कि एपद्वारा प्रदान गरिने भेरिएबलहरू वरिपरि 'mustache' सिन्ट्याक्स छ, जस्तै भविष्यवाणी पाठ: `{{}}`। त्यहाँ `/predict` रूटमा पोस्ट गर्ने फर्म पनि छ।

    अन्ततः, तपाईंले मोडेलको उपभोग र भविष्यवाणीहरूको प्रदर्शनलाई चलाउने python फाइल निर्माण गर्न तयार हुनुहुन्छ:

1. `app.py` मा थप्नुहोस्:

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

    > 💡 टिप: जब तपाईं Flask प्रयोग गरेर वेब एप चलाउँदा [`debug=True`](https://www.askpython.com/python-modules/flask/flask-debug-mode) थप्नुहुन्छ, तपाईंले आफ्नो एप्लिकेसनमा गरेको कुनै पनि परिवर्तन तुरुन्तै प्रतिबिम्बित हुनेछ बिना सर्भर पुनः सुरु गर्न आवश्यक। सावधान रहनुहोस्! उत्पादन एपमा यो मोड सक्षम नगर्नुहोस्।

यदि तपाईं `python app.py` वा `python3 app.py` चलाउनुहुन्छ - तपाईंको वेब सर्भर स्थानीय रूपमा सुरु हुन्छ, र तपाईं एक छोटो फर्म भर्न सक्नुहुन्छ ताकि UFO देखाइहरू कहाँ भएको छ भन्ने प्रश्नको उत्तर प्राप्त गर्न सकियोस्!

त्यसअघि, `app.py` का भागहरू हेर्नुहोस्:

1. पहिलो, निर्भरता लोड गरिन्छ र एप सुरु हुन्छ।
1. त्यसपछि, मोडेल आयात गरिन्छ।
1. त्यसपछि, home रूटमा index.html प्रस्तुत गरिन्छ।

`/predict` रूटमा, फर्म पोस्ट हुँदा धेरै कुराहरू हुन्छन्:

1. फर्म भेरिएबलहरू संकलन गरिन्छ र numpy एरेमा रूपान्तरण गरिन्छ। तिनीहरू मोडेलमा पठाइन्छन् र भविष्यवाणी फिर्ता गरिन्छ।
2. हामीले प्रदर्शन गर्न चाहेको देशहरू भविष्यवाणी गरिएको देश कोडबाट पठनीय पाठको रूपमा पुनः प्रस्तुत गरिन्छ, र त्यो मान index.html मा टेम्प्लेटमा प्रस्तुत गर्न फिर्ता पठाइन्छ।

Flask र pickled मोडेलको साथ मोडेल प्रयोग गर्ने यो तरिका तुलनात्मक रूपमा सरल छ। सबैभन्दा कठिन कुरा भनेको मोडेललाई भविष्यवाणी प्राप्त गर्न पठाउनुपर्ने डेटा कस्तो आकारको छ भन्ने बुझ्नु हो। त्यो सबै मोडेल कसरी प्रशिक्षण गरिएको थियो भन्नेमा निर्भर गर्दछ। यसमा भविष्यवाणी प्राप्त गर्न तीन डेटा बिन्दुहरू इनपुट गर्नुपर्छ।

व्यावसायिक सेटिङमा, तपाईं देख्न सक्नुहुन्छ कि मोडेल प्रशिक्षण गर्ने व्यक्तिहरू र वेब वा मोबाइल एपमा उपभोग गर्ने व्यक्तिहरू बीच राम्रो संचार आवश्यक छ। हाम्रो मामलामा, यो केवल एक व्यक्ति हो, तपाईं!

---

## 🚀 चुनौती

नोटबुकमा काम गरेर मोडेललाई Flask एपमा आयात गर्ने सट्टा, तपाईं मोडेललाई Flask एपमै प्रशिक्षण दिन सक्नुहुन्छ! आफ्नो Python कोडलाई नोटबुकमा रूपान्तरण गर्ने प्रयास गर्नुहोस्, सम्भवतः आफ्नो डेटा सफा गरेपछि, एपभित्र `train` नामक रूटमा मोडेल प्रशिक्षण गर्न। यस विधि अपनाउँदा के फाइदा र बेफाइदा छन्?

## [पाठ पछि क्विज](https://ff-quizzes.netlify.app/en/ml/)

## समीक्षा र आत्म अध्ययन

मेसिन लर्निङ मोडेललाई उपभोग गर्न वेब एप निर्माण गर्ने धेरै तरिकाहरू छन्। JavaScript वा Python प्रयोग गरेर वेब एप निर्माण गर्न सकिने तरिकाहरूको सूची बनाउनुहोस्। आर्किटेक्चर विचार गर्नुहोस्: के मोडेल एपमै रहनुपर्छ कि क्लाउडमा? यदि पछिल्लो हो भने, तपाईंले यसलाई कसरी पहुँच गर्नुहुन्छ? लागू गरिएको ML वेब समाधानको लागि आर्किटेक्चरल मोडेल बनाउनुहोस्।

## असाइनमेन्ट

[अर्को मोडेल प्रयास गर्नुहोस्](assignment.md)

---

**अस्वीकरण**:  
यो दस्तावेज़ AI अनुवाद सेवा [Co-op Translator](https://github.com/Azure/co-op-translator) प्रयोग गरेर अनुवाद गरिएको छ। हामी शुद्धताको लागि प्रयास गर्छौं, तर कृपया ध्यान दिनुहोस् कि स्वचालित अनुवादहरूमा त्रुटि वा अशुद्धता हुन सक्छ। यसको मूल भाषा मा रहेको मूल दस्तावेज़लाई आधिकारिक स्रोत मानिनुपर्छ। महत्वपूर्ण जानकारीको लागि, व्यावसायिक मानव अनुवाद सिफारिस गरिन्छ। यस अनुवादको प्रयोगबाट उत्पन्न हुने कुनै पनि गलतफहमी वा गलत व्याख्याको लागि हामी जिम्मेवार हुने छैनौं।