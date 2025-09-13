<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "e0b75f73e4a90d45181dc5581fe2ef5c",
  "translation_date": "2025-09-05T10:25:48+00:00",
  "source_file": "3-Web-App/1-Web-App/README.md",
  "language_code": "hi"
}
-->
# वेब ऐप बनाएं जो ML मॉडल का उपयोग करे

इस पाठ में, आप एक डेटा सेट पर ML मॉडल को प्रशिक्षित करेंगे जो अद्भुत है: _पिछली सदी में UFO देखे जाने की घटनाएं_, जो NUFORC के डेटाबेस से ली गई हैं।

आप सीखेंगे:

- प्रशिक्षित मॉडल को 'पिकल' कैसे करें
- उस मॉडल का उपयोग Flask ऐप में कैसे करें

हम नोटबुक्स का उपयोग जारी रखेंगे ताकि डेटा को साफ किया जा सके और मॉडल को प्रशिक्षित किया जा सके, लेकिन आप इस प्रक्रिया को एक कदम आगे ले जा सकते हैं और मॉडल को 'वाइल्ड' में उपयोग करने का अनुभव प्राप्त कर सकते हैं, यानी एक वेब ऐप में।

इसके लिए, आपको Flask का उपयोग करके एक वेब ऐप बनाना होगा।

## [प्री-लेक्चर क्विज़](https://ff-quizzes.netlify.app/en/ml/)

## ऐप बनाना

मशीन लर्निंग मॉडल को उपयोग करने के लिए वेब ऐप बनाने के कई तरीके हैं। आपकी वेब आर्किटेक्चर आपके मॉडल के प्रशिक्षण के तरीके को प्रभावित कर सकती है। कल्पना करें कि आप एक व्यवसाय में काम कर रहे हैं जहां डेटा साइंस समूह ने एक मॉडल प्रशिक्षित किया है जिसे वे ऐप में उपयोग करना चाहते हैं।

### विचार करने योग्य बातें

आपको कई सवाल पूछने की आवश्यकता होगी:

- **क्या यह वेब ऐप है या मोबाइल ऐप?** यदि आप मोबाइल ऐप बना रहे हैं या मॉडल को IoT संदर्भ में उपयोग करने की आवश्यकता है, तो आप [TensorFlow Lite](https://www.tensorflow.org/lite/) का उपयोग कर सकते हैं और मॉडल को Android या iOS ऐप में उपयोग कर सकते हैं।
- **मॉडल कहां रहेगा?** क्लाउड में या लोकल?
- **ऑफ़लाइन समर्थन।** क्या ऐप को ऑफ़लाइन काम करना होगा?
- **मॉडल को प्रशिक्षित करने के लिए कौन सी तकनीक का उपयोग किया गया था?** चुनी गई तकनीक आपके उपयोग किए जाने वाले टूलिंग को प्रभावित कर सकती है।
    - **TensorFlow का उपयोग।** यदि आप TensorFlow का उपयोग करके मॉडल को प्रशिक्षित कर रहे हैं, तो उदाहरण के लिए, यह इकोसिस्टम [TensorFlow.js](https://www.tensorflow.org/js/) का उपयोग करके वेब ऐप में उपयोग के लिए TensorFlow मॉडल को कनवर्ट करने की क्षमता प्रदान करता है।
    - **PyTorch का उपयोग।** यदि आप [PyTorch](https://pytorch.org/) जैसी लाइब्रेरी का उपयोग करके मॉडल बना रहे हैं, तो आपके पास इसे [ONNX](https://onnx.ai/) (Open Neural Network Exchange) फॉर्मेट में एक्सपोर्ट करने का विकल्प है ताकि इसे JavaScript वेब ऐप्स में उपयोग किया जा सके जो [Onnx Runtime](https://www.onnxruntime.ai/) का उपयोग कर सकते हैं। इस विकल्प को भविष्य के पाठ में Scikit-learn-प्रशिक्षित मॉडल के लिए एक्सप्लोर किया जाएगा।
    - **Lobe.ai या Azure Custom Vision का उपयोग।** यदि आप [Lobe.ai](https://lobe.ai/) या [Azure Custom Vision](https://azure.microsoft.com/services/cognitive-services/custom-vision-service/?WT.mc_id=academic-77952-leestott) जैसे ML SaaS (Software as a Service) सिस्टम का उपयोग करके मॉडल को प्रशिक्षित कर रहे हैं, तो इस प्रकार का सॉफ़्टवेयर कई प्लेटफार्मों के लिए मॉडल को एक्सपोर्ट करने के तरीके प्रदान करता है, जिसमें क्लाउड में आपके ऑनलाइन एप्लिकेशन द्वारा क्वेरी किए जाने वाले एक bespoke API का निर्माण शामिल है।

आपके पास एक पूरा Flask वेब ऐप बनाने का अवसर भी है जो वेब ब्राउज़र में ही मॉडल को प्रशिक्षित कर सकता है। इसे JavaScript संदर्भ में TensorFlow.js का उपयोग करके भी किया जा सकता है।

हमारे उद्देश्यों के लिए, चूंकि हम Python-आधारित नोटबुक्स के साथ काम कर रहे हैं, आइए उन चरणों का पता लगाएं जिन्हें आपको प्रशिक्षित मॉडल को नोटबुक से Python-निर्मित वेब ऐप द्वारा पढ़े जाने वाले फॉर्मेट में एक्सपोर्ट करने के लिए उठाने की आवश्यकता है।

## टूल

इस कार्य के लिए, आपको दो टूल्स की आवश्यकता होगी: Flask और Pickle, दोनों Python पर चलते हैं।

✅ [Flask](https://palletsprojects.com/p/flask/) क्या है? इसके निर्माताओं द्वारा 'माइक्रो-फ्रेमवर्क' के रूप में परिभाषित, Flask Python का उपयोग करके वेब फ्रेमवर्क की बुनियादी विशेषताएं प्रदान करता है और वेब पेज बनाने के लिए एक टेम्पलेटिंग इंजन का उपयोग करता है। Flask के साथ निर्माण का अभ्यास करने के लिए [इस Learn module](https://docs.microsoft.com/learn/modules/python-flask-build-ai-web-app?WT.mc_id=academic-77952-leestott) को देखें।

✅ [Pickle](https://docs.python.org/3/library/pickle.html) क्या है? Pickle 🥒 एक Python मॉड्यूल है जो Python ऑब्जेक्ट संरचना को सीरियलाइज़ और डी-सीरियलाइज़ करता है। जब आप किसी मॉडल को 'पिकल' करते हैं, तो आप उसकी संरचना को वेब पर उपयोग के लिए सीरियलाइज़ या फ्लैटन करते हैं। सावधान रहें: पिकल स्वाभाविक रूप से सुरक्षित नहीं है, इसलिए यदि किसी फ़ाइल को 'अन-पिकल' करने के लिए कहा जाए तो सावधान रहें। एक पिकल्ड फ़ाइल का उपसर्ग `.pkl` होता है।

## अभ्यास - अपने डेटा को साफ करें

इस पाठ में आप 80,000 UFO देखे जाने की घटनाओं के डेटा का उपयोग करेंगे, जिसे [NUFORC](https://nuforc.org) (The National UFO Reporting Center) द्वारा एकत्र किया गया है। इस डेटा में UFO देखे जाने की कुछ दिलचस्प विवरण हैं, जैसे:

- **लंबा विवरण उदाहरण।** "एक आदमी रात में एक घास के मैदान पर चमकने वाली रोशनी की किरण से बाहर निकलता है और वह टेक्सास इंस्ट्रूमेंट्स पार्किंग लॉट की ओर दौड़ता है।"
- **छोटा विवरण उदाहरण।** "लाइट्स ने हमारा पीछा किया।"

[ufos.csv](../../../../3-Web-App/1-Web-App/data/ufos.csv) स्प्रेडशीट में `city`, `state` और `country` जहां घटना हुई, ऑब्जेक्ट का `shape` और उसका `latitude` और `longitude` जैसे कॉलम शामिल हैं।

इस पाठ में शामिल खाली [notebook](../../../../3-Web-App/1-Web-App/notebook.ipynb) में:

1. पिछले पाठों में आपने जैसे `pandas`, `matplotlib`, और `numpy` को इम्पोर्ट किया था, वैसे ही करें और ufos स्प्रेडशीट को इम्पोर्ट करें। आप डेटा सेट का नमूना देख सकते हैं:

    ```python
    import pandas as pd
    import numpy as np
    
    ufos = pd.read_csv('./data/ufos.csv')
    ufos.head()
    ```

1. ufos डेटा को नए शीर्षकों के साथ एक छोटे डेटा फ्रेम में बदलें। `Country` फ़ील्ड में अद्वितीय मानों की जांच करें।

    ```python
    ufos = pd.DataFrame({'Seconds': ufos['duration (seconds)'], 'Country': ufos['country'],'Latitude': ufos['latitude'],'Longitude': ufos['longitude']})
    
    ufos.Country.unique()
    ```

1. अब, आप उन डेटा को कम कर सकते हैं जिनसे हमें निपटना है, किसी भी null मानों को हटाकर और केवल 1-60 सेकंड के बीच की घटनाओं को इम्पोर्ट करके:

    ```python
    ufos.dropna(inplace=True)
    
    ufos = ufos[(ufos['Seconds'] >= 1) & (ufos['Seconds'] <= 60)]
    
    ufos.info()
    ```

1. Scikit-learn की `LabelEncoder` लाइब्रेरी को इम्पोर्ट करें ताकि देशों के टेक्स्ट मानों को एक संख्या में बदल सकें:

    ✅ LabelEncoder डेटा को वर्णानुक्रम में एन्कोड करता है

    ```python
    from sklearn.preprocessing import LabelEncoder
    
    ufos['Country'] = LabelEncoder().fit_transform(ufos['Country'])
    
    ufos.head()
    ```

    आपका डेटा ऐसा दिखना चाहिए:

    ```output
    	Seconds	Country	Latitude	Longitude
    2	20.0	3		53.200000	-2.916667
    3	20.0	4		28.978333	-96.645833
    14	30.0	4		35.823889	-80.253611
    23	60.0	4		45.582778	-122.352222
    24	3.0		3		51.783333	-0.783333
    ```

## अभ्यास - अपना मॉडल बनाएं

अब आप डेटा को प्रशिक्षण और परीक्षण समूह में विभाजित करके मॉडल को प्रशिक्षित करने के लिए तैयार हो सकते हैं।

1. तीन फीचर्स चुनें जिन पर आप प्रशिक्षण देना चाहते हैं, जो आपका X वेक्टर होगा, और y वेक्टर `Country` होगा। आप `Seconds`, `Latitude` और `Longitude` को इनपुट करना चाहते हैं और एक देश का आईडी प्राप्त करना चाहते हैं।

    ```python
    from sklearn.model_selection import train_test_split
    
    Selected_features = ['Seconds','Latitude','Longitude']
    
    X = ufos[Selected_features]
    y = ufos['Country']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    ```

1. अपने मॉडल को लॉजिस्टिक रिग्रेशन का उपयोग करके प्रशिक्षित करें:

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

सटीकता बुरी नहीं है **(लगभग 95%)**, जो आश्चर्यजनक नहीं है, क्योंकि `Country` और `Latitude/Longitude` सहसंबद्ध हैं।

आपका बनाया गया मॉडल बहुत क्रांतिकारी नहीं है क्योंकि आप `Latitude` और `Longitude` से देश का अनुमान लगा सकते हैं, लेकिन यह कच्चे डेटा से प्रशिक्षण देने, उसे साफ करने, एक्सपोर्ट करने और फिर इस मॉडल का वेब ऐप में उपयोग करने का एक अच्छा अभ्यास है।

## अभ्यास - अपने मॉडल को 'पिकल' करें

अब, समय है अपने मॉडल को _पिकल_ करने का! आप इसे कुछ लाइनों के कोड में कर सकते हैं। एक बार जब यह _पिकल_ हो जाए, तो अपने पिकल्ड मॉडल को लोड करें और इसे सेकंड, अक्षांश और देशांतर के मानों वाले नमूना डेटा ऐरे के खिलाफ परीक्षण करें।

```python
import pickle
model_filename = 'ufo-model.pkl'
pickle.dump(model, open(model_filename,'wb'))

model = pickle.load(open('ufo-model.pkl','rb'))
print(model.predict([[50,44,-12]]))
```

मॉडल **'3'** लौटाता है, जो UK का देश कोड है। अद्भुत! 👽

## अभ्यास - Flask ऐप बनाएं

अब आप एक Flask ऐप बना सकते हैं जो आपके मॉडल को कॉल करता है और समान परिणाम लौटाता है, लेकिन अधिक आकर्षक तरीके से।

1. **web-app** नामक एक फ़ोल्डर बनाएं, जो _notebook.ipynb_ फ़ाइल के बगल में हो, जहां आपका _ufo-model.pkl_ फ़ाइल स्थित है।

1. उस फ़ोल्डर में तीन और फ़ोल्डर बनाएं: **static**, जिसमें **css** नामक एक फ़ोल्डर हो, और **templates**। अब आपके पास निम्नलिखित फ़ाइलें और डायरेक्टरी होनी चाहिए:

    ```output
    web-app/
      static/
        css/
      templates/
    notebook.ipynb
    ufo-model.pkl
    ```

    ✅ समाधान फ़ोल्डर को अंतिम ऐप का दृश्य देखने के लिए देखें

1. _web-app_ फ़ोल्डर में बनाने वाली पहली फ़ाइल **requirements.txt** है। JavaScript ऐप में _package.json_ की तरह, यह फ़ाइल ऐप द्वारा आवश्यक डिपेंडेंसी को सूचीबद्ध करती है। **requirements.txt** में निम्नलिखित पंक्तियां जोड़ें:

    ```text
    scikit-learn
    pandas
    numpy
    flask
    ```

1. अब, इस फ़ाइल को _web-app_ में नेविगेट करके चलाएं:

    ```bash
    cd web-app
    ```

1. अपने टर्मिनल में `pip install` टाइप करें, ताकि _requirements.txt_ में सूचीबद्ध लाइब्रेरीज़ को इंस्टॉल किया जा सके:

    ```bash
    pip install -r requirements.txt
    ```

1. अब, आप ऐप को पूरा करने के लिए तीन और फ़ाइलें बनाने के लिए तैयार हैं:

    1. **app.py** को रूट में बनाएं।
    2. **index.html** को _templates_ डायरेक्टरी में बनाएं।
    3. **styles.css** को _static/css_ डायरेक्टरी में बनाएं।

1. _styles.css_ फ़ाइल को कुछ स्टाइल्स के साथ बनाएं:

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

1. अगला, _index.html_ फ़ाइल बनाएं:

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

    इस फ़ाइल में टेम्पलेटिंग को देखें। ध्यान दें कि ऐप द्वारा प्रदान किए जाने वाले वेरिएबल्स के चारों ओर 'मस्टाच' सिंटैक्स है, जैसे कि प्रेडिक्शन टेक्स्ट: `{{}}`। इसमें `/predict` रूट पर पोस्ट करने वाला एक फॉर्म भी है।

    अंत में, आप उस Python फ़ाइल को बनाने के लिए तैयार हैं जो मॉडल की खपत और प्रेडिक्शन के प्रदर्शन को संचालित करती है:

1. `app.py` में जोड़ें:

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

    > 💡 टिप: जब आप Flask का उपयोग करके वेब ऐप चलाते समय [`debug=True`](https://www.askpython.com/python-modules/flask/flask-debug-mode) जोड़ते हैं, तो आपके एप्लिकेशन में किए गए किसी भी बदलाव तुरंत प्रतिबिंबित होंगे, बिना सर्वर को पुनः आरंभ किए। सावधान रहें! इस मोड को प्रोडक्शन ऐप में सक्षम न करें।

यदि आप `python app.py` या `python3 app.py` चलाते हैं - आपका वेब सर्वर स्थानीय रूप से शुरू हो जाता है, और आप एक छोटा फॉर्म भर सकते हैं ताकि आपको UFO देखे जाने की घटनाओं के बारे में अपने सवाल का जवाब मिल सके!

ऐसा करने से पहले, `app.py` के भागों पर एक नज़र डालें:

1. सबसे पहले, डिपेंडेंसी लोड होती हैं और ऐप शुरू होता है।
1. फिर, मॉडल इम्पोर्ट होता है।
1. फिर, होम रूट पर index.html रेंडर होता है।

`/predict` रूट पर, जब फॉर्म पोस्ट किया जाता है, तो कई चीजें होती हैं:

1. फॉर्म वेरिएबल्स को इकट्ठा किया जाता है और numpy ऐरे में कनवर्ट किया जाता है। फिर उन्हें मॉडल में भेजा जाता है और एक प्रेडिक्शन लौटाया जाता है।
2. जिन देशों को हम प्रदर्शित करना चाहते हैं, उन्हें उनके प्रेडिक्टेड देश कोड से पठनीय टेक्स्ट के रूप में पुनः रेंडर किया जाता है, और वह मान index.html में टेम्पलेट में रेंडर करने के लिए वापस भेजा जाता है।

इस तरह Flask और पिकल्ड मॉडल के साथ मॉडल का उपयोग करना अपेक्षाकृत सरल है। सबसे कठिन बात यह समझना है कि मॉडल को प्रेडिक्शन प्राप्त करने के लिए किस प्रकार का डेटा भेजा जाना चाहिए। यह पूरी तरह से इस बात पर निर्भर करता है कि मॉडल को कैसे प्रशिक्षित किया गया था। इस मॉडल में प्रेडिक्शन प्राप्त करने के लिए तीन डेटा पॉइंट्स को इनपुट करना होता है।

एक पेशेवर सेटिंग में, आप देख सकते हैं कि मॉडल को प्रशिक्षित करने वाले लोगों और इसे वेब या मोबाइल ऐप में उपयोग करने वाले लोगों के बीच अच्छा संचार कितना आवश्यक है। हमारे मामले में, यह केवल एक व्यक्ति है, आप!

---

## 🚀 चुनौती

नोटबुक में काम करने और मॉडल को Flask ऐप में इम्पोर्ट करने के बजाय, आप मॉडल को Flask ऐप के भीतर ही प्रशिक्षित कर सकते हैं! अपने Python कोड को नोटबुक में कनवर्ट करने का प्रयास करें, शायद आपके डेटा को साफ करने के बाद, ताकि ऐप के भीतर `train` नामक रूट पर मॉडल को प्रशिक्षित किया जा सके। इस विधि को अपनाने के फायदे और नुकसान क्या हैं?

## [पोस्ट-लेक्चर क्विज़](https://ff-quizzes.netlify.app/en/ml/)

## समीक्षा और स्व-अध्ययन

ML मॉडल को उपयोग करने के लिए वेब ऐप बनाने के कई तरीके हैं। उन तरीकों की सूची बनाएं जिनसे आप JavaScript या Python का उपयोग करके वेब ऐप बना सकते हैं। आर्किटेक्चर पर विचार करें: क्या मॉडल ऐप में रहना चाहिए या क्लाउड में? यदि बाद वाला, तो आप इसे कैसे एक्सेस करेंगे? एक लागू ML वेब समाधान के लिए एक आर्किटेक्चरल मॉडल बनाएं।

## असाइनमेंट

[एक अलग मॉडल आज़माएं](assignment.md)

---

**अस्वीकरण**:  
यह दस्तावेज़ AI अनुवाद सेवा [Co-op Translator](https://github.com/Azure/co-op-translator) का उपयोग करके अनुवादित किया गया है। जबकि हम सटीकता के लिए प्रयासरत हैं, कृपया ध्यान दें कि स्वचालित अनुवाद में त्रुटियां या अशुद्धियां हो सकती हैं। मूल भाषा में उपलब्ध मूल दस्तावेज़ को आधिकारिक स्रोत माना जाना चाहिए। महत्वपूर्ण जानकारी के लिए, पेशेवर मानव अनुवाद की सिफारिश की जाती है। इस अनुवाद के उपयोग से उत्पन्न किसी भी गलतफहमी या गलत व्याख्या के लिए हम उत्तरदायी नहीं हैं।  