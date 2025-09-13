<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "e0b75f73e4a90d45181dc5581fe2ef5c",
  "translation_date": "2025-09-06T08:54:15+00:00",
  "source_file": "3-Web-App/1-Web-App/README.md",
  "language_code": "ur"
}
-->
# ویب ایپ بنائیں جو مشین لرننگ ماڈل استعمال کرے

اس سبق میں، آپ ایک مشین لرننگ ماڈل کو ایک ایسے ڈیٹا سیٹ پر تربیت دیں گے جو واقعی منفرد ہے: _پچھلی صدی کے دوران UFO مشاہدات_، جو NUFORC کے ڈیٹا بیس سے حاصل کیے گئے ہیں۔

آپ سیکھیں گے:

- تربیت یافتہ ماڈل کو 'pickle' کیسے کریں
- اس ماڈل کو Flask ایپ میں کیسے استعمال کریں

ہم نوٹ بکس کا استعمال جاری رکھیں گے تاکہ ڈیٹا کو صاف کریں اور ماڈل کو تربیت دیں، لیکن آپ اس عمل کو ایک قدم آگے لے جا سکتے ہیں اور ماڈل کو "جنگل میں" استعمال کرنے کا تجربہ کر سکتے ہیں، یعنی ایک ویب ایپ میں۔

اس کے لیے، آپ کو Flask استعمال کرتے ہوئے ایک ویب ایپ بنانی ہوگی۔

## [لیکچر سے پہلے کا کوئز](https://ff-quizzes.netlify.app/en/ml/)

## ایپ بنانا

مشین لرننگ ماڈلز کو استعمال کرنے کے لیے ویب ایپس بنانے کے کئی طریقے ہیں۔ آپ کی ویب آرکیٹیکچر اس بات پر اثر ڈال سکتی ہے کہ آپ کا ماڈل کیسے تربیت یافتہ ہے۔ تصور کریں کہ آپ ایک ایسے کاروبار میں کام کر رہے ہیں جہاں ڈیٹا سائنس گروپ نے ایک ماڈل تربیت دیا ہے جسے وہ چاہتے ہیں کہ آپ ایپ میں استعمال کریں۔

### غور و فکر

آپ کو کئی سوالات پوچھنے کی ضرورت ہوگی:

- **کیا یہ ویب ایپ ہے یا موبائل ایپ؟** اگر آپ موبائل ایپ بنا رہے ہیں یا ماڈل کو IoT کے تناظر میں استعمال کرنے کی ضرورت ہے، تو آپ [TensorFlow Lite](https://www.tensorflow.org/lite/) استعمال کر سکتے ہیں اور ماڈل کو Android یا iOS ایپ میں استعمال کر سکتے ہیں۔
- **ماڈل کہاں ہوگا؟** کلاؤڈ میں یا مقامی طور پر؟
- **آف لائن سپورٹ۔** کیا ایپ کو آف لائن کام کرنا ہوگا؟
- **ماڈل کو تربیت دینے کے لیے کونسی ٹیکنالوجی استعمال کی گئی؟** منتخب کردہ ٹیکنالوجی اس ٹولنگ پر اثر ڈال سکتی ہے جسے آپ استعمال کریں گے۔
    - **TensorFlow کا استعمال۔** اگر آپ TensorFlow استعمال کرتے ہوئے ماڈل تربیت دے رہے ہیں، تو یہ ایکو سسٹم [TensorFlow.js](https://www.tensorflow.org/js/) کے ذریعے ماڈل کو ویب ایپ میں استعمال کے لیے تبدیل کرنے کی صلاحیت فراہم کرتا ہے۔
    - **PyTorch کا استعمال۔** اگر آپ [PyTorch](https://pytorch.org/) جیسی لائبریری استعمال کرتے ہوئے ماڈل بنا رہے ہیں، تو آپ اسے [ONNX](https://onnx.ai/) (Open Neural Network Exchange) فارمیٹ میں ایکسپورٹ کر سکتے ہیں تاکہ جاوا اسکرپٹ ویب ایپس میں استعمال کیا جا سکے جو [Onnx Runtime](https://www.onnxruntime.ai/) استعمال کرتی ہیں۔ یہ آپشن ایک آئندہ سبق میں Scikit-learn تربیت یافتہ ماڈل کے لیے دریافت کیا جائے گا۔
    - **Lobe.ai یا Azure Custom Vision کا استعمال۔** اگر آپ [Lobe.ai](https://lobe.ai/) یا [Azure Custom Vision](https://azure.microsoft.com/services/cognitive-services/custom-vision-service/?WT.mc_id=academic-77952-leestott) جیسے ML SaaS (Software as a Service) سسٹم کا استعمال کرتے ہوئے ماڈل تربیت دے رہے ہیں، تو یہ قسم کا سافٹ ویئر کئی پلیٹ فارمز کے لیے ماڈل ایکسپورٹ کرنے کے طریقے فراہم کرتا ہے، بشمول کلاؤڈ میں آپ کی آن لائن ایپلیکیشن کے ذریعے استفسار کرنے کے لیے ایک حسب ضرورت API بنانا۔

آپ کے پاس ایک مکمل Flask ویب ایپ بنانے کا موقع بھی ہے جو خود ماڈل کو ویب براؤزر میں تربیت دے سکتی ہے۔ یہ کام جاوا اسکرپٹ کے تناظر میں TensorFlow.js کا استعمال کرتے ہوئے بھی کیا جا سکتا ہے۔

ہمارے مقصد کے لیے، چونکہ ہم Python پر مبنی نوٹ بکس کے ساتھ کام کر رہے ہیں، آئیے ان مراحل کو دریافت کریں جو آپ کو ایک تربیت یافتہ ماڈل کو ایک ایسے فارمیٹ میں ایکسپورٹ کرنے کے لیے لینے کی ضرورت ہے جو Python سے بنی ویب ایپ کے ذریعے پڑھا جا سکے۔

## ٹول

اس کام کے لیے آپ کو دو ٹولز کی ضرورت ہوگی: Flask اور Pickle، دونوں Python پر چلتے ہیں۔

✅ [Flask](https://palletsprojects.com/p/flask/) کیا ہے؟ اس کے تخلیق کاروں کے مطابق، Flask ایک 'مائیکرو فریم ورک' ہے جو Python کا استعمال کرتے ہوئے ویب فریم ورک کی بنیادی خصوصیات فراہم کرتا ہے اور ویب صفحات بنانے کے لیے ایک ٹیمپلیٹنگ انجن فراہم کرتا ہے۔ [اس لرن ماڈیول](https://docs.microsoft.com/learn/modules/python-flask-build-ai-web-app?WT.mc_id=academic-77952-leestott) کو دیکھیں تاکہ Flask کے ساتھ کام کرنے کی مشق کریں۔

✅ [Pickle](https://docs.python.org/3/library/pickle.html) کیا ہے؟ Pickle 🥒 ایک Python ماڈیول ہے جو Python آبجیکٹ کی ساخت کو سیریلائز اور ڈی سیریلائز کرتا ہے۔ جب آپ ماڈل کو 'pickle' کرتے ہیں، تو آپ اس کی ساخت کو ویب پر استعمال کے لیے سیریلائز یا فلیٹ کرتے ہیں۔ محتاط رہیں: pickle بنیادی طور پر محفوظ نہیں ہے، لہذا اگر آپ کو کسی فائل کو 'un-pickle' کرنے کے لیے کہا جائے تو محتاط رہیں۔ ایک pickled فائل کا لاحقہ `.pkl` ہوتا ہے۔

## مشق - اپنے ڈیٹا کو صاف کریں

اس سبق میں آپ 80,000 UFO مشاہدات کے ڈیٹا کا استعمال کریں گے، جو [NUFORC](https://nuforc.org) (The National UFO Reporting Center) کے ذریعے جمع کیے گئے ہیں۔ اس ڈیٹا میں UFO مشاہدات کی دلچسپ تفصیلات شامل ہیں، جیسے:

- **لمبی مثال تفصیل۔** "ایک آدمی روشنی کی ایک شعاع سے نکلتا ہے جو رات کے وقت ایک گھاس کے میدان پر چمکتی ہے اور وہ Texas Instruments کی پارکنگ کی طرف بھاگتا ہے"۔
- **مختصر مثال تفصیل۔** "روشنیوں نے ہمارا پیچھا کیا"۔

[ufos.csv](../../../../3-Web-App/1-Web-App/data/ufos.csv) اسپریڈشیٹ میں `city`, `state`, اور `country` کے کالم شامل ہیں جہاں مشاہدہ ہوا، آبجیکٹ کی `shape` اور اس کے `latitude` اور `longitude`۔

خالی [notebook](../../../../3-Web-App/1-Web-App/notebook.ipynb) میں شامل کریں:

1. `pandas`, `matplotlib`, اور `numpy` کو درآمد کریں جیسا کہ آپ نے پچھلے اسباق میں کیا تھا اور ufos اسپریڈشیٹ کو درآمد کریں۔ آپ ڈیٹا سیٹ کا نمونہ دیکھ سکتے ہیں:

    ```python
    import pandas as pd
    import numpy as np
    
    ufos = pd.read_csv('./data/ufos.csv')
    ufos.head()
    ```

1. ufos ڈیٹا کو ایک چھوٹے ڈیٹا فریم میں تبدیل کریں جس میں تازہ عنوانات ہوں۔ `Country` فیلڈ میں منفرد اقدار چیک کریں۔

    ```python
    ufos = pd.DataFrame({'Seconds': ufos['duration (seconds)'], 'Country': ufos['country'],'Latitude': ufos['latitude'],'Longitude': ufos['longitude']})
    
    ufos.Country.unique()
    ```

1. اب، آپ ان ڈیٹا کو کم کر سکتے ہیں جن سے ہمیں نمٹنے کی ضرورت ہے، کسی بھی null اقدار کو ہٹا کر اور صرف 1-60 سیکنڈ کے درمیان مشاہدات کو درآمد کر کے:

    ```python
    ufos.dropna(inplace=True)
    
    ufos = ufos[(ufos['Seconds'] >= 1) & (ufos['Seconds'] <= 60)]
    
    ufos.info()
    ```

1. Scikit-learn کی `LabelEncoder` لائبریری درآمد کریں تاکہ ممالک کے متن کی اقدار کو ایک نمبر میں تبدیل کیا جا سکے:

    ✅ LabelEncoder ڈیٹا کو حروف تہجی کے لحاظ سے انکوڈ کرتا ہے

    ```python
    from sklearn.preprocessing import LabelEncoder
    
    ufos['Country'] = LabelEncoder().fit_transform(ufos['Country'])
    
    ufos.head()
    ```

    آپ کا ڈیٹا اس طرح نظر آنا چاہیے:

    ```output
    	Seconds	Country	Latitude	Longitude
    2	20.0	3		53.200000	-2.916667
    3	20.0	4		28.978333	-96.645833
    14	30.0	4		35.823889	-80.253611
    23	60.0	4		45.582778	-122.352222
    24	3.0		3		51.783333	-0.783333
    ```

## مشق - اپنا ماڈل بنائیں

اب آپ ڈیٹا کو تربیت اور ٹیسٹنگ گروپ میں تقسیم کر کے ماڈل کو تربیت دینے کے لیے تیار ہو سکتے ہیں۔

1. تین فیچرز منتخب کریں جن پر آپ تربیت دینا چاہتے ہیں، جو آپ کا X ویکٹر ہوگا، اور y ویکٹر `Country` ہوگا۔ آپ `Seconds`, `Latitude` اور `Longitude` کو ان پٹ کرنا چاہتے ہیں اور ایک ملک کا آئی ڈی حاصل کرنا چاہتے ہیں۔

    ```python
    from sklearn.model_selection import train_test_split
    
    Selected_features = ['Seconds','Latitude','Longitude']
    
    X = ufos[Selected_features]
    y = ufos['Country']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    ```

1. اپنے ماڈل کو لاجسٹک ریگریشن کا استعمال کرتے ہوئے تربیت دیں:

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

درستگی بری نہیں ہے **(تقریباً 95%)**، جو حیرت انگیز نہیں ہے، کیونکہ `Country` اور `Latitude/Longitude` آپس میں جڑے ہوئے ہیں۔

آپ کا بنایا ہوا ماڈل بہت انقلابی نہیں ہے کیونکہ آپ کو `Latitude` اور `Longitude` سے ایک `Country` کا اندازہ لگانے کے قابل ہونا چاہیے، لیکن یہ ایک اچھا مشق ہے کہ آپ خام ڈیٹا کو صاف کریں، اسے ایکسپورٹ کریں، اور پھر اس ماڈل کو ایک ویب ایپ میں استعمال کریں۔

## مشق - اپنے ماڈل کو 'pickle' کریں

اب، وقت ہے کہ آپ اپنے ماڈل کو _pickle_ کریں! آپ یہ چند لائنوں کے کوڈ میں کر سکتے ہیں۔ ایک بار جب یہ _pickled_ ہو جائے، تو اپنے pickled ماڈل کو لوڈ کریں اور اسے سیکنڈز، latitude اور longitude کے لیے ایک نمونہ ڈیٹا آرے کے خلاف ٹیسٹ کریں،

```python
import pickle
model_filename = 'ufo-model.pkl'
pickle.dump(model, open(model_filename,'wb'))

model = pickle.load(open('ufo-model.pkl','rb'))
print(model.predict([[50,44,-12]]))
```

ماڈل **'3'** واپس کرتا ہے، جو UK کے لیے ملک کا کوڈ ہے۔ حیرت انگیز! 👽

## مشق - Flask ایپ بنائیں

اب آپ ایک Flask ایپ بنا سکتے ہیں جو آپ کے ماڈل کو کال کرے اور اسی طرح کے نتائج واپس کرے، لیکن زیادہ بصری طور پر خوشنما انداز میں۔

1. _notebook.ipynb_ فائل کے ساتھ ایک فولڈر **web-app** بنائیں جہاں آپ کی _ufo-model.pkl_ فائل موجود ہے۔

1. اس فولڈر میں مزید تین فولڈرز بنائیں: **static**, جس کے اندر ایک فولڈر **css** ہو، اور **templates**۔ آپ کے پاس اب درج ذیل فائلیں اور ڈائریکٹریز ہونی چاہئیں:

    ```output
    web-app/
      static/
        css/
      templates/
    notebook.ipynb
    ufo-model.pkl
    ```

    ✅ حل فولڈر کا حوالہ دیں تاکہ مکمل ایپ کا نظارہ حاصل کریں

1. _web-app_ فولڈر میں پہلی فائل **requirements.txt** بنائیں۔ جیسے جاوا اسکرپٹ ایپ میں _package.json_، یہ فائل ایپ کے لیے مطلوبہ ڈپینڈنسیز کی فہرست دیتی ہے۔ **requirements.txt** میں درج ذیل لائنیں شامل کریں:

    ```text
    scikit-learn
    pandas
    numpy
    flask
    ```

1. اب، اس فائل کو _web-app_ میں نیویگیٹ کر کے چلائیں:

    ```bash
    cd web-app
    ```

1. اپنے ٹرمینل میں `pip install` ٹائپ کریں تاکہ _requirements.txt_ میں درج لائبریریوں کو انسٹال کریں:

    ```bash
    pip install -r requirements.txt
    ```

1. اب، آپ ایپ کو مکمل کرنے کے لیے مزید تین فائلیں بنانے کے لیے تیار ہیں:

    1. **app.py** کو روٹ میں بنائیں۔
    2. _templates_ ڈائریکٹری میں **index.html** بنائیں۔
    3. _static/css_ ڈائریکٹری میں **styles.css** بنائیں۔

1. _styles.css_ فائل میں چند اسٹائلز شامل کریں:

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

1. اگلا، _index.html_ فائل بنائیں:

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

    اس فائل میں ٹیمپلیٹنگ کو دیکھیں۔ متغیرات کے ارد گرد 'mustache' نحو پر غور کریں جو ایپ کے ذریعے فراہم کیے جائیں گے، جیسے پیش گوئی کا متن: `{{}}`۔ یہاں ایک فارم بھی ہے جو پیش گوئی کو `/predict` روٹ پر پوسٹ کرتا ہے۔

    آخر میں، آپ Python فائل بنانے کے لیے تیار ہیں جو ماڈل کو استعمال کرے اور پیش گوئیوں کو ظاہر کرے:

1. `app.py` میں شامل کریں:

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

    > 💡 ٹپ: جب آپ Flask کا استعمال کرتے ہوئے ویب ایپ چلاتے وقت [`debug=True`](https://www.askpython.com/python-modules/flask/flask-debug-mode) شامل کرتے ہیں، تو آپ کی ایپلیکیشن میں کی گئی کوئی بھی تبدیلی فوراً ظاہر ہوگی بغیر سرور کو دوبارہ شروع کرنے کی ضرورت کے۔ خبردار! پروڈکشن ایپ میں اس موڈ کو فعال نہ کریں۔

اگر آپ `python app.py` یا `python3 app.py` چلائیں - آپ کا ویب سرور مقامی طور پر شروع ہو جاتا ہے، اور آپ ایک مختصر فارم بھر کر UFO مشاہدات کے بارے میں اپنے سوال کا جواب حاصل کر سکتے ہیں!

اس سے پہلے کہ آپ ایسا کریں، `app.py` کے حصوں پر ایک نظر ڈالیں:

1. پہلے، ڈپینڈنسیز لوڈ کی جاتی ہیں اور ایپ شروع ہوتی ہے۔
1. پھر، ماڈل درآمد کیا جاتا ہے۔
1. پھر، home روٹ پر index.html رینڈر کیا جاتا ہے۔

`/predict` روٹ پر، جب فارم پوسٹ کیا جاتا ہے تو کئی چیزیں ہوتی ہیں:

1. فارم کے متغیرات جمع کیے جاتے ہیں اور numpy آرے میں تبدیل کیے جاتے ہیں۔ پھر انہیں ماڈل پر بھیجا جاتا ہے اور ایک پیش گوئی واپس کی جاتی ہے۔
2. وہ ممالک جنہیں ہم دکھانا چاہتے ہیں، ان کے پیش گوئی کردہ ملک کے کوڈ سے دوبارہ قابل پڑھنے متن کے طور پر رینڈر کیے جاتے ہیں، اور وہ قدر index.html میں ٹیمپلیٹ میں رینڈر کرنے کے لیے واپس بھیجی جاتی ہے۔

اس طرح ماڈل کو Flask اور pickled ماڈل کے ساتھ استعمال کرنا نسبتاً آسان ہے۔ سب سے مشکل چیز یہ سمجھنا ہے کہ ماڈل کو پیش گوئی حاصل کرنے کے لیے کس شکل کے ڈیٹا کی ضرورت ہے۔ یہ سب اس بات پر منحصر ہے کہ ماڈل کو کیسے تربیت دی گئی تھی۔ اس ماڈل میں پیش گوئی حاصل کرنے کے لیے تین ڈیٹا پوائنٹس ان پٹ کرنے کی ضرورت ہے۔

پیشہ ورانہ ماحول میں، آپ دیکھ سکتے ہیں کہ ماڈل کو تربیت دینے والے افراد اور اسے ویب یا موبائل ایپ میں استعمال کرنے والے افراد کے درمیان اچھا مواصلت کتنا ضروری ہے۔ ہمارے معاملے میں، یہ صرف ایک شخص ہے، آپ!

---

## 🚀 چیلنج

نوٹ بک میں کام کرنے اور ماڈل کو Flask ایپ میں درآمد کرنے کے بجائے، آپ ایپ کے اندر ہی ماڈل کو تربیت دے سکتے ہیں! اپنے Python کوڈ کو نوٹ بک میں تبدیل کرنے کی کوشش کریں، شاید ڈیٹا صاف کرنے کے بعد، تاکہ ایپ کے اندر `train` نامی روٹ پر ماڈل کو تربیت دیں۔ اس طریقے کو اپنانے کے فوائد اور نقصانات کیا ہیں؟

## [لیکچر کے بعد کا کوئز](https://ff-quizzes.netlify.app/en/ml/)

## جائزہ اور خود مطالعہ

مشین لرننگ ماڈلز کو استعمال کرنے کے لیے ویب ایپ بنانے کے کئی طریقے ہیں۔ ان طریقوں کی فہرست بنائیں جن کے ذریعے آپ جاوا اسکرپٹ یا Python کا استعمال کرتے ہوئے ویب ایپ بنا سکتے ہیں۔ آرکیٹیکچر پر غور کریں: کیا ماڈل ایپ میں رہنا چاہیے یا کلاؤڈ میں؟ اگر کلاؤڈ میں، تو آپ اسے کیسے حاصل کریں گے؟ مشین لرننگ کے ایک عملی ویب حل کے لیے ایک آرکیٹیکچرل ماڈل بنائیں۔

## اسائنمنٹ

[ایک مختلف ماڈل آزمائیں](assignment.md)

---

**ڈسکلیمر**:  
یہ دستاویز AI ترجمہ سروس [Co-op Translator](https://github.com/Azure/co-op-translator) کا استعمال کرتے ہوئے ترجمہ کی گئی ہے۔ ہم درستگی کے لیے کوشش کرتے ہیں، لیکن براہ کرم آگاہ رہیں کہ خودکار ترجمے میں غلطیاں یا غیر درستیاں ہو سکتی ہیں۔ اصل دستاویز کو اس کی اصل زبان میں مستند ذریعہ سمجھا جانا چاہیے۔ اہم معلومات کے لیے، پیشہ ور انسانی ترجمہ کی سفارش کی جاتی ہے۔ ہم اس ترجمے کے استعمال سے پیدا ہونے والی کسی بھی غلط فہمی یا غلط تشریح کے ذمہ دار نہیں ہیں۔