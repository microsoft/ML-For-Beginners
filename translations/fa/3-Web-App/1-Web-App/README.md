<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "e0b75f73e4a90d45181dc5581fe2ef5c",
  "translation_date": "2025-09-04T22:41:40+00:00",
  "source_file": "3-Web-App/1-Web-App/README.md",
  "language_code": "fa"
}
-->
# ساخت یک اپلیکیشن وب برای استفاده از مدل یادگیری ماشین

در این درس، شما یک مدل یادگیری ماشین را بر روی مجموعه داده‌ای که از این دنیا نیست آموزش خواهید داد: _مشاهدات بشقاب پرنده در طول قرن گذشته_، که از پایگاه داده NUFORC جمع‌آوری شده است.

شما یاد خواهید گرفت:

- چگونه یک مدل آموزش‌دیده را «pickle» کنید
- چگونه از آن مدل در یک اپلیکیشن Flask استفاده کنید

ما همچنان از نوت‌بوک‌ها برای پاکسازی داده‌ها و آموزش مدل استفاده خواهیم کرد، اما می‌توانید این فرآیند را یک قدم جلوتر ببرید و مدل را در دنیای واقعی، به عبارتی در یک اپلیکیشن وب، به کار ببرید.

برای انجام این کار، باید یک اپلیکیشن وب با استفاده از Flask بسازید.

## [پیش‌ آزمون](https://ff-quizzes.netlify.app/en/ml/)

## ساخت اپلیکیشن

راه‌های مختلفی برای ساخت اپلیکیشن‌های وب وجود دارد که بتوانند مدل‌های یادگیری ماشین را مصرف کنند. معماری وب شما ممکن است بر نحوه آموزش مدل تأثیر بگذارد. تصور کنید که در یک کسب‌وکار کار می‌کنید که گروه داده‌کاوی آن یک مدل آموزش داده‌اند و می‌خواهند شما از آن در اپلیکیشن استفاده کنید.

### ملاحظات

سؤالات زیادی وجود دارد که باید بپرسید:

- **آیا اپلیکیشن وب است یا موبایل؟** اگر در حال ساخت یک اپلیکیشن موبایل هستید یا نیاز دارید مدل را در یک زمینه IoT استفاده کنید، می‌توانید از [TensorFlow Lite](https://www.tensorflow.org/lite/) استفاده کنید و مدل را در اپلیکیشن اندروید یا iOS به کار ببرید.
- **مدل کجا قرار خواهد گرفت؟** در فضای ابری یا به صورت محلی؟
- **پشتیبانی آفلاین.** آیا اپلیکیشن باید به صورت آفلاین کار کند؟
- **از چه تکنولوژی برای آموزش مدل استفاده شده است؟** تکنولوژی انتخاب‌شده ممکن است بر ابزارهایی که باید استفاده کنید تأثیر بگذارد.
    - **استفاده از TensorFlow.** اگر مدل را با استفاده از TensorFlow آموزش می‌دهید، این اکوسیستم امکان تبدیل مدل TensorFlow برای استفاده در اپلیکیشن وب را با استفاده از [TensorFlow.js](https://www.tensorflow.org/js/) فراهم می‌کند.
    - **استفاده از PyTorch.** اگر مدل را با استفاده از کتابخانه‌ای مانند [PyTorch](https://pytorch.org/) می‌سازید، می‌توانید آن را در قالب [ONNX](https://onnx.ai/) (Open Neural Network Exchange) برای استفاده در اپلیکیشن‌های وب جاوااسکریپت که از [Onnx Runtime](https://www.onnxruntime.ai/) استفاده می‌کنند، صادر کنید. این گزینه در درس آینده برای مدل آموزش‌دیده با Scikit-learn بررسی خواهد شد.
    - **استفاده از Lobe.ai یا Azure Custom Vision.** اگر از یک سیستم SaaS یادگیری ماشین مانند [Lobe.ai](https://lobe.ai/) یا [Azure Custom Vision](https://azure.microsoft.com/services/cognitive-services/custom-vision-service/?WT.mc_id=academic-77952-leestott) برای آموزش مدل استفاده می‌کنید، این نوع نرم‌افزار راه‌هایی برای صادر کردن مدل برای پلتفرم‌های مختلف، از جمله ساخت یک API سفارشی برای پرس‌وجو در فضای ابری توسط اپلیکیشن آنلاین شما، فراهم می‌کند.

شما همچنین می‌توانید یک اپلیکیشن وب کامل با Flask بسازید که بتواند مدل را در خود مرورگر وب آموزش دهد. این کار همچنین می‌تواند با استفاده از TensorFlow.js در زمینه جاوااسکریپت انجام شود.

برای اهداف ما، از آنجا که با نوت‌بوک‌های مبتنی بر پایتون کار کرده‌ایم، بیایید مراحل لازم برای صادر کردن یک مدل آموزش‌دیده از چنین نوت‌بوکی به فرمتی که توسط یک اپلیکیشن وب ساخته‌شده با پایتون قابل خواندن باشد را بررسی کنیم.

## ابزار

برای این کار، به دو ابزار نیاز دارید: Flask و Pickle، که هر دو بر روی پایتون اجرا می‌شوند.

✅ [Flask](https://palletsprojects.com/p/flask/) چیست؟ Flask که توسط سازندگانش به عنوان یک «میکرو-فریم‌ورک» تعریف شده است، ویژگی‌های پایه‌ای فریم‌ورک‌های وب را با استفاده از پایتون و یک موتور قالب‌سازی برای ساخت صفحات وب فراهم می‌کند. به [این ماژول آموزشی](https://docs.microsoft.com/learn/modules/python-flask-build-ai-web-app?WT.mc_id=academic-77952-leestott) نگاهی بیندازید تا با Flask کار کنید.

✅ [Pickle](https://docs.python.org/3/library/pickle.html) چیست؟ Pickle 🥒 یک ماژول پایتون است که ساختار شیء پایتون را سریال‌سازی و دی‌سریال‌سازی می‌کند. وقتی یک مدل را «pickle» می‌کنید، ساختار آن را برای استفاده در وب سریال‌سازی یا تخت می‌کنید. مراقب باشید: pickle ذاتاً امن نیست، بنابراین اگر از شما خواسته شد یک فایل «un-pickle» کنید، احتیاط کنید. فایل‌های pickled پسوند `.pkl` دارند.

## تمرین - پاکسازی داده‌ها

در این درس، شما از داده‌های ۸۰,۰۰۰ مشاهده بشقاب پرنده که توسط [NUFORC](https://nuforc.org) (مرکز ملی گزارش‌دهی بشقاب پرنده) جمع‌آوری شده است استفاده خواهید کرد. این داده‌ها شامل توضیحات جالبی از مشاهدات بشقاب پرنده هستند، برای مثال:

- **توضیح طولانی نمونه.** "یک مرد از یک پرتو نور که در شب بر روی یک میدان چمن می‌تابد بیرون می‌آید و به سمت پارکینگ Texas Instruments می‌دود".
- **توضیح کوتاه نمونه.** "چراغ‌ها ما را دنبال کردند".

صفحه‌گسترده [ufos.csv](../../../../3-Web-App/1-Web-App/data/ufos.csv) شامل ستون‌هایی درباره `شهر`، `ایالت` و `کشور` محل مشاهده، `شکل` شیء و `عرض جغرافیایی` و `طول جغرافیایی` آن است.

در [نوت‌بوک](../../../../3-Web-App/1-Web-App/notebook.ipynb) خالی که در این درس گنجانده شده است:

1. `pandas`، `matplotlib` و `numpy` را همانطور که در درس‌های قبلی انجام دادید وارد کنید و صفحه‌گسترده ufos را وارد کنید. می‌توانید به یک نمونه داده نگاه کنید:

    ```python
    import pandas as pd
    import numpy as np
    
    ufos = pd.read_csv('./data/ufos.csv')
    ufos.head()
    ```

1. داده‌های ufos را به یک dataframe کوچک با عناوین تازه تبدیل کنید. مقادیر منحصربه‌فرد در فیلد `Country` را بررسی کنید.

    ```python
    ufos = pd.DataFrame({'Seconds': ufos['duration (seconds)'], 'Country': ufos['country'],'Latitude': ufos['latitude'],'Longitude': ufos['longitude']})
    
    ufos.Country.unique()
    ```

1. اکنون می‌توانید مقدار داده‌هایی که باید با آن‌ها کار کنید را با حذف مقادیر null و فقط وارد کردن مشاهدات بین ۱-۶۰ ثانیه کاهش دهید:

    ```python
    ufos.dropna(inplace=True)
    
    ufos = ufos[(ufos['Seconds'] >= 1) & (ufos['Seconds'] <= 60)]
    
    ufos.info()
    ```

1. کتابخانه `LabelEncoder` از Scikit-learn را وارد کنید تا مقادیر متنی کشورها را به عدد تبدیل کنید:

    ✅ LabelEncoder داده‌ها را به صورت الفبایی کدگذاری می‌کند

    ```python
    from sklearn.preprocessing import LabelEncoder
    
    ufos['Country'] = LabelEncoder().fit_transform(ufos['Country'])
    
    ufos.head()
    ```

    داده‌های شما باید به این شکل باشند:

    ```output
    	Seconds	Country	Latitude	Longitude
    2	20.0	3		53.200000	-2.916667
    3	20.0	4		28.978333	-96.645833
    14	30.0	4		35.823889	-80.253611
    23	60.0	4		45.582778	-122.352222
    24	3.0		3		51.783333	-0.783333
    ```

## تمرین - ساخت مدل

اکنون می‌توانید آماده شوید تا مدل را با تقسیم داده‌ها به گروه‌های آموزشی و آزمایشی آموزش دهید.

1. سه ویژگی‌ای که می‌خواهید بر اساس آن‌ها آموزش دهید را به عنوان بردار X انتخاب کنید، و بردار y فیلد `Country` خواهد بود. شما می‌خواهید بتوانید `ثانیه‌ها`، `عرض جغرافیایی` و `طول جغرافیایی` را وارد کنید و یک شناسه کشور دریافت کنید.

    ```python
    from sklearn.model_selection import train_test_split
    
    Selected_features = ['Seconds','Latitude','Longitude']
    
    X = ufos[Selected_features]
    y = ufos['Country']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    ```

1. مدل خود را با استفاده از رگرسیون لجستیک آموزش دهید:

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

دقت بد نیست **(حدود ۹۵٪)**، که تعجب‌آور نیست، زیرا `Country` و `عرض/طول جغرافیایی` با هم مرتبط هستند.

مدلی که ایجاد کردید خیلی انقلابی نیست، زیرا باید بتوانید یک `Country` را از `عرض جغرافیایی` و `طول جغرافیایی` استنباط کنید، اما این یک تمرین خوب است که سعی کنید از داده‌های خامی که پاکسازی کرده‌اید، مدل را آموزش دهید، صادر کنید و سپس از این مدل در یک اپلیکیشن وب استفاده کنید.

## تمرین - «pickle» کردن مدل

اکنون زمان آن است که مدل خود را _pickle_ کنید! می‌توانید این کار را در چند خط کد انجام دهید. پس از _pickle_ کردن، مدل خود را بارگذاری کنید و آن را با یک آرایه داده نمونه که شامل مقادیر ثانیه‌ها، عرض جغرافیایی و طول جغرافیایی است آزمایش کنید.

```python
import pickle
model_filename = 'ufo-model.pkl'
pickle.dump(model, open(model_filename,'wb'))

model = pickle.load(open('ufo-model.pkl','rb'))
print(model.predict([[50,44,-12]]))
```

مدل **'3'** را بازمی‌گرداند، که کد کشور برای بریتانیا است. شگفت‌انگیز! 👽

## تمرین - ساخت اپلیکیشن Flask

اکنون می‌توانید یک اپلیکیشن Flask بسازید تا مدل خود را فراخوانی کنید و نتایج مشابهی را به شکلی زیباتر نمایش دهید.

1. ابتدا یک پوشه به نام **web-app** در کنار فایل _notebook.ipynb_ که فایل _ufo-model.pkl_ شما در آن قرار دارد ایجاد کنید.

1. در آن پوشه سه پوشه دیگر ایجاد کنید: **static**، با یک پوشه **css** داخل آن، و **templates**. اکنون باید فایل‌ها و دایرکتوری‌های زیر را داشته باشید:

    ```output
    web-app/
      static/
        css/
      templates/
    notebook.ipynb
    ufo-model.pkl
    ```

    ✅ به پوشه solution برای مشاهده اپلیکیشن نهایی مراجعه کنید

1. اولین فایلی که باید در پوشه _web-app_ ایجاد کنید فایل **requirements.txt** است. مانند _package.json_ در یک اپلیکیشن جاوااسکریپت، این فایل وابستگی‌های مورد نیاز اپلیکیشن را لیست می‌کند. در **requirements.txt** خطوط زیر را اضافه کنید:

    ```text
    scikit-learn
    pandas
    numpy
    flask
    ```

1. اکنون این فایل را با حرکت به _web-app_ اجرا کنید:

    ```bash
    cd web-app
    ```

1. در ترمینال خود تایپ کنید `pip install`، تا کتابخانه‌های لیست‌شده در _requirements.txt_ نصب شوند:

    ```bash
    pip install -r requirements.txt
    ```

1. اکنون آماده هستید تا سه فایل دیگر برای تکمیل اپلیکیشن ایجاد کنید:

    1. فایل **app.py** را در ریشه ایجاد کنید.
    2. فایل **index.html** را در دایرکتوری _templates_ ایجاد کنید.
    3. فایل **styles.css** را در دایرکتوری _static/css_ ایجاد کنید.

1. فایل _styles.css_ را با چند سبک بسازید:

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

1. سپس فایل _index.html_ را بسازید:

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

    به قالب‌بندی در این فایل نگاه کنید. به سینتکس 'mustache' اطراف متغیرهایی که توسط اپلیکیشن ارائه خواهند شد، مانند متن پیش‌بینی: `{{}}` توجه کنید. همچنین یک فرم وجود دارد که یک پیش‌بینی را به مسیر `/predict` ارسال می‌کند.

    در نهایت، آماده هستید تا فایل پایتون که مصرف مدل و نمایش پیش‌بینی‌ها را هدایت می‌کند بسازید:

1. در `app.py` اضافه کنید:

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

    > 💡 نکته: وقتی [`debug=True`](https://www.askpython.com/python-modules/flask/flask-debug-mode) را هنگام اجرای اپلیکیشن وب با Flask اضافه می‌کنید، هر تغییری که در اپلیکیشن خود ایجاد کنید بلافاصله بدون نیاز به راه‌اندازی مجدد سرور منعکس می‌شود. مراقب باشید! این حالت را در اپلیکیشن تولیدی فعال نکنید.

اگر `python app.py` یا `python3 app.py` را اجرا کنید - سرور وب شما به صورت محلی راه‌اندازی می‌شود و می‌توانید یک فرم کوتاه را پر کنید تا پاسخ سوال خود درباره محل مشاهده بشقاب پرنده‌ها را دریافت کنید!

قبل از انجام این کار، به بخش‌های `app.py` نگاه کنید:

1. ابتدا وابستگی‌ها بارگذاری می‌شوند و اپلیکیشن شروع می‌شود.
1. سپس مدل وارد می‌شود.
1. سپس index.html در مسیر اصلی رندر می‌شود.

در مسیر `/predict`، چندین اتفاق رخ می‌دهد وقتی فرم ارسال می‌شود:

1. متغیرهای فرم جمع‌آوری شده و به یک آرایه numpy تبدیل می‌شوند. سپس به مدل ارسال می‌شوند و یک پیش‌بینی بازگردانده می‌شود.
2. کشورهایی که می‌خواهیم نمایش داده شوند به متن قابل خواندن از کد کشور پیش‌بینی‌شده تبدیل می‌شوند و آن مقدار به index.html ارسال می‌شود تا در قالب رندر شود.

استفاده از مدل به این روش، با Flask و یک مدل pickled، نسبتاً ساده است. سخت‌ترین چیز این است که بفهمید داده‌هایی که باید به مدل ارسال شوند تا یک پیش‌بینی دریافت شود چه شکلی دارند. این کاملاً به نحوه آموزش مدل بستگی دارد. این مدل سه نقطه داده برای ورودی نیاز دارد تا یک پیش‌بینی ارائه دهد.

در یک محیط حرفه‌ای، می‌توانید ببینید که ارتباط خوب بین افرادی که مدل را آموزش می‌دهند و کسانی که آن را در اپلیکیشن وب یا موبایل مصرف می‌کنند چقدر ضروری است. در مورد ما، فقط یک نفر هستید، شما!

---

## 🚀 چالش

به جای کار در یک نوت‌بوک و وارد کردن مدل به اپلیکیشن Flask، می‌توانید مدل را مستقیماً در اپلیکیشن Flask آموزش دهید! سعی کنید کد پایتون خود را در نوت‌بوک تبدیل کنید، شاید پس از پاکسازی داده‌ها، تا مدل را از داخل اپلیکیشن در یک مسیر به نام `train` آموزش دهید. مزایا و معایب دنبال کردن این روش چیست؟

## [پس‌ آزمون](https://ff-quizzes.netlify.app/en/ml/)

## مرور و مطالعه شخصی

راه‌های زیادی برای ساخت اپلیکیشن وب برای مصرف مدل‌های یادگیری ماشین وجود دارد. لیستی از روش‌هایی که می‌توانید با استفاده از جاوااسکریپت یا پایتون اپلیکیشن وب بسازید تا یادگیری ماشین را به کار ببرید تهیه کنید. معماری را در نظر بگیرید: آیا مدل باید در اپلیکیشن باقی بماند یا در فضای ابری قرار گیرد؟ اگر گزینه دوم، چگونه به آن دسترسی پیدا می‌کنید؟ یک مدل معماری برای یک راه‌حل وب یادگیری ماشین طراحی کنید.

## تکلیف

[یک مدل متفاوت را امتحان کنید](assignment.md)

---

**سلب مسئولیت**:  
این سند با استفاده از سرویس ترجمه هوش مصنوعی [Co-op Translator](https://github.com/Azure/co-op-translator) ترجمه شده است. در حالی که ما تلاش می‌کنیم دقت را حفظ کنیم، لطفاً توجه داشته باشید که ترجمه‌های خودکار ممکن است شامل خطاها یا نادرستی‌ها باشند. سند اصلی به زبان اصلی آن باید به عنوان منبع معتبر در نظر گرفته شود. برای اطلاعات حساس، توصیه می‌شود از ترجمه حرفه‌ای انسانی استفاده کنید. ما مسئولیتی در قبال سوءتفاهم‌ها یا تفسیرهای نادرست ناشی از استفاده از این ترجمه نداریم.