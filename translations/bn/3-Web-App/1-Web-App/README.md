<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "e0b75f73e4a90d45181dc5581fe2ef5c",
  "translation_date": "2025-09-04T21:09:27+00:00",
  "source_file": "3-Web-App/1-Web-App/README.md",
  "language_code": "bn"
}
-->
# একটি মেশিন লার্নিং মডেল ব্যবহার করার জন্য একটি ওয়েব অ্যাপ তৈরি করুন

এই পাঠে, আপনি একটি ডেটাসেটে একটি মেশিন লার্নিং মডেল প্রশিক্ষণ দেবেন যা একেবারে অনন্য: _গত শতাব্দীর UFO দর্শন_, যা NUFORC-এর ডেটাবেস থেকে সংগ্রহ করা হয়েছে।

আপনি শিখবেন:

- কীভাবে একটি প্রশিক্ষিত মডেল 'পিকল' করতে হয়
- কীভাবে সেই মডেলটি একটি Flask অ্যাপে ব্যবহার করতে হয়

আমরা ডেটা পরিষ্কার এবং আমাদের মডেল প্রশিক্ষণ দেওয়ার জন্য নোটবুক ব্যবহার চালিয়ে যাব, তবে আপনি প্রক্রিয়াটি আরও এক ধাপ এগিয়ে নিয়ে যেতে পারেন একটি মডেলকে বাস্তব জীবনে ব্যবহার করার মাধ্যমে: একটি ওয়েব অ্যাপে।

এটি করতে, আপনাকে Flask ব্যবহার করে একটি ওয়েব অ্যাপ তৈরি করতে হবে।

## [পাঠের আগে কুইজ](https://ff-quizzes.netlify.app/en/ml/)

## একটি অ্যাপ তৈরি করা

মেশিন লার্নিং মডেল ব্যবহার করার জন্য ওয়েব অ্যাপ তৈরি করার বিভিন্ন উপায় রয়েছে। আপনার ওয়েব আর্কিটেকচার আপনার মডেল প্রশিক্ষণের পদ্ধতিকে প্রভাবিত করতে পারে। কল্পনা করুন যে আপনি একটি ব্যবসায় কাজ করছেন যেখানে ডেটা সায়েন্স দল একটি মডেল প্রশিক্ষণ দিয়েছে যা তারা চায় আপনি একটি অ্যাপে ব্যবহার করুন।

### বিবেচ্য বিষয়

অনেক প্রশ্ন আপনাকে করতে হবে:

- **এটি কি একটি ওয়েব অ্যাপ নাকি একটি মোবাইল অ্যাপ?** যদি আপনি একটি মোবাইল অ্যাপ তৈরি করছেন বা IoT প্রসঙ্গে মডেলটি ব্যবহার করতে চান, আপনি [TensorFlow Lite](https://www.tensorflow.org/lite/) ব্যবহার করতে পারেন এবং মডেলটি একটি Android বা iOS অ্যাপে ব্যবহার করতে পারেন।
- **মডেলটি কোথায় থাকবে?** ক্লাউডে নাকি লোকালিতে?
- **অফলাইন সাপোর্ট।** অ্যাপটি কি অফলাইনে কাজ করতে হবে?
- **মডেল প্রশিক্ষণের জন্য কোন প্রযুক্তি ব্যবহার করা হয়েছে?** নির্বাচিত প্রযুক্তি আপনার ব্যবহৃত টুলিংকে প্রভাবিত করতে পারে।
    - **TensorFlow ব্যবহার করা।** উদাহরণস্বরূপ, যদি আপনি TensorFlow ব্যবহার করে একটি মডেল প্রশিক্ষণ দেন, সেই ইকোসিস্টেম [TensorFlow.js](https://www.tensorflow.org/js/) ব্যবহার করে একটি ওয়েব অ্যাপে মডেলটি ব্যবহার করার জন্য এটি রূপান্তর করার ক্ষমতা প্রদান করে।
    - **PyTorch ব্যবহার করা।** যদি আপনি [PyTorch](https://pytorch.org/) এর মতো একটি লাইব্রেরি ব্যবহার করে একটি মডেল তৈরি করেন, আপনি এটি [ONNX](https://onnx.ai/) (Open Neural Network Exchange) ফরম্যাটে রপ্তানি করার বিকল্প পাবেন যা [Onnx Runtime](https://www.onnxruntime.ai/) ব্যবহার করে জাভাস্ক্রিপ্ট ওয়েব অ্যাপে ব্যবহার করা যায়। এই বিকল্পটি ভবিষ্যতের পাঠে Scikit-learn-প্রশিক্ষিত মডেলের জন্য অন্বেষণ করা হবে।
    - **Lobe.ai বা Azure Custom Vision ব্যবহার করা।** যদি আপনি [Lobe.ai](https://lobe.ai/) বা [Azure Custom Vision](https://azure.microsoft.com/services/cognitive-services/custom-vision-service/?WT.mc_id=academic-77952-leestott) এর মতো একটি ML SaaS (Software as a Service) সিস্টেম ব্যবহার করে একটি মডেল প্রশিক্ষণ দেন, এই ধরনের সফটওয়্যার অনেক প্ল্যাটফর্মের জন্য মডেল রপ্তানি করার উপায় প্রদান করে, যার মধ্যে রয়েছে একটি কাস্টম API তৈরি করা যা আপনার অনলাইন অ্যাপ্লিকেশন দ্বারা ক্লাউডে প্রশ্ন করা যেতে পারে।

আপনার কাছে একটি সম্পূর্ণ Flask ওয়েব অ্যাপ তৈরি করার সুযোগও রয়েছে যা একটি ওয়েব ব্রাউজারে নিজেই মডেলটি প্রশিক্ষণ দিতে সক্ষম হবে। এটি TensorFlow.js ব্যবহার করে একটি জাভাস্ক্রিপ্ট প্রসঙ্গে করা যেতে পারে।

আমাদের উদ্দেশ্যে, যেহেতু আমরা Python-ভিত্তিক নোটবুক নিয়ে কাজ করছি, আসুন আমরা সেই পদক্ষেপগুলি অন্বেষণ করি যা আপনাকে একটি প্রশিক্ষিত মডেলকে এমন একটি ফরম্যাটে রপ্তানি করতে হবে যা একটি Python-নির্মিত ওয়েব অ্যাপ দ্বারা পড়া যায়।

## টুল

এই কাজের জন্য, আপনার দুটি টুল দরকার: Flask এবং Pickle, উভয়ই Python-এ চলে।

✅ [Flask](https://palletsprojects.com/p/flask/) কী? এর নির্মাতারা এটিকে 'মাইক্রো-ফ্রেমওয়ার্ক' হিসাবে সংজ্ঞায়িত করেছেন। Flask Python ব্যবহার করে ওয়েব ফ্রেমওয়ার্কের মৌলিক বৈশিষ্ট্য এবং ওয়েব পেজ তৈরি করার জন্য একটি টেমপ্লেটিং ইঞ্জিন প্রদান করে। Flask দিয়ে তৈরি করার অনুশীলন করতে [এই Learn module](https://docs.microsoft.com/learn/modules/python-flask-build-ai-web-app?WT.mc_id=academic-77952-leestott) দেখুন।

✅ [Pickle](https://docs.python.org/3/library/pickle.html) কী? Pickle 🥒 একটি Python মডিউল যা একটি Python অবজেক্ট স্ট্রাকচারকে সিরিয়ালাইজ এবং ডি-সিরিয়ালাইজ করে। যখন আপনি একটি মডেল 'পিকল' করেন, আপনি এর স্ট্রাকচারকে সিরিয়ালাইজ বা ফ্ল্যাটেন করেন ওয়েবে ব্যবহারের জন্য। সতর্ক থাকুন: Pickle স্বাভাবিকভাবে নিরাপদ নয়, তাই যদি আপনাকে একটি ফাইল 'আন-পিকল' করতে বলা হয় তবে সতর্ক থাকুন। একটি পিকল করা ফাইলের `.pkl` সাফিক্স থাকে।

## অনুশীলন - আপনার ডেটা পরিষ্কার করুন

এই পাঠে আপনি 80,000 UFO দর্শনের ডেটা ব্যবহার করবেন, যা [NUFORC](https://nuforc.org) (The National UFO Reporting Center) দ্বারা সংগ্রহ করা হয়েছে। এই ডেটায় UFO দর্শনের কিছু আকর্ষণীয় বর্ণনা রয়েছে, যেমন:

- **দীর্ঘ উদাহরণ বর্ণনা।** "একটি আলো রশ্মি থেকে একজন মানুষ বেরিয়ে আসে যা রাতে একটি ঘাসের মাঠে পড়ে এবং তিনি Texas Instruments পার্কিং লটের দিকে দৌড়ান।"
- **সংক্ষিপ্ত উদাহরণ বর্ণনা।** "আলো আমাদের তাড়া করেছিল।"

[ufos.csv](../../../../3-Web-App/1-Web-App/data/ufos.csv) স্প্রেডশিটে `city`, `state` এবং `country` যেখানে দর্শন ঘটেছে, বস্তুটির `shape` এবং এর `latitude` এবং `longitude` সম্পর্কে কলাম রয়েছে।

এই পাঠে অন্তর্ভুক্ত খালি [notebook](../../../../3-Web-App/1-Web-App/notebook.ipynb)-এ:

1. আগের পাঠে যেমন করেছিলেন, `pandas`, `matplotlib`, এবং `numpy` আমদানি করুন এবং ufos স্প্রেডশিট আমদানি করুন। আপনি একটি নমুনা ডেটাসেট দেখতে পারেন:

    ```python
    import pandas as pd
    import numpy as np
    
    ufos = pd.read_csv('./data/ufos.csv')
    ufos.head()
    ```

1. ufos ডেটাকে নতুন শিরোনাম সহ একটি ছোট ডেটাফ্রেমে রূপান্তর করুন। `Country` ফিল্ডে অনন্য মানগুলি পরীক্ষা করুন।

    ```python
    ufos = pd.DataFrame({'Seconds': ufos['duration (seconds)'], 'Country': ufos['country'],'Latitude': ufos['latitude'],'Longitude': ufos['longitude']})
    
    ufos.Country.unique()
    ```

1. এখন, আপনি প্রয়োজনীয় ডেটার পরিমাণ কমাতে পারেন null মানগুলি বাদ দিয়ে এবং শুধুমাত্র 1-60 সেকেন্ডের মধ্যে দর্শনগুলি আমদানি করে:

    ```python
    ufos.dropna(inplace=True)
    
    ufos = ufos[(ufos['Seconds'] >= 1) & (ufos['Seconds'] <= 60)]
    
    ufos.info()
    ```

1. Scikit-learn-এর `LabelEncoder` লাইব্রেরি আমদানি করুন যাতে দেশগুলির টেক্সট মানগুলি একটি সংখ্যায় রূপান্তর করা যায়:

    ✅ LabelEncoder ডেটাকে বর্ণানুক্রমিকভাবে এনকোড করে

    ```python
    from sklearn.preprocessing import LabelEncoder
    
    ufos['Country'] = LabelEncoder().fit_transform(ufos['Country'])
    
    ufos.head()
    ```

    আপনার ডেটা দেখতে এরকম হওয়া উচিত:

    ```output
    	Seconds	Country	Latitude	Longitude
    2	20.0	3		53.200000	-2.916667
    3	20.0	4		28.978333	-96.645833
    14	30.0	4		35.823889	-80.253611
    23	60.0	4		45.582778	-122.352222
    24	3.0		3		51.783333	-0.783333
    ```

## অনুশীলন - আপনার মডেল তৈরি করুন

এখন আপনি ডেটাকে প্রশিক্ষণ এবং পরীক্ষার গ্রুপে ভাগ করে মডেল প্রশিক্ষণের জন্য প্রস্তুত হতে পারেন।

1. আপনার X ভেক্টর হিসাবে প্রশিক্ষণ দেওয়ার জন্য তিনটি বৈশিষ্ট্য নির্বাচন করুন, এবং y ভেক্টর হবে `Country`। আপনি `Seconds`, `Latitude` এবং `Longitude` ইনপুট করতে চান এবং একটি country id ফেরত পেতে চান।

    ```python
    from sklearn.model_selection import train_test_split
    
    Selected_features = ['Seconds','Latitude','Longitude']
    
    X = ufos[Selected_features]
    y = ufos['Country']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    ```

1. আপনার মডেলটি লজিস্টিক রিগ্রেশন ব্যবহার করে প্রশিক্ষণ দিন:

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

সঠিকতা খারাপ নয় **(প্রায় 95%)**, যা আশ্চর্যজনক নয়, কারণ `Country` এবং `Latitude/Longitude` সম্পর্কিত।

আপনার তৈরি মডেলটি খুব বিপ্লবী নয় কারণ আপনি `Latitude` এবং `Longitude` থেকে একটি `Country` অনুমান করতে সক্ষম হওয়া উচিত, তবে এটি একটি ভাল অনুশীলন যা আপনাকে পরিষ্কার করা কাঁচা ডেটা থেকে প্রশিক্ষণ দেওয়া, রপ্তানি করা এবং তারপর এই মডেলটি একটি ওয়েব অ্যাপে ব্যবহার করার চেষ্টা করতে দেয়।

## অনুশীলন - আপনার মডেল 'পিকল' করুন

এখন, আপনার মডেলটি _পিকল_ করার সময়! আপনি এটি কয়েকটি কোড লাইনে করতে পারেন। একবার এটি _পিকল_ হয়ে গেলে, আপনার পিকল করা মডেলটি লোড করুন এবং সেকেন্ড, latitude এবং longitude এর মান সহ একটি নমুনা ডেটা অ্যারের বিরুদ্ধে এটি পরীক্ষা করুন,

```python
import pickle
model_filename = 'ufo-model.pkl'
pickle.dump(model, open(model_filename,'wb'))

model = pickle.load(open('ufo-model.pkl','rb'))
print(model.predict([[50,44,-12]]))
```

মডেলটি **'3'** ফেরত দেয়, যা UK-এর country code। আশ্চর্যজনক! 👽

## অনুশীলন - একটি Flask অ্যাপ তৈরি করুন

এখন আপনি একটি Flask অ্যাপ তৈরি করতে পারেন যা আপনার মডেলকে কল করে এবং অনুরূপ ফলাফল ফেরত দেয়, তবে আরও চিত্তাকর্ষকভাবে।

1. _notebook.ipynb_ ফাইলের পাশে যেখানে আপনার _ufo-model.pkl_ ফাইল রয়েছে, একটি **web-app** নামক ফোল্ডার তৈরি করুন।

1. সেই ফোল্ডারে আরও তিনটি ফোল্ডার তৈরি করুন: **static**, যার ভিতরে একটি **css** ফোল্ডার থাকবে, এবং **templates**। এখন আপনার নিম্নলিখিত ফাইল এবং ডিরেক্টরি থাকা উচিত:

    ```output
    web-app/
      static/
        css/
      templates/
    notebook.ipynb
    ufo-model.pkl
    ```

    ✅ সমাধান ফোল্ডারটি সম্পূর্ণ অ্যাপের একটি দৃশ্যের জন্য দেখুন

1. _web-app_ ফোল্ডারে তৈরি করার প্রথম ফাইলটি হল **requirements.txt** ফাইল। একটি জাভাস্ক্রিপ্ট অ্যাপে _package.json_-এর মতো, এই ফাইলটি অ্যাপের প্রয়োজনীয় নির্ভরতা তালিকাভুক্ত করে। **requirements.txt**-এ লাইনগুলি যোগ করুন:

    ```text
    scikit-learn
    pandas
    numpy
    flask
    ```

1. এখন, _web-app_ এ নেভিগেট করে এই ফাইলটি চালান:

    ```bash
    cd web-app
    ```

1. আপনার টার্মিনালে `pip install` টাইপ করুন, _requirements.txt_ এ তালিকাভুক্ত লাইব্রেরিগুলি ইনস্টল করতে:

    ```bash
    pip install -r requirements.txt
    ```

1. এখন, আপনি অ্যাপটি শেষ করতে আরও তিনটি ফাইল তৈরি করতে প্রস্তুত:

    1. **app.py** রুটে তৈরি করুন।
    2. _templates_ ডিরেক্টরিতে **index.html** তৈরি করুন।
    3. _static/css_ ডিরেক্টরিতে **styles.css** তৈরি করুন।

1. _styles.css_ ফাইলটি কয়েকটি স্টাইল দিয়ে তৈরি করুন:

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

1. পরবর্তী, _index.html_ ফাইলটি তৈরি করুন:

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

    এই ফাইলের টেমপ্লেটিংটি দেখুন। লক্ষ্য করুন যে অ্যাপ দ্বারা প্রদত্ত ভেরিয়েবলগুলির চারপাশে 'mustache' সিনট্যাক্স রয়েছে, যেমন prediction টেক্সট: `{{}}`। এখানে একটি ফর্মও রয়েছে যা `/predict` রুটে একটি prediction পোস্ট করে।

    অবশেষে, আপনি Python ফাইলটি তৈরি করতে প্রস্তুত যা মডেলটি ব্যবহার এবং prediction প্রদর্শনের জন্য চালিত করে:

1. `app.py`-এ যোগ করুন:

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

    > 💡 টিপ: যখন আপনি Flask ব্যবহার করে ওয়েব অ্যাপ চালানোর সময় [`debug=True`](https://www.askpython.com/python-modules/flask/flask-debug-mode) যোগ করেন, তখন আপনার অ্যাপ্লিকেশনে করা যেকোনো পরিবর্তন অবিলম্বে প্রতিফলিত হবে সার্ভার পুনরায় চালু করার প্রয়োজন ছাড়াই। সতর্ক থাকুন! প্রোডাকশন অ্যাপে এই মোডটি সক্রিয় করবেন না।

যদি আপনি `python app.py` বা `python3 app.py` চালান - আপনার ওয়েব সার্ভার স্থানীয়ভাবে শুরু হয়, এবং আপনি একটি ছোট ফর্ম পূরণ করতে পারেন UFO দর্শন সম্পর্কে আপনার জ্বলন্ত প্রশ্নের উত্তর পেতে!

এর আগে, `app.py`-এর অংশগুলি দেখুন:

1. প্রথমে, নির্ভরতাগুলি লোড হয় এবং অ্যাপটি শুরু হয়।
1. তারপর, মডেলটি আমদানি করা হয়।
1. তারপর, হোম রুটে index.html রেন্ডার করা হয়।

`/predict` রুটে, ফর্ম পোস্ট করার সময় কয়েকটি জিনিস ঘটে:

1. ফর্ম ভেরিয়েবলগুলি সংগ্রহ করা হয় এবং একটি numpy অ্যারে-তে রূপান্তরিত হয়। তারপর সেগুলি মডেলে পাঠানো হয় এবং একটি prediction ফেরত দেওয়া হয়।
2. আমরা প্রদর্শন করতে চাই এমন দেশগুলি তাদের পূর্বাভাসিত country code থেকে পাঠযোগ্য টেক্সটে পুনরায় রেন্ডার করা হয়, এবং সেই মানটি index.html-এ টেমপ্লেটে রেন্ডার করার জন্য ফেরত পাঠানো হয়।

Flask এবং একটি পিকল করা মডেল ব্যবহার করে একটি মডেল ব্যবহার করা তুলনামূলকভাবে সহজ। সবচেয়ে কঠিন বিষয়টি হল ডেটার আকার বোঝা যা মডেলে পাঠানো উচিত একটি prediction পেতে। এটি সম্পূর্ণরূপে নির্ভর করে মডেলটি কীভাবে প্রশিক্ষণ দেওয়া হয়েছিল। এই মডেলটিতে একটি prediction পেতে তিনটি ডেটা পয়েন্ট ইনপুট করতে হবে।

একটি পেশাদার সেটিংয়ে, আপনি দেখতে পারেন যে মডেল প্রশিক্ষণকারী এবং যারা এটি একটি ওয়েব বা মোবাইল অ্যাপে ব্যবহার করেন তাদের মধ্যে ভাল যোগাযোগ কতটা গুরুত্বপূর্ণ। আমাদের ক্ষেত্রে, এটি শুধুমাত্র একজন ব্যক্তি, আপনি!

---

## 🚀 চ্যালেঞ্জ

নোটবুকে কাজ করার এবং Flask অ্যাপে মডেল আমদানি করার পরিবর্তে, আপনি Flask অ্যাপের মধ্যেই মডেলটি প্রশিক্ষণ দিতে পারেন! আপনার নোটবুকে থাকা Python কোডটি রূপান্তর করার চেষ্টা করুন, সম্ভবত আপনার ডেটা পরিষ্কার হওয়ার পরে, অ্যাপের মধ্যে একটি `train` রুটে মডেলটি প্রশিক্ষণ দেওয়ার জন্য। এই পদ্ধতি অনুসরণ করার সুবিধা এবং অসুবিধাগুলি কী?

## [পাঠের পরে কুইজ](https://ff-quizzes.netlify.app/en/ml/)

## পর্যালোচনা এবং স্ব-অধ্যয়ন

মেশিন লার্নিং মডেল ব্যবহার করার জন্য একটি ওয়েব অ্যাপ তৈরি করার অনেক উপায় রয়েছে। আপনি জাভাস্ক্রিপ্ট বা Python ব্যবহার করে মেশিন লার্নিং লিভারেজ করার জন্য একটি ওয়েব অ্যাপ তৈরি করার উপায়গুলির একটি তালিকা তৈরি করুন। আর্কিটেকচার বিবেচনা করুন: মডেলটি কি অ্যাপে থাকা উচিত নাকি ক্লাউডে থাকা উচিত? যদি ক্লাউডে থাকে, আপনি কীভাবে এটি অ্যাক্সেস করবেন? একটি প্রয়োগকৃত ML ওয়েব সমাধানের জন্য একটি আর্কিটেকচারাল মডেল আঁকুন।

## অ্যাসাইনমেন্ট

[একটি ভিন্ন মডেল চেষ্টা করুন](assignment.md)

---

**অস্বীকৃতি**:  
এই নথিটি AI অনুবাদ পরিষেবা [Co-op Translator](https://github.com/Azure/co-op-translator) ব্যবহার করে অনুবাদ করা হয়েছে। আমরা যথাসম্ভব সঠিকতা নিশ্চিত করার চেষ্টা করি, তবে অনুগ্রহ করে মনে রাখবেন যে স্বয়ংক্রিয় অনুবাদে ত্রুটি বা অসঙ্গতি থাকতে পারে। মূল ভাষায় থাকা নথিটিকে প্রামাণিক উৎস হিসেবে বিবেচনা করা উচিত। গুরুত্বপূর্ণ তথ্যের জন্য, পেশাদার মানব অনুবাদ সুপারিশ করা হয়। এই অনুবাদ ব্যবহারের ফলে কোনো ভুল বোঝাবুঝি বা ভুল ব্যাখ্যা হলে আমরা দায়বদ্ধ থাকব না।