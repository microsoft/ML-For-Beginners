<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "e0b75f73e4a90d45181dc5581fe2ef5c",
  "translation_date": "2025-09-05T13:00:15+00:00",
  "source_file": "3-Web-App/1-Web-App/README.md",
  "language_code": "my"
}
-->
# ML မော်ဒယ်ကို အသုံးပြုရန် Web App တည်ဆောက်ခြင်း

ဒီသင်ခန်းစာမှာ သင်သည် _နောက်ဆုံးရာစုအတွင်း UFO တွေ့ရှိမှုများ_ ဆိုတဲ့ အထူးအဆန်းသော ဒေတာစနစ်ပေါ်မှာ ML မော်ဒယ်ကို လေ့ကျင့်ပါမည်။ ဒေတာများကို NUFORC ရဲ့ ဒေတာဘေ့စ်မှ ရယူထားသည်။

သင်လေ့လာရမည့်အရာများမှာ:

- လေ့ကျင့်ပြီးသော မော်ဒယ်ကို 'pickle' လုပ်နည်း
- Flask app မှာ မော်ဒယ်ကို အသုံးပြုနည်း

ကျွန်တော်တို့သည် ဒေတာကို သန့်စင်ခြင်းနှင့် မော်ဒယ်ကို လေ့ကျင့်ခြင်းအတွက် notebook များကို ဆက်လက်အသုံးပြုမည်ဖြစ်ပြီး၊ သင်သည် မော်ဒယ်ကို 'အပြင်မှာ' အသုံးပြုခြင်းကို စူးစမ်းခြင်းဖြင့် တစ်ဆင့်အဆင့်တက်နိုင်ပါသည်။ ဒါကိုလုပ်ဖို့ Flask ကို အသုံးပြုပြီး web app တစ်ခုကို တည်ဆောက်ရပါမည်။

## [Pre-lecture quiz](https://ff-quizzes.netlify.app/en/ml/)

## App တစ်ခုတည်ဆောက်ခြင်း

Machine learning မော်ဒယ်များကို အသုံးပြုရန် web app များတည်ဆောက်ရန် နည်းလမ်းများစွာရှိသည်။ သင့် web architecture သည် မော်ဒယ်ကို လေ့ကျင့်ပုံကို သက်ရောက်စေနိုင်သည်။ သင်သည် data science group မှ လေ့ကျင့်ထားသော မော်ဒယ်ကို app မှာ အသုံးပြုရန်လိုအပ်သော စီးပွားရေးလုပ်ငန်းတွင် အလုပ်လုပ်နေသည်ဟု စဉ်းစားပါ။

### စဉ်းစားရန်အချက်များ

သင်မေးရမည့်မေးခွန်းများစွာရှိသည်:

- **Web app လား mobile app လား?** သင် mobile app တစ်ခုတည်ဆောက်နေသည်၊ ဒါမှမဟုတ် IoT context မှာ မော်ဒယ်ကို အသုံးပြုရန်လိုအပ်ပါက [TensorFlow Lite](https://www.tensorflow.org/lite/) ကို အသုံးပြုပြီး Android သို့မဟုတ် iOS app မှာ မော်ဒယ်ကို အသုံးပြုနိုင်ပါသည်။
- **မော်ဒယ်ကို ဘယ်မှာထားမလဲ?** Cloud မှာလား ဒေသတွင်းမှာလား?
- **Offline support.** App သည် offline မှာ အလုပ်လုပ်ရမလား?
- **မော်ဒယ်ကို လေ့ကျင့်ရန် ဘယ်နည်းပညာကို အသုံးပြုခဲ့သလဲ?** ရွေးချယ်ထားသော နည်းပညာသည် သင်အသုံးပြုရမည့် tooling ကို သက်ရောက်စေပါမည်။
    - **TensorFlow ကို အသုံးပြုခြင်း။** TensorFlow ကို အသုံးပြု၍ မော်ဒယ်ကို လေ့ကျင့်နေပါက [TensorFlow.js](https://www.tensorflow.org/js/) ကို အသုံးပြု၍ web app မှာ အသုံးပြုရန် TensorFlow မော်ဒယ်ကို ပြောင်းနိုင်စွမ်းပေးသည်။
    - **PyTorch ကို အသုံးပြုခြင်း။** [PyTorch](https://pytorch.org/) ကဲ့သို့သော library ကို အသုံးပြု၍ မော်ဒယ်တစ်ခုကို တည်ဆောက်နေပါက [ONNX](https://onnx.ai/) (Open Neural Network Exchange) format ကို JavaScript web app များတွင် အသုံးပြုနိုင်ရန် [Onnx Runtime](https://www.onnxruntime.ai/) ကို အသုံးပြုနိုင်သည်။ ဒီ option ကို Scikit-learn-trained မော်ဒယ်အတွက် နောက်ဆုံးသင်ခန်းစာမှာ စူးစမ်းပါမည်။
    - **Lobe.ai သို့မဟုတ် Azure Custom Vision ကို အသုံးပြုခြင်း။** [Lobe.ai](https://lobe.ai/) သို့မဟုတ် [Azure Custom Vision](https://azure.microsoft.com/services/cognitive-services/custom-vision-service/?WT.mc_id=academic-77952-leestott) ကဲ့သို့သော ML SaaS (Software as a Service) စနစ်ကို အသုံးပြု၍ မော်ဒယ်ကို လေ့ကျင့်နေပါက၊ ဒီ software များသည် မော်ဒယ်ကို အမျိုးမျိုးသော platform များအတွက် export လုပ်ရန် နည်းလမ်းများပေးသည်။ Cloud မှာ query လုပ်နိုင်သော bespoke API တစ်ခုတည်ဆောက်ရန်လည်း ရနိုင်သည်။

သင်သည် web browser မှာ မော်ဒယ်ကို ကိုယ်တိုင်လေ့ကျင့်နိုင်သော Flask web app တစ်ခုလုံးကို တည်ဆောက်နိုင်သည်။ JavaScript context မှာ TensorFlow.js ကို အသုံးပြု၍လည်း လုပ်နိုင်ပါသည်။

ကျွန်တော်တို့ရဲ့ ရည်ရွယ်ချက်အရ၊ Python-based notebooks များကို အသုံးပြုနေသောကြောင့် notebook မှ trained မော်ဒယ်ကို Python-built web app မှာ ဖတ်နိုင်သော format သို့ export လုပ်ရန် လိုအပ်သော အဆင့်များကို စူးစမ်းကြည့်ပါမည်။

## Tool

ဒီ task အတွက် သင်သည် Flask နှင့် Pickle ဆိုသော Python မှ run လုပ်သော tool နှစ်ခုလိုအပ်ပါမည်။

✅ [Flask](https://palletsprojects.com/p/flask/) ဆိုတာဘာလဲ? Flask ကို 'micro-framework' ဟု ဖန်တီးသူများက ဖော်ပြထားပြီး Python ကို အသုံးပြု၍ web frameworks ရဲ့ အခြေခံ features များနှင့် web pages တည်ဆောက်ရန် templating engine ကို ပေးသည်။ Flask ကို အသုံးပြု၍ AI web app တစ်ခုတည်ဆောက်ရန် [ဒီ Learn module](https://docs.microsoft.com/learn/modules/python-flask-build-ai-web-app?WT.mc_id=academic-77952-leestott) ကို ကြည့်ပါ။

✅ [Pickle](https://docs.python.org/3/library/pickle.html) ဆိုတာဘာလဲ? Pickle 🥒 သည် Python module တစ်ခုဖြစ်ပြီး Python object structure ကို serialize နှင့် de-serialize လုပ်ပေးသည်။ မော်ဒယ်ကို 'pickle' လုပ်သည်ဆိုတာ မော်ဒယ်ရဲ့ structure ကို web မှာ အသုံးပြုရန် serialize သို့မဟုတ် flatten လုပ်ခြင်းဖြစ်သည်။ သတိထားပါ: pickle သည် intrinsic security မရှိသောကြောင့် 'un-pickle' လုပ်ရန် prompt လုပ်ပါက သတိထားပါ။ Pickled file တွင် `.pkl` suffix ပါသည်။

## လေ့ကျင့်ခန်း - သင့်ဒေတာကို သန့်စင်ပါ

ဒီသင်ခန်းစာမှာ သင်သည် [NUFORC](https://nuforc.org) (The National UFO Reporting Center) မှ စုဆောင်းထားသော 80,000 UFO တွေ့ရှိမှုများ၏ ဒေတာကို အသုံးပြုပါမည်။ ဒီဒေတာမှာ UFO တွေ့ရှိမှုများ၏ စိတ်ဝင်စားဖွယ်ဖော်ပြချက်များပါဝင်သည်၊ ဥပမာ:

- **ရှည်လျားသော ဖော်ပြချက်။** "အလင်းတန်းတစ်ခုက ညဘက် မြက်ခင်းပေါ်ကို ထွန်းလင်းနေပြီး အလင်းတန်းထဲက လူတစ်ယောက် ထွက်လာပြီး Texas Instruments parking lot ကို ပြေးသွားသည်။"
- **တိုတောင်းသော ဖော်ပြချက်။** "အလင်းတွေက ကျွန်တော်တို့ကို လိုက်ခဲ့တယ်။"

[ufos.csv](../../../../3-Web-App/1-Web-App/data/ufos.csv) spreadsheet တွင် `city`, `state`, `country` (တွေ့ရှိမှုဖြစ်ပွားသောနေရာ), object's `shape`, `latitude` နှင့် `longitude` column များပါဝင်သည်။

ဒီသင်ခန်းစာတွင် ပါဝင်သော [notebook](../../../../3-Web-App/1-Web-App/notebook.ipynb) မှာ:

1. `pandas`, `matplotlib`, နှင့် `numpy` ကို ယခင်သင်ခန်းစာများတွင်လုပ်ခဲ့သလို import လုပ်ပြီး ufos spreadsheet ကို import လုပ်ပါ။ ဒေတာစနစ်တစ်ခုကို ကြည့်နိုင်သည်:

    ```python
    import pandas as pd
    import numpy as np
    
    ufos = pd.read_csv('./data/ufos.csv')
    ufos.head()
    ```

1. ufos ဒေတာကို fresh titles ပါသော dataframe သေးငယ်တစ်ခုသို့ ပြောင်းပါ။ `Country` field ရဲ့ unique values များကို စစ်ဆေးပါ။

    ```python
    ufos = pd.DataFrame({'Seconds': ufos['duration (seconds)'], 'Country': ufos['country'],'Latitude': ufos['latitude'],'Longitude': ufos['longitude']})
    
    ufos.Country.unique()
    ```

1. အခုတော့ null values များကို drop လုပ်ပြီး 1-60 seconds ကြားမှာ ဖြစ်ပွားသော တွေ့ရှိမှုများကိုသာ import လုပ်ခြင်းဖြင့် ကျွန်တော်တို့ကိုလိုအပ်သော ဒေတာပမာဏကို လျှော့ချနိုင်ပါသည်:

    ```python
    ufos.dropna(inplace=True)
    
    ufos = ufos[(ufos['Seconds'] >= 1) & (ufos['Seconds'] <= 60)]
    
    ufos.info()
    ```

1. Scikit-learn ရဲ့ `LabelEncoder` library ကို import လုပ်ပြီး text values များကို number သို့ ပြောင်းပါ:

    ✅ LabelEncoder သည် ဒေတာကို အက္ခရာစဉ်အတိုင်း encode လုပ်သည်

    ```python
    from sklearn.preprocessing import LabelEncoder
    
    ufos['Country'] = LabelEncoder().fit_transform(ufos['Country'])
    
    ufos.head()
    ```

    သင့်ဒေတာသည် ဒီလိုပုံစံရှိသင့်သည်:

    ```output
    	Seconds	Country	Latitude	Longitude
    2	20.0	3		53.200000	-2.916667
    3	20.0	4		28.978333	-96.645833
    14	30.0	4		35.823889	-80.253611
    23	60.0	4		45.582778	-122.352222
    24	3.0		3		51.783333	-0.783333
    ```

## လေ့ကျင့်ခန်း - မော်ဒယ်ကို တည်ဆောက်ပါ

အခုတော့ ဒေတာကို လေ့ကျင့်ခြင်းနှင့် စမ်းသပ်ခြင်းအုပ်စုသို့ ခွဲခြားပြီး မော်ဒယ်ကို လေ့ကျင့်ရန် ပြင်ဆင်နိုင်ပါပြီ။

1. သင်လေ့ကျင့်လိုသော feature သုံးခုကို X vector အဖြစ် ရွေးချယ်ပါ၊ y vector သည် `Country` ဖြစ်ပါမည်။ သင်သည် `Seconds`, `Latitude` နှင့် `Longitude` ကို input လုပ်ပြီး country id ကို return လုပ်နိုင်ရန်လိုအပ်သည်။

    ```python
    from sklearn.model_selection import train_test_split
    
    Selected_features = ['Seconds','Latitude','Longitude']
    
    X = ufos[Selected_features]
    y = ufos['Country']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    ```

1. Logistic regression ကို အသုံးပြု၍ မော်ဒယ်ကို လေ့ကျင့်ပါ:

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

Accuracy သည် **(95% ခန့်)** မဆိုးပါဘူး၊ အံ့မခန်းပဲ၊ `Country` နှင့် `Latitude/Longitude` သည် correlation ရှိသောကြောင့်ဖြစ်သည်။

သင်ဖန်တီးထားသော မော်ဒယ်သည် `Latitude` နှင့် `Longitude` မှ `Country` ကို အတိအကျသုံးသပ်နိုင်သည့်အတွက် အလွန်ထူးခြားသော မော်ဒယ်မဟုတ်ပါ၊ ဒါပေမယ့် crude ဒေတာကို သန့်စင်ပြီး export လုပ်ကာ web app မှာ မော်ဒယ်ကို အသုံးပြုရန် လေ့ကျင့်ခြင်းကို လေ့ကျင့်ရန်ကောင်းသော လေ့ကျင့်ခန်းဖြစ်သည်။

## လေ့ကျင့်ခန်း - မော်ဒယ်ကို 'pickle' လုပ်ပါ

အခုတော့ သင့်မော်ဒယ်ကို _pickle_ လုပ်ရန် အချိန်ရောက်ပါပြီ! ```python
import pickle
model_filename = 'ufo-model.pkl'
pickle.dump(model, open(model_filename,'wb'))

model = pickle.load(open('ufo-model.pkl','rb'))
print(model.predict([[50,44,-12]]))
```

မော်ဒယ်သည် **'3'** ကို return လုပ်သည်၊ UK ရဲ့ country code ဖြစ်သည်။ အံ့မခန်း! 👽

## လေ့ကျင့်ခန်း - Flask app တစ်ခုတည်ဆောက်ပါ

အခုတော့ သင့်မော်ဒယ်ကို ခေါ်ပြီး ရလဒ်များကို ပိုမိုကြည့်လို့ကောင်းသောပုံစံဖြင့် ပြသနိုင်သော Flask app တစ်ခုကို တည်ဆောက်နိုင်ပါပြီ။

1. **web-app** folder တစ်ခုကို _notebook.ipynb_ ဖိုင်နှင့် _ufo-model.pkl_ ဖိုင်ရှိရာနေရာအနီးတွင် ဖန်တီးပါ။

1. အဲဒီ folder မှာ **static** folder တစ်ခု (အတွင်းမှာ **css** folder ပါ) နှင့် **templates** folder တစ်ခု ဖန်တီးပါ။ သင့်တွင် အောက်ပါဖိုင်များနှင့် directories ရှိသင့်သည်:

    ```output
    web-app/
      static/
        css/
      templates/
    notebook.ipynb
    ufo-model.pkl
    ```

    ✅ အပြီးသတ် app ရဲ့ solution folder ကို ကြည့်ပါ

1. _web-app_ folder မှာ ဖန်တီးရမည့် ပထမဆုံးဖိုင်မှာ **requirements.txt** ဖြစ်သည်။ JavaScript app ရဲ့ _package.json_ ကဲ့သို့ requirements.txt ဖိုင်သည် app မှာလိုအပ်သော dependencies များကို ဖော်ပြသည်။ **requirements.txt** မှာ အောက်ပါလိုင်းများထည့်ပါ:

    ```text
    scikit-learn
    pandas
    numpy
    flask
    ```

1. အခုတော့ _web-app_ folder ကို navigate လုပ်ပြီး ဒီဖိုင်ကို run လုပ်ပါ:

    ```bash
    cd web-app
    ```

1. Terminal မှာ `pip install` ကို ရိုက်ပါ၊ _requirements.txt_ မှာ ဖော်ပြထားသော libraries များကို install လုပ်ရန်:

    ```bash
    pip install -r requirements.txt
    ```

1. အခုတော့ app ကို အပြီးသတ်ရန် ဖိုင်သုံးခုကို ဖန်တီးရန် ပြင်ဆင်ထားပါ:

    1. **app.py** ကို root မှာ ဖန်တီးပါ။
    2. **index.html** ကို _templates_ directory မှာ ဖန်တီးပါ။
    3. **styles.css** ကို _static/css_ directory မှာ ဖန်တီးပါ။

1. _styles.css_ ဖိုင်ကို အနည်းငယ် styles ဖြင့် တည်ဆောက်ပါ:

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

1. နောက်ဆုံး _index.html_ ဖိုင်ကို တည်ဆောက်ပါ:

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

    ဒီဖိုင်မှာ templating ကို ကြည့်ပါ။ `{{}}` ကဲ့သို့ app မှပေးမည့် variables များကို 'mustache' syntax ဖြင့် ဖော်ပြထားသည်။ `/predict` route မှာ prediction ကို post လုပ်မည့် form တစ်ခုလည်း ပါဝင်သည်။

    နောက်ဆုံးတော့ မော်ဒယ်ကို အသုံးပြုခြင်းနှင့် prediction များကို ပြသရန် Python ဖိုင်ကို တည်ဆောက်ရန် ပြင်ဆင်ထားပါ:

1. `app.py` မှာ အောက်ပါအတိုင်း ထည့်ပါ:

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

    > 💡 Tip: Flask ကို အသုံးပြု၍ web app ကို run လုပ်နေစဉ် `debug=True` ကို ထည့်ပါက app မှာ ပြောင်းလဲမှုများကို server ကို restart မလုပ်ဘဲ ချက်ချင်းပြောင်းလဲနိုင်သည်။ သတိထားပါ! ဒီ mode ကို production app မှာ enable မလုပ်ပါနှင့်။

`python app.py` သို့မဟုတ် `python3 app.py` ကို run လုပ်ပါ - သင့် web server သည် locally မှာ စတင်ပြီး သင့်ရဲ့ UFO တွေ့ရှိမှုများအကြောင်း burning question ကို ဖြေရှင်းရန် short form တစ်ခုကို ဖြည့်နိုင်ပါသည်!

ဒီလုပ်ဆောင်မှုကိုလုပ်မည်မပြုမီ `app.py` ရဲ့ အပိုင်းများကို ကြည့်ပါ:

1. ပထမဆုံး dependencies များကို load လုပ်ပြီး app ကို စတင်ပါသည်။
1. နောက်ဆုံး မော်ဒယ်ကို import လုပ်ပါသည်။
1. index.html ကို home route မှာ render လုပ်ပါသည်။

`/predict` route မှာ form ကို post လုပ်သောအခါ အချက်အလက်များစွာဖြစ်ပျက်သည်:

1. Form variables များကို စုဆောင်းပြီး numpy array သို့ ပြောင်းပါ။ မော်ဒယ်ထံပို့ပြီး prediction ကို return လုပ်ပါသည်။
2. Countries များကို readable text အဖြစ် ပြန်လည် render လုပ်ပြီး predicted country code မှာ ပြသရန် index.html သို့ ပြန်ပို့ပါသည်။

Flask နှင့် pickled မော်ဒယ်ကို အသုံးပြု၍ မော်ဒယ်ကို ဒီလိုအသုံးပြုခြင်းသည် အလွန်ရိုးရှင်းသည်။ မော်ဒယ်ထံ prediction ရရန် ပို့ရမည့် ဒေတာရဲ့ ပုံစံကို နားလည်ရန်သာ အခက်အခဲရှိသည်။ ဒါဟာ မော်ဒယ်ကို လေ့ကျင့်ပုံပေါ်မူတည်သည်။ ဒီမော်ဒယ်မှာ prediction ရရန် input လုပ်ရမည့် data points သုံးခုရှိသည်။

ပရော်ဖက်ရှင်နယ် setting မှာ မော်ဒယ်ကို လေ့ကျင့်သူများနှင့် web သို့မဟုတ် mobile app မှာ မော်ဒယ်ကို အသုံးပြုသူများအကြား ကောင်းမွန်သော ဆက်သွယ်မှုလိုအပ်သည်ကို သင်မြင်နိုင်ပါသည်။ ကျွန်တော်တို့ရဲ့ အခြေအနေမှာတော့ တစ်ယောက်တည်းဖြစ်သည်၊ သင်ပဲ!

---

## 🚀 Challenge

Notebook မှာ အလုပ်လုပ်ပြီး မော်ဒယ်ကို Flask app သို့ import လုပ်ခြင်းအစား၊ Flask app မှာတင် မော်ဒယ်ကို လေ့ကျင့်နိုင်ပါသည်! သင့် notebook မှ Python code ကို app မှာ data ကို သန့်စင်ပြီးနောက် `train` route မှာ မော်ဒယ်ကို လေ့ကျင့်ရန် ပြောင်းပါ။ ဒီနည်းလမ်းကို လိုက်နာခြင်းရဲ့ အကျိုးကျေးဇူးများနှင့် အနုတ်လက္ခဏာများကို စဉ်းစားပါ။

## [Post-lecture quiz](https://ff-quizzes.netlify.app/en/ml/)

## Review & Self Study

ML မော်ဒယ်များကို အသုံးပြုရန် web app တစ်ခုတည်ဆောက်ရန် နည်းလမ်းများစွာရှိသည်။ JavaScript သို့မဟုတ် Python ကို အသုံးပြု၍ ML ကို leverage လုပ်ရန် web app တစ်ခုတည်ဆောက်နိုင်သော နည်းလမ်းများကို စဉ်းစားပါ။ Architecture ကို စဉ်းစားပါ: မော်ဒယ်ကို app မှာထားသင့်သလား cloud မှာထားသင့်သလား? Cloud မှာထားပါက ဘယ်လို access လုပ်မလဲ? Applied ML web solution အတွက် architectural model တစ်ခုကို ရေးဆွဲပါ။

## Assignment

[မော်ဒယ်တစ်ခုကို စမ်းကြည့်ပါ](assignment.md)

---

**အကြောင်းကြားချက်**:  
ဤစာရွက်စာတမ်းကို AI ဘာသာပြန်ဝန်ဆောင်မှု [Co-op Translator](https://github.com/Azure/co-op-translator) ကို အသုံးပြု၍ ဘာသာပြန်ထားပါသည်။ ကျွန်ုပ်တို့သည် တိကျမှုအတွက် ကြိုးစားနေသော်လည်း၊ အလိုအလျောက် ဘာသာပြန်ခြင်းတွင် အမှားများ သို့မဟုတ် မတိကျမှုများ ပါဝင်နိုင်သည်ကို သတိပြုပါ။ မူရင်းစာရွက်စာတမ်းကို ၎င်း၏ မူရင်းဘာသာစကားဖြင့် အာဏာတရားရှိသော အရင်းအမြစ်အဖြစ် သတ်မှတ်သင့်ပါသည်။ အရေးကြီးသော အချက်အလက်များအတွက် လူ့ဘာသာပြန်ပညာရှင်များမှ ပရော်ဖက်ရှင်နယ် ဘာသာပြန်ခြင်းကို အကြံပြုပါသည်။ ဤဘာသာပြန်ကို အသုံးပြုခြင်းမှ ဖြစ်ပေါ်လာသော အလွဲအလွတ်များ သို့မဟုတ် အနားလွဲမှုများအတွက် ကျွန်ုပ်တို့သည် တာဝန်မယူပါ။