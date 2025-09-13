<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "e0b75f73e4a90d45181dc5581fe2ef5c",
  "translation_date": "2025-09-05T21:46:29+00:00",
  "source_file": "3-Web-App/1-Web-App/README.md",
  "language_code": "th"
}
-->
# สร้างเว็บแอปเพื่อใช้งานโมเดล ML

ในบทเรียนนี้ คุณจะฝึกโมเดล ML บนชุดข้อมูลที่น่าสนใจมาก: _การพบเห็น UFO ในช่วงศตวรรษที่ผ่านมา_ ซึ่งมาจากฐานข้อมูลของ NUFORC

คุณจะได้เรียนรู้:

- วิธี 'pickle' โมเดลที่ฝึกแล้ว
- วิธีใช้โมเดลนั้นในแอป Flask

เราจะใช้โน้ตบุ๊กเพื่อทำความสะอาดข้อมูลและฝึกโมเดลของเรา แต่คุณสามารถนำกระบวนการนี้ไปอีกขั้นโดยการสำรวจการใช้งานโมเดลในโลกจริง เช่น ในเว็บแอป

เพื่อทำสิ่งนี้ คุณจำเป็นต้องสร้างเว็บแอปโดยใช้ Flask

## [แบบทดสอบก่อนเรียน](https://ff-quizzes.netlify.app/en/ml/)

## การสร้างแอป

มีหลายวิธีในการสร้างเว็บแอปเพื่อใช้งานโมเดล Machine Learning สถาปัตยกรรมเว็บของคุณอาจมีผลต่อวิธีการฝึกโมเดล ลองจินตนาการว่าคุณกำลังทำงานในองค์กรที่กลุ่ม Data Science ได้ฝึกโมเดลที่พวกเขาต้องการให้คุณใช้งานในแอป

### สิ่งที่ต้องพิจารณา

มีคำถามหลายข้อที่คุณต้องถาม:

- **เป็นเว็บแอปหรือแอปมือถือ?** หากคุณกำลังสร้างแอปมือถือหรือจำเป็นต้องใช้โมเดลในบริบท IoT คุณสามารถใช้ [TensorFlow Lite](https://www.tensorflow.org/lite/) และใช้โมเดลในแอป Android หรือ iOS
- **โมเดลจะอยู่ที่ไหน?** บนคลาวด์หรือในเครื่อง?
- **การรองรับแบบออฟไลน์** แอปจำเป็นต้องทำงานแบบออฟไลน์หรือไม่?
- **เทคโนโลยีที่ใช้ฝึกโมเดลคืออะไร?** เทคโนโลยีที่เลือกอาจมีผลต่อเครื่องมือที่คุณต้องใช้
    - **การใช้ TensorFlow** หากคุณฝึกโมเดลโดยใช้ TensorFlow ตัวระบบนี้มีความสามารถในการแปลงโมเดล TensorFlow เพื่อใช้งานในเว็บแอปโดยใช้ [TensorFlow.js](https://www.tensorflow.org/js/)
    - **การใช้ PyTorch** หากคุณสร้างโมเดลโดยใช้ไลบรารี เช่น [PyTorch](https://pytorch.org/) คุณมีตัวเลือกในการส่งออกโมเดลในรูปแบบ [ONNX](https://onnx.ai/) (Open Neural Network Exchange) เพื่อใช้งานในเว็บแอป JavaScript ที่สามารถใช้ [Onnx Runtime](https://www.onnxruntime.ai/) ตัวเลือกนี้จะถูกสำรวจในบทเรียนอนาคตสำหรับโมเดลที่ฝึกด้วย Scikit-learn
    - **การใช้ Lobe.ai หรือ Azure Custom Vision** หากคุณใช้ระบบ ML SaaS (Software as a Service) เช่น [Lobe.ai](https://lobe.ai/) หรือ [Azure Custom Vision](https://azure.microsoft.com/services/cognitive-services/custom-vision-service/?WT.mc_id=academic-77952-leestott) เพื่อฝึกโมเดล ซอฟต์แวร์ประเภทนี้มีวิธีการส่งออกโมเดลสำหรับหลายแพลตฟอร์ม รวมถึงการสร้าง API เฉพาะเพื่อเรียกใช้งานในคลาวด์โดยแอปออนไลน์ของคุณ

คุณยังมีโอกาสสร้างเว็บแอป Flask ทั้งหมดที่สามารถฝึกโมเดลได้เองในเว็บเบราว์เซอร์ สิ่งนี้สามารถทำได้โดยใช้ TensorFlow.js ในบริบทของ JavaScript

สำหรับวัตถุประสงค์ของเรา เนื่องจากเราได้ทำงานกับโน้ตบุ๊กที่ใช้ Python ลองสำรวจขั้นตอนที่คุณต้องทำเพื่อส่งออกโมเดลที่ฝึกแล้วจากโน้ตบุ๊กไปยังรูปแบบที่เว็บแอปที่สร้างด้วย Python สามารถอ่านได้

## เครื่องมือ

สำหรับงานนี้ คุณต้องใช้เครื่องมือสองอย่าง: Flask และ Pickle ซึ่งทั้งสองทำงานบน Python

✅ [Flask](https://palletsprojects.com/p/flask/) คืออะไร? Flask ถูกนิยามว่าเป็น 'micro-framework' โดยผู้สร้างของมัน Flask ให้ฟีเจอร์พื้นฐานของเว็บเฟรมเวิร์กโดยใช้ Python และเครื่องมือสร้างเทมเพลตเพื่อสร้างหน้าเว็บ ลองดู [โมดูลการเรียนรู้นี้](https://docs.microsoft.com/learn/modules/python-flask-build-ai-web-app?WT.mc_id=academic-77952-leestott) เพื่อฝึกสร้างด้วย Flask

✅ [Pickle](https://docs.python.org/3/library/pickle.html) คืออะไร? Pickle 🥒 เป็นโมดูล Python ที่ใช้ในการ serialize และ de-serialize โครงสร้างวัตถุ Python เมื่อคุณ 'pickle' โมเดล คุณจะ serialize หรือ flatten โครงสร้างของมันเพื่อใช้งานบนเว็บ ระวัง: pickle ไม่ปลอดภัยโดยธรรมชาติ ดังนั้นควรระวังหากถูกขอให้ 'un-pickle' ไฟล์ ไฟล์ที่ถูก pickle จะมีนามสกุล `.pkl`

## แบบฝึกหัด - ทำความสะอาดข้อมูลของคุณ

ในบทเรียนนี้ คุณจะใช้ข้อมูลจากการพบเห็น UFO จำนวน 80,000 ครั้ง ซึ่งรวบรวมโดย [NUFORC](https://nuforc.org) (ศูนย์รายงาน UFO แห่งชาติ) ข้อมูลนี้มีคำอธิบายที่น่าสนใจเกี่ยวกับการพบเห็น UFO เช่น:

- **คำอธิบายตัวอย่างแบบยาว** "ชายคนหนึ่งปรากฏตัวจากลำแสงที่ส่องลงบนทุ่งหญ้าในเวลากลางคืน และเขาวิ่งไปยังลานจอดรถของ Texas Instruments"
- **คำอธิบายตัวอย่างแบบสั้น** "แสงไล่ตามเรา"

สเปรดชีต [ufos.csv](../../../../3-Web-App/1-Web-App/data/ufos.csv) มีคอลัมน์เกี่ยวกับ `city`, `state` และ `country` ที่การพบเห็นเกิดขึ้น รูปร่างของวัตถุ (`shape`) และ `latitude` และ `longitude`

ใน [notebook](../../../../3-Web-App/1-Web-App/notebook.ipynb) ที่ว่างเปล่าที่รวมอยู่ในบทเรียนนี้:

1. import `pandas`, `matplotlib`, และ `numpy` เหมือนที่คุณทำในบทเรียนก่อนหน้า และ import สเปรดชีต ufos คุณสามารถดูตัวอย่างชุดข้อมูล:

    ```python
    import pandas as pd
    import numpy as np
    
    ufos = pd.read_csv('./data/ufos.csv')
    ufos.head()
    ```

1. แปลงข้อมูล ufos เป็น dataframe ขนาดเล็กพร้อมชื่อใหม่ ตรวจสอบค่าที่ไม่ซ้ำกันในฟิลด์ `Country`

    ```python
    ufos = pd.DataFrame({'Seconds': ufos['duration (seconds)'], 'Country': ufos['country'],'Latitude': ufos['latitude'],'Longitude': ufos['longitude']})
    
    ufos.Country.unique()
    ```

1. ตอนนี้ คุณสามารถลดปริมาณข้อมูลที่เราต้องจัดการโดยการลบค่าที่เป็น null และนำเข้าการพบเห็นที่มีระยะเวลา 1-60 วินาทีเท่านั้น:

    ```python
    ufos.dropna(inplace=True)
    
    ufos = ufos[(ufos['Seconds'] >= 1) & (ufos['Seconds'] <= 60)]
    
    ufos.info()
    ```

1. Import ไลบรารี `LabelEncoder` ของ Scikit-learn เพื่อแปลงค่าข้อความสำหรับประเทศให้เป็นตัวเลข:

    ✅ LabelEncoder เข้ารหัสข้อมูลตามลำดับตัวอักษร

    ```python
    from sklearn.preprocessing import LabelEncoder
    
    ufos['Country'] = LabelEncoder().fit_transform(ufos['Country'])
    
    ufos.head()
    ```

    ข้อมูลของคุณควรมีลักษณะดังนี้:

    ```output
    	Seconds	Country	Latitude	Longitude
    2	20.0	3		53.200000	-2.916667
    3	20.0	4		28.978333	-96.645833
    14	30.0	4		35.823889	-80.253611
    23	60.0	4		45.582778	-122.352222
    24	3.0		3		51.783333	-0.783333
    ```

## แบบฝึกหัด - สร้างโมเดลของคุณ

ตอนนี้คุณสามารถเตรียมพร้อมที่จะฝึกโมเดลโดยแบ่งข้อมูลออกเป็นกลุ่มการฝึกและการทดสอบ

1. เลือกสามฟีเจอร์ที่คุณต้องการฝึกเป็นเวกเตอร์ X ของคุณ และเวกเตอร์ y จะเป็น `Country` คุณต้องการสามารถป้อน `Seconds`, `Latitude` และ `Longitude` และรับรหัสประเทศกลับมา

    ```python
    from sklearn.model_selection import train_test_split
    
    Selected_features = ['Seconds','Latitude','Longitude']
    
    X = ufos[Selected_features]
    y = ufos['Country']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    ```

1. ฝึกโมเดลของคุณโดยใช้ logistic regression:

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

ความแม่นยำไม่เลวเลย **(ประมาณ 95%)** ซึ่งไม่น่าแปลกใจ เนื่องจาก `Country` และ `Latitude/Longitude` มีความสัมพันธ์กัน

โมเดลที่คุณสร้างไม่ได้ปฏิวัติอะไรมากนัก เนื่องจากคุณควรสามารถอนุมาน `Country` จาก `Latitude` และ `Longitude` ได้ แต่เป็นการฝึกที่ดีในการลองฝึกจากข้อมูลดิบที่คุณทำความสะอาด ส่งออก และใช้โมเดลนี้ในเว็บแอป

## แบบฝึกหัด - 'pickle' โมเดลของคุณ

ตอนนี้ถึงเวลาที่จะ _pickle_ โมเดลของคุณ! คุณสามารถทำได้ในไม่กี่บรรทัดโค้ด เมื่อมันถูก _pickled_ แล้ว ให้โหลดโมเดลที่ถูก pickle และทดสอบกับตัวอย่างข้อมูลอาร์เรย์ที่มีค่าของ seconds, latitude และ longitude

```python
import pickle
model_filename = 'ufo-model.pkl'
pickle.dump(model, open(model_filename,'wb'))

model = pickle.load(open('ufo-model.pkl','rb'))
print(model.predict([[50,44,-12]]))
```

โมเดลคืนค่า **'3'** ซึ่งเป็นรหัสประเทศสำหรับ UK น่าทึ่งมาก! 👽

## แบบฝึกหัด - สร้างแอป Flask

ตอนนี้คุณสามารถสร้างแอป Flask เพื่อเรียกโมเดลของคุณและคืนผลลัพธ์ที่คล้ายกัน แต่ในรูปแบบที่ดูดีขึ้น

1. เริ่มต้นด้วยการสร้างโฟลเดอร์ชื่อ **web-app** ข้างไฟล์ _notebook.ipynb_ ที่ไฟล์ _ufo-model.pkl_ ของคุณอยู่

1. ในโฟลเดอร์นั้น สร้างอีกสามโฟลเดอร์: **static** โดยมีโฟลเดอร์ **css** อยู่ข้างใน และ **templates** คุณควรมีไฟล์และไดเรกทอรีดังนี้:

    ```output
    web-app/
      static/
        css/
      templates/
    notebook.ipynb
    ufo-model.pkl
    ```

    ✅ ดูโฟลเดอร์ solution เพื่อดูแอปที่เสร็จสมบูรณ์

1. ไฟล์แรกที่สร้างในโฟลเดอร์ _web-app_ คือไฟล์ **requirements.txt** เหมือน _package.json_ ในแอป JavaScript ไฟล์นี้แสดงรายการ dependencies ที่แอปต้องการ ใน **requirements.txt** เพิ่มบรรทัด:

    ```text
    scikit-learn
    pandas
    numpy
    flask
    ```

1. ตอนนี้ รันไฟล์นี้โดยไปที่ _web-app_:

    ```bash
    cd web-app
    ```

1. ใน terminal ของคุณ พิมพ์ `pip install` เพื่อ install ไลบรารีที่ระบุใน _requirements.txt_:

    ```bash
    pip install -r requirements.txt
    ```

1. ตอนนี้ คุณพร้อมที่จะสร้างไฟล์อีกสามไฟล์เพื่อจบแอป:

    1. สร้าง **app.py** ใน root
    2. สร้าง **index.html** ในไดเรกทอรี _templates_
    3. สร้าง **styles.css** ในไดเรกทอรี _static/css_

1. สร้างไฟล์ _styles.css_ ด้วยสไตล์เล็กน้อย:

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

1. ต่อไป สร้างไฟล์ _index.html_:

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

    ลองดูการสร้างเทมเพลตในไฟล์นี้ สังเกตไวยากรณ์ 'mustache' รอบตัวแปรที่จะถูกส่งโดยแอป เช่น ข้อความการทำนาย: `{{}}` นอกจากนี้ยังมีฟอร์มที่โพสต์การทำนายไปยัง route `/predict`

    สุดท้าย คุณพร้อมที่จะสร้างไฟล์ python ที่ขับเคลื่อนการใช้งานโมเดลและการแสดงผลการทำนาย:

1. ใน `app.py` เพิ่ม:

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

    > 💡 เคล็ดลับ: เมื่อคุณเพิ่ม [`debug=True`](https://www.askpython.com/python-modules/flask/flask-debug-mode) ขณะรันเว็บแอปโดยใช้ Flask การเปลี่ยนแปลงใดๆ ที่คุณทำกับแอปพลิเคชันของคุณจะสะท้อนทันทีโดยไม่ต้องรีสตาร์ทเซิร์ฟเวอร์ ระวัง! อย่าเปิดใช้งานโหมดนี้ในแอปที่ใช้งานจริง

หากคุณรัน `python app.py` หรือ `python3 app.py` - เซิร์ฟเวอร์เว็บของคุณจะเริ่มต้นขึ้นในเครื่อง และคุณสามารถกรอกฟอร์มสั้นๆ เพื่อรับคำตอบสำหรับคำถามที่คุณสงสัยเกี่ยวกับสถานที่ที่มีการพบเห็น UFO!

ก่อนทำสิ่งนั้น ลองดูส่วนต่างๆ ของ `app.py`:

1. ขั้นแรก dependencies ถูกโหลดและแอปเริ่มต้น
1. จากนั้น โมเดลถูก import
1. จากนั้น index.html ถูก render บน home route

บน route `/predict` มีหลายสิ่งเกิดขึ้นเมื่อฟอร์มถูกโพสต์:

1. ตัวแปรในฟอร์มถูกรวบรวมและแปลงเป็น numpy array จากนั้นส่งไปยังโมเดลและผลการทำนายจะถูกคืนค่า
2. ประเทศที่เราต้องการแสดงผลถูก render ใหม่เป็นข้อความที่อ่านได้จากรหัสประเทศที่ถูกทำนาย และค่านั้นถูกส่งกลับไปยัง index.html เพื่อ render ในเทมเพลต

การใช้โมเดลในลักษณะนี้ ด้วย Flask และโมเดลที่ถูก pickle นั้นค่อนข้างตรงไปตรงมา สิ่งที่ยากที่สุดคือการเข้าใจว่าข้อมูลที่ต้องส่งไปยังโมเดลเพื่อให้ได้ผลการทำนายมีรูปแบบอย่างไร ทั้งหมดนี้ขึ้นอยู่กับวิธีที่โมเดลถูกฝึก โมเดลนี้มีจุดข้อมูลสามจุดที่ต้องป้อนเพื่อให้ได้ผลการทำนาย

ในสภาพแวดล้อมการทำงาน คุณจะเห็นว่าการสื่อสารที่ดีมีความจำเป็นระหว่างผู้ที่ฝึกโมเดลและผู้ที่ใช้งานโมเดลในเว็บหรือแอปมือถือ ในกรณีของเรา มีเพียงคนเดียว นั่นคือคุณ!

---

## 🚀 ความท้าทาย

แทนที่จะทำงานในโน้ตบุ๊กและ import โมเดลไปยังแอป Flask คุณสามารถฝึกโมเดลได้ภายในแอป Flask! ลองแปลงโค้ด Python ของคุณในโน้ตบุ๊ก อาจหลังจากที่คุณทำความสะอาดข้อมูล เพื่อฝึกโมเดลจากภายในแอปบน route ที่เรียกว่า `train` ข้อดีและข้อเสียของการใช้วิธีนี้คืออะไร?

## [แบบทดสอบหลังเรียน](https://ff-quizzes.netlify.app/en/ml/)

## ทบทวนและศึกษาด้วยตนเอง

มีหลายวิธีในการสร้างเว็บแอปเพื่อใช้งานโมเดล ML ลองทำรายการวิธีที่คุณสามารถใช้ JavaScript หรือ Python เพื่อสร้างเว็บแอปที่ใช้ Machine Learning พิจารณาสถาปัตยกรรม: โมเดลควรอยู่ในแอปหรืออยู่ในคลาวด์? หากเป็นแบบหลัง คุณจะเข้าถึงมันได้อย่างไร? ลองวาดโมเดลสถาปัตยกรรมสำหรับโซลูชัน ML เว็บที่ใช้งานจริง

## งานที่ได้รับมอบหมาย

[ลองใช้โมเดลอื่น](assignment.md)

---

**ข้อจำกัดความรับผิดชอบ**:  
เอกสารนี้ได้รับการแปลโดยใช้บริการแปลภาษา AI [Co-op Translator](https://github.com/Azure/co-op-translator) แม้ว่าเราจะพยายามให้การแปลมีความถูกต้อง แต่โปรดทราบว่าการแปลอัตโนมัติอาจมีข้อผิดพลาดหรือความไม่แม่นยำ เอกสารต้นฉบับในภาษาดั้งเดิมควรถือเป็นแหล่งข้อมูลที่เชื่อถือได้ สำหรับข้อมูลที่สำคัญ แนะนำให้ใช้บริการแปลภาษาจากผู้เชี่ยวชาญ เราไม่รับผิดชอบต่อความเข้าใจผิดหรือการตีความที่ผิดพลาดซึ่งเกิดจากการใช้การแปลนี้