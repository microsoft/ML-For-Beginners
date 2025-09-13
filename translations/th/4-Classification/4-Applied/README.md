<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "61bdec27ed2da8b098cd9065405d9bb0",
  "translation_date": "2025-09-05T21:51:46+00:00",
  "source_file": "4-Classification/4-Applied/README.md",
  "language_code": "th"
}
-->
# สร้างเว็บแอปแนะนำเมนูอาหาร

ในบทเรียนนี้ คุณจะสร้างโมเดลการจำแนกประเภทโดยใช้เทคนิคที่คุณได้เรียนรู้ในบทเรียนก่อนหน้า พร้อมกับชุดข้อมูลเมนูอาหารที่น่ารับประทานซึ่งใช้ตลอดซีรีส์นี้ นอกจากนี้ คุณจะสร้างเว็บแอปเล็กๆ เพื่อใช้งานโมเดลที่บันทึกไว้ โดยใช้ Onnx Web Runtime

หนึ่งในประโยชน์ที่สำคัญของการเรียนรู้ด้วยเครื่องคือการสร้างระบบแนะนำ และวันนี้คุณสามารถเริ่มต้นในเส้นทางนั้นได้!

[![นำเสนอเว็บแอปนี้](https://img.youtube.com/vi/17wdM9AHMfg/0.jpg)](https://youtu.be/17wdM9AHMfg "Applied ML")

> 🎥 คลิกที่ภาพด้านบนเพื่อดูวิดีโอ: Jen Looper สร้างเว็บแอปโดยใช้ข้อมูลเมนูอาหารที่จำแนกไว้

## [แบบทดสอบก่อนเรียน](https://ff-quizzes.netlify.app/en/ml/)

ในบทเรียนนี้คุณจะได้เรียนรู้:

- วิธีสร้างโมเดลและบันทึกเป็นโมเดล Onnx
- วิธีใช้ Netron เพื่อตรวจสอบโมเดล
- วิธีใช้โมเดลในเว็บแอปเพื่อการคาดการณ์

## สร้างโมเดลของคุณ

การสร้างระบบ ML ที่ใช้งานได้จริงเป็นส่วนสำคัญในการนำเทคโนโลยีเหล่านี้มาใช้ในระบบธุรกิจของคุณ คุณสามารถใช้โมเดลในแอปพลิเคชันเว็บของคุณ (และใช้งานในบริบทออฟไลน์หากจำเป็น) โดยใช้ Onnx

ใน [บทเรียนก่อนหน้า](../../3-Web-App/1-Web-App/README.md) คุณได้สร้างโมเดล Regression เกี่ยวกับการพบเห็น UFO และ "pickled" โมเดลนั้นเพื่อใช้งานในแอป Flask แม้ว่าโครงสร้างนี้จะมีประโยชน์มาก แต่เป็นแอป Python แบบเต็มรูปแบบ และความต้องการของคุณอาจรวมถึงการใช้แอปพลิเคชัน JavaScript

ในบทเรียนนี้ คุณสามารถสร้างระบบพื้นฐานที่ใช้ JavaScript สำหรับการคาดการณ์ได้ แต่ก่อนอื่น คุณต้องฝึกโมเดลและแปลงมันเพื่อใช้งานกับ Onnx

## แบบฝึกหัด - ฝึกโมเดลการจำแนกประเภท

เริ่มต้นด้วยการฝึกโมเดลการจำแนกประเภทโดยใช้ชุดข้อมูลเมนูอาหารที่ทำความสะอาดแล้วที่เราใช้ก่อนหน้านี้

1. เริ่มต้นด้วยการนำเข้าห้องสมุดที่มีประโยชน์:

    ```python
    !pip install skl2onnx
    import pandas as pd 
    ```

    คุณต้องใช้ '[skl2onnx](https://onnx.ai/sklearn-onnx/)' เพื่อช่วยแปลงโมเดล Scikit-learn ของคุณเป็นรูปแบบ Onnx

1. จากนั้นทำงานกับข้อมูลในลักษณะเดียวกับที่คุณทำในบทเรียนก่อนหน้า โดยการอ่านไฟล์ CSV ด้วย `read_csv()`:

    ```python
    data = pd.read_csv('../data/cleaned_cuisines.csv')
    data.head()
    ```

1. ลบสองคอลัมน์แรกที่ไม่จำเป็นออก และบันทึกข้อมูลที่เหลือเป็น 'X':

    ```python
    X = data.iloc[:,2:]
    X.head()
    ```

1. บันทึกป้ายกำกับเป็น 'y':

    ```python
    y = data[['cuisine']]
    y.head()
    
    ```

### เริ่มต้นกระบวนการฝึกโมเดล

เราจะใช้ห้องสมุด 'SVC' ซึ่งมีความแม่นยำดี

1. นำเข้าห้องสมุดที่เหมาะสมจาก Scikit-learn:

    ```python
    from sklearn.model_selection import train_test_split
    from sklearn.svm import SVC
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import accuracy_score,precision_score,confusion_matrix,classification_report
    ```

1. แยกชุดข้อมูลการฝึกและการทดสอบ:

    ```python
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3)
    ```

1. สร้างโมเดลการจำแนกประเภท SVC เหมือนที่คุณทำในบทเรียนก่อนหน้า:

    ```python
    model = SVC(kernel='linear', C=10, probability=True,random_state=0)
    model.fit(X_train,y_train.values.ravel())
    ```

1. ทดสอบโมเดลของคุณโดยเรียกใช้ `predict()`:

    ```python
    y_pred = model.predict(X_test)
    ```

1. พิมพ์รายงานการจำแนกประเภทเพื่อตรวจสอบคุณภาพของโมเดล:

    ```python
    print(classification_report(y_test,y_pred))
    ```

    ดังที่เราเห็นก่อนหน้านี้ ความแม่นยำดี:

    ```output
                    precision    recall  f1-score   support
    
         chinese       0.72      0.69      0.70       257
          indian       0.91      0.87      0.89       243
        japanese       0.79      0.77      0.78       239
          korean       0.83      0.79      0.81       236
            thai       0.72      0.84      0.78       224
    
        accuracy                           0.79      1199
       macro avg       0.79      0.79      0.79      1199
    weighted avg       0.79      0.79      0.79      1199
    ```

### แปลงโมเดลของคุณเป็น Onnx

ตรวจสอบให้แน่ใจว่าการแปลงใช้จำนวน Tensor ที่เหมาะสม ชุดข้อมูลนี้มีส่วนผสม 380 รายการ ดังนั้นคุณต้องระบุจำนวนดังกล่าวใน `FloatTensorType`:

1. แปลงโดยใช้จำนวน Tensor เท่ากับ 380:

    ```python
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType
    
    initial_type = [('float_input', FloatTensorType([None, 380]))]
    options = {id(model): {'nocl': True, 'zipmap': False}}
    ```

1. สร้างไฟล์ onx และบันทึกเป็น **model.onnx**:

    ```python
    onx = convert_sklearn(model, initial_types=initial_type, options=options)
    with open("./model.onnx", "wb") as f:
        f.write(onx.SerializeToString())
    ```

    > หมายเหตุ คุณสามารถส่ง [options](https://onnx.ai/sklearn-onnx/parameterized.html) ในสคริปต์การแปลงของคุณได้ ในกรณีนี้ เราได้ตั้งค่า 'nocl' เป็น True และ 'zipmap' เป็น False เนื่องจากนี่เป็นโมเดลการจำแนกประเภท คุณมีตัวเลือกในการลบ ZipMap ซึ่งสร้างรายการของ dictionary (ไม่จำเป็น) `nocl` หมายถึงข้อมูลคลาสที่รวมอยู่ในโมเดล ลดขนาดโมเดลของคุณโดยตั้งค่า `nocl` เป็น 'True'

การรันโน้ตบุ๊กทั้งหมดจะสร้างโมเดล Onnx และบันทึกไว้ในโฟลเดอร์นี้

## ดูโมเดลของคุณ

โมเดล Onnx ไม่สามารถมองเห็นได้ชัดเจนใน Visual Studio Code แต่มีซอฟต์แวร์ฟรีที่ดีมากที่นักวิจัยหลายคนใช้เพื่อดูโมเดลเพื่อให้แน่ใจว่าสร้างขึ้นอย่างถูกต้อง ดาวน์โหลด [Netron](https://github.com/lutzroeder/Netron) และเปิดไฟล์ model.onnx ของคุณ คุณจะเห็นโมเดลง่ายๆ ของคุณที่มี 380 อินพุตและตัวจำแนกประเภท:

![Netron visual](../../../../4-Classification/4-Applied/images/netron.png)

Netron เป็นเครื่องมือที่มีประโยชน์ในการดูโมเดลของคุณ

ตอนนี้คุณพร้อมที่จะใช้โมเดลที่น่าสนใจนี้ในเว็บแอปแล้ว มาสร้างแอปที่มีประโยชน์เมื่อคุณมองเข้าไปในตู้เย็นและพยายามหาว่าส่วนผสมที่เหลืออยู่สามารถใช้ทำอาหารประเภทใดได้บ้างตามที่โมเดลของคุณกำหนด

## สร้างเว็บแอปแนะนำเมนูอาหาร

คุณสามารถใช้โมเดลของคุณโดยตรงในเว็บแอป โครงสร้างนี้ยังช่วยให้คุณรันมันในเครื่องและแม้กระทั่งออฟไลน์หากจำเป็น เริ่มต้นด้วยการสร้างไฟล์ `index.html` ในโฟลเดอร์เดียวกับที่คุณบันทึกไฟล์ `model.onnx`

1. ในไฟล์นี้ _index.html_ เพิ่มมาร์กอัปต่อไปนี้:

    ```html
    <!DOCTYPE html>
    <html>
        <header>
            <title>Cuisine Matcher</title>
        </header>
        <body>
            ...
        </body>
    </html>
    ```

1. ตอนนี้ ทำงานภายในแท็ก `body` เพิ่มมาร์กอัปเล็กน้อยเพื่อแสดงรายการของกล่องเลือกที่สะท้อนถึงส่วนผสมบางอย่าง:

    ```html
    <h1>Check your refrigerator. What can you create?</h1>
            <div id="wrapper">
                <div class="boxCont">
                    <input type="checkbox" value="4" class="checkbox">
                    <label>apple</label>
                </div>
            
                <div class="boxCont">
                    <input type="checkbox" value="247" class="checkbox">
                    <label>pear</label>
                </div>
            
                <div class="boxCont">
                    <input type="checkbox" value="77" class="checkbox">
                    <label>cherry</label>
                </div>
    
                <div class="boxCont">
                    <input type="checkbox" value="126" class="checkbox">
                    <label>fenugreek</label>
                </div>
    
                <div class="boxCont">
                    <input type="checkbox" value="302" class="checkbox">
                    <label>sake</label>
                </div>
    
                <div class="boxCont">
                    <input type="checkbox" value="327" class="checkbox">
                    <label>soy sauce</label>
                </div>
    
                <div class="boxCont">
                    <input type="checkbox" value="112" class="checkbox">
                    <label>cumin</label>
                </div>
            </div>
            <div style="padding-top:10px">
                <button onClick="startInference()">What kind of cuisine can you make?</button>
            </div> 
    ```

    สังเกตว่ากล่องเลือกแต่ละกล่องมีค่า ซึ่งสะท้อนถึงดัชนีที่ส่วนผสมอยู่ตามชุดข้อมูล ตัวอย่างเช่น แอปเปิ้ลในรายการตัวอักษรนี้อยู่ในคอลัมน์ที่ห้า ดังนั้นค่าของมันคือ '4' เนื่องจากเราเริ่มนับจาก 0 คุณสามารถดู [สเปรดชีตส่วนผสม](../../../../4-Classification/data/ingredient_indexes.csv) เพื่อค้นหาดัชนีของส่วนผสมที่กำหนด

    ดำเนินการต่อในไฟล์ index.html เพิ่มบล็อกสคริปต์ที่เรียกโมเดลหลังจากปิดแท็ก `</div>` สุดท้าย

1. ก่อนอื่น นำเข้า [Onnx Runtime](https://www.onnxruntime.ai/):

    ```html
    <script src="https://cdn.jsdelivr.net/npm/onnxruntime-web@1.9.0/dist/ort.min.js"></script> 
    ```

    > Onnx Runtime ใช้เพื่อช่วยให้สามารถรันโมเดล Onnx ของคุณบนแพลตฟอร์มฮาร์ดแวร์ที่หลากหลาย รวมถึงการปรับแต่งและ API เพื่อใช้งาน

1. เมื่อ Runtime อยู่ในที่แล้ว คุณสามารถเรียกใช้งานได้:

    ```html
    <script>
        const ingredients = Array(380).fill(0);
        
        const checks = [...document.querySelectorAll('.checkbox')];
        
        checks.forEach(check => {
            check.addEventListener('change', function() {
                // toggle the state of the ingredient
                // based on the checkbox's value (1 or 0)
                ingredients[check.value] = check.checked ? 1 : 0;
            });
        });

        function testCheckboxes() {
            // validate if at least one checkbox is checked
            return checks.some(check => check.checked);
        }

        async function startInference() {

            let atLeastOneChecked = testCheckboxes()

            if (!atLeastOneChecked) {
                alert('Please select at least one ingredient.');
                return;
            }
            try {
                // create a new session and load the model.
                
                const session = await ort.InferenceSession.create('./model.onnx');

                const input = new ort.Tensor(new Float32Array(ingredients), [1, 380]);
                const feeds = { float_input: input };

                // feed inputs and run
                const results = await session.run(feeds);

                // read from results
                alert('You can enjoy ' + results.label.data[0] + ' cuisine today!')

            } catch (e) {
                console.log(`failed to inference ONNX model`);
                console.error(e);
            }
        }
               
    </script>
    ```

ในโค้ดนี้ มีหลายสิ่งที่เกิดขึ้น:

1. คุณสร้างอาร์เรย์ที่มีค่า 380 ค่า (1 หรือ 0) เพื่อกำหนดและส่งไปยังโมเดลเพื่อการคาดการณ์ ขึ้นอยู่กับว่ากล่องเลือกส่วนผสมถูกเลือกหรือไม่
2. คุณสร้างอาร์เรย์ของกล่องเลือกและวิธีการตรวจสอบว่ากล่องเลือกถูกเลือกในฟังก์ชัน `init` ที่ถูกเรียกเมื่อแอปพลิเคชันเริ่มต้น เมื่อกล่องเลือกถูกเลือก อาร์เรย์ `ingredients` จะถูกเปลี่ยนแปลงเพื่อสะท้อนส่วนผสมที่เลือก
3. คุณสร้างฟังก์ชัน `testCheckboxes` ที่ตรวจสอบว่ามีกล่องเลือกใดถูกเลือกหรือไม่
4. คุณใช้ฟังก์ชัน `startInference` เมื่อปุ่มถูกกด และหากมีกล่องเลือกใดถูกเลือก คุณจะเริ่มการคาดการณ์
5. รูทีนการคาดการณ์รวมถึง:
   1. การตั้งค่าการโหลดแบบอะซิงโครนัสของโมเดล
   2. การสร้างโครงสร้าง Tensor เพื่อส่งไปยังโมเดล
   3. การสร้าง 'feeds' ที่สะท้อนถึงอินพุต `float_input` ที่คุณสร้างเมื่อฝึกโมเดลของคุณ (คุณสามารถใช้ Netron เพื่อตรวจสอบชื่อนั้น)
   4. การส่ง 'feeds' เหล่านี้ไปยังโมเดลและรอการตอบกลับ

## ทดสอบแอปพลิเคชันของคุณ

เปิดเซสชันเทอร์มินัลใน Visual Studio Code ในโฟลเดอร์ที่ไฟล์ index.html ของคุณอยู่ ตรวจสอบให้แน่ใจว่าคุณได้ติดตั้ง [http-server](https://www.npmjs.com/package/http-server) ไว้ทั่วโลก และพิมพ์ `http-server` ที่พรอมต์ localhost จะเปิดขึ้นและคุณสามารถดูเว็บแอปของคุณได้ ตรวจสอบว่าเมนูอาหารใดที่แนะนำตามส่วนผสมต่างๆ:

![ingredient web app](../../../../4-Classification/4-Applied/images/web-app.png)

ยินดีด้วย คุณได้สร้างเว็บแอป 'แนะนำ' พร้อมฟิลด์ไม่กี่ฟิลด์ ใช้เวลาสร้างระบบนี้เพิ่มเติม!

## 🚀ความท้าทาย

เว็บแอปของคุณยังค่อนข้างเรียบง่าย ดังนั้นให้สร้างมันต่อโดยใช้ส่วนผสมและดัชนีของมันจากข้อมูล [ingredient_indexes](../../../../4-Classification/data/ingredient_indexes.csv) รสชาติแบบไหนที่เหมาะสมในการสร้างอาหารประจำชาติ?

## [แบบทดสอบหลังเรียน](https://ff-quizzes.netlify.app/en/ml/)

## ทบทวนและศึกษาด้วยตนเอง

แม้ว่าบทเรียนนี้จะกล่าวถึงประโยชน์ของการสร้างระบบแนะนำสำหรับส่วนผสมอาหารเพียงเล็กน้อย แต่พื้นที่ของแอปพลิเคชัน ML นี้มีตัวอย่างที่หลากหลาย อ่านเพิ่มเติมเกี่ยวกับวิธีการสร้างระบบเหล่านี้:

- https://www.sciencedirect.com/topics/computer-science/recommendation-engine
- https://www.technologyreview.com/2014/08/25/171547/the-ultimate-challenge-for-recommendation-engines/
- https://www.technologyreview.com/2015/03/23/168831/everything-is-a-recommendation/

## งานที่ได้รับมอบหมาย

[สร้างระบบแนะนำใหม่](assignment.md)

---

**ข้อจำกัดความรับผิดชอบ**:  
เอกสารนี้ได้รับการแปลโดยใช้บริการแปลภาษา AI [Co-op Translator](https://github.com/Azure/co-op-translator) แม้ว่าเราจะพยายามให้การแปลมีความถูกต้องมากที่สุด แต่โปรดทราบว่าการแปลอัตโนมัติอาจมีข้อผิดพลาดหรือความไม่ถูกต้อง เอกสารต้นฉบับในภาษาดั้งเดิมควรถือเป็นแหล่งข้อมูลที่เชื่อถือได้ สำหรับข้อมูลที่สำคัญ ขอแนะนำให้ใช้บริการแปลภาษาจากผู้เชี่ยวชาญ เราไม่รับผิดชอบต่อความเข้าใจผิดหรือการตีความที่ผิดพลาดซึ่งเกิดจากการใช้การแปลนี้