<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "61bdec27ed2da8b098cd9065405d9bb0",
  "translation_date": "2025-09-05T13:12:28+00:00",
  "source_file": "4-Classification/4-Applied/README.md",
  "language_code": "my"
}
-->
# အစားအစာအမျိုးအစား အကြံပေးဝက်ဘ်အက်ပ် တည်ဆောက်ခြင်း

ဒီသင်ခန်းစာမှာ သင်ဟာ အတိတ်သင်ခန်းစာတွေမှာ သင်ယူခဲ့တဲ့နည်းလမ်းတွေကို အသုံးပြုပြီး အစားအစာအမျိုးအစားဒေတာစနစ်ကို အသုံးပြုကာ အမျိုးအစားခွဲခြားမော်ဒယ်တစ်ခု တည်ဆောက်ပါမယ်။ ထို့အပြင် Onnx ရဲ့ web runtime ကို အသုံးပြုကာ သိမ်းဆည်းထားတဲ့မော်ဒယ်ကို အသုံးပြုနိုင်တဲ့ ဝက်ဘ်အက်ပ်လေးတစ်ခုကိုလည်း တည်ဆောက်ပါမယ်။

Machine Learning ရဲ့ အကျိုးရှိတဲ့ လက်တွေ့အသုံးချမှုတစ်ခုက အကြံပေးစနစ်တွေ တည်ဆောက်ခြင်းဖြစ်ပြီး ဒီနေ့မှာ သင်အဲဒီလမ်းကြောင်းကို စတင်နိုင်ပါပြီ!

[![ဒီဝက်ဘ်အက်ပ်ကို ဖော်ပြခြင်း](https://img.youtube.com/vi/17wdM9AHMfg/0.jpg)](https://youtu.be/17wdM9AHMfg "Applied ML")

> 🎥 အပေါ်ကပုံကို နှိပ်ပြီး ဗီဒီယိုကြည့်ပါ: Jen Looper က အစားအစာအမျိုးအစားဒေတာကို အသုံးပြုကာ ဝက်ဘ်အက်ပ်တစ်ခု တည်ဆောက်နေသည်

## [သင်ခန်းစာမတိုင်မီ မေးခွန်း](https://ff-quizzes.netlify.app/en/ml/)

ဒီသင်ခန်းစာမှာ သင်လေ့လာရမယ့်အရာတွေက:

- မော်ဒယ်တစ်ခုကို တည်ဆောက်ပြီး Onnx မော်ဒယ်အဖြစ် သိမ်းဆည်းနည်း
- Netron ကို အသုံးပြုကာ မော်ဒယ်ကို စစ်ဆေးနည်း
- မော်ဒယ်ကို ဝက်ဘ်အက်ပ်မှာ အသုံးပြုကာ အတိအကျခန့်မှန်းနည်း

## မော်ဒယ်ကို တည်ဆောက်ပါ

Applied ML စနစ်တွေကို တည်ဆောက်ခြင်းဟာ ဒီနည်းပညာတွေကို သင့်လုပ်ငန်းစနစ်တွေမှာ အသုံးချဖို့ အရေးကြီးတဲ့အပိုင်းတစ်ခုဖြစ်ပါတယ်။ Onnx ကို အသုံးပြုကာ သင့်ဝက်ဘ်အက်ပ်တွေမှာ မော်ဒယ်တွေကို အသုံးပြုနိုင်ပါတယ် (လိုအပ်ပါက offline context မှာလည်း အသုံးပြုနိုင်ပါတယ်)။

[အတိတ်သင်ခန်းစာ](../../3-Web-App/1-Web-App/README.md) မှာ သင်ဟာ UFO တွေကို ရှာဖွေတဲ့ Regression မော်ဒယ်တစ်ခုကို တည်ဆောက်ပြီး "pickle" လုပ်ကာ Flask app မှာ အသုံးပြုခဲ့ပါတယ်။ ဒီ architecture ဟာ သိထားသင့်တဲ့ အသုံးဝင်တဲ့နည်းလမ်းတစ်ခုဖြစ်ပေမယ့် Python အပြည့်အစုံ stack app ဖြစ်ပြီး သင့်လိုအပ်ချက်တွေမှာ JavaScript application ကို အသုံးပြုဖို့လိုအပ်နိုင်ပါတယ်။

ဒီသင်ခန်းစာမှာ JavaScript-based စနစ်တစ်ခုကို အတိအကျခန့်မှန်းဖို့ တည်ဆောက်နိုင်ပါတယ်။ ဒါပေမယ့် အရင်ဆုံး မော်ဒယ်တစ်ခုကို လေ့ကျင့်ပြီး Onnx နဲ့ အသုံးပြုနိုင်ဖို့ ပြောင်းလဲရပါမယ်။

## လေ့ကျင့်မှု - အမျိုးအစားခွဲခြားမော်ဒယ်ကို လေ့ကျင့်ပါ

အရင်ဆုံး သင်အသုံးပြုခဲ့တဲ့ အစားအစာအမျိုးအစားဒေတာစနစ်ကို အသုံးပြုကာ အမျိုးအစားခွဲခြားမော်ဒယ်တစ်ခုကို လေ့ကျင့်ပါ။

1. အသုံးဝင်တဲ့ library တွေကို အရင် Import လုပ်ပါ:

    ```python
    !pip install skl2onnx
    import pandas as pd 
    ```

    '[skl2onnx](https://onnx.ai/sklearn-onnx/)' ကို သင့် Scikit-learn မော်ဒယ်ကို Onnx format ပြောင်းဖို့ အထောက်အကူပြုပါမယ်။

1. အတိတ်သင်ခန်းစာတွေမှာ လုပ်ခဲ့သလို `read_csv()` ကို အသုံးပြုကာ သင့်ဒေတာကို အလုပ်လုပ်ပါ:

    ```python
    data = pd.read_csv('../data/cleaned_cuisines.csv')
    data.head()
    ```

1. မလိုအပ်တဲ့ အတန်းနှစ်ခုကို ဖယ်ရှားပြီး ကျန်ရှိတဲ့ဒေတာကို 'X' အဖြစ် သိမ်းဆည်းပါ:

    ```python
    X = data.iloc[:,2:]
    X.head()
    ```

1. Label တွေကို 'y' အဖြစ် သိမ်းဆည်းပါ:

    ```python
    y = data[['cuisine']]
    y.head()
    
    ```

### လေ့ကျင့်မှုစနစ်ကို စတင်ပါ

'SVC' library ကို အသုံးပြုပါမယ်၊ accuracy ကောင်းပါတယ်။

1. Scikit-learn မှ သင့် library တွေကို Import လုပ်ပါ:

    ```python
    from sklearn.model_selection import train_test_split
    from sklearn.svm import SVC
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import accuracy_score,precision_score,confusion_matrix,classification_report
    ```

1. လေ့ကျင့်မှုနှင့် စမ်းသပ်မှု set တွေကို ခွဲခြားပါ:

    ```python
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3)
    ```

1. အတိတ်သင်ခန်းစာမှာ လုပ်ခဲ့သလို SVC Classification မော်ဒယ်တစ်ခုကို တည်ဆောက်ပါ:

    ```python
    model = SVC(kernel='linear', C=10, probability=True,random_state=0)
    model.fit(X_train,y_train.values.ravel())
    ```

1. အခုတော့ သင့်မော်ဒယ်ကို စမ်းသပ်ပြီး `predict()` ကို ခေါ်ပါ:

    ```python
    y_pred = model.predict(X_test)
    ```

1. Classification report ကို print ထုတ်ပြီး မော်ဒယ်ရဲ့ အရည်အသွေးကို စစ်ဆေးပါ:

    ```python
    print(classification_report(y_test,y_pred))
    ```

    အရင်မြင်ခဲ့သလို accuracy ကောင်းပါတယ်:

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

### သင့်မော်ဒယ်ကို Onnx format ပြောင်းပါ

Tensor နံပါတ်ကို သင့်တော်အောင် ပြောင်းလဲပါ။ ဒီဒေတာစနစ်မှာ 380 ခုသော အစားအစာပါဝင်ပြီး `FloatTensorType` မှာ အဲဒီနံပါတ်ကို မှတ်သားရပါမယ်။

1. Tensor နံပါတ် 380 ကို အသုံးပြုကာ ပြောင်းပါ:

    ```python
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType
    
    initial_type = [('float_input', FloatTensorType([None, 380]))]
    options = {id(model): {'nocl': True, 'zipmap': False}}
    ```

1. **model.onnx** အဖြစ် ဖိုင်အနေနဲ့ သိမ်းဆည်းပါ:

    ```python
    onx = convert_sklearn(model, initial_types=initial_type, options=options)
    with open("./model.onnx", "wb") as f:
        f.write(onx.SerializeToString())
    ```

    > မှတ်ချက် - သင့် conversion script မှာ [options](https://onnx.ai/sklearn-onnx/parameterized.html) တွေကို pass လုပ်နိုင်ပါတယ်။ ဒီအခါမှာ 'nocl' ကို True အဖြစ်၊ 'zipmap' ကို False အဖြစ် pass လုပ်ထားပါတယ်။ Classification မော်ဒယ်ဖြစ်တဲ့အတွက် ZipMap ကို ဖယ်ရှားနိုင်ပါတယ် (မလိုအပ်ပါဘူး)။ `nocl` ဟာ မော်ဒယ်မှာ class အချက်အလက်ကို ထည့်သွင်းခြင်းကို ရည်ညွှန်းပါတယ်။ `nocl` ကို 'True' အဖြစ် သတ်မှတ်ခြင်းအားဖြင့် မော်ဒယ်ရဲ့အရွယ်အစားကို လျှော့ချနိုင်ပါတယ်။

Notebook အားလုံးကို run လုပ်ပြီး Onnx မော်ဒယ်တစ်ခုကို တည်ဆောက်ကာ ဒီ folder မှာ သိမ်းဆည်းပါ။

## သင့်မော်ဒယ်ကို ကြည့်ပါ

Onnx မော်ဒယ်တွေဟာ Visual Studio Code မှာ မမြင်နိုင်ပါဘူး၊ ဒါပေမယ့် မော်ဒယ်တစ်ခုကို visualization လုပ်ဖို့ အခမဲ့ software ကောင်းတစ်ခုရှိပါတယ်။ [Netron](https://github.com/lutzroeder/Netron) ကို download လုပ်ပြီး သင့် model.onnx ဖိုင်ကို ဖွင့်ပါ။ 380 inputs နဲ့ classifier ပါဝင်တဲ့ သင့်မော်ဒယ်ကို ရိုးရှင်းစွာ visualization လုပ်နိုင်ပါတယ်:

![Netron visual](../../../../4-Classification/4-Applied/images/netron.png)

Netron ဟာ သင့်မော်ဒယ်တွေကို ကြည့်ရှုဖို့ အထောက်အကူပြုတဲ့ tool တစ်ခုဖြစ်ပါတယ်။

## အကြံပေးဝက်ဘ်အက်ပ် တည်ဆောက်ပါ

သင့်မော်ဒယ်ကို ဝက်ဘ်အက်ပ်မှာ တိုက်ရိုက်အသုံးပြုနိုင်ပါတယ်။ ဒီ architecture ဟာ local မှာ run လုပ်နိုင်ပြီး offline မှာလည်း အသုံးပြုနိုင်ပါတယ်။ သင့် `model.onnx` ဖိုင်ကို သိမ်းဆည်းထားတဲ့ folder မှာ `index.html` ဖိုင်တစ်ခုကို စတင်ဖန်တီးပါ။

1. ဒီဖိုင် _index.html_ မှာ အောက်ပါ markup ကို ထည့်ပါ:

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

1. `body` tag တွေထဲမှာ အစားအစာအချို့ကို ပြသတဲ့ checkbox တွေကို ထည့်ပါ:

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

    Checkbox တစ်ခုစီကို value သတ်မှတ်ထားပါတယ်။ ဒါဟာ dataset အရ အစားအစာရဲ့ index ကို ရည်ညွှန်းပါတယ်။ ဥပမာ - Apple ဟာ alphabetic စဉ်အရ column 5 မှာရှိပြီး value ဟာ '4' ဖြစ်ပါတယ် (0 မှ စတင်ရေတွက်ပါတယ်)။ [ingredients spreadsheet](../../../../4-Classification/data/ingredient_indexes.csv) ကို ကြည့်ပြီး အစားအစာတစ်ခုရဲ့ index ကို ရှာဖွေပါ။

    index.html ဖိုင်မှာ ဆက်လက်လုပ်ဆောင်ပြီး script block ကို နောက်ဆုံး `</div>` tag ရဲ့ အောက်မှာ ထည့်ပါ။

1. အရင်ဆုံး [Onnx Runtime](https://www.onnxruntime.ai/) ကို Import လုပ်ပါ:

    ```html
    <script src="https://cdn.jsdelivr.net/npm/onnxruntime-web@1.9.0/dist/ort.min.js"></script> 
    ```

    > Onnx Runtime ကို အသုံးပြုကာ သင့် Onnx မော်ဒယ်တွေကို hardware platform အမျိုးမျိုးမှာ run လုပ်နိုင်ပါတယ်၊ optimization တွေကိုလည်း ပါဝင်ပါတယ်။

1. Runtime ကို အသုံးပြုပါ:

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

ဒီ code မှာ အောက်ပါအရာတွေကို လုပ်ဆောင်ထားပါတယ်:

1. 380 ခုသော value (1 or 0) တွေကို သတ်မှတ်ပြီး မော်ဒယ်ကို inference အတွက် ပို့ရန် array တစ်ခုကို ဖန်တီးထားပါတယ်။
2. Checkbox array တစ်ခုကို ဖန်တီးပြီး `init` function မှာ checkbox တွေ checked ဖြစ်/မဖြစ်ကို စစ်ဆေးထားပါတယ်။
3. `testCheckboxes` function ကို ဖန်တီးပြီး checkbox checked ဖြစ်/မဖြစ်ကို စစ်ဆေးထားပါတယ်။
4. Button ကို နှိပ်တဲ့အခါ `startInference` function ကို အသုံးပြုကာ inference စတင်ပါ။
5. Inference routine မှာ:
   1. မော်ဒယ်ကို asynchronous load လုပ်ခြင်း
   2. Tensor structure ကို ဖန်တီးပြီး မော်ဒယ်ကို ပို့ခြင်း
   3. `float_input` input ကို feeds အဖြစ် ဖန်တီးခြင်း (Netron မှာ အဲဒီနာမည်ကို စစ်ဆေးနိုင်ပါတယ်)
   4. Feed တွေကို မော်ဒယ်ကို ပို့ပြီး အဖြေကို စောင့်ဆိုင်းခြင်း

## သင့်အက်ပ်ကို စမ်းသပ်ပါ

Visual Studio Code မှာ terminal session ကို ဖွင့်ပြီး သင့် index.html ဖိုင်ရှိတဲ့ folder မှာ `http-server` ကို run လုပ်ပါ။ localhost ကို ဖွင့်ပြီး သင့်ဝက်ဘ်အက်ပ်ကို ကြည့်ရှုနိုင်ပါတယ်။ အစားအစာအမျိုးအစားကို အကြံပေးမှုအတွက် စမ်းသပ်ပါ:

![ingredient web app](../../../../4-Classification/4-Applied/images/web-app.png)

အောင်မြင်ပါတယ်၊ သင့်အစားအစာအမျိုးအစားအကြံပေးဝက်ဘ်အက်ပ်ကို တည်ဆောက်ပြီး ပြီးဆုံးပါပြီ။ ဒီစနစ်ကို ဆက်လက်တိုးချဲ့ဖို့ အချိန်ယူပါ!

## 🚀စိန်ခေါ်မှု

သင့်ဝက်ဘ်အက်ပ်ဟာ အလွန်ရိုးရှင်းပါတယ်၊ [ingredient_indexes](../../../../4-Classification/data/ingredient_indexes.csv) ဒေတာမှ အစားအစာနဲ့ index တွေကို အသုံးပြုကာ ဆက်လက်တိုးချဲ့ပါ။ ဘယ်အစားအစာအရသာပေါင်းစပ်မှုတွေက အမျိုးသားအစားအစာကို ဖန်တီးနိုင်မလဲ?

## [သင်ခန်းစာပြီးနောက် မေးခွန်း](https://ff-quizzes.netlify.app/en/ml/)

## ပြန်လည်သုံးသပ်ခြင်းနှင့် ကိုယ်တိုင်လေ့လာခြင်း

ဒီသင်ခန်းစာမှာ အစားအစာအမျိုးအစားအကြံပေးစနစ်တစ်ခုကို တည်ဆောက်ခြင်းရဲ့ အသုံးဝင်မှုကို အနည်းငယ်သာ ထိတွေ့ခဲ့ပါတယ်။ ဒီ ML application နယ်ပယ်ဟာ ဥပမာများစွာပါဝင်တဲ့ နယ်ပယ်တစ်ခုဖြစ်ပါတယ်။ ဒီစနစ်တွေကို ဘယ်လိုတည်ဆောက်ကြောင်းကို ဆက်လက်ဖတ်ရှုပါ:

- https://www.sciencedirect.com/topics/computer-science/recommendation-engine
- https://www.technologyreview.com/2014/08/25/171547/the-ultimate-challenge-for-recommendation-engines/
- https://www.technologyreview.com/2015/03/23/168831/everything-is-a-recommendation/

## လုပ်ငန်းတာဝန်

[အကြံပေးစနစ်အသစ်တစ်ခု တည်ဆောက်ပါ](assignment.md)

---

**ဝက်ဘ်ဆိုက်မှတ်ချက်**:  
ဤစာရွက်စာတမ်းကို AI ဘာသာပြန်ဝန်ဆောင်မှု [Co-op Translator](https://github.com/Azure/co-op-translator) ကို အသုံးပြု၍ ဘာသာပြန်ထားပါသည်။ ကျွန်ုပ်တို့သည် တိကျမှန်ကန်မှုအတွက် ကြိုးစားနေသော်လည်း၊ အလိုအလျောက်ဘာသာပြန်ဆိုမှုများတွင် အမှားများ သို့မဟုတ် မမှန်ကန်မှုများ ပါဝင်နိုင်ကြောင်း သတိပြုပါ။ မူရင်းစာရွက်စာတမ်းကို ၎င်း၏ မူလဘာသာစကားဖြင့် အာဏာတည်သောရင်းမြစ်အဖြစ် သတ်မှတ်သင့်ပါသည်။ အရေးကြီးသောအချက်အလက်များအတွက် လူ့ဘာသာပြန်ပညာရှင်များမှ ပြန်ဆိုမှုကို အကြံပြုပါသည်။ ဤဘာသာပြန်ကို အသုံးပြုခြင်းမှ ဖြစ်ပေါ်လာသော နားလည်မှုမှားများ သို့မဟုတ် အဓိပ္ပါယ်မှားများအတွက် ကျွန်ုပ်တို့သည် တာဝန်မယူပါ။