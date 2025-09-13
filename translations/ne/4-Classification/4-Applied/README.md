<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "61bdec27ed2da8b098cd9065405d9bb0",
  "translation_date": "2025-09-06T06:37:11+00:00",
  "source_file": "4-Classification/4-Applied/README.md",
  "language_code": "ne"
}
-->
# क्युइजिन सिफारिस वेब एप निर्माण गर्नुहोस्

यस पाठमा, तपाईंले अघिल्लो पाठहरूमा सिकेका केही प्रविधिहरू प्रयोग गरेर वर्गीकरण मोडेल निर्माण गर्नुहुनेछ र यस शृंखलामा प्रयोग गरिएको स्वादिष्ट क्युइजिन डेटासेटको साथ काम गर्नुहुनेछ। साथै, तपाईंले एक सानो वेब एप निर्माण गर्नुहुनेछ जसले बचत गरिएको मोडेल प्रयोग गर्दछ, Onnx को वेब रनटाइमलाई उपयोग गर्दै।

मेसिन लर्निङको सबैभन्दा उपयोगी व्यावहारिक प्रयोगहरू मध्ये एक सिफारिस प्रणाली निर्माण गर्नु हो, र तपाईं आज त्यस दिशामा पहिलो कदम चाल्न सक्नुहुन्छ!

> 🎥 माथिको छवि क्लिक गर्नुहोस् भिडियोका लागि: जेन लूपरले वर्गीकृत क्युइजिन डाटाको प्रयोग गरेर वेब एप निर्माण गर्छिन्

## [पाठ अघि क्विज](https://ff-quizzes.netlify.app/en/ml/)

यस पाठमा तपाईंले सिक्नुहुनेछ:

- कसरी मोडेल निर्माण गर्ने र यसलाई Onnx मोडेलको रूपमा बचत गर्ने
- कसरी Netron प्रयोग गरेर मोडेल निरीक्षण गर्ने
- कसरी वेब एपमा आफ्नो मोडेल प्रयोग गर्ने

## आफ्नो मोडेल निर्माण गर्नुहोस्

व्यावसायिक प्रणालीहरूको लागि यी प्रविधिहरूलाई उपयोग गर्न लागू गरिएको मेसिन लर्निङ प्रणाली निर्माण गर्नु महत्त्वपूर्ण छ। तपाईं आफ्नो वेब एप्लिकेसनहरूमा मोडेलहरू प्रयोग गर्न सक्नुहुन्छ (र आवश्यक परेमा अफलाइन सन्दर्भमा पनि) Onnx प्रयोग गरेर।

[अघिल्लो पाठमा](../../3-Web-App/1-Web-App/README.md), तपाईंले UFO sightings सम्बन्धी एक Regression मोडेल निर्माण गर्नुभयो, यसलाई "पिकल" गर्नुभयो, र यसलाई Flask एपमा प्रयोग गर्नुभयो। यो आर्किटेक्चर जान्न उपयोगी छ, तर यो पूर्ण-स्ट्याक Python एप हो, र तपाईंको आवश्यकताहरूमा JavaScript एप्लिकेसनको प्रयोग समावेश हुन सक्छ।

यस पाठमा, तपाईंले JavaScript-आधारित प्रणाली निर्माण गर्न सक्नुहुन्छ। तर, पहिले, तपाईंले मोडेल प्रशिक्षण गर्नुपर्छ र यसलाई Onnx को साथ प्रयोग गर्न रूपान्तरण गर्नुपर्छ।

## अभ्यास - वर्गीकरण मोडेल प्रशिक्षण गर्नुहोस्

पहिले, सफा गरिएको क्युइजिन डेटासेट प्रयोग गरेर वर्गीकरण मोडेल प्रशिक्षण गर्नुहोस्।

1. उपयोगी लाइब्रेरीहरू आयात गरेर सुरु गर्नुहोस्:

    ```python
    !pip install skl2onnx
    import pandas as pd 
    ```

    तपाईंलाई '[skl2onnx](https://onnx.ai/sklearn-onnx/)' चाहिन्छ जसले तपाईंको Scikit-learn मोडेललाई Onnx ढाँचामा रूपान्तरण गर्न मद्दत गर्दछ।

1. त्यसपछि, अघिल्लो पाठहरूमा जस्तै, `read_csv()` प्रयोग गरेर CSV फाइलसँग काम गर्नुहोस्:

    ```python
    data = pd.read_csv('../data/cleaned_cuisines.csv')
    data.head()
    ```

1. पहिलो दुई अनावश्यक स्तम्भहरू हटाउनुहोस् र बाँकी डाटालाई 'X' को रूपमा बचत गर्नुहोस्:

    ```python
    X = data.iloc[:,2:]
    X.head()
    ```

1. लेबलहरूलाई 'y' को रूपमा बचत गर्नुहोस्:

    ```python
    y = data[['cuisine']]
    y.head()
    
    ```

### प्रशिक्षण प्रक्रिया सुरु गर्नुहोस्

हामी 'SVC' लाइब्रेरी प्रयोग गर्नेछौं जसको राम्रो शुद्धता छ।

1. Scikit-learn बाट उपयुक्त लाइब्रेरीहरू आयात गर्नुहोस्:

    ```python
    from sklearn.model_selection import train_test_split
    from sklearn.svm import SVC
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import accuracy_score,precision_score,confusion_matrix,classification_report
    ```

1. प्रशिक्षण र परीक्षण सेटहरू अलग गर्नुहोस्:

    ```python
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3)
    ```

1. अघिल्लो पाठमा जस्तै SVC वर्गीकरण मोडेल निर्माण गर्नुहोस्:

    ```python
    model = SVC(kernel='linear', C=10, probability=True,random_state=0)
    model.fit(X_train,y_train.values.ravel())
    ```

1. अब, आफ्नो मोडेल परीक्षण गर्नुहोस्, `predict()` कल गरेर:

    ```python
    y_pred = model.predict(X_test)
    ```

1. मोडेलको गुणस्तर जाँच गर्न वर्गीकरण रिपोर्ट प्रिन्ट गर्नुहोस्:

    ```python
    print(classification_report(y_test,y_pred))
    ```

    जस्तै हामीले पहिले देख्यौं, शुद्धता राम्रो छ:

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

### आफ्नो मोडेललाई Onnx मा रूपान्तरण गर्नुहोस्

सुनिश्चित गर्नुहोस् कि रूपान्तरण सही Tensor नम्बरको साथ गरिएको छ। यस डेटासेटमा 380 सामग्रीहरू सूचीबद्ध छन्, त्यसैले तपाईंले `FloatTensorType` मा त्यो नम्बर उल्लेख गर्नुपर्छ:

1. 380 को Tensor नम्बर प्रयोग गरेर रूपान्तरण गर्नुहोस्।

    ```python
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType
    
    initial_type = [('float_input', FloatTensorType([None, 380]))]
    options = {id(model): {'nocl': True, 'zipmap': False}}
    ```

1. **model.onnx** नामक फाइलको रूपमा onx सिर्जना गर्नुहोस् र बचत गर्नुहोस्:

    ```python
    onx = convert_sklearn(model, initial_types=initial_type, options=options)
    with open("./model.onnx", "wb") as f:
        f.write(onx.SerializeToString())
    ```

    > नोट, तपाईं आफ्नो रूपान्तरण स्क्रिप्टमा [विकल्पहरू](https://onnx.ai/sklearn-onnx/parameterized.html) पास गर्न सक्नुहुन्छ। यस अवस्थामा, हामीले 'nocl' लाई True र 'zipmap' लाई False सेट गरेका छौं। किनभने यो वर्गीकरण मोडेल हो, तपाईं ZipMap हटाउन सक्नुहुन्छ जसले शब्दकोशहरूको सूची उत्पादन गर्दछ (आवश्यक छैन)। `nocl` मोडेलमा वर्ग जानकारी समावेश गरिएको हो। `nocl` लाई 'True' सेट गरेर आफ्नो मोडेलको आकार घटाउनुहोस्।

पूरा नोटबुक चलाउँदा अब Onnx मोडेल निर्माण हुनेछ र यसलाई यस फोल्डरमा बचत गरिनेछ।

## आफ्नो मोडेल हेर्नुहोस्

Onnx मोडेलहरू Visual Studio कोडमा धेरै देखिने छैनन्, तर त्यहाँ एक धेरै राम्रो निःशुल्क सफ्टवेयर छ जुन धेरै अनुसन्धानकर्ताहरूले मोडेललाई हेर्न प्रयोग गर्छन्। [Netron](https://github.com/lutzroeder/Netron) डाउनलोड गर्नुहोस् र आफ्नो model.onnx फाइल खोल्नुहोस्। तपाईं आफ्नो सरल मोडेललाई 380 इनपुटहरू र वर्गीकरणकर्ता सहित देख्न सक्नुहुन्छ:

Netron मोडेलहरू हेर्नको लागि उपयोगी उपकरण हो।

अब तपाईंले यो राम्रो मोडेललाई वेब एपमा प्रयोग गर्न तयार हुनुहुन्छ। एउटा एप निर्माण गरौं जुन तपाईंको फ्रिजमा हेर्दा र बाँकी सामग्रीहरूको संयोजनले तपाईंको मोडेलले निर्धारण गरेको क्युइजिन पकाउन सक्नेछ।

## सिफारिस वेब एप्लिकेसन निर्माण गर्नुहोस्

तपाईं आफ्नो मोडेललाई सिधै वेब एपमा प्रयोग गर्न सक्नुहुन्छ। यो आर्किटेक्चरले तपाईंलाई यसलाई स्थानीय रूपमा चलाउन र आवश्यक परेमा अफलाइन पनि चलाउन अनुमति दिन्छ। `index.html` फाइल सिर्जना गरेर सुरु गर्नुहोस् जहाँ तपाईंले आफ्नो `model.onnx` फाइल बचत गर्नुभएको छ।

1. यस फाइल _index.html_ मा, निम्न मार्कअप थप्नुहोस्:

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

1. अब, `body` ट्यागहरू भित्र काम गर्दै, केही सामग्रीहरू प्रतिबिम्बित गर्ने चेकबक्सहरूको सूची देखाउन थोरै मार्कअप थप्नुहोस्:

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

    ध्यान दिनुहोस् कि प्रत्येक चेकबक्सलाई एउटा मान दिइएको छ। यो डेटासेट अनुसार सामग्री पाइने स्तम्भको सूचकांकलाई प्रतिबिम्बित गर्दछ। उदाहरणका लागि, Apple यस वर्णानुक्रमिक सूचीमा पाँचौं स्तम्भमा छ, त्यसैले यसको मान '4' हो किनभने हामी 0 बाट गणना सुरु गर्छौं। [ingredients spreadsheet](../../../../4-Classification/data/ingredient_indexes.csv) परामर्श गरेर कुनै सामग्रीको सूचकांक पत्ता लगाउन सक्नुहुन्छ।

    आफ्नो कामलाई index.html फाइलमा जारी राख्दै, स्क्रिप्ट ब्लक थप्नुहोस् जहाँ अन्तिम बन्द `</div>` पछि मोडेल कल गरिन्छ।

1. पहिले, [Onnx Runtime](https://www.onnxruntime.ai/) आयात गर्नुहोस्:

    ```html
    <script src="https://cdn.jsdelivr.net/npm/onnxruntime-web@1.9.0/dist/ort.min.js"></script> 
    ```

    > Onnx Runtime विभिन्न हार्डवेयर प्लेटफर्महरूमा तपाईंको Onnx मोडेलहरू चलाउन सक्षम बनाउन प्रयोग गरिन्छ, जसमा अनुकूलनहरू र प्रयोग गर्न API समावेश छ।

1. Runtime सेट भएपछि, तपाईं यसलाई कल गर्न सक्नुहुन्छ:

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

यस कोडमा, धेरै कुराहरू भइरहेको छ:

1. तपाईंले 380 सम्भावित मानहरूको (1 वा 0) एक एरे सिर्जना गर्नुभयो जुन सामग्री चेकबक्स जाँच गरिएको छ कि छैन भन्ने आधारमा मोडेलमा पठाइनेछ।
2. तपाईंले चेकबक्सहरूको एरे सिर्जना गर्नुभयो र एप्लिकेसन सुरु हुँदा कल गरिने `init` फङ्सनमा चेकबक्स जाँच गरिएको छ कि छैन भन्ने निर्धारण गर्ने तरिका बनाउनु भयो। जब चेकबक्स जाँच गरिन्छ, `ingredients` एरेलाई चयन गरिएको सामग्रीलाई प्रतिबिम्बित गर्न परिवर्तन गरिन्छ।
3. तपाईंले `testCheckboxes` फङ्सन सिर्जना गर्नुभयो जसले कुनै चेकबक्स जाँच गरिएको छ कि छैन जाँच गर्दछ।
4. तपाईंले बटन थिच्दा `startInference` फङ्सन प्रयोग गर्नुभयो, र यदि कुनै चेकबक्स जाँच गरिएको छ भने, तपाईंले अनुमान सुरु गर्नुहुन्छ।
5. अनुमान प्रक्रिया समावेश गर्दछ:
   1. मोडेलको एसिंक्रोनस लोड सेटअप गर्नु
   2. मोडेलमा पठाउन टेन्सर संरचना सिर्जना गर्नु
   3. 'feeds' सिर्जना गर्नु जुन तपाईंले आफ्नो मोडेल प्रशिक्षण गर्दा सिर्जना गरेको `float_input` इनपुटलाई प्रतिबिम्बित गर्दछ (तपाईं Netron प्रयोग गरेर नाम प्रमाणित गर्न सक्नुहुन्छ)
   4. यी 'feeds' मोडेलमा पठाउनु र प्रतिक्रिया पर्खनु

## आफ्नो एप्लिकेसन परीक्षण गर्नुहोस्

Visual Studio Code मा टर्मिनल सत्र खोल्नुहोस् जहाँ तपाईंको index.html फाइल रहेको छ। सुनिश्चित गर्नुहोस् कि तपाईंले [http-server](https://www.npmjs.com/package/http-server) ग्लोबल रूपमा स्थापना गर्नुभएको छ, र प्रम्प्टमा `http-server` टाइप गर्नुहोस्। एक localhost खुल्नेछ र तपाईं आफ्नो वेब एप हेर्न सक्नुहुन्छ। विभिन्न सामग्रीहरूको आधारमा कुन क्युइजिन सिफारिस गरिएको छ जाँच गर्नुहोस्:

बधाई छ, तपाईंले केही क्षेत्रहरू भएको 'सिफारिस' वेब एप निर्माण गर्नुभयो। यस प्रणालीलाई निर्माण गर्न समय लिनुहोस्!

## 🚀चुनौती

तपाईंको वेब एप धेरै न्यूनतम छ, त्यसैले [ingredient_indexes](../../../../4-Classification/data/ingredient_indexes.csv) डाटाबाट सामग्रीहरू र तिनीहरूको सूचकांकहरू प्रयोग गरेर यसलाई निर्माण गर्न जारी राख्नुहोस्। कुन स्वाद संयोजनहरूले कुनै राष्ट्रिय परिकार निर्माण गर्न काम गर्छ?

## [पाठ पछि क्विज](https://ff-quizzes.netlify.app/en/ml/)

## समीक्षा र आत्म अध्ययन

यो पाठले खाना सामग्रीहरूको लागि सिफारिस प्रणाली निर्माणको उपयोगितालाई मात्र छोयो, तर मेसिन लर्निङ अनुप्रयोगहरूको यो क्षेत्र उदाहरणहरूमा धेरै धनी छ। यी प्रणालीहरू कसरी निर्माण गरिन्छन् भन्ने बारे थप पढ्नुहोस्:

- https://www.sciencedirect.com/topics/computer-science/recommendation-engine
- https://www.technologyreview.com/2014/08/25/171547/the-ultimate-challenge-for-recommendation-engines/
- https://www.technologyreview.com/2015/03/23/168831/everything-is-a-recommendation/

## असाइनमेन्ट 

[नयाँ सिफारिस प्रणाली निर्माण गर्नुहोस्](assignment.md)

---

**अस्वीकरण**:  
यो दस्तावेज़ AI अनुवाद सेवा [Co-op Translator](https://github.com/Azure/co-op-translator) प्रयोग गरेर अनुवाद गरिएको छ। हामी शुद्धताको लागि प्रयास गर्छौं, तर कृपया ध्यान दिनुहोस् कि स्वचालित अनुवादहरूमा त्रुटि वा अशुद्धता हुन सक्छ। यसको मूल भाषा मा रहेको मूल दस्तावेज़लाई आधिकारिक स्रोत मानिनुपर्छ। महत्वपूर्ण जानकारीको लागि, व्यावसायिक मानव अनुवाद सिफारिस गरिन्छ। यस अनुवादको प्रयोगबाट उत्पन्न हुने कुनै पनि गलतफहमी वा गलत व्याख्याको लागि हामी जिम्मेवार हुने छैनौं।