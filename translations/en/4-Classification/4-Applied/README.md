<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "61bdec27ed2da8b098cd9065405d9bb0",
  "translation_date": "2025-09-06T10:56:37+00:00",
  "source_file": "4-Classification/4-Applied/README.md",
  "language_code": "en"
}
-->
# Build a Cuisine Recommender Web App

In this lesson, you will create a classification model using techniques learned in previous lessons and the delicious cuisine dataset used throughout this series. Additionally, you will develop a small web app to utilize a saved model, leveraging Onnx's web runtime.

Recommendation systems are one of the most practical applications of machine learning, and today you’ll take your first step in building one!

[![Presenting this web app](https://img.youtube.com/vi/17wdM9AHMfg/0.jpg)](https://youtu.be/17wdM9AHMfg "Applied ML")

> 🎥 Click the image above for a video: Jen Looper builds a web app using classified cuisine data

## [Pre-lecture quiz](https://ff-quizzes.netlify.app/en/ml/)

In this lesson, you will learn:

- How to build a model and save it as an Onnx model
- How to use Netron to inspect the model
- How to use your model in a web app for inference

## Build your model

Building applied ML systems is an essential part of integrating these technologies into business systems. By using Onnx, you can incorporate models into web applications, enabling offline usage if necessary.

In a [previous lesson](../../3-Web-App/1-Web-App/README.md), you created a regression model about UFO sightings, "pickled" it, and used it in a Flask app. While this architecture is valuable, it is a full-stack Python app, and your requirements might call for a JavaScript-based application.

In this lesson, you’ll build a basic JavaScript-based system for inference. First, you need to train a model and convert it for use with Onnx.

## Exercise - Train a Classification Model

Start by training a classification model using the cleaned cuisines dataset we’ve worked with before.

1. Begin by importing the necessary libraries:

    ```python
    !pip install skl2onnx
    import pandas as pd 
    ```

    You’ll need '[skl2onnx](https://onnx.ai/sklearn-onnx/)' to convert your Scikit-learn model to Onnx format.

2. Process your data as you did in previous lessons by reading a CSV file using `read_csv()`:

    ```python
    data = pd.read_csv('../data/cleaned_cuisines.csv')
    data.head()
    ```

3. Remove the first two unnecessary columns and save the remaining data as 'X':

    ```python
    X = data.iloc[:,2:]
    X.head()
    ```

4. Save the labels as 'y':

    ```python
    y = data[['cuisine']]
    y.head()
    
    ```

### Start the Training Process

We’ll use the 'SVC' library, which provides good accuracy.

1. Import the relevant libraries from Scikit-learn:

    ```python
    from sklearn.model_selection import train_test_split
    from sklearn.svm import SVC
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import accuracy_score,precision_score,confusion_matrix,classification_report
    ```

2. Split the data into training and test sets:

    ```python
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3)
    ```

3. Build an SVC classification model as you did in the previous lesson:

    ```python
    model = SVC(kernel='linear', C=10, probability=True,random_state=0)
    model.fit(X_train,y_train.values.ravel())
    ```

4. Test your model by calling `predict()`:

    ```python
    y_pred = model.predict(X_test)
    ```

5. Print a classification report to evaluate the model’s performance:

    ```python
    print(classification_report(y_test,y_pred))
    ```

    As seen before, the accuracy is strong:

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

### Convert Your Model to Onnx

Ensure the conversion uses the correct tensor number. This dataset includes 380 ingredients, so you’ll need to specify that number in `FloatTensorType`.

1. Convert the model using a tensor number of 380:

    ```python
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType
    
    initial_type = [('float_input', FloatTensorType([None, 380]))]
    options = {id(model): {'nocl': True, 'zipmap': False}}
    ```

2. Save the Onnx model as a file named **model.onnx**:

    ```python
    onx = convert_sklearn(model, initial_types=initial_type, options=options)
    with open("./model.onnx", "wb") as f:
        f.write(onx.SerializeToString())
    ```

    > Note: You can pass [options](https://onnx.ai/sklearn-onnx/parameterized.html) in your conversion script. In this case, we set 'nocl' to True and 'zipmap' to False. Since this is a classification model, you can remove ZipMap, which produces a list of dictionaries (not needed). `nocl` refers to class information being included in the model. Reduce the model’s size by setting `nocl` to 'True'.

Running the entire notebook will now create an Onnx model and save it in the current folder.

## View Your Model

Onnx models aren’t easily viewable in Visual Studio Code, but there’s excellent free software called [Netron](https://github.com/lutzroeder/Netron) that researchers use to visualize models. Open your model.onnx file in Netron to see your simple model, including its 380 inputs and classifier:

![Netron visual](../../../../4-Classification/4-Applied/images/netron.png)

Netron is a useful tool for inspecting models.

Now you’re ready to use this model in a web app. Let’s build an app to help you decide which cuisine you can prepare based on the leftover ingredients in your refrigerator, as determined by your model.

## Build a Recommender Web Application

You can use your model directly in a web app. This architecture allows you to run it locally and even offline if needed. Start by creating an `index.html` file in the same folder as your `model.onnx` file.

1. In the file _index.html_, add the following markup:

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

2. Within the `body` tags, add markup to display a list of checkboxes representing various ingredients:

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

    Each checkbox is assigned a value corresponding to the index of the ingredient in the dataset. For example, Apple occupies the fifth column in the alphabetic list, so its value is '4' (counting starts at 0). Refer to the [ingredients spreadsheet](../../../../4-Classification/data/ingredient_indexes.csv) to find an ingredient’s index.

    After the closing `</div>` tag, add a script block to call the model.

3. First, import the [Onnx Runtime](https://www.onnxruntime.ai/):

    ```html
    <script src="https://cdn.jsdelivr.net/npm/onnxruntime-web@1.9.0/dist/ort.min.js"></script> 
    ```

    > Onnx Runtime enables running Onnx models across various hardware platforms, offering optimizations and an API for usage.

4. Once the runtime is set up, call it:

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

In this code, several things happen:

1. An array of 380 possible values (1 or 0) is created to send to the model for inference, depending on whether an ingredient checkbox is checked.
2. An array of checkboxes is created, along with a way to determine whether they are checked, using an `init` function called when the app starts. When a checkbox is checked, the `ingredients` array is updated to reflect the selected ingredient.
3. A `testCheckboxes` function checks if any checkbox is selected.
4. The `startInference` function is triggered when the button is pressed. If any checkbox is checked, inference begins.
5. The inference routine includes:
   1. Setting up an asynchronous model load
   2. Creating a Tensor structure to send to the model
   3. Creating 'feeds' that match the `float_input` input created during model training (use Netron to verify the name)
   4. Sending these 'feeds' to the model and awaiting a response

## Test Your Application

Open a terminal in Visual Studio Code in the folder containing your index.html file. Ensure [http-server](https://www.npmjs.com/package/http-server) is installed globally, then type `http-server` at the prompt. A localhost will open, allowing you to view your web app. Check which cuisine is recommended based on selected ingredients:

![ingredient web app](../../../../4-Classification/4-Applied/images/web-app.png)

Congratulations! You’ve created a recommendation web app with a few fields. Take some time to expand this system.

## 🚀Challenge

Your web app is quite basic, so enhance it using ingredients and their indexes from the [ingredient_indexes](../../../../4-Classification/data/ingredient_indexes.csv) dataset. What flavor combinations work to create a specific national dish?

## [Post-lecture quiz](https://ff-quizzes.netlify.app/en/ml/)

## Review & Self Study

This lesson briefly introduced the concept of creating a recommendation system for food ingredients. This area of ML applications is rich with examples. Explore more about how these systems are built:

- https://www.sciencedirect.com/topics/computer-science/recommendation-engine
- https://www.technologyreview.com/2014/08/25/171547/the-ultimate-challenge-for-recommendation-engines/
- https://www.technologyreview.com/2015/03/23/168831/everything-is-a-recommendation/

## Assignment 

[Build a new recommender](assignment.md)

---

**Disclaimer**:  
This document has been translated using the AI translation service [Co-op Translator](https://github.com/Azure/co-op-translator). While we strive for accuracy, please note that automated translations may contain errors or inaccuracies. The original document in its native language should be regarded as the authoritative source. For critical information, professional human translation is recommended. We are not responsible for any misunderstandings or misinterpretations resulting from the use of this translation.