<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "e0b75f73e4a90d45181dc5581fe2ef5c",
  "translation_date": "2025-09-05T19:45:06+00:00",
  "source_file": "3-Web-App/1-Web-App/README.md",
  "language_code": "he"
}
-->
# בניית אפליקציית אינטרנט לשימוש במודל למידת מכונה

בשיעור הזה, תאמנו מודל למידת מכונה על סט נתונים יוצא דופן: _תצפיות עב"מים במאה האחרונה_, שנאספו ממאגר הנתונים של NUFORC.

תלמדו:

- איך 'לשמר' מודל מאומן
- איך להשתמש במודל הזה באפליקציית Flask

נמשיך להשתמש במחברות לניקוי נתונים ולאימון המודל שלנו, אבל תוכלו לקחת את התהליך צעד אחד קדימה על ידי חקר השימוש במודל "בשדה", כלומר: באפליקציית אינטרנט.

כדי לעשות זאת, תצטרכו לבנות אפליקציית אינטרנט באמצעות Flask.

## [שאלון לפני השיעור](https://ff-quizzes.netlify.app/en/ml/)

## בניית אפליקציה

ישנן מספר דרכים לבנות אפליקציות אינטרנט לצריכת מודלים של למידת מכונה. הארכיטקטורה של האינטרנט שלכם עשויה להשפיע על הדרך שבה המודל שלכם מאומן. דמיינו שאתם עובדים בעסק שבו קבוצת מדעני הנתונים אימנה מודל שהם רוצים שתשתמשו בו באפליקציה.

### שיקולים

ישנן שאלות רבות שעליכם לשאול:

- **האם זו אפליקציית אינטרנט או אפליקציה לנייד?** אם אתם בונים אפליקציה לנייד או צריכים להשתמש במודל בהקשר של IoT, תוכלו להשתמש ב-[TensorFlow Lite](https://www.tensorflow.org/lite/) ולהשתמש במודל באפליקציות אנדרואיד או iOS.
- **היכן המודל יימצא?** בענן או מקומית?
- **תמיכה לא מקוונת.** האם האפליקציה צריכה לעבוד במצב לא מקוון?
- **איזו טכנולוגיה שימשה לאימון המודל?** הטכנולוגיה שנבחרה עשויה להשפיע על הכלים שתצטרכו להשתמש בהם.
    - **שימוש ב-TensorFlow.** אם אתם מאמנים מודל באמצעות TensorFlow, למשל, האקוסיסטם הזה מספק את היכולת להמיר מודל TensorFlow לשימוש באפליקציית אינטרנט באמצעות [TensorFlow.js](https://www.tensorflow.org/js/).
    - **שימוש ב-PyTorch.** אם אתם בונים מודל באמצעות ספרייה כמו [PyTorch](https://pytorch.org/), יש לכם אפשרות לייצא אותו בפורמט [ONNX](https://onnx.ai/) (Open Neural Network Exchange) לשימוש באפליקציות אינטרנט JavaScript שיכולות להשתמש ב-[Onnx Runtime](https://www.onnxruntime.ai/). אפשרות זו תיחקר בשיעור עתידי עבור מודל שאומן באמצעות Scikit-learn.
    - **שימוש ב-Lobe.ai או Azure Custom Vision.** אם אתם משתמשים במערכת SaaS (תוכנה כשירות) ללמידת מכונה כמו [Lobe.ai](https://lobe.ai/) או [Azure Custom Vision](https://azure.microsoft.com/services/cognitive-services/custom-vision-service/?WT.mc_id=academic-77952-leestott) לאימון מודל, סוג זה של תוכנה מספק דרכים לייצא את המודל לפלטפורמות רבות, כולל בניית API מותאם אישית שניתן לשאול בענן על ידי האפליקציה המקוונת שלכם.

יש לכם גם את האפשרות לבנות אפליקציית אינטרנט שלמה ב-Flask שתוכל לאמן את המודל בעצמה בדפדפן אינטרנט. ניתן לעשות זאת גם באמצעות TensorFlow.js בהקשר של JavaScript.

למטרותינו, מכיוון שעבדנו עם מחברות מבוססות Python, בואו נחקור את השלבים שעליכם לבצע כדי לייצא מודל מאומן ממחברת כזו לפורמט שניתן לקריאה על ידי אפליקציית אינטרנט שנבנתה ב-Python.

## כלי

למשימה זו, תצטרכו שני כלים: Flask ו-Pickle, שניהם פועלים על Python.

✅ מהו [Flask](https://palletsprojects.com/p/flask/)? מוגדר כ'מיקרו-פריימוורק' על ידי יוצריו, Flask מספק את התכונות הבסיסיות של פריימוורקים לאינטרנט באמצעות Python ומנוע תבניות לבניית דפי אינטרנט. עיינו ב-[מודול הלמידה הזה](https://docs.microsoft.com/learn/modules/python-flask-build-ai-web-app?WT.mc_id=academic-77952-leestott) כדי לתרגל בנייה עם Flask.

✅ מהו [Pickle](https://docs.python.org/3/library/pickle.html)? Pickle 🥒 הוא מודול Python שמבצע סריאליזציה ודס-סריאליזציה של מבנה אובייקט ב-Python. כשאתם 'משמרים' מודל, אתם מבצעים סריאליזציה או משטחים את המבנה שלו לשימוש באינטרנט. שימו לב: Pickle אינו בטוח באופן אינהרנטי, אז היו זהירים אם תתבקשו 'לפרוק' קובץ. קובץ משומר מסומן בסיומת `.pkl`.

## תרגיל - ניקוי הנתונים שלכם

בשיעור הזה תשתמשו בנתונים מ-80,000 תצפיות עב"מים, שנאספו על ידי [NUFORC](https://nuforc.org) (המרכז הלאומי לדיווח על עב"מים). לנתונים האלה יש תיאורים מעניינים של תצפיות עב"מים, לדוגמה:

- **תיאור ארוך לדוגמה.** "אדם יוצא מקרן אור שמאירה על שדה דשא בלילה ורץ לכיוון מגרש החניה של Texas Instruments".
- **תיאור קצר לדוגמה.** "האורות רדפו אחרינו".

גיליון הנתונים [ufos.csv](../../../../3-Web-App/1-Web-App/data/ufos.csv) כולל עמודות על `עיר`, `מדינה` ו`ארץ` שבהן התצפית התרחשה, `צורה` של האובייקט ו`קו רוחב` ו`קו אורך`.

ב-[מחברת](../../../../3-Web-App/1-Web-App/notebook.ipynb) הריקה שמצורפת לשיעור הזה:

1. ייבאו את `pandas`, `matplotlib`, ו-`numpy` כפי שעשיתם בשיעורים קודמים וייבאו את גיליון הנתונים של עב"מים. תוכלו להסתכל על דוגמת סט נתונים:

    ```python
    import pandas as pd
    import numpy as np
    
    ufos = pd.read_csv('./data/ufos.csv')
    ufos.head()
    ```

1. המירו את נתוני העב"מים למסגרת נתונים קטנה עם כותרות חדשות. בדקו את הערכים הייחודיים בשדה `Country`.

    ```python
    ufos = pd.DataFrame({'Seconds': ufos['duration (seconds)'], 'Country': ufos['country'],'Latitude': ufos['latitude'],'Longitude': ufos['longitude']})
    
    ufos.Country.unique()
    ```

1. עכשיו, תוכלו לצמצם את כמות הנתונים שעלינו להתמודד איתם על ידי הסרת ערכים ריקים וייבוא תצפיות בין 1-60 שניות בלבד:

    ```python
    ufos.dropna(inplace=True)
    
    ufos = ufos[(ufos['Seconds'] >= 1) & (ufos['Seconds'] <= 60)]
    
    ufos.info()
    ```

1. ייבאו את ספריית `LabelEncoder` של Scikit-learn כדי להמיר את ערכי הטקסט של מדינות למספר:

    ✅ LabelEncoder מקודד נתונים לפי סדר אלפביתי

    ```python
    from sklearn.preprocessing import LabelEncoder
    
    ufos['Country'] = LabelEncoder().fit_transform(ufos['Country'])
    
    ufos.head()
    ```

    הנתונים שלכם צריכים להיראות כך:

    ```output
    	Seconds	Country	Latitude	Longitude
    2	20.0	3		53.200000	-2.916667
    3	20.0	4		28.978333	-96.645833
    14	30.0	4		35.823889	-80.253611
    23	60.0	4		45.582778	-122.352222
    24	3.0		3		51.783333	-0.783333
    ```

## תרגיל - בניית המודל שלכם

עכשיו תוכלו להתכונן לאמן מודל על ידי חלוקת הנתונים לקבוצת אימון ובדיקה.

1. בחרו את שלושת המאפיינים שתרצו לאמן עליהם כוקטור X שלכם, והוקטור y יהיה `Country`. אתם רוצים להיות מסוגלים להזין `Seconds`, `Latitude` ו-`Longitude` ולקבל מזהה מדינה להחזרה.

    ```python
    from sklearn.model_selection import train_test_split
    
    Selected_features = ['Seconds','Latitude','Longitude']
    
    X = ufos[Selected_features]
    y = ufos['Country']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    ```

1. אימנו את המודל שלכם באמצעות רגרסיה לוגיסטית:

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

הדיוק לא רע **(כ-95%)**, ולא מפתיע, מכיוון ש-`Country` ו-`Latitude/Longitude` מתואמים.

המודל שיצרתם אינו מאוד מהפכני מכיוון שאתם אמורים להיות מסוגלים להסיק `Country` מ-`Latitude` ו-`Longitude`, אבל זהו תרגיל טוב לנסות לאמן מנתונים גולמיים שניקיתם, ייצאתם, ואז להשתמש במודל הזה באפליקציית אינטרנט.

## תרגיל - 'שימור' המודל שלכם

עכשיו, הגיע הזמן _לשמר_ את המודל שלכם! תוכלו לעשות זאת בכמה שורות קוד. לאחר שהוא _משומר_, טענו את המודל המשומר ובדקו אותו מול מערך נתונים לדוגמה שמכיל ערכים עבור שניות, קו רוחב וקו אורך.

```python
import pickle
model_filename = 'ufo-model.pkl'
pickle.dump(model, open(model_filename,'wb'))

model = pickle.load(open('ufo-model.pkl','rb'))
print(model.predict([[50,44,-12]]))
```

המודל מחזיר **'3'**, שזהו קוד המדינה עבור בריטניה. מדהים! 👽

## תרגיל - בניית אפליקציית Flask

עכשיו תוכלו לבנות אפליקציית Flask שתוכל לקרוא את המודל שלכם ולהחזיר תוצאות דומות, אבל בצורה יותר נעימה לעין.

1. התחילו ביצירת תיקייה בשם **web-app** ליד קובץ _notebook.ipynb_ שבו נמצא קובץ _ufo-model.pkl_ שלכם.

1. בתיקייה הזו צרו עוד שלוש תיקיות: **static**, עם תיקייה **css** בתוכה, ו-**templates**. עכשיו אמורים להיות לכם הקבצים והתיקיות הבאים:

    ```output
    web-app/
      static/
        css/
      templates/
    notebook.ipynb
    ufo-model.pkl
    ```

    ✅ עיינו בתיקיית הפתרון כדי לראות את האפליקציה המוגמרת

1. הקובץ הראשון שיש ליצור בתיקיית _web-app_ הוא קובץ **requirements.txt**. כמו _package.json_ באפליקציית JavaScript, קובץ זה מפרט את התלויות הנדרשות על ידי האפליקציה. ב-**requirements.txt** הוסיפו את השורות:

    ```text
    scikit-learn
    pandas
    numpy
    flask
    ```

1. עכשיו, הריצו את הקובץ הזה על ידי ניווט ל-_web-app_:

    ```bash
    cd web-app
    ```

1. בטרמינל שלכם הקלידו `pip install`, כדי להתקין את הספריות המפורטות ב-_requirements.txt_:

    ```bash
    pip install -r requirements.txt
    ```

1. עכשיו, אתם מוכנים ליצור עוד שלושה קבצים כדי לסיים את האפליקציה:

    1. צרו **app.py** בשורש.
    2. צרו **index.html** בתיקיית _templates_.
    3. צרו **styles.css** בתיקיית _static/css_.

1. בנו את קובץ _styles.css_ עם כמה סגנונות:

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

1. לאחר מכן, בנו את קובץ _index.html_:

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

    שימו לב לתבניות בקובץ הזה. שימו לב לסינטקס 'mustache' סביב משתנים שיסופקו על ידי האפליקציה, כמו טקסט התחזית: `{{}}`. יש גם טופס ששולח תחזית לנתיב `/predict`.

    לבסוף, אתם מוכנים לבנות את קובץ ה-Python שמניע את צריכת המודל והצגת התחזיות:

1. ב-`app.py` הוסיפו:

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

    > 💡 טיפ: כשאתם מוסיפים [`debug=True`](https://www.askpython.com/python-modules/flask/flask-debug-mode) בזמן הרצת אפליקציית האינטרנט באמצעות Flask, כל שינוי שתעשו באפליקציה שלכם ישתקף מיד ללא צורך להפעיל מחדש את השרת. שימו לב! אל תפעילו מצב זה באפליקציה בסביבת ייצור.

אם תריצו `python app.py` או `python3 app.py` - שרת האינטרנט שלכם יתחיל לפעול, מקומית, ותוכלו למלא טופס קצר כדי לקבל תשובה לשאלה הבוערת שלכם על היכן נצפו עב"מים!

לפני שתעשו זאת, הסתכלו על החלקים של `app.py`:

1. קודם כל, התלויות נטענות והאפליקציה מתחילה.
1. לאחר מכן, המודל מיובא.
1. לאחר מכן, index.html מוצג בנתיב הבית.

בנתיב `/predict`, מספר דברים קורים כשהטופס נשלח:

1. משתני הטופס נאספים ומומרים למערך numpy. הם נשלחים למודל ותחזית מוחזרת.
2. המדינות שאנחנו רוצים להציג מוצגות מחדש כטקסט קריא מקוד המדינה החזוי שלהן, והערך הזה נשלח חזרה ל-index.html כדי להיות מוצג בתבנית.

שימוש במודל בדרך זו, עם Flask ומודל משומר, הוא יחסית פשוט. הדבר הקשה ביותר הוא להבין באיזו צורה הנתונים צריכים להיות כדי להישלח למודל ולקבל תחזית. זה תלוי לחלוטין באיך המודל אומן. למודל הזה יש שלוש נקודות נתונים שצריך להזין כדי לקבל תחזית.

בסביבה מקצועית, תוכלו לראות עד כמה תקשורת טובה היא הכרחית בין האנשים שמאמנים את המודל לבין אלה שצורכים אותו באפליקציית אינטרנט או נייד. במקרה שלנו, זה רק אדם אחד, אתם!

---

## 🚀 אתגר

במקום לעבוד במחברת ולייבא את המודל לאפליקציית Flask, תוכלו לאמן את המודל ממש בתוך אפליקציית Flask! נסו להמיר את קוד ה-Python במחברת, אולי לאחר ניקוי הנתונים שלכם, כדי לאמן את המודל מתוך האפליקציה בנתיב שנקרא `train`. מה היתרונות והחסרונות של שיטה זו?

## [שאלון אחרי השיעור](https://ff-quizzes.netlify.app/en/ml/)

## סקירה ולימוד עצמי

ישנן דרכים רבות לבנות אפליקציית אינטרנט לצריכת מודלים של למידת מכונה. הכינו רשימה של הדרכים שבהן תוכלו להשתמש ב-JavaScript או Python כדי לבנות אפליקציית אינטרנט שתנצל למידת מכונה. שקלו ארכיטקטורה: האם המודל צריך להישאר באפליקציה או לחיות בענן? אם האפשרות השנייה, איך הייתם ניגשים אליו? ציירו מודל ארכיטקטוני לפתרון אינטרנטי של למידת מכונה.

## משימה

[נסו מודל אחר](assignment.md)

---

**כתב ויתור**:  
מסמך זה תורגם באמצעות שירות תרגום מבוסס בינה מלאכותית [Co-op Translator](https://github.com/Azure/co-op-translator). למרות שאנו שואפים לדיוק, יש לקחת בחשבון שתרגומים אוטומטיים עשויים להכיל שגיאות או אי דיוקים. המסמך המקורי בשפתו המקורית צריך להיחשב כמקור סמכותי. עבור מידע קריטי, מומלץ להשתמש בתרגום מקצועי על ידי אדם. איננו נושאים באחריות לאי הבנות או לפרשנויות שגויות הנובעות משימוש בתרגום זה.