<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "49047911108adc49d605cddfb455749c",
  "translation_date": "2025-09-05T19:56:10+00:00",
  "source_file": "4-Classification/3-Classifiers-2/README.md",
  "language_code": "he"
}
-->
# מסווגי מטבחים 2

בשיעור הסיווג השני הזה, תחקור דרכים נוספות לסווג נתונים מספריים. בנוסף, תלמד על ההשלכות של בחירת מסווג אחד על פני אחר.

## [מבחן מקדים להרצאה](https://ff-quizzes.netlify.app/en/ml/)

### דרישות מוקדמות

אנו מניחים שסיימת את השיעורים הקודמים ויש לך מערך נתונים מנוקה בתיקיית `data` בשם _cleaned_cuisines.csv_ שנמצא בשורש תיקיית ארבעת השיעורים.

### הכנה

טענו את קובץ _notebook.ipynb_ שלך עם מערך הנתונים המנוקה וחילקנו אותו למסגרות נתונים X ו-y, מוכנות לתהליך בניית המודל.

## מפת סיווג

בשיעור הקודם, למדת על האפשרויות השונות שיש לך בעת סיווג נתונים באמצעות דף העזר של Microsoft. Scikit-learn מציעה דף עזר דומה אך מפורט יותר שיכול לעזור לצמצם את הבחירה במעריכים (מונח נוסף למסווגים):

![מפת ML מ-Scikit-learn](../../../../4-Classification/3-Classifiers-2/images/map.png)
> טיפ: [בקר במפה הזו אונליין](https://scikit-learn.org/stable/tutorial/machine_learning_map/) ולחץ לאורך המסלול כדי לקרוא את התיעוד.

### התוכנית

המפה הזו מאוד מועילה ברגע שיש לך הבנה ברורה של הנתונים שלך, שכן ניתן 'ללכת' לאורך המסלולים שלה כדי להגיע להחלטה:

- יש לנו >50 דגימות
- אנחנו רוצים לחזות קטגוריה
- יש לנו נתונים מתויגים
- יש לנו פחות מ-100K דגימות
- ✨ אנחנו יכולים לבחור ב-Linear SVC
- אם זה לא עובד, מכיוון שיש לנו נתונים מספריים
    - אנחנו יכולים לנסות ✨ KNeighbors Classifier 
      - אם זה לא עובד, לנסות ✨ SVC ו-✨ Ensemble Classifiers

זהו מסלול מאוד מועיל לעקוב אחריו.

## תרגיל - חלוקת הנתונים

בהתאם למסלול הזה, כדאי להתחיל בייבוא כמה ספריות לשימוש.

1. ייבא את הספריות הנדרשות:

    ```python
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import accuracy_score,precision_score,confusion_matrix,classification_report, precision_recall_curve
    import numpy as np
    ```

1. חלק את נתוני האימון והבדיקה שלך:

    ```python
    X_train, X_test, y_train, y_test = train_test_split(cuisines_feature_df, cuisines_label_df, test_size=0.3)
    ```

## מסווג Linear SVC

סיווג באמצעות Support-Vector (SVC) הוא חלק ממשפחת טכניקות ה-ML של Support-Vector Machines (למידע נוסף על אלו למטה). בשיטה זו, ניתן לבחור 'גרעין' כדי להחליט כיצד לקבץ את התוויות. הפרמטר 'C' מתייחס ל'רגולריזציה' שמווסתת את השפעת הפרמטרים. הגרעין יכול להיות אחד מ-[כמה](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC); כאן אנו מגדירים אותו כ'לינארי' כדי להבטיח שנשתמש ב-Linear SVC. ברירת המחדל של הסתברות היא 'false'; כאן אנו מגדירים אותה כ'true' כדי לקבל הערכות הסתברות. אנו מגדירים את מצב האקראיות כ-'0' כדי לערבב את הנתונים ולקבל הסתברויות.

### תרגיל - יישום Linear SVC

התחל ביצירת מערך מסווגים. תוסיף בהדרגה למערך הזה ככל שנבדוק.

1. התחל עם Linear SVC:

    ```python
    C = 10
    # Create different classifiers.
    classifiers = {
        'Linear SVC': SVC(kernel='linear', C=C, probability=True,random_state=0)
    }
    ```

2. אמן את המודל שלך באמצעות Linear SVC והדפס דוח:

    ```python
    n_classifiers = len(classifiers)
    
    for index, (name, classifier) in enumerate(classifiers.items()):
        classifier.fit(X_train, np.ravel(y_train))
    
        y_pred = classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print("Accuracy (train) for %s: %0.1f%% " % (name, accuracy * 100))
        print(classification_report(y_test,y_pred))
    ```

    התוצאה די טובה:

    ```output
    Accuracy (train) for Linear SVC: 78.6% 
                  precision    recall  f1-score   support
    
         chinese       0.71      0.67      0.69       242
          indian       0.88      0.86      0.87       234
        japanese       0.79      0.74      0.76       254
          korean       0.85      0.81      0.83       242
            thai       0.71      0.86      0.78       227
    
        accuracy                           0.79      1199
       macro avg       0.79      0.79      0.79      1199
    weighted avg       0.79      0.79      0.79      1199
    ```

## מסווג K-Neighbors

K-Neighbors הוא חלק ממשפחת שיטות ה-ML של "שכנים", שניתן להשתמש בהן ללמידה מונחית ולא מונחית. בשיטה זו, נוצר מספר מוגדר מראש של נקודות, והנתונים נאספים סביב נקודות אלו כך שניתן לחזות תוויות כלליות עבור הנתונים.

### תרגיל - יישום מסווג K-Neighbors

המסווג הקודם היה טוב ועבד היטב עם הנתונים, אבל אולי נוכל להשיג דיוק טוב יותר. נסה מסווג K-Neighbors.

1. הוסף שורה למערך המסווגים שלך (הוסף פסיק אחרי הפריט של Linear SVC):

    ```python
    'KNN classifier': KNeighborsClassifier(C),
    ```

    התוצאה קצת פחות טובה:

    ```output
    Accuracy (train) for KNN classifier: 73.8% 
                  precision    recall  f1-score   support
    
         chinese       0.64      0.67      0.66       242
          indian       0.86      0.78      0.82       234
        japanese       0.66      0.83      0.74       254
          korean       0.94      0.58      0.72       242
            thai       0.71      0.82      0.76       227
    
        accuracy                           0.74      1199
       macro avg       0.76      0.74      0.74      1199
    weighted avg       0.76      0.74      0.74      1199
    ```

    ✅ למד על [K-Neighbors](https://scikit-learn.org/stable/modules/neighbors.html#neighbors)

## מסווג Support Vector

מסווגי Support-Vector הם חלק ממשפחת [Support-Vector Machine](https://wikipedia.org/wiki/Support-vector_machine) של שיטות ML המשמשות למשימות סיווג ורגרסיה. SVMs "ממפים דוגמאות אימון לנקודות במרחב" כדי למקסם את המרחק בין שתי קטגוריות. נתונים עוקבים ממופים למרחב הזה כך שניתן לחזות את הקטגוריה שלהם.

### תרגיל - יישום מסווג Support Vector

בואו ננסה להשיג דיוק קצת יותר טוב עם מסווג Support Vector.

1. הוסף פסיק אחרי הפריט של K-Neighbors, ואז הוסף את השורה הזו:

    ```python
    'SVC': SVC(),
    ```

    התוצאה די טובה!

    ```output
    Accuracy (train) for SVC: 83.2% 
                  precision    recall  f1-score   support
    
         chinese       0.79      0.74      0.76       242
          indian       0.88      0.90      0.89       234
        japanese       0.87      0.81      0.84       254
          korean       0.91      0.82      0.86       242
            thai       0.74      0.90      0.81       227
    
        accuracy                           0.83      1199
       macro avg       0.84      0.83      0.83      1199
    weighted avg       0.84      0.83      0.83      1199
    ```

    ✅ למד על [Support-Vectors](https://scikit-learn.org/stable/modules/svm.html#svm)

## מסווגי Ensemble

בואו נעקוב אחרי המסלול עד הסוף, למרות שהבדיקה הקודמת הייתה די טובה. ננסה כמה מסווגי 'Ensemble', במיוחד Random Forest ו-AdaBoost:

```python
  'RFST': RandomForestClassifier(n_estimators=100),
  'ADA': AdaBoostClassifier(n_estimators=100)
```

התוצאה מאוד טובה, במיוחד עבור Random Forest:

```output
Accuracy (train) for RFST: 84.5% 
              precision    recall  f1-score   support

     chinese       0.80      0.77      0.78       242
      indian       0.89      0.92      0.90       234
    japanese       0.86      0.84      0.85       254
      korean       0.88      0.83      0.85       242
        thai       0.80      0.87      0.83       227

    accuracy                           0.84      1199
   macro avg       0.85      0.85      0.84      1199
weighted avg       0.85      0.84      0.84      1199

Accuracy (train) for ADA: 72.4% 
              precision    recall  f1-score   support

     chinese       0.64      0.49      0.56       242
      indian       0.91      0.83      0.87       234
    japanese       0.68      0.69      0.69       254
      korean       0.73      0.79      0.76       242
        thai       0.67      0.83      0.74       227

    accuracy                           0.72      1199
   macro avg       0.73      0.73      0.72      1199
weighted avg       0.73      0.72      0.72      1199
```

✅ למד על [מסווגי Ensemble](https://scikit-learn.org/stable/modules/ensemble.html)

שיטה זו של למידת מכונה "משלבת את התחזיות של כמה מעריכים בסיסיים" כדי לשפר את איכות המודל. בדוגמה שלנו, השתמשנו ב-Random Trees ו-AdaBoost. 

- [Random Forest](https://scikit-learn.org/stable/modules/ensemble.html#forest), שיטה ממוצעת, בונה 'יער' של 'עצים החלטה' עם אקראיות כדי להימנע מהתאמת יתר. הפרמטר n_estimators מוגדר למספר העצים.

- [AdaBoost](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html) מתאים מסווג למערך נתונים ואז מתאים עותקים של אותו מסווג לאותו מערך נתונים. הוא מתמקד במשקל של פריטים שסווגו באופן שגוי ומכוונן את ההתאמה למסווג הבא כדי לתקן.

---

## 🚀אתגר

לכל אחת מהטכניקות הללו יש מספר רב של פרמטרים שניתן לכוונן. חקור את פרמטרי ברירת המחדל של כל אחת מהן וחשוב על מה משמעות כוונון הפרמטרים הללו עבור איכות המודל.

## [מבחן לאחר ההרצאה](https://ff-quizzes.netlify.app/en/ml/)

## סקירה ולימוד עצמי

יש הרבה מונחים מקצועיים בשיעורים האלה, אז קח רגע לעיין [ברשימה הזו](https://docs.microsoft.com/dotnet/machine-learning/resources/glossary?WT.mc_id=academic-77952-leestott) של מונחים שימושיים!

## משימה 

[משחק פרמטרים](assignment.md)

---

**כתב ויתור**:  
מסמך זה תורגם באמצעות שירות תרגום מבוסס בינה מלאכותית [Co-op Translator](https://github.com/Azure/co-op-translator). למרות שאנו שואפים לדיוק, יש לקחת בחשבון שתרגומים אוטומטיים עשויים להכיל שגיאות או אי דיוקים. המסמך המקורי בשפתו המקורית צריך להיחשב כמקור סמכותי. עבור מידע קריטי, מומלץ להשתמש בתרגום מקצועי על ידי אדם. איננו נושאים באחריות לאי הבנות או לפרשנויות שגויות הנובעות משימוש בתרגום זה.