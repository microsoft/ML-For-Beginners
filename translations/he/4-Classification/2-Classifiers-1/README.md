<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "1a6e9e46b34a2e559fbbfc1f95397c7b",
  "translation_date": "2025-09-05T19:49:37+00:00",
  "source_file": "4-Classification/2-Classifiers-1/README.md",
  "language_code": "he"
}
-->
# מסווגי מטבחים 1

בשיעור הזה, תשתמשו במאגר הנתונים ששמרתם מהשיעור הקודם, שמלא בנתונים מאוזנים ונקיים על מטבחים.

תשתמשו במאגר הנתונים הזה עם מגוון מסווגים כדי _לחזות מטבח לאומי מסוים בהתבסס על קבוצת מרכיבים_. תוך כדי כך, תלמדו יותר על כמה מהדרכים שבהן ניתן להשתמש באלגוריתמים למשימות סיווג.

## [שאלון לפני השיעור](https://ff-quizzes.netlify.app/en/ml/)
# הכנה

בהנחה שסיימתם את [שיעור 1](../1-Introduction/README.md), ודאו שקובץ _cleaned_cuisines.csv_ נמצא בתיקיית השורש `/data` עבור ארבעת השיעורים הללו.

## תרגיל - חיזוי מטבח לאומי

1. בעבודה בתיקיית _notebook.ipynb_ של השיעור הזה, ייבאו את הקובץ יחד עם ספריית Pandas:

    ```python
    import pandas as pd
    cuisines_df = pd.read_csv("../data/cleaned_cuisines.csv")
    cuisines_df.head()
    ```

    הנתונים נראים כך:

|     | Unnamed: 0 | cuisine | almond | angelica | anise | anise_seed | apple | apple_brandy | apricot | armagnac | ... | whiskey | white_bread | white_wine | whole_grain_wheat_flour | wine | wood | yam | yeast | yogurt | zucchini |
| --- | ---------- | ------- | ------ | -------- | ----- | ---------- | ----- | ------------ | ------- | -------- | --- | ------- | ----------- | ---------- | ----------------------- | ---- | ---- | --- | ----- | ------ | -------- |
| 0   | 0          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 1   | 1          | indian  | 1      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 2   | 2          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 3   | 3          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 4   | 4          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 1      | 0        |
  

1. עכשיו, ייבאו עוד כמה ספריות:

    ```python
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import accuracy_score,precision_score,confusion_matrix,classification_report, precision_recall_curve
    from sklearn.svm import SVC
    import numpy as np
    ```

1. חלקו את הקואורדינטות X ו-y לשני מסגרות נתונים עבור אימון. `cuisine` יכולה להיות מסגרת הנתונים של התוויות:

    ```python
    cuisines_label_df = cuisines_df['cuisine']
    cuisines_label_df.head()
    ```

    זה ייראה כך:

    ```output
    0    indian
    1    indian
    2    indian
    3    indian
    4    indian
    Name: cuisine, dtype: object
    ```

1. הסירו את העמודה `Unnamed: 0` ואת העמודה `cuisine` באמצעות `drop()`. שמרו את שאר הנתונים כמאפיינים לאימון:

    ```python
    cuisines_feature_df = cuisines_df.drop(['Unnamed: 0', 'cuisine'], axis=1)
    cuisines_feature_df.head()
    ```

    המאפיינים שלכם ייראו כך:

|      | almond | angelica | anise | anise_seed | apple | apple_brandy | apricot | armagnac | artemisia | artichoke |  ... | whiskey | white_bread | white_wine | whole_grain_wheat_flour | wine | wood |  yam | yeast | yogurt | zucchini |
| ---: | -----: | -------: | ----: | ---------: | ----: | -----------: | ------: | -------: | --------: | --------: | ---: | ------: | ----------: | ---------: | ----------------------: | ---: | ---: | ---: | ----: | -----: | -------: |
|    0 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    1 |      1 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    2 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    3 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    4 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      1 |        0 | 0 |

עכשיו אתם מוכנים לאמן את המודל שלכם!

## בחירת המסווג

עכשיו כשהנתונים שלכם נקיים ומוכנים לאימון, עליכם להחליט איזה אלגוריתם להשתמש בו למשימה.

Scikit-learn מקטלגת סיווג תחת למידה מונחית, ובקטגוריה הזו תמצאו דרכים רבות לסווג. [המגוון](https://scikit-learn.org/stable/supervised_learning.html) עשוי להיות מבלבל במבט ראשון. השיטות הבאות כולן כוללות טכניקות סיווג:

- מודלים ליניאריים
- מכונות וקטור תמיכה
- ירידה גרדיאנטית סטוכסטית
- שכנים קרובים
- תהליכים גאוסיאניים
- עצי החלטה
- שיטות אנסמבל (מסווג הצבעה)
- אלגוריתמים רב-מחלקתיים ורב-תוצאות (סיווג רב-מחלקתי ורב-תוויתי, סיווג רב-מחלקתי-רב-תוצאות)

> ניתן גם להשתמש [ברשתות נוירונים לסיווג נתונים](https://scikit-learn.org/stable/modules/neural_networks_supervised.html#classification), אבל זה מחוץ לתחום השיעור הזה.

### איזה מסווג לבחור?

אז, איזה מסווג כדאי לבחור? לעיתים, מעבר על כמה מהם וחיפוש אחר תוצאה טובה היא דרך לבדוק. Scikit-learn מציעה [השוואה צד-לצד](https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html) על מאגר נתונים שנוצר, המשווה בין KNeighbors, SVC בשתי דרכים, GaussianProcessClassifier, DecisionTreeClassifier, RandomForestClassifier, MLPClassifier, AdaBoostClassifier, GaussianNB ו-QuadraticDiscrinationAnalysis, ומציגה את התוצאות בצורה חזותית:

![השוואת מסווגים](../../../../4-Classification/2-Classifiers-1/images/comparison.png)
> גרפים שנוצרו בתיעוד של Scikit-learn

> AutoML פותר את הבעיה הזו בצורה מסודרת על ידי הרצת ההשוואות הללו בענן, ומאפשר לכם לבחור את האלגוריתם הטוב ביותר עבור הנתונים שלכם. נסו זאת [כאן](https://docs.microsoft.com/learn/modules/automate-model-selection-with-azure-automl/?WT.mc_id=academic-77952-leestott)

### גישה טובה יותר

גישה טובה יותר מאשר לנחש באופן אקראי היא לעקוב אחר הרעיונות בדף העזר להורדה [ML Cheat sheet](https://docs.microsoft.com/azure/machine-learning/algorithm-cheat-sheet?WT.mc_id=academic-77952-leestott). כאן, אנו מגלים שבשביל הבעיה הרב-מחלקתית שלנו, יש לנו כמה אפשרויות:

![דף עזר לבעיות רב-מחלקתיות](../../../../4-Classification/2-Classifiers-1/images/cheatsheet.png)
> חלק מדף העזר של Microsoft לאלגוריתמים, המתאר אפשרויות סיווג רב-מחלקתיות

✅ הורידו את דף העזר הזה, הדפיסו אותו ותלו אותו על הקיר שלכם!

### נימוקים

בואו נראה אם נוכל לנמק את דרכנו דרך גישות שונות בהתחשב במגבלות שיש לנו:

- **רשתות נוירונים כבדות מדי**. בהתחשב במאגר הנתונים הנקי אך המינימלי שלנו, ובעובדה שאנחנו מריצים את האימון באופן מקומי דרך מחברות, רשתות נוירונים כבדות מדי למשימה הזו.
- **אין מסווג דו-מחלקתי**. אנחנו לא משתמשים במסווג דו-מחלקתי, כך שזה שולל את האפשרות של one-vs-all.
- **עץ החלטה או רגרסיה לוגיסטית יכולים לעבוד**. עץ החלטה עשוי לעבוד, או רגרסיה לוגיסטית עבור נתונים רב-מחלקתיים.
- **עצים החלטה מחוזקים רב-מחלקתיים פותרים בעיה אחרת**. עץ החלטה מחוזק רב-מחלקתי מתאים ביותר למשימות לא פרמטריות, למשל משימות שנועדו לבנות דירוגים, ולכן הוא לא שימושי עבורנו.

### שימוש ב-Scikit-learn 

נשתמש ב-Scikit-learn כדי לנתח את הנתונים שלנו. עם זאת, ישנן דרכים רבות להשתמש ברגרסיה לוגיסטית ב-Scikit-learn. הסתכלו על [הפרמטרים להעברה](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html?highlight=logistic%20regressio#sklearn.linear_model.LogisticRegression).  

בעיקרון, ישנם שני פרמטרים חשובים - `multi_class` ו-`solver` - שעלינו לציין, כאשר אנו מבקשים מ-Scikit-learn לבצע רגרסיה לוגיסטית. הערך של `multi_class` מיישם התנהגות מסוימת. הערך של `solver` הוא האלגוריתם לשימוש. לא כל הפותרים יכולים להיות משולבים עם כל ערכי `multi_class`.

לפי התיעוד, במקרה הרב-מחלקתי, אלגוריתם האימון:

- **משתמש בתוכנית one-vs-rest (OvR)**, אם אפשרות `multi_class` מוגדרת כ-`ovr`
- **משתמש באובדן קרוס-אנטרופי**, אם אפשרות `multi_class` מוגדרת כ-`multinomial`. (כרגע אפשרות `multinomial` נתמכת רק על ידי הפותרים ‘lbfgs’, ‘sag’, ‘saga’ ו-‘newton-cg’.)"

> 🎓 ה'תוכנית' כאן יכולה להיות 'ovr' (one-vs-rest) או 'multinomial'. מכיוון שרגרסיה לוגיסטית נועדה למעשה לתמוך בסיווג בינארי, תוכניות אלו מאפשרות לה להתמודד טוב יותר עם משימות סיווג רב-מחלקתיות. [מקור](https://machinelearningmastery.com/one-vs-rest-and-one-vs-one-for-multi-class-classification/)

> 🎓 ה'פותר' מוגדר כ"האלגוריתם לשימוש בבעיה האופטימיזציה". [מקור](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html?highlight=logistic%20regressio#sklearn.linear_model.LogisticRegression).

Scikit-learn מציעה את הטבלה הזו כדי להסביר כיצד פותרים מתמודדים עם אתגרים שונים שמוצגים על ידי מבני נתונים שונים:

![פותרים](../../../../4-Classification/2-Classifiers-1/images/solvers.png)

## תרגיל - פיצול הנתונים

נוכל להתמקד ברגרסיה לוגיסטית לניסוי האימון הראשון שלנו מכיוון שלמדתם עליה לאחרונה בשיעור קודם.
פצלו את הנתונים שלכם לקבוצות אימון ובדיקה על ידי קריאה ל-`train_test_split()`:

```python
X_train, X_test, y_train, y_test = train_test_split(cuisines_feature_df, cuisines_label_df, test_size=0.3)
```

## תרגיל - יישום רגרסיה לוגיסטית

מכיוון שאתם משתמשים במקרה הרב-מחלקתי, עליכם לבחור איזו _תוכנית_ להשתמש ואיזה _פותר_ להגדיר. השתמשו ב-LogisticRegression עם הגדרה רב-מחלקתית והפותר **liblinear** לאימון.

1. צרו רגרסיה לוגיסטית עם multi_class מוגדר כ-`ovr` והפותר מוגדר כ-`liblinear`:

    ```python
    lr = LogisticRegression(multi_class='ovr',solver='liblinear')
    model = lr.fit(X_train, np.ravel(y_train))
    
    accuracy = model.score(X_test, y_test)
    print ("Accuracy is {}".format(accuracy))
    ```

    ✅ נסו פותר אחר כמו `lbfgs`, שלרוב מוגדר כברירת מחדל
> שים לב, השתמש בפונקציה [`ravel`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.ravel.html) של Pandas כדי לשטח את הנתונים שלך בעת הצורך.
הדיוק טוב ביותר מ-**80%**!

1. ניתן לראות את המודל בפעולה על ידי בדיקת שורה אחת של נתונים (#50):

    ```python
    print(f'ingredients: {X_test.iloc[50][X_test.iloc[50]!=0].keys()}')
    print(f'cuisine: {y_test.iloc[50]}')
    ```

    התוצאה מודפסת:

   ```output
   ingredients: Index(['cilantro', 'onion', 'pea', 'potato', 'tomato', 'vegetable_oil'], dtype='object')
   cuisine: indian
   ```

   ✅ נסו מספר שורה שונה ובדקו את התוצאות

1. אם נעמיק, ניתן לבדוק את דיוק התחזית הזו:

    ```python
    test= X_test.iloc[50].values.reshape(-1, 1).T
    proba = model.predict_proba(test)
    classes = model.classes_
    resultdf = pd.DataFrame(data=proba, columns=classes)
    
    topPrediction = resultdf.T.sort_values(by=[0], ascending = [False])
    topPrediction.head()
    ```

    התוצאה מודפסת - המטבח ההודי הוא הניחוש הטוב ביותר, עם הסתברות גבוהה:

    |          |        0 |
    | -------: | -------: |
    |   indian | 0.715851 |
    |  chinese | 0.229475 |
    | japanese | 0.029763 |
    |   korean | 0.017277 |
    |     thai | 0.007634 |

    ✅ האם תוכלו להסביר מדוע המודל די בטוח שזהו מטבח הודי?

1. קבלו פרטים נוספים על ידי הדפסת דוח סיווג, כפי שעשיתם בשיעורי הרגרסיה:

    ```python
    y_pred = model.predict(X_test)
    print(classification_report(y_test,y_pred))
    ```

    |              | precision | recall | f1-score | support |
    | ------------ | --------- | ------ | -------- | ------- |
    | chinese      | 0.73      | 0.71   | 0.72     | 229     |
    | indian       | 0.91      | 0.93   | 0.92     | 254     |
    | japanese     | 0.70      | 0.75   | 0.72     | 220     |
    | korean       | 0.86      | 0.76   | 0.81     | 242     |
    | thai         | 0.79      | 0.85   | 0.82     | 254     |
    | accuracy     | 0.80      | 1199   |          |         |
    | macro avg    | 0.80      | 0.80   | 0.80     | 1199    |
    | weighted avg | 0.80      | 0.80   | 0.80     | 1199    |

## 🚀אתגר

בשיעור זה השתמשתם בנתונים הנקיים שלכם כדי לבנות מודל למידת מכונה שיכול לחזות מטבח לאומי בהתבסס על סדרת מרכיבים. הקדישו זמן לקרוא על האפשרויות הרבות ש-Scikit-learn מציעה לסיווג נתונים. העמיקו במושג 'solver' כדי להבין מה מתרחש מאחורי הקלעים.

## [שאלון לאחר השיעור](https://ff-quizzes.netlify.app/en/ml/)

## סקירה ולימוד עצמי

העמיקו מעט במתמטיקה שמאחורי רגרסיה לוגיסטית ב[שיעור הזה](https://people.eecs.berkeley.edu/~russell/classes/cs194/f11/lectures/CS194%20Fall%202011%20Lecture%2006.pdf)
## משימה 

[לימדו על ה-solvers](assignment.md)

---

**כתב ויתור**:  
מסמך זה תורגם באמצעות שירות תרגום מבוסס בינה מלאכותית [Co-op Translator](https://github.com/Azure/co-op-translator). למרות שאנו שואפים לדיוק, יש לקחת בחשבון שתרגומים אוטומטיים עשויים להכיל שגיאות או אי דיוקים. המסמך המקורי בשפתו המקורית צריך להיחשב כמקור סמכותי. עבור מידע קריטי, מומלץ להשתמש בתרגום מקצועי על ידי אדם. איננו נושאים באחריות לאי הבנות או לפרשנויות שגויות הנובעות משימוש בתרגום זה.