<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "aaf391d922bd6de5efba871d514c6d47",
  "translation_date": "2025-09-05T19:58:31+00:00",
  "source_file": "4-Classification/1-Introduction/README.md",
  "language_code": "he"
}
-->
# מבוא לסיווג

בארבעת השיעורים הללו, תחקור את אחד הנושאים המרכזיים בלמידת מכונה קלאסית - _סיווג_. נעבור יחד על שימוש באלגוריתמים שונים לסיווג עם מערך נתונים על כל המטבחים המדהימים של אסיה והודו. מקווים שאתה רעב!

![רק קורטוב!](../../../../4-Classification/1-Introduction/images/pinch.png)

> חוגגים את המטבחים הפאן-אסייתיים בשיעורים האלה! תמונה מאת [Jen Looper](https://twitter.com/jenlooper)

סיווג הוא סוג של [למידה מונחית](https://wikipedia.org/wiki/Supervised_learning) שיש לה הרבה מן המשותף עם טכניקות רגרסיה. אם למידת מכונה עוסקת בניבוי ערכים או שמות לדברים באמצעות מערכי נתונים, אז סיווג בדרך כלל מתחלק לשתי קבוצות: _סיווג בינארי_ ו_סיווג רב-קטגורי_.

[![מבוא לסיווג](https://img.youtube.com/vi/eg8DJYwdMyg/0.jpg)](https://youtu.be/eg8DJYwdMyg "מבוא לסיווג")

> 🎥 לחץ על התמונה למעלה לצפייה בסרטון: ג'ון גוטאג מ-MIT מציג את נושא הסיווג

זכור:

- **רגרסיה ליניארית** עזרה לך לנבא קשרים בין משתנים ולבצע תחזיות מדויקות על מיקום נקודת נתונים חדשה ביחס לקו. לדוגמה, יכולת לנבא _מה יהיה מחיר דלעת בספטמבר לעומת דצמבר_.
- **רגרסיה לוגיסטית** עזרה לך לגלות "קטגוריות בינאריות": בנקודת מחיר זו, _האם הדלעת כתומה או לא כתומה_?

סיווג משתמש באלגוריתמים שונים כדי לקבוע דרכים אחרות להגדיר את התווית או הקטגוריה של נקודת נתונים. בואו נעבוד עם נתוני המטבחים האלה כדי לראות האם, על ידי התבוננות בקבוצת מרכיבים, נוכל לקבוע את מקור המטבח.

## [שאלון לפני השיעור](https://ff-quizzes.netlify.app/en/ml/)

> ### [השיעור הזה זמין ב-R!](../../../../4-Classification/1-Introduction/solution/R/lesson_10.html)

### מבוא

סיווג הוא אחת הפעילויות המרכזיות של חוקרי למידת מכונה ומדעני נתונים. החל מסיווג בסיסי של ערך בינארי ("האם האימייל הזה הוא ספאם או לא?"), ועד לסיווג תמונות מורכב וחלוקה באמצעות ראייה ממוחשבת, תמיד מועיל להיות מסוגל למיין נתונים לקטגוריות ולשאול שאלות עליהם.

במונחים מדעיים יותר, שיטת הסיווג שלך יוצרת מודל חיזוי שמאפשר לך למפות את הקשר בין משתני קלט למשתני פלט.

![סיווג בינארי לעומת רב-קטגורי](../../../../4-Classification/1-Introduction/images/binary-multiclass.png)

> בעיות בינאריות לעומת רב-קטגוריות עבור אלגוריתמי סיווג. אינפוגרפיקה מאת [Jen Looper](https://twitter.com/jenlooper)

לפני שנתחיל בתהליך ניקוי הנתונים, ויזואליזציה שלהם והכנתם למשימות למידת המכונה שלנו, בואו נלמד מעט על הדרכים השונות שבהן ניתן להשתמש בלמידת מכונה כדי לסווג נתונים.

בהשראת [סטטיסטיקה](https://wikipedia.org/wiki/Statistical_classification), סיווג באמצעות למידת מכונה קלאסית משתמש בתכונות כמו `smoker`, `weight`, ו-`age` כדי לקבוע _סבירות לפתח מחלה X_. כטכניקת למידה מונחית הדומה לתרגילי הרגרסיה שביצעתם קודם לכן, הנתונים שלכם מתויגים והאלגוריתמים של למידת המכונה משתמשים בתוויות אלו כדי לסווג ולחזות קטגוריות (או 'תכונות') של מערך נתונים ולהקצות אותם לקבוצה או לתוצאה.

✅ הקדש רגע לדמיין מערך נתונים על מטבחים. מה מודל רב-קטגורי יוכל לענות עליו? מה מודל בינארי יוכל לענות עליו? מה אם היית רוצה לקבוע האם מטבח מסוים נוטה להשתמש בחילבה? מה אם היית רוצה לראות אם, בהתחשב בשקית מצרכים מלאה בכוכב אניס, ארטישוק, כרובית וחזרת, תוכל ליצור מנה הודית טיפוסית?

[![סלים מסתוריים משוגעים](https://img.youtube.com/vi/GuTeDbaNoEU/0.jpg)](https://youtu.be/GuTeDbaNoEU "סלים מסתוריים משוגעים")

> 🎥 לחץ על התמונה למעלה לצפייה בסרטון. כל הרעיון של התוכנית 'Chopped' הוא 'סל מסתורין' שבו שפים צריכים להכין מנה מתוך בחירה אקראית של מרכיבים. בטוח שמודל למידת מכונה היה עוזר!

## שלום 'מסווג'

השאלה שאנחנו רוצים לשאול על מערך הנתונים של המטבחים היא למעשה שאלה **רב-קטגורית**, מכיוון שיש לנו כמה מטבחים לאומיים פוטנציאליים לעבוד איתם. בהתחשב בקבוצת מרכיבים, לאיזו מהקטגוריות הרבות הנתונים יתאימו?

Scikit-learn מציעה מספר אלגוריתמים שונים לשימוש בסיווג נתונים, בהתאם לסוג הבעיה שברצונך לפתור. בשני השיעורים הבאים תלמד על כמה מהאלגוריתמים הללו.

## תרגיל - ניקוי ואיזון הנתונים שלך

המשימה הראשונה, לפני תחילת הפרויקט, היא לנקות ול**אזן** את הנתונים שלך כדי לקבל תוצאות טובות יותר. התחל עם הקובץ הריק _notebook.ipynb_ בתיקיית השורש של תיקייה זו.

הדבר הראשון להתקין הוא [imblearn](https://imbalanced-learn.org/stable/). זהו חבילת Scikit-learn שתאפשר לך לאזן את הנתונים בצורה טובה יותר (תלמד יותר על משימה זו בעוד רגע).

1. כדי להתקין `imblearn`, הרץ `pip install`, כך:

    ```python
    pip install imblearn
    ```

1. ייבא את החבילות שאתה צריך כדי לייבא את הנתונים שלך ולבצע ויזואליזציה, וגם ייבא `SMOTE` מ-`imblearn`.

    ```python
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    import numpy as np
    from imblearn.over_sampling import SMOTE
    ```

    עכשיו אתה מוכן לייבא את הנתונים.

1. המשימה הבאה תהיה לייבא את הנתונים:

    ```python
    df  = pd.read_csv('../data/cuisines.csv')
    ```

   שימוש ב-`read_csv()` יקרא את תוכן קובץ ה-csv _cusines.csv_ וימקם אותו במשתנה `df`.

1. בדוק את צורת הנתונים:

    ```python
    df.head()
    ```

   חמש השורות הראשונות נראות כך:

    ```output
    |     | Unnamed: 0 | cuisine | almond | angelica | anise | anise_seed | apple | apple_brandy | apricot | armagnac | ... | whiskey | white_bread | white_wine | whole_grain_wheat_flour | wine | wood | yam | yeast | yogurt | zucchini |
    | --- | ---------- | ------- | ------ | -------- | ----- | ---------- | ----- | ------------ | ------- | -------- | --- | ------- | ----------- | ---------- | ----------------------- | ---- | ---- | --- | ----- | ------ | -------- |
    | 0   | 65         | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
    | 1   | 66         | indian  | 1      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
    | 2   | 67         | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
    | 3   | 68         | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
    | 4   | 69         | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 1      | 0        |
    ```

1. קבל מידע על הנתונים האלה על ידי קריאה ל-`info()`:

    ```python
    df.info()
    ```

    הפלט שלך נראה כך:

    ```output
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 2448 entries, 0 to 2447
    Columns: 385 entries, Unnamed: 0 to zucchini
    dtypes: int64(384), object(1)
    memory usage: 7.2+ MB
    ```

## תרגיל - ללמוד על מטבחים

עכשיו העבודה מתחילה להיות מעניינת יותר. בואו נגלה את התפלגות הנתונים, לפי מטבח.

1. הצג את הנתונים כעמודות על ידי קריאה ל-`barh()`:

    ```python
    df.cuisine.value_counts().plot.barh()
    ```

    ![התפלגות נתוני מטבחים](../../../../4-Classification/1-Introduction/images/cuisine-dist.png)

    יש מספר סופי של מטבחים, אבל התפלגות הנתונים אינה אחידה. אתה יכול לתקן את זה! לפני כן, חקור קצת יותר.

1. גלה כמה נתונים זמינים לכל מטבח והדפס אותם:

    ```python
    thai_df = df[(df.cuisine == "thai")]
    japanese_df = df[(df.cuisine == "japanese")]
    chinese_df = df[(df.cuisine == "chinese")]
    indian_df = df[(df.cuisine == "indian")]
    korean_df = df[(df.cuisine == "korean")]
    
    print(f'thai df: {thai_df.shape}')
    print(f'japanese df: {japanese_df.shape}')
    print(f'chinese df: {chinese_df.shape}')
    print(f'indian df: {indian_df.shape}')
    print(f'korean df: {korean_df.shape}')
    ```

    הפלט נראה כך:

    ```output
    thai df: (289, 385)
    japanese df: (320, 385)
    chinese df: (442, 385)
    indian df: (598, 385)
    korean df: (799, 385)
    ```

## גילוי מרכיבים

עכשיו אתה יכול להעמיק בנתונים וללמוד מהם המרכיבים הטיפוסיים לכל מטבח. כדאי לנקות נתונים חוזרים שיוצרים בלבול בין מטבחים, אז בואו נלמד על הבעיה הזו.

1. צור פונקציה `create_ingredient()` ב-Python כדי ליצור מסגרת נתונים של מרכיבים. פונקציה זו תתחיל בהשמטת עמודה לא מועילה ותמיין מרכיבים לפי הספירה שלהם:

    ```python
    def create_ingredient_df(df):
        ingredient_df = df.T.drop(['cuisine','Unnamed: 0']).sum(axis=1).to_frame('value')
        ingredient_df = ingredient_df[(ingredient_df.T != 0).any()]
        ingredient_df = ingredient_df.sort_values(by='value', ascending=False,
        inplace=False)
        return ingredient_df
    ```

   עכשיו תוכל להשתמש בפונקציה הזו כדי לקבל מושג על עשרת המרכיבים הפופולריים ביותר לפי מטבח.

1. קרא ל-`create_ingredient()` והצג את הנתונים על ידי קריאה ל-`barh()`:

    ```python
    thai_ingredient_df = create_ingredient_df(thai_df)
    thai_ingredient_df.head(10).plot.barh()
    ```

    ![תאילנדי](../../../../4-Classification/1-Introduction/images/thai.png)

1. עשה את אותו הדבר עבור הנתונים היפניים:

    ```python
    japanese_ingredient_df = create_ingredient_df(japanese_df)
    japanese_ingredient_df.head(10).plot.barh()
    ```

    ![יפני](../../../../4-Classification/1-Introduction/images/japanese.png)

1. עכשיו עבור המרכיבים הסיניים:

    ```python
    chinese_ingredient_df = create_ingredient_df(chinese_df)
    chinese_ingredient_df.head(10).plot.barh()
    ```

    ![סיני](../../../../4-Classification/1-Introduction/images/chinese.png)

1. הצג את המרכיבים ההודיים:

    ```python
    indian_ingredient_df = create_ingredient_df(indian_df)
    indian_ingredient_df.head(10).plot.barh()
    ```

    ![הודי](../../../../4-Classification/1-Introduction/images/indian.png)

1. לבסוף, הצג את המרכיבים הקוריאניים:

    ```python
    korean_ingredient_df = create_ingredient_df(korean_df)
    korean_ingredient_df.head(10).plot.barh()
    ```

    ![קוריאני](../../../../4-Classification/1-Introduction/images/korean.png)

1. עכשיו, השמט את המרכיבים הנפוצים ביותר שיוצרים בלבול בין מטבחים שונים, על ידי קריאה ל-`drop()`:

   כולם אוהבים אורז, שום וג'ינג'ר!

    ```python
    feature_df= df.drop(['cuisine','Unnamed: 0','rice','garlic','ginger'], axis=1)
    labels_df = df.cuisine #.unique()
    feature_df.head()
    ```

## איזון מערך הנתונים

עכשיו, לאחר שניקית את הנתונים, השתמש ב-[SMOTE](https://imbalanced-learn.org/dev/references/generated/imblearn.over_sampling.SMOTE.html) - "טכניקת דגימה יתר סינתטית" - כדי לאזן אותם.

1. קרא ל-`fit_resample()`, אסטרטגיה זו יוצרת דגימות חדשות באמצעות אינטרפולציה.

    ```python
    oversample = SMOTE()
    transformed_feature_df, transformed_label_df = oversample.fit_resample(feature_df, labels_df)
    ```

    על ידי איזון הנתונים שלך, תקבל תוצאות טובות יותר בעת סיווגם. חשב על סיווג בינארי. אם רוב הנתונים שלך הם מקטגוריה אחת, מודל למידת מכונה הולך לנבא את הקטגוריה הזו בתדירות גבוהה יותר, פשוט כי יש יותר נתונים עבורה. איזון הנתונים לוקח נתונים מוטים ועוזר להסיר את חוסר האיזון הזה.

1. עכשיו תוכל לבדוק את מספרי התוויות לפי מרכיב:

    ```python
    print(f'new label count: {transformed_label_df.value_counts()}')
    print(f'old label count: {df.cuisine.value_counts()}')
    ```

    הפלט שלך נראה כך:

    ```output
    new label count: korean      799
    chinese     799
    indian      799
    japanese    799
    thai        799
    Name: cuisine, dtype: int64
    old label count: korean      799
    indian      598
    chinese     442
    japanese    320
    thai        289
    Name: cuisine, dtype: int64
    ```

    הנתונים נקיים, מאוזנים, ומאוד טעימים!

1. השלב האחרון הוא לשמור את הנתונים המאוזנים שלך, כולל תוויות ותכונות, למסגרת נתונים חדשה שניתן לייצא לקובץ:

    ```python
    transformed_df = pd.concat([transformed_label_df,transformed_feature_df],axis=1, join='outer')
    ```

1. תוכל להציץ שוב בנתונים באמצעות `transformed_df.head()` ו-`transformed_df.info()`. שמור עותק של הנתונים האלה לשימוש בשיעורים עתידיים:

    ```python
    transformed_df.head()
    transformed_df.info()
    transformed_df.to_csv("../data/cleaned_cuisines.csv")
    ```

    קובץ ה-CSV החדש הזה נמצא עכשיו בתיקיית הנתונים הראשית.

---

## 🚀אתגר

תוכנית הלימודים הזו מכילה כמה מערכי נתונים מעניינים. חפש בתיקיות `data` וראה אם יש מערכי נתונים שמתאימים לסיווג בינארי או רב-קטגורי? אילו שאלות היית שואל על מערך הנתונים הזה?

## [שאלון לאחר השיעור](https://ff-quizzes.netlify.app/en/ml/)

## סקירה ולימוד עצמי

חקור את ה-API של SMOTE. לאילו מקרי שימוש הוא מתאים ביותר? אילו בעיות הוא פותר?

## משימה 

[חקור שיטות סיווג](assignment.md)

---

**כתב ויתור**:  
מסמך זה תורגם באמצעות שירות תרגום מבוסס בינה מלאכותית [Co-op Translator](https://github.com/Azure/co-op-translator). למרות שאנו שואפים לדיוק, יש לקחת בחשבון שתרגומים אוטומטיים עשויים להכיל שגיאות או אי דיוקים. המסמך המקורי בשפתו המקורית צריך להיחשב כמקור סמכותי. עבור מידע קריטי, מומלץ להשתמש בתרגום מקצועי על ידי אדם. איננו נושאים באחריות לאי הבנות או לפרשנויות שגויות הנובעות משימוש בתרגום זה.