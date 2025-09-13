<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "482bccabe1df958496ea71a3667995cd",
  "translation_date": "2025-09-05T19:07:29+00:00",
  "source_file": "7-TimeSeries/3-SVR/README.md",
  "language_code": "he"
}
-->
# חיזוי סדרות זמן עם Support Vector Regressor

בשיעור הקודם, למדת כיצד להשתמש במודל ARIMA כדי לבצע תחזיות של סדרות זמן. עכשיו תכיר את מודל Support Vector Regressor, שהוא מודל רגרסיה המשמש לחיזוי נתונים רציפים.

## [מבחן מקדים](https://ff-quizzes.netlify.app/en/ml/) 

## מבוא

בשיעור זה, תגלו דרך ספציפית לבנות מודלים עם [**SVM**: **S**upport **V**ector **M**achine](https://en.wikipedia.org/wiki/Support-vector_machine) עבור רגרסיה, או **SVR: Support Vector Regressor**. 

### SVR בהקשר של סדרות זמן [^1]

לפני שנבין את החשיבות של SVR בחיזוי סדרות זמן, הנה כמה מושגים חשובים שעליכם להכיר:

- **רגרסיה:** טכניקת למידה מונחית לחיזוי ערכים רציפים מתוך קבוצת נתונים נתונה. הרעיון הוא להתאים עקומה (או קו) במרחב התכונות שיש לה את המספר המרבי של נקודות נתונים. [לחצו כאן](https://en.wikipedia.org/wiki/Regression_analysis) למידע נוסף.
- **Support Vector Machine (SVM):** סוג של מודל למידת מכונה מונחית המשמש לסיווג, רגרסיה וזיהוי חריגות. המודל הוא היפר-מישור במרחב התכונות, שבמקרה של סיווג משמש כגבול, ובמקרה של רגרסיה משמש כקו ההתאמה הטוב ביותר. ב-SVM, פונקציית Kernel משמשת בדרך כלל כדי להפוך את קבוצת הנתונים למרחב בעל מספר ממדים גבוה יותר, כך שניתן להפריד אותם בקלות. [לחצו כאן](https://en.wikipedia.org/wiki/Support-vector_machine) למידע נוסף על SVMs.
- **Support Vector Regressor (SVR):** סוג של SVM, שמטרתו למצוא את קו ההתאמה הטוב ביותר (שבמקרה של SVM הוא היפר-מישור) שיש לו את המספר המרבי של נקודות נתונים.

### למה SVR? [^1]

בשיעור הקודם למדתם על ARIMA, שהוא שיטה סטטיסטית ליניארית מוצלחת לחיזוי נתוני סדרות זמן. עם זאת, במקרים רבים, נתוני סדרות זמן מכילים *אי-ליניאריות*, שלא ניתן למפות באמצעות מודלים ליניאריים. במקרים כאלה, היכולת של SVM להתחשב באי-ליניאריות בנתונים עבור משימות רגרסיה הופכת את SVR למוצלח בחיזוי סדרות זמן.

## תרגיל - בניית מודל SVR

השלבים הראשונים להכנת הנתונים זהים לאלה של השיעור הקודם על [ARIMA](https://github.com/microsoft/ML-For-Beginners/tree/main/7-TimeSeries/2-ARIMA). 

פתחו את [_/working_](https://github.com/microsoft/ML-For-Beginners/tree/main/7-TimeSeries/3-SVR/working) בתיקייה של שיעור זה ומצאו את הקובץ [_notebook.ipynb_](https://github.com/microsoft/ML-For-Beginners/blob/main/7-TimeSeries/3-SVR/working/notebook.ipynb).[^2]

1. הריצו את המחברת וייבאו את הספריות הנדרשות:  [^2]

   ```python
   import sys
   sys.path.append('../../')
   ```

   ```python
   import os
   import warnings
   import matplotlib.pyplot as plt
   import numpy as np
   import pandas as pd
   import datetime as dt
   import math
   
   from sklearn.svm import SVR
   from sklearn.preprocessing import MinMaxScaler
   from common.utils import load_data, mape
   ```

2. טענו את הנתונים מתוך הקובץ `/data/energy.csv` לתוך DataFrame של Pandas והסתכלו עליהם:  [^2]

   ```python
   energy = load_data('../../data')[['load']]
   ```

3. צרו גרף של כל נתוני האנרגיה הזמינים מינואר 2012 עד דצמבר 2014: [^2]

   ```python
   energy.plot(y='load', subplots=True, figsize=(15, 8), fontsize=12)
   plt.xlabel('timestamp', fontsize=12)
   plt.ylabel('load', fontsize=12)
   plt.show()
   ```

   ![נתונים מלאים](../../../../7-TimeSeries/3-SVR/images/full-data.png)

   עכשיו, בואו נבנה את מודל ה-SVR שלנו.

### יצירת קבוצות אימון ובדיקה

עכשיו הנתונים שלכם טעונים, כך שתוכלו להפריד אותם לקבוצות אימון ובדיקה. לאחר מכן תעצבו מחדש את הנתונים כדי ליצור קבוצת נתונים מבוססת צעדי זמן, שתידרש עבור SVR. תאמנו את המודל שלכם על קבוצת האימון. לאחר שהמודל סיים את האימון, תעריכו את דיוקו על קבוצת האימון, קבוצת הבדיקה ולאחר מכן על כל קבוצת הנתונים כדי לראות את הביצועים הכוללים. עליכם לוודא שקבוצת הבדיקה מכסה תקופה מאוחרת יותר בזמן מקבוצת האימון כדי להבטיח שהמודל לא יקבל מידע מתקופות זמן עתידיות [^2] (מצב המכונה *Overfitting*).

1. הקצו תקופה של חודשיים מה-1 בספטמבר עד ה-31 באוקטובר 2014 לקבוצת האימון. קבוצת הבדיקה תכלול את התקופה של חודשיים מה-1 בנובמבר עד ה-31 בדצמבר 2014: [^2]

   ```python
   train_start_dt = '2014-11-01 00:00:00'
   test_start_dt = '2014-12-30 00:00:00'
   ```

2. הציגו את ההבדלים: [^2]

   ```python
   energy[(energy.index < test_start_dt) & (energy.index >= train_start_dt)][['load']].rename(columns={'load':'train'}) \
       .join(energy[test_start_dt:][['load']].rename(columns={'load':'test'}), how='outer') \
       .plot(y=['train', 'test'], figsize=(15, 8), fontsize=12)
   plt.xlabel('timestamp', fontsize=12)
   plt.ylabel('load', fontsize=12)
   plt.show()
   ```

   ![נתוני אימון ובדיקה](../../../../7-TimeSeries/3-SVR/images/train-test.png)



### הכנת הנתונים לאימון

עכשיו, עליכם להכין את הנתונים לאימון על ידי ביצוע סינון וסקיילינג של הנתונים שלכם. סננו את קבוצת הנתונים כך שתכלול רק את התקופות והעמודות הדרושות, וסקיילינג כדי להבטיח שהנתונים יוקרנו בטווח 0,1.

1. סננו את קבוצת הנתונים המקורית כך שתכלול רק את התקופות שהוזכרו לכל קבוצה ותכלול רק את העמודה הדרושה 'load' בנוסף לתאריך: [^2]

   ```python
   train = energy.copy()[(energy.index >= train_start_dt) & (energy.index < test_start_dt)][['load']]
   test = energy.copy()[energy.index >= test_start_dt][['load']]
   
   print('Training data shape: ', train.shape)
   print('Test data shape: ', test.shape)
   ```

   ```output
   Training data shape:  (1416, 1)
   Test data shape:  (48, 1)
   ```
   
2. בצעו סקיילינג לנתוני האימון כך שיהיו בטווח (0, 1): [^2]

   ```python
   scaler = MinMaxScaler()
   train['load'] = scaler.fit_transform(train)
   ```
   
4. עכשיו, בצעו סקיילינג לנתוני הבדיקה: [^2]

   ```python
   test['load'] = scaler.transform(test)
   ```

### יצירת נתונים עם צעדי זמן [^1]

עבור SVR, אתם ממירים את נתוני הקלט לצורה `[batch, timesteps]`. לכן, תעצבו מחדש את `train_data` ו-`test_data` כך שתהיה ממד חדש שמתייחס לצעדי הזמן. 

```python
# Converting to numpy arrays
train_data = train.values
test_data = test.values
```

לדוגמה זו, ניקח `timesteps = 5`. כך שהקלטים למודל הם הנתונים עבור 4 צעדי הזמן הראשונים, והפלט יהיה הנתונים עבור צעד הזמן החמישי.

```python
timesteps=5
```

המרת נתוני האימון לטנסור דו-ממדי באמצעות list comprehension מקונן:

```python
train_data_timesteps=np.array([[j for j in train_data[i:i+timesteps]] for i in range(0,len(train_data)-timesteps+1)])[:,:,0]
train_data_timesteps.shape
```

```output
(1412, 5)
```

המרת נתוני הבדיקה לטנסור דו-ממדי:

```python
test_data_timesteps=np.array([[j for j in test_data[i:i+timesteps]] for i in range(0,len(test_data)-timesteps+1)])[:,:,0]
test_data_timesteps.shape
```

```output
(44, 5)
```

בחירת קלטים ופלטים מנתוני האימון והבדיקה:

```python
x_train, y_train = train_data_timesteps[:,:timesteps-1],train_data_timesteps[:,[timesteps-1]]
x_test, y_test = test_data_timesteps[:,:timesteps-1],test_data_timesteps[:,[timesteps-1]]

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)
```

```output
(1412, 4) (1412, 1)
(44, 4) (44, 1)
```

### יישום SVR [^1]

עכשיו, הגיע הזמן ליישם SVR. לקריאה נוספת על יישום זה, תוכלו לעיין ב-[תיעוד הזה](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html). עבור היישום שלנו, נבצע את השלבים הבאים:

  1. הגדירו את המודל על ידי קריאה ל-`SVR()` והעברת היפר-פרמטרים של המודל: kernel, gamma, c ו-epsilon
  2. הכינו את המודל לנתוני האימון על ידי קריאה לפונקציה `fit()`
  3. בצעו תחזיות על ידי קריאה לפונקציה `predict()`

עכשיו ניצור מודל SVR. כאן נשתמש ב-[RBF kernel](https://scikit-learn.org/stable/modules/svm.html#parameters-of-the-rbf-kernel), ונגדיר את היפר-פרמטרים gamma, C ו-epsilon כ-0.5, 10 ו-0.05 בהתאמה.

```python
model = SVR(kernel='rbf',gamma=0.5, C=10, epsilon = 0.05)
```

#### התאמת המודל לנתוני האימון [^1]

```python
model.fit(x_train, y_train[:,0])
```

```output
SVR(C=10, cache_size=200, coef0=0.0, degree=3, epsilon=0.05, gamma=0.5,
    kernel='rbf', max_iter=-1, shrinking=True, tol=0.001, verbose=False)
```

#### ביצוע תחזיות עם המודל [^1]

```python
y_train_pred = model.predict(x_train).reshape(-1,1)
y_test_pred = model.predict(x_test).reshape(-1,1)

print(y_train_pred.shape, y_test_pred.shape)
```

```output
(1412, 1) (44, 1)
```

יצרתם את ה-SVR שלכם! עכשיו נצטרך להעריך אותו.

### הערכת המודל שלכם [^1]

להערכה, קודם כל נחזיר את הנתונים לסקייל המקורי שלנו. לאחר מכן, כדי לבדוק את הביצועים, ניצור גרף של סדרת הזמן המקורית והתחזית, ונדפיס גם את תוצאת ה-MAPE.

החזירו את הפלטים המנובאים והמקוריים לסקייל המקורי:

```python
# Scaling the predictions
y_train_pred = scaler.inverse_transform(y_train_pred)
y_test_pred = scaler.inverse_transform(y_test_pred)

print(len(y_train_pred), len(y_test_pred))
```

```python
# Scaling the original values
y_train = scaler.inverse_transform(y_train)
y_test = scaler.inverse_transform(y_test)

print(len(y_train), len(y_test))
```

#### בדיקת ביצועי המודל על נתוני האימון והבדיקה [^1]

נחלץ את חותמות הזמן מקבוצת הנתונים כדי להציג בציר ה-x של הגרף שלנו. שימו לב שאנחנו משתמשים ב-```timesteps-1``` הערכים הראשונים כקלט עבור הפלט הראשון, כך שחותמות הזמן עבור הפלט יתחילו לאחר מכן.

```python
train_timestamps = energy[(energy.index < test_start_dt) & (energy.index >= train_start_dt)].index[timesteps-1:]
test_timestamps = energy[test_start_dt:].index[timesteps-1:]

print(len(train_timestamps), len(test_timestamps))
```

```output
1412 44
```

צרו גרף של התחזיות עבור נתוני האימון:

```python
plt.figure(figsize=(25,6))
plt.plot(train_timestamps, y_train, color = 'red', linewidth=2.0, alpha = 0.6)
plt.plot(train_timestamps, y_train_pred, color = 'blue', linewidth=0.8)
plt.legend(['Actual','Predicted'])
plt.xlabel('Timestamp')
plt.title("Training data prediction")
plt.show()
```

![תחזית נתוני אימון](../../../../7-TimeSeries/3-SVR/images/train-data-predict.png)

הדפיסו את MAPE עבור נתוני האימון

```python
print('MAPE for training data: ', mape(y_train_pred, y_train)*100, '%')
```

```output
MAPE for training data: 1.7195710200875551 %
```

צרו גרף של התחזיות עבור נתוני הבדיקה

```python
plt.figure(figsize=(10,3))
plt.plot(test_timestamps, y_test, color = 'red', linewidth=2.0, alpha = 0.6)
plt.plot(test_timestamps, y_test_pred, color = 'blue', linewidth=0.8)
plt.legend(['Actual','Predicted'])
plt.xlabel('Timestamp')
plt.show()
```

![תחזית נתוני בדיקה](../../../../7-TimeSeries/3-SVR/images/test-data-predict.png)

הדפיסו את MAPE עבור נתוני הבדיקה

```python
print('MAPE for testing data: ', mape(y_test_pred, y_test)*100, '%')
```

```output
MAPE for testing data:  1.2623790187854018 %
```

🏆 קיבלתם תוצאה טובה מאוד על קבוצת הבדיקה!

### בדיקת ביצועי המודל על כל קבוצת הנתונים [^1]

```python
# Extracting load values as numpy array
data = energy.copy().values

# Scaling
data = scaler.transform(data)

# Transforming to 2D tensor as per model input requirement
data_timesteps=np.array([[j for j in data[i:i+timesteps]] for i in range(0,len(data)-timesteps+1)])[:,:,0]
print("Tensor shape: ", data_timesteps.shape)

# Selecting inputs and outputs from data
X, Y = data_timesteps[:,:timesteps-1],data_timesteps[:,[timesteps-1]]
print("X shape: ", X.shape,"\nY shape: ", Y.shape)
```

```output
Tensor shape:  (26300, 5)
X shape:  (26300, 4) 
Y shape:  (26300, 1)
```

```python
# Make model predictions
Y_pred = model.predict(X).reshape(-1,1)

# Inverse scale and reshape
Y_pred = scaler.inverse_transform(Y_pred)
Y = scaler.inverse_transform(Y)
```

```python
plt.figure(figsize=(30,8))
plt.plot(Y, color = 'red', linewidth=2.0, alpha = 0.6)
plt.plot(Y_pred, color = 'blue', linewidth=0.8)
plt.legend(['Actual','Predicted'])
plt.xlabel('Timestamp')
plt.show()
```

![תחזית נתונים מלאים](../../../../7-TimeSeries/3-SVR/images/full-data-predict.png)

```python
print('MAPE: ', mape(Y_pred, Y)*100, '%')
```

```output
MAPE:  2.0572089029888656 %
```



🏆 גרפים מרשימים מאוד, שמראים מודל עם דיוק טוב. כל הכבוד!

---

## 🚀אתגר

- נסו לשנות את היפר-פרמטרים (gamma, C, epsilon) בזמן יצירת המודל והעריכו את הנתונים כדי לראות איזה סט של היפר-פרמטרים נותן את התוצאות הטובות ביותר על נתוני הבדיקה. למידע נוסף על היפר-פרמטרים אלה, תוכלו לעיין בתיעוד [כאן](https://scikit-learn.org/stable/modules/svm.html#parameters-of-the-rbf-kernel). 
- נסו להשתמש בפונקציות kernel שונות עבור המודל ונתחו את ביצועיהן על קבוצת הנתונים. מסמך מועיל ניתן למצוא [כאן](https://scikit-learn.org/stable/modules/svm.html#kernel-functions).
- נסו להשתמש בערכים שונים עבור `timesteps` כדי שהמודל יוכל להסתכל אחורה ולבצע תחזית.

## [מבחן מסכם](https://ff-quizzes.netlify.app/en/ml/)

## סקירה ולימוד עצמי

שיעור זה נועד להציג את השימוש ב-SVR לחיזוי סדרות זמן. לקריאה נוספת על SVR, תוכלו לעיין ב-[בלוג הזה](https://www.analyticsvidhya.com/blog/2020/03/support-vector-regression-tutorial-for-machine-learning/). תיעוד זה ב-[scikit-learn](https://scikit-learn.org/stable/modules/svm.html) מספק הסבר מקיף יותר על SVMs באופן כללי, [SVRs](https://scikit-learn.org/stable/modules/svm.html#regression) וגם פרטי יישום אחרים כמו [פונקציות kernel](https://scikit-learn.org/stable/modules/svm.html#kernel-functions) שונות שניתן להשתמש בהן, והפרמטרים שלהן.

## משימה

[מודל SVR חדש](assignment.md)



## קרדיטים


[^1]: הטקסט, הקוד והתוצאות בסעיף זה נתרמו על ידי [@AnirbanMukherjeeXD](https://github.com/AnirbanMukherjeeXD)
[^2]: הטקסט, הקוד והתוצאות בסעיף זה נלקחו מ-[ARIMA](https://github.com/microsoft/ML-For-Beginners/tree/main/7-TimeSeries/2-ARIMA)

---

**כתב ויתור**:  
מסמך זה תורגם באמצעות שירות תרגום מבוסס בינה מלאכותית [Co-op Translator](https://github.com/Azure/co-op-translator). בעוד שאנו שואפים לדיוק, יש לקחת בחשבון שתרגומים אוטומטיים עשויים להכיל שגיאות או אי דיוקים. המסמך המקורי בשפתו המקורית צריך להיחשב כמקור סמכותי. עבור מידע קריטי, מומלץ להשתמש בתרגום מקצועי על ידי אדם. איננו נושאים באחריות לאי הבנות או לפרשנויות שגויות הנובעות משימוש בתרגום זה.