<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "917dbf890db71a322f306050cb284749",
  "translation_date": "2025-09-05T18:59:03+00:00",
  "source_file": "7-TimeSeries/2-ARIMA/README.md",
  "language_code": "he"
}
-->
# חיזוי סדרות זמן עם ARIMA

בשיעור הקודם, למדתם מעט על חיזוי סדרות זמן וטעינת מערך נתונים שמציג את התנודות בעומס החשמלי לאורך תקופת זמן.

> 🎥 לחצו על התמונה למעלה לצפייה בסרטון: מבוא קצר למודלים של ARIMA. הדוגמה נעשתה ב-R, אך הרעיונות הם אוניברסליים.

## [שאלון לפני השיעור](https://ff-quizzes.netlify.app/en/ml/)

## מבוא

בשיעור זה, תגלו דרך ספציפית לבנות מודלים עם [ARIMA: *A*uto*R*egressive *I*ntegrated *M*oving *A*verage](https://wikipedia.org/wiki/Autoregressive_integrated_moving_average). מודלים של ARIMA מתאימים במיוחד לנתונים שמציגים [אי-סטציונריות](https://wikipedia.org/wiki/Stationary_process).

## מושגים כלליים

כדי לעבוד עם ARIMA, ישנם כמה מושגים שחשוב להכיר:

- 🎓 **סטציונריות**. בהקשר סטטיסטי, סטציונריות מתייחסת לנתונים שההתפלגות שלהם אינה משתנה כאשר הם מוזזים בזמן. נתונים שאינם סטציונריים מציגים תנודות עקב מגמות שיש להפוך אותן כדי לנתח. עונתיות, למשל, יכולה להכניס תנודות לנתונים וניתן להסיר אותה באמצעות תהליך של 'הבדל עונתי'.

- 🎓 **[הבדלה](https://wikipedia.org/wiki/Autoregressive_integrated_moving_average#Differencing)**. הבדלה של נתונים, שוב בהקשר סטטיסטי, מתייחסת לתהליך של הפיכת נתונים שאינם סטציונריים לסטציונריים על ידי הסרת המגמה הלא-קבועה שלהם. "הבדלה מסירה את השינויים ברמת סדרת הזמן, מבטלת מגמות ועונתיות ובכך מייצבת את הממוצע של סדרת הזמן." [מאמר מאת Shixiong et al](https://arxiv.org/abs/1904.07632)

## ARIMA בהקשר של סדרות זמן

בואו נפרק את החלקים של ARIMA כדי להבין טוב יותר כיצד הוא עוזר לנו לבנות מודלים של סדרות זמן ולבצע תחזיות.

- **AR - עבור AutoRegressive**. מודלים אוטורגרסיביים, כפי שהשם מרמז, מסתכלים 'אחורה' בזמן כדי לנתח ערכים קודמים בנתונים שלכם ולבצע הנחות לגביהם. ערכים קודמים אלו נקראים 'פיגורים'. דוגמה לכך תהיה נתונים שמציגים מכירות חודשיות של עפרונות. סך המכירות של כל חודש ייחשב כ'משתנה מתפתח' במערך הנתונים. מודל זה נבנה כאשר "המשתנה המתפתח של העניין מוערך על ערכיו המפגרים (כלומר, הקודמים)." [ויקיפדיה](https://wikipedia.org/wiki/Autoregressive_integrated_moving_average)

- **I - עבור Integrated**. בניגוד למודלים דומים כמו 'ARMA', ה-'I' ב-ARIMA מתייחס להיבט ה*[משולב](https://wikipedia.org/wiki/Order_of_integration)* שלו. הנתונים 'משולבים' כאשר מיושמים שלבי הבדלה כדי לבטל אי-סטציונריות.

- **MA - עבור Moving Average**. ההיבט של [ממוצע נע](https://wikipedia.org/wiki/Moving-average_model) במודל זה מתייחס למשתנה הפלט שנקבע על ידי התבוננות בערכים הנוכחיים והעבריים של פיגורים.

שורה תחתונה: ARIMA משמש כדי להתאים מודל בצורה הקרובה ביותר לנתונים המיוחדים של סדרות זמן.

## תרגיל - בניית מודל ARIMA

פתחו את תיקיית [_/working_](https://github.com/microsoft/ML-For-Beginners/tree/main/7-TimeSeries/2-ARIMA/working) בשיעור זה ומצאו את הקובץ [_notebook.ipynb_](https://github.com/microsoft/ML-For-Beginners/blob/main/7-TimeSeries/2-ARIMA/working/notebook.ipynb).

1. הריצו את המחברת כדי לטעון את ספריית Python `statsmodels`; תזדקקו לה עבור מודלים של ARIMA.

1. טענו ספריות נחוצות.

1. כעת, טענו מספר ספריות נוספות שימושיות לשרטוט נתונים:

    ```python
    import os
    import warnings
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import datetime as dt
    import math

    from pandas.plotting import autocorrelation_plot
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from sklearn.preprocessing import MinMaxScaler
    from common.utils import load_data, mape
    from IPython.display import Image

    %matplotlib inline
    pd.options.display.float_format = '{:,.2f}'.format
    np.set_printoptions(precision=2)
    warnings.filterwarnings("ignore") # specify to ignore warning messages
    ```

1. טענו את הנתונים מקובץ `/data/energy.csv` לתוך DataFrame של Pandas והסתכלו עליהם:

    ```python
    energy = load_data('./data')[['load']]
    energy.head(10)
    ```

1. שרטטו את כל נתוני האנרגיה הזמינים מינואר 2012 עד דצמבר 2014. לא אמורות להיות הפתעות, כפי שראינו את הנתונים בשיעור הקודם:

    ```python
    energy.plot(y='load', subplots=True, figsize=(15, 8), fontsize=12)
    plt.xlabel('timestamp', fontsize=12)
    plt.ylabel('load', fontsize=12)
    plt.show()
    ```

    כעת, בואו נבנה מודל!

### יצירת מערכי נתונים לאימון ובדיקה

כעת הנתונים שלכם טעונים, כך שתוכלו להפריד אותם למערכי אימון ובדיקה. תאמנו את המודל שלכם על מערך האימון. כרגיל, לאחר שהמודל סיים את האימון, תעריכו את דיוקו באמצעות מערך הבדיקה. עליכם לוודא שמערך הבדיקה מכסה תקופה מאוחרת יותר בזמן ממערך האימון כדי להבטיח שהמודל לא יקבל מידע מתקופות זמן עתידיות.

1. הקצו תקופה של חודשיים מה-1 בספטמבר עד ה-31 באוקטובר 2014 למערך האימון. מערך הבדיקה יכלול את התקופה של חודשיים מה-1 בנובמבר עד ה-31 בדצמבר 2014:

    ```python
    train_start_dt = '2014-11-01 00:00:00'
    test_start_dt = '2014-12-30 00:00:00'
    ```

    מכיוון שנתונים אלו משקפים את צריכת האנרגיה היומית, ישנו דפוס עונתי חזק, אך הצריכה דומה ביותר לצריכה בימים האחרונים.

1. ויזואליזציה של ההבדלים:

    ```python
    energy[(energy.index < test_start_dt) & (energy.index >= train_start_dt)][['load']].rename(columns={'load':'train'}) \
        .join(energy[test_start_dt:][['load']].rename(columns={'load':'test'}), how='outer') \
        .plot(y=['train', 'test'], figsize=(15, 8), fontsize=12)
    plt.xlabel('timestamp', fontsize=12)
    plt.ylabel('load', fontsize=12)
    plt.show()
    ```

    ![נתוני אימון ובדיקה](../../../../7-TimeSeries/2-ARIMA/images/train-test.png)

    לכן, שימוש בחלון זמן קטן יחסית לאימון הנתונים אמור להיות מספיק.

    > הערה: מכיוון שהפונקציה שבה אנו משתמשים להתאמת מודל ARIMA משתמשת באימות בתוך הדגימה במהלך ההתאמה, נוותר על נתוני אימות.

### הכנת הנתונים לאימון

כעת, עליכם להכין את הנתונים לאימון על ידי ביצוע סינון וסקיילינג של הנתונים שלכם. סננו את מערך הנתונים שלכם כך שיכלול רק את התקופות והעמודות הנדרשות, וסקיילינג כדי להבטיח שהנתונים יוקרנו בטווח 0,1.

1. סננו את מערך הנתונים המקורי כך שיכלול רק את התקופות שהוזכרו לכל מערך ורק את העמודה הנדרשת 'load' בנוסף לתאריך:

    ```python
    train = energy.copy()[(energy.index >= train_start_dt) & (energy.index < test_start_dt)][['load']]
    test = energy.copy()[energy.index >= test_start_dt][['load']]

    print('Training data shape: ', train.shape)
    print('Test data shape: ', test.shape)
    ```

    תוכלו לראות את הצורה של הנתונים:

    ```output
    Training data shape:  (1416, 1)
    Test data shape:  (48, 1)
    ```

1. בצעו סקיילינג לנתונים כך שיהיו בטווח (0, 1).

    ```python
    scaler = MinMaxScaler()
    train['load'] = scaler.fit_transform(train)
    train.head(10)
    ```

1. ויזואליזציה של הנתונים המקוריים מול הנתונים המוקנים:

    ```python
    energy[(energy.index >= train_start_dt) & (energy.index < test_start_dt)][['load']].rename(columns={'load':'original load'}).plot.hist(bins=100, fontsize=12)
    train.rename(columns={'load':'scaled load'}).plot.hist(bins=100, fontsize=12)
    plt.show()
    ```

    ![מקורי](../../../../7-TimeSeries/2-ARIMA/images/original.png)

    > הנתונים המקוריים

    ![מוקנה](../../../../7-TimeSeries/2-ARIMA/images/scaled.png)

    > הנתונים המוקנים

1. כעת, לאחר שכיילתם את הנתונים המוקנים, תוכלו לכייל את נתוני הבדיקה:

    ```python
    test['load'] = scaler.transform(test)
    test.head()
    ```

### יישום ARIMA

הגיע הזמן ליישם ARIMA! כעת תשתמשו בספריית `statsmodels` שהתקנתם קודם.

כעת עליכם לבצע מספר שלבים:

   1. הגדירו את המודל על ידי קריאה ל-`SARIMAX()` והעברת פרמטרי המודל: פרמטרים p, d, ו-q, ופרמטרים P, D, ו-Q.
   2. הכינו את המודל לנתוני האימון על ידי קריאה לפונקציה fit().
   3. בצעו תחזיות על ידי קריאה לפונקציה `forecast()` וציון מספר הצעדים (ה'אופק') לתחזית.

> 🎓 מה משמעות כל הפרמטרים הללו? במודל ARIMA ישנם 3 פרמטרים המשמשים לסייע במידול ההיבטים המרכזיים של סדרת זמן: עונתיות, מגמה ורעש. הפרמטרים הם:

`p`: הפרמטר הקשור להיבט האוטורגרסיבי של המודל, שמשלב ערכים *עבריים*.
`d`: הפרמטר הקשור לחלק המשולב של המודל, שמשפיע על כמות ה-*הבדלה* (🎓 זוכרים הבדלה 👆?) שיש ליישם על סדרת זמן.
`q`: הפרמטר הקשור לחלק הממוצע הנע של המודל.

> הערה: אם לנתונים שלכם יש היבט עונתי - כמו במקרה זה - , אנו משתמשים במודל ARIMA עונתי (SARIMA). במקרה זה עליכם להשתמש בקבוצת פרמטרים נוספת: `P`, `D`, ו-`Q` שמתארים את אותם קשרים כמו `p`, `d`, ו-`q`, אך מתייחסים לרכיבים העונתיים של המודל.

1. התחילו בהגדרת ערך האופק המועדף עליכם. בואו ננסה 3 שעות:

    ```python
    # Specify the number of steps to forecast ahead
    HORIZON = 3
    print('Forecasting horizon:', HORIZON, 'hours')
    ```

    בחירת הערכים הטובים ביותר עבור פרמטרי מודל ARIMA יכולה להיות מאתגרת מכיוון שהיא מעט סובייקטיבית וגוזלת זמן. ייתכן שתרצו לשקול שימוש בפונקציה `auto_arima()` מתוך ספריית [`pyramid`](https://alkaline-ml.com/pmdarima/0.9.0/modules/generated/pyramid.arima.auto_arima.html).

1. לעת עתה נסו כמה בחירות ידניות כדי למצוא מודל טוב.

    ```python
    order = (4, 1, 0)
    seasonal_order = (1, 1, 0, 24)

    model = SARIMAX(endog=train, order=order, seasonal_order=seasonal_order)
    results = model.fit()

    print(results.summary())
    ```

    טבלה של תוצאות מודפסת.

בניתם את המודל הראשון שלכם! כעת עלינו למצוא דרך להעריך אותו.

### הערכת המודל שלכם

כדי להעריך את המודל שלכם, תוכלו לבצע את מה שנקרא `walk forward` validation. בפועל, מודלים של סדרות זמן מאומנים מחדש בכל פעם שנתונים חדשים הופכים זמינים. זה מאפשר למודל לבצע את התחזית הטובה ביותר בכל שלב זמן.

מתחילים בתחילת סדרת הזמן באמצעות טכניקה זו, מאמנים את המודל על מערך נתוני האימון. לאחר מכן מבצעים תחזית על שלב הזמן הבא. התחזית מוערכת מול הערך הידוע. מערך האימון מורחב כך שיכלול את הערך הידוע והתהליך חוזר על עצמו.

> הערה: עליכם לשמור על חלון מערך האימון קבוע לצורך אימון יעיל יותר כך שבכל פעם שאתם מוסיפים תצפית חדשה למערך האימון, אתם מסירים את התצפית מתחילת המערך.

תהליך זה מספק הערכה חזקה יותר של איך המודל יפעל בפועל. עם זאת, הוא מגיע בעלות חישובית של יצירת כל כך הרבה מודלים. זה מקובל אם הנתונים קטנים או אם המודל פשוט, אך יכול להיות בעייתי בקנה מידה גדול.

Walk-forward validation הוא תקן הזהב להערכת מודלים של סדרות זמן ומומלץ לפרויקטים שלכם.

1. ראשית, צרו נקודת נתוני בדיקה עבור כל שלב אופק.

    ```python
    test_shifted = test.copy()

    for t in range(1, HORIZON+1):
        test_shifted['load+'+str(t)] = test_shifted['load'].shift(-t, freq='H')

    test_shifted = test_shifted.dropna(how='any')
    test_shifted.head(5)
    ```

    |            |          | load | load+1 | load+2 |
    | ---------- | -------- | ---- | ------ | ------ |
    | 2014-12-30 | 00:00:00 | 0.33 | 0.29   | 0.27   |
    | 2014-12-30 | 01:00:00 | 0.29 | 0.27   | 0.27   |
    | 2014-12-30 | 02:00:00 | 0.27 | 0.27   | 0.30   |
    | 2014-12-30 | 03:00:00 | 0.27 | 0.30   | 0.41   |
    | 2014-12-30 | 04:00:00 | 0.30 | 0.41   | 0.57   |

    הנתונים מוזזים אופקית בהתאם לנקודת האופק שלהם.

1. בצעו תחזיות על נתוני הבדיקה שלכם באמצעות גישה זו של חלון הזזה בלולאה בגודל אורך נתוני הבדיקה:

    ```python
    %%time
    training_window = 720 # dedicate 30 days (720 hours) for training

    train_ts = train['load']
    test_ts = test_shifted

    history = [x for x in train_ts]
    history = history[(-training_window):]

    predictions = list()

    order = (2, 1, 0)
    seasonal_order = (1, 1, 0, 24)

    for t in range(test_ts.shape[0]):
        model = SARIMAX(endog=history, order=order, seasonal_order=seasonal_order)
        model_fit = model.fit()
        yhat = model_fit.forecast(steps = HORIZON)
        predictions.append(yhat)
        obs = list(test_ts.iloc[t])
        # move the training window
        history.append(obs[0])
        history.pop(0)
        print(test_ts.index[t])
        print(t+1, ': predicted =', yhat, 'expected =', obs)
    ```

    תוכלו לצפות באימון מתרחש:

    ```output
    2014-12-30 00:00:00
    1 : predicted = [0.32 0.29 0.28] expected = [0.32945389435989236, 0.2900626678603402, 0.2739480752014323]

    2014-12-30 01:00:00
    2 : predicted = [0.3  0.29 0.3 ] expected = [0.2900626678603402, 0.2739480752014323, 0.26812891674127126]

    2014-12-30 02:00:00
    3 : predicted = [0.27 0.28 0.32] expected = [0.2739480752014323, 0.26812891674127126, 0.3025962399283795]
    ```

1. השוו את התחזיות לעומס בפועל:

    ```python
    eval_df = pd.DataFrame(predictions, columns=['t+'+str(t) for t in range(1, HORIZON+1)])
    eval_df['timestamp'] = test.index[0:len(test.index)-HORIZON+1]
    eval_df = pd.melt(eval_df, id_vars='timestamp', value_name='prediction', var_name='h')
    eval_df['actual'] = np.array(np.transpose(test_ts)).ravel()
    eval_df[['prediction', 'actual']] = scaler.inverse_transform(eval_df[['prediction', 'actual']])
    eval_df.head()
    ```

    פלט
    |     |            | timestamp | h   | prediction | actual   |
    | --- | ---------- | --------- | --- | ---------- | -------- |
    | 0   | 2014-12-30 | 00:00:00  | t+1 | 3,008.74   | 3,023.00 |
    | 1   | 2014-12-30 | 01:00:00  | t+1 | 2,955.53   | 2,935.00 |
    | 2   | 2014-12-30 | 02:00:00  | t+1 | 2,900.17   | 2,899.00 |
    | 3   | 2014-12-30 | 03:00:00  | t+1 | 2,917.69   | 2,886.00 |
    | 4   | 2014-12-30 | 04:00:00  | t+1 | 2,946.99   | 2,963.00 |

    התבוננו בתחזית הנתונים השעתית, בהשוואה לעומס בפועל. עד כמה זה מדויק?

### בדיקת דיוק המודל

בדקו את דיוק המודל שלכם על ידי בדיקת שגיאת האחוז הממוצעת המוחלטת (MAPE) שלו על כל התחזיות.
> **🧮 הצג לי את המתמטיקה**
>
> ![MAPE](../../../../7-TimeSeries/2-ARIMA/images/mape.png)
>
> [MAPE](https://www.linkedin.com/pulse/what-mape-mad-msd-time-series-allameh-statistics/) משמש להציג את דיוק התחזית כיחס שמוגדר על ידי הנוסחה לעיל. ההפרש בין הערך האמיתי לערך החזוי מחולק בערך האמיתי.  
> "הערך המוחלט בחישוב זה מסוכם עבור כל נקודת תחזית בזמן ומחולק במספר הנקודות המותאמות n." [ויקיפדיה](https://wikipedia.org/wiki/Mean_absolute_percentage_error)
1. כתיבת משוואה בקוד:

    ```python
    if(HORIZON > 1):
        eval_df['APE'] = (eval_df['prediction'] - eval_df['actual']).abs() / eval_df['actual']
        print(eval_df.groupby('h')['APE'].mean())
    ```

1. חישוב MAPE של צעד אחד:

    ```python
    print('One step forecast MAPE: ', (mape(eval_df[eval_df['h'] == 't+1']['prediction'], eval_df[eval_df['h'] == 't+1']['actual']))*100, '%')
    ```

    MAPE של תחזית צעד אחד: 0.5570581332313952 %

1. הדפסת MAPE של תחזית רב-שלבית:

    ```python
    print('Multi-step forecast MAPE: ', mape(eval_df['prediction'], eval_df['actual'])*100, '%')
    ```

    ```output
    Multi-step forecast MAPE:  1.1460048657704118 %
    ```

    מספר נמוך הוא הטוב ביותר: יש לקחת בחשבון שתחזית עם MAPE של 10 היא תחזית עם סטייה של 10%.

1. אבל כמו תמיד, קל יותר לראות מדידה כזו של דיוק באופן חזותי, אז בואו נשרטט את זה:

    ```python
     if(HORIZON == 1):
        ## Plotting single step forecast
        eval_df.plot(x='timestamp', y=['actual', 'prediction'], style=['r', 'b'], figsize=(15, 8))

    else:
        ## Plotting multi step forecast
        plot_df = eval_df[(eval_df.h=='t+1')][['timestamp', 'actual']]
        for t in range(1, HORIZON+1):
            plot_df['t+'+str(t)] = eval_df[(eval_df.h=='t+'+str(t))]['prediction'].values

        fig = plt.figure(figsize=(15, 8))
        ax = plt.plot(plot_df['timestamp'], plot_df['actual'], color='red', linewidth=4.0)
        ax = fig.add_subplot(111)
        for t in range(1, HORIZON+1):
            x = plot_df['timestamp'][(t-1):]
            y = plot_df['t+'+str(t)][0:len(x)]
            ax.plot(x, y, color='blue', linewidth=4*math.pow(.9,t), alpha=math.pow(0.8,t))

        ax.legend(loc='best')

    plt.xlabel('timestamp', fontsize=12)
    plt.ylabel('load', fontsize=12)
    plt.show()
    ```

    ![מודל סדרת זמן](../../../../7-TimeSeries/2-ARIMA/images/accuracy.png)

🏆 גרף יפה מאוד, שמציג מודל עם דיוק טוב. עבודה מצוינת!

---

## 🚀אתגר

חקרו את הדרכים לבחון את דיוקו של מודל סדרת זמן. בשיעור זה נגענו ב-MAPE, אבל האם יש שיטות נוספות שתוכלו להשתמש בהן? חקרו אותן והוסיפו הערות. מסמך מועיל ניתן למצוא [כאן](https://otexts.com/fpp2/accuracy.html)

## [שאלון לאחר השיעור](https://ff-quizzes.netlify.app/en/ml/)

## סקירה ולימוד עצמי

שיעור זה נוגע רק ביסודות של תחזיות סדרת זמן עם ARIMA. הקדישו זמן להעמיק את הידע שלכם על ידי חקר [מאגר זה](https://microsoft.github.io/forecasting/) וסוגי המודלים השונים שבו כדי ללמוד דרכים נוספות לבנות מודלים של סדרות זמן.

## משימה

[מודל ARIMA חדש](assignment.md)

---

**כתב ויתור**:  
מסמך זה תורגם באמצעות שירות תרגום מבוסס בינה מלאכותית [Co-op Translator](https://github.com/Azure/co-op-translator). למרות שאנו שואפים לדיוק, יש לקחת בחשבון שתרגומים אוטומטיים עשויים להכיל שגיאות או אי דיוקים. המסמך המקורי בשפתו המקורית צריך להיחשב כמקור סמכותי. עבור מידע קריטי, מומלץ להשתמש בתרגום מקצועי על ידי אדם. איננו נושאים באחריות לאי הבנות או לפרשנויות שגויות הנובעות משימוש בתרגום זה.