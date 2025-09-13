<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "917dbf890db71a322f306050cb284749",
  "translation_date": "2025-09-06T08:46:38+00:00",
  "source_file": "7-TimeSeries/2-ARIMA/README.md",
  "language_code": "ur"
}
-->
# وقت کی سیریز کی پیش گوئی ARIMA کے ساتھ

پچھلے سبق میں، آپ نے وقت کی سیریز کی پیش گوئی کے بارے میں کچھ سیکھا اور ایک ڈیٹا سیٹ لوڈ کیا جو ایک وقت کے عرصے میں بجلی کے لوڈ میں اتار چڑھاؤ کو ظاہر کرتا ہے۔

[![ARIMA کا تعارف](https://img.youtube.com/vi/IUSk-YDau10/0.jpg)](https://youtu.be/IUSk-YDau10 "ARIMA کا تعارف")

> 🎥 اوپر دی گئی تصویر پر کلک کریں ایک ویڈیو کے لیے: ARIMA ماڈلز کا مختصر تعارف۔ مثال R میں کی گئی ہے، لیکن تصورات عالمی ہیں۔

## [سبق سے پہلے کا کوئز](https://ff-quizzes.netlify.app/en/ml/)

## تعارف

اس سبق میں، آپ ایک خاص طریقہ دریافت کریں گے جس کے ذریعے ماڈلز کو [ARIMA: *A*uto*R*egressive *I*ntegrated *M*oving *A*verage](https://wikipedia.org/wiki/Autoregressive_integrated_moving_average) کے ساتھ بنایا جا سکتا ہے۔ ARIMA ماڈلز خاص طور پر ایسے ڈیٹا کے لیے موزوں ہیں جو [غیر مستحکم](https://wikipedia.org/wiki/Stationary_process) رجحانات ظاہر کرتا ہے۔

## عمومی تصورات

ARIMA کے ساتھ کام کرنے کے لیے، آپ کو کچھ بنیادی تصورات کے بارے میں جاننا ضروری ہے:

- 🎓 **مستحکمی**۔ شماریاتی سیاق و سباق میں، مستحکمی اس ڈیٹا کو کہتے ہیں جس کی تقسیم وقت کے ساتھ تبدیل نہیں ہوتی۔ غیر مستحکم ڈیٹا ایسے اتار چڑھاؤ ظاہر کرتا ہے جو رجحانات کی وجہ سے ہوتے ہیں اور تجزیہ کے لیے اسے تبدیل کرنا ضروری ہوتا ہے۔ مثال کے طور پر، موسمی رجحانات ڈیٹا میں اتار چڑھاؤ پیدا کر سکتے ہیں، جنہیں 'موسمی فرق' کے عمل سے ختم کیا جا سکتا ہے۔

- 🎓 **[فرق لینا](https://wikipedia.org/wiki/Autoregressive_integrated_moving_average#Differencing)**۔ شماریاتی سیاق و سباق میں، فرق لینا غیر مستحکم ڈیٹا کو مستحکم بنانے کے عمل کو کہتے ہیں، جس میں غیر مستقل رجحان کو ختم کیا جاتا ہے۔ "فرق لینا وقت کی سیریز کی سطح میں تبدیلیوں کو ختم کرتا ہے، رجحان اور موسمی اثرات کو ختم کرتا ہے اور نتیجتاً وقت کی سیریز کے اوسط کو مستحکم کرتا ہے۔" [Shixiong et al کا مقالہ](https://arxiv.org/abs/1904.07632)

## وقت کی سیریز کے سیاق و سباق میں ARIMA

آئیے ARIMA کے اجزاء کو کھول کر دیکھتے ہیں تاکہ یہ سمجھ سکیں کہ یہ وقت کی سیریز کو ماڈل کرنے اور اس کے خلاف پیش گوئی کرنے میں کیسے مدد کرتا ہے۔

- **AR - آٹو ریگریسیو کے لیے**۔ آٹو ریگریسیو ماڈلز، جیسا کہ نام سے ظاہر ہے، وقت میں 'پیچھے' دیکھتے ہیں تاکہ آپ کے ڈیٹا میں پچھلی قدروں کا تجزیہ کریں اور ان کے بارے میں مفروضے بنائیں۔ ان پچھلی قدروں کو 'لیگز' کہا جاتا ہے۔ مثال کے طور پر، وہ ڈیٹا جو پنسلوں کی ماہانہ فروخت کو ظاہر کرتا ہے۔ ہر مہینے کی فروخت کا کل ڈیٹا سیٹ میں ایک 'ارتقائی متغیر' سمجھا جائے گا۔ یہ ماڈل اس طرح بنایا جاتا ہے کہ "دلچسپی کا ارتقائی متغیر اپنی ہی پچھلی (یعنی، سابقہ) قدروں پر ریگریس کیا جاتا ہے۔" [ویکیپیڈیا](https://wikipedia.org/wiki/Autoregressive_integrated_moving_average)

- **I - انٹیگریٹڈ کے لیے**۔ ARMA ماڈلز کے برعکس، ARIMA میں 'I' اس کے *[انٹیگریٹڈ](https://wikipedia.org/wiki/Order_of_integration)* پہلو کی طرف اشارہ کرتا ہے۔ ڈیٹا کو 'انٹیگریٹ' کیا جاتا ہے جب فرق لینے کے اقدامات غیر مستحکمی کو ختم کرنے کے لیے لاگو کیے جاتے ہیں۔

- **MA - موونگ ایوریج کے لیے**۔ اس ماڈل کا [موونگ ایوریج](https://wikipedia.org/wiki/Moving-average_model) پہلو اس آؤٹ پٹ متغیر کی طرف اشارہ کرتا ہے جو لیگز کی موجودہ اور پچھلی قدروں کو دیکھ کر طے کیا جاتا ہے۔

خلاصہ: ARIMA وقت کی سیریز کے ڈیٹا کی خاص شکل کو زیادہ سے زیادہ قریب سے فٹ کرنے کے لیے استعمال کیا جاتا ہے۔

## مشق - ARIMA ماڈل بنائیں

اس سبق کے [_/working_](https://github.com/microsoft/ML-For-Beginners/tree/main/7-TimeSeries/2-ARIMA/working) فولڈر کو کھولیں اور [_notebook.ipynb_](https://github.com/microsoft/ML-For-Beginners/blob/main/7-TimeSeries/2-ARIMA/working/notebook.ipynb) فائل تلاش کریں۔

1. نوٹ بک کو چلائیں تاکہ `statsmodels` Python لائبریری لوڈ ہو؛ آپ کو ARIMA ماڈلز کے لیے اس کی ضرورت ہوگی۔

1. ضروری لائبریریاں لوڈ کریں۔

1. اب، ڈیٹا کو پلاٹ کرنے کے لیے مزید لائبریریاں لوڈ کریں:

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

1. `/data/energy.csv` فائل سے ڈیٹا کو ایک Pandas ڈیٹا فریم میں لوڈ کریں اور دیکھیں:

    ```python
    energy = load_data('./data')[['load']]
    energy.head(10)
    ```

1. جنوری 2012 سے دسمبر 2014 تک دستیاب تمام توانائی کے ڈیٹا کو پلاٹ کریں۔ اس میں کوئی حیرت نہیں ہونی چاہیے کیونکہ ہم نے یہ ڈیٹا پچھلے سبق میں دیکھا تھا:

    ```python
    energy.plot(y='load', subplots=True, figsize=(15, 8), fontsize=12)
    plt.xlabel('timestamp', fontsize=12)
    plt.ylabel('load', fontsize=12)
    plt.show()
    ```

    اب، آئیے ایک ماڈل بناتے ہیں!

### تربیتی اور جانچ کے ڈیٹا سیٹس بنائیں

اب آپ کا ڈیٹا لوڈ ہو چکا ہے، لہذا آپ اسے تربیتی اور جانچ کے سیٹس میں تقسیم کر سکتے ہیں۔ آپ اپنے ماڈل کو تربیتی سیٹ پر تربیت دیں گے۔ حسب معمول، ماڈل کی تربیت مکمل ہونے کے بعد، آپ اس کی درستگی کا اندازہ جانچ کے سیٹ کا استعمال کرتے ہوئے کریں گے۔ آپ کو یہ یقینی بنانا ہوگا کہ جانچ کا سیٹ تربیتی سیٹ کے بعد کے وقت کے عرصے کا احاطہ کرتا ہے تاکہ ماڈل مستقبل کے وقت کے عرصے سے معلومات حاصل نہ کرے۔

1. ستمبر 1 سے اکتوبر 31، 2014 تک کے دو ماہ کے عرصے کو تربیتی سیٹ کے لیے مختص کریں۔ جانچ کا سیٹ نومبر 1 سے دسمبر 31، 2014 کے دو ماہ کے عرصے پر مشتمل ہوگا:

    ```python
    train_start_dt = '2014-11-01 00:00:00'
    test_start_dt = '2014-12-30 00:00:00'
    ```

    چونکہ یہ ڈیٹا توانائی کے روزانہ استعمال کو ظاہر کرتا ہے، اس میں ایک مضبوط موسمی نمونہ موجود ہے، لیکن حالیہ دنوں کے استعمال کے ساتھ سب سے زیادہ مشابہت ہے۔

1. فرق کو بصری طور پر دیکھیں:

    ```python
    energy[(energy.index < test_start_dt) & (energy.index >= train_start_dt)][['load']].rename(columns={'load':'train'}) \
        .join(energy[test_start_dt:][['load']].rename(columns={'load':'test'}), how='outer') \
        .plot(y=['train', 'test'], figsize=(15, 8), fontsize=12)
    plt.xlabel('timestamp', fontsize=12)
    plt.ylabel('load', fontsize=12)
    plt.show()
    ```

    ![تربیتی اور جانچ کا ڈیٹا](../../../../7-TimeSeries/2-ARIMA/images/train-test.png)

    لہذا، ڈیٹا کو تربیت دینے کے لیے وقت کی ایک نسبتاً چھوٹی ونڈو کا استعمال کافی ہونا چاہیے۔

    > نوٹ: چونکہ ہم ARIMA ماڈل کو فٹ کرنے کے لیے استعمال ہونے والے فنکشن میں تربیت کے دوران اندرونی نمونہ کی توثیق کا استعمال کرتے ہیں، ہم توثیقی ڈیٹا کو چھوڑ دیں گے۔

### تربیت کے لیے ڈیٹا تیار کریں

اب، آپ کو ڈیٹا کو تربیت کے لیے تیار کرنے کی ضرورت ہے، جس میں فلٹرنگ اور اسکیلنگ شامل ہے۔ اپنے ڈیٹا سیٹ کو صرف ان وقت کے عرصوں اور کالموں تک محدود کریں جن کی آپ کو ضرورت ہے، اور اسکیلنگ کو یقینی بنائیں کہ ڈیٹا 0 اور 1 کے وقفے میں پیش کیا گیا ہے۔

1. اصل ڈیٹا سیٹ کو فلٹر کریں تاکہ صرف مذکورہ وقت کے عرصے اور صرف مطلوبہ کالم 'لوڈ' اور تاریخ شامل ہوں:

    ```python
    train = energy.copy()[(energy.index >= train_start_dt) & (energy.index < test_start_dt)][['load']]
    test = energy.copy()[energy.index >= test_start_dt][['load']]

    print('Training data shape: ', train.shape)
    print('Test data shape: ', test.shape)
    ```

    آپ ڈیٹا کی شکل دیکھ سکتے ہیں:

    ```output
    Training data shape:  (1416, 1)
    Test data shape:  (48, 1)
    ```

1. ڈیٹا کو (0, 1) کی حد میں اسکیل کریں۔

    ```python
    scaler = MinMaxScaler()
    train['load'] = scaler.fit_transform(train)
    train.head(10)
    ```

1. اصل بمقابلہ اسکیل شدہ ڈیٹا کو بصری طور پر دیکھیں:

    ```python
    energy[(energy.index >= train_start_dt) & (energy.index < test_start_dt)][['load']].rename(columns={'load':'original load'}).plot.hist(bins=100, fontsize=12)
    train.rename(columns={'load':'scaled load'}).plot.hist(bins=100, fontsize=12)
    plt.show()
    ```

    ![اصل](../../../../7-TimeSeries/2-ARIMA/images/original.png)

    > اصل ڈیٹا

    ![اسکیل شدہ](../../../../7-TimeSeries/2-ARIMA/images/scaled.png)

    > اسکیل شدہ ڈیٹا

1. اب جب کہ آپ نے اسکیل شدہ ڈیٹا کو کیلیبریٹ کر لیا ہے، آپ جانچ کے ڈیٹا کو اسکیل کر سکتے ہیں:

    ```python
    test['load'] = scaler.transform(test)
    test.head()
    ```

### ARIMA کو نافذ کریں

اب وقت آ گیا ہے کہ ARIMA کو نافذ کریں! آپ اس کے لیے پہلے سے انسٹال کردہ `statsmodels` لائبریری کا استعمال کریں گے۔

اب آپ کو کئی مراحل پر عمل کرنے کی ضرورت ہے:

   1. ماڈل کو `SARIMAX()` کال کر کے اور ماڈل کے پیرامیٹرز: p، d، اور q پیرامیٹرز، اور P، D، اور Q پیرامیٹرز پاس کر کے ڈیفائن کریں۔
   2. تربیتی ڈیٹا کے لیے ماڈل کو تیار کریں `fit()` فنکشن کو کال کر کے۔
   3. پیش گوئی کرنے کے لیے `forecast()` فنکشن کو کال کریں اور پیش گوئی کے لیے قدموں کی تعداد (یعنی `horizon`) کو مخصوص کریں۔

> 🎓 یہ تمام پیرامیٹرز کس لیے ہیں؟ ARIMA ماڈل میں 3 پیرامیٹرز ہوتے ہیں جو وقت کی سیریز کے اہم پہلوؤں کو ماڈل کرنے میں مدد کرتے ہیں: موسمی رجحان، رجحان، اور شور۔ یہ پیرامیٹرز ہیں:

`p`: ماڈل کے آٹو ریگریسیو پہلو سے وابستہ پیرامیٹر، جو *ماضی* کی قدروں کو شامل کرتا ہے۔  
`d`: ماڈل کے انٹیگریٹڈ حصے سے وابستہ پیرامیٹر، جو وقت کی سیریز پر *فرق لینے* کی مقدار کو متاثر کرتا ہے۔  
`q`: ماڈل کے موونگ ایوریج حصے سے وابستہ پیرامیٹر۔

> نوٹ: اگر آپ کے ڈیٹا میں موسمی پہلو موجود ہو - جیسا کہ اس ڈیٹا میں ہے - تو ہم موسمی ARIMA ماڈل (SARIMA) استعمال کرتے ہیں۔ اس صورت میں آپ کو ایک اور سیٹ کے پیرامیٹرز استعمال کرنے کی ضرورت ہوگی: `P`، `D`، اور `Q` جو `p`، `d`، اور `q` کے جیسے ہی تعلقات کو بیان کرتے ہیں، لیکن ماڈل کے موسمی اجزاء سے متعلق ہیں۔

1. اپنے پسندیدہ افق کی قیمت مقرر کریں۔ آئیے 3 گھنٹے آزماتے ہیں:

    ```python
    # Specify the number of steps to forecast ahead
    HORIZON = 3
    print('Forecasting horizon:', HORIZON, 'hours')
    ```

    ARIMA ماڈل کے پیرامیٹرز کے لیے بہترین قدروں کا انتخاب کرنا چیلنجنگ ہو سکتا ہے کیونکہ یہ کسی حد تک موضوعی اور وقت طلب ہے۔ آپ [`pyramid` لائبریری](https://alkaline-ml.com/pmdarima/0.9.0/modules/generated/pyramid.arima.auto_arima.html) کے `auto_arima()` فنکشن پر غور کر سکتے ہیں۔

1. فی الحال کچھ دستی انتخاب آزمائیں تاکہ ایک اچھا ماڈل تلاش کیا جا سکے۔

    ```python
    order = (4, 1, 0)
    seasonal_order = (1, 1, 0, 24)

    model = SARIMAX(endog=train, order=order, seasonal_order=seasonal_order)
    results = model.fit()

    print(results.summary())
    ```

    نتائج کی ایک جدول پرنٹ کی گئی ہے۔

آپ نے اپنا پہلا ماڈل بنا لیا ہے! اب ہمیں اسے جانچنے کا ایک طریقہ تلاش کرنے کی ضرورت ہے۔

### اپنے ماڈل کا جائزہ لیں

اپنے ماڈل کا جائزہ لینے کے لیے، آپ نام نہاد `walk forward` توثیق انجام دے سکتے ہیں۔ عملی طور پر، وقت کی سیریز کے ماڈلز کو ہر بار جب نیا ڈیٹا دستیاب ہوتا ہے دوبارہ تربیت دی جاتی ہے۔ یہ ماڈل کو ہر وقت کے قدم پر بہترین پیش گوئی کرنے کی اجازت دیتا ہے۔

اس تکنیک کا استعمال کرتے ہوئے وقت کی سیریز کے آغاز سے شروع کریں، تربیتی ڈیٹا سیٹ پر ماڈل کو تربیت دیں۔ پھر اگلے وقت کے قدم پر پیش گوئی کریں۔ پیش گوئی کو معلوم قدر کے خلاف جانچا جاتا ہے۔ تربیتی سیٹ کو معلوم قدر شامل کرنے کے لیے بڑھایا جاتا ہے اور یہ عمل دہرایا جاتا ہے۔

> نوٹ: آپ کو تربیتی سیٹ کی ونڈو کو زیادہ موثر تربیت کے لیے مقررہ رکھنا چاہیے تاکہ ہر بار جب آپ تربیتی سیٹ میں ایک نیا مشاہدہ شامل کریں، آپ سیٹ کے آغاز سے مشاہدہ ہٹا دیں۔

یہ عمل اس بات کا زیادہ مضبوط تخمینہ فراہم کرتا ہے کہ ماڈل عملی طور پر کیسا کارکردگی دکھائے گا۔ تاہم، اس میں اتنے زیادہ ماڈلز بنانے کی کمپیوٹیشنل لاگت آتی ہے۔ اگر ڈیٹا چھوٹا ہو یا ماڈل سادہ ہو تو یہ قابل قبول ہے، لیکن بڑے پیمانے پر یہ مسئلہ بن سکتا ہے۔

`walk-forward` توثیق وقت کی سیریز کے ماڈلز کے جائزے کا سنہری معیار ہے اور آپ کے اپنے منصوبوں کے لیے تجویز کیا جاتا ہے۔

1. سب سے پہلے، ہر HORIZON قدم کے لیے ایک ٹیسٹ ڈیٹا پوائنٹ بنائیں۔

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

    ڈیٹا کو اس کے افق کے نقطہ کے مطابق افقی طور پر منتقل کیا گیا ہے۔

1. اپنے ٹیسٹ ڈیٹا پر اس سلائیڈنگ ونڈو اپروچ کا استعمال کرتے ہوئے پیش گوئیاں کریں، ٹیسٹ ڈیٹا کی لمبائی کے سائز کے ایک لوپ میں:

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

    آپ تربیت کو دیکھ سکتے ہیں:

    ```output
    2014-12-30 00:00:00
    1 : predicted = [0.32 0.29 0.28] expected = [0.32945389435989236, 0.2900626678603402, 0.2739480752014323]

    2014-12-30 01:00:00
    2 : predicted = [0.3  0.29 0.3 ] expected = [0.2900626678603402, 0.2739480752014323, 0.26812891674127126]

    2014-12-30 02:00:00
    3 : predicted = [0.27 0.28 0.32] expected = [0.2739480752014323, 0.26812891674127126, 0.3025962399283795]
    ```

1. پیش گوئیوں کا اصل لوڈ کے ساتھ موازنہ کریں:

    ```python
    eval_df = pd.DataFrame(predictions, columns=['t+'+str(t) for t in range(1, HORIZON+1)])
    eval_df['timestamp'] = test.index[0:len(test.index)-HORIZON+1]
    eval_df = pd.melt(eval_df, id_vars='timestamp', value_name='prediction', var_name='h')
    eval_df['actual'] = np.array(np.transpose(test_ts)).ravel()
    eval_df[['prediction', 'actual']] = scaler.inverse_transform(eval_df[['prediction', 'actual']])
    eval_df.head()
    ```

    آؤٹ پٹ  
    |     |            | timestamp | h   | prediction | actual   |
    | --- | ---------- | --------- | --- | ---------- | -------- |
    | 0   | 2014-12-30 | 00:00:00  | t+1 | 3,008.74   | 3,023.00 |
    | 1   | 2014-12-30 | 01:00:00  | t+1 | 2,955.53   | 2,935.00 |
    | 2   | 2014-12-30 | 02:00:00  | t+1 | 2,900.17   | 2,899.00 |
    | 3   | 2014-12-30 | 03:00:00  | t+1 | 2,917.69   | 2,886.00 |
    | 4   | 2014-12-30 | 04:00:00  | t+1 | 2,946.99   | 2,963.00 |

    گھنٹہ وار ڈیٹا کی پیش گوئی کو اصل لوڈ کے ساتھ ملاحظہ کریں۔ یہ کتنا درست ہے؟

### ماڈل کی درستگی چیک کریں

اپنے ماڈل کی درستگی کو چیک کریں، تمام پیش گوئیوں پر اس کی اوسط مطلق فیصد غلطی (MAPE) کی جانچ کر کے۔
> **🧮 ریاضی دیکھیں**
>
> ![MAPE](../../../../7-TimeSeries/2-ARIMA/images/mape.png)
>
> [MAPE](https://www.linkedin.com/pulse/what-mape-mad-msd-time-series-allameh-statistics/) پیش گوئی کی درستگی کو ایک تناسب کے طور پر ظاہر کرنے کے لیے استعمال کیا جاتا ہے، جو اوپر دیے گئے فارمولے سے بیان کیا گیا ہے۔ اصل اور پیش گوئی کے درمیان فرق کو اصل سے تقسیم کیا جاتا ہے۔
"اس حساب میں مطلق قدر کو وقت کے ہر پیش گوئی شدہ نقطے کے لیے جمع کیا جاتا ہے اور فٹ کیے گئے نقاط کی تعداد n سے تقسیم کیا جاتا ہے۔" [ویکیپیڈیا](https://wikipedia.org/wiki/Mean_absolute_percentage_error)
1. کوڈ میں مساوات ظاہر کریں:

    ```python
    if(HORIZON > 1):
        eval_df['APE'] = (eval_df['prediction'] - eval_df['actual']).abs() / eval_df['actual']
        print(eval_df.groupby('h')['APE'].mean())
    ```

1. ایک قدم کے MAPE کا حساب لگائیں:

    ```python
    print('One step forecast MAPE: ', (mape(eval_df[eval_df['h'] == 't+1']['prediction'], eval_df[eval_df['h'] == 't+1']['actual']))*100, '%')
    ```

    ایک قدم کی پیش گوئی کا MAPE:  0.5570581332313952 %

1. کثیر قدم کی پیش گوئی کا MAPE پرنٹ کریں:

    ```python
    print('Multi-step forecast MAPE: ', mape(eval_df['prediction'], eval_df['actual'])*100, '%')
    ```

    ```output
    Multi-step forecast MAPE:  1.1460048657704118 %
    ```

    ایک اچھا کم نمبر بہترین ہے: غور کریں کہ اگر پیش گوئی کا MAPE 10 ہو تو یہ 10% سے غلط ہے۔

1. لیکن ہمیشہ کی طرح، اس قسم کی درستگی کی پیمائش کو بصری طور پر دیکھنا آسان ہے، تو آئیے اسے پلاٹ کرتے ہیں:

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

    ![ایک وقت کی سیریز ماڈل](../../../../7-TimeSeries/2-ARIMA/images/accuracy.png)

🏆 ایک بہت اچھا پلاٹ، جو ایک ماڈل کو اچھی درستگی کے ساتھ دکھاتا ہے۔ شاباش!

---

## 🚀چیلنج

وقت کی سیریز ماڈل کی درستگی کو جانچنے کے مختلف طریقوں پر غور کریں۔ ہم نے اس سبق میں MAPE پر بات کی ہے، لیکن کیا آپ دوسرے طریقے استعمال کر سکتے ہیں؟ ان پر تحقیق کریں اور ان کا تجزیہ کریں۔ ایک مددگار دستاویز [یہاں](https://otexts.com/fpp2/accuracy.html) دستیاب ہے۔

## [لیکچر کے بعد کا کوئز](https://ff-quizzes.netlify.app/en/ml/)

## جائزہ اور خود مطالعہ

یہ سبق ARIMA کے ساتھ وقت کی سیریز کی پیش گوئی کے صرف بنیادی اصولوں پر بات کرتا ہے۔ اپنے علم کو گہرا کرنے کے لیے کچھ وقت نکالیں اور [اس ریپوزٹری](https://microsoft.github.io/forecasting/) اور اس کے مختلف ماڈل اقسام کو دیکھیں تاکہ وقت کی سیریز کے ماڈل بنانے کے دوسرے طریقے سیکھ سکیں۔

## اسائنمنٹ

[ایک نیا ARIMA ماڈل](assignment.md)

---

**ڈسکلیمر**:  
یہ دستاویز AI ترجمہ سروس [Co-op Translator](https://github.com/Azure/co-op-translator) کا استعمال کرتے ہوئے ترجمہ کی گئی ہے۔ ہم درستگی کے لیے کوشش کرتے ہیں، لیکن براہ کرم آگاہ رہیں کہ خودکار ترجمے میں غلطیاں یا غیر درستیاں ہو سکتی ہیں۔ اصل دستاویز کو اس کی اصل زبان میں مستند ذریعہ سمجھا جانا چاہیے۔ اہم معلومات کے لیے، پیشہ ور انسانی ترجمہ کی سفارش کی جاتی ہے۔ ہم اس ترجمے کے استعمال سے پیدا ہونے والی کسی بھی غلط فہمی یا غلط تشریح کے ذمہ دار نہیں ہیں۔