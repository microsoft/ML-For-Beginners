<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "917dbf890db71a322f306050cb284749",
  "translation_date": "2025-09-04T20:41:47+00:00",
  "source_file": "7-TimeSeries/2-ARIMA/README.md",
  "language_code": "ar"
}
-->
# التنبؤ بالسلاسل الزمنية باستخدام ARIMA

في الدرس السابق، تعلمت قليلاً عن التنبؤ بالسلاسل الزمنية وقمت بتحميل مجموعة بيانات تُظهر تقلبات الحمل الكهربائي على مدى فترة زمنية.

[![مقدمة إلى ARIMA](https://img.youtube.com/vi/IUSk-YDau10/0.jpg)](https://youtu.be/IUSk-YDau10 "Introduction to ARIMA")

> 🎥 انقر على الصورة أعلاه لمشاهدة فيديو: مقدمة قصيرة عن نماذج ARIMA. المثال مُنفذ باستخدام R، لكن المفاهيم عامة.

## [اختبار ما قبل المحاضرة](https://ff-quizzes.netlify.app/en/ml/)

## المقدمة

في هذا الدرس، ستتعرف على طريقة محددة لبناء النماذج باستخدام [ARIMA: *A*uto*R*egressive *I*ntegrated *M*oving *A*verage](https://wikipedia.org/wiki/Autoregressive_integrated_moving_average). تُعتبر نماذج ARIMA مناسبة بشكل خاص لتلائم البيانات التي تُظهر [عدم الثبات](https://wikipedia.org/wiki/Stationary_process).

## المفاهيم العامة

لكي تتمكن من العمل مع ARIMA، هناك بعض المفاهيم التي تحتاج إلى معرفتها:

- 🎓 **الثبات**. من منظور إحصائي، يشير الثبات إلى البيانات التي لا يتغير توزيعها عند تحريكها عبر الزمن. أما البيانات غير الثابتة، فتُظهر تقلبات بسبب الاتجاهات التي يجب تحويلها لتحليلها. على سبيل المثال، يمكن أن تُدخل الموسمية تقلبات في البيانات ويمكن التخلص منها من خلال عملية "الفرق الموسمي".

- 🎓 **[الفرق](https://wikipedia.org/wiki/Autoregressive_integrated_moving_average#Differencing)**. يشير الفرق في البيانات، من منظور إحصائي، إلى عملية تحويل البيانات غير الثابتة لجعلها ثابتة عن طريق إزالة الاتجاه غير الثابت. "الفرق يزيل التغيرات في مستوى السلسلة الزمنية، مما يلغي الاتجاه والموسمية وبالتالي يثبت المتوسط للسلسلة الزمنية." [ورقة بحثية لـ Shixiong وآخرين](https://arxiv.org/abs/1904.07632)

## ARIMA في سياق السلاسل الزمنية

دعونا نفكك أجزاء ARIMA لفهم كيفية مساعدتها في نمذجة السلاسل الزمنية ومساعدتنا في التنبؤ بها.

- **AR - الانحدار الذاتي**. كما يوحي الاسم، تنظر النماذج الانحدارية الذاتية إلى "الماضي" لتحليل القيم السابقة في بياناتك وتكوين افتراضات حولها. تُسمى هذه القيم السابقة بـ "الفجوات الزمنية". مثال على ذلك هو البيانات التي تُظهر مبيعات الأقلام الشهرية. إجمالي مبيعات كل شهر يُعتبر "متغيرًا متطورًا" في مجموعة البيانات. يتم بناء هذا النموذج على أساس أن "المتغير المتطور محل الاهتمام يتم انحداره على قيمه السابقة (أي الفجوات الزمنية)." [ويكيبيديا](https://wikipedia.org/wiki/Autoregressive_integrated_moving_average)

- **I - التكامل**. على عكس النماذج المشابهة مثل 'ARMA'، يشير الحرف 'I' في ARIMA إلى الجانب *[المتكامل](https://wikipedia.org/wiki/Order_of_integration)*. يتم "تكامل" البيانات عند تطبيق خطوات الفرق لإزالة عدم الثبات.

- **MA - المتوسط المتحرك**. يشير جانب [المتوسط المتحرك](https://wikipedia.org/wiki/Moving-average_model) في هذا النموذج إلى المتغير الناتج الذي يتم تحديده من خلال مراقبة القيم الحالية والسابقة للفجوات الزمنية.

الخلاصة: تُستخدم ARIMA لجعل النموذج يتلاءم مع الشكل الخاص لبيانات السلاسل الزمنية بأكبر قدر ممكن من الدقة.

## تمرين - بناء نموذج ARIMA

افتح المجلد [_/working_](https://github.com/microsoft/ML-For-Beginners/tree/main/7-TimeSeries/2-ARIMA/working) في هذا الدرس وابحث عن الملف [_notebook.ipynb_](https://github.com/microsoft/ML-For-Beginners/blob/main/7-TimeSeries/2-ARIMA/working/notebook.ipynb).

1. قم بتشغيل الدفتر لتحميل مكتبة Python `statsmodels`؛ ستحتاجها لنماذج ARIMA.

1. قم بتحميل المكتبات اللازمة.

1. الآن، قم بتحميل المزيد من المكتبات المفيدة لرسم البيانات:

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

1. قم بتحميل البيانات من ملف `/data/energy.csv` إلى إطار بيانات Pandas وألقِ نظرة:

    ```python
    energy = load_data('./data')[['load']]
    energy.head(10)
    ```

1. قم برسم جميع بيانات الطاقة المتاحة من يناير 2012 إلى ديسمبر 2014. لا ينبغي أن تكون هناك مفاجآت لأننا رأينا هذه البيانات في الدرس السابق:

    ```python
    energy.plot(y='load', subplots=True, figsize=(15, 8), fontsize=12)
    plt.xlabel('timestamp', fontsize=12)
    plt.ylabel('load', fontsize=12)
    plt.show()
    ```

    الآن، دعونا نبني نموذجًا!

### إنشاء مجموعات بيانات التدريب والاختبار

الآن بعد أن تم تحميل بياناتك، يمكنك فصلها إلى مجموعات تدريب واختبار. ستقوم بتدريب النموذج الخاص بك على مجموعة التدريب. وكالعادة، بعد أن ينتهي النموذج من التدريب، ستقوم بتقييم دقته باستخدام مجموعة الاختبار. تحتاج إلى التأكد من أن مجموعة الاختبار تغطي فترة زمنية لاحقة لمجموعة التدريب لضمان أن النموذج لا يحصل على معلومات من فترات زمنية مستقبلية.

1. خصص فترة شهرين من 1 سبتمبر إلى 31 أكتوبر 2014 لمجموعة التدريب. ستشمل مجموعة الاختبار فترة الشهرين من 1 نوفمبر إلى 31 ديسمبر 2014:

    ```python
    train_start_dt = '2014-11-01 00:00:00'
    test_start_dt = '2014-12-30 00:00:00'
    ```

    نظرًا لأن هذه البيانات تعكس استهلاك الطاقة اليومي، هناك نمط موسمي قوي، ولكن الاستهلاك يكون أكثر تشابهًا مع الاستهلاك في الأيام الأخيرة.

1. قم بتصور الفروقات:

    ```python
    energy[(energy.index < test_start_dt) & (energy.index >= train_start_dt)][['load']].rename(columns={'load':'train'}) \
        .join(energy[test_start_dt:][['load']].rename(columns={'load':'test'}), how='outer') \
        .plot(y=['train', 'test'], figsize=(15, 8), fontsize=12)
    plt.xlabel('timestamp', fontsize=12)
    plt.ylabel('load', fontsize=12)
    plt.show()
    ```

    ![بيانات التدريب والاختبار](../../../../7-TimeSeries/2-ARIMA/images/train-test.png)

    لذلك، استخدام نافذة زمنية صغيرة نسبيًا لتدريب البيانات يجب أن يكون كافيًا.

    > ملاحظة: نظرًا لأن الوظيفة التي نستخدمها لتلائم نموذج ARIMA تستخدم التحقق داخل العينة أثناء التلاؤم، سنقوم بتجاهل بيانات التحقق.

### تحضير البيانات للتدريب

الآن، تحتاج إلى تحضير البيانات للتدريب عن طريق تصفيتها وتوسيع نطاقها. قم بتصفية مجموعة البيانات الخاصة بك لتشمل فقط الفترات الزمنية والأعمدة التي تحتاجها، وقم بتوسيع النطاق لضمان أن البيانات تقع في النطاق 0,1.

1. قم بتصفية مجموعة البيانات الأصلية لتشمل فقط الفترات الزمنية المذكورة لكل مجموعة، مع تضمين العمود المطلوب 'load' بالإضافة إلى التاريخ:

    ```python
    train = energy.copy()[(energy.index >= train_start_dt) & (energy.index < test_start_dt)][['load']]
    test = energy.copy()[energy.index >= test_start_dt][['load']]

    print('Training data shape: ', train.shape)
    print('Test data shape: ', test.shape)
    ```

    يمكنك رؤية شكل البيانات:

    ```output
    Training data shape:  (1416, 1)
    Test data shape:  (48, 1)
    ```

1. قم بتوسيع نطاق البيانات لتكون في النطاق (0, 1).

    ```python
    scaler = MinMaxScaler()
    train['load'] = scaler.fit_transform(train)
    train.head(10)
    ```

1. قم بتصور البيانات الأصلية مقابل البيانات الموسعة:

    ```python
    energy[(energy.index >= train_start_dt) & (energy.index < test_start_dt)][['load']].rename(columns={'load':'original load'}).plot.hist(bins=100, fontsize=12)
    train.rename(columns={'load':'scaled load'}).plot.hist(bins=100, fontsize=12)
    plt.show()
    ```

    ![الأصلية](../../../../7-TimeSeries/2-ARIMA/images/original.png)

    > البيانات الأصلية

    ![الموسعة](../../../../7-TimeSeries/2-ARIMA/images/scaled.png)

    > البيانات الموسعة

1. الآن بعد أن قمت بمعايرة البيانات الموسعة، يمكنك توسيع نطاق بيانات الاختبار:

    ```python
    test['load'] = scaler.transform(test)
    test.head()
    ```

### تنفيذ ARIMA

حان الوقت لتنفيذ ARIMA! ستستخدم الآن مكتبة `statsmodels` التي قمت بتثبيتها سابقًا.

الآن تحتاج إلى اتباع عدة خطوات:

   1. قم بتعريف النموذج عن طريق استدعاء `SARIMAX()` وتمرير معلمات النموذج: المعلمات p، d، و q، بالإضافة إلى المعلمات P، D، و Q.
   2. قم بتحضير النموذج لبيانات التدريب عن طريق استدعاء وظيفة `fit()`.
   3. قم بإجراء التنبؤات عن طريق استدعاء وظيفة `forecast()` وتحديد عدد الخطوات (الأفق) للتنبؤ.

> 🎓 ما هي كل هذه المعلمات؟ في نموذج ARIMA، هناك 3 معلمات تُستخدم للمساعدة في نمذجة الجوانب الرئيسية للسلسلة الزمنية: الموسمية، الاتجاه، والضوضاء. هذه المعلمات هي:

`p`: المعلمة المرتبطة بجانب الانحدار الذاتي للنموذج، والذي يدمج القيم *السابقة*.

`d`: المعلمة المرتبطة بالجزء المتكامل للنموذج، والتي تؤثر على مقدار *الفرق* (🎓 تذكر الفرق 👆؟) الذي يتم تطبيقه على السلسلة الزمنية.

`q`: المعلمة المرتبطة بجانب المتوسط المتحرك للنموذج.

> ملاحظة: إذا كانت بياناتك تحتوي على جانب موسمي - كما هو الحال هنا -، نستخدم نموذج ARIMA الموسمي (SARIMA). في هذه الحالة، تحتاج إلى استخدام مجموعة أخرى من المعلمات: `P`، `D`، و `Q` التي تصف نفس الارتباطات مثل `p`، `d`، و `q`، ولكنها تتعلق بالمكونات الموسمية للنموذج.

1. ابدأ بتحديد قيمة الأفق المفضلة لديك. لنحاول 3 ساعات:

    ```python
    # Specify the number of steps to forecast ahead
    HORIZON = 3
    print('Forecasting horizon:', HORIZON, 'hours')
    ```

    اختيار أفضل القيم لمعلمات نموذج ARIMA يمكن أن يكون تحديًا لأنه يعتمد إلى حد كبير على التقدير الشخصي ويستغرق وقتًا. يمكنك التفكير في استخدام وظيفة `auto_arima()` من مكتبة [`pyramid`](https://alkaline-ml.com/pmdarima/0.9.0/modules/generated/pyramid.arima.auto_arima.html).

1. في الوقت الحالي، جرب بعض الاختيارات اليدوية للعثور على نموذج جيد.

    ```python
    order = (4, 1, 0)
    seasonal_order = (1, 1, 0, 24)

    model = SARIMAX(endog=train, order=order, seasonal_order=seasonal_order)
    results = model.fit()

    print(results.summary())
    ```

    يتم طباعة جدول النتائج.

لقد قمت ببناء نموذجك الأول! الآن نحتاج إلى إيجاد طريقة لتقييمه.

### تقييم النموذج الخاص بك

لتقييم النموذج الخاص بك، يمكنك تنفيذ ما يُسمى بـ `التحقق التدريجي`. في الممارسة العملية، يتم إعادة تدريب نماذج السلاسل الزمنية في كل مرة تصبح فيها بيانات جديدة متاحة. يسمح هذا للنموذج بتقديم أفضل تنبؤ في كل خطوة زمنية.

ابدأ من بداية السلسلة الزمنية باستخدام هذه التقنية، قم بتدريب النموذج على مجموعة بيانات التدريب. ثم قم بإجراء تنبؤ على الخطوة الزمنية التالية. يتم تقييم التنبؤ مقابل القيمة المعروفة. يتم بعد ذلك توسيع مجموعة التدريب لتشمل القيمة المعروفة ويتم تكرار العملية.

> ملاحظة: يجب أن تحافظ على نافذة مجموعة التدريب ثابتة لتحقيق تدريب أكثر كفاءة، بحيث في كل مرة تضيف ملاحظة جديدة إلى مجموعة التدريب، تقوم بإزالة الملاحظة من بداية المجموعة.

توفر هذه العملية تقديرًا أكثر دقة لكيفية أداء النموذج في الممارسة العملية. ومع ذلك، فإنها تأتي بتكلفة حسابية بسبب إنشاء العديد من النماذج. هذا مقبول إذا كانت البيانات صغيرة أو إذا كان النموذج بسيطًا، ولكنه قد يكون مشكلة على نطاق واسع.

يُعتبر التحقق التدريجي المعيار الذهبي لتقييم نماذج السلاسل الزمنية ويوصى به لمشاريعك الخاصة.

1. أولاً، قم بإنشاء نقطة بيانات اختبار لكل خطوة أفقية.

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

    يتم إزاحة البيانات أفقيًا وفقًا لنقطة الأفق.

1. قم بإجراء التنبؤات على بيانات الاختبار باستخدام هذا النهج المتحرك في حلقة بحجم طول بيانات الاختبار:

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

    يمكنك مشاهدة عملية التدريب:

    ```output
    2014-12-30 00:00:00
    1 : predicted = [0.32 0.29 0.28] expected = [0.32945389435989236, 0.2900626678603402, 0.2739480752014323]

    2014-12-30 01:00:00
    2 : predicted = [0.3  0.29 0.3 ] expected = [0.2900626678603402, 0.2739480752014323, 0.26812891674127126]

    2014-12-30 02:00:00
    3 : predicted = [0.27 0.28 0.32] expected = [0.2739480752014323, 0.26812891674127126, 0.3025962399283795]
    ```

1. قارن التنبؤات بالحمل الفعلي:

    ```python
    eval_df = pd.DataFrame(predictions, columns=['t+'+str(t) for t in range(1, HORIZON+1)])
    eval_df['timestamp'] = test.index[0:len(test.index)-HORIZON+1]
    eval_df = pd.melt(eval_df, id_vars='timestamp', value_name='prediction', var_name='h')
    eval_df['actual'] = np.array(np.transpose(test_ts)).ravel()
    eval_df[['prediction', 'actual']] = scaler.inverse_transform(eval_df[['prediction', 'actual']])
    eval_df.head()
    ```

    المخرجات
    |     |            | timestamp | h   | prediction | actual   |
    | --- | ---------- | --------- | --- | ---------- | -------- |
    | 0   | 2014-12-30 | 00:00:00  | t+1 | 3,008.74   | 3,023.00 |
    | 1   | 2014-12-30 | 01:00:00  | t+1 | 2,955.53   | 2,935.00 |
    | 2   | 2014-12-30 | 02:00:00  | t+1 | 2,900.17   | 2,899.00 |
    | 3   | 2014-12-30 | 03:00:00  | t+1 | 2,917.69   | 2,886.00 |
    | 4   | 2014-12-30 | 04:00:00  | t+1 | 2,946.99   | 2,963.00 |

    لاحظ التنبؤ بالبيانات لكل ساعة، مقارنة بالحمل الفعلي. ما مدى دقة هذا؟

### تحقق من دقة النموذج

تحقق من دقة النموذج الخاص بك عن طريق اختبار متوسط نسبة الخطأ المطلق (MAPE) لجميع التنبؤات.
> **🧮 أظهر لي الرياضيات**
>
> ![MAPE](../../../../7-TimeSeries/2-ARIMA/images/mape.png)
>
> [MAPE](https://www.linkedin.com/pulse/what-mape-mad-msd-time-series-allameh-statistics/) تُستخدم لعرض دقة التنبؤ كنسبة تُعرّفها الصيغة أعلاه. يتم قسمة الفرق بين القيم الفعلية والمتوقعة على القيم الفعلية.  
> "يتم جمع القيمة المطلقة في هذا الحساب لكل نقطة متوقعة في الزمن وقسمتها على عدد النقاط الملائمة n." [wikipedia](https://wikipedia.org/wiki/Mean_absolute_percentage_error)
1. عبر عن المعادلة في الكود:

    ```python
    if(HORIZON > 1):
        eval_df['APE'] = (eval_df['prediction'] - eval_df['actual']).abs() / eval_df['actual']
        print(eval_df.groupby('h')['APE'].mean())
    ```

1. احسب MAPE لخطوة واحدة:

    ```python
    print('One step forecast MAPE: ', (mape(eval_df[eval_df['h'] == 't+1']['prediction'], eval_df[eval_df['h'] == 't+1']['actual']))*100, '%')
    ```

    MAPE لتوقع خطوة واحدة: 0.5570581332313952 %

1. اطبع MAPE لتوقع متعدد الخطوات:

    ```python
    print('Multi-step forecast MAPE: ', mape(eval_df['prediction'], eval_df['actual'])*100, '%')
    ```

    ```output
    Multi-step forecast MAPE:  1.1460048657704118 %
    ```

    رقم منخفض جميل هو الأفضل: ضع في اعتبارك أن التوقع الذي يحتوي على MAPE بقيمة 10 يعني أنه خاطئ بنسبة 10%.

1. ولكن كما هو الحال دائمًا، من الأسهل رؤية هذا النوع من قياس الدقة بصريًا، لذا دعنا نرسمه:

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

    ![نموذج سلسلة زمنية](../../../../7-TimeSeries/2-ARIMA/images/accuracy.png)

🏆 رسم جميل جدًا، يظهر نموذجًا بدقة جيدة. عمل رائع!

---

## 🚀التحدي

استكشف الطرق المختلفة لاختبار دقة نموذج السلاسل الزمنية. لقد تناولنا MAPE في هذا الدرس، ولكن هل هناك طرق أخرى يمكنك استخدامها؟ قم بالبحث عنها وقم بتوضيحها. يمكن العثور على وثيقة مفيدة [هنا](https://otexts.com/fpp2/accuracy.html)

## [اختبار ما بعد الدرس](https://ff-quizzes.netlify.app/en/ml/)

## المراجعة والدراسة الذاتية

يتناول هذا الدرس فقط أساسيات التنبؤ بالسلاسل الزمنية باستخدام ARIMA. خذ بعض الوقت لتعميق معرفتك من خلال استكشاف [هذا المستودع](https://microsoft.github.io/forecasting/) وأنواع النماذج المختلفة فيه لتتعلم طرقًا أخرى لبناء نماذج السلاسل الزمنية.

## الواجب

[نموذج ARIMA جديد](assignment.md)

---

**إخلاء المسؤولية**:  
تم ترجمة هذا المستند باستخدام خدمة الترجمة بالذكاء الاصطناعي [Co-op Translator](https://github.com/Azure/co-op-translator). بينما نسعى لتحقيق الدقة، يرجى العلم أن الترجمات الآلية قد تحتوي على أخطاء أو معلومات غير دقيقة. يجب اعتبار المستند الأصلي بلغته الأصلية هو المصدر الموثوق. للحصول على معلومات حاسمة، يُوصى بالاستعانة بترجمة بشرية احترافية. نحن غير مسؤولين عن أي سوء فهم أو تفسيرات خاطئة ناتجة عن استخدام هذه الترجمة.