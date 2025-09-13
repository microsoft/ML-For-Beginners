<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "aaf391d922bd6de5efba871d514c6d47",
  "translation_date": "2025-09-04T20:50:26+00:00",
  "source_file": "4-Classification/1-Introduction/README.md",
  "language_code": "ar"
}
-->
# مقدمة إلى التصنيف

في هذه الدروس الأربعة، ستستكشف أحد الجوانب الأساسية لتعلم الآلة الكلاسيكي - _التصنيف_. سنقوم باستخدام مجموعة متنوعة من خوارزميات التصنيف مع مجموعة بيانات عن جميع المأكولات الرائعة في آسيا والهند. نأمل أن تكون جائعًا!

![مجرد رشة!](../../../../4-Classification/1-Introduction/images/pinch.png)

> احتفل بالمأكولات الآسيوية في هذه الدروس! الصورة بواسطة [Jen Looper](https://twitter.com/jenlooper)

التصنيف هو شكل من أشكال [التعلم الموجّه](https://wikipedia.org/wiki/Supervised_learning) الذي يشترك كثيرًا مع تقنيات الانحدار. إذا كان تعلم الآلة يدور حول التنبؤ بالقيم أو الأسماء باستخدام مجموعات البيانات، فإن التصنيف ينقسم عمومًا إلى مجموعتين: _التصنيف الثنائي_ و _التصنيف متعدد الفئات_.

[![مقدمة إلى التصنيف](https://img.youtube.com/vi/eg8DJYwdMyg/0.jpg)](https://youtu.be/eg8DJYwdMyg "مقدمة إلى التصنيف")

> 🎥 انقر على الصورة أعلاه لمشاهدة الفيديو: يقدم جون غوتاج من MIT التصنيف

تذكر:

- **الانحدار الخطي** ساعدك في التنبؤ بالعلاقات بين المتغيرات وإجراء توقعات دقيقة حول مكان وقوع نقطة بيانات جديدة بالنسبة لذلك الخط. على سبيل المثال، يمكنك التنبؤ _بسعر اليقطين في سبتمبر مقابل ديسمبر_.
- **الانحدار اللوجستي** ساعدك في اكتشاف "الفئات الثنائية": عند نقطة السعر هذه، _هل هذا اليقطين برتقالي أم غير برتقالي_؟

يستخدم التصنيف خوارزميات مختلفة لتحديد طرق أخرى لتحديد تسمية أو فئة نقطة البيانات. دعونا نعمل مع بيانات المأكولات لنرى ما إذا كان بإمكاننا، من خلال مراقبة مجموعة من المكونات، تحديد أصل المأكولات.

## [اختبار ما قبل المحاضرة](https://ff-quizzes.netlify.app/en/ml/)

> ### [هذا الدرس متاح بلغة R!](../../../../4-Classification/1-Introduction/solution/R/lesson_10.html)

### مقدمة

التصنيف هو أحد الأنشطة الأساسية للباحثين في تعلم الآلة وعلماء البيانات. من التصنيف الأساسي لقيمة ثنائية ("هل هذا البريد الإلكتروني مزعج أم لا؟") إلى التصنيف المعقد للصور وتقسيمها باستخدام رؤية الكمبيوتر، من المفيد دائمًا أن تكون قادرًا على تصنيف البيانات إلى فئات وطرح الأسئلة عليها.

لصياغة العملية بطريقة أكثر علمية، فإن طريقة التصنيف الخاصة بك تنشئ نموذجًا تنبؤيًا يمكّنك من رسم العلاقة بين المتغيرات المدخلة والمتغيرات الناتجة.

![التصنيف الثنائي مقابل التصنيف متعدد الفئات](../../../../4-Classification/1-Introduction/images/binary-multiclass.png)

> مشاكل التصنيف الثنائي مقابل متعدد الفئات التي تتعامل معها خوارزميات التصنيف. الرسم البياني بواسطة [Jen Looper](https://twitter.com/jenlooper)

قبل البدء في عملية تنظيف بياناتنا، تصورها، وتجهيزها لمهام تعلم الآلة، دعونا نتعلم قليلاً عن الطرق المختلفة التي يمكن من خلالها استخدام تعلم الآلة لتصنيف البيانات.

مستمدة من [الإحصائيات](https://wikipedia.org/wiki/Statistical_classification)، يستخدم التصنيف باستخدام تعلم الآلة الكلاسيكي ميزات مثل `smoker`، `weight`، و `age` لتحديد _احتمالية الإصابة بمرض معين_. كطريقة تعلم موجّهة مشابهة لتمارين الانحدار التي أجريتها سابقًا، يتم تصنيف بياناتك وتستخدم خوارزميات تعلم الآلة هذه التصنيفات لتصنيف وتوقع الفئات (أو "الميزات") لمجموعة البيانات وتعيينها إلى مجموعة أو نتيجة.

✅ خذ لحظة لتخيل مجموعة بيانات عن المأكولات. ما الذي يمكن لنموذج متعدد الفئات الإجابة عليه؟ وما الذي يمكن لنموذج ثنائي الإجابة عليه؟ ماذا لو أردت تحديد ما إذا كانت مأكولات معينة من المحتمل أن تستخدم الحلبة؟ ماذا لو أردت معرفة ما إذا كان بإمكانك، بالنظر إلى حقيبة بقالة مليئة باليانسون النجمي، الخرشوف، القرنبيط، والفجل، إعداد طبق هندي نموذجي؟

[![سلال غامضة مجنونة](https://img.youtube.com/vi/GuTeDbaNoEU/0.jpg)](https://youtu.be/GuTeDbaNoEU "سلال غامضة مجنونة")

> 🎥 انقر على الصورة أعلاه لمشاهدة الفيديو. الفكرة الأساسية لبرنامج 'Chopped' هي "السلة الغامضة" حيث يتعين على الطهاة إعداد طبق من اختيار عشوائي من المكونات. بالتأكيد كان نموذج تعلم الآلة سيساعد!

## مرحبًا بـ 'المصنف'

السؤال الذي نريد طرحه على مجموعة بيانات المأكولات هو في الواقع سؤال **متعدد الفئات**، حيث لدينا العديد من المأكولات الوطنية المحتملة للعمل معها. بالنظر إلى مجموعة من المكونات، أي من هذه الفئات العديدة ستتناسب معها البيانات؟

يوفر Scikit-learn العديد من الخوارزميات المختلفة لاستخدامها لتصنيف البيانات، اعتمادًا على نوع المشكلة التي تريد حلها. في الدروس التالية، ستتعلم عن العديد من هذه الخوارزميات.

## تمرين - تنظيف وتوازن البيانات

المهمة الأولى التي يجب القيام بها، قبل بدء هذا المشروع، هي تنظيف وتوازن البيانات للحصول على نتائج أفضل. ابدأ بملف _notebook.ipynb_ الفارغ الموجود في جذر هذا المجلد.

أول شيء يجب تثبيته هو [imblearn](https://imbalanced-learn.org/stable/). هذه حزمة Scikit-learn ستتيح لك تحقيق توازن أفضل للبيانات (ستتعلم المزيد عن هذه المهمة قريبًا).

1. لتثبيت `imblearn`، قم بتشغيل `pip install`، كما يلي:

    ```python
    pip install imblearn
    ```

1. قم باستيراد الحزم التي تحتاجها لاستيراد بياناتك وتصورها، واستيراد `SMOTE` من `imblearn`.

    ```python
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    import numpy as np
    from imblearn.over_sampling import SMOTE
    ```

    الآن أصبحت جاهزًا لاستيراد البيانات.

1. المهمة التالية ستكون استيراد البيانات:

    ```python
    df  = pd.read_csv('../data/cuisines.csv')
    ```

   استخدام `read_csv()` سيقرأ محتوى ملف csv _cusines.csv_ ويضعه في المتغير `df`.

1. تحقق من شكل البيانات:

    ```python
    df.head()
    ```

   تبدو الصفوف الخمسة الأولى كما يلي:

    ```output
    |     | Unnamed: 0 | cuisine | almond | angelica | anise | anise_seed | apple | apple_brandy | apricot | armagnac | ... | whiskey | white_bread | white_wine | whole_grain_wheat_flour | wine | wood | yam | yeast | yogurt | zucchini |
    | --- | ---------- | ------- | ------ | -------- | ----- | ---------- | ----- | ------------ | ------- | -------- | --- | ------- | ----------- | ---------- | ----------------------- | ---- | ---- | --- | ----- | ------ | -------- |
    | 0   | 65         | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
    | 1   | 66         | indian  | 1      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
    | 2   | 67         | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
    | 3   | 68         | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
    | 4   | 69         | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 1      | 0        |
    ```

1. احصل على معلومات حول هذه البيانات عن طريق استدعاء `info()`:

    ```python
    df.info()
    ```

    يشبه الناتج:

    ```output
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 2448 entries, 0 to 2447
    Columns: 385 entries, Unnamed: 0 to zucchini
    dtypes: int64(384), object(1)
    memory usage: 7.2+ MB
    ```

## تمرين - التعرف على المأكولات

الآن يبدأ العمل ليصبح أكثر إثارة. دعونا نكتشف توزيع البيانات لكل نوع من المأكولات.

1. قم برسم البيانات كأشرطة باستخدام `barh()`:

    ```python
    df.cuisine.value_counts().plot.barh()
    ```

    ![توزيع بيانات المأكولات](../../../../4-Classification/1-Introduction/images/cuisine-dist.png)

    هناك عدد محدود من المأكولات، لكن توزيع البيانات غير متساوٍ. يمكنك إصلاح ذلك! قبل القيام بذلك، استكشف قليلاً.

1. اكتشف مقدار البيانات المتاحة لكل نوع من المأكولات واطبعها:

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

    يبدو الناتج كما يلي:

    ```output
    thai df: (289, 385)
    japanese df: (320, 385)
    chinese df: (442, 385)
    indian df: (598, 385)
    korean df: (799, 385)
    ```

## اكتشاف المكونات

الآن يمكنك التعمق أكثر في البيانات ومعرفة ما هي المكونات النموذجية لكل نوع من المأكولات. يجب تنظيف البيانات المتكررة التي تسبب ارتباكًا بين المأكولات، لذا دعونا نتعلم عن هذه المشكلة.

1. قم بإنشاء وظيفة `create_ingredient()` في Python لإنشاء إطار بيانات للمكونات. ستبدأ هذه الوظيفة بإسقاط عمود غير مفيد وفرز المكونات حسب عددها:

    ```python
    def create_ingredient_df(df):
        ingredient_df = df.T.drop(['cuisine','Unnamed: 0']).sum(axis=1).to_frame('value')
        ingredient_df = ingredient_df[(ingredient_df.T != 0).any()]
        ingredient_df = ingredient_df.sort_values(by='value', ascending=False,
        inplace=False)
        return ingredient_df
    ```

   الآن يمكنك استخدام هذه الوظيفة للحصول على فكرة عن أكثر عشرة مكونات شيوعًا لكل نوع من المأكولات.

1. قم باستدعاء `create_ingredient()` وقم برسمها باستخدام `barh()`:

    ```python
    thai_ingredient_df = create_ingredient_df(thai_df)
    thai_ingredient_df.head(10).plot.barh()
    ```

    ![التايلاندية](../../../../4-Classification/1-Introduction/images/thai.png)

1. قم بنفس الشيء لبيانات المأكولات اليابانية:

    ```python
    japanese_ingredient_df = create_ingredient_df(japanese_df)
    japanese_ingredient_df.head(10).plot.barh()
    ```

    ![اليابانية](../../../../4-Classification/1-Introduction/images/japanese.png)

1. الآن بالنسبة لمكونات المأكولات الصينية:

    ```python
    chinese_ingredient_df = create_ingredient_df(chinese_df)
    chinese_ingredient_df.head(10).plot.barh()
    ```

    ![الصينية](../../../../4-Classification/1-Introduction/images/chinese.png)

1. قم برسم مكونات المأكولات الهندية:

    ```python
    indian_ingredient_df = create_ingredient_df(indian_df)
    indian_ingredient_df.head(10).plot.barh()
    ```

    ![الهندية](../../../../4-Classification/1-Introduction/images/indian.png)

1. أخيرًا، قم برسم مكونات المأكولات الكورية:

    ```python
    korean_ingredient_df = create_ingredient_df(korean_df)
    korean_ingredient_df.head(10).plot.barh()
    ```

1. الآن، قم بإسقاط المكونات الأكثر شيوعًا التي تسبب ارتباكًا بين المأكولات المختلفة، باستخدام `drop()`:

   الجميع يحب الأرز، الثوم، والزنجبيل!

    ```python
    feature_df= df.drop(['cuisine','Unnamed: 0','rice','garlic','ginger'], axis=1)
    labels_df = df.cuisine #.unique()
    feature_df.head()
    ```

## توازن مجموعة البيانات

الآن بعد أن قمت بتنظيف البيانات، استخدم [SMOTE](https://imbalanced-learn.org/dev/references/generated/imblearn.over_sampling.SMOTE.html) - "تقنية الإفراط في أخذ العينات للأقلية الاصطناعية" - لتحقيق التوازن.

1. قم باستدعاء `fit_resample()`، هذه الاستراتيجية تولد عينات جديدة عن طريق الاستيفاء.

    ```python
    oversample = SMOTE()
    transformed_feature_df, transformed_label_df = oversample.fit_resample(feature_df, labels_df)
    ```

    من خلال تحقيق التوازن في بياناتك، ستحصل على نتائج أفضل عند تصنيفها. فكر في التصنيف الثنائي. إذا كانت معظم بياناتك تنتمي إلى فئة واحدة، فإن نموذج تعلم الآلة سيتنبأ بتلك الفئة بشكل أكثر تكرارًا، فقط لأن هناك المزيد من البيانات لها. تحقيق التوازن في البيانات يزيل أي انحراف ويساعد في حل هذه المشكلة.

1. الآن يمكنك التحقق من أعداد التصنيفات لكل مكون:

    ```python
    print(f'new label count: {transformed_label_df.value_counts()}')
    print(f'old label count: {df.cuisine.value_counts()}')
    ```

    يبدو الناتج كما يلي:

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

    البيانات نظيفة ومتوازنة ولذيذة جدًا!

1. الخطوة الأخيرة هي حفظ بياناتك المتوازنة، بما في ذلك التصنيفات والميزات، في إطار بيانات جديد يمكن تصديره إلى ملف:

    ```python
    transformed_df = pd.concat([transformed_label_df,transformed_feature_df],axis=1, join='outer')
    ```

1. يمكنك إلقاء نظرة أخيرة على البيانات باستخدام `transformed_df.head()` و `transformed_df.info()`. احفظ نسخة من هذه البيانات لاستخدامها في الدروس المستقبلية:

    ```python
    transformed_df.head()
    transformed_df.info()
    transformed_df.to_csv("../data/cleaned_cuisines.csv")
    ```

    يمكن الآن العثور على ملف CSV الجديد في مجلد البيانات الجذر.

---

## 🚀تحدي

تحتوي هذه المناهج على العديد من مجموعات البيانات المثيرة للاهتمام. ابحث في مجلدات `data` لترى ما إذا كانت تحتوي على مجموعات بيانات مناسبة للتصنيف الثنائي أو متعدد الفئات؟ ما الأسئلة التي ستطرحها على هذه المجموعة؟

## [اختبار ما بعد المحاضرة](https://ff-quizzes.netlify.app/en/ml/)

## المراجعة والدراسة الذاتية

استكشف واجهة برمجة تطبيقات SMOTE. ما هي حالات الاستخدام التي تناسبها؟ ما المشاكل التي تحلها؟

## الواجب 

[استكشاف طرق التصنيف](assignment.md)

---

**إخلاء المسؤولية**:  
تمت ترجمة هذا المستند باستخدام خدمة الترجمة الآلية [Co-op Translator](https://github.com/Azure/co-op-translator). بينما نسعى لتحقيق الدقة، يرجى العلم أن الترجمات الآلية قد تحتوي على أخطاء أو معلومات غير دقيقة. يجب اعتبار المستند الأصلي بلغته الأصلية هو المصدر الموثوق. للحصول على معلومات حساسة أو هامة، يُوصى بالاستعانة بترجمة بشرية احترافية. نحن غير مسؤولين عن أي سوء فهم أو تفسيرات خاطئة تنشأ عن استخدام هذه الترجمة.