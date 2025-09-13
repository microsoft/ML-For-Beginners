<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "7cdd17338d9bbd7e2171c2cd462eb081",
  "translation_date": "2025-09-04T20:44:23+00:00",
  "source_file": "5-Clustering/2-K-Means/README.md",
  "language_code": "ar"
}
-->
# التجميع باستخدام K-Means

## [اختبار ما قبل المحاضرة](https://ff-quizzes.netlify.app/en/ml/)

في هذه الدرس، ستتعلم كيفية إنشاء مجموعات باستخدام مكتبة Scikit-learn وبيانات الموسيقى النيجيرية التي قمت باستيرادها سابقًا. سنغطي أساسيات K-Means للتجميع. تذكر أنه كما تعلمت في الدرس السابق، هناك العديد من الطرق للعمل مع المجموعات والطريقة التي تستخدمها تعتمد على بياناتك. سنجرب K-Means لأنه التقنية الأكثر شيوعًا للتجميع. لنبدأ!

المصطلحات التي ستتعلم عنها:

- تقييم Silhouette
- طريقة الكوع
- القصور الذاتي (Inertia)
- التباين

## المقدمة

[التجميع باستخدام K-Means](https://wikipedia.org/wiki/K-means_clustering) هو طريقة مشتقة من مجال معالجة الإشارات. تُستخدم لتقسيم وتجزئة مجموعات البيانات إلى "k" مجموعات باستخدام سلسلة من الملاحظات. تعمل كل ملاحظة على تجميع نقطة البيانات الأقرب إلى "المتوسط" أو النقطة المركزية للمجموعة.

يمكن تصور المجموعات كـ [مخططات Voronoi](https://wikipedia.org/wiki/Voronoi_diagram)، والتي تتضمن نقطة (أو "بذرة") ومنطقتها المقابلة.

![مخطط Voronoi](../../../../5-Clustering/2-K-Means/images/voronoi.png)

> رسم توضيحي بواسطة [Jen Looper](https://twitter.com/jenlooper)

عملية التجميع باستخدام K-Means [تُنفذ في ثلاث خطوات](https://scikit-learn.org/stable/modules/clustering.html#k-means):

1. يختار الخوارزمية عددًا من النقاط المركزية "k" عن طريق أخذ عينات من مجموعة البيانات. بعد ذلك، تبدأ في التكرار:
    1. تُخصص كل عينة إلى أقرب نقطة مركزية.
    2. تُنشئ نقاط مركزية جديدة عن طريق حساب متوسط القيم لجميع العينات المخصصة للنقاط المركزية السابقة.
    3. ثم تُحسب الفرق بين النقاط المركزية الجديدة والقديمة وتكرر العملية حتى تستقر النقاط المركزية.

أحد العيوب في استخدام K-Means هو أنك ستحتاج إلى تحديد "k"، أي عدد النقاط المركزية. لحسن الحظ، تساعد طريقة "الكوع" في تقدير قيمة جيدة كبداية لـ "k". ستجربها قريبًا.

## المتطلبات الأساسية

ستعمل في ملف [_notebook.ipynb_](https://github.com/microsoft/ML-For-Beginners/blob/main/5-Clustering/2-K-Means/notebook.ipynb) الخاص بهذا الدرس، والذي يتضمن استيراد البيانات والتنظيف الأولي الذي قمت به في الدرس السابق.

## التمرين - التحضير

ابدأ بإلقاء نظرة أخرى على بيانات الأغاني.

1. أنشئ مخططًا صندوقيًا باستخدام `boxplot()` لكل عمود:

    ```python
    plt.figure(figsize=(20,20), dpi=200)
    
    plt.subplot(4,3,1)
    sns.boxplot(x = 'popularity', data = df)
    
    plt.subplot(4,3,2)
    sns.boxplot(x = 'acousticness', data = df)
    
    plt.subplot(4,3,3)
    sns.boxplot(x = 'energy', data = df)
    
    plt.subplot(4,3,4)
    sns.boxplot(x = 'instrumentalness', data = df)
    
    plt.subplot(4,3,5)
    sns.boxplot(x = 'liveness', data = df)
    
    plt.subplot(4,3,6)
    sns.boxplot(x = 'loudness', data = df)
    
    plt.subplot(4,3,7)
    sns.boxplot(x = 'speechiness', data = df)
    
    plt.subplot(4,3,8)
    sns.boxplot(x = 'tempo', data = df)
    
    plt.subplot(4,3,9)
    sns.boxplot(x = 'time_signature', data = df)
    
    plt.subplot(4,3,10)
    sns.boxplot(x = 'danceability', data = df)
    
    plt.subplot(4,3,11)
    sns.boxplot(x = 'length', data = df)
    
    plt.subplot(4,3,12)
    sns.boxplot(x = 'release_date', data = df)
    ```

    هذه البيانات تحتوي على بعض الضوضاء: من خلال ملاحظة كل عمود كمخطط صندوقي، يمكنك رؤية القيم الشاذة.

    ![القيم الشاذة](../../../../5-Clustering/2-K-Means/images/boxplots.png)

يمكنك المرور عبر مجموعة البيانات وإزالة هذه القيم الشاذة، ولكن ذلك سيجعل البيانات قليلة جدًا.

1. في الوقت الحالي، اختر الأعمدة التي ستستخدمها في تمرين التجميع. اختر الأعمدة ذات النطاقات المتشابهة وقم بترميز عمود `artist_top_genre` كبيانات رقمية:

    ```python
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    
    X = df.loc[:, ('artist_top_genre','popularity','danceability','acousticness','loudness','energy')]
    
    y = df['artist_top_genre']
    
    X['artist_top_genre'] = le.fit_transform(X['artist_top_genre'])
    
    y = le.transform(y)
    ```

1. الآن تحتاج إلى اختيار عدد المجموعات المستهدفة. تعرف أن هناك 3 أنواع من الأغاني التي استخرجناها من مجموعة البيانات، لذا دعنا نجرب 3:

    ```python
    from sklearn.cluster import KMeans
    
    nclusters = 3 
    seed = 0
    
    km = KMeans(n_clusters=nclusters, random_state=seed)
    km.fit(X)
    
    # Predict the cluster for each data point
    
    y_cluster_kmeans = km.predict(X)
    y_cluster_kmeans
    ```

سترى مصفوفة مطبوعة تحتوي على المجموعات المتوقعة (0، 1، أو 2) لكل صف من إطار البيانات.

1. استخدم هذه المصفوفة لحساب "تقييم Silhouette":

    ```python
    from sklearn import metrics
    score = metrics.silhouette_score(X, y_cluster_kmeans)
    score
    ```

## تقييم Silhouette

ابحث عن تقييم Silhouette أقرب إلى 1. يتراوح هذا التقييم بين -1 و 1، وإذا كان التقييم 1، فإن المجموعة كثيفة ومفصولة جيدًا عن المجموعات الأخرى. القيمة القريبة من 0 تمثل مجموعات متداخلة مع عينات قريبة جدًا من حدود القرار للمجموعات المجاورة. [(المصدر)](https://dzone.com/articles/kmeans-silhouette-score-explained-with-python-exam)

تقييمنا هو **0.53**، لذا فهو في المنتصف. يشير ذلك إلى أن بياناتنا ليست مناسبة بشكل خاص لهذا النوع من التجميع، ولكن دعنا نواصل.

### التمرين - بناء نموذج

1. قم باستيراد `KMeans` وابدأ عملية التجميع.

    ```python
    from sklearn.cluster import KMeans
    wcss = []
    
    for i in range(1, 11):
        kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)
    
    ```

    هناك بعض الأجزاء هنا التي تستحق التوضيح.

    > 🎓 النطاق: هذه هي التكرارات لعملية التجميع.

    > 🎓 random_state: "يحدد توليد الأرقام العشوائية لتحديد النقاط المركزية." [المصدر](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html#sklearn.cluster.KMeans)

    > 🎓 WCSS: "مجموع المربعات داخل المجموعة" يقيس متوسط المسافة المربعة لجميع النقاط داخل المجموعة إلى النقطة المركزية للمجموعة. [المصدر](https://medium.com/@ODSC/unsupervised-learning-evaluating-clusters-bd47eed175ce).

    > 🎓 القصور الذاتي: تحاول خوارزمية K-Means اختيار النقاط المركزية لتقليل "القصور الذاتي"، "وهو مقياس لمدى تماسك المجموعات داخليًا." [المصدر](https://scikit-learn.org/stable/modules/clustering.html). يتم إلحاق القيمة إلى متغير wcss في كل تكرار.

    > 🎓 k-means++: في [Scikit-learn](https://scikit-learn.org/stable/modules/clustering.html#k-means) يمكنك استخدام تحسين "k-means++"، الذي "يبدأ النقاط المركزية بحيث تكون (عادةً) بعيدة عن بعضها البعض، مما يؤدي إلى نتائج أفضل على الأرجح مقارنة بالتحديد العشوائي."

### طريقة الكوع

سابقًا، افترضت أنه نظرًا لأنك استهدفت 3 أنواع من الأغاني، يجب اختيار 3 مجموعات. ولكن هل هذا صحيح؟

1. استخدم طريقة "الكوع" للتأكد.

    ```python
    plt.figure(figsize=(10,5))
    sns.lineplot(x=range(1, 11), y=wcss, marker='o', color='red')
    plt.title('Elbow')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    plt.show()
    ```

    استخدم متغير `wcss` الذي أنشأته في الخطوة السابقة لإنشاء مخطط يظهر فيه "الانحناء" في الكوع، مما يشير إلى العدد الأمثل للمجموعات. ربما يكون **3** بالفعل!

    ![طريقة الكوع](../../../../5-Clustering/2-K-Means/images/elbow.png)

## التمرين - عرض المجموعات

1. جرب العملية مرة أخرى، هذه المرة بتحديد ثلاث مجموعات، واعرض المجموعات كمخطط مبعثر:

    ```python
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters = 3)
    kmeans.fit(X)
    labels = kmeans.predict(X)
    plt.scatter(df['popularity'],df['danceability'],c = labels)
    plt.xlabel('popularity')
    plt.ylabel('danceability')
    plt.show()
    ```

1. تحقق من دقة النموذج:

    ```python
    labels = kmeans.labels_
    
    correct_labels = sum(y == labels)
    
    print("Result: %d out of %d samples were correctly labeled." % (correct_labels, y.size))
    
    print('Accuracy score: {0:0.2f}'. format(correct_labels/float(y.size)))
    ```

    دقة هذا النموذج ليست جيدة جدًا، وشكل المجموعات يعطيك فكرة عن السبب.

    ![المجموعات](../../../../5-Clustering/2-K-Means/images/clusters.png)

    هذه البيانات غير متوازنة جدًا، وغير مترابطة بشكل كافٍ وهناك الكثير من التباين بين قيم الأعمدة لتجميعها بشكل جيد. في الواقع، المجموعات التي تتشكل ربما تكون متأثرة بشكل كبير أو منحرفة بسبب الفئات الثلاثة التي حددناها أعلاه. كانت هذه عملية تعلم!

    في وثائق Scikit-learn، يمكنك أن ترى أن نموذجًا مثل هذا، مع مجموعات غير محددة بشكل جيد، لديه مشكلة "التباين":

    ![نماذج بها مشاكل](../../../../5-Clustering/2-K-Means/images/problems.png)
    > رسم توضيحي من Scikit-learn

## التباين

يُعرف التباين بأنه "متوسط الفروق المربعة من المتوسط" [(المصدر)](https://www.mathsisfun.com/data/standard-deviation.html). في سياق مشكلة التجميع هذه، يشير إلى البيانات التي تميل أرقام مجموعة البيانات إلى التباعد كثيرًا عن المتوسط.

✅ هذه لحظة رائعة للتفكير في جميع الطرق التي يمكنك من خلالها تصحيح هذه المشكلة. هل يمكنك تعديل البيانات أكثر؟ استخدام أعمدة مختلفة؟ استخدام خوارزمية مختلفة؟ تلميح: جرب [توسيع نطاق بياناتك](https://www.mygreatlearning.com/blog/learning-data-science-with-k-means-clustering/) لتطبيعها واختبار أعمدة أخرى.

> جرب هذا '[حاسبة التباين](https://www.calculatorsoup.com/calculators/statistics/variance-calculator.php)' لفهم المفهوم بشكل أفضل.

---

## 🚀التحدي

اقضِ بعض الوقت مع هذا الدفتر، وقم بتعديل المعلمات. هل يمكنك تحسين دقة النموذج عن طريق تنظيف البيانات أكثر (إزالة القيم الشاذة، على سبيل المثال)؟ يمكنك استخدام أوزان لإعطاء وزن أكبر لعينات بيانات معينة. ماذا يمكنك أن تفعل أيضًا لإنشاء مجموعات أفضل؟

تلميح: جرب توسيع نطاق بياناتك. هناك كود معلق في الدفتر يضيف التوسيع القياسي لجعل أعمدة البيانات تبدو أكثر تشابهًا من حيث النطاق. ستجد أنه بينما ينخفض تقييم Silhouette، يصبح "الانحناء" في مخطط الكوع أكثر سلاسة. هذا لأن ترك البيانات غير موسعة يسمح للبيانات ذات التباين الأقل بأن تحمل وزنًا أكبر. اقرأ المزيد عن هذه المشكلة [هنا](https://stats.stackexchange.com/questions/21222/are-mean-normalization-and-feature-scaling-needed-for-k-means-clustering/21226#21226).

## [اختبار ما بعد المحاضرة](https://ff-quizzes.netlify.app/en/ml/)

## المراجعة والدراسة الذاتية

ألقِ نظرة على محاكي K-Means [مثل هذا](https://user.ceng.metu.edu.tr/~akifakkus/courses/ceng574/k-means/). يمكنك استخدام هذه الأداة لتصور نقاط البيانات النموذجية وتحديد النقاط المركزية لها. يمكنك تعديل عشوائية البيانات، عدد المجموعات وعدد النقاط المركزية. هل يساعدك هذا في الحصول على فكرة عن كيفية تجميع البيانات؟

كما يمكنك الاطلاع على [هذه النشرة حول K-Means](https://stanford.edu/~cpiech/cs221/handouts/kmeans.html) من جامعة ستانفورد.

## الواجب

[جرب طرق تجميع مختلفة](assignment.md)

---

**إخلاء المسؤولية**:  
تمت ترجمة هذا المستند باستخدام خدمة الترجمة بالذكاء الاصطناعي [Co-op Translator](https://github.com/Azure/co-op-translator). بينما نسعى لتحقيق الدقة، يرجى العلم أن الترجمات الآلية قد تحتوي على أخطاء أو عدم دقة. يجب اعتبار المستند الأصلي بلغته الأصلية هو المصدر الموثوق. للحصول على معلومات حساسة أو هامة، يُوصى بالاستعانة بترجمة بشرية احترافية. نحن غير مسؤولين عن أي سوء فهم أو تفسيرات خاطئة تنشأ عن استخدام هذه الترجمة.