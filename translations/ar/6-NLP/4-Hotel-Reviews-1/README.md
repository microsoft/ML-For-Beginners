<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "8d32dadeda93c6fb5c43619854882ab1",
  "translation_date": "2025-09-04T20:53:38+00:00",
  "source_file": "6-NLP/4-Hotel-Reviews-1/README.md",
  "language_code": "ar"
}
-->
# تحليل المشاعر باستخدام تقييمات الفنادق - معالجة البيانات

في هذا القسم، ستستخدم التقنيات التي تعلمتها في الدروس السابقة لإجراء تحليل استكشافي للبيانات لمجموعة بيانات كبيرة. بمجرد أن تحصل على فهم جيد لفائدة الأعمدة المختلفة، ستتعلم:

- كيفية إزالة الأعمدة غير الضرورية
- كيفية حساب بيانات جديدة بناءً على الأعمدة الموجودة
- كيفية حفظ مجموعة البيانات الناتجة لاستخدامها في التحدي النهائي

## [اختبار ما قبل المحاضرة](https://ff-quizzes.netlify.app/en/ml/)

### المقدمة

حتى الآن، تعلمت أن البيانات النصية تختلف تمامًا عن البيانات الرقمية. إذا كانت النصوص مكتوبة أو منطوقة من قبل البشر، يمكن تحليلها لاكتشاف الأنماط والتكرارات، المشاعر والمعاني. تأخذك هذه الدرسة إلى مجموعة بيانات حقيقية مع تحدٍ حقيقي: **[515K تقييمات فنادق في أوروبا](https://www.kaggle.com/jiashenliu/515k-hotel-reviews-data-in-europe)** وتشمل [رخصة CC0: Public Domain](https://creativecommons.org/publicdomain/zero/1.0/). تم جمع البيانات من Booking.com من مصادر عامة. منشئ مجموعة البيانات هو Jiashen Liu.

### التحضير

ستحتاج إلى:

* القدرة على تشغيل دفاتر .ipynb باستخدام Python 3
* مكتبة pandas
* مكتبة NLTK، [والتي يجب تثبيتها محليًا](https://www.nltk.org/install.html)
* مجموعة البيانات المتوفرة على Kaggle [515K تقييمات فنادق في أوروبا](https://www.kaggle.com/jiashenliu/515k-hotel-reviews-data-in-europe). حجمها حوالي 230 ميجابايت بعد فك الضغط. قم بتنزيلها إلى مجلد `/data` الجذر المرتبط بهذه الدروس.

## تحليل استكشافي للبيانات

يفترض هذا التحدي أنك تقوم ببناء روبوت توصيات للفنادق باستخدام تحليل المشاعر وتقييمات الضيوف. تتضمن مجموعة البيانات التي ستستخدمها تقييمات لـ 1493 فندقًا مختلفًا في 6 مدن.

باستخدام Python، مجموعة بيانات تقييمات الفنادق، وتحليل المشاعر باستخدام NLTK، يمكنك اكتشاف:

* ما هي الكلمات والعبارات الأكثر استخدامًا في التقييمات؟
* هل تتطابق *العلامات* الرسمية التي تصف الفندق مع درجات التقييم (على سبيل المثال، هل التقييمات السلبية لفندق معين أكثر شيوعًا من قبل *عائلة مع أطفال صغار* مقارنة بـ *مسافر منفرد*، مما قد يشير إلى أنه أفضل لـ *المسافرين المنفردين*)؟
* هل تتفق درجات المشاعر في NLTK مع التقييم الرقمي للمراجع؟

#### مجموعة البيانات

دعونا نستكشف مجموعة البيانات التي قمت بتنزيلها وحفظها محليًا. افتح الملف في محرر مثل VS Code أو حتى Excel.

رؤوس الأعمدة في مجموعة البيانات هي كما يلي:

*Hotel_Address, Additional_Number_of_Scoring, Review_Date, Average_Score, Hotel_Name, Reviewer_Nationality, Negative_Review, Review_Total_Negative_Word_Counts, Total_Number_of_Reviews, Positive_Review, Review_Total_Positive_Word_Counts, Total_Number_of_Reviews_Reviewer_Has_Given, Reviewer_Score, Tags, days_since_review, lat, lng*

هنا تم تجميعها بطريقة قد تكون أسهل للفحص:
##### أعمدة الفندق

* `Hotel_Name`, `Hotel_Address`, `lat` (خط العرض), `lng` (خط الطول)
  * باستخدام *lat* و *lng* يمكنك رسم خريطة باستخدام Python تُظهر مواقع الفنادق (ربما يتم تلوينها حسب التقييمات السلبية والإيجابية)
  * من الواضح أن Hotel_Address ليس مفيدًا لنا، وربما نستبدله بدولة لتسهيل الفرز والبحث

**أعمدة المراجعات العامة للفندق**

* `Average_Score`
  * وفقًا لمنشئ مجموعة البيانات، هذا العمود هو *متوسط تقييم الفندق، محسوب بناءً على أحدث تعليق في العام الماضي*. يبدو أن هذه طريقة غير عادية لحساب التقييم، ولكن بما أن هذه هي البيانات المستخرجة، يمكننا أخذها كما هي الآن.

  ✅ بناءً على الأعمدة الأخرى في هذه البيانات، هل يمكنك التفكير في طريقة أخرى لحساب متوسط التقييم؟

* `Total_Number_of_Reviews`
  * إجمالي عدد التقييمات التي تلقاها هذا الفندق - ليس من الواضح (دون كتابة بعض التعليمات البرمجية) ما إذا كان هذا يشير إلى التقييمات في مجموعة البيانات.
* `Additional_Number_of_Scoring`
  * يعني أن التقييم تم تقديمه ولكن لم يتم كتابة مراجعة إيجابية أو سلبية من قبل المراجع

**أعمدة المراجعات**

- `Reviewer_Score`
  - هذه قيمة رقمية بأقصى عدد عشري واحد بين القيم الدنيا والقصوى 2.5 و 10
  - لم يتم توضيح سبب كون 2.5 هو أدنى تقييم ممكن
- `Negative_Review`
  - إذا لم يكتب المراجع شيئًا، سيكون هذا الحقل "**No Negative**"
  - لاحظ أن المراجع قد يكتب مراجعة إيجابية في عمود المراجعة السلبية (على سبيل المثال، "لا يوجد شيء سيء في هذا الفندق")
- `Review_Total_Negative_Word_Counts`
  - تشير أعداد الكلمات السلبية الأعلى إلى تقييم أقل (دون التحقق من المشاعر)
- `Positive_Review`
  - إذا لم يكتب المراجع شيئًا، سيكون هذا الحقل "**No Positive**"
  - لاحظ أن المراجع قد يكتب مراجعة سلبية في عمود المراجعة الإيجابية (على سبيل المثال، "لا يوجد شيء جيد في هذا الفندق على الإطلاق")
- `Review_Total_Positive_Word_Counts`
  - تشير أعداد الكلمات الإيجابية الأعلى إلى تقييم أعلى (دون التحقق من المشاعر)
- `Review_Date` و `days_since_review`
  - يمكن تطبيق مقياس حداثة أو قدم على المراجعة (قد لا تكون المراجعات القديمة دقيقة مثل الجديدة بسبب تغييرات في إدارة الفندق، أو تجديدات، أو إضافة مرافق مثل مسبح)
- `Tags`
  - هذه أوصاف قصيرة قد يختارها المراجع لوصف نوع الضيف الذي كان عليه (على سبيل المثال، منفرد أو عائلة)، نوع الغرفة التي حصل عليها، مدة الإقامة وكيفية تقديم المراجعة.
  - لسوء الحظ، استخدام هذه العلامات يمثل مشكلة، تحقق من القسم أدناه الذي يناقش فائدتها

**أعمدة المراجع**

- `Total_Number_of_Reviews_Reviewer_Has_Given`
  - قد يكون هذا عاملًا في نموذج التوصيات، على سبيل المثال، إذا كان بإمكانك تحديد أن المراجعين الأكثر إنتاجية الذين لديهم مئات المراجعات كانوا أكثر ميلًا لأن يكونوا سلبيين بدلاً من إيجابيين. ومع ذلك، فإن المراجع لأي مراجعة معينة غير معرف برمز فريد، وبالتالي لا يمكن ربطه بمجموعة من المراجعات. هناك 30 مراجعًا لديهم 100 مراجعة أو أكثر، ولكن من الصعب رؤية كيف يمكن أن يساعد ذلك في نموذج التوصيات.
- `Reviewer_Nationality`
  - قد يعتقد البعض أن بعض الجنسيات أكثر ميلًا لتقديم مراجعة إيجابية أو سلبية بسبب ميول وطنية. كن حذرًا عند بناء مثل هذه الآراء القصصية في نماذجك. هذه قوالب نمطية وطنية (وأحيانًا عرقية)، وكل مراجع هو فرد كتب مراجعة بناءً على تجربته. قد تكون التجربة قد تمت تصفيتها من خلال العديد من العدسات مثل إقاماتهم الفندقية السابقة، المسافة التي قطعوها، ومزاجهم الشخصي. من الصعب تبرير أن جنسيتهم كانت السبب وراء تقييمهم.

##### أمثلة

| متوسط التقييم | إجمالي عدد التقييمات | تقييم المراجع | المراجعة السلبية                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  | المراجعة الإيجابية                 | العلامات                                                                                      |
| -------------- | ---------------------- | ---------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------- | ----------------------------------------------------------------------------------------- |
| 7.8            | 1945                   | 2.5              | هذا ليس فندقًا حاليًا بل موقع بناء. تعرضت للإزعاج من الصباح الباكر وطوال اليوم بضوضاء بناء غير مقبولة أثناء الراحة بعد رحلة طويلة والعمل في الغرفة. كان الناس يعملون طوال اليوم باستخدام المطارق في الغرف المجاورة. طلبت تغيير الغرفة ولكن لم تكن هناك غرفة هادئة متاحة. لجعل الأمور أسوأ، تم تحميلي رسومًا زائدة. غادرت في المساء لأنني كنت بحاجة إلى مغادرة رحلة مبكرة جدًا وتلقيت فاتورة مناسبة. بعد يوم، قام الفندق بخصم آخر دون موافقتي بمبلغ يزيد عن السعر المحجوز. إنه مكان فظيع. لا تعاقب نفسك بالحجز هنا. | لا شيء. مكان رهيب. ابتعد. | رحلة عمل، زوجان، غرفة مزدوجة قياسية، أقاموا ليلتين |

كما ترى، لم يكن هذا الضيف سعيدًا بإقامته في هذا الفندق. الفندق لديه متوسط تقييم جيد يبلغ 7.8 و1945 مراجعة، لكن هذا المراجع أعطاه 2.5 وكتب 115 كلمة عن مدى سلبية إقامته. إذا لم يكتبوا شيئًا على الإطلاق في عمود المراجعة الإيجابية، يمكنك أن تستنتج أنه لم يكن هناك شيء إيجابي، ولكن للأسف كتبوا 7 كلمات تحذيرية. إذا قمنا فقط بعد الكلمات بدلاً من معنى الكلمات أو مشاعرها، فقد نحصل على رؤية مشوهة لنية المراجع. الغريب أن تقييمهم بـ 2.5 محير، لأنه إذا كانت الإقامة في الفندق سيئة جدًا، فلماذا يعطونه أي نقاط على الإطلاق؟ عند التحقيق في مجموعة البيانات عن كثب، ستلاحظ أن أدنى تقييم ممكن هو 2.5، وليس 0. وأعلى تقييم ممكن هو 10.

##### العلامات

كما ذكر أعلاه، عند النظر لأول مرة، يبدو أن فكرة استخدام `Tags` لتصنيف البيانات منطقية. لسوء الحظ، هذه العلامات ليست موحدة، مما يعني أنه في فندق معين، قد تكون الخيارات *غرفة فردية*، *غرفة مزدوجة*، و*غرفة توأم*، ولكن في الفندق التالي، تكون *غرفة فردية ديلوكس*، *غرفة كوين كلاسيكية*، و*غرفة كينغ تنفيذية*. قد تكون هذه نفس الأشياء، ولكن هناك العديد من الاختلافات بحيث يصبح الخيار:

1. محاولة تغيير جميع المصطلحات إلى معيار واحد، وهو أمر صعب للغاية، لأنه ليس من الواضح ما هو مسار التحويل في كل حالة (على سبيل المثال، *غرفة فردية كلاسيكية* تتطابق مع *غرفة فردية* ولكن *غرفة كوين متفوقة مع إطلالة على الحديقة أو المدينة* أصعب في المطابقة)

1. يمكننا اتخاذ نهج NLP وقياس تكرار مصطلحات معينة مثل *منفرد*، *مسافر عمل*، أو *عائلة مع أطفال صغار* كما تنطبق على كل فندق، وإدخال ذلك في نموذج التوصية  

عادةً ما تكون العلامات (ولكن ليس دائمًا) حقلًا واحدًا يحتوي على قائمة من 5 إلى 6 قيم مفصولة بفواصل تتماشى مع *نوع الرحلة*، *نوع الضيوف*، *نوع الغرفة*، *عدد الليالي*، و*نوع الجهاز الذي تم تقديم المراجعة عليه*. ومع ذلك، نظرًا لأن بعض المراجعين لا يملأون كل حقل (قد يتركون واحدًا فارغًا)، فإن القيم ليست دائمًا بنفس الترتيب.

كمثال، خذ *نوع المجموعة*. هناك 1025 إمكانية فريدة في هذا الحقل في عمود `Tags`، وللأسف فقط بعض منها يشير إلى مجموعة (بعضها يشير إلى نوع الغرفة وما إلى ذلك). إذا قمت بتصفية فقط القيم التي تذكر العائلة، تحتوي النتائج على العديد من النتائج من نوع *غرفة عائلية*. إذا قمت بتضمين مصطلح *مع*، أي عد القيم *عائلة مع*، تكون النتائج أفضل، مع أكثر من 80,000 من 515,000 نتيجة تحتوي على العبارة "عائلة مع أطفال صغار" أو "عائلة مع أطفال كبار".

هذا يعني أن عمود العلامات ليس عديم الفائدة تمامًا بالنسبة لنا، ولكنه سيتطلب بعض العمل لجعله مفيدًا.

##### متوسط تقييم الفندق

هناك عدد من الغرائب أو التناقضات مع مجموعة البيانات التي لا أستطيع فهمها، ولكنها موضحة هنا لتكون على دراية بها عند بناء نماذجك. إذا اكتشفتها، يرجى إخبارنا في قسم المناقشة!

تحتوي مجموعة البيانات على الأعمدة التالية المتعلقة بمتوسط التقييم وعدد التقييمات:

1. Hotel_Name
2. Additional_Number_of_Scoring
3. Average_Score
4. Total_Number_of_Reviews
5. Reviewer_Score  

الفندق الوحيد الذي يحتوي على أكبر عدد من التقييمات في هذه المجموعة هو *Britannia International Hotel Canary Wharf* مع 4789 تقييمًا من أصل 515,000. ولكن إذا نظرنا إلى قيمة `Total_Number_of_Reviews` لهذا الفندق، فهي 9086. قد تستنتج أن هناك العديد من التقييمات بدون مراجعات، لذا ربما يجب أن نضيف قيمة عمود `Additional_Number_of_Scoring`. تلك القيمة هي 2682، وإضافتها إلى 4789 تعطينا 7471، وهو ما يزال أقل بـ 1615 من `Total_Number_of_Reviews`.

إذا أخذت عمود `Average_Score`، قد تستنتج أنه متوسط التقييمات في مجموعة البيانات، ولكن الوصف من Kaggle هو "*متوسط تقييم الفندق، محسوب بناءً على أحدث تعليق في العام الماضي*". هذا لا يبدو مفيدًا جدًا، ولكن يمكننا حساب متوسطنا الخاص بناءً على تقييمات المراجعين في مجموعة البيانات. باستخدام نفس الفندق كمثال، متوسط تقييم الفندق المعطى هو 7.1 ولكن التقييم المحسوب (متوسط تقييمات المراجعين *في* مجموعة البيانات) هو 6.8. هذا قريب، ولكنه ليس نفس القيمة، ويمكننا فقط التخمين بأن التقييمات المعطاة في مراجعات `Additional_Number_of_Scoring` زادت المتوسط إلى 7.1. لسوء الحظ، مع عدم وجود طريقة لاختبار أو إثبات هذا الافتراض، من الصعب استخدام أو الوثوق بـ `Average_Score`، `Additional_Number_of_Scoring` و `Total_Number_of_Reviews` عندما تستند إلى، أو تشير إلى، بيانات لا نملكها.

لتعقيد الأمور أكثر، الفندق الذي يحتوي على ثاني أعلى عدد من التقييمات لديه متوسط تقييم محسوب يبلغ 8.12 ومتوسط التقييم في مجموعة البيانات هو 8.1. هل هذا التقييم الصحيح مجرد صدفة أم أن الفندق الأول هو التناقض؟

على احتمال أن يكون هذا الفندق حالة شاذة، وأن معظم القيم قد تتطابق (ولكن بعضها لا يتطابق لسبب ما)، سنكتب برنامجًا قصيرًا بعد ذلك لاستكشاف القيم في مجموعة البيانات وتحديد الاستخدام الصحيح (أو عدم الاستخدام) للقيم.
> 🚨 ملاحظة تحذيرية  
>  
> عند العمل مع هذه المجموعة من البيانات، ستكتب كودًا يحسب شيئًا من النص دون الحاجة إلى قراءة النص أو تحليله بنفسك. هذه هي جوهر معالجة اللغة الطبيعية (NLP)، تفسير المعنى أو المشاعر دون الحاجة إلى تدخل بشري. ومع ذلك، من الممكن أن تقرأ بعض المراجعات السلبية. أنصحك بعدم القيام بذلك، لأنك لست بحاجة لذلك. بعض هذه المراجعات سخيفة أو غير ذات صلة، مثل "الطقس لم يكن جيدًا"، وهو أمر خارج عن سيطرة الفندق أو أي شخص آخر. ولكن هناك جانب مظلم لبعض المراجعات أيضًا. أحيانًا تكون المراجعات السلبية عنصرية، أو متحيزة ضد الجنس، أو العمر. هذا أمر مؤسف ولكنه متوقع في مجموعة بيانات مأخوذة من موقع عام. بعض المراجعين يتركون مراجعات قد تجدها غير مستساغة، أو غير مريحة، أو مزعجة. من الأفضل أن تدع الكود يقيس المشاعر بدلاً من قراءتها بنفسك والتأثر بها. ومع ذلك، فإن من يكتبون مثل هذه الأمور هم أقلية، لكنهم موجودون بكل الأحوال.
## التمرين - استكشاف البيانات
### تحميل البيانات

لقد اكتفينا من فحص البيانات بصريًا، الآن حان الوقت لكتابة بعض الأكواد والحصول على إجابات! هذا القسم يستخدم مكتبة pandas. أول مهمة لك هي التأكد من أنك تستطيع تحميل وقراءة بيانات CSV. مكتبة pandas تحتوي على أداة سريعة لتحميل ملفات CSV، والنتيجة يتم وضعها في DataFrame، كما رأينا في الدروس السابقة. ملف CSV الذي نقوم بتحميله يحتوي على أكثر من نصف مليون صف، ولكن فقط 17 عمودًا. توفر pandas العديد من الطرق القوية للتفاعل مع DataFrame، بما في ذلك القدرة على إجراء عمليات على كل صف.

من الآن فصاعدًا في هذا الدرس، ستجد مقتطفات من الأكواد مع بعض الشروحات عنها ومناقشة حول ما تعنيه النتائج. استخدم الملف _notebook.ipynb_ المرفق لكتابة الأكواد الخاصة بك.

لنبدأ بتحميل ملف البيانات الذي ستستخدمه:

```python
# Load the hotel reviews from CSV
import pandas as pd
import time
# importing time so the start and end time can be used to calculate file loading time
print("Loading data file now, this could take a while depending on file size")
start = time.time()
# df is 'DataFrame' - make sure you downloaded the file to the data folder
df = pd.read_csv('../../data/Hotel_Reviews.csv')
end = time.time()
print("Loading took " + str(round(end - start, 2)) + " seconds")
```

الآن بعد أن تم تحميل البيانات، يمكننا إجراء بعض العمليات عليها. احتفظ بهذا الكود في أعلى برنامجك للجزء التالي.

## استكشاف البيانات

في هذه الحالة، البيانات بالفعل *نظيفة*، مما يعني أنها جاهزة للعمل عليها، ولا تحتوي على أحرف بلغات أخرى قد تسبب مشاكل للخوارزميات التي تتوقع فقط أحرفًا إنجليزية.

✅ قد تضطر أحيانًا إلى العمل مع بيانات تحتاج إلى معالجة أولية لتنسيقها قبل تطبيق تقنيات معالجة اللغة الطبيعية (NLP)، ولكن ليس هذه المرة. إذا كان عليك ذلك، كيف ستتعامل مع الأحرف غير الإنجليزية؟

خذ لحظة للتأكد من أنه بمجرد تحميل البيانات، يمكنك استكشافها باستخدام الكود. من السهل جدًا أن ترغب في التركيز على العمودين `Negative_Review` و`Positive_Review`. هذان العمودان مليئان بالنصوص الطبيعية التي يمكن لخوارزميات NLP الخاصة بك معالجتها. ولكن انتظر! قبل أن تبدأ في معالجة اللغة الطبيعية وتحليل المشاعر، يجب أن تتبع الكود أدناه للتأكد من أن القيم الموجودة في مجموعة البيانات تتطابق مع القيم التي تحسبها باستخدام pandas.

## عمليات DataFrame

المهمة الأولى في هذا الدرس هي التحقق مما إذا كانت الافتراضات التالية صحيحة عن طريق كتابة كود يفحص DataFrame (دون تغييرها).

> كما هو الحال مع العديد من مهام البرمجة، هناك عدة طرق لإكمال هذه المهمة، ولكن النصيحة الجيدة هي القيام بذلك بأبسط وأسهل طريقة ممكنة، خاصة إذا كان ذلك سيجعل الكود أسهل للفهم عند العودة إليه في المستقبل. مع DataFrames، هناك واجهة برمجية شاملة (API) غالبًا ما تحتوي على طريقة فعالة لتحقيق ما تريد.

تعامل مع الأسئلة التالية كمهام برمجية وحاول الإجابة عليها دون النظر إلى الحل.

1. اطبع *الشكل* (shape) الخاص بـ DataFrame الذي قمت بتحميله للتو (الشكل هو عدد الصفوف والأعمدة).
2. احسب تكرار الجنسيات للمراجعين:
   1. كم عدد القيم المميزة الموجودة في العمود `Reviewer_Nationality` وما هي؟
   2. ما هي الجنسية الأكثر شيوعًا بين المراجعين في مجموعة البيانات (اطبع البلد وعدد المراجعات)؟
   3. ما هي أكثر 10 جنسيات تكرارًا بعد ذلك، وعدد مرات تكرارها؟
3. ما هو الفندق الذي تمت مراجعته بشكل متكرر لكل من أكثر 10 جنسيات للمراجعين؟
4. كم عدد المراجعات لكل فندق (تكرار المراجعات لكل فندق) في مجموعة البيانات؟
5. على الرغم من وجود عمود `Average_Score` لكل فندق في مجموعة البيانات، يمكنك أيضًا حساب متوسط التقييم (بحساب متوسط جميع تقييمات المراجعين في مجموعة البيانات لكل فندق). أضف عمودًا جديدًا إلى DataFrame بعنوان `Calc_Average_Score` يحتوي على المتوسط المحسوب.
6. هل هناك فنادق لها نفس القيمة (بعد التقريب إلى منزلة عشرية واحدة) في عمودي `Average_Score` و`Calc_Average_Score`؟
   1. حاول كتابة دالة Python تأخذ Series (صف) كمدخل وتقارن القيم، وتطبع رسالة عندما تكون القيم غير متساوية. ثم استخدم الطريقة `.apply()` لمعالجة كل صف باستخدام الدالة.
7. احسب واطبع عدد الصفوف التي تحتوي على القيم "No Negative" في العمود `Negative_Review`.
8. احسب واطبع عدد الصفوف التي تحتوي على القيم "No Positive" في العمود `Positive_Review`.
9. احسب واطبع عدد الصفوف التي تحتوي على القيم "No Positive" في العمود `Positive_Review` **و** "No Negative" في العمود `Negative_Review`.

### إجابات الكود

1. اطبع *الشكل* الخاص بـ DataFrame الذي قمت بتحميله للتو (الشكل هو عدد الصفوف والأعمدة).

   ```python
   print("The shape of the data (rows, cols) is " + str(df.shape))
   > The shape of the data (rows, cols) is (515738, 17)
   ```

2. احسب تكرار الجنسيات للمراجعين:

   1. كم عدد القيم المميزة الموجودة في العمود `Reviewer_Nationality` وما هي؟
   2. ما هي الجنسية الأكثر شيوعًا بين المراجعين في مجموعة البيانات (اطبع البلد وعدد المراجعات)؟

   ```python
   # value_counts() creates a Series object that has index and values in this case, the country and the frequency they occur in reviewer nationality
   nationality_freq = df["Reviewer_Nationality"].value_counts()
   print("There are " + str(nationality_freq.size) + " different nationalities")
   # print first and last rows of the Series. Change to nationality_freq.to_string() to print all of the data
   print(nationality_freq) 
   
   There are 227 different nationalities
    United Kingdom               245246
    United States of America      35437
    Australia                     21686
    Ireland                       14827
    United Arab Emirates          10235
                                  ...  
    Comoros                           1
    Palau                             1
    Northern Mariana Islands          1
    Cape Verde                        1
    Guinea                            1
   Name: Reviewer_Nationality, Length: 227, dtype: int64
   ```

   3. ما هي أكثر 10 جنسيات تكرارًا بعد ذلك، وعدد مرات تكرارها؟

      ```python
      print("The highest frequency reviewer nationality is " + str(nationality_freq.index[0]).strip() + " with " + str(nationality_freq[0]) + " reviews.")
      # Notice there is a leading space on the values, strip() removes that for printing
      # What is the top 10 most common nationalities and their frequencies?
      print("The next 10 highest frequency reviewer nationalities are:")
      print(nationality_freq[1:11].to_string())
      
      The highest frequency reviewer nationality is United Kingdom with 245246 reviews.
      The next 10 highest frequency reviewer nationalities are:
       United States of America     35437
       Australia                    21686
       Ireland                      14827
       United Arab Emirates         10235
       Saudi Arabia                  8951
       Netherlands                   8772
       Switzerland                   8678
       Germany                       7941
       Canada                        7894
       France                        7296
      ```

3. ما هو الفندق الذي تمت مراجعته بشكل متكرر لكل من أكثر 10 جنسيات للمراجعين؟

   ```python
   # What was the most frequently reviewed hotel for the top 10 nationalities
   # Normally with pandas you will avoid an explicit loop, but wanted to show creating a new dataframe using criteria (don't do this with large amounts of data because it could be very slow)
   for nat in nationality_freq[:10].index:
      # First, extract all the rows that match the criteria into a new dataframe
      nat_df = df[df["Reviewer_Nationality"] == nat]   
      # Now get the hotel freq
      freq = nat_df["Hotel_Name"].value_counts()
      print("The most reviewed hotel for " + str(nat).strip() + " was " + str(freq.index[0]) + " with " + str(freq[0]) + " reviews.") 
      
   The most reviewed hotel for United Kingdom was Britannia International Hotel Canary Wharf with 3833 reviews.
   The most reviewed hotel for United States of America was Hotel Esther a with 423 reviews.
   The most reviewed hotel for Australia was Park Plaza Westminster Bridge London with 167 reviews.
   The most reviewed hotel for Ireland was Copthorne Tara Hotel London Kensington with 239 reviews.
   The most reviewed hotel for United Arab Emirates was Millennium Hotel London Knightsbridge with 129 reviews.
   The most reviewed hotel for Saudi Arabia was The Cumberland A Guoman Hotel with 142 reviews.
   The most reviewed hotel for Netherlands was Jaz Amsterdam with 97 reviews.
   The most reviewed hotel for Switzerland was Hotel Da Vinci with 97 reviews.
   The most reviewed hotel for Germany was Hotel Da Vinci with 86 reviews.
   The most reviewed hotel for Canada was St James Court A Taj Hotel London with 61 reviews.
   ```

4. كم عدد المراجعات لكل فندق (تكرار المراجعات لكل فندق) في مجموعة البيانات؟

   ```python
   # First create a new dataframe based on the old one, removing the uneeded columns
   hotel_freq_df = df.drop(["Hotel_Address", "Additional_Number_of_Scoring", "Review_Date", "Average_Score", "Reviewer_Nationality", "Negative_Review", "Review_Total_Negative_Word_Counts", "Positive_Review", "Review_Total_Positive_Word_Counts", "Total_Number_of_Reviews_Reviewer_Has_Given", "Reviewer_Score", "Tags", "days_since_review", "lat", "lng"], axis = 1)
   
   # Group the rows by Hotel_Name, count them and put the result in a new column Total_Reviews_Found
   hotel_freq_df['Total_Reviews_Found'] = hotel_freq_df.groupby('Hotel_Name').transform('count')
   
   # Get rid of all the duplicated rows
   hotel_freq_df = hotel_freq_df.drop_duplicates(subset = ["Hotel_Name"])
   display(hotel_freq_df) 
   ```
   |                 اسم الفندق                 | إجمالي عدد المراجعات | عدد المراجعات الموجودة |
   | :----------------------------------------: | :-------------------: | :---------------------: |
   | Britannia International Hotel Canary Wharf |          9086         |          4789          |
   |    Park Plaza Westminster Bridge London    |         12158         |          4169          |
   |   Copthorne Tara Hotel London Kensington   |          7105         |          3578          |
   |                    ...                     |          ...          |           ...          |
   |       Mercure Paris Porte d Orleans        |          110          |           10           |
   |                Hotel Wagner                |          135          |           10           |
   |            Hotel Gallitzinberg             |          173          |            8           |

   قد تلاحظ أن النتائج *المحسوبة في مجموعة البيانات* لا تتطابق مع القيمة في `Total_Number_of_Reviews`. من غير الواضح ما إذا كانت هذه القيمة في مجموعة البيانات تمثل إجمالي عدد المراجعات التي حصل عليها الفندق، ولكن لم يتم جمعها جميعًا، أو إذا كان هناك حساب آخر. لا يتم استخدام `Total_Number_of_Reviews` في النموذج بسبب هذا الغموض.

5. على الرغم من وجود عمود `Average_Score` لكل فندق في مجموعة البيانات، يمكنك أيضًا حساب متوسط التقييم (بحساب متوسط جميع تقييمات المراجعين في مجموعة البيانات لكل فندق). أضف عمودًا جديدًا إلى DataFrame بعنوان `Calc_Average_Score` يحتوي على المتوسط المحسوب. اطبع الأعمدة `Hotel_Name`، `Average_Score`، و`Calc_Average_Score`.

   ```python
   # define a function that takes a row and performs some calculation with it
   def get_difference_review_avg(row):
     return row["Average_Score"] - row["Calc_Average_Score"]
   
   # 'mean' is mathematical word for 'average'
   df['Calc_Average_Score'] = round(df.groupby('Hotel_Name').Reviewer_Score.transform('mean'), 1)
   
   # Add a new column with the difference between the two average scores
   df["Average_Score_Difference"] = df.apply(get_difference_review_avg, axis = 1)
   
   # Create a df without all the duplicates of Hotel_Name (so only 1 row per hotel)
   review_scores_df = df.drop_duplicates(subset = ["Hotel_Name"])
   
   # Sort the dataframe to find the lowest and highest average score difference
   review_scores_df = review_scores_df.sort_values(by=["Average_Score_Difference"])
   
   display(review_scores_df[["Average_Score_Difference", "Average_Score", "Calc_Average_Score", "Hotel_Name"]])
   ```

   قد تتساءل أيضًا عن قيمة `Average_Score` ولماذا تكون أحيانًا مختلفة عن متوسط التقييم المحسوب. بما أننا لا نستطيع معرفة سبب تطابق بعض القيم، ولكن وجود اختلاف في أخرى، فمن الأفضل في هذه الحالة استخدام تقييمات المراجعين التي لدينا لحساب المتوسط بأنفسنا. ومع ذلك، فإن الفروقات عادة ما تكون صغيرة، وهذه هي الفنادق التي لديها أكبر انحراف عن متوسط التقييم في مجموعة البيانات والمتوسط المحسوب:

   | فرق متوسط التقييم | متوسط التقييم | المتوسط المحسوب |                                  اسم الفندق |
   | :----------------: | :-----------: | :--------------: | ------------------------------------------: |
   |        -0.8        |      7.7      |        8.5       |                  Best Western Hotel Astoria |
   |        -0.7        |      8.8      |        9.5       | Hotel Stendhal Place Vend me Paris MGallery |
   |        -0.7        |      7.5      |        8.2       |               Mercure Paris Porte d Orleans |
   |        -0.7        |      7.9      |        8.6       |             Renaissance Paris Vendome Hotel |
   |        -0.5        |      7.0      |        7.5       |                         Hotel Royal Elys es |
   |        ...         |      ...      |        ...       |                                         ... |
   |         0.7        |      7.5      |        6.8       |     Mercure Paris Op ra Faubourg Montmartre |
   |         0.8        |      7.1      |        6.3       |      Holiday Inn Paris Montparnasse Pasteur |
   |         0.9        |      6.8      |        5.9       |                               Villa Eugenie |
   |         0.9        |      8.6      |        7.7       |   MARQUIS Faubourg St Honor Relais Ch teaux |
   |         1.3        |      7.2      |        5.9       |                          Kube Hotel Ice Bar |

   مع وجود فندق واحد فقط لديه فرق في التقييم أكبر من 1، فهذا يعني أنه يمكننا على الأرجح تجاهل الفرق واستخدام متوسط التقييم المحسوب.

6. احسب واطبع عدد الصفوف التي تحتوي على القيم "No Negative" في العمود `Negative_Review`.

7. احسب واطبع عدد الصفوف التي تحتوي على القيم "No Positive" في العمود `Positive_Review`.

8. احسب واطبع عدد الصفوف التي تحتوي على القيم "No Positive" في العمود `Positive_Review` **و** "No Negative" في العمود `Negative_Review`.

   ```python
   # with lambdas:
   start = time.time()
   no_negative_reviews = df.apply(lambda x: True if x['Negative_Review'] == "No Negative" else False , axis=1)
   print("Number of No Negative reviews: " + str(len(no_negative_reviews[no_negative_reviews == True].index)))
   
   no_positive_reviews = df.apply(lambda x: True if x['Positive_Review'] == "No Positive" else False , axis=1)
   print("Number of No Positive reviews: " + str(len(no_positive_reviews[no_positive_reviews == True].index)))
   
   both_no_reviews = df.apply(lambda x: True if x['Negative_Review'] == "No Negative" and x['Positive_Review'] == "No Positive" else False , axis=1)
   print("Number of both No Negative and No Positive reviews: " + str(len(both_no_reviews[both_no_reviews == True].index)))
   end = time.time()
   print("Lambdas took " + str(round(end - start, 2)) + " seconds")
   
   Number of No Negative reviews: 127890
   Number of No Positive reviews: 35946
   Number of both No Negative and No Positive reviews: 127
   Lambdas took 9.64 seconds
   ```

## طريقة أخرى

طريقة أخرى لحساب العناصر بدون استخدام Lambdas، واستخدام sum لحساب الصفوف:

   ```python
   # without lambdas (using a mixture of notations to show you can use both)
   start = time.time()
   no_negative_reviews = sum(df.Negative_Review == "No Negative")
   print("Number of No Negative reviews: " + str(no_negative_reviews))
   
   no_positive_reviews = sum(df["Positive_Review"] == "No Positive")
   print("Number of No Positive reviews: " + str(no_positive_reviews))
   
   both_no_reviews = sum((df.Negative_Review == "No Negative") & (df.Positive_Review == "No Positive"))
   print("Number of both No Negative and No Positive reviews: " + str(both_no_reviews))
   
   end = time.time()
   print("Sum took " + str(round(end - start, 2)) + " seconds")
   
   Number of No Negative reviews: 127890
   Number of No Positive reviews: 35946
   Number of both No Negative and No Positive reviews: 127
   Sum took 0.19 seconds
   ```

   قد تكون لاحظت أن هناك 127 صفًا تحتوي على القيم "No Negative" و"No Positive" في العمودين `Negative_Review` و`Positive_Review` على التوالي. هذا يعني أن المراجع أعطى الفندق تقييمًا رقميًا، لكنه امتنع عن كتابة مراجعة إيجابية أو سلبية. لحسن الحظ، هذا عدد صغير من الصفوف (127 من أصل 515738، أو 0.02%)، لذا من المحتمل ألا يؤثر ذلك على النموذج أو النتائج في أي اتجاه معين، ولكن قد لا تتوقع أن تحتوي مجموعة بيانات المراجعات على صفوف بدون مراجعات، لذا من المفيد استكشاف البيانات لاكتشاف صفوف مثل هذه.

الآن بعد أن قمت باستكشاف مجموعة البيانات، في الدرس التالي ستقوم بتصفية البيانات وإضافة بعض تحليلات المشاعر.

---
## 🚀تحدي

يوضح هذا الدرس، كما رأينا في الدروس السابقة، مدى أهمية فهم بياناتك وخصائصها قبل إجراء العمليات عليها. البيانات النصية، على وجه الخصوص، تتطلب تدقيقًا دقيقًا. قم بالبحث في مجموعات بيانات مختلفة تحتوي على نصوص واكتشف المجالات التي قد تؤدي إلى إدخال تحيز أو مشاعر مشوهة في النموذج.

## [اختبار ما بعد المحاضرة](https://ff-quizzes.netlify.app/en/ml/)

## المراجعة والدراسة الذاتية

خذ [مسار التعلم هذا حول معالجة اللغة الطبيعية](https://docs.microsoft.com/learn/paths/explore-natural-language-processing/?WT.mc_id=academic-77952-leestott) لاكتشاف الأدوات التي يمكنك تجربتها عند بناء نماذج تعتمد على النصوص والكلام.

## الواجب

[NLTK](assignment.md)

---

**إخلاء المسؤولية**:  
تمت ترجمة هذا المستند باستخدام خدمة الترجمة الآلية [Co-op Translator](https://github.com/Azure/co-op-translator). بينما نسعى لتحقيق الدقة، يرجى العلم أن الترجمات الآلية قد تحتوي على أخطاء أو معلومات غير دقيقة. يجب اعتبار المستند الأصلي بلغته الأصلية هو المصدر الموثوق. للحصول على معلومات حساسة أو هامة، يُوصى بالاستعانة بترجمة بشرية احترافية. نحن غير مسؤولين عن أي سوء فهم أو تفسيرات خاطئة تنشأ عن استخدام هذه الترجمة.