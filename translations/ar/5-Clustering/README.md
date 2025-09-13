<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "b28a3a4911584062772c537b653ebbc7",
  "translation_date": "2025-08-29T13:18:28+00:00",
  "source_file": "5-Clustering/README.md",
  "language_code": "ar"
}
-->
# نماذج التجميع في تعلم الآلة

التجميع هو مهمة في تعلم الآلة تهدف إلى العثور على الأشياء التي تشبه بعضها البعض وتجميعها في مجموعات تُسمى "العناقيد". ما يميز التجميع عن الأساليب الأخرى في تعلم الآلة هو أن الأمور تحدث تلقائيًا، في الواقع، يمكن القول إنه عكس التعلم الموجّه.

## موضوع إقليمي: نماذج التجميع لتفضيلات الجمهور النيجيري الموسيقية 🎧

الجمهور النيجيري المتنوع لديه أذواق موسيقية متنوعة. باستخدام البيانات المستخرجة من Spotify (مستوحاة من [هذه المقالة](https://towardsdatascience.com/country-wise-visual-analysis-of-music-taste-using-spotify-api-seaborn-in-python-77f5b749b421))، دعونا نلقي نظرة على بعض الموسيقى الشعبية في نيجيريا. تتضمن هذه المجموعة من البيانات معلومات حول درجات "القابلية للرقص"، "الصوتية"، مستوى الصوت، "الكلامية"، الشعبية والطاقة للأغاني المختلفة. سيكون من المثير اكتشاف الأنماط في هذه البيانات!

![جهاز تشغيل الأسطوانات](../../../translated_images/turntable.f2b86b13c53302dc106aa741de9dc96ac372864cf458dd6f879119857aab01da.ar.jpg)

> صورة بواسطة <a href="https://unsplash.com/@marcelalaskoski?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Marcela Laskoski</a> على <a href="https://unsplash.com/s/photos/nigerian-music?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Unsplash</a>
  
في هذه السلسلة من الدروس، ستكتشف طرقًا جديدة لتحليل البيانات باستخدام تقنيات التجميع. التجميع مفيد بشكل خاص عندما تفتقر مجموعة البيانات إلى تسميات. إذا كانت تحتوي على تسميات، فإن تقنيات التصنيف مثل تلك التي تعلمتها في الدروس السابقة قد تكون أكثر فائدة. ولكن في الحالات التي تبحث فيها عن تجميع بيانات غير مُسماة، فإن التجميع طريقة رائعة لاكتشاف الأنماط.

> هناك أدوات منخفضة الكود يمكن أن تساعدك في تعلم كيفية العمل مع نماذج التجميع. جرب [Azure ML لهذه المهمة](https://docs.microsoft.com/learn/modules/create-clustering-model-azure-machine-learning-designer/?WT.mc_id=academic-77952-leestott)

## الدروس

1. [مقدمة في التجميع](1-Visualize/README.md)
2. [تجميع K-Means](2-K-Means/README.md)

## الشكر

تم كتابة هذه الدروس مع 🎶 بواسطة [Jen Looper](https://www.twitter.com/jenlooper) مع مراجعات مفيدة من [Rishit Dagli](https://rishit_dagli) و [Muhammad Sakib Khan Inan](https://twitter.com/Sakibinan).

تم الحصول على مجموعة بيانات [الأغاني النيجيرية](https://www.kaggle.com/sootersaalu/nigerian-songs-spotify) من Kaggle بعد استخراجها من Spotify.

أمثلة مفيدة عن K-Means ساعدت في إنشاء هذا الدرس تشمل [استكشاف زهرة السوسن](https://www.kaggle.com/bburns/iris-exploration-pca-k-means-and-gmm-clustering)، [دفتر ملاحظات تمهيدي](https://www.kaggle.com/prashant111/k-means-clustering-with-python)، و[مثال افتراضي لمنظمة غير حكومية](https://www.kaggle.com/ankandash/pca-k-means-clustering-hierarchical-clustering).

---

**إخلاء المسؤولية**:  
تم ترجمة هذا المستند باستخدام خدمة الترجمة الآلية [Co-op Translator](https://github.com/Azure/co-op-translator). بينما نسعى لتحقيق الدقة، يرجى العلم أن الترجمات الآلية قد تحتوي على أخطاء أو معلومات غير دقيقة. يجب اعتبار المستند الأصلي بلغته الأصلية هو المصدر الموثوق. للحصول على معلومات حساسة أو هامة، يُوصى بالاستعانة بترجمة بشرية احترافية. نحن غير مسؤولين عن أي سوء فهم أو تفسيرات خاطئة ناتجة عن استخدام هذه الترجمة.