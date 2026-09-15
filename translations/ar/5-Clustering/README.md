# نماذج التجميع للتعلّم الآلي

التجميع هو مهمة في التعلّم الآلي تهدف إلى إيجاد الأشياء المتشابهة وتجمّعها ضمن مجموعات تُسمى تجمّعات. ما يميّز التجميع عن الأساليب الأخرى في التعلّم الآلي هو أن العمليات تتم تلقائيًا، في الواقع، يمكن القول إنه عكس التعلّم المُشرف.

## الموضوع الإقليمي: نماذج التجميع لذوق الجمهور النيجيري الموسيقي 🎧

يمتلك جمهور نيجيريا المتنوع أذواقًا موسيقية متنوعة. باستخدام بيانات تم جمعها من سبوتيفاي (مستلهمة من [هذه المقالة](https://towardsdatascience.com/country-wise-visual-analysis-of-music-taste-using-spotify-api-seaborn-in-python-77f5b749b421))، دعونا نلقي نظرة على بعض الموسيقى الشهيرة في نيجيريا. تتضمن هذه المجموعة بيانات حول درجات 'رقص' الأغاني، و'الصوتية'، والضوضاء، و'الكلامية'، والشعبية والطاقة. سيكون من الممتع اكتشاف الأنماط في هذه البيانات!

![طاولة تشغيل الأسطوانات](../../../translated_images/ar/turntable.f2b86b13c53302dc.webp)

> صورة بواسطة <a href="https://unsplash.com/@marcelalaskoski?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">مارسلا لاسكوسكي</a> على <a href="https://unsplash.com/s/photos/nigerian-music?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">أنسبلاش</a>
  
في هذه السلسلة من الدروس، ستكتشف طرقًا جديدة لتحليل البيانات باستخدام تقنيات التجميع. يعد التجميع مفيدًا بشكل خاص عندما تفتقر مجموعة بياناتك إلى تسميات. إذا كانت تحتوي على تسميات، فقد تكون تقنيات التصنيف مثل تلك التي تعلمتها في الدروس السابقة أكثر فائدة. ولكن في الحالات التي تسعى فيها لتجميع بيانات غير مصنفة، فإن التجميع هو وسيلة رائعة لاكتشاف الأنماط.

> هناك أدوات منخفضة البرمجة مفيدة يمكن أن تساعدك في تعلم العمل مع نماذج التجميع. جرّب [Azure ML لهذه المهمة](https://docs.microsoft.com/learn/modules/create-clustering-model-azure-machine-learning-designer/?WT.mc_id=academic-77952-leestott)

## الدروس

1. [مقدمة في التجميع](1-Visualize/README.md)
2. [تجميع K-Means](2-K-Means/README.md)

## الاعتمادات

كُتبت هذه الدروس مع 🎶 بواسطة [جين لوبير](https://www.twitter.com/jenlooper) بمراجعات مفيدة من [ريشيت داغلي](https://rishit_dagli/) و [محمد سكيب خان إنان](https://twitter.com/Sakibinan).

تم الحصول على مجموعة بيانات [الأغاني النيجيرية](https://www.kaggle.com/sootersaalu/nigerian-songs-spotify) من Kaggle تم جمعها من سبوتيفاي.

الأمثلة المفيدة لتجميع K-Means التي ساعدت في إنشاء هذا الدرس تشمل هذا [استكشاف إيريس](https://www.kaggle.com/bburns/iris-exploration-pca-k-means-and-gmm-clustering)، وهذا [دفتر تمهيدي](https://www.kaggle.com/prashant111/k-means-clustering-with-python)، وهذا [مثال من منظمة غير حكومية افتراضية](https://www.kaggle.com/ankandash/pca-k-means-clustering-hierarchical-clustering).

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**تنويه**:
تمت ترجمة هذا المستند باستخدام خدمة الترجمة بالذكاء الاصطناعي [Co-op Translator](https://github.com/Azure/co-op-translator). بينما نسعى للدقة، يرجى العلم أن الترجمات الآلية قد تحتوي على أخطاء أو عدم دقة. يجب اعتبار المستند الأصلي بلغته الأصلية المصدر الرسمي والمعتمد. للمعلومات الهامة، يُنصح بالاستعانة بترجمة بشرية محترفة. نحن غير مسؤولين عن أي سوء فهم أو تفسير ناتج عن استخدام هذه الترجمة.
<!-- CO-OP TRANSLATOR DISCLAIMER END -->