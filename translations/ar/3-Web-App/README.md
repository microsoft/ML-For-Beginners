<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "9836ff53cfef716ddfd70e06c5f43436",
  "translation_date": "2025-08-29T13:47:34+00:00",
  "source_file": "3-Web-App/README.md",
  "language_code": "ar"
}
-->
# بناء تطبيق ويب لاستخدام نموذج التعلم الآلي الخاص بك

في هذا القسم من المنهج، ستتعرف على موضوع عملي في التعلم الآلي: كيفية حفظ نموذج Scikit-learn كملف يمكن استخدامه لإجراء التنبؤات داخل تطبيق ويب. بمجرد حفظ النموذج، ستتعلم كيفية استخدامه في تطبيق ويب مبني باستخدام Flask. ستقوم أولاً بإنشاء نموذج باستخدام بعض البيانات المتعلقة بمشاهدات الأجسام الطائرة المجهولة (UFO)! بعد ذلك، ستبني تطبيق ويب يسمح لك بإدخال عدد من الثواني مع قيمة خط العرض وخط الطول للتنبؤ بالدولة التي أبلغت عن رؤية جسم طائر مجهول.

![UFO Parking](../../../translated_images/ufo.9e787f5161da9d4d1dafc537e1da09be8210f2ee996cb638aa5cee1d92867a04.ar.jpg)

صورة بواسطة <a href="https://unsplash.com/@mdherren?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Michael Herren</a> على <a href="https://unsplash.com/s/photos/ufo?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Unsplash</a>

## الدروس

1. [بناء تطبيق ويب](1-Web-App/README.md)

## الشكر

تم كتابة "بناء تطبيق ويب" بحب ♥️ بواسطة [Jen Looper](https://twitter.com/jenlooper).

♥️ تم كتابة الاختبارات بواسطة روهان راج.

تم الحصول على مجموعة البيانات من [Kaggle](https://www.kaggle.com/NUFORC/ufo-sightings).

تم اقتراح بنية تطبيق الويب جزئيًا من خلال [هذا المقال](https://towardsdatascience.com/how-to-easily-deploy-machine-learning-models-using-flask-b95af8fe34d4) و[هذا المستودع](https://github.com/abhinavsagar/machine-learning-deployment) بواسطة Abhinav Sagar.

---

**إخلاء المسؤولية**:  
تمت ترجمة هذا المستند باستخدام خدمة الترجمة الآلية [Co-op Translator](https://github.com/Azure/co-op-translator). بينما نسعى لتحقيق الدقة، يرجى العلم أن الترجمات الآلية قد تحتوي على أخطاء أو معلومات غير دقيقة. يجب اعتبار المستند الأصلي بلغته الأصلية هو المصدر الموثوق. للحصول على معلومات حساسة أو هامة، يُوصى بالاستعانة بترجمة بشرية احترافية. نحن غير مسؤولين عن أي سوء فهم أو تفسيرات خاطئة تنشأ عن استخدام هذه الترجمة.