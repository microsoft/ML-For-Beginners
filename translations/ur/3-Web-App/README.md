<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "9836ff53cfef716ddfd70e06c5f43436",
  "translation_date": "2025-08-29T13:47:43+00:00",
  "source_file": "3-Web-App/README.md",
  "language_code": "ur"
}
-->
# اپنی مشین لرننگ ماڈل کے استعمال کے لیے ویب ایپ بنائیں

اس نصاب کے اس حصے میں، آپ کو ایک عملی مشین لرننگ موضوع سے متعارف کرایا جائے گا: اپنے Scikit-learn ماڈل کو ایک فائل کے طور پر محفوظ کرنے کا طریقہ جو ویب ایپلیکیشن کے اندر پیش گوئی کرنے کے لیے استعمال کی جا سکتی ہے۔ جب ماڈل محفوظ ہو جائے گا، تو آپ سیکھیں گے کہ اسے Flask میں بنائی گئی ویب ایپ میں کیسے استعمال کریں۔ آپ پہلے کچھ ڈیٹا کا استعمال کرتے ہوئے ایک ماڈل بنائیں گے جو UFO دیکھنے کے بارے میں ہے! پھر، آپ ایک ویب ایپ بنائیں گے جو آپ کو سیکنڈز کی تعداد، عرض بلد اور طول بلد کی قدر درج کرنے کی اجازت دے گی تاکہ یہ پیش گوئی کی جا سکے کہ کس ملک نے UFO دیکھنے کی اطلاع دی۔

![UFO Parking](../../../translated_images/ufo.9e787f5161da9d4d1dafc537e1da09be8210f2ee996cb638aa5cee1d92867a04.ur.jpg)

تصویر از <a href="https://unsplash.com/@mdherren?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">مائیکل ہیرن</a> پر <a href="https://unsplash.com/s/photos/ufo?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Unsplash</a>

## اسباق

1. [ویب ایپ بنائیں](1-Web-App/README.md)

## کریڈٹس

"ویب ایپ بنائیں" کو ♥️ کے ساتھ [Jen Looper](https://twitter.com/jenlooper) نے لکھا۔

♥️ کوئزز کو روہن راج نے لکھا۔

ڈیٹا سیٹ [Kaggle](https://www.kaggle.com/NUFORC/ufo-sightings) سے حاصل کیا گیا ہے۔

ویب ایپ کی آرکیٹیکچر جزوی طور پر [اس مضمون](https://towardsdatascience.com/how-to-easily-deploy-machine-learning-models-using-flask-b95af8fe34d4) اور [اس ریپو](https://github.com/abhinavsagar/machine-learning-deployment) کے ذریعے تجویز کی گئی تھی، جو ابھینو ساگر نے لکھی۔

---

**ڈسکلیمر**:  
یہ دستاویز AI ترجمہ سروس [Co-op Translator](https://github.com/Azure/co-op-translator) کا استعمال کرتے ہوئے ترجمہ کی گئی ہے۔ ہم درستگی کے لیے کوشش کرتے ہیں، لیکن براہ کرم آگاہ رہیں کہ خودکار ترجمے میں غلطیاں یا عدم درستگی ہو سکتی ہیں۔ اصل دستاویز، جو اس کی مقامی زبان میں ہے، کو مستند ذریعہ سمجھا جانا چاہیے۔ اہم معلومات کے لیے، پیشہ ور انسانی ترجمہ کی سفارش کی جاتی ہے۔ اس ترجمے کے استعمال سے پیدا ہونے والی کسی بھی غلط فہمی یا غلط تشریح کے لیے ہم ذمہ دار نہیں ہیں۔