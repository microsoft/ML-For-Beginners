<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "6d130dffca5db70d7e615f926cb1ad4c",
  "translation_date": "2025-08-29T13:50:46+00:00",
  "source_file": "quiz-app/README.md",
  "language_code": "ar"
}
-->
# الاختبارات

هذه الاختبارات هي اختبارات ما قبل وبعد المحاضرات لمنهج تعلم الآلة على الرابط https://aka.ms/ml-beginners

## إعداد المشروع

```
npm install
```

### التجميع وإعادة التحميل السريع للتطوير

```
npm run serve
```

### التجميع والتصغير للإنتاج

```
npm run build
```

### التحقق من الأخطاء وإصلاح الملفات

```
npm run lint
```

### تخصيص الإعدادات

راجع [مرجع الإعدادات](https://cli.vuejs.org/config/).

الشكر: شكراً للإصدار الأصلي لهذا التطبيق الخاص بالاختبارات: https://github.com/arpan45/simple-quiz-vue

## النشر على Azure

إليك دليل خطوة بخطوة لمساعدتك على البدء:

1. نسخ مستودع GitHub
تأكد من أن كود تطبيق الويب الثابت موجود في مستودع GitHub الخاص بك. قم بنسخ هذا المستودع.

2. إنشاء تطبيق ويب ثابت على Azure
- قم بإنشاء [حساب Azure](http://azure.microsoft.com)
- انتقل إلى [بوابة Azure](https://portal.azure.com)
- انقر على "إنشاء مورد" وابحث عن "تطبيق ويب ثابت".
- انقر على "إنشاء".

3. إعداد تطبيق الويب الثابت
- الأساسيات: الاشتراك: اختر اشتراك Azure الخاص بك.
- مجموعة الموارد: قم بإنشاء مجموعة موارد جديدة أو استخدم مجموعة موجودة.
- الاسم: قدم اسمًا لتطبيق الويب الثابت الخاص بك.
- المنطقة: اختر المنطقة الأقرب لمستخدميك.

- #### تفاصيل النشر:
- المصدر: اختر "GitHub".
- حساب GitHub: قم بتفويض Azure للوصول إلى حساب GitHub الخاص بك.
- المنظمة: اختر المنظمة الخاصة بك على GitHub.
- المستودع: اختر المستودع الذي يحتوي على تطبيق الويب الثابت الخاص بك.
- الفرع: اختر الفرع الذي تريد النشر منه.

- #### تفاصيل البناء:
- إعدادات البناء: اختر الإطار الذي تم بناء تطبيقك عليه (مثل React، Angular، Vue، إلخ).
- موقع التطبيق: حدد المجلد الذي يحتوي على كود التطبيق الخاص بك (مثل / إذا كان في الجذر).
- موقع API: إذا كان لديك API، حدد موقعه (اختياري).
- موقع الإخراج: حدد المجلد الذي يتم فيه إنشاء إخراج البناء (مثل build أو dist).

4. المراجعة والإنشاء
راجع إعداداتك وانقر على "إنشاء". سيقوم Azure بإعداد الموارد اللازمة وإنشاء ملف سير عمل GitHub Actions في مستودعك.

5. سير عمل GitHub Actions
سيقوم Azure تلقائيًا بإنشاء ملف سير عمل GitHub Actions في مستودعك (.github/workflows/azure-static-web-apps-<name>.yml). هذا الملف سيتولى عملية البناء والنشر.

6. مراقبة النشر
انتقل إلى علامة التبويب "الإجراءات" في مستودع GitHub الخاص بك.
يجب أن ترى سير عمل قيد التشغيل. هذا السير سيقوم ببناء ونشر تطبيق الويب الثابت الخاص بك على Azure.
بمجرد اكتمال سير العمل، سيكون تطبيقك متاحًا على عنوان URL الذي يوفره Azure.

### مثال على ملف سير العمل

إليك مثال على ما قد يبدو عليه ملف سير عمل GitHub Actions:
name: Azure Static Web Apps CI/CD
```
on:
  push:
    branches:
      - main
  pull_request:
    types: [opened, synchronize, reopened, closed]
    branches:
      - main

jobs:
  build_and_deploy_job:
    runs-on: ubuntu-latest
    name: Build and Deploy Job
    steps:
      - uses: actions/checkout@v2
      - name: Build And Deploy
        id: builddeploy
        uses: Azure/static-web-apps-deploy@v1
        with:
          azure_static_web_apps_api_token: ${{ secrets.AZURE_STATIC_WEB_APPS_API_TOKEN }}
          repo_token: ${{ secrets.GITHUB_TOKEN }}
          action: "upload"
          app_location: "/quiz-app" # App source code path
          api_location: ""API source code path optional
          output_location: "dist" #Built app content directory - optional
```

### موارد إضافية
- [وثائق تطبيقات الويب الثابتة على Azure](https://learn.microsoft.com/azure/static-web-apps/getting-started)
- [وثائق GitHub Actions](https://docs.github.com/actions/use-cases-and-examples/deploying/deploying-to-azure-static-web-app)

---

**إخلاء المسؤولية**:  
تمت ترجمة هذا المستند باستخدام خدمة الترجمة بالذكاء الاصطناعي [Co-op Translator](https://github.com/Azure/co-op-translator). بينما نسعى لتحقيق الدقة، يرجى العلم أن الترجمات الآلية قد تحتوي على أخطاء أو عدم دقة. يجب اعتبار المستند الأصلي بلغته الأصلية المصدر الموثوق. للحصول على معلومات حساسة أو هامة، يُوصى بالاستعانة بترجمة بشرية احترافية. نحن غير مسؤولين عن أي سوء فهم أو تفسيرات خاطئة تنشأ عن استخدام هذه الترجمة.