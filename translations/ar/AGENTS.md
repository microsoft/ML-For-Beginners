<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "93fdaa0fd38836e50c4793e2f2f25e8b",
  "translation_date": "2025-10-03T10:58:35+00:00",
  "source_file": "AGENTS.md",
  "language_code": "ar"
}
-->
# AGENTS.md

## نظرة عامة على المشروع

هذا هو **تعلم الآلة للمبتدئين**، منهج شامل لمدة 12 أسبوعًا يتضمن 26 درسًا يغطي مفاهيم تعلم الآلة الكلاسيكية باستخدام Python (بشكل أساسي مع Scikit-learn) وR. تم تصميم المستودع كمورد تعليمي ذاتي مع مشاريع عملية، اختبارات، وتمارين. يستكشف كل درس مفاهيم تعلم الآلة من خلال بيانات واقعية من ثقافات ومناطق مختلفة حول العالم.

المكونات الرئيسية:
- **المحتوى التعليمي**: 26 درسًا تغطي مقدمة في تعلم الآلة، الانحدار، التصنيف، التجميع، معالجة اللغة الطبيعية، السلاسل الزمنية، والتعلم المعزز
- **تطبيق الاختبارات**: تطبيق اختبارات يعتمد على Vue.js مع تقييمات قبل وبعد الدرس
- **دعم متعدد اللغات**: ترجمات تلقائية لأكثر من 40 لغة عبر GitHub Actions
- **دعم لغتين**: الدروس متوفرة باللغتين Python (دفاتر Jupyter) وR (ملفات R Markdown)
- **تعلم قائم على المشاريع**: كل موضوع يتضمن مشاريع عملية وتمارين

## هيكل المستودع

```
ML-For-Beginners/
├── 1-Introduction/         # ML basics, history, fairness, techniques
├── 2-Regression/          # Regression models with Python/R
├── 3-Web-App/            # Flask web app for ML model deployment
├── 4-Classification/      # Classification algorithms
├── 5-Clustering/         # Clustering techniques
├── 6-NLP/               # Natural Language Processing
├── 7-TimeSeries/        # Time series forecasting
├── 8-Reinforcement/     # Reinforcement learning
├── 9-Real-World/        # Real-world ML applications
├── quiz-app/           # Vue.js quiz application
├── translations/       # Auto-generated translations
└── sketchnotes/       # Visual learning aids
```

عادةً ما يحتوي مجلد الدرس على:
- `README.md` - محتوى الدرس الرئيسي
- `notebook.ipynb` - دفتر Jupyter بلغة Python
- `solution/` - كود الحل (إصدارات Python وR)
- `assignment.md` - تمارين عملية
- `images/` - موارد بصرية

## أوامر الإعداد

### لدروس Python

تستخدم معظم الدروس دفاتر Jupyter. قم بتثبيت التبعيات المطلوبة:

```bash
# Install Python 3.8+ if not already installed
python --version

# Install Jupyter
pip install jupyter

# Install common ML libraries
pip install scikit-learn pandas numpy matplotlib seaborn

# For specific lessons, check lesson-specific requirements
# Example: Web App lesson
pip install flask
```

### لدروس R

دروس R موجودة في مجلدات `solution/R/` كملفات `.rmd` أو `.ipynb`:

```bash
# Install R and required packages
# In R console:
install.packages(c("tidyverse", "tidymodels", "caret"))
```

### لتطبيق الاختبارات

تطبيق الاختبارات هو تطبيق Vue.js موجود في دليل `quiz-app/`:

```bash
cd quiz-app
npm install
```

### لموقع التوثيق

لتشغيل التوثيق محليًا:

```bash
# Install Docsify
npm install -g docsify-cli

# Serve from repository root
docsify serve

# Access at http://localhost:3000
```

## سير العمل التطويري

### العمل مع دفاتر الدروس

1. انتقل إلى دليل الدرس (مثل `2-Regression/1-Tools/`)
2. افتح دفتر Jupyter:
   ```bash
   jupyter notebook notebook.ipynb
   ```
3. اعمل على محتوى الدرس والتمارين
4. تحقق من الحلول في مجلد `solution/` إذا لزم الأمر

### تطوير Python

- تستخدم الدروس مكتبات Python القياسية لتحليل البيانات
- دفاتر Jupyter للتعلم التفاعلي
- كود الحل متوفر في مجلد `solution/` لكل درس

### تطوير R

- دروس R تأتي بصيغة `.rmd` (R Markdown)
- الحلول موجودة في مجلدات فرعية `solution/R/`
- استخدم RStudio أو Jupyter مع نواة R لتشغيل دفاتر R

### تطوير تطبيق الاختبارات

```bash
cd quiz-app

# Start development server
npm run serve
# Access at http://localhost:8080

# Build for production
npm run build

# Lint and fix files
npm run lint
```

## تعليمات الاختبار

### اختبار تطبيق الاختبارات

```bash
cd quiz-app

# Lint code
npm run lint

# Build to verify no errors
npm run build
```

**ملاحظة**: هذا المستودع مخصص بشكل أساسي للتعليم. لا توجد اختبارات تلقائية لمحتوى الدروس. يتم التحقق من الصحة من خلال:
- إكمال تمارين الدروس
- تشغيل خلايا الدفاتر بنجاح
- مقارنة النتائج مع الحلول المتوقعة

## إرشادات أسلوب الكود

### كود Python
- اتبع إرشادات أسلوب PEP 8
- استخدم أسماء متغيرات واضحة وموصوفة
- أضف تعليقات للعمليات المعقدة
- يجب أن تحتوي دفاتر Jupyter على خلايا Markdown تشرح المفاهيم

### JavaScript/Vue.js (تطبيق الاختبارات)
- يتبع دليل أسلوب Vue.js
- تكوين ESLint في `quiz-app/package.json`
- قم بتشغيل `npm run lint` للتحقق من المشكلات وإصلاحها تلقائيًا

### التوثيق
- يجب أن تكون ملفات Markdown واضحة ومنظمة بشكل جيد
- تضمين أمثلة الكود في كتل الكود المحددة
- استخدام الروابط النسبية للإشارات الداخلية
- اتباع الاتفاقيات التنسيقية الحالية

## البناء والنشر

### نشر تطبيق الاختبارات

يمكن نشر تطبيق الاختبارات على Azure Static Web Apps:

1. **المتطلبات الأساسية**:
   - حساب Azure
   - مستودع GitHub (تم نسخه بالفعل)

2. **النشر على Azure**:
   - إنشاء مورد Azure Static Web App
   - الاتصال بمستودع GitHub
   - تحديد موقع التطبيق: `/quiz-app`
   - تحديد موقع الإخراج: `dist`
   - يقوم Azure تلقائيًا بإنشاء سير عمل GitHub Actions

3. **سير عمل GitHub Actions**:
   - يتم إنشاء ملف سير العمل في `.github/workflows/azure-static-web-apps-*.yml`
   - يتم البناء والنشر تلقائيًا عند الدفع إلى الفرع الرئيسي

### وثيقة PDF

إنشاء PDF من التوثيق:

```bash
npm install
npm run convert
```

## سير عمل الترجمة

**هام**: يتم تنفيذ الترجمات تلقائيًا عبر GitHub Actions باستخدام Co-op Translator.

- يتم إنشاء الترجمات تلقائيًا عند دفع التغييرات إلى الفرع `main`
- **لا تقم بترجمة المحتوى يدويًا** - النظام يتولى ذلك
- يتم تعريف سير العمل في `.github/workflows/co-op-translator.yml`
- يستخدم خدمات Azure AI/OpenAI للترجمة
- يدعم أكثر من 40 لغة

## إرشادات المساهمة

### للمساهمين في المحتوى

1. **قم بنسخ المستودع** وأنشئ فرعًا جديدًا
2. **قم بإجراء تغييرات على محتوى الدرس** إذا كنت تضيف أو تحدث الدروس
3. **لا تقم بتعديل الملفات المترجمة** - يتم إنشاؤها تلقائيًا
4. **اختبر الكود الخاص بك** - تأكد من تشغيل جميع خلايا الدفاتر بنجاح
5. **تحقق من الروابط والصور** للتأكد من عملها بشكل صحيح
6. **قدم طلب سحب** مع وصف واضح

### إرشادات طلب السحب

- **تنسيق العنوان**: `[القسم] وصف مختصر للتغييرات`
  - مثال: `[Regression] تصحيح خطأ في الدرس 5`
  - مثال: `[Quiz-App] تحديث التبعيات`
- **قبل التقديم**:
  - تأكد من تنفيذ جميع خلايا الدفاتر دون أخطاء
  - قم بتشغيل `npm run lint` إذا كنت تعدل تطبيق الاختبارات
  - تحقق من تنسيق Markdown
  - اختبر أي أمثلة كود جديدة
- **يجب أن يتضمن طلب السحب**:
  - وصف التغييرات
  - سبب التغييرات
  - لقطات شاشة إذا كانت هناك تغييرات في واجهة المستخدم
- **مدونة السلوك**: اتبع [مدونة قواعد السلوك مفتوحة المصدر من Microsoft](CODE_OF_CONDUCT.md)
- **CLA**: ستحتاج إلى توقيع اتفاقية ترخيص المساهم

## هيكل الدرس

يتبع كل درس نمطًا ثابتًا:

1. **اختبار ما قبل المحاضرة** - اختبار المعرفة الأساسية
2. **محتوى الدرس** - تعليمات وشروحات مكتوبة
3. **عروض الكود** - أمثلة عملية في الدفاتر
4. **فحوصات المعرفة** - التحقق من الفهم خلال الدرس
5. **التحدي** - تطبيق المفاهيم بشكل مستقل
6. **التمرين** - ممارسة موسعة
7. **اختبار ما بعد المحاضرة** - تقييم نتائج التعلم

## مرجع الأوامر الشائعة

```bash
# Python/Jupyter
jupyter notebook                    # Start Jupyter server
jupyter notebook notebook.ipynb     # Open specific notebook
pip install -r requirements.txt     # Install dependencies (where available)

# Quiz App
cd quiz-app
npm install                        # Install dependencies
npm run serve                      # Development server
npm run build                      # Production build
npm run lint                       # Lint and fix

# Documentation
docsify serve                      # Serve documentation locally
npm run convert                    # Generate PDF

# Git workflow
git checkout -b feature/my-change  # Create feature branch
git add .                         # Stage changes
git commit -m "Description"       # Commit changes
git push origin feature/my-change # Push to remote
```

## موارد إضافية

- **مجموعة Microsoft Learn**: [وحدات تعلم الآلة للمبتدئين](https://learn.microsoft.com/en-us/collections/qrqzamz1nn2wx3?WT.mc_id=academic-77952-bethanycheum)
- **تطبيق الاختبارات**: [اختبارات عبر الإنترنت](https://ff-quizzes.netlify.app/en/ml/)
- **لوحة المناقشة**: [مناقشات GitHub](https://github.com/microsoft/ML-For-Beginners/discussions)
- **مقاطع الفيديو التوضيحية**: [قائمة تشغيل YouTube](https://aka.ms/ml-beginners-videos)

## التقنيات الرئيسية

- **Python**: اللغة الأساسية لدروس تعلم الآلة (Scikit-learn، Pandas، NumPy، Matplotlib)
- **R**: تنفيذ بديل باستخدام tidyverse، tidymodels، caret
- **Jupyter**: دفاتر تفاعلية لدروس Python
- **R Markdown**: مستندات لدروس R
- **Vue.js 3**: إطار عمل تطبيق الاختبارات
- **Flask**: إطار عمل تطبيقات الويب لنشر نماذج تعلم الآلة
- **Docsify**: مولد موقع التوثيق
- **GitHub Actions**: CI/CD وترجمات تلقائية

## اعتبارات الأمان

- **لا توجد أسرار في الكود**: لا تقم أبدًا بتضمين مفاتيح API أو بيانات اعتماد
- **التبعيات**: حافظ على تحديث حزم npm وpip
- **مدخلات المستخدم**: أمثلة تطبيقات الويب باستخدام Flask تتضمن تحققًا أساسيًا من المدخلات
- **البيانات الحساسة**: مجموعات البيانات المستخدمة عامة وغير حساسة

## استكشاف الأخطاء وإصلاحها

### دفاتر Jupyter

- **مشاكل النواة**: أعد تشغيل النواة إذا توقفت الخلايا: Kernel → Restart
- **أخطاء الاستيراد**: تأكد من تثبيت جميع الحزم المطلوبة باستخدام pip
- **مشاكل المسار**: قم بتشغيل الدفاتر من الدليل الذي يحتوي عليها

### تطبيق الاختبارات

- **فشل تثبيت npm**: قم بمسح ذاكرة التخزين المؤقت لـ npm: `npm cache clean --force`
- **تعارض المنافذ**: قم بتغيير المنفذ باستخدام: `npm run serve -- --port 8081`
- **أخطاء البناء**: احذف `node_modules` وأعد التثبيت: `rm -rf node_modules && npm install`

### دروس R

- **الحزمة غير موجودة**: قم بالتثبيت باستخدام: `install.packages("package-name")`
- **عرض RMarkdown**: تأكد من تثبيت حزمة rmarkdown
- **مشاكل النواة**: قد تحتاج إلى تثبيت IRkernel لـ Jupyter

## ملاحظات خاصة بالمشروع

- هذا في الأساس **منهج تعليمي** وليس كود إنتاج
- التركيز على **فهم مفاهيم تعلم الآلة** من خلال الممارسة العملية
- أمثلة الكود تركز على **الوضوح بدلاً من التحسين**
- معظم الدروس **مستقلة** ويمكن إكمالها بشكل منفصل
- **الحلول متوفرة** ولكن يجب على المتعلمين محاولة حل التمارين أولاً
- يستخدم المستودع **Docsify** لتوثيق الويب بدون خطوة البناء
- **Sketchnotes** توفر ملخصات بصرية للمفاهيم
- **الدعم متعدد اللغات** يجعل المحتوى متاحًا عالميًا

---

**إخلاء المسؤولية**:  
تمت ترجمة هذا المستند باستخدام خدمة الترجمة بالذكاء الاصطناعي [Co-op Translator](https://github.com/Azure/co-op-translator). بينما نسعى لتحقيق الدقة، يرجى العلم أن الترجمات الآلية قد تحتوي على أخطاء أو عدم دقة. يجب اعتبار المستند الأصلي بلغته الأصلية المصدر الموثوق. للحصول على معلومات حاسمة، يُوصى بالترجمة البشرية الاحترافية. نحن غير مسؤولين عن أي سوء فهم أو تفسيرات خاطئة ناتجة عن استخدام هذه الترجمة.