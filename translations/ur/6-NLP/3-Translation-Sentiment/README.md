<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "be03c8182982b87ced155e4e9d1438e8",
  "translation_date": "2025-09-06T09:02:30+00:00",
  "source_file": "6-NLP/3-Translation-Sentiment/README.md",
  "language_code": "ur"
}
-->
# مشین لرننگ کے ذریعے ترجمہ اور جذبات کا تجزیہ

پچھلے اسباق میں آپ نے سیکھا کہ `TextBlob` لائبریری کا استعمال کرتے ہوئے ایک بنیادی بوٹ کیسے بنایا جائے، جو پس منظر میں مشین لرننگ کو استعمال کرتے ہوئے بنیادی قدرتی زبان کے کام انجام دیتی ہے، جیسے کہ اسم جملے نکالنا۔ کمپیوٹیشنل لسانیات میں ایک اور اہم چیلنج ایک زبان سے دوسری زبان میں جملے کا درست _ترجمہ_ کرنا ہے۔

## [لیکچر سے پہلے کا کوئز](https://ff-quizzes.netlify.app/en/ml/)

ترجمہ ایک بہت مشکل مسئلہ ہے کیونکہ دنیا میں ہزاروں زبانیں ہیں اور ہر زبان کے اپنے منفرد گرامر کے اصول ہیں۔ ایک طریقہ یہ ہے کہ ایک زبان، جیسے انگریزی، کے گرامر کے اصولوں کو ایک غیر زبان پر مبنی ڈھانچے میں تبدیل کیا جائے، اور پھر اسے دوسری زبان میں واپس تبدیل کیا جائے۔ اس طریقے میں درج ذیل مراحل شامل ہیں:

1. **شناخت**۔ ان پٹ زبان کے الفاظ کو اسم، فعل وغیرہ میں ٹیگ کریں۔
2. **ترجمہ بنائیں**۔ ہدف زبان کے فارمیٹ میں ہر لفظ کا براہ راست ترجمہ تیار کریں۔

### مثال جملہ، انگریزی سے آئرش

'انگریزی' میں، جملہ _I feel happy_ تین الفاظ پر مشتمل ہے اور ترتیب یہ ہے:

- **فاعل** (I)
- **فعل** (feel)
- **صفت** (happy)

لیکن 'آئرش' زبان میں، یہی جملہ ایک بالکل مختلف گرامر کے ڈھانچے کا حامل ہے - جذبات جیسے "*happy*" یا "*sad*" کو آپ پر ہونے کے طور پر بیان کیا جاتا ہے۔

انگریزی جملہ `I feel happy` آئرش میں `Tá athas orm` ہوگا۔ ایک *لفظی* ترجمہ ہوگا `Happy is upon me`۔

ایک آئرش بولنے والا جب انگریزی میں ترجمہ کرے گا تو وہ کہے گا `I feel happy`، نہ کہ `Happy is upon me`، کیونکہ وہ جملے کے معنی کو سمجھتا ہے، چاہے الفاظ اور جملے کی ساخت مختلف ہو۔

آئرش میں جملے کی رسمی ترتیب یہ ہے:

- **فعل** (Tá یا is)
- **صفت** (athas، یا happy)
- **فاعل** (orm، یا upon me)

## ترجمہ

ایک سادہ ترجمہ پروگرام صرف الفاظ کا ترجمہ کرے گا اور جملے کی ساخت کو نظر انداز کرے گا۔

✅ اگر آپ نے بالغ ہونے کے بعد دوسری (یا تیسری یا اس سے زیادہ) زبان سیکھی ہے، تو آپ نے شاید اپنی مادری زبان میں سوچنا شروع کیا ہوگا، ایک تصور کو اپنے ذہن میں لفظ بہ لفظ دوسری زبان میں ترجمہ کیا ہوگا، اور پھر اپنی ترجمہ شدہ بات کہی ہوگی۔ یہ وہی ہے جو سادہ ترجمہ کمپیوٹر پروگرام کرتے ہیں۔ روانی حاصل کرنے کے لیے اس مرحلے سے آگے بڑھنا ضروری ہے!

سادہ ترجمہ خراب (اور بعض اوقات مزاحیہ) غلط تراجم کی طرف لے جاتا ہے: `I feel happy` کا لفظی ترجمہ آئرش میں `Mise bhraitheann athas` ہوگا۔ اس کا مطلب (لفظی طور پر) `me feel happy` ہے اور یہ ایک درست آئرش جملہ نہیں ہے۔ حالانکہ انگریزی اور آئرش دو قریبی جزیروں پر بولی جانے والی زبانیں ہیں، لیکن یہ بہت مختلف زبانیں ہیں جن کے گرامر کے ڈھانچے مختلف ہیں۔

> آپ آئرش لسانی روایات کے بارے میں کچھ ویڈیوز دیکھ سکتے ہیں، جیسے کہ [یہ ویڈیو](https://www.youtube.com/watch?v=mRIaLSdRMMs)

### مشین لرننگ کے طریقے

اب تک، آپ نے قدرتی زبان کی پروسیسنگ کے رسمی اصولوں کے بارے میں سیکھا ہے۔ ایک اور طریقہ یہ ہے کہ الفاظ کے معنی کو نظر انداز کریں، اور _مشین لرننگ کا استعمال کرتے ہوئے پیٹرن کا پتہ لگائیں_۔ اگر آپ کے پاس اصل اور ہدف زبانوں میں بہت زیادہ متن (ایک *corpus*) یا متون (*corpora*) ہوں تو یہ ترجمہ میں کام کر سکتا ہے۔

مثال کے طور پر، *Pride and Prejudice* کو دیکھیں، جو جین آسٹن کا 1813 میں لکھا گیا ایک مشہور انگریزی ناول ہے۔ اگر آپ کتاب کو انگریزی میں اور اس کا انسانی ترجمہ *فرانسیسی* میں دیکھیں، تو آپ ایک زبان کے جملوں کو دوسری زبان میں *محاوراتی* طور پر ترجمہ شدہ دیکھ سکتے ہیں۔ آپ ابھی یہ کریں گے۔

مثال کے طور پر، جب ایک انگریزی جملہ `I have no money` کو فرانسیسی میں لفظی طور پر ترجمہ کیا جاتا ہے، تو یہ `Je n'ai pas de monnaie` بن سکتا ہے۔ "Monnaie" ایک مشکل فرانسیسی 'false cognate' ہے، کیونکہ 'money' اور 'monnaie' مترادف نہیں ہیں۔ ایک بہتر انسانی ترجمہ ہوگا `Je n'ai pas d'argent`، کیونکہ یہ بہتر طور پر اس بات کو ظاہر کرتا ہے کہ آپ کے پاس پیسے نہیں ہیں (نہ کہ 'loose change' جو 'monnaie' کا مطلب ہے)۔

![monnaie](../../../../6-NLP/3-Translation-Sentiment/images/monnaie.png)

> تصویر بشکریہ [Jen Looper](https://twitter.com/jenlooper)

اگر کسی مشین لرننگ ماڈل کے پاس انسانی تراجم کی کافی مقدار ہو، تو یہ ماڈل بنا کر تراجم کی درستگی کو بہتر بنا سکتا ہے، اور ان متون میں عام پیٹرن کی شناخت کر سکتا ہے جو پہلے ماہر انسانی مترجمین نے کیے ہوں۔

### مشق - ترجمہ

آپ جملوں کا ترجمہ کرنے کے لیے `TextBlob` استعمال کر سکتے ہیں۔ **Pride and Prejudice** کی مشہور پہلی لائن آزمائیں:

```python
from textblob import TextBlob

blob = TextBlob(
    "It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want of a wife!"
)
print(blob.translate(to="fr"))

```

`TextBlob` ترجمہ کافی اچھا کرتا ہے: "C'est une vérité universellement reconnue, qu'un homme célibataire en possession d'une bonne fortune doit avoir besoin d'une femme!".

یہ کہا جا سکتا ہے کہ TextBlob کا ترجمہ 1932 میں V. Leconte اور Ch. Pressoir کے فرانسیسی ترجمے سے کہیں زیادہ درست ہے:

"C'est une vérité universelle qu'un célibataire pourvu d'une belle fortune doit avoir envie de se marier, et, si peu que l'on sache de son sentiment à cet egard, lorsqu'il arrive dans une nouvelle résidence, cette idée est si bien fixée dans l'esprit de ses voisins qu'ils le considèrent sur-le-champ comme la propriété légitime de l'une ou l'autre de leurs filles."

اس معاملے میں، مشین لرننگ سے مطلع شدہ ترجمہ انسانی مترجم سے بہتر کام کرتا ہے، جو غیر ضروری طور پر اصل مصنف کے الفاظ میں وضاحت کے لیے اضافے کر رہا ہے۔

> یہاں کیا ہو رہا ہے؟ اور TextBlob ترجمہ میں اتنا اچھا کیوں ہے؟ دراصل، یہ پس منظر میں Google Translate استعمال کر رہا ہے، جو ایک جدید AI ہے جو لاکھوں جملوں کو تجزیہ کر کے کام کے لیے بہترین جملے پیش کرتا ہے۔ یہاں کچھ بھی دستی نہیں ہو رہا اور آپ کو `blob.translate` استعمال کرنے کے لیے انٹرنیٹ کنکشن کی ضرورت ہے۔

✅ مزید جملے آزمائیں۔ کون بہتر ہے، مشین لرننگ یا انسانی ترجمہ؟ کن معاملات میں؟

## جذبات کا تجزیہ

مشین لرننگ ایک اور شعبے میں بھی بہت اچھا کام کر سکتی ہے، اور وہ ہے جذبات کا تجزیہ۔ جذبات کا غیر مشین لرننگ طریقہ یہ ہے کہ 'مثبت' اور 'منفی' الفاظ اور جملے کی شناخت کی جائے۔ پھر، ایک نئے متن کو دیکھتے ہوئے، مثبت، منفی اور غیر جانبدار الفاظ کی کل قیمت کا حساب لگائیں تاکہ مجموعی جذبات کی شناخت ہو سکے۔

یہ طریقہ آسانی سے دھوکہ کھا سکتا ہے جیسا کہ آپ نے مارون کے کام میں دیکھا ہوگا - جملہ `Great, that was a wonderful waste of time, I'm glad we are lost on this dark road` ایک طنزیہ، منفی جذبات والا جملہ ہے، لیکن سادہ الگورتھم 'great', 'wonderful', 'glad' کو مثبت اور 'waste', 'lost' اور 'dark' کو منفی کے طور پر شناخت کرتا ہے۔ متضاد الفاظ کی وجہ سے مجموعی جذبات متاثر ہوتے ہیں۔

✅ ایک لمحے کے لیے رکیں اور سوچیں کہ ہم بطور انسان بولنے والے طنز کو کیسے ظاہر کرتے ہیں۔ لہجے کی جھلک اس میں بڑا کردار ادا کرتی ہے۔ جملہ "Well, that film was awesome" مختلف طریقوں سے کہنے کی کوشش کریں تاکہ معلوم ہو کہ آپ کی آواز معنی کیسے ظاہر کرتی ہے۔

### مشین لرننگ کے طریقے

مشین لرننگ کا طریقہ یہ ہوگا کہ منفی اور مثبت متون کو دستی طور پر جمع کیا جائے - جیسے ٹویٹس، فلمی جائزے، یا کوئی بھی چیز جہاں انسان نے اسکور *اور* تحریری رائے دی ہو۔ پھر NLP تکنیکوں کو رائے اور اسکور پر لاگو کیا جا سکتا ہے تاکہ پیٹرن ابھر سکیں (مثلاً، مثبت فلمی جائزوں میں 'Oscar worthy' کا ذکر منفی فلمی جائزوں کے مقابلے میں زیادہ ہوتا ہے، یا مثبت ریستوران کے جائزے 'gourmet' کا ذکر 'disgusting' کے مقابلے میں زیادہ کرتے ہیں)۔

> ⚖️ **مثال**: فرض کریں آپ کسی سیاستدان کے دفتر میں کام کرتے ہیں اور کوئی نیا قانون زیر بحث ہے۔ شہری دفتر کو اس قانون کے حق میں یا اس کے خلاف ای میلز لکھ سکتے ہیں۔ فرض کریں آپ کو ان ای میلز کو پڑھ کر دو ڈھیروں میں تقسیم کرنے کا کام دیا گیا ہے، *حق میں* اور *خلاف*۔ اگر ای میلز کی تعداد زیادہ ہو، تو آپ ان سب کو پڑھنے کی کوشش میں مغلوب ہو سکتے ہیں۔ کیا یہ اچھا نہ ہوگا کہ ایک بوٹ ان سب کو آپ کے لیے پڑھ لے، انہیں سمجھے اور بتائے کہ کون سی ای میل کس ڈھیر میں جانی چاہیے؟ 
> 
> اس مقصد کو حاصل کرنے کا ایک طریقہ مشین لرننگ ہے۔ آپ ماڈل کو کچھ *خلاف* ای میلز اور کچھ *حق میں* ای میلز کے ساتھ تربیت دیں گے۔ ماڈل الفاظ اور جملوں کو خلاف یا حق میں ای میلز کے ساتھ منسلک کرے گا، *لیکن یہ مواد کو نہیں سمجھے گا*، صرف یہ کہ کچھ الفاظ اور پیٹرن خلاف یا حق میں ای میلز میں زیادہ ظاہر ہوتے ہیں۔ آپ اسے کچھ ایسی ای میلز کے ساتھ جانچ سکتے ہیں جو آپ نے ماڈل کو تربیت دینے کے لیے استعمال نہیں کیں، اور دیکھ سکتے ہیں کہ آیا یہ آپ کے نتیجے سے متفق ہے۔ پھر، جب آپ ماڈل کی درستگی سے مطمئن ہوں، تو آپ مستقبل کی ای میلز کو پڑھے بغیر پروسیس کر سکتے ہیں۔

✅ کیا یہ عمل آپ کے پچھلے اسباق میں استعمال کیے گئے عمل سے مشابہت رکھتا ہے؟

## مشق - جذباتی جملے

جذبات کو *پولیرٹی* کے ساتھ ماپا جاتا ہے، جو -1 سے 1 تک ہوتی ہے، جہاں -1 سب سے زیادہ منفی جذبات ہے، اور 1 سب سے زیادہ مثبت۔ جذبات کو 0 - 1 کے اسکور کے ساتھ معروضیت (0) اور موضوعیت (1) کے لیے بھی ماپا جاتا ہے۔

جین آسٹن کے *Pride and Prejudice* پر دوبارہ نظر ڈالیں۔ متن یہاں دستیاب ہے: [Project Gutenberg](https://www.gutenberg.org/files/1342/1342-h/1342-h.htm)۔ نیچے دیا گیا نمونہ ایک مختصر پروگرام دکھاتا ہے جو کتاب کے پہلے اور آخری جملوں کے جذبات کا تجزیہ کرتا ہے اور اس کے جذبات کی پولیرٹی اور موضوعیت/معروضیت کا اسکور دکھاتا ہے۔

آپ کو `TextBlob` لائبریری (جیسا کہ اوپر بیان کیا گیا ہے) کا استعمال کرتے ہوئے `sentiment` کا تعین کرنا چاہیے (آپ کو اپنا جذبات کیلکولیٹر لکھنے کی ضرورت نہیں ہے)۔

```python
from textblob import TextBlob

quote1 = """It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want of a wife."""

quote2 = """Darcy, as well as Elizabeth, really loved them; and they were both ever sensible of the warmest gratitude towards the persons who, by bringing her into Derbyshire, had been the means of uniting them."""

sentiment1 = TextBlob(quote1).sentiment
sentiment2 = TextBlob(quote2).sentiment

print(quote1 + " has a sentiment of " + str(sentiment1))
print(quote2 + " has a sentiment of " + str(sentiment2))
```

آپ کو درج ذیل آؤٹ پٹ نظر آتا ہے:

```output
It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want # of a wife. has a sentiment of Sentiment(polarity=0.20952380952380953, subjectivity=0.27142857142857146)

Darcy, as well as Elizabeth, really loved them; and they were
     both ever sensible of the warmest gratitude towards the persons
      who, by bringing her into Derbyshire, had been the means of
      uniting them. has a sentiment of Sentiment(polarity=0.7, subjectivity=0.8)
```

## چیلنج - جذباتی پولیرٹی کی جانچ کریں

آپ کا کام یہ طے کرنا ہے کہ آیا *Pride and Prejudice* میں بالکل مثبت جملے زیادہ ہیں یا بالکل منفی جملے۔ اس کام کے لیے، آپ فرض کر سکتے ہیں کہ پولیرٹی اسکور 1 یا -1 بالکل مثبت یا منفی ہے۔

**مراحل:**

1. [Pride and Prejudice کا ایک نسخہ](https://www.gutenberg.org/files/1342/1342-h/1342-h.htm) Project Gutenberg سے .txt فائل کے طور پر ڈاؤن لوڈ کریں۔ فائل کے آغاز اور اختتام پر موجود میٹا ڈیٹا کو ہٹا دیں، صرف اصل متن چھوڑ دیں۔
2. فائل کو Python میں کھولیں اور مواد کو ایک سٹرنگ کے طور پر نکالیں۔
3. کتاب کی سٹرنگ کا استعمال کرتے ہوئے ایک TextBlob بنائیں۔
4. کتاب کے ہر جملے کا ایک لوپ میں تجزیہ کریں۔
   1. اگر پولیرٹی 1 یا -1 ہو تو جملے کو مثبت یا منفی پیغامات کی ایک فہرست میں محفوظ کریں۔
5. آخر میں، تمام مثبت جملے اور منفی جملے (الگ الگ) اور ان کی تعداد پرنٹ کریں۔

یہاں ایک نمونہ [حل](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/3-Translation-Sentiment/solution/notebook.ipynb) ہے۔

✅ علم کی جانچ

1. جذبات الفاظ کے استعمال پر مبنی ہیں، لیکن کیا کوڈ الفاظ کو *سمجھتا* ہے؟
2. کیا آپ کو جذبات کی پولیرٹی درست لگتی ہے، یا دوسرے الفاظ میں، کیا آپ اسکورز سے *متفق* ہیں؟
   1. خاص طور پر، کیا آپ درج ذیل جملوں کی بالکل **مثبت** پولیرٹی سے متفق ہیں؟
      * “What an excellent father you have, girls!” said she, when the door was shut.
      * “Your examination of Mr. Darcy is over, I presume,” said Miss Bingley; “and pray what is the result?” “I am perfectly convinced by it that Mr. Darcy has no defect.
      * How wonderfully these sort of things occur!
      * I have the greatest dislike in the world to that sort of thing.
      * Charlotte is an excellent manager, I dare say.
      * “This is delightful indeed!
      * I am so happy!
      * Your idea of the ponies is delightful.
   2. اگلے 3 جملوں کو بالکل مثبت جذبات کے ساتھ اسکور کیا گیا، لیکن قریب سے پڑھنے پر، یہ مثبت جملے نہیں ہیں۔ جذبات کے تجزیے نے کیوں سوچا کہ یہ مثبت جملے ہیں؟
      * Happy shall I be, when his stay at Netherfield is over!” “I wish I could say anything to comfort you,” replied Elizabeth; “but it is wholly out of my power.
      * If I could but see you as happy!
      * Our distress, my dear Lizzy, is very great.
   3. کیا آپ درج ذیل جملوں کی بالکل **منفی** پولیرٹی سے متفق ہیں؟
      - Everybody is disgusted with his pride.
      - “I should like to know how he behaves among strangers.” “You shall hear then—but prepare yourself for something very dreadful.
      - The pause was to Elizabeth’s feelings dreadful.
      - It would be dreadful!

✅ جین آسٹن کے کسی بھی شیدائی کو یہ سمجھنے میں دیر نہیں لگے گی کہ وہ اکثر اپنی کتابوں میں انگریزی ریجنسی معاشرے کے زیادہ مضحکہ خیز پہلوؤں پر تنقید کرتی ہیں۔ *Pride and Prejudice* کی مرکزی کردار، الزبتھ بینیٹ، ایک تیز مشاہدہ کرنے والی ہے (مصنف کی طرح) اور اس کی زبان اکثر گہرے معنی رکھتی ہے۔ حتیٰ کہ مسٹر ڈارسی (کہانی کے محبت کے کردار) بھی الزبتھ کی زبان کے چنچل اور طنزیہ استعمال کو نوٹ کرتے ہیں: "I have had the pleasure of your acquaintance long enough to know that you find great enjoyment in occasionally professing opinions which in fact are not your own."

---

## 🚀چیلنج

کیا آپ مارون کو مزید بہتر بنا سکتے ہیں تاکہ صارف کے ان پٹ سے دیگر خصوصیات نکالی جا سکیں؟

## [لیکچر کے بعد کا کوئز](https://ff-quizzes.netlify.app/en/ml/)

## جائزہ اور خود مطالعہ
متن سے جذبات نکالنے کے کئی طریقے ہیں۔ ان کاروباری ایپلیکیشنز کے بارے میں سوچیں جو اس تکنیک کا استعمال کر سکتی ہیں۔ یہ بھی سوچیں کہ یہ کیسے غلط ہو سکتی ہے۔ جذبات کا تجزیہ کرنے والے جدید اور کاروباری نظاموں کے بارے میں مزید پڑھیں جیسے [Azure Text Analysis](https://docs.microsoft.com/azure/cognitive-services/Text-Analytics/how-tos/text-analytics-how-to-sentiment-analysis?tabs=version-3-1?WT.mc_id=academic-77952-leestott)۔ اوپر دی گئی "Pride and Prejudice" کی کچھ جملے آزمائیں اور دیکھیں کہ آیا یہ باریکیوں کو سمجھ سکتا ہے۔

## اسائنمنٹ

[Poetic license](assignment.md)

---

**ڈسکلیمر**:  
یہ دستاویز AI ترجمہ سروس [Co-op Translator](https://github.com/Azure/co-op-translator) کا استعمال کرتے ہوئے ترجمہ کی گئی ہے۔ ہم درستگی کے لیے کوشش کرتے ہیں، لیکن براہ کرم آگاہ رہیں کہ خودکار ترجمے میں غلطیاں یا عدم درستگی ہو سکتی ہیں۔ اصل دستاویز، جو اس کی اصل زبان میں ہے، کو مستند ذریعہ سمجھا جانا چاہیے۔ اہم معلومات کے لیے، پیشہ ور انسانی ترجمہ کی سفارش کی جاتی ہے۔ اس ترجمے کے استعمال سے پیدا ہونے والی کسی بھی غلط فہمی یا غلط تشریح کے لیے ہم ذمہ دار نہیں ہیں۔