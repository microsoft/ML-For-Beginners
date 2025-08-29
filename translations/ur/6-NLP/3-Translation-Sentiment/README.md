<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "6396d5d8617572cd2ac1de74fb0deb22",
  "translation_date": "2025-08-29T14:33:12+00:00",
  "source_file": "6-NLP/3-Translation-Sentiment/README.md",
  "language_code": "ur"
}
-->
# مشین لرننگ کے ذریعے ترجمہ اور جذبات کا تجزیہ

پچھلے اسباق میں آپ نے `TextBlob` لائبریری کا استعمال کرتے ہوئے ایک بنیادی بوٹ بنانے کا طریقہ سیکھا، جو بنیادی NLP کاموں جیسے اسم جملے نکالنے کے لیے مشین لرننگ کو پس پردہ استعمال کرتی ہے۔ کمپیوٹیشنل لسانیات میں ایک اور اہم چیلنج ایک جملے کو ایک زبان سے دوسری زبان میں درست طریقے سے _ترجمہ_ کرنا ہے۔

## [لیکچر سے پہلے کا کوئز](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/35/)

ترجمہ ایک بہت مشکل مسئلہ ہے کیونکہ دنیا میں ہزاروں زبانیں ہیں اور ہر ایک کے اپنے مختلف قواعد ہیں۔ ایک طریقہ یہ ہے کہ ایک زبان کے قواعد، جیسے انگریزی، کو ایک غیر زبان پر مبنی ڈھانچے میں تبدیل کیا جائے، اور پھر اسے دوسری زبان میں واپس تبدیل کر کے ترجمہ کیا جائے۔ اس طریقے میں درج ذیل مراحل شامل ہیں:

1. **شناخت**۔ ان پٹ زبان کے الفاظ کو اسم، فعل وغیرہ میں ٹیگ کریں۔
2. **ترجمہ بنائیں**۔ ہر لفظ کا براہ راست ترجمہ ہدف زبان کے فارمیٹ میں تیار کریں۔

### مثال جملہ، انگریزی سے آئرش

انگریزی میں، جملہ _I feel happy_ تین الفاظ پر مشتمل ہے اور ترتیب کچھ یوں ہے:

- **موضوع** (I)
- **فعل** (feel)
- **صفت** (happy)

تاہم، آئرش زبان میں، یہی جملہ ایک بالکل مختلف گرامر کے ڈھانچے میں ہوتا ہے - جذبات جیسے "*happy*" یا "*sad*" کو آپ پر ہونے کے طور پر ظاہر کیا جاتا ہے۔

انگریزی جملہ `I feel happy` آئرش میں `Tá athas orm` ہوگا۔ ایک *لفظی* ترجمہ ہوگا `Happy is upon me`.

ایک آئرش بولنے والا شخص انگریزی میں ترجمہ کرتے وقت کہے گا `I feel happy`, نہ کہ `Happy is upon me`, کیونکہ وہ جملے کا مطلب سمجھتا ہے، چاہے الفاظ اور جملے کی ساخت مختلف ہو۔

آئرش میں جملے کی رسمی ترتیب کچھ یوں ہے:

- **فعل** (Tá یا is)
- **صفت** (athas، یا happy)
- **موضوع** (orm، یا upon me)

## ترجمہ

ایک سادہ ترجمہ پروگرام صرف الفاظ کا ترجمہ کرے گا اور جملے کی ساخت کو نظر انداز کرے گا۔

✅ اگر آپ نے بالغ ہونے کے بعد دوسری (یا تیسری یا مزید) زبان سیکھی ہے، تو آپ نے شاید اپنی مادری زبان میں سوچنا شروع کیا ہوگا، ایک تصور کو ذہن میں لفظ بہ لفظ دوسری زبان میں ترجمہ کیا ہوگا، اور پھر اپنی ترجمہ شدہ بات کو بول دیا ہوگا۔ یہ بالکل وہی ہے جو سادہ ترجمہ کمپیوٹر پروگرام کرتے ہیں۔ روانی حاصل کرنے کے لیے اس مرحلے سے آگے بڑھنا ضروری ہے!

سادہ ترجمہ خراب (اور کبھی کبھار مزاحیہ) غلط ترجمے کی طرف لے جاتا ہے: `I feel happy` آئرش میں لفظی طور پر `Mise bhraitheann athas` میں ترجمہ ہوتا ہے۔ اس کا مطلب (لفظی طور پر) `me feel happy` ہے اور یہ آئرش میں درست جملہ نہیں ہے۔ حالانکہ انگریزی اور آئرش دو قریبی پڑوسی جزیروں پر بولی جانے والی زبانیں ہیں، وہ بہت مختلف زبانیں ہیں جن کے گرامر کے ڈھانچے مختلف ہیں۔

> آپ آئرش لسانی روایات کے بارے میں کچھ ویڈیوز دیکھ سکتے ہیں جیسے [یہ ویڈیو](https://www.youtube.com/watch?v=mRIaLSdRMMs)

### مشین لرننگ کے طریقے

اب تک، آپ نے قدرتی زبان کی پروسیسنگ کے رسمی قواعد کے طریقے کے بارے میں سیکھا ہے۔ ایک اور طریقہ یہ ہے کہ الفاظ کے معنی کو نظر انداز کریں، اور _اس کے بجائے مشین لرننگ کا استعمال کرتے ہوئے پیٹرنز کا پتہ لگائیں_۔ یہ ترجمہ میں کام کر سکتا ہے اگر آپ کے پاس اصل اور ہدف زبانوں میں بہت سا متن (ایک *corpus*) یا متون (*corpora*) موجود ہوں۔

مثال کے طور پر، *Pride and Prejudice* کو دیکھیں، جو جین آسٹن کی 1813 میں لکھی گئی ایک مشہور انگریزی ناول ہے۔ اگر آپ کتاب کو انگریزی میں اور کتاب کے *فرانسیسی* انسانی ترجمے میں دیکھیں، تو آپ ایک زبان میں ایسے جملے کا پتہ لگا سکتے ہیں جو دوسری زبان میں _محاوراتی طور پر_ ترجمہ کیا گیا ہو۔ آپ یہ کام ابھی کریں گے۔

مثال کے طور پر، جب ایک انگریزی جملہ جیسے `I have no money` کو فرانسیسی میں لفظی طور پر ترجمہ کیا جاتا ہے، تو یہ `Je n'ai pas de monnaie` بن سکتا ہے۔ "Monnaie" ایک مشکل فرانسیسی 'false cognate' ہے، کیونکہ 'money' اور 'monnaie' مترادف نہیں ہیں۔ ایک بہتر ترجمہ جو ایک انسان کر سکتا ہے وہ ہوگا `Je n'ai pas d'argent`, کیونکہ یہ بہتر طور پر یہ مطلب دیتا ہے کہ آپ کے پاس پیسے نہیں ہیں (نہ کہ 'loose change' جو 'monnaie' کا مطلب ہے)۔

![monnaie](../../../../translated_images/monnaie.606c5fa8369d5c3b3031ef0713e2069485c87985dd475cd9056bdf4c76c1f4b8.ur.png)

> تصویر [Jen Looper](https://twitter.com/jenlooper) کی جانب سے

اگر ایک ML ماڈل کے پاس کافی انسانی ترجمے موجود ہوں جن پر ماڈل بنایا جا سکے، تو یہ پہلے سے ماہر انسانی مترجمین کے ذریعے ترجمہ کیے گئے متون میں عام پیٹرنز کی شناخت کر کے ترجمے کی درستگی کو بہتر بنا سکتا ہے۔

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

یہ کہا جا سکتا ہے کہ TextBlob کا ترجمہ 1932 کے فرانسیسی ترجمے سے کہیں زیادہ درست ہے جو V. Leconte اور Ch. Pressoir نے کیا تھا:

"C'est une vérité universelle qu'un célibataire pourvu d'une belle fortune doit avoir envie de se marier, et, si peu que l'on sache de son sentiment à cet egard, lorsqu'il arrive dans une nouvelle résidence, cette idée est si bien fixée dans l'esprit de ses voisins qu'ils le considèrent sur-le-champ comme la propriété légitime de l'une ou l'autre de leurs filles."

اس معاملے میں، مشین لرننگ سے مطلع ترجمہ انسانی مترجم سے بہتر کام کرتا ہے جو غیر ضروری طور پر اصل مصنف کے الفاظ میں وضاحت کے لیے اضافی الفاظ ڈال رہا ہے۔

> یہاں کیا ہو رہا ہے؟ اور TextBlob ترجمہ میں اتنا اچھا کیوں ہے؟ دراصل، پس پردہ یہ Google Translate استعمال کر رہا ہے، جو ایک پیچیدہ AI ہے جو لاکھوں جملوں کو تجزیہ کر کے کام کے لیے بہترین سٹرنگز کی پیش گوئی کرتا ہے۔ یہاں کچھ بھی دستی نہیں ہو رہا اور آپ کو `blob.translate` استعمال کرنے کے لیے انٹرنیٹ کنکشن کی ضرورت ہے۔

✅ کچھ مزید جملے آزمائیں۔ کون بہتر ہے، مشین لرننگ یا انسانی ترجمہ؟ کن معاملات میں؟

## جذبات کا تجزیہ

مشین لرننگ ایک اور شعبے میں بھی بہت اچھا کام کر سکتی ہے، اور وہ ہے جذبات کا تجزیہ۔ جذبات کے لیے غیر مشین لرننگ طریقہ یہ ہے کہ 'مثبت' اور 'منفی' الفاظ اور جملے کی شناخت کی جائے۔ پھر، ایک نئے متن کو دیکھتے ہوئے، مثبت، منفی اور غیر جانبدار الفاظ کی کل قیمت کا حساب لگائیں تاکہ مجموعی جذبات کی شناخت کی جا سکے۔

یہ طریقہ آسانی سے دھوکہ دیا جا سکتا ہے جیسا کہ آپ نے مارون کے کام میں دیکھا ہوگا - جملہ `Great, that was a wonderful waste of time, I'm glad we are lost on this dark road` ایک طنزیہ، منفی جذبات والا جملہ ہے، لیکن سادہ الگورتھم 'great', 'wonderful', 'glad' کو مثبت اور 'waste', 'lost' اور 'dark' کو منفی کے طور پر شناخت کرتا ہے۔ مجموعی جذبات ان متضاد الفاظ سے متاثر ہوتے ہیں۔

✅ ایک لمحہ رکیں اور سوچیں کہ ہم بطور انسان بولنے والے طنز کو کیسے ظاہر کرتے ہیں۔ لہجے کی تبدیلی اس میں بڑا کردار ادا کرتی ہے۔ جملہ "Well, that film was awesome" کو مختلف طریقوں سے کہنے کی کوشش کریں تاکہ معلوم ہو کہ آپ کی آواز معنی کیسے ظاہر کرتی ہے۔

### مشین لرننگ کے طریقے

مشین لرننگ کا طریقہ یہ ہوگا کہ منفی اور مثبت متن کے مجموعے - ٹویٹس، یا فلم کے جائزے، یا کچھ بھی جہاں انسان نے اسکور *اور* تحریری رائے دی ہو - کو دستی طور پر جمع کیا جائے۔ پھر NLP تکنیکوں کو رائے اور اسکور پر لاگو کیا جا سکتا ہے، تاکہ پیٹرنز ظاہر ہوں (مثلاً، مثبت فلم کے جائزے میں 'Oscar worthy' کا ذکر منفی فلم کے جائزے کے مقابلے میں زیادہ ہوتا ہے، یا مثبت ریستوران کے جائزے میں 'gourmet' کا ذکر 'disgusting' کے مقابلے میں زیادہ ہوتا ہے)۔

> ⚖️ **مثال**: اگر آپ کسی سیاستدان کے دفتر میں کام کرتے ہیں اور کوئی نیا قانون زیر بحث ہے، تو عوام دفتر کو اس قانون کے حق میں یا اس کے خلاف ای میلز لکھ سکتے ہیں۔ فرض کریں کہ آپ کو ای میلز پڑھنے اور انہیں 2 ڈھیروں میں ترتیب دینے کا کام سونپا گیا ہے، *حق میں* اور *خلاف*۔ اگر ای میلز کی تعداد زیادہ ہو، تو آپ انہیں پڑھنے کی کوشش میں مغلوب ہو سکتے ہیں۔ کیا یہ اچھا نہ ہوگا کہ ایک بوٹ تمام ای میلز کو پڑھ سکے، انہیں سمجھ سکے اور آپ کو بتا سکے کہ کون سی ای میل کس ڈھیر میں جانی چاہیے؟ 
> 
> اس کو حاصل کرنے کا ایک طریقہ مشین لرننگ کا استعمال ہے۔ آپ ماڈل کو کچھ *خلاف* ای میلز اور کچھ *حق میں* ای میلز کے ساتھ تربیت دیں گے۔ ماڈل الفاظ اور پیٹرنز کو خلاف یا حق میں ای میلز کے ساتھ منسلک کرے گا، *لیکن یہ مواد کو نہیں سمجھے گا*، صرف یہ کہ کچھ الفاظ اور پیٹرنز خلاف یا حق میں ای میلز میں ظاہر ہونے کا زیادہ امکان رکھتے ہیں۔ آپ اسے کچھ ای میلز کے ساتھ آزمائیں گے جو آپ نے ماڈل کو تربیت دینے کے لیے استعمال نہیں کیں، اور دیکھیں گے کہ آیا یہ آپ کے نتیجے سے متفق ہے۔ پھر، جب آپ ماڈل کی درستگی سے مطمئن ہوں گے، تو آپ مستقبل کی ای میلز کو بغیر ہر ایک کو پڑھنے کے پروسیس کر سکیں گے۔

✅ کیا یہ عمل آپ نے پچھلے اسباق میں استعمال کیے گئے عمل سے مشابہت رکھتا ہے؟

## مشق - جذباتی جملے

جذبات کو *پولیریٹی* کے ساتھ -1 سے 1 تک ماپا جاتا ہے، جس کا مطلب ہے کہ -1 سب سے زیادہ منفی جذبات ہے، اور 1 سب سے زیادہ مثبت۔ جذبات کو 0 - 1 اسکور کے ساتھ معروضیت (0) اور موضوعیت (1) کے لیے بھی ماپا جاتا ہے۔

جین آسٹن کے *Pride and Prejudice* پر دوبارہ نظر ڈالیں۔ متن یہاں [Project Gutenberg](https://www.gutenberg.org/files/1342/1342-h/1342-h.htm) پر دستیاب ہے۔ نیچے دیا گیا نمونہ ایک مختصر پروگرام دکھاتا ہے جو کتاب کے پہلے اور آخری جملے کے جذبات کا تجزیہ کرتا ہے اور اس کے جذبات کی پولیریٹی اور موضوعیت/معروضیت کا اسکور ظاہر کرتا ہے۔

آپ کو درج ذیل کام میں جذبات کا تعین کرنے کے لیے `TextBlob` لائبریری (اوپر بیان کی گئی) استعمال کرنی چاہیے (آپ کو اپنا جذبات کیلکولیٹر لکھنے کی ضرورت نہیں ہے)۔

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

## چیلنج - جذباتی پولیریٹی چیک کریں

آپ کا کام یہ ہے کہ جذباتی پولیریٹی کا استعمال کرتے ہوئے یہ تعین کریں کہ *Pride and Prejudice* میں بالکل مثبت جملے زیادہ ہیں یا بالکل منفی جملے۔ اس کام کے لیے، آپ فرض کر سکتے ہیں کہ پولیریٹی اسکور 1 یا -1 بالکل مثبت یا منفی ہے۔

**مراحل:**

1. [Pride and Prejudice](https://www.gutenberg.org/files/1342/1342-h/1342-h.htm) کی ایک کاپی .txt فائل کے طور پر Project Gutenberg سے ڈاؤنلوڈ کریں۔ فائل کے آغاز اور اختتام پر موجود میٹا ڈیٹا کو ہٹا دیں، صرف اصل متن چھوڑ دیں۔
2. فائل کو Python میں کھولیں اور مواد کو ایک سٹرنگ کے طور پر نکالیں۔
3. کتاب کی سٹرنگ کا استعمال کرتے ہوئے ایک TextBlob بنائیں۔
4. کتاب میں ہر جملے کا ایک لوپ میں تجزیہ کریں۔
   1. اگر پولیریٹی 1 یا -1 ہو تو جملے کو مثبت یا منفی پیغامات کی ایک array یا list میں محفوظ کریں۔
5. آخر میں، تمام مثبت جملے اور منفی جملے (الگ الگ) اور ان کی تعداد پرنٹ کریں۔

یہاں ایک نمونہ [حل](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/3-Translation-Sentiment/solution/notebook.ipynb) موجود ہے۔

✅ علم کی جانچ

1. جذبات الفاظ کے استعمال پر مبنی ہیں، لیکن کیا کوڈ *الفاظ کو سمجھتا ہے*؟
2. کیا آپ کو جذباتی پولیریٹی درست لگتی ہے، یا دوسرے الفاظ میں، کیا آپ اس اسکور سے *متفق* ہیں؟
   1. خاص طور پر، کیا آپ درج ذیل جملوں کی بالکل **مثبت** پولیریٹی سے متفق ہیں؟
      * “What an excellent father you have, girls!” said she, when the door was shut.
      * “Your examination of Mr. Darcy is over, I presume,” said Miss Bingley; “and pray what is the result?” “I am perfectly convinced by it that Mr. Darcy has no defect.
      * How wonderfully these sort of things occur!
      * I have the greatest dislike in the world to that sort of thing.
      * Charlotte is an excellent manager, I dare say.
      * “This is delightful indeed!
      * I am so happy!
      * Your idea of the ponies is delightful.
   2. اگلے 3 جملے بالکل مثبت جذبات کے ساتھ اسکور کیے گئے تھے، لیکن قریب سے پڑھنے پر، وہ مثبت جملے نہیں ہیں۔ جذباتی تجزیہ نے انہیں مثبت جملے کیوں سمجھا؟
      * Happy shall I be, when his stay at Netherfield is over!” “I wish I could say anything to comfort you,” replied Elizabeth; “but it is wholly out of my power.
      * If I could but see you as happy!
      * Our distress, my dear Lizzy, is very great.
   3. کیا آپ درج ذیل جملوں کی بالکل **منفی** پولیریٹی سے متفق ہیں؟
      - Everybody is disgusted with his pride.
      - “I should like to know how he behaves among strangers.” “You shall hear then—but prepare yourself for something very dreadful.
      - The pause was to Elizabeth’s feelings dreadful.
      - It would be dreadful!

✅ جین آسٹن کے کسی بھی شیدائی کو یہ سمجھنا مشکل نہیں ہوگا کہ وہ اکثر اپنی کتابوں میں انگریزی ریجنسی معاشرے کے زیادہ مضحکہ خیز پہلوؤں پر تنقید کرتی ہیں۔ *Pride and Prejudice* کی مرکزی کردار، الزبتھ بینٹ، ایک تیز سماجی مبصر ہیں (جیسے مصنفہ) اور ان کی زبان اکثر بہت زیادہ معنی خیز ہوتی ہے۔ یہاں تک کہ مسٹر ڈارسی (کہانی میں محبت کا کردار) بھی الزبتھ کی زبان کے چنچل اور طنزیہ استعمال کو نوٹ کرتے ہیں: "I have had the pleasure of your acquaintance long enough to know that you find great enjoyment in occasionally professing opinions which in fact are not your own."

---

## 🚀چیلنج

کیا آپ مارون کو مزید بہتر بنا سکتے ہیں تاکہ صارف کے ان پٹ سے دیگر خصوصیات نکالی جا سکیں؟

## [لیکچر کے بعد کا کوئز](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/36/)

## جائزہ اور خود مطالعہ
متن سے جذبات نکالنے کے کئی طریقے ہیں۔ ان کاروباری اطلاقات کے بارے میں سوچیں جو اس تکنیک کا استعمال کر سکتے ہیں۔ یہ بھی سوچیں کہ یہ کیسے غلط ہو سکتا ہے۔ جذبات کا تجزیہ کرنے والے جدید اور کاروباری نظاموں کے بارے میں مزید پڑھیں جیسے [Azure Text Analysis](https://docs.microsoft.com/azure/cognitive-services/Text-Analytics/how-tos/text-analytics-how-to-sentiment-analysis?tabs=version-3-1?WT.mc_id=academic-77952-leestott)۔ اوپر دی گئی "Pride and Prejudice" کی کچھ جملے آزمائیں اور دیکھیں کہ آیا یہ باریکیوں کو سمجھ سکتا ہے۔

## اسائنمنٹ

[Poetic license](assignment.md)

---

**ڈسکلیمر**:  
یہ دستاویز AI ترجمہ سروس [Co-op Translator](https://github.com/Azure/co-op-translator) کا استعمال کرتے ہوئے ترجمہ کی گئی ہے۔ ہم درستگی کے لیے کوشش کرتے ہیں، لیکن براہ کرم آگاہ رہیں کہ خودکار ترجمے میں غلطیاں یا عدم درستگی ہو سکتی ہیں۔ اصل دستاویز کو اس کی اصل زبان میں مستند ذریعہ سمجھا جانا چاہیے۔ اہم معلومات کے لیے، پیشہ ور انسانی ترجمہ کی سفارش کی جاتی ہے۔ اس ترجمے کے استعمال سے پیدا ہونے والی کسی بھی غلط فہمی یا غلط تشریح کے لیے ہم ذمہ دار نہیں ہیں۔