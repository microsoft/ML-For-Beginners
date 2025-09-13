<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "be03c8182982b87ced155e4e9d1438e8",
  "translation_date": "2025-09-05T14:16:35+00:00",
  "source_file": "6-NLP/3-Translation-Sentiment/README.md",
  "language_code": "my"
}
-->
# Machine Learning ဖြင့် ဘာသာပြန်ခြင်းနှင့်ခံစားချက်ခွဲခြားခြင်း

ယခင်သင်ခန်းစာများတွင် `TextBlob` ကို အသုံးပြု၍ အခြေခံ bot တစ်ခုကို တည်ဆောက်နည်းကို သင်လေ့လာခဲ့ပါသည်။ `TextBlob` သည် noun phrase extraction ကဲ့သို့သော အခြေခံ NLP လုပ်ငန်းများကို လုပ်ဆောင်ရန် ML ကို နောက်ကွယ်တွင် ပေါင်းစပ်ထားသော library တစ်ခုဖြစ်သည်။ Computational linguistics တွင် အရေးကြီးသော စိန်ခေါ်မှုတစ်ခုမှာ စကားလုံးများကို တစ်ဘာသာမှ တစ်ဘာသာသို့ တိကျစွာ _ဘာသာပြန်ခြင်း_ ဖြစ်သည်။

## [Pre-lecture quiz](https://ff-quizzes.netlify.app/en/ml/)

ဘာသာပြန်ခြင်းသည် အလွန်ခက်ခဲသော ပြဿနာတစ်ခုဖြစ်ပြီး ဘာသာစကားများ အထောင်ပေါင်းများစွာရှိပြီး တစ်ခုနှင့်တစ်ခု၏ သဒ္ဒါစည်းမျဉ်းများက အလွန်ကွဲပြားနိုင်သည်။ တစ်ခုသော နည်းလမ်းမှာ ဘာသာစကားတစ်ခု၏ သဒ္ဒါစည်းမျဉ်းများကို non-language dependent structure သို့ ပြောင်းလဲပြီး ထို structure ကို အခြားဘာသာစကားသို့ ပြန်လည်ပြောင်းလဲခြင်းဖြစ်သည်။ ဤနည်းလမ်းသည် အောက်ပါအဆင့်များကို လိုက်နာရမည်ဖြစ်သည်-

1. **Identification** - input ဘာသာစကားရှိ စကားလုံးများကို noun, verb စသည်ဖြင့် သတ်မှတ်ခြင်း။
2. **Create translation** - target ဘာသာစကား format အတိုင်း တိုက်ရိုက်ဘာသာပြန်ခြင်း။

### ဥပမာ စာကြောင်း - အင်္ဂလိပ်မှ အိုင်းရစ်

အင်္ဂလိပ်ဘာသာစကားတွင် _I feel happy_ ဆိုသော စာကြောင်းသည် စကားလုံးသုံးခုပါရှိပြီး အစီအစဉ်မှာ-

- **subject** (I)
- **verb** (feel)
- **adjective** (happy)

သို့သော် အိုင်းရစ်ဘာသာစကားတွင် ထိုစာကြောင်း၏ သဒ္ဒါစည်းမျဉ်းများက အလွန်ကွဲပြားသည်။ "*happy*" သို့ "*sad*" ကဲ့သို့သော ခံစားချက်များကို *upon* you အဖြစ်ဖော်ပြသည်။

အင်္ဂလိပ်စာကြောင်း `I feel happy` ကို အိုင်းရစ်ဘာသာစကားတွင် `Tá athas orm` ဟု ပြောဆိုသည်။ *literal* translation သည် `Happy is upon me` ဖြစ်သည်။

အိုင်းရစ် speaker တစ်ဦးသည် အင်္ဂလိပ်ဘာသာစကားသို့ ဘာသာပြန်သောအခါ `Happy is upon me` ဟု မပြောပါ၊ `I feel happy` ဟု ပြောပါသည်။ အကြောင်းမှာ စာကြောင်း၏ အဓိပ္ပါယ်ကို နားလည်သောကြောင့်ဖြစ်သည်။ စကားလုံးများနှင့် စာကြောင်းဖွဲ့စည်းမှုက ကွဲပြားနေသော်လည်း အဓိပ္ပါယ်ကို နားလည်နိုင်သည်။

အိုင်းရစ်ဘာသာစကားတွင် စာကြောင်း၏ formal order သည်-

- **verb** (Tá or is)
- **adjective** (athas, or happy)
- **subject** (orm, or upon me)

## ဘာသာပြန်ခြင်း

naive translation program တစ်ခုသည် စာကြောင်းဖွဲ့စည်းမှုကို မထည့်သွင်းဘဲ စကားလုံးများကိုသာ ဘာသာပြန်နိုင်သည်။

✅ သင်အရွယ်ရောက်ပြီးနောက် ဘာသာစကားတစ်ခု (သို့မဟုတ် နှစ်ခု၊ သုံးခု) ကို သင်ယူခဲ့ပါက သင်၏ native ဘာသာစကားတွင် စဉ်းစားပြီး concept ကို စကားလုံးတစ်လုံးချင်းစီ translation ပြုလုပ်ကာ ဒုတိယဘာသာစကားသို့ ပြောဆိုခဲ့ဖူးနိုင်သည်။ ဤနည်းလမ်းသည် naive translation computer programs လုပ်ဆောင်နည်းနှင့် ဆင်တူသည်။ fluency ရရှိရန် ဤအဆင့်ကို ကျော်လွှားရန် အရေးကြီးသည်။

naive translation သည် မကောင်းသော (တစ်ခါတစ်ရံ အလွဲလွဲအချော်ချော်) ဘာသာပြန်မှုများကို ဖြစ်စေသည်- `I feel happy` ကို literal translation ပြုလုပ်ပါက `Mise bhraitheann athas` ဟု အိုင်းရစ်ဘာသာစကားသို့ ပြောင်းလဲသည်။ ၎င်းသည် (literal) `me feel happy` ဟု အဓိပ္ပါယ်ရပြီး အိုင်းရစ်စာကြောင်းတစ်ခုအဖြစ် မမှန်ကန်ပါ။

> [ဤဗီဒီယို](https://www.youtube.com/watch?v=mRIaLSdRMMs) ကဲ့သို့သော အိုင်းရစ်ဘာသာစကား၏ သမိုင်းနှင့်ယဉ်ကျေးမှုအကြောင်းကို ကြည့်ရှုနိုင်ပါသည်။

### Machine learning နည်းလမ်းများ

ယခုအထိ သင်သည် natural language processing အတွက် formal rules နည်းလမ်းကို လေ့လာခဲ့ပါသည်။ အခြားနည်းလမ်းတစ်ခုမှာ စကားလုံးများ၏ အဓိပ္ပါယ်ကို မထည့်သွင်းဘဲ _machine learning ကို အသုံးပြု၍ pattern များကို ရှာဖွေခြင်း_ ဖြစ်သည်။ origin နှင့် target ဘာသာစကားများတွင် text များ (*corpus*) သို့မဟုတ် texts (*corpora*) အများအပြားရှိပါက translation အတွက် ဤနည်းလမ်းကို အသုံးပြုနိုင်သည်။

ဥပမာအားဖြင့် Jane Austen ရေးသားသော 1813 ခုနှစ်ထုတ် *Pride and Prejudice* အင်္ဂလိပ်ဝတ္ထုကို ရှုပါ။ အင်္ဂလိပ်စာအုပ်နှင့် *French* ဘာသာစကားသို့ လူသားဘာသာပြန်မှုကို ကြည့်ရှုပါက phrase များကို _idiomatically_ translation ပြုလုပ်ထားသည်ကို တွေ့နိုင်သည်။ သင်မကြာမီ ဤလုပ်ငန်းကို လုပ်ဆောင်မည်။

ဥပမာအားဖြင့် `I have no money` ဆိုသော အင်္ဂလိပ် phrase ကို literal translation ပြုလုပ်ပါက `Je n'ai pas de monnaie` ဟု French ဘာသာစကားသို့ ပြောင်းလဲနိုင်သည်။ "Monnaie" သည် tricky French 'false cognate' တစ်ခုဖြစ်ပြီး 'money' နှင့် 'monnaie' သည် အဓိပ္ပါယ်တူမဟုတ်ပါ။ လူသားဘာသာပြန်သူတစ်ဦးက `Je n'ai pas d'argent` ဟု ပြောင်းလဲနိုင်သည်။ ၎င်းသည် 'loose change' (monnaie) မဟုတ်ဘဲ 'money' (argent) ကို ပိုမိုတိကျစွာဖော်ပြသည်။

![monnaie](../../../../6-NLP/3-Translation-Sentiment/images/monnaie.png)

> [Jen Looper](https://twitter.com/jenlooper) မှ ရေးသားထားသော ပုံ

ML model တွင် လူသားဘာသာပြန်မှုများလုံလောက်စွာရှိပါက expert human speakers နှစ်ဦး၏ translation pattern များကို ရှာဖွေကာ translation တိကျမှုကို တိုးတက်စေနိုင်သည်။

### လေ့ကျင့်ခန်း - ဘာသာပြန်ခြင်း

`TextBlob` ကို အသုံးပြု၍ စာကြောင်းများကို ဘာသာပြန်နိုင်သည်။ **Pride and Prejudice** ၏ ပထမဆုံးစာကြောင်းကို စမ်းကြည့်ပါ-

```python
from textblob import TextBlob

blob = TextBlob(
    "It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want of a wife!"
)
print(blob.translate(to="fr"))

```

`TextBlob` သည် translation ကို အလွန်ကောင်းစွာ ပြုလုပ်နိုင်သည်- "C'est une vérité universellement reconnue, qu'un homme célibataire en possession d'une bonne fortune doit avoir besoin d'une femme!". 

1932 ခုနှစ် V. Leconte နှင့် Ch. Pressoir မှ French translation နှင့် နှိုင်းယှဉ်ပါက TextBlob ၏ translation သည် original author ၏ စကားလုံးများကို ပိုမိုတိကျစွာ ဖော်ပြထားသည်-

"C'est une vérité universelle qu'un célibataire pourvu d'une belle fortune doit avoir envie de se marier, et, si peu que l'on sache de son sentiment à cet egard, lorsqu'il arrive dans une nouvelle résidence, cette idée est si bien fixée dans l'esprit de ses voisins qu'ils le considèrent sur-le-champ comme la propriété légitime de l'une ou l'autre de leurs filles."

ဤကိစ္စတွင် ML ဖြင့် translation ပြုလုပ်ခြင်းသည် လူသားဘာသာပြန်သူ၏ translation ထက် ပိုမိုကောင်းမွန်သည်။

> TextBlob သည် translation ကို အလွန်ကောင်းစွာ ပြုလုပ်နိုင်ခြင်း၏ အကြောင်းရင်းမှာ ML သုံး Google Translate ကို နောက်ကွယ်တွင် အသုံးပြုထားခြင်းဖြစ်သည်။ ၎င်းသည် AI တစ်ခုဖြစ်ပြီး phrase များကို မီလီယံချီ parse ပြုလုပ်ကာ အကောင်းဆုံး string များကို ခန့်မှန်းနိုင်သည်။ manual လုပ်ငန်းစဉ်မရှိဘဲ internet connection လိုအပ်သည်။

✅ စာကြောင်းများကို ထပ်မံစမ်းကြည့်ပါ။ ML translation နှင့် လူသားဘာသာပြန်မှုတို့တွင် ဘယ်ဟာက ပိုကောင်းသနည်း။ ဘယ်အခြေအနေတွင်ကောင်းသနည်း။

## ခံစားချက်ခွဲခြားခြင်း

Machine learning သည် sentiment analysis တွင်လည်း အလွန်ကောင်းစွာ လုပ်ဆောင်နိုင်သည်။ non-ML နည်းလမ်းမှာ 'positive' နှင့် 'negative' စကားလုံးများကို သတ်မှတ်ကာ sentiment ကိုတွက်ချက်ခြင်းဖြစ်သည်။

ဤနည်းလမ်းသည် `Great, that was a wonderful waste of time, I'm glad we are lost on this dark road` ကဲ့သို့သော sarcastic sentence များကို trick ပြုနိုင်သည်။ simple algorithm သည် 'great', 'wonderful', 'glad' ကို positive ဟု သတ်မှတ်ပြီး 'waste', 'lost', 'dark' ကို negative ဟု သတ်မှတ်သည်။ conflicting words များကြောင့် overall sentiment ကို မမှန်ကန်စေသည်။

✅ လူသား speaker အနေဖြင့် sarcasm ကို ဘယ်လိုဖော်ပြသနည်း စဉ်းစားကြည့်ပါ။ tone inflection သည် အဓိကအခန်းကဏ္ဍ ပါဝင်သည်။ "Well, that film was awesome" ဟု အမျိုးမျိုးသောနည်းလမ်းဖြင့် ပြောကြည့်ပါ။

### ML နည်းလမ်းများ

ML နည်းလမ်းမှာ negative နှင့် positive text များကို manually စုဆောင်းခြင်းဖြစ်သည်- tweets, movie reviews, သို့မဟုတ် opinion နှင့် score ပါဝင်သော text များ။ opinion နှင့် score များကို NLP techniques ဖြင့် ခွဲခြားကာ pattern များကို ရှာဖွေသည်။

> ⚖️ **ဥပမာ**: နိုင်ငံရေးသမား၏ရုံးတွင် ဥပဒေသစ်တစ်ခုကို ဆွေးနွေးနေသည်။ constituents များသည် email များကို support သို့မဟုတ် against အဖြစ် ရေးသားနိုင်သည်။ email များအများကြီးရှိပါက bot တစ်ခုကို အသုံးပြု၍ email များကို ဖတ်စစ်နိုင်မည်။

✅ ယခင်သင်ခန်းစာများတွင် သင်အသုံးပြုခဲ့သော လုပ်ငန်းစဉ်များနှင့် ဆင်တူပါသလား။

## လေ့ကျင့်ခန်း - sentimental စာကြောင်းများ

sentiment ကို -1 မှ 1 အထိ *polarity* ဖြင့် တိုင်းတာသည်။ -1 သည် အဆိုးဆုံး negative sentiment ဖြစ်ပြီး 1 သည် အကောင်းဆုံး positive sentiment ဖြစ်သည်။ sentiment ကို 0 မှ 1 အထိ objectivity (0) နှင့် subjectivity (1) ဖြင့်လည်း တိုင်းတာသည်။

Jane Austen ရေးသားသော *Pride and Prejudice* ကို ထပ်မံကြည့်ရှုပါ။ [Project Gutenberg](https://www.gutenberg.org/files/1342/1342-h/1342-h.htm) တွင် text ရရှိနိုင်သည်။ အောက်ပါ program သည် စာအုပ်၏ ပထမဆုံးနှင့် နောက်ဆုံးစာကြောင်းများ၏ sentiment polarity နှင့် subjectivity/objectivity score ကို ခွဲခြားပြသည်။

`TextBlob` library ကို sentiment ကို သတ်မှတ်ရန် အသုံးပြုပါ (သင်၏ sentiment calculator ကို ရေးရန် မလိုအပ်ပါ)။

```python
from textblob import TextBlob

quote1 = """It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want of a wife."""

quote2 = """Darcy, as well as Elizabeth, really loved them; and they were both ever sensible of the warmest gratitude towards the persons who, by bringing her into Derbyshire, had been the means of uniting them."""

sentiment1 = TextBlob(quote1).sentiment
sentiment2 = TextBlob(quote2).sentiment

print(quote1 + " has a sentiment of " + str(sentiment1))
print(quote2 + " has a sentiment of " + str(sentiment2))
```

output အဖြစ်-

```output
It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want # of a wife. has a sentiment of Sentiment(polarity=0.20952380952380953, subjectivity=0.27142857142857146)

Darcy, as well as Elizabeth, really loved them; and they were
     both ever sensible of the warmest gratitude towards the persons
      who, by bringing her into Derbyshire, had been the means of
      uniting them. has a sentiment of Sentiment(polarity=0.7, subjectivity=0.8)
```

## စိန်ခေါ်မှု - sentiment polarity ကို စစ်ဆေးပါ

သင်၏ task သည် *Pride and Prejudice* တွင် absolutely positive sentences များသည် absolutely negative sentences များထက် ပိုများသလားဆိုသည်ကို sentiment polarity ဖြင့် သတ်မှတ်ခြင်းဖြစ်သည်။ polarity score 1 သို့မဟုတ် -1 ကို absolutely positive သို့မဟုတ် negative ဟု သတ်မှတ်နိုင်သည်။

**အဆင့်များ**:

1. [Pride and Prejudice](https://www.gutenberg.org/files/1342/1342-h/1342-h.htm) ကို .txt file အဖြစ် download ပြုလုပ်ပါ။ metadata များကို ဖယ်ရှားပြီး original text ကိုသာထားပါ။
2. Python တွင် file ကို ဖွင့်ကာ string အဖြစ် extract ပြုလုပ်ပါ။
3. book string ကို TextBlob ဖြင့် ဖန်တီးပါ။
4. စာအုပ်ရှိ စာကြောင်းတစ်ခုချင်းစီကို loop ဖြင့် ခွဲခြားပါ။
   1. polarity သည် 1 သို့မဟုတ် -1 ဖြစ်ပါက positive သို့မဟုတ် negative messages များကို array သို့မဟုတ် list တွင် သိမ်းဆည်းပါ။
5. နောက်ဆုံးတွင် positive sentences နှင့် negative sentences (သီးသန့်) နှင့် ၎င်းတို့၏ အရေအတွက်ကို print ပြပါ။

[solution](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/3-Translation-Sentiment/solution/notebook.ipynb) ကို ကြည့်ရှုနိုင်သည်။

✅ Knowledge Check

1. sentiment သည် စာကြောင်းတွင် အသုံးပြုထားသော စကားလုံးများအပေါ် အခြေခံသည်။ code သည် စကားလုံးများကို *နားလည်* သလား။
2. sentiment polarity သည် တိကျသလား၊ သို့မဟုတ် score များနှင့် သင် *သဘောတူ* သလား။
   1. အထူးသဖြင့် အောက်ပါ absolute **positive** polarity စာကြောင်းများနှင့် သဘောတူသလား-
      * “What an excellent father you have, girls!” said she, when the door was shut.
      * “Your examination of Mr. Darcy is over, I presume,” said Miss Bingley; “and pray what is the result?” “I am perfectly convinced by it that Mr. Darcy has no defect.
      * How wonderfully these sort of things occur!
      * I have the greatest dislike in the world to that sort of thing.
      * Charlotte is an excellent manager, I dare say.
      * “This is delightful indeed!
      * I am so happy!
      * Your idea of the ponies is delightful.
   2. အောက်ပါ absolute positive sentiment စာကြောင်းများသည် positive မဟုတ်သော်လည်း sentiment analysis သည် positive ဟု သတ်မှတ်ခဲ့သည်။ အဘယ်ကြောင့် positive ဟု သတ်မှတ်ခဲ့သနည်း-
      * Happy shall I be, when his stay at Netherfield is over!” “I wish I could say anything to comfort you,” replied Elizabeth; “but it is wholly out of my power.
      * If I could but see you as happy!
      * Our distress, my dear Lizzy, is very great.
   3. အောက်ပါ absolute **negative** polarity စာကြောင်းများနှင့် သဘောတူသလား-
      - Everybody is disgusted with his pride.
      - “I should like to know how he behaves among strangers.” “You shall hear then—but prepare yourself for something very dreadful.
      - The pause was to Elizabeth’s feelings dreadful.
      - It would be dreadful!

✅ Jane Austen ၏ ဝတ္ထုများကို နားလည်သူများသည် Regency England ၏ ရိုးရာများကို critique ပြုလုပ်ထားသည်ကို သိရှိနိုင်သည်။ Elizabeth Bennett သည် *Pride and Prejudice* ၏ အဓိကဇာတ်ကောင်ဖြစ်ပြီး ၎င်း၏ စကားလုံးများသည် nuance အများအပြားပါဝင်သည်။ Mr. Darcy သည် Elizabeth ၏ playful language ကိုမှတ်ချက်ပြုထားသည်- "I have had the pleasure of your acquaintance long enough to know that you find great enjoyment in occasionally professing opinions which in fact are not your own."

---

## 🚀စိန်ခေါ်မှု

Marvin ကို user input မှ feature များကို ထုတ်ယူခြင်းဖြင့် ပိုမိုကောင်းမွန်အောင် ပြုလုပ်နိုင်ပါသလား။

## [Post-lecture quiz](https://ff-quizzes.netlify.app/en/ml/)

## Review & Self Study
စာသားမှခံစားချက်ကိုထုတ်ယူရန်နည်းလမ်းများစွာရှိသည်။ ဒီနည်းလမ်းကိုအသုံးပြုနိုင်မည့်လုပ်ငန်းဆိုင်ရာအက်ပလီကေးရှင်းများကိုစဉ်းစားပါ။ ဒါဟာဘယ်လိုမှားယွင်းနိုင်တယ်ဆိုတာကိုလည်းစဉ်းစားပါ။ ခံစားချက်ကိုခွဲခြားစစ်ဆေးပေးသော [Azure Text Analysis](https://docs.microsoft.com/azure/cognitive-services/Text-Analytics/how-tos/text-analytics-how-to-sentiment-analysis?tabs=version-3-1?WT.mc_id=academic-77952-leestott) ကဲ့သို့သောစီးပွားရေးလုပ်ငန်းများအတွက်အသင့်ဖြစ်သောစနစ်များအကြောင်းပိုမိုဖတ်ရှုပါ။ အထက်တွင်ဖော်ပြထားသော Pride and Prejudice စာကြောင်းများကိုစမ်းသပ်ပြီး၊ အနုနုကျကျခံစားချက်များကိုသိနိုင်မလားစစ်ဆေးပါ။

## လုပ်ငန်းတာဝန်

[Poetic license](assignment.md)

---

**ဝက်ဘ်ဆိုက်မှတ်ချက်**:  
ဤစာရွက်စာတမ်းကို AI ဘာသာပြန်ဝန်ဆောင်မှု [Co-op Translator](https://github.com/Azure/co-op-translator) ကို အသုံးပြု၍ ဘာသာပြန်ထားပါသည်။ ကျွန်ုပ်တို့သည် တိကျမှန်ကန်မှုအတွက် ကြိုးစားနေပါသော်လည်း၊ အလိုအလျောက်ဘာသာပြန်ဆိုမှုများတွင် အမှားများ သို့မဟုတ် မတိကျမှုများ ပါဝင်နိုင်သည်ကို ကျေးဇူးပြု၍ သတိပြုပါ။ မူရင်းစာရွက်စာတမ်းကို ၎င်း၏ မူလဘာသာစကားဖြင့် အာဏာတည်သောရင်းမြစ်အဖြစ် သတ်မှတ်သင့်ပါသည်။ အရေးကြီးသော အချက်အလက်များအတွက် လူသားပညာရှင်များမှ ဘာသာပြန်ဆိုမှုကို အကြံပြုပါသည်။ ဤဘာသာပြန်ဆိုမှုကို အသုံးပြုခြင်းမှ ဖြစ်ပေါ်လာသော နားလည်မှုမှားများ သို့မဟုတ် အဓိပ္ပါယ်မှားများအတွက် ကျွန်ုပ်တို့သည် တာဝန်မယူပါ။