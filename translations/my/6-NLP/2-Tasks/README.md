<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "5f3cb462e3122e1afe7ab0050ccf2bd3",
  "translation_date": "2025-09-05T13:56:24+00:00",
  "source_file": "6-NLP/2-Tasks/README.md",
  "language_code": "my"
}
-->
# သဘာဝဘာသာစကားကို အလုပ်လုပ်စေခြင်းဆိုင်ရာ အလုပ်များနှင့် နည်းလမ်းများ

*သဘာဝဘာသာစကားကို အလုပ်လုပ်စေခြင်း* အလုပ်များအတွက်၊ အလုပ်လုပ်စေလိုသော စာသားကို အစိတ်အပိုင်းများ ခွဲခြားပြီး၊ စစ်ဆေးကာ၊ ရလဒ်များကို စည်းမျဉ်းများနှင့် ဒေတာအစုများနှင့် တွဲဆက်ထားရမည်။ အလုပ်များသည်၊ အရေးအသား၏ *အဓိပ္ပါယ်*၊ *ရည်ရွယ်ချက်* သို့မဟုတ် *စကားလုံးများ၏ ထပ်တလဲလဲဖြစ်မှု* ကို ရယူရန် အခွင့်အရေးပေးသည်။

## [Pre-lecture quiz](https://ff-quizzes.netlify.app/en/ml/)

စာသားကို အလုပ်လုပ်စေရာတွင် အသုံးပြုသော နည်းလမ်းများကို ရှာဖွေကြမည်။ စက်ရုပ်သင်ယူမှုနှင့် ပေါင်းစပ်ပြီး၊ ဤနည်းလမ်းများသည် စာသားအများအပြားကို ထိရောက်စွာ ခွဲခြားစစ်ဆေးရန် ကူညီပေးသည်။ သို့သော် စက်ရုပ်သင်ယူမှုကို ဤအလုပ်များတွင် အသုံးပြုမီ၊ သဘာဝဘာသာစကားကို အလုပ်လုပ်စေသူများ ရင်ဆိုင်ရသော ပြဿနာများကို နားလည်ရမည်။

## သဘာဝဘာသာစကားကို အလုပ်လုပ်စေခြင်းဆိုင်ရာ အလုပ်များ

သင်လုပ်ဆောင်လိုသော စာသားကို ခွဲခြားစစ်ဆေးရန် နည်းလမ်းများ မျိုးစုံရှိသည်။ သင်လုပ်ဆောင်နိုင်သော အလုပ်များရှိပြီး၊ ဤအလုပ်များမှတစ်ဆင့် စာသားကို နားလည်နိုင်ပြီး သုံးသပ်ချက်များ ဆွဲယူနိုင်သည်။ အလုပ်များကို အဆင့်ဆင့် လုပ်ဆောင်လေ့ရှိသည်။

### Tokenization

NLP အယ်လဂိုရစ်သမားများအတွက် ပထမဆုံးလုပ်ဆောင်ရမည့်အရာမှာ စာသားကို token သို့မဟုတ် စကားလုံးများအဖြစ် ခွဲခြားခြင်းဖြစ်သည်။ ဤအရာသည် ရိုးရှင်းသလို ထင်ရပေမယ့်၊ စာကြောင်းအတွင်းရှိ သတ်မှတ်ချက်များနှင့် ဘာသာစကားအမျိုးမျိုး၏ စကားလုံးနှင့် စာကြောင်းအဆုံးသတ်ချက်များကို ထည့်သွင်းစဉ်းစားရခြင်းကြောင့် ခက်ခဲနိုင်သည်။ သတ်မှတ်ချက်များကို ဆုံးဖြတ်ရန် နည်းလမ်းများ မျိုးစုံကို အသုံးပြုရနိုင်သည်။

![tokenization](../../../../6-NLP/2-Tasks/images/tokenization.png)
> **Pride and Prejudice** စာအုပ်မှ စာကြောင်းတစ်ကြောင်းကို token ခွဲခြင်း။ [Jen Looper](https://twitter.com/jenlooper) မှ Infographic

### Embeddings

[Word embeddings](https://wikipedia.org/wiki/Word_embedding) သည် စာသားဒေတာကို ကိန်းဂဏန်းအဖြစ် ပြောင်းလဲရန် နည်းလမ်းတစ်ခုဖြစ်သည်။ Embeddings ကို အဓိပ္ပါယ်တူသော စကားလုံးများ သို့မဟုတ် အတူတူအသုံးပြုသော စကားလုံးများကို အစုအဖွဲ့တစ်ခုအဖြစ် စုပုံစေသော နည်းလမ်းဖြင့် ပြုလုပ်သည်။

![word embeddings](../../../../6-NLP/2-Tasks/images/embedding.png)
> "I have the highest respect for your nerves, they are my old friends." - **Pride and Prejudice** စာအုပ်မှ စာကြောင်းတစ်ကြောင်းအတွက် Word embeddings။ [Jen Looper](https://twitter.com/jenlooper) မှ Infographic

✅ [ဤစိတ်ဝင်စားဖွယ် tool](https://projector.tensorflow.org/) ကို အသုံးပြု၍ word embeddings ကို စမ်းသပ်ပါ။ စကားလုံးတစ်လုံးကို နှိပ်ပါက အဓိပ္ပါယ်တူသော စကားလုံးများ၏ အစုအဖွဲ့ကို ပြသသည်။ ဥပမာ - 'toy' သည် 'disney', 'lego', 'playstation', နှင့် 'console' တို့နှင့် အစုအဖွဲ့တူသည်။

### Parsing & Part-of-speech Tagging

Tokenized ဖြစ်ထားသော စကားလုံးတစ်ခုစီကို noun, verb, adjective စသည့် part of speech အဖြစ် tag လုပ်နိုင်သည်။ ဥပမာ - `the quick red fox jumped over the lazy brown dog` စာကြောင်းကို POS tagging လုပ်ပါက fox = noun, jumped = verb ဖြစ်နိုင်သည်။

![parsing](../../../../6-NLP/2-Tasks/images/parse.png)

> **Pride and Prejudice** စာအုပ်မှ စာကြောင်းတစ်ကြောင်းကို Parsing လုပ်ခြင်း။ [Jen Looper](https://twitter.com/jenlooper) မှ Infographic

Parsing သည် စာကြောင်းအတွင်း စကားလုံးများသည် အချင်းချင်း ဆက်စပ်နေမှုကို သိရှိခြင်းဖြစ်သည်။ ဥပမာ - `the quick red fox jumped` သည် adjective-noun-verb အစဉ်ဖြစ်ပြီး၊ `lazy brown dog` အစဉ်နှင့် သီးသန့်ဖြစ်သည်။

### Word and Phrase Frequencies

စာသားအများအပြားကို ခွဲခြားစစ်ဆေးရာတွင် အသုံးဝင်သော နည်းလမ်းတစ်ခုမှာ စိတ်ဝင်စားဖွယ် စကားလုံး သို့မဟုတ် စကားစုများ၏ အကြိမ်ရေကို စာရင်းပြုစုခြင်းဖြစ်သည်။ `the quick red fox jumped over the lazy brown dog` စာကြောင်းတွင် `the` စကားလုံး၏ frequency သည် 2 ဖြစ်သည်။

ဥပမာစာသားတစ်ခုကို ကြည့်ပြီး စကားလုံးများ၏ frequency ကို ရေတွက်ကြည့်ပါ။ Rudyard Kipling ရေးသားသော The Winners ကဗျာတွင် အောက်ပါ စာပိုဒ်ပါဝင်သည် -

```output
What the moral? Who rides may read.
When the night is thick and the tracks are blind
A friend at a pinch is a friend, indeed,
But a fool to wait for the laggard behind.
Down to Gehenna or up to the Throne,
He travels the fastest who travels alone.
```

Phrase frequencies သည် case sensitive သို့မဟုတ် case insensitive ဖြစ်နိုင်သည်။ ဥပမာ - `a friend` phrase ၏ frequency သည် 2 ဖြစ်ပြီး၊ `the` phrase ၏ frequency သည် 6 ဖြစ်သည်။ `travels` phrase ၏ frequency သည် 2 ဖြစ်သည်။

### N-grams

စာသားကို စကားလုံးအရေအတွက် သတ်မှတ်ထားသော အစဉ်အတိုင်း ခွဲခြားနိုင်သည်။ တစ်လုံး (unigram), နှစ်လုံး (bigram), သုံးလုံး (trigram) သို့မဟုတ် စကားလုံးအရေအတွက် မည်သည့်အတိုင်း (n-grams) ဖြစ်နိုင်သည်။

ဥပမာ - `the quick red fox jumped over the lazy brown dog` ကို n-gram score 2 ဖြင့် ခွဲခြားပါက အောက်ပါ n-grams များရရှိသည် -

1. the quick  
2. quick red  
3. red fox  
4. fox jumped  
5. jumped over  
6. over the  
7. the lazy  
8. lazy brown  
9. brown dog  

ဤအရာကို sliding box အဖြစ် စဉ်းစားပါက ပိုမိုလွယ်ကူနိုင်သည်။ ဤသည်မှာ 3 စကားလုံး n-grams အတွက် sliding box ဖြစ်သည် -

1.   <u>**the quick red**</u> fox jumped over the lazy brown dog  
2.   the **<u>quick red fox</u>** jumped over the lazy brown dog  
3.   the quick **<u>red fox jumped</u>** over the lazy brown dog  
4.   the quick red **<u>fox jumped over</u>** the lazy brown dog  
5.   the quick red fox **<u>jumped over the</u>** lazy brown dog  
6.   the quick red fox jumped **<u>over the lazy</u>** brown dog  
7.   the quick red fox jumped over <u>**the lazy brown**</u> dog  
8.   the quick red fox jumped over the **<u>lazy brown dog</u>**  

![n-grams sliding window](../../../../6-NLP/2-Tasks/images/n-grams.gif)

> N-gram value of 3: [Jen Looper](https://twitter.com/jenlooper) မှ Infographic

### Noun phrase Extraction

စာကြောင်းအများစုတွင် subject သို့မဟုတ် object ဖြစ်သော noun phrase တစ်ခုရှိသည်။ အင်္ဂလိပ်ဘာသာစကားတွင် `a`, `an`, `the` စသည်ဖြင့် စတင်သော noun phrase ကို ရှာဖွေခြင်းသည် စာကြောင်း၏ အဓိပ္ပါယ်ကို နားလည်ရန် အသုံးဝင်သော NLP အလုပ်တစ်ခုဖြစ်သည်။

✅ "I cannot fix on the hour, or the spot, or the look or the words, which laid the foundation. It is too long ago. I was in the middle before I knew that I had begun." စာကြောင်းတွင် noun phrases ကို ရှာဖွေပါ။

`the quick red fox jumped over the lazy brown dog` စာကြောင်းတွင် noun phrases 2 ခုရှိသည် - **quick red fox** နှင့် **lazy brown dog**။

### Sentiment analysis

စာကြောင်း သို့မဟုတ် စာသားကို *အပျော်* သို့မဟုတ် *အနည်းငယ်စိတ်ဆိုး* ဖြစ်မှုအပေါ် အခြေခံ၍ sentiment ကို ခွဲခြားနိုင်သည်။ Sentiment ကို *polarity* နှင့် *objectivity/subjectivity* ဖြင့် တိုင်းတာသည်။ Polarity သည် -1.0 မှ 1.0 (negative to positive) ဖြစ်ပြီး၊ 0.0 မှ 1.0 (most objective to most subjective) ဖြစ်သည်။

✅ နောက်ပိုင်းတွင် sentiment ကို စက်ရုပ်သင်ယူမှုဖြင့် ဆုံးဖြတ်နိုင်သော နည်းလမ်းများကို သင်လေ့လာမည်။ သို့သော် လူ့အတတ်ပညာရှင်က စိတ်ဝင်စားဖွယ် positive သို့မဟုတ် negative စကားလုံးများနှင့် phrase များကို စာရင်းပြုစုထားသော နည်းလမ်းတစ်ခုကို စဉ်းစားပါ။ ဤနည်းလမ်းသည် အချို့သောအခြေအနေများတွင် အလုပ်လုပ်နိုင်ပြီး အခြားအခြေအနေများတွင် မလုပ်နိုင်ခြင်းကို သင်မြင်နိုင်ပါသလား။

### Inflection

Inflection သည် စကားလုံးတစ်လုံးကို singular သို့မဟုတ် plural အဖြစ် ပြောင်းလဲရန် ခွင့်ပြုသည်။

### Lemmatization

*Lemma* သည် စကားလုံးများအစုအဖွဲ့၏ root သို့မဟုတ် headword ဖြစ်သည်။ ဥပမာ - *flew*, *flies*, *flying* ၏ lemma သည် *fly* ဖြစ်သည်။

NLP သုတေသနသူများအတွက် အသုံးဝင်သော ဒေတာဘေ့စ်များလည်း ရရှိနိုင်သည်၊ အထူးသဖြင့် -

### WordNet

[WordNet](https://wordnet.princeton.edu/) သည် စကားလုံးများ၊ အဓိပ္ပါယ်တူစကားလုံးများ၊ အဓိပ္ပါယ်ဆန်စကားလုံးများနှင့် အခြားသော အသေးစိတ်များကို ဘာသာစကားအမျိုးမျိုးအတွက် စုစည်းထားသော ဒေတာဘေ့စ်ဖြစ်သည်။ ဘာသာပြန်များ၊ စာလုံးပေါင်းစစ်ဆေးသူများ သို့မဟုတ် ဘာသာစကား tools များကို တည်ဆောက်ရာတွင် အလွန်အသုံးဝင်သည်။

## NLP Libraries

ကံကောင်းစွာ၊ ဤနည်းလမ်းများအားလုံးကို ကိုယ်တိုင် တည်ဆောက်ရန် မလိုအပ်ပါ၊ အလွန်ထိရောက်သော Python libraries များ ရရှိနိုင်ပြီး၊ သဘာဝဘာသာစကားကို အလုပ်လုပ်စေခြင်း သို့မဟုတ် စက်ရုပ်သင်ယူမှုတွင် အထူးကျွမ်းကျင်မဟုတ်သော developer များအတွက် ပိုမိုလွယ်ကူစေသည်။ နောက်ပိုင်းသင်ခန်းစာများတွင် ဤ libraries များ၏ နမူနာများကို ပိုမိုလေ့လာမည်၊ သို့သော် အခုမှာ သင်၏ အလုပ်ကို ကူညီနိုင်သော အသုံးဝင်သော နမူနာများကို လေ့လာပါ။

### Exercise - `TextBlob` library ကို အသုံးပြုခြင်း

TextBlob ဟုခေါ်သော library ကို အသုံးပြုကြည့်ပါ၊ ဤ library တွင် ဤအမျိုးအစားအလုပ်များကို လုပ်ဆောင်ရန် API များပါဝင်သည်။ TextBlob သည် "[NLTK](https://nltk.org) နှင့် [pattern](https://github.com/clips/pattern) ၏ အကြီးမားသော အခြေခံအရပ်များပေါ်တွင် ရပ်တည်ပြီး၊ နှစ်ခုစလုံးနှင့် သင့်တော်စွာ အလုပ်လုပ်သည်။" ၎င်း၏ API တွင် ML အများအပြား ပါဝင်သည်။

> Note: [Quick Start](https://textblob.readthedocs.io/en/dev/quickstart.html#quickstart) လမ်းညွှန်သည် အတွေ့အကြုံရှိ Python developer များအတွက် အကြံပြုထားသည်။

*Noun phrases* ကို ရှာဖွေရာတွင် TextBlob သည် extractors များစွာကို ပေးထားသည်။

1. `ConllExtractor` ကို ကြည့်ပါ။

    ```python
    from textblob import TextBlob
    from textblob.np_extractors import ConllExtractor
    # import and create a Conll extractor to use later 
    extractor = ConllExtractor()
    
    # later when you need a noun phrase extractor:
    user_input = input("> ")
    user_input_blob = TextBlob(user_input, np_extractor=extractor)  # note non-default extractor specified
    np = user_input_blob.noun_phrases                                    
    ```

    > ဤနေရာတွင် ဘာဖြစ်နေသနည်း? [ConllExtractor](https://textblob.readthedocs.io/en/dev/api_reference.html?highlight=Conll#textblob.en.np_extractors.ConllExtractor) သည် "ConLL-2000 training corpus ဖြင့် လေ့ကျင့်ထားသော chunk parsing ကို အသုံးပြုသော noun phrase extractor" ဖြစ်သည်။ ConLL-2000 သည် 2000 ခုနှစ် Computational Natural Language Learning Conference ကို ရည်ညွှန်းသည်။ 2000 ခုနှစ်တွင် noun chunking ကို လေ့လာရန် model တစ်ခုကို Wall Street Journal မှ training data (211727 tokens) နှင့် test data (47377 tokens) ဖြင့် လေ့ကျင့်ခဲ့သည်။ [Procedures](https://www.clips.uantwerpen.be/conll2000/chunking/) နှင့် [results](https://ifarm.nl/erikt/research/np-chunking.html) ကို ကြည့်နိုင်သည်။

### Challenge - သင့် bot ကို NLP ဖြင့် တိုးတက်စေခြင်း

ယခင်သင်ခန်းစာတွင် သင်အလွန်ရိုးရှင်းသော Q&A bot တစ်ခုကို တည်ဆောက်ခဲ့သည်။ အခုမှာ Marvin ကို သင့် input ကို sentiment အပေါ်အခြေခံ၍ သုံးသပ်ပြီး၊ sentiment နှင့် ကိုက်ညီသော အဖြေကို ပေးနိုင်စေရန် တိုးတက်စေပါမည်။ သင် noun phrase ကို ရှာဖွေပြီး၊ pluralize လုပ်ကာ အဲဒီအကြောင်းအရာအပေါ် input ပေးရန် မေးမြန်းရပါမည်။

သင့် bot ကို ပိုမိုကောင်းမွန်စေရန် လုပ်ဆောင်ရမည့် အဆင့်များ -

1. အသုံးပြုသူကို bot နှင့် အပြန်အလှန် ဆက်သွယ်ရန် လမ်းညွှန်ချက်များကို print လုပ်ပါ  
2. loop ကို စတင်ပါ  
   1. အသုံးပြုသူ input ကို လက်ခံပါ  
   2. အသုံးပြုသူသည် exit မေးမြန်းပါက ထွက်ပါ  
   3. အသုံးပြုသူ input ကို စစ်ဆေးပြီး sentiment response ကို ဆုံးဖြတ်ပါ  
   4. sentiment တွင် noun phrase ရှိပါက pluralize လုပ်ပြီး အဲဒီအကြောင်းအရာအပေါ် input ပေးရန် မေးမြန်းပါ  
   5. အဖြေကို print လုပ်ပါ  
3. အဆင့် 2 သို့ ပြန်သွားပါ  

TextBlob ကို အသုံးပြု၍ sentiment ကို ဆုံးဖြတ်ရန် code snippet ကို ကြည့်ပါ။ Sentiment response ၏ *gradients* သာ 4 ခုရှိသည် (သင်လိုချင်ပါက ပိုမိုများစွာ ထည့်နိုင်သည်) -

```python
if user_input_blob.polarity <= -0.5:
  response = "Oh dear, that sounds bad. "
elif user_input_blob.polarity <= 0:
  response = "Hmm, that's not great. "
elif user_input_blob.polarity <= 0.5:
  response = "Well, that sounds positive. "
elif user_input_blob.polarity <= 1:
  response = "Wow, that sounds great. "
```

ဤနမူနာ output ကို လမ်းညွှန်အဖြစ် အသုံးပြုပါ (အသုံးပြုသူ input သည် > ဖြင့် စတင်သော အကြောင်းအရာများတွင် ရှိသည်) -

```output
Hello, I am Marvin, the friendly robot.
You can end this conversation at any time by typing 'bye'
After typing each answer, press 'enter'
How are you today?
> I am ok
Well, that sounds positive. Can you tell me more?
> I went for a walk and saw a lovely cat
Well, that sounds positive. Can you tell me more about lovely cats?
> cats are the best. But I also have a cool dog
Wow, that sounds great. Can you tell me more about cool dogs?
> I have an old hounddog but he is sick
Hmm, that's not great. Can you tell me more about old hounddogs?
> bye
It was nice talking to you, goodbye!
```

ဤအလုပ်ကို ပြုလုပ်ရန် အဖြေတစ်ခုကို [ဒီမှာ](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/2-Tasks/solution/bot.py) ကြည့်နိုင်သည်။

✅ Knowledge Check

1. Sympathetic responses သည် bot ကို အသုံးပြုသူက အမှန်တကယ် နားလည်နေသည်ဟု ထင်ရစေမည်လား?  
2. Noun phrase ကို ရှာဖွေခြင်းသည် bot ကို ပိုမို 'ယုံကြည်စရာကောင်း' စေမည်လား?  
3. စာကြောင်းမှ 'noun phrase' ကို ရှာဖွေခြင်းသည် ဘာကြောင့် အသုံးဝင်သနည်း?  

---

ယခင် knowledge check တွင် bot ကို တည်ဆောက်ပြီး၊ သူငယ်ချင်းတစ်ဦးကို စမ်းသပ်ပါ။ Bot သည် သူတို့ကို လှည့်စားနိုင်ပါသလား? Bot ကို ပိုမို 'ယုံကြည်စရာကောင်း'

---

**ဝက်ဘ်ဆိုက်မှတ်ချက်**:  
ဒီစာရွက်စာတမ်းကို AI ဘာသာပြန်ဝန်ဆောင်မှု [Co-op Translator](https://github.com/Azure/co-op-translator) ကို အသုံးပြု၍ ဘာသာပြန်ထားပါသည်။ ကျွန်ုပ်တို့သည် တိကျမှန်ကန်မှုအတွက် ကြိုးစားနေပါသော်လည်း၊ အလိုအလျောက်ဘာသာပြန်မှုများတွင် အမှားများ သို့မဟုတ် မမှန်ကန်မှုများ ပါဝင်နိုင်သည်ကို သတိပြုပါ။ မူလဘာသာစကားဖြင့် ရေးသားထားသော စာရွက်စာတမ်းကို အာဏာတည်သော ရင်းမြစ်အဖြစ် သတ်မှတ်သင့်ပါသည်။ အရေးကြီးသော အချက်အလက်များအတွက် လူ့ဘာသာပြန်ပညာရှင်များကို အကြံပြုပါသည်။ ဒီဘာသာပြန်မှုကို အသုံးပြုခြင်းမှ ဖြစ်ပေါ်လာသော နားလည်မှုမှားမှုများ သို့မဟုတ် အဓိပ္ပာယ်မှားမှုများအတွက် ကျွန်ုပ်တို့သည် တာဝန်မယူပါ။