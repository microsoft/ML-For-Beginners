<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "83320d6b6994909e35d830cebf214039",
  "translation_date": "2025-09-05T12:27:38+00:00",
  "source_file": "9-Real-World/1-Applications/README.md",
  "language_code": "my"
}
-->
# Postscript: အမှန်တကယ်သောကမ္ဘာတွင် Machine Learning

![အမှန်တကယ်သောကမ္ဘာတွင် Machine Learning အကျဉ်းချုပ်ကို Sketchnote အနေနဲ့](../../../../sketchnotes/ml-realworld.png)
> Sketchnote by [Tomomi Imura](https://www.twitter.com/girlie_mac)

ဒီသင်ခန်းစာတွဲမှာ သင်သည် training အတွက် data ကိုပြင်ဆင်နည်းများနှင့် machine learning models ဖန်တီးနည်းများကိုလေ့လာခဲ့ပါသည်။ သင်သည် regression, clustering, classification, natural language processing, နှင့် time series models များကိုတစ်စဉ်တစ်စဉ်တည်ဆောက်ခဲ့ပါသည်။ ဂုဏ်ယူပါတယ်! အခုတော့ သင်သည် "ဒါတွေဘာအတွက်လဲ..." "ဒီ models တွေကို အမှန်တကယ်ဘယ်လိုအသုံးချနိုင်မလဲ" ဆိုပြီး စဉ်းစားနေရနိုင်ပါတယ်။

AI သည် deep learning ကိုအခြေခံပြီး စက်မှုလုပ်ငန်းများတွင် အလွန်စိတ်ဝင်စားမှုရရှိထားသော်လည်း classical machine learning models များအတွက်လည်း အရေးပါသောအသုံးချမှုများရှိနေဆဲဖြစ်သည်။ သင်သည် ယနေ့တိုင်အောင် ဒီအသုံးချမှုများကိုတချို့အသုံးပြုနေတတ်ပါသည်။ ဒီသင်ခန်းစာမှာ သင်သည် အခြားသောလုပ်ငန်းများနှင့် အထူးကျွမ်းကျင်မှုရှိသောနယ်ပယ် ၈ ခုက ဒီ models များကို application များကိုပိုမိုထိရောက်စေခြင်း၊ ယုံကြည်စိတ်ချစေခြင်း၊ ဉာဏ်ရည်ရှိစေခြင်း၊ နှင့် အသုံးပြုသူများအတွက်တန်ဖိုးရှိစေခြင်းအတွက် ဘယ်လိုအသုံးချကြောင်းကိုလေ့လာပါမည်။

## [Pre-lecture quiz](https://ff-quizzes.netlify.app/en/ml/)

## 💰 ဘဏ္ဍာရေး

ဘဏ္ဍာရေးကဏ္ဍသည် machine learning အတွက် အခွင့်အလမ်းများစွာပေးနိုင်သည်။ ဒီကဏ္ဍရှိပြဿနာများစွာသည် ML ကိုအသုံးပြု၍ မော်ဒယ်တည်ဆောက်ခြင်းနှင့် ဖြေရှင်းခြင်းအတွက် သင့်လျော်သည်။

### ခရက်ဒစ်ကတ်လိမ်လည်မှုရှာဖွေခြင်း

ကျွန်ုပ်တို့သည် [k-means clustering](../../5-Clustering/2-K-Means/README.md) ကို သင်ခန်းစာတွင်လေ့လာခဲ့ပါသည်၊ ဒါပေမယ့် ဒါကို ခရက်ဒစ်ကတ်လိမ်လည်မှုဆိုင်ရာပြဿနာများကို ဘယ်လိုဖြေရှင်းနိုင်မလဲ?

K-means clustering သည် **outlier detection** ဟုခေါ်သော ခရက်ဒစ်ကတ်လိမ်လည်မှုရှာဖွေခြင်းနည်းလမ်းတွင် အလွန်အသုံးဝင်သည်။ Outliers သည် data set တစ်ခုအပေါ်ရှိ observation များတွင် အထူးပြောင်းလဲမှုများဖြစ်ပြီး ခရက်ဒစ်ကတ်ကို သာမန်အသုံးပြုမှုဖြစ်မဖြစ်၊ သို့မဟုတ် ထူးခြားသောအရာတစ်ခုဖြစ်နေမဖြစ်ကို ပြောပြနိုင်သည်။ အောက်ပါစာတမ်းတွင် ဖော်ပြထားသည့်အတိုင်း သင်သည် k-means clustering algorithm ကိုအသုံးပြု၍ ခရက်ဒစ်ကတ် data ကို စီစစ်နိုင်ပြီး transaction တစ်ခုစီကို outlier ဖြစ်ပုံကိုအခြေခံ၍ cluster တစ်ခုသို့ သတ်မှတ်နိုင်သည်။ ထို့နောက် သင်သည် fraudulent versus legitimate transactions အတွက် အန္တရာယ်များသော clusters များကို အကဲဖြတ်နိုင်သည်။
[Reference](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.680.1195&rep=rep1&type=pdf)

### Wealth management

Wealth management တွင် တစ်ဦးတစ်ယောက် သို့မဟုတ် ကုမ္ပဏီတစ်ခုသည် သူတို့၏ client များအတွက် ရင်းနှီးမြှုပ်နှံမှုများကို စီမံခန့်ခွဲသည်။ သူတို့၏အလုပ်သည် ရေရှည်တွင် ငွေကြေးကို ထိန်းသိမ်းပြီး တိုးတက်စေခြင်းဖြစ်သည်၊ ထို့ကြောင့် အကောင်းဆုံးလုပ်ဆောင်သော ရင်းနှီးမြှုပ်နှံမှုများကို ရွေးချယ်ရန် အရေးကြီးသည်။

ရင်းနှီးမြှုပ်နှံမှုတစ်ခုသည် ဘယ်လိုလုပ်ဆောင်သလဲဆိုတာကို အကဲဖြတ်ရန် statistical regression သည် အလွန်တန်ဖိုးရှိသော tools ဖြစ်သည်။ [Linear regression](../../2-Regression/1-Tools/README.md) သည် fund တစ်ခုသည် benchmark တစ်ခုနှင့် ဆက်စပ်မှုကို နားလည်ရန် အရေးပါသော tools ဖြစ်သည်။ Regression ရလဒ်များသည် statistically significant ဖြစ်မဖြစ်၊ သို့မဟုတ် client ၏ ရင်းနှီးမြှုပ်နှံမှုများကို ဘယ်လောက်ထိခိုက်စေမည်ကိုလည်း သုံးသပ်နိုင်သည်။ သင်သည် multiple regression ကို အသုံးပြု၍ အခြားသော risk factors များကိုပါ ထည့်သွင်းပြီး analysis ကို တိုးချဲ့နိုင်သည်။ Fund တစ်ခုအတွက် ဒီနည်းလမ်းက ဘယ်လိုအလုပ်လုပ်မည်ဆိုတာကို အောက်ပါစာတမ်းတွင် ကြည့်ရှုနိုင်ပါသည်။
[Reference](http://www.brightwoodventures.com/evaluating-fund-performance-using-regression/)

## 🎓 ပညာရေး

ပညာရေးကဏ္ဍသည် ML ကို အသုံးချနိုင်သော စိတ်ဝင်စားဖွယ်ရာနယ်ပယ်တစ်ခုဖြစ်သည်။ စမ်းသပ်မှုများ သို့မဟုတ် စာတမ်းများတွင် လိမ်လည်မှုကို ရှာဖွေခြင်း၊ correction process တွင် bias (မတူညီမှု) ကို စီမံခန့်ခွဲခြင်းစသည်တို့က စိတ်ဝင်စားဖွယ်ရာပြဿနာများဖြစ်သည်။

### ကျောင်းသားအပြုအမူကိုခန့်မှန်းခြင်း

[Coursera](https://coursera.com) သည် online open course provider တစ်ခုဖြစ်ပြီး သူတို့ engineering ဆုံးဖြတ်ချက်များစွာကို ဆွေးနွေးသော tech blog တစ်ခုရှိသည်။ ဒီ case study တွင် သူတို့သည် regression line တစ်ခုကို plot လုပ်ပြီး low NPS (Net Promoter Score) rating နှင့် course retention သို့မဟုတ် drop-off အကြား correlation ရှိမရှိကို ရှာဖွေခဲ့သည်။
[Reference](https://medium.com/coursera-engineering/controlled-regression-quantifying-the-impact-of-course-quality-on-learner-retention-31f956bd592a)

### Bias ကိုလျှော့ချခြင်း

[Grammarly](https://grammarly.com) သည် spelling နှင့် grammar အမှားများကို စစ်ဆေးပေးသော writing assistant တစ်ခုဖြစ်ပြီး သူတို့၏ product များတွင် sophisticated [natural language processing systems](../../6-NLP/README.md) များကို အသုံးပြုသည်။ သူတို့ tech blog တွင် gender bias ကို machine learning တွင် ဘယ်လိုကိုင်တွယ်ခဲ့သည်ဆိုတာကို case study အနေနဲ့ ဖော်ပြထားသည်။ သင်သည် [introductory fairness lesson](../../1-Introduction/3-fairness/README.md) တွင်လည်း ဒီအကြောင်းကိုလေ့လာခဲ့ပါသည်။
[Reference](https://www.grammarly.com/blog/engineering/mitigating-gender-bias-in-autocorrect/)

## 👜 လက်လီရောင်းဝယ်ရေး

လက်လီရောင်းဝယ်ရေးကဏ္ဍသည် ML ကို အသုံးပြု၍ customer journey ကိုပိုမိုကောင်းမွန်စေခြင်း၊ inventory ကို အကောင်းဆုံးစီမံခန့်ခွဲခြင်းစသည်တို့တွင် အကျိုးရှိစေသည်။

### Customer journey ကို personalize လုပ်ခြင်း

Wayfair သည် furniture ကဲ့သို့သော home goods များကိုရောင်းချသောကုမ္ပဏီတစ်ခုဖြစ်ပြီး customer များအတွက် taste နှင့်လိုအပ်ချက်များကိုဖြည့်ဆည်းပေးရန် အရေးကြီးသည်။ ဒီ article တွင် Wayfair ၏ engineers များက ML နှင့် NLP ကို "customer များအတွက် အမှန်တကယ်သောရလဒ်များကို surface လုပ်ရန်" ဘယ်လိုအသုံးပြုကြောင်းကို ဖော်ပြထားသည်။ အထူးသဖြင့် သူတို့၏ Query Intent Engine သည် entity extraction, classifier training, asset နှင့် opinion extraction, နှင့် sentiment tagging ကို customer reviews တွင် အသုံးပြုထားသည်။ ဒါဟာ online retail တွင် NLP ဘယ်လိုအလုပ်လုပ်တတ်သလဲဆိုတာကို classic use case တစ်ခုဖြစ်သည်။
[Reference](https://www.aboutwayfair.com/tech-innovation/how-we-use-machine-learning-and-natural-language-processing-to-empower-search)

### Inventory management

[StitchFix](https://stitchfix.com) ကဲ့သို့သော innovative, nimble ကုမ္ပဏီများသည် ML ကို recommendation နှင့် inventory management အတွက် အလွန်အကျိုးရှိစွာအသုံးပြုသည်။ သူတို့၏ styling teams များသည် merchandising teams များနှင့်ပေါင်းစည်းလုပ်ဆောင်ကြသည်။ "ကျွန်ုပ်တို့၏ data scientist တစ်ဦးသည် genetic algorithm တစ်ခုကို apparel တွင် အသုံးပြု၍ ယနေ့မရှိသေးသော successful piece of clothing ကိုခန့်မှန်းခဲ့သည်။"
[Reference](https://www.zdnet.com/article/how-stitch-fix-uses-machine-learning-to-master-the-science-of-styling/)

## 🏥 ကျန်းမာရေး

ကျန်းမာရေးကဏ္ဍသည် ML ကို အသုံးပြု၍ သုတေသနလုပ်ငန်းများနှင့် logistic ပြဿနာများကို optimize လုပ်နိုင်သည်။ ဥပမာ - လူနာများကိုပြန်လည်လက်ခံခြင်း၊ သို့မဟုတ် ရောဂါများပျံ့နှံ့မှုကိုတားဆီးခြင်း။

### Clinical trials စီမံခန့်ခွဲခြင်း

Clinical trials တွင် toxicity သည် drug makers များအတွက် အရေးကြီးသောပြဿနာတစ်ခုဖြစ်သည်။ Toxicity ဘယ်လောက်ထိခံနိုင်ရမလဲ? ဒီသုတေသနတွင် clinical trial methods များကို analysis လုပ်ပြီး clinical trial outcomes ကိုခန့်မှန်းရန်နည်းလမ်းအသစ်တစ်ခုကို ဖန်တီးခဲ့သည်။ အထူးသဖြင့် random forest ကိုအသုံးပြု၍ [classifier](../../4-Classification/README.md) တစ်ခုကို ဖန်တီးခဲ့သည်။
[Reference](https://www.sciencedirect.com/science/article/pii/S2451945616302914)

### လူနာပြန်လည်လက်ခံမှုစီမံခန့်ခွဲခြင်း

ဆေးရုံ care သည် အလွန်ကုန်ကျစရိတ်များရှိပြီး လူနာများကိုပြန်လည်လက်ခံရခြင်းသည် အထူးကုန်ကျစရိတ်များရှိသည်။ ဒီစာတမ်းတွင် ML ကို clustering algorithms အသုံးပြု၍ readmission potential ကိုခန့်မှန်းရန် ဘယ်လိုအသုံးပြုကြောင်းကို ဖော်ပြထားသည်။ ဒီ clusters များက "readmissions များတွင် common cause ရှိနိုင်သောအုပ်စုများကို ရှာဖွေရန်" analyst များကိုကူညီပေးသည်။
[Reference](https://healthmanagement.org/c/healthmanagement/issuearticle/hospital-readmissions-and-machine-learning)

### ရောဂါစီမံခန့်ခွဲခြင်း

နောက်ဆုံးကာလ pandemic သည် ML ကိုရောဂါပျံ့နှံ့မှုကိုတားဆီးရန် ဘယ်လိုအသုံးပြုနိုင်သည်ဆိုတာကို အလင်းရောင်ပေးခဲ့သည်။ ဒီ article တွင် ARIMA, logistic curves, linear regression, နှင့် SARIMA ကိုအသုံးပြုထားသည်။ "ဒီအလုပ်သည် virus ၏ပျံ့နှံ့နှုန်းကိုတွက်ချက်ရန်နှင့် သေဆုံးမှုများ၊ ပြန်လည်ကောင်းမွန်မှုများ၊ နှင့် အတည်ပြုမှုများကိုခန့်မှန်းရန် ကြိုးစားမှုတစ်ခုဖြစ်သည်။"
[Reference](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7979218/)

## 🌲 သဘာဝပတ်ဝန်းကျင်နှင့် Green Tech

သဘာဝနှင့်သဘာဝပတ်ဝန်းကျင်သည် အလွန်နူးညံ့သောစနစ်များဖြစ်ပြီး တိရစ္ဆာန်များနှင့်သဘာဝအကြားဆက်ဆံရေးကိုအဓိကထားသည်။ ဒီစနစ်များကိုတိကျစွာတိုင်းတာနိုင်ရန်နှင့် တစ်စုံတစ်ခုဖြစ်ပျက်ပါက သင့်တော်သောအရေးယူမှုများလုပ်ဆောင်ရန် အရေးကြီးသည်။

### သစ်တောစီမံခန့်ခွဲမှု

သင်သည် [Reinforcement Learning](../../8-Reinforcement/README.md) ကို ယခင်သင်ခန်းစာများတွင်လေ့လာခဲ့ပါသည်။ သဘာဝတွင် pattern များကိုခန့်မှန်းရန် အလွန်အသုံးဝင်သည်။ အထူးသဖြင့် သစ်တောမီးလောင်မှုများနှင့် invasive species များပျံ့နှံ့မှုကဲ့သို့သော ecological ပြဿနာများကို tracking လုပ်ရန် အသုံးပြုနိုင်သည်။ ကနေဒါတွင် သုတေသနလုပ်ငန်းတစ်ခုသည် satellite images ကိုအသုံးပြု၍ forest wildfire dynamics models များကို reinforcement learning ဖြင့်တည်ဆောက်ခဲ့သည်။
[Reference](https://www.frontiersin.org/articles/10.3389/fict.2018.00006/full)

### တိရစ္ဆာန်များ၏လှုပ်ရှားမှုကိုစစ်ဆေးခြင်း

Deep learning သည် တိရစ္ဆာန်လှုပ်ရှားမှုများကို visually tracking လုပ်ရန် revolution တစ်ခုဖန်တီးခဲ့သော်လည်း classic ML သည် ဒီအလုပ်တွင် အရေးပါနေဆဲဖြစ်သည်။ 

Farm animals များ၏လှုပ်ရှားမှုများကို tracking လုပ်ရန် sensor များနှင့် IoT ကိုအသုံးပြုသော်လည်း data ကို preprocess လုပ်ရန် basic ML techniques များကိုအသုံးပြုသည်။ ဥပမာ - ဒီစာတမ်းတွင် classifier algorithms များကိုအသုံးပြု၍ သိုးများ၏ posture များကိုစစ်ဆေးခဲ့သည်။
[Reference](https://druckhaus-hofmann.de/gallery/31-wj-feb-2020.pdf)

### ⚡️ စွမ်းအင်စီမံခန့်ခွဲမှု

[time series forecasting](../../7-TimeSeries/README.md) သင်ခန်းစာများတွင် supply နှင့် demand ကိုနားလည်ခြင်းအပေါ်အခြေခံပြီး smart parking meters ကို revenue ရရှိရန်အသုံးပြုခဲ့သည်။ ဒီ article တွင် clustering, regression နှင့် time series forecasting ကိုပေါင်းစပ်ပြီး smart metering အပေါ်အခြေခံ၍ အနာဂတ်စွမ်းအင်အသုံးပြုမှုကိုခန့်မှန်းခဲ့သည်။
[Reference](https://www-cdn.knime.com/sites/default/files/inline-images/knime_bigdata_energy_timeseries_whitepaper.pdf)

## 💼 အာမခံ

အာမခံကဏ္ဍသည် ML ကိုအသုံးပြု၍ financial နှင့် actuarial models များကိုတည်ဆောက်ခြင်းနှင့် optimize လုပ်ခြင်းအတွက် အသုံးပြုသည်။

### Volatility Management

MetLife သည် life insurance provider တစ်ခုဖြစ်ပြီး သူတို့ financial models တွင် volatility ကိုဘယ်လိုခန့်မှန်းပြီး လျှော့ချကြောင်းကို ဖော်ပြထားသည်။ ဒီ article တွင် binary နှင့် ordinal classification visualizations များကိုတွေ့နိုင်သည်။
[Reference](https://investments.metlife.com/content/dam/metlifecom/us/investments/insights/research-topics/macro-strategy/pdf/MetLifeInvestmentManagement_MachineLearnedRanking_070920.pdf)

## 🎨 အနုပညာ၊ ယဉ်ကျေးမှုနှင့်စာပေ

အနုပညာတွင် ဥပမာ - သတင်းစာပညာတွင် စိတ်ဝင်စားဖွယ်ရာပြဿနာများစွာရှိသည်။ Fake news ကိုရှာဖွေခြင်းသည် လူများ၏အမြင်ကိုသက်ရောက်စေခြင်းနှင့် ဒီမိုကရေစီများကိုတုန်လှုပ်စေခြင်းအထိ သက်ရောက်မှုရှိသည်ဟု သက်သေပြထားသည်။ ပြတိုက်များသည် artifacts များအကြားဆက်စပ်မှုများကိုရှာဖွေခြင်းမှစ၍ resource planning အထိ ML ကိုအသုံးပြုနိုင်သည်။

### Fake news detection

Fake news ကိုရှာဖွေခြင်းသည် ယနေ့မီဒီယာတွင် ကြောင်နှင့်ကြွက်ကစားပွဲတစ်ခုဖြစ်လာသည်။ ဒီ article တွင် သုတေသနလုပ်ငန်းများက ML techniques များစွာကိုပေါင်းစပ်ပြီး အကောင်းဆုံးမော်ဒယ်ကို deploy လုပ်နိုင်ကြောင်းကိုဖော်ပြထားသည်။ "ဒီစနစ်သည် data မှ features များကို extract လုပ်ရန် natural language processing ကို
## [Post-lecture quiz](https://ff-quizzes.netlify.app/en/ml/)

## ပြန်လည်သုံးသပ်ခြင်းနှင့် ကိုယ်တိုင်လေ့လာခြင်း

Wayfair ရဲ့ ဒေတာသိပ္ပံအဖွဲ့က ML ကို သူတို့ကုမ္ပဏီမှာ ဘယ်လိုအသုံးပြုတယ်ဆိုတာနဲ့ပတ်သက်ပြီး စိတ်ဝင်စားဖွယ်ဗီဒီယိုများစွာရှိပါတယ်။ [ကြည့်ရှုဖို့](https://www.youtube.com/channel/UCe2PjkQXqOuwkW1gw6Ameuw/videos) တန်ပါတယ်!

## လုပ်ငန်းတာဝန်

[A ML scavenger hunt](assignment.md)

---

**ဝက်ဘ်ဆိုက်မှတ်ချက်**:  
ဤစာရွက်စာတမ်းကို AI ဘာသာပြန်ဝန်ဆောင်မှု [Co-op Translator](https://github.com/Azure/co-op-translator) ကို အသုံးပြု၍ ဘာသာပြန်ထားပါသည်။ ကျွန်ုပ်တို့သည် တိကျမှန်ကန်မှုအတွက် ကြိုးစားနေပါသော်လည်း၊ အလိုအလျောက်ဘာသာပြန်ဆိုမှုများတွင် အမှားများ သို့မဟုတ် မတိကျမှုများ ပါဝင်နိုင်သည်ကို ကျေးဇူးပြု၍ သတိပြုပါ။ မူရင်းဘာသာစကားဖြင့် ရေးသားထားသော စာရွက်စာတမ်းကို အာဏာတည်သော ရင်းမြစ်အဖြစ် သတ်မှတ်သင့်ပါသည်။ အရေးကြီးသော အချက်အလက်များအတွက် လူပညာရှင်များမှ ဘာသာပြန်ဆိုမှုကို အကြံပြုပါသည်။ ဤဘာသာပြန်ဆိုမှုကို အသုံးပြုခြင်းမှ ဖြစ်ပေါ်လာသော နားလည်မှုမှားမှုများ သို့မဟုတ် အဓိပ္ပာယ်မှားမှုများအတွက် ကျွန်ုပ်တို့သည် တာဝန်မယူပါ။