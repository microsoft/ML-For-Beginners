<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "8d32dadeda93c6fb5c43619854882ab1",
  "translation_date": "2025-09-05T14:06:07+00:00",
  "source_file": "6-NLP/4-Hotel-Reviews-1/README.md",
  "language_code": "my"
}
-->
# ဟိုတယ်ပြန်လည်သုံးသပ်မှုများနှင့် စိတ်ခံစားမှုခွဲခြမ်းစိတ်ဖြာမှု - ဒေတာကို အလုပ်လုပ်စေခြင်း

ဤအပိုင်းတွင် သင်သည် ယခင်သင်ခန်းစာများတွင် သင်ယူခဲ့သော နည်းလမ်းများကို အသုံးပြု၍ ဒေတာအစုအဝေးကြီးတစ်ခုကို စူးစမ်းလေ့လာမှု ဒေတာခွဲခြမ်းစိတ်ဖြာမှု (EDA) ပြုလုပ်မည်ဖြစ်သည်။ အမျိုးမျိုးသော ကော်လံများ၏ အသုံးဝင်မှုကို နားလည်ပြီးနောက် သင်သည် အောက်ပါအရာများကို သင်ယူမည်ဖြစ်သည်-

- မလိုအပ်သော ကော်လံများကို ဖယ်ရှားနည်း
- ရှိပြီးသား ကော်လံများအပေါ် အခြေခံပြီး ဒေတာအသစ်များကို တွက်ချက်နည်း
- နောက်ဆုံး စိန်ခေါ်မှုတွင် အသုံးပြုရန်အတွက် ရလဒ်အဖြစ်ရသော ဒေတာအစုအဝေးကို သိမ်းဆည်းနည်း

## [Pre-lecture quiz](https://ff-quizzes.netlify.app/en/ml/)

### အကျဉ်းချုပ်

ယခုအချိန်ထိ သင်သည် စာသားဒေတာသည် ကိန်းဂဏန်းဒေတာများနှင့် မတူကြောင်း သင်ယူခဲ့ပြီဖြစ်သည်။ လူသားများရေးသားထားသည့် သို့မဟုတ် ပြောဆိုထားသည့် စာသားများကို ပုံစံများနှင့် မကြာခဏဖြစ်သော စကားလုံးများ၊ စိတ်ခံစားမှုများနှင့် အဓိပ္ပါယ်များကို ရှာဖွေခွဲခြမ်းနိုင်သည်။ ဤသင်ခန်းစာတွင် သင်သည် အမှန်တကယ်သော ဒေတာအစုအဝေးနှင့် အမှန်တကယ်သော စိန်ခေါ်မှုတစ်ခုကို ရင်ဆိုင်မည်ဖြစ်သည် - **[515K Hotel Reviews Data in Europe](https://www.kaggle.com/jiashenliu/515k-hotel-reviews-data-in-europe)** (ဤဒေတာသည် [CC0: Public Domain license](https://creativecommons.org/publicdomain/zero/1.0/) ဖြင့် ထုတ်ဝေထားသည်)။ ဒေတာကို Booking.com မှ ပြည်သူ့ရင်းမြစ်များမှ စုဆောင်းထားပြီး ဒေတာဖန်တီးသူမှာ Jiashen Liu ဖြစ်သည်။

### ပြင်ဆင်မှု

သင်လိုအပ်မည့်အရာများ-

* Python 3 ဖြင့် .ipynb notebooks များကို အလုပ်လုပ်စေနိုင်ရမည်
* pandas
* NLTK, [ဤနေရာတွင် ဒေသတွင်းတွင် ထည့်သွင်းပါ](https://www.nltk.org/install.html)
* Kaggle တွင် ရရှိနိုင်သည့် ဒေတာအစုအဝေး [515K Hotel Reviews Data in Europe](https://www.kaggle.com/jiashenliu/515k-hotel-reviews-data-in-europe)။ ဒေတာကို ဖိုင်ဖွင့်ပြီးနောက် 230 MB ခန့်ရှိသည်။ ဒေတာကို ဤ NLP သင်ခန်းစာများနှင့် ဆက်စပ်သော `/data` ဖိုလ်ဒါတွင် ဒေါင်းလုပ်ဆွဲထားပါ။

## စူးစမ်းလေ့လာမှု ဒေတာခွဲခြမ်းစိတ်ဖြာမှု

ဤစိန်ခေါ်မှုသည် စိတ်ခံစားမှုခွဲခြမ်းစိတ်ဖြာမှုနှင့် ဧည့်သည်ပြန်လည်သုံးသပ်မှုအမှတ်များကို အသုံးပြု၍ ဟိုတယ်အကြံပြုမှုဘော့တစ်ခုကို တည်ဆောက်နေသည်ဟု သင်ယူထားသည်။ သင်အသုံးပြုမည့် ဒေတာအစုအဝေးတွင် မြို့ကြီး ၆ ခုရှိ ဟိုတယ် ၁၄၉၃ ခုမှ ပြန်လည်သုံးသပ်မှုများ ပါဝင်သည်။

Python, ဟိုတယ်ပြန်လည်သုံးသပ်မှုဒေတာအစုအဝေးနှင့် NLTK ၏ စိတ်ခံစားမှုခွဲခြမ်းစိတ်ဖြာမှုကို အသုံးပြု၍ သင်သည် အောက်ပါအရာများကို ရှာဖွေနိုင်သည်-

* ပြန်လည်သုံးသပ်မှုများတွင် အကြိမ်ရေအများဆုံးဖြစ်သော စကားလုံးများနှင့် စကားစုများက ဘာတွေလဲ။
* ဟိုတယ်ကို ဖော်ပြသည့် *tags* များသည် ပြန်လည်သုံးသပ်မှုအမှတ်များနှင့် ဆက်စပ်နေပါသလား (ဥပမာ- *Family with young children* အတွက် အနုတ်လက္ခဏာပြန်လည်သုံးသပ်မှုများသည် *Solo traveller* ထက် ပိုများနေပါသလား၊ ဤဟိုတယ်သည် *Solo travellers* အတွက် ပိုသင့်တော်ကြောင်း ပြသနိုင်သည်)။
* NLTK ၏ စိတ်ခံစားမှုအမှတ်များသည် ဟိုတယ်ပြန်လည်သုံးသပ်သူ၏ ကိန်းဂဏန်းအမှတ်နှင့် 'သဘောတူ' နေပါသလား။

#### ဒေတာအစုအဝေး

သင်ဒေါင်းလုပ်ဆွဲပြီး ဒေသတွင်းတွင် သိမ်းဆည်းထားသော ဒေတာအစုအဝေးကို စူးစမ်းကြည့်ပါ။ ဖိုင်ကို VS Code သို့မဟုတ် Excel ကဲ့သို့သော တည်းဖြတ်ရေးဆော့ဖ်ဝဲတစ်ခုဖြင့် ဖွင့်ပါ။

ဒေတာအစုအဝေးတွင် ပါဝင်သော ခေါင်းစဉ်များမှာ အောက်ပါအတိုင်းဖြစ်သည်-

*Hotel_Address, Additional_Number_of_Scoring, Review_Date, Average_Score, Hotel_Name, Reviewer_Nationality, Negative_Review, Review_Total_Negative_Word_Counts, Total_Number_of_Reviews, Positive_Review, Review_Total_Positive_Word_Counts, Total_Number_of_Reviews_Reviewer_Has_Given, Reviewer_Score, Tags, days_since_review, lat, lng*

ဤခေါင်းစဉ်များကို အောက်ပါအတိုင်း အုပ်စုဖွဲ့ထားသည်-

##### ဟိုတယ်ကော်လံများ

* `Hotel_Name`, `Hotel_Address`, `lat` (latitude), `lng` (longitude)
  * *lat* နှင့် *lng* ကို အသုံးပြု၍ Python ဖြင့် ဟိုတယ်တည်နေရာများကို မြေပုံပေါ်တွင် ရှုထောင့်အရောင်သတ်မှတ်ထားပြီး ရေးဆွဲနိုင်သည်။
  * Hotel_Address သည် ကျွန်ုပ်တို့အတွက် ထင်ရှားစွာ အသုံးမဝင်သည့်အရာဖြစ်ပြီး၊ အလွယ်တကူ စီစစ်ခြင်းနှင့် ရှာဖွေမှုအတွက် နိုင်ငံတစ်ခုဖြင့် အစားထိုးမည်ဖြစ်သည်။

**ဟိုတယ် Meta-review ကော်လံများ**

* `Average_Score`
  * ဒေတာဖန်တီးသူအဆိုအရ၊ ဤကော်လံသည် *နောက်ဆုံးနှစ်အတွင်းမှ နောက်ဆုံးမှတ်ချက်အပေါ် အခြေခံ၍ တွက်ချက်ထားသော ဟိုတယ်၏ ပျမ်းမျှအမှတ်* ဖြစ်သည်။ ၎င်းသည် အမှတ်ကို တွက်ချက်သည့် ပုံစံအနေနှင့် ထူးဆန်းသော်လည်း၊ ယခုအချိန်အတွက် ကျွန်ုပ်တို့သည် ၎င်းကို ယုံကြည်ရမည်ဖြစ်သည်။
  
  ✅ ဤဒေတာအတွင်းရှိ အခြားသော ကော်လံများအပေါ် အခြေခံ၍ ပျမ်းမျှအမှတ်ကို တွက်ချက်ရန် နည်းလမ်းတစ်ခုကို သင်စဉ်းစားနိုင်ပါသလား။

* `Total_Number_of_Reviews`
  * ဤဟိုတယ်သည် ရရှိထားသော ပြန်လည်သုံးသပ်မှုအရေအတွက် - ၎င်းသည် (အချို့သော ကုဒ်ရေးသားမှုမပြုလုပ်မီ) ဒေတာအစုအဝေးတွင် ပါဝင်သော ပြန်လည်သုံးသပ်မှုများကို ဆိုလိုသည်ဟု မရှင်းလင်းပါ။
* `Additional_Number_of_Scoring`
  * ၎င်းသည် ပြန်လည်သုံးသပ်မှုအမှတ်ကို ပေးခဲ့သော်လည်း ပြန်လည်သုံးသပ်မှုအနုတ်လက္ခဏာ သို့မဟုတ် အပေါင်းလက္ခဏာကို မရေးသားခဲ့သော အခြေအနေကို ဆိုလိုသည်။

**ပြန်လည်သုံးသပ်မှုကော်လံများ**

- `Reviewer_Score`
  - ၎င်းသည် အနည်းဆုံးနှင့် အများဆုံးတန်ဖိုး 2.5 နှင့် 10 အကြားတွင် အများဆုံး ဒသမ ၁ ချက်ရှိသော ကိန်းဂဏန်းတန်ဖိုးဖြစ်သည်။
  - အနည်းဆုံးအမှတ်ဖြစ်နိုင်သော 2.5 သည် အနိမ့်ဆုံးအမှတ်ဖြစ်ရမည့်အကြောင်းကို မရှင်းလင်းထားပါ။
- `Negative_Review`
  - ပြန်လည်သုံးသပ်သူသည် ဘာမှ မရေးသားပါက၊ ဤကွင်းသည် "**No Negative**" ဟု ရေးထားမည်။
  - ပြန်လည်သုံးသပ်သူသည် အနုတ်လက္ခဏာကွင်းတွင် အပေါင်းလက္ခဏာပြန်လည်သုံးသပ်မှုကို ရေးသားနိုင်သည် (ဥပမာ- "ဤဟိုတယ်တွင် မကောင်းသောအရာမရှိပါ")။
- `Review_Total_Negative_W
🚨 သတိပေးချက်  
ဒီ dataset ကို အသုံးပြုတဲ့အခါမှာ သင်ရေးတဲ့ code က text ကို ဖတ်စရာမလိုဘဲ text ထဲကနေ တစ်ခုခုကိုတွက်ချက်ပေးနိုင်ရမယ်။ ဒါဟာ NLP ရဲ့ အဓိကအချက်ဖြစ်ပြီး လူ့အဖွဲ့အစည်းမပါဘဲ အဓိပ္ပါယ်နဲ့ခံစားချက်ကို အနက်ဖွင့်ပေးနိုင်တာပဲဖြစ်ပါတယ်။ သို့သော် သင်အချို့သော အနုတ်လက္ခဏာရှိတဲ့ review တွေကို ဖတ်မိနိုင်ပါတယ်။ အဲ့ဒီ review တွေကို မဖတ်ဖို့ အကြံပေးချင်ပါတယ်၊ ဘာဖြစ်လို့လဲဆိုတော့ သင်ဖတ်စရာမလိုလို့ပါ။ အချို့ review တွေက အလွဲလွဲအချော်ချော်ဖြစ်ပြီး hotel နဲ့မသက်ဆိုင်တဲ့ အနုတ်လက္ခဏာရှိတဲ့ review တွေဖြစ်နိုင်ပါတယ်၊ ဥပမာ "ရာသီဥတုက မကောင်းဘူး" ဆိုတဲ့အရာတွေ၊ hotel ရဲ့ ထိန်းချုပ်မှုအောက်မှာမရှိတဲ့အရာတွေ၊ ဒါမှမဟုတ် ဘယ်သူ့အတွက်မှ ထိန်းချုပ်လို့မရတဲ့အရာတွေပါ။ ဒါပေမယ့် review တွေထဲမှာ အနောက်ဖက်မှောင်မိုက်တဲ့အပိုင်းလည်းရှိပါတယ်။ အချို့သော အနုတ်လက္ခဏာရှိတဲ့ review တွေက လူမျိုးရေး၊ လိင်ရေး၊ အသက်အရွယ်ရေး အနက်ဖွင့်မှုတွေပါဝင်နိုင်ပါတယ်။ ဒါဟာ စိတ်မကောင်းစရာကောင်းပေမယ့် public website မှာ scrape လုပ်ထားတဲ့ dataset မှာ ဖြစ်နိုင်တဲ့အရာပါ။ အချို့သော reviewer တွေက သင်မကြိုက်မယ်၊ သက်တောင့်သက်သာမရှိမယ်၊ ဒါမှမဟုတ် စိတ်မကောင်းစရာကောင်းမယ်လို့ ခံစားရမယ့် review တွေကို ရေးသားထားနိုင်ပါတယ်။ sentiment ကို code က တိုင်းတာပေးစေပြီး review တွေကို ကိုယ်တိုင်ဖတ်ပြီး စိတ်မကောင်းဖြစ်စေဖို့ မလုပ်ပါနဲ့။ ဒါဆိုလည်း အဲ့ဒီလိုအရာတွေကို ရေးသားတဲ့သူတွေက အနည်းငယ်ပဲရှိပေမယ့် သူတို့ရှိနေတယ်ဆိုတာတော့ အမှန်ပါပဲ။
## လေ့ကျင့်မှု - ဒေတာစူးစမ်းခြင်း  
### ဒေတာကို တင်ပါ  

ဒေတာကို မျက်မြင်ဖြင့် စစ်ဆေးတာလုံလောက်ပြီဆိုရင်၊ အခုတော့ သင့်ရဲ့ ကုဒ်ကိုရေးပြီး အဖြေတွေကို ရယူကြည့်ပါ။ ဒီအပိုင်းမှာ pandas library ကို အသုံးပြုမှာဖြစ်ပါတယ်။ သင့်ရဲ့ ပထမဆုံးတာဝန်က CSV ဒေတာကို load လုပ်ပြီး ဖတ်နိုင်တာ သေချာစေဖို့ပါ။ pandas library မှာ CSV loader တစ်ခုရှိပြီး၊ ရလဒ်ကို dataframe အနေနဲ့ ထည့်သွင်းပေးပါတယ်၊ ယခင် သင်ခန်းစာတွေမှာလိုပဲ။ ကျွန်တော်တို့ load လုပ်မယ့် CSV ဖိုင်မှာ တန်းစီထားတဲ့ အတန်း ၅ သိန်းကျော်ရှိပေမယ့် ကော်လံ ၁၇ ခုသာ ပါဝင်ပါတယ်။ Pandas က dataframe နဲ့ အလုပ်လုပ်ဖို့ အစွမ်းထက်နည်းလမ်းတွေ အများကြီးပေးထားပြီး၊ တန်းစီထားတဲ့ အတန်းတိုင်းမှာ လုပ်ဆောင်ချက်တွေ ပြုလုပ်နိုင်ပါတယ်။  

ဒီသင်ခန်းစာမှာ ဒီနေရာကစပြီး ကုဒ် snippet တွေ၊ ကုဒ်ရဲ့ ရှင်းလင်းချက်တွေ၊ ရလဒ်တွေက ဘာကို ဆိုလိုတာလဲဆိုတာ ဆွေးနွေးချက်တွေ ပါဝင်မှာဖြစ်ပါတယ်။ သင့်ရဲ့ ကုဒ်အတွက် _notebook.ipynb_ ကို အသုံးပြုပါ။  

အရင်ဆုံး သင်အသုံးပြုမယ့် ဒေတာဖိုင်ကို load လုပ်တာကစပါမယ် -  

```python
# Load the hotel reviews from CSV
import pandas as pd
import time
# importing time so the start and end time can be used to calculate file loading time
print("Loading data file now, this could take a while depending on file size")
start = time.time()
# df is 'DataFrame' - make sure you downloaded the file to the data folder
df = pd.read_csv('../../data/Hotel_Reviews.csv')
end = time.time()
print("Loading took " + str(round(end - start, 2)) + " seconds")
```  

ဒေတာကို load လုပ်ပြီးပြီဆိုရင်၊ အခုတော့ ဒေတာပေါ်မှာ လုပ်ဆောင်ချက်တွေ ပြုလုပ်နိုင်ပါပြီ။ ဒီကုဒ်ကို သင့်ရဲ့ အစီအစဉ်ရဲ့ အပေါ်ဆုံးမှာ ထည့်ထားပါ။  

## ဒေတာကို စူးစမ်းပါ  

ဒီအခါမှာတော့ ဒေတာဟာ *သန့်ရှင်း* ဖြစ်ပြီး၊ အလုပ်လုပ်ဖို့ အဆင်သင့်ဖြစ်နေပါပြီ။ ဒါက ဘာကိုဆိုလိုတာလဲဆိုရင်၊ အင်္ဂလိပ်စာလုံးတွေကိုသာ မျှော်လင့်နေတဲ့ အယ်လဂိုရီသမ်တွေကို အနှောက်အယှက်ဖြစ်စေမယ့် အခြားဘာသာစကားတွေရဲ့ စာလုံးတွေ မပါဝင်ဘူးဆိုတာပါ။  

✅ သင့်အနေနဲ့ NLP နည်းပညာတွေကို အသုံးချမယ့်အခါ၊ ဒေတာကို အစပိုင်းမှာ format ပြင်ဆင်ဖို့ လိုအပ်တတ်ပါတယ်၊ ဒါပေမယ့် ဒီအခါမှာတော့ မလိုအပ်ပါဘူး။ သင့်အနေနဲ့ အင်္ဂလိပ်မဟုတ်တဲ့ စာလုံးတွေကို ဘယ်လိုကိုင်တွယ်မလဲ?  

ဒေတာကို load လုပ်ပြီးတာနဲ့၊ ကုဒ်နဲ့အတူ စူးစမ်းနိုင်တာ သေချာစေပါ။ `Negative_Review` နဲ့ `Positive_Review` ကော်လံတွေကို အလွယ်တကူ အာရုံစိုက်ချင်တတ်ပါတယ်။ ဒီကော်လံတွေမှာ သင့်ရဲ့ NLP အယ်လဂိုရီသမ်တွေ အလုပ်လုပ်ဖို့ သဘာဝစာသားတွေ ပါဝင်ပါတယ်။ ဒါပေမယ့် ခဏစောင့်ပါ! NLP နဲ့ စိတ်ခံစားမှုကို စတင်မလုပ်ခင်၊ pandas နဲ့ ရှာဖွေတွေ့ရှိထားတဲ့ တန်ဖိုးတွေဟာ ဒေတာစဉ်မှာပေးထားတဲ့ တန်ဖိုးတွေနဲ့ ကိုက်ညီမကိုက်ညီ စစ်ဆေးဖို့ အောက်ပါကုဒ်ကို လိုက်နာပါ။  

## Dataframe လုပ်ဆောင်ချက်များ  

ဒီသင်ခန်းစာရဲ့ ပထမဆုံးတာဝန်ကတော့ အောက်ပါ အတည်ပြုချက်တွေကို စစ်ဆေးဖို့ ကုဒ်ရေးပါ။ ဒေတာဖရိမ်ကို ပြောင်းလဲမလုပ်ဘဲ စစ်ဆေးပါ။  

> Programming လုပ်ငန်းတာဝန်အများစုလိုပဲ၊ ဒီအလုပ်ကို ပြီးမြောက်ဖို့ နည်းလမ်းအများကြီး ရှိပါတယ်၊ ဒါပေမယ့် အကောင်းဆုံးအကြံပေးချက်ကတော့ သင်နောက်ပိုင်းမှာ ဒီကုဒ်ကို ပြန်ကြည့်တဲ့အခါ အလွယ်တကူ နားလည်နိုင်အောင် ရိုးရှင်းပြီး လွယ်ကူတဲ့ နည်းလမ်းနဲ့ လုပ်ပါ။ Dataframe တွေအတွက် Comprehensive API တစ်ခုရှိပြီး၊ သင်လိုချင်တာကို ထိရောက်စွာ ပြုလုပ်နိုင်မယ့် နည်းလမ်းတစ်ခု ရှိတတ်ပါတယ်။  

အောက်ပါမေးခွန်းတွေကို coding task အနေနဲ့ သတ်မှတ်ပြီး၊ ဖြေရှင်းချက်ကို မကြည့်ဘဲ ကြိုးစားဖြေပါ။  

1. သင် load လုပ်ထားတဲ့ dataframe ရဲ့ *shape* ကို print ထုတ်ပါ (shape ဆိုတာက အတန်းနဲ့ ကော်လံအရေအတွက်ပါ)  
2. Reviewer နိုင်ငံသားများအတွက် frequency count ကိုတွက်ပါ -  
   1. `Reviewer_Nationality` ကော်လံမှာ ဘယ်လောက် distinct value တွေရှိပြီး၊ အဲဒီ value တွေက ဘာတွေလဲ?  
   2. Dataset မှာ အများဆုံးတွေ့ရတဲ့ reviewer နိုင်ငံသားက ဘယ်နိုင်ငံသားလဲ (နိုင်ငံနဲ့ review အရေအတွက်ကို print ထုတ်ပါ)  
   3. နောက်ထပ် အများဆုံးတွေ့ရတဲ့ နိုင်ငံသား ၁၀ ယောက်နဲ့ သူတို့ရဲ့ frequency count တွေက ဘယ်လောက်လဲ?  
3. အများဆုံး review ရရှိတဲ့ ဟိုတယ်ကို top 10 reviewer နိုင်ငံသားအလိုက် တွက်ပါ။  
4. Dataset မှာ ဟိုတယ်တစ်ခုစီအတွက် review အရေအတွက် (frequency count) ကို တွက်ပါ။  
5. Dataset မှာ ဟိုတယ်တစ်ခုစီအတွက် `Average_Score` ကော်လံတစ်ခုရှိပေမယ့်၊ reviewer score တွေကို အသုံးပြုပြီး သင့်ကိုယ်တိုင်လည်း အလယ်ပျမ်းမျှကို တွက်နိုင်ပါတယ်။ `Calc_Average_Score` ဆိုတဲ့ ကော်လံခေါင်းစဉ်နဲ့ အသစ်တစ်ခုထည့်ပါ။  
6. `Average_Score` နဲ့ `Calc_Average_Score` တန်ဖိုးတွေဟာ တူညီတဲ့ ဟိုတယ်တွေ ရှိပါသလား (တစ်ဆင့်တည်းအနက် rounded)?  
   1. Python function တစ်ခုရေးပါ၊ Series (row) တစ်ခုကို argument အနေနဲ့ ယူပြီး၊ တန်ဖိုးတွေ မတူညီတဲ့အခါ message တစ်ခု print ထုတ်ပါ။ `.apply()` method ကို အသုံးပြုပြီး row တစ်ခုစီကို process လုပ်ပါ။  
7. `Negative_Review` ကော်လံမှာ "No Negative" ဆိုတဲ့ တန်ဖိုးရှိတဲ့ row အရေအတွက်ကို တွက်ပြီး print ထုတ်ပါ။  
8. `Positive_Review` ကော်လံမှာ "No Positive" ဆိုတဲ့ တန်ဖိုးရှိတဲ့ row အရေအတွက်ကို တွက်ပြီး print ထုတ်ပါ။  
9. `Positive_Review` ကော်လံမှာ "No Positive" နဲ့ `Negative_Review` ကော်လံမှာ "No Negative" ဆိုတဲ့ တန်ဖိုးနှစ်ခုလုံးရှိတဲ့ row အရေအတွက်ကို တွက်ပြီး print ထုတ်ပါ။  

### ကုဒ်ဖြေရှင်းချက်  

1. ```python
   print("The shape of the data (rows, cols) is " + str(df.shape))
   > The shape of the data (rows, cols) is (515738, 17)
   ```  

2. ```python
   # value_counts() creates a Series object that has index and values in this case, the country and the frequency they occur in reviewer nationality
   nationality_freq = df["Reviewer_Nationality"].value_counts()
   print("There are " + str(nationality_freq.size) + " different nationalities")
   # print first and last rows of the Series. Change to nationality_freq.to_string() to print all of the data
   print(nationality_freq) 
   
   There are 227 different nationalities
    United Kingdom               245246
    United States of America      35437
    Australia                     21686
    Ireland                       14827
    United Arab Emirates          10235
                                  ...  
    Comoros                           1
    Palau                             1
    Northern Mariana Islands          1
    Cape Verde                        1
    Guinea                            1
   Name: Reviewer_Nationality, Length: 227, dtype: int64
   ```  

3. ```python
   # What was the most frequently reviewed hotel for the top 10 nationalities
   # Normally with pandas you will avoid an explicit loop, but wanted to show creating a new dataframe using criteria (don't do this with large amounts of data because it could be very slow)
   for nat in nationality_freq[:10].index:
      # First, extract all the rows that match the criteria into a new dataframe
      nat_df = df[df["Reviewer_Nationality"] == nat]   
      # Now get the hotel freq
      freq = nat_df["Hotel_Name"].value_counts()
      print("The most reviewed hotel for " + str(nat).strip() + " was " + str(freq.index[0]) + " with " + str(freq[0]) + " reviews.") 
      
   The most reviewed hotel for United Kingdom was Britannia International Hotel Canary Wharf with 3833 reviews.
   The most reviewed hotel for United States of America was Hotel Esther a with 423 reviews.
   The most reviewed hotel for Australia was Park Plaza Westminster Bridge London with 167 reviews.
   The most reviewed hotel for Ireland was Copthorne Tara Hotel London Kensington with 239 reviews.
   The most reviewed hotel for United Arab Emirates was Millennium Hotel London Knightsbridge with 129 reviews.
   The most reviewed hotel for Saudi Arabia was The Cumberland A Guoman Hotel with 142 reviews.
   The most reviewed hotel for Netherlands was Jaz Amsterdam with 97 reviews.
   The most reviewed hotel for Switzerland was Hotel Da Vinci with 97 reviews.
   The most reviewed hotel for Germany was Hotel Da Vinci with 86 reviews.
   The most reviewed hotel for Canada was St James Court A Taj Hotel London with 61 reviews.
   ```  

4. ```python
   # First create a new dataframe based on the old one, removing the uneeded columns
   hotel_freq_df = df.drop(["Hotel_Address", "Additional_Number_of_Scoring", "Review_Date", "Average_Score", "Reviewer_Nationality", "Negative_Review", "Review_Total_Negative_Word_Counts", "Positive_Review", "Review_Total_Positive_Word_Counts", "Total_Number_of_Reviews_Reviewer_Has_Given", "Reviewer_Score", "Tags", "days_since_review", "lat", "lng"], axis = 1)
   
   # Group the rows by Hotel_Name, count them and put the result in a new column Total_Reviews_Found
   hotel_freq_df['Total_Reviews_Found'] = hotel_freq_df.groupby('Hotel_Name').transform('count')
   
   # Get rid of all the duplicated rows
   hotel_freq_df = hotel_freq_df.drop_duplicates(subset = ["Hotel_Name"])
   display(hotel_freq_df) 
   ```  

5. ```python
   # define a function that takes a row and performs some calculation with it
   def get_difference_review_avg(row):
     return row["Average_Score"] - row["Calc_Average_Score"]
   
   # 'mean' is mathematical word for 'average'
   df['Calc_Average_Score'] = round(df.groupby('Hotel_Name').Reviewer_Score.transform('mean'), 1)
   
   # Add a new column with the difference between the two average scores
   df["Average_Score_Difference"] = df.apply(get_difference_review_avg, axis = 1)
   
   # Create a df without all the duplicates of Hotel_Name (so only 1 row per hotel)
   review_scores_df = df.drop_duplicates(subset = ["Hotel_Name"])
   
   # Sort the dataframe to find the lowest and highest average score difference
   review_scores_df = review_scores_df.sort_values(by=["Average_Score_Difference"])
   
   display(review_scores_df[["Average_Score_Difference", "Average_Score", "Calc_Average_Score", "Hotel_Name"]])
   ```  

6. ```python
   # with lambdas:
   start = time.time()
   no_negative_reviews = df.apply(lambda x: True if x['Negative_Review'] == "No Negative" else False , axis=1)
   print("Number of No Negative reviews: " + str(len(no_negative_reviews[no_negative_reviews == True].index)))
   
   no_positive_reviews = df.apply(lambda x: True if x['Positive_Review'] == "No Positive" else False , axis=1)
   print("Number of No Positive reviews: " + str(len(no_positive_reviews[no_positive_reviews == True].index)))
   
   both_no_reviews = df.apply(lambda x: True if x['Negative_Review'] == "No Negative" and x['Positive_Review'] == "No Positive" else False , axis=1)
   print("Number of both No Negative and No Positive reviews: " + str(len(both_no_reviews[both_no_reviews == True].index)))
   end = time.time()
   print("Lambdas took " + str(round(end - start, 2)) + " seconds")
   
   Number of No Negative reviews: 127890
   Number of No Positive reviews: 35946
   Number of both No Negative and No Positive reviews: 127
   Lambdas took 9.64 seconds
   ```  

## အခြားနည်းလမ်း  

Lambda မသုံးဘဲ item တွေကို count လုပ်ပြီး၊ row တွေကို count လုပ်ဖို့ sum ကို အသုံးပြုပါ -  

   ```python
   # without lambdas (using a mixture of notations to show you can use both)
   start = time.time()
   no_negative_reviews = sum(df.Negative_Review == "No Negative")
   print("Number of No Negative reviews: " + str(no_negative_reviews))
   
   no_positive_reviews = sum(df["Positive_Review"] == "No Positive")
   print("Number of No Positive reviews: " + str(no_positive_reviews))
   
   both_no_reviews = sum((df.Negative_Review == "No Negative") & (df.Positive_Review == "No Positive"))
   print("Number of both No Negative and No Positive reviews: " + str(both_no_reviews))
   
   end = time.time()
   print("Sum took " + str(round(end - start, 2)) + " seconds")
   
   Number of No Negative reviews: 127890
   Number of No Positive reviews: 35946
   Number of both No Negative and No Positive reviews: 127
   Sum took 0.19 seconds
   ```  

127 rows မှာ `Negative_Review` နဲ့ `Positive_Review` ကော်လံတွေမှာ "No Negative" နဲ့ "No Positive" တန်ဖိုးတွေ ရှိနေတယ်ဆိုတာ သတိထားမိမှာပါ။ ဒါကဆိုရင် reviewer က ဟိုတယ်ကို နံပါတ်အဆင့်ပေးခဲ့ပေမယ့်၊ အပြုသဘော သို့မဟုတ် အနုတ်သဘော review မရေးခဲ့တာဖြစ်ပါတယ်။ ဒေတာစဉ်မှာ review မပါဝင်တဲ့ row တွေ ရှိတယ်ဆိုတာ မမျှော်လင့်ထားတာဖြစ်ပေမယ့်၊ ဒေတာကို စူးစမ်းတဲ့အခါ ဒီလို row တွေကို ရှာဖွေတွေ့ရှိရမှာပါ။  

အခုတော့ dataset ကို စူးစမ်းပြီးဖြစ်တဲ့အတွက်၊ နောက်သင်ခန်းစာမှာ ဒေတာကို filter လုပ်ပြီး sentiment analysis တစ်ခု ထည့်သွင်းပါမယ်။  

---  
## 🚀 စိန်ခေါ်မှု  

ဒီသင်ခန်းစာက ပြသသလို၊ သင့်ဒေတာနဲ့ ဒေတာရဲ့ အားနည်းချက်တွေကို နားလည်ဖို့ အရေးကြီးကြောင်း ပြသပါတယ်။ စာသားအခြေပြု ဒေတာတွေကို အထူးသဖြင့် သေချာစွာ စစ်ဆေးဖို့ လိုအပ်ပါတယ်။ စာသားအများကြီးပါတဲ့ dataset တွေကို စူးစမ်းပြီး၊ မော်ဒယ်မှာ bias သို့မဟုတ် skewed sentiment ဖြစ်စေနိုင်တဲ့ အပိုင်းတွေ ရှာဖွေကြည့်ပါ။  

## [Post-lecture quiz](https://ff-quizzes.netlify.app/en/ml/)  

## ပြန်လည်သုံးသပ်ခြင်းနှင့် ကိုယ်တိုင်လေ့လာခြင်း  

[ဒီ Learning Path on NLP](https://docs.microsoft.com/learn/paths/explore-natural-language-processing/?WT.mc_id=academic-77952-leestott) ကို လေ့လာပြီး၊ စကားပြောနဲ့ စာသားအခြေပြု မော်ဒယ်တွေ တည်ဆောက်တဲ့အခါ အသုံးပြုနိုင်မယ့် ကိရိယာတွေကို ရှာဖွေပါ။  

## လုပ်ငန်းတာဝန်  

[NLTK](assignment.md)  

---

**ဝက်ဘ်ဆိုက်မှတ်ချက်**:  
ဤစာရွက်စာတမ်းကို AI ဘာသာပြန်ဝန်ဆောင်မှု [Co-op Translator](https://github.com/Azure/co-op-translator) ကို အသုံးပြု၍ ဘာသာပြန်ထားပါသည်။ ကျွန်ုပ်တို့သည် တိကျမှန်ကန်မှုအတွက် ကြိုးစားနေသော်လည်း၊ အလိုအလျောက်ဘာသာပြန်ခြင်းတွင် အမှားများ သို့မဟုတ် မမှန်ကန်မှုများ ပါဝင်နိုင်ကြောင်း သတိပြုပါ။ မူလဘာသာစကားဖြင့် ရေးသားထားသော စာရွက်စာတမ်းကို အာဏာတည်သော ရင်းမြစ်အဖြစ် သတ်မှတ်သင့်ပါသည်။ အရေးကြီးသော အချက်အလက်များအတွက် လူ့ဘာသာပြန်ပညာရှင်များကို အသုံးပြုရန် အကြံပြုပါသည်။ ဤဘာသာပြန်ကို အသုံးပြုခြင်းမှ ဖြစ်ပေါ်လာသော နားလည်မှုမှားများ သို့မဟုတ် အဓိပ္ပါယ်မှားများအတွက် ကျွန်ုပ်တို့သည် တာဝန်မယူပါ။ 