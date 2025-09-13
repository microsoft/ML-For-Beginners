<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "b28a3a4911584062772c537b653ebbc7",
  "translation_date": "2025-09-05T12:09:13+00:00",
  "source_file": "5-Clustering/README.md",
  "language_code": "my"
}
-->
# စက်ရုပ်သင်ယူမှုအတွက် Clustering မော်ဒယ်များ

Clustering သည် စက်ရုပ်သင်ယူမှုလုပ်ငန်းတစ်ခုဖြစ်ပြီး၊ တူညီမှုရှိသော အရာများကို ရှာဖွေပြီး အုပ်စုများ (clusters) အဖြစ် စုစည်းပေးသည်။ Clustering သည် စက်ရုပ်သင်ယူမှု၏ အခြားနည်းလမ်းများနှင့် ကွဲပြားခြားနားပြီး၊ အလိုအလျောက်ဖြစ်ပျက်သည်။ တကယ်တော့ supervised learning နှင့် ဆန့်ကျင်ဘက်ဖြစ်သည်ဟု ဆိုနိုင်သည်။

## ဒေသဆိုင်ရာ ခေါင်းစဉ် - နိုင်ဂျီးရီးယားပရိသတ်၏ ဂီတအရသာအတွက် clustering မော်ဒယ်များ 🎧

နိုင်ဂျီးရီးယား၏ အမျိုးမျိုးသောပရိသတ်များတွင် အမျိုးမျိုးသော ဂီတအရသာများရှိသည်။ [ဒီဆောင်းပါး](https://towardsdatascience.com/country-wise-visual-analysis-of-music-taste-using-spotify-api-seaborn-in-python-77f5b749b421) မှ အားကိုးပြီး Spotify မှ ရှာဖွေထားသော ဒေတာကို အသုံးပြု၍ နိုင်ဂျီးရီးယားတွင် လူကြိုက်များသော ဂီတအချို့ကို ကြည့်ကြမည်။ ဒီဒေတာစဉ်တွင် သီချင်းများ၏ 'danceability' အဆင့်၊ 'acousticness'၊ 'loudness'၊ 'speechiness'၊ လူကြိုက်များမှုနှင့် 'energy' အကြောင်းအရာများ ပါဝင်သည်။ ဒီဒေတာတွင် ပုံစံများကို ရှာဖွေဖို့ စိတ်ဝင်စားဖွယ်ကောင်းပါသည်။

![A turntable](../../../5-Clustering/images/turntable.jpg)

> <a href="https://unsplash.com/@marcelalaskoski?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Marcela Laskoski</a> မှ <a href="https://unsplash.com/s/photos/nigerian-music?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Unsplash</a> တွင် ရိုက်ထားသော ဓာတ်ပုံ
  
ဒီသင်ခန်းစာများတွင် Clustering နည်းလမ်းများကို အသုံးပြု၍ ဒေတာကို လေ့လာသုံးသပ်ရန် နည်းလမ်းအသစ်များကို ရှာဖွေမည်ဖြစ်သည်။ Clustering သည် သင်၏ ဒေတာစဉ်တွင် label မပါဝင်သောအခါ အထူးအသုံးဝင်သည်။ Label ပါဝင်ပါက၊ ယခင်သင်ခန်းစာများတွင် သင်လေ့လာခဲ့သော classification နည်းလမ်းများက ပိုအသုံးဝင်နိုင်သည်။ သို့သော် label မပါဝင်သော ဒေတာကို အုပ်စုဖွဲ့လိုသောအခါ Clustering သည် ပုံစံများကို ရှာဖွေဖို့ အကောင်းဆုံးနည်းလမ်းဖြစ်သည်။

> Clustering မော်ဒယ်များနှင့် အလုပ်လုပ်ခြင်းကို လေ့လာရန် အထောက်အကူပြုသော low-code tools များရှိသည်။ [Azure ML](https://docs.microsoft.com/learn/modules/create-clustering-model-azure-machine-learning-designer/?WT.mc_id=academic-77952-leestott) ကို စမ်းကြည့်ပါ။

## သင်ခန်းစာများ

1. [Clustering ကိုမိတ်ဆက်ခြင်း](1-Visualize/README.md)
2. [K-Means Clustering](2-K-Means/README.md)

## အကျိုးတူ

ဒီသင်ခန်းစာများကို 🎶 [Jen Looper](https://www.twitter.com/jenlooper) မှ ရေးသားပြီး၊ [Rishit Dagli](https://rishit_dagli) နှင့် [Muhammad Sakib Khan Inan](https://twitter.com/Sakibinan) မှ အထောက်အကူပြုသုံးသပ်ချက်များဖြင့် အကောင်းဆုံးဖြစ်စေရန် ပြုလုပ်ထားသည်။

[Nigerian Songs](https://www.kaggle.com/sootersaalu/nigerian-songs-spotify) ဒေတာစဉ်ကို Kaggle မှ Spotify မှ ရှာဖွေထားသည်။

ဒီသင်ခန်းစာကို ဖန်တီးရာတွင် အထောက်အကူပြုခဲ့သော အသုံးဝင်သော K-Means နမူနာများမှာ [iris exploration](https://www.kaggle.com/bburns/iris-exploration-pca-k-means-and-gmm-clustering)၊ [introductory notebook](https://www.kaggle.com/prashant111/k-means-clustering-with-python) နှင့် [hypothetical NGO example](https://www.kaggle.com/ankandash/pca-k-means-clustering-hierarchical-clustering) တို့ဖြစ်သည်။

---

**အကြောင်းကြားချက်**:  
ဤစာရွက်စာတမ်းကို AI ဘာသာပြန်ဝန်ဆောင်မှု [Co-op Translator](https://github.com/Azure/co-op-translator) ကို အသုံးပြု၍ ဘာသာပြန်ထားပါသည်။ ကျွန်ုပ်တို့သည် တိကျမှုအတွက် ကြိုးစားနေပါသော်လည်း၊ အလိုအလျောက် ဘာသာပြန်မှုများတွင် အမှားများ သို့မဟုတ် မမှန်ကန်မှုများ ပါဝင်နိုင်သည်ကို သတိပြုပါ။ မူရင်းဘာသာစကားဖြင့် ရေးသားထားသော စာရွက်စာတမ်းကို အာဏာတရ အရင်းအမြစ်အဖြစ် သတ်မှတ်သင့်ပါသည်။ အရေးကြီးသော အချက်အလက်များအတွက် လူ့ဘာသာပြန်ပညာရှင်များမှ ပရော်ဖက်ရှင်နယ် ဘာသာပြန်မှုကို အကြံပြုပါသည်။ ဤဘာသာပြန်မှုကို အသုံးပြုခြင်းမှ ဖြစ်ပေါ်လာသော အလွဲအလွဲအချော်များ သို့မဟုတ် အနားလွဲမှုများအတွက် ကျွန်ုပ်တို့သည် တာဝန်မယူပါ။