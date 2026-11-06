# မက်ရှင်လေ့လာမှုအတွက် ကလပ်စတာအမျိုးအစား မော်ဒယ်များ

ကလပ်စတာက မက်ရှင်လေ့လာမှုလုပ်ငန်းတစ်ခုဖြစ်ပြီး ပစ္စည်းများသည် တူညီသောပစ္စည်းများကို ရှာဖွေကာ ကလပ်စတာဟုခေါ်သောအုပ်စုများသို့ စုစည်းပေးသည်။ ကလပ်စတာသည် မက်ရှင်လေ့လာမှုတွင် အခြားနည်းလမ်းများနှင့် ကွဲပြားသည့်အချက်မှာ ပရိုဆက်များသည် အလိုအလျောက်ဖြစ်ပွားခြင်း ဖြစ်ပြီး၊ ၎င်းသည် supervised learning (ထိန်းကြပ်သင်ကြားမှု) နှင့် ဆန့်ကျင်နေသောအရာဟု ဆိုနိုင်သည်။

## ဒေသဆိုင်ရာ အကြောင်းအရာ။ ညီဂျီးရီးယား ပြည်သူ့ ရောနှောဂီတအရသာအတွက် ကလပ်စတာ မော်ဒယ်များ 🎧

ညီဂျီးရီးယား၏ အမျိုးမျိုးသော ပြည်သူ့အုပ်စုများသည် အမျိုးမျိုးသောဂီတအရသာများရှိသည်။ Spotify မှ ရယူထားသည့် ဒေတာကို အသုံးပြုကာ ([ဒီဆောင်းပါး](https://towardsdatascience.com/country-wise-visual-analysis-of-music-taste-using-spotify-api-seaborn-in-python-77f5b749b421) မှ စိတ်ဓာတ်ယူ၍) ညီဂျီးရီးယားတွင် လူနှစ်သက်သောဂီတများကို ကြည့်မည်။ ဒီ dataset တွင် သီချင်းအမျိုးမျိုး၏ 'danceability' အမှတ်၊ 'acousticness'၊ အသံကြိမ်နှုန်း၊ 'speechiness'၊ လူကြိုက်နှုန်းနှင့် စွမ်းအင်တို့အကြောင်း ပါဝင်သည်။ ဒီဒေတာတွင် ပုံစံများကို ရှာဖွေတွေ့ရှိရမည်ဖြစ်၍ စိတ်ဝင်စားဖွယ်ကောင်းပါသည်။

![A turntable](../../../translated_images/my/turntable.f2b86b13c53302dc.webp)

> ဓာတ်ပုံကို <a href="https://unsplash.com/@marcelalaskoski?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Marcela Laskoski</a> မှ <a href="https://unsplash.com/s/photos/nigerian-music?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Unsplash</a> တွင်
  
ဒီသင်ခန်းစာတွဲတွင် သင်သည် ကလပ်စတာနည်းစနစ်များဖြင့် ဒေတာအသစ်များကို ခွဲခြားစိစစ်နိုင်သည့် နည်းလမ်းများကို ရှာဖွေတွေ့ရှိမည်။ ကလပ်စတာသည် သင့်ဒေတာတွင် label မပါသောအခါ အထူးအသုံးဝင်သည်။ label ပါက၊ ယခင်သင်ခန်းစာများတွင် သင်ယူခဲ့သည့် classification နည်းစနစ်များကို အသုံးပြုသင့်သည်။ သို့သော် label မပါသော ဒေတာများကို အုပ်စုခွဲချင်သည့်အခါ ကလပ်စတာသည် ပုံစံများကို ရှာဖွေရန် ကောင်းသောနည်းဖြစ်သည်။

> ကလပ်စတာ မော်ဒယ်များရှေ့ပြေးလေ့လာရာတွင် အထောက်အကူ ဖြစ်စေသည့် low-code tools များ ရှိသည်။ ဒီအလုပ်အတွက် [Azure ML](https://docs.microsoft.com/learn/modules/create-clustering-model-azure-machine-learning-designer/?WT.mc_id=academic-77952-leestott) ကို စမ်းကြည့်ပါ။

## သင်ခန်းစာများ

1. [ကလပ်စတာနည်းစနစ်များ၏ နိဒါန်း](1-Visualize/README.md)
2. [K-Means ကလပ်စတာ](2-K-Means/README.md)

## အခွင့်အရေးများ

ဒီသင်ခန်းစာများကို 🎶 [Jen Looper](https://www.twitter.com/jenlooper) မှ ရေးသားပြီး [Rishit Dagli](https://rishit_dagli/) နှင့် [Muhammad Sakib Khan Inan](https://twitter.com/Sakibinan) တို့ မှ ပြန်လည်သုံးသပ်မှုကူညီမှုများပါဝင်သည်။

[Nigerian Songs](https://www.kaggle.com/sootersaalu/nigerian-songs-spotify) dataset သည် Spotify မှ ကတ်ပြီး Kaggle မှ ရယူထားပါသည်။

ဒီသင်ခန်းစာဖန်တီးရာတွင် အထောက်အကူဖြစ်လာသည့် အသုံးဝင်သော K-Means နမူနာများမှာ [iris exploration](https://www.kaggle.com/bburns/iris-exploration-pca-k-means-and-gmm-clustering), [introductory notebook](https://www.kaggle.com/prashant111/k-means-clustering-with-python), နှင့် [hypothetical NGO example](https://www.kaggle.com/ankandash/pca-k-means-clustering-hierarchical-clustering) တို့ဖြစ်ပါသည်။

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**ပြောကြားချက်**
ဤစာတမ်းကို AI ဘာသာပြန်ဝန်ဆောင်မှု [Co-op Translator](https://github.com/Azure/co-op-translator) အသုံးပြု၍ ဘာသာပြန်ထားပါသည်။ ကျွန်ုပ်တို့သည် တိကျမှန်ကန်မှုအတွက် ကြိုးပမ်းနေသော်လည်း၊ စက်ကိရိယာဘာသာပြန်ခြင်းများတွင် အမှားများ သို့မဟုတ် မှားယွင်းချက်များ ပါဝင်နိုင်ကြောင်း သတိပြုပါရန် လိုအပ်ပါသည်။ မူလစာတမ်းကို မူရင်းဘာသာဖြင့်သာ ယုံကြည်စိတ်ချရသော အချက်အလက်အဖြစ် သတ်မှတ်သင့်သည်။ အရေးကြီးသည့် သတင်းအချက်အလက်များအတွက် ပရော်ဖက်ရှင်နယ် လူသားဘာသာပြန်သူဝန်ဆောင်မှုကို အကြံပြုပါသည်။ ဤဘာသာပြန်ချက်ကို အသုံးပြုခြင်းမှ ဖြစ်ပေါ်လာသော နားလည်မှုကွာခြားမှုများ သို့မဟုတ် မမှန်ကန်သော အသုံးပြုမှုများအတွက် ကျွန်ုပ်တို့ တာဝန်မခံပါ။
<!-- CO-OP TRANSLATOR DISCLAIMER END -->