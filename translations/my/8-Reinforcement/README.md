<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "20ca019012b1725de956681d036d8b18",
  "translation_date": "2025-09-05T13:32:51+00:00",
  "source_file": "8-Reinforcement/README.md",
  "language_code": "my"
}
-->
# reinforcement learning အကြောင်းမိတ်ဆက်

Reinforcement learning (RL) သည် supervised learning နှင့် unsupervised learning အနောက်တွင် machine learning ရဲ့ အခြေခံ paradigm တစ်ခုအဖြစ်လေ့လာခံရသော နည်းလမ်းတစ်ခုဖြစ်သည်။ RL သည် ဆုံးဖြတ်ချက်များနှင့်ပတ်သက်ပြီးဖြစ်သည်။ မှန်ကန်သော ဆုံးဖြတ်ချက်များပေးခြင်း သို့မဟုတ် အနည်းဆုံး အဲဒီဆုံးဖြတ်ချက်များမှ သင်ယူခြင်းကို အဓိကထားသည်။

သင် stock market ကဲ့သို့သော simulation environment တစ်ခုရှိသည်ဟု စဉ်းစားပါ။ သတ်မှတ်ထားသော regulation တစ်ခုကို ထည့်သွင်းလိုက်ရင် ဘာဖြစ်မလဲ? အဲဒါက အကျိုးသက်ရောက်မှုက အကောင်းတစ်ခုလား၊ အဆိုးတစ်ခုလား? အဆိုးတစ်ခုဖြစ်လာရင် _negative reinforcement_ ကို သင်ယူပြီး လမ်းကြောင်းပြောင်းဖို့လိုအပ်သည်။ အကောင်းတစ်ခုဖြစ်လာရင် _positive reinforcement_ ကို အခြေခံပြီး ဆက်လက်တိုးတက်ဖို့လိုအပ်သည်။

![peter and the wolf](../../../8-Reinforcement/images/peter.png)

> Peter နဲ့ သူ့မိတ်ဆွေတွေဟာ ဝက်ခြံဆာတဲ့ ဝက်ကို လွတ်မြောက်ဖို့ လိုအပ်ပါတယ်! [Jen Looper](https://twitter.com/jenlooper) ရဲ့ ပုံ

## ဒေသဆိုင်ရာအကြောင်းအရာ: Peter and the Wolf (ရုရှား)

[Peter and the Wolf](https://en.wikipedia.org/wiki/Peter_and_the_Wolf) သည် ရုရှား composer [Sergei Prokofiev](https://en.wikipedia.org/wiki/Sergei_Prokofiev) ရေးသားထားသော ဂီတပုံပြင်တစ်ခုဖြစ်သည်။ အဲဒါက သူရဲကောင်းလေး Peter ရဲ့ အကြောင်းဖြစ်ပြီး သူဟာ ဝက်ကို လိုက်ဖမ်းဖို့ အိမ်ကနေ တောထဲကို သွားတဲ့ ပုံပြင်ဖြစ်သည်။ ဒီအပိုင်းမှာ Peter ကို အကူအညီပေးမယ့် machine learning algorithm တွေကို သင်ကြမယ်။

- **ရှာဖွေ** ပတ်ဝန်းကျင်ကို လေ့လာပြီး အကောင်းဆုံး navigation map တစ်ခုတည်ဆောက်ရန်
- **သင်ယူ** skateboard ကို အသုံးပြုနည်းနဲ့ balance လုပ်နည်းကို သင်ယူပြီး ပိုမြန်မြန်ရွေ့လျားနိုင်ရန်

[![Peter and the Wolf](https://img.youtube.com/vi/Fmi5zHg4QSM/0.jpg)](https://www.youtube.com/watch?v=Fmi5zHg4QSM)

> 🎥 Prokofiev ရဲ့ Peter and the Wolf ကို နားဆင်ဖို့ အထက်ပါပုံကို နှိပ်ပါ

## reinforcement learning

အရင်အပိုင်းတွေမှာ machine learning problem နှစ်ခုကို တွေ့မြင်ခဲ့ပါပြီ။

- **Supervised learning** သည် သင်လိုက်ဖျက်လိုသော ပြဿနာအတွက် နမူနာဖြေရှင်းချက်များကို အကြံပြုသော dataset များကို အသုံးပြုသည်။ [Classification](../4-Classification/README.md) နဲ့ [regression](../2-Regression/README.md) သည် supervised learning task များဖြစ်သည်။
- **Unsupervised learning** သည် labeled training data မရှိသော နည်းလမ်းဖြစ်သည်။ Unsupervised learning ရဲ့ အဓိကဥပမာမှာ [Clustering](../5-Clustering/README.md) ဖြစ်သည်။

ဒီအပိုင်းမှာ labeled training data မလိုအပ်တဲ့ learning problem အသစ်တစ်ခုကို မိတ်ဆက်ပေးပါမယ်။ ဒီလို problem တွေမှာ အမျိုးအစားအများကြီးရှိပါတယ်။

- **[Semi-supervised learning](https://wikipedia.org/wiki/Semi-supervised_learning)** သည် unlabeled data များစွာကို model ကို pre-train လုပ်ရန် အသုံးပြုနိုင်သည်။
- **[Reinforcement learning](https://wikipedia.org/wiki/Reinforcement_learning)** သည် agent တစ်ခုက simulated environment တစ်ခုမှာ စမ်းသပ်မှုများလုပ်ပြီး အပြုအမူကို သင်ယူသည်။

### ဥပမာ - ကွန်ပျူတာဂိမ်း

ကွန်ပျူတာကို chess သို့မဟုတ် [Super Mario](https://wikipedia.org/wiki/Super_Mario) ကဲ့သို့သော ဂိမ်းတစ်ခုကို ကစားဖို့ သင်ပေးချင်တယ်ဆိုပါစို့။ ကွန်ပျူတာကို ဂိမ်းကစားစေဖို့ ဂိမ်းရဲ့ state တစ်ခုစီမှာ ဘယ်လိုလှုပ်ရှားမှုကို လုပ်မလဲဆိုတာကို ခန့်မှန်းနိုင်ဖို့လိုအပ်သည်။ ဒါဟာ classification problem တစ်ခုလိုပုံရပေမယ့် အဲဒါမဟုတ်ပါဘူး - အကြောင်းက states နဲ့ အတူတူသော actions ရှိတဲ့ dataset မရှိလို့ပါ။ Chess match တွေ သို့မဟုတ် Super Mario ကစားနေတဲ့ player တွေကို record လုပ်ထားတဲ့ data ရှိနေပေမယ့် အဲဒီ data က states များစွာကို လုံလောက်စွာ မဖုံးလွှမ်းနိုင်ပါ။

ဂိမ်း data ရှာဖွေခြင်းကို မလုပ်ဘဲ **Reinforcement Learning** (RL) သည် *ကွန်ပျူတာကို အကြိမ်ကြိမ် ကစားစေပြီး ရလဒ်ကို ကြည့်ရှုခြင်း* ဆိုတဲ့ အတွေးအခေါ်ကို အခြေခံထားသည်။ ဒါကြောင့် RL ကို အသုံးပြုဖို့အတွက် အဓိကလိုအပ်ချက်နှစ်ခုရှိပါတယ်။

- **Environment** နဲ့ **Simulator** တစ်ခုလိုအပ်သည်။ ဂိမ်းကို အကြိမ်ကြိမ် ကစားနိုင်ရန် simulator က ဂိမ်းရဲ့ rule တွေ၊ state တွေ၊ action တွေကို သတ်မှတ်ပေးရမယ်။

- **Reward function** တစ်ခုလိုအပ်သည်။ အဲဒါက အကြိမ်စီမှာ သင်ဘယ်လိုလုပ်ဆောင်ခဲ့တယ်ဆိုတာကို ပြောပြပေးမယ်။

Machine learning အခြားနည်းလမ်းတွေနဲ့ RL ရဲ့ အဓိကကွာခြားချက်က RL မှာ ဂိမ်းပြီးဆုံးမှသာ အနိုင်ရ/အရှုံးပေါ်မယ်ဆိုတာကို သိနိုင်ခြင်းဖြစ်သည်။ ဒါကြောင့် move တစ်ခုတည်းက အကောင်းတစ်ခုလားဆိုတာကို မသိနိုင်ပါဘူး - ဂိမ်းပြီးဆုံးမှသာ reward ကို ရရှိနိုင်သည်။ အဲဒီလို မသေချာတဲ့အခြေအနေတွေအောက်မှာ model ကို train လုပ်နိုင်တဲ့ algorithm တွေကို ဒီဇိုင်းဆွဲဖို့ ကျွန်တော်တို့ရဲ့ ရည်မှန်းချက်ဖြစ်ပါတယ်။ ကျွန်တော်တို့ **Q-learning** ဆိုတဲ့ RL algorithm တစ်ခုကို လေ့လာပါမယ်။

## သင်ခန်းစာများ

1. [Reinforcement learning နဲ့ Q-Learning အကြောင်းမိတ်ဆက်](1-QLearning/README.md)
2. [Gym simulation environment ကို အသုံးပြုခြင်း](2-Gym/README.md)

## Credit

"Introduction to Reinforcement Learning" ကို [Dmitry Soshnikov](http://soshnikov.com) မှ ♥️ ဖြင့် ရေးသားထားသည်။

---

**အကြောင်းကြားချက်**:  
ဤစာရွက်စာတမ်းကို AI ဘာသာပြန်ဝန်ဆောင်မှု [Co-op Translator](https://github.com/Azure/co-op-translator) ကို အသုံးပြု၍ ဘာသာပြန်ထားပါသည်။ ကျွန်ုပ်တို့သည် တိကျမှုအတွက် ကြိုးစားနေသော်လည်း၊ အလိုအလျောက် ဘာသာပြန်ခြင်းတွင် အမှားများ သို့မဟုတ် မမှန်ကန်မှုများ ပါဝင်နိုင်သည်ကို သတိပြုပါ။ မူရင်းဘာသာစကားဖြင့် ရေးသားထားသော စာရွက်စာတမ်းကို အာဏာရှိသော ရင်းမြစ်အဖြစ် သတ်မှတ်သင့်ပါသည်။ အရေးကြီးသော အချက်အလက်များအတွက် လူက ဘာသာပြန်ခြင်းကို အကြံပြုပါသည်။ ဤဘာသာပြန်ကို အသုံးပြုခြင်းမှ ဖြစ်ပေါ်လာသော အလွဲအလွဲအချော်များ သို့မဟုတ် အနားယူမှုများအတွက် ကျွန်ုပ်တို့သည် တာဝန်မယူပါ။