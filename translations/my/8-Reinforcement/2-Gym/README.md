<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "107d5bb29da8a562e7ae72262d251a75",
  "translation_date": "2025-09-05T13:49:26+00:00",
  "source_file": "8-Reinforcement/2-Gym/README.md",
  "language_code": "my"
}
-->
# ကတ်ပိုးလ် စကိတ်စီးခြင်း

ယခင်သင်ခန်းစာတွင် ကျွန်ုပ်တို့ဖြေရှင်းခဲ့သော ပြဿနာသည် အလွန်ရိုးရှင်းသော ပြဿနာတစ်ခုဖြစ်ပြီး၊ အမှန်တကယ်ဘဝတွင် အသုံးမဝင်နိုင်သလို ထင်ရနိုင်ပါသည်။ သို့သော် အမှန်တကယ်ဘဝရှိ ပြဿနာများစွာသည်လည်း ယင်းအခြေအနေနှင့် ဆင်တူပါသည် - ဥပမာ Chess သို့မဟုတ် Go ကစားခြင်းတို့ပါဝင်သည်။ ၎င်းတို့သည် ဆင်တူသည်မှာ ကျွန်ုပ်တို့တွင် စည်းမျဉ်းများနှင့် **Discrete State** ပါဝင်သော ဘုတ်အဖွဲ့တစ်ခုရှိသောကြောင့်ဖြစ်သည်။

## [Pre-lecture quiz](https://ff-quizzes.netlify.app/en/ml/)

## အကျဉ်းချုပ်

ဒီသင်ခန်းစာမှာ ကျွန်ုပ်တို့ Q-Learning ၏ အခြေခံအချက်များကို **Continuous State** (တစ်ခု သို့မဟုတ် တစ်ခုထက်ပိုသော အမှန်တကယ်ဂဏန်းများဖြင့် ဖော်ပြထားသော state) တွင် အသုံးပြုမည်ဖြစ်သည်။ ကျွန်ုပ်တို့သည် အောက်ပါပြဿနာကို ကိုင်တွယ်မည်ဖြစ်သည် -

> **ပြဿနာ**: Peter သည် ဝက်ဝံမှ လွတ်မြောက်ရန် ပိုမိုမြန်ဆန်စွာ ရွေ့လျားနိုင်ရမည်။ Q-Learning ကို အသုံးပြု၍ Peter သည် စကိတ်စီးခြင်းကို, အထူးသဖြင့် တိုင်းတာမှုကို ထိန်းသိမ်းရန် လေ့လာနိုင်ပုံကို ကြည့်ရှုမည်။

![The great escape!](../../../../8-Reinforcement/2-Gym/images/escape.png)

> ဝက်ဝံမှ လွတ်မြောက်ရန် Peter နှင့် သူ၏ မိတ်ဆွေများသည် ဖန်တီးမှုအပြည့်အဝဖြစ်နေသည်! ပုံ - [Jen Looper](https://twitter.com/jenlooper)

ကျွန်ုပ်တို့သည် **CartPole** ပြဿနာအဖြစ် သိရှိထားသော တိုင်းတာမှုကို ထိန်းသိမ်းရန် ရိုးရှင်းသော ဗားရှင်းကို အသုံးပြုမည်။ CartPole ကမ္ဘာတွင်, ကျွန်ုပ်တို့တွင် ဘယ်ဘက် သို့မဟုတ် ညာဘက်သို့ ရွေ့လျားနိုင်သော အလျားလိုက် slider တစ်ခုရှိပြီး, ရည်မှန်းချက်မှာ slider အပေါ်တွင် တည်နေသော တိုင်တစ်ခုကို ထိန်းသိမ်းရန်ဖြစ်သည်။

## လိုအပ်ချက်များ

ဒီသင်ခန်းစာတွင်, **OpenAI Gym** ဟုခေါ်သော စာကြည့်တိုက်ကို အသုံးပြု၍ အမျိုးမျိုးသော **ပတ်ဝန်းကျင်များ** ကို အတုယူမည်။ သင်ဤသင်ခန်းစာ၏ ကုဒ်ကို ဒေသတွင်းတွင် (ဥပမာ Visual Studio Code မှ) အလုပ်လည်နိုင်ပြီး, simulation သည် ပြတင်းပေါ်တွင် ဖွင့်လှစ်မည်ဖြစ်သည်။ အွန်လိုင်းတွင် ကုဒ်ကို အလုပ်လည်စေရန်, [ဒီမှာ](https://towardsdatascience.com/rendering-openai-gym-envs-on-binder-and-google-colab-536f99391cc7) ဖော်ပြထားသည့်အတိုင်း အချို့ပြင်ဆင်မှုများ လိုအပ်နိုင်သည်။

## OpenAI Gym

ယခင်သင်ခန်းစာတွင်, ကစားပွဲ၏ စည်းမျဉ်းများနှင့် state ကို ကျွန်ုပ်တို့ကိုယ်တိုင် သတ်မှတ်ထားသော `Board` class မှပေးထားသည်။ ဒီနေရာမှာတော့, **simulation environment** အထူးတစ်ခုကို အသုံးပြုမည်ဖြစ်ပြီး, ၎င်းသည် တိုင်ကို ထိန်းသိမ်းရန် ရူပဗေဒကို အတုယူမည်။ reinforcement learning algorithm များကို လေ့ကျင့်ရန် အများဆုံးအသုံးပြုသော simulation environment တစ်ခုမှာ [Gym](https://gym.openai.com/) ဖြစ်ပြီး, ၎င်းကို [OpenAI](https://openai.com/) မှ ထိန်းသိမ်းထားသည်။ Gym ကို အသုံးပြုခြင်းဖြင့်, cartpole simulation မှ Atari ဂိမ်းများအထိ အမျိုးမျိုးသော **ပတ်ဝန်းကျင်များ** ကို ဖန်တီးနိုင်သည်။

> **မှတ်ချက်**: OpenAI Gym မှ ရရှိနိုင်သော အခြားသော ပတ်ဝန်းကျင်များကို [ဒီမှာ](https://gym.openai.com/envs/#classic_control) ကြည့်ရှုနိုင်သည်။

ပထမဦးစွာ, gym ကို install လုပ်ပြီး လိုအပ်သော စာကြည့်တိုက်များကို import လုပ်ပါ (code block 1):

```python
import sys
!{sys.executable} -m pip install gym 

import gym
import matplotlib.pyplot as plt
import numpy as np
import random
```

## လေ့ကျင့်မှု - cartpole ပတ်ဝန်းကျင်ကို initialize လုပ်ပါ

Cartpole တိုင်ထိန်းသိမ်းမှု ပြဿနာကို ကိုင်တွယ်ရန်, သက်ဆိုင်ရာ ပတ်ဝန်းကျင်ကို initialize လုပ်ရန် လိုအပ်သည်။ ပတ်ဝန်းကျင်တစ်ခုစီသည် အောက်ပါအရာများနှင့် ဆက်စပ်သည် -

- **Observation space**: ပတ်ဝန်းကျင်မှ ရရှိသော အချက်အလက်၏ ဖွဲ့စည်းမှုကို သတ်မှတ်သည်။ Cartpole ပြဿနာအတွက်, တိုင်၏ တည်နေရာ, အရှိန်နှင့် အခြားတန်ဖိုးများကို ရရှိသည်။

- **Action space**: ဖြစ်နိုင်သော လုပ်ဆောင်မှုများကို သတ်မှတ်သည်။ ကျွန်ုပ်တို့၏ အမှုအတွက်, action space သည် discrete ဖြစ်ပြီး, **ဘယ်ဘက်** နှင့် **ညာဘက်** ဆိုသော လုပ်ဆောင်မှုနှစ်ခုပါဝင်သည်။ (code block 2)

1. Initialize လုပ်ရန်, အောက်ပါကုဒ်ကို ရိုက်ထည့်ပါ:

    ```python
    env = gym.make("CartPole-v1")
    print(env.action_space)
    print(env.observation_space)
    print(env.action_space.sample())
    ```

ပတ်ဝန်းကျင်သည် မည်သို့အလုပ်လုပ်သည်ကို ကြည့်ရန်, 100 ခြေလှမ်းအတွက် simulation တစ်ခုကို အတိုချုပ်ပြုလုပ်ပါ။ ခြေလှမ်းတစ်ခုစီတွင်, ကျွန်ုပ်တို့သည် လုပ်ဆောင်ရန် လုပ်ဆောင်မှုတစ်ခုကို ပေးရမည် - ဒီ simulation တွင် ကျွန်ုပ်တို့သည် `action_space` မှ လုပ်ဆောင်မှုတစ်ခုကို အလွတ်ရွေးချယ်သည်။

1. အောက်ပါကုဒ်ကို အလုပ်လည်စေပြီး, ၎င်းက ဘာဖြစ်လာမည်ကို ကြည့်ပါ။

    ✅ ဒီကုဒ်ကို ဒေသတွင်း Python installation တွင် အလုပ်လည်စေရန် အကြံပြုသည်! (code block 3)

    ```python
    env.reset()
    
    for i in range(100):
       env.render()
       env.step(env.action_space.sample())
    env.close()
    ```

    သင်သည် အောက်ပါပုံနှင့် ဆင်တူသော အရာကို မြင်ရမည် -

    ![non-balancing cartpole](../../../../8-Reinforcement/2-Gym/images/cartpole-nobalance.gif)

1. Simulation အတွင်း, လုပ်ဆောင်ရန် ဆုံးဖြတ်ရန် observation များကို ရယူရန် လိုအပ်သည်။ အမှန်တကယ်, step function သည် လက်ရှိ observation များ, reward function နှင့် simulation ကို ဆက်လက်လုပ်ဆောင်ရန် make sense ဖြစ်မဖြစ်ကို ပြသသော done flag ကို ပြန်ပေးသည်။ (code block 4)

    ```python
    env.reset()
    
    done = False
    while not done:
       env.render()
       obs, rew, done, info = env.step(env.action_space.sample())
       print(f"{obs} -> {rew}")
    env.close()
    ```

    သင်သည် notebook output တွင် အောက်ပါအတိုင်း အရာတစ်ခုကို မြင်ရမည် -

    ```text
    [ 0.03403272 -0.24301182  0.02669811  0.2895829 ] -> 1.0
    [ 0.02917248 -0.04828055  0.03248977  0.00543839] -> 1.0
    [ 0.02820687  0.14636075  0.03259854 -0.27681916] -> 1.0
    [ 0.03113408  0.34100283  0.02706215 -0.55904489] -> 1.0
    [ 0.03795414  0.53573468  0.01588125 -0.84308041] -> 1.0
    ...
    [ 0.17299878  0.15868546 -0.20754175 -0.55975453] -> 1.0
    [ 0.17617249  0.35602306 -0.21873684 -0.90998894] -> 1.0
    ```

    Simulation ၏ ခြေလှမ်းတစ်ခုစီတွင် ပြန်ပေးသော observation vector တွင် အောက်ပါတန်ဖိုးများပါဝင်သည် -
    - ကတ်၏ တည်နေရာ
    - ကတ်၏ အရှိန်
    - တိုင်၏ ထောင့်
    - တိုင်၏ လည်ပတ်နှုန်း

1. အဲဒီတန်ဖိုးများ၏ အနိမ့်နှင့် အမြင့်တန်ဖိုးကို ရယူပါ - (code block 5)

    ```python
    print(env.observation_space.low)
    print(env.observation_space.high)
    ```

    သင်သည် simulation ခြေလှမ်းတစ်ခုစီတွင် reward တန်ဖိုးသည် အမြဲ 1 ဖြစ်နေသည်ကိုလည်း သတိပြုမိနိုင်သည်။ ၎င်းသည် ကျွန်ုပ်တို့၏ ရည်မှန်းချက်မှာ အချိန်အတော်ကြာအထိ အသက်ရှင်နေခြင်းဖြစ်သောကြောင့်ဖြစ်သည်၊ အတည်တကျ တိုင်ကို တည်နေရာမှန်ကန်စွာ ထိန်းသိမ်းထားရန်ဖြစ်သည်။

    ✅ အမှန်တကယ်, CartPole simulation ကို 100 ကြိမ်ဆက်တိုက် စမ်းသပ်မှုများအတွင်း 195 အထက်ရှိ ပျမ်းမျှ reward ရရှိနိုင်ပါက ဖြေရှင်းပြီးဖြစ်သည်ဟု သတ်မှတ်သည်။

## State Discretization

Q-Learning တွင်, state တစ်ခုစီတွင် ဘာလုပ်ရမည်ကို သတ်မှတ်ထားသော Q-Table တစ်ခုကို တည်ဆောက်ရန် လိုအပ်သည်။ ၎င်းကို ပြုလုပ်နိုင်ရန်, state သည် **discrete** ဖြစ်ရမည်၊ ပိုမိုတိကျစွာဆိုရမည်ဆိုပါက, ၎င်းသည် discrete တန်ဖိုးများ အနည်းငယ်သာ ပါဝင်ရမည်။ ထို့ကြောင့်, observation များကို **discretize** ပြုလုပ်ပြီး, ၎င်းတို့ကို discrete state များအဖြစ် သတ်မှတ်ရန် လိုအပ်သည်။

ဒီအတွက် ကျွန်ုပ်တို့မှာ အနည်းဆုံးနည်းလမ်းနှစ်ခုရှိသည် -

- **Bins အဖြစ် ခွဲခြားခြင်း**: တန်ဖိုးတစ်ခု၏ အကွာအဝေးကို သိပါက, ၎င်းကို **bins** အရေအတွက်တစ်ခုအဖြင့် ခွဲခြားနိုင်ပြီး, ထို့နောက် ၎င်းတန်ဖိုးကို ၎င်းပါဝင်သော bin အမှတ်ဖြင့် အစားထိုးနိုင်သည်။ ၎င်းကို numpy [`digitize`](https://numpy.org/doc/stable/reference/generated/numpy.digitize.html) method ကို အသုံးပြု၍ ပြုလုပ်နိုင်သည်။ 

✅ Linear interpolation ကို အသုံးပြု၍ တန်ဖိုးများကို သတ်မှတ်ထားသော အကွာအဝေး (ဥပမာ -20 မှ 20) သို့ ဆွဲဆောင်ပြီး, ထို့နောက် အနီးဆုံး integer သို့ ပြောင်းနိုင်သည်။ 

ဒီဥပမာတွင်, ကျွန်ုပ်တို့သည် ဒုတိယနည်းလမ်းကို အသုံးပြုမည်။ ၎င်းကို နောက်ပိုင်းတွင် သတိပြုမိနိုင်သလို, အကွာအဝေးမသတ်မှတ်ထားသော တန်ဖိုးများသည် အလွန်ရှားပါးသောကြောင့်, အလွန်ရှားပါးသော state များသာ ဖြစ်ပေါ်မည်။

1. ကျွန်ုပ်တို့၏ မော်ဒယ်မှ observation ကို ယူပြီး, 4 ခုသော integer တန်ဖိုးများ၏ tuple ကို ထုတ်ပေးမည့် function ကို အောက်တွင် ဖော်ပြထားသည် - (code block 6)

    ```python
    def discretize(x):
        return tuple((x/np.array([0.25, 0.25, 0.01, 0.1])).astype(np.int))
    ```

1. Bins အသုံးပြု၍ discretization နည်းလမ်းတစ်ခုကိုလည်း စမ်းကြည့်ပါ - (code block 7)

    ```python
    def create_bins(i,num):
        return np.arange(num+1)*(i[1]-i[0])/num+i[0]
    
    print("Sample bins for interval (-5,5) with 10 bins\n",create_bins((-5,5),10))
    
    ints = [(-5,5),(-2,2),(-0.5,0.5),(-2,2)] # intervals of values for each parameter
    nbins = [20,20,10,10] # number of bins for each parameter
    bins = [create_bins(ints[i],nbins[i]) for i in range(4)]
    
    def discretize_bins(x):
        return tuple(np.digitize(x[i],bins[i]) for i in range(4))
    ```

1. ယခု simulation အတိုချုပ်တစ်ခုကို အလုပ်လည်စေပြီး, discrete environment တန်ဖိုးများကို ကြည့်ရှုပါ။ `discretize` နှင့် `discretize_bins` နှစ်ခုစလုံးကို စမ်းကြည့်ပြီး, မည်သည့်ကွာခြားချက်ရှိသည်ကို ကြည့်ပါ။

    ✅ `discretize_bins` သည် bin အမှတ်ကို ပြန်ပေးသည်, ၎င်းသည် 0-based ဖြစ်သည်။ `discretize` တွင်, output တန်ဖိုးများ၏ အကွာအဝေးကို ဂရုမစိုက်ဘဲ, 0 သည် 0 ကို ကိုယ်စားပြုသည်။ (code block 8)

    ```python
    env.reset()
    
    done = False
    while not done:
       #env.render()
       obs, rew, done, info = env.step(env.action_space.sample())
       #print(discretize_bins(obs))
       print(discretize(obs))
    env.close()
    ```

    ✅ ပတ်ဝန်းကျင်အလုပ်လုပ်ပုံကို ကြည့်လိုပါက, `env.render` ဖြင့် စတန်းထားသောလိုင်းကို uncomment လုပ်ပါ။ 

## Q-Table ဖွဲ့စည်းမှု

ယခင်သင်ခန်းစာတွင်, state သည် 0 မှ 8 အတွင်းရှိ နံပါတ်နှစ်ခုဖြစ်ပြီး, Q-Table ကို 8x8x2 အရွယ်အစားရှိ numpy tensor ဖြင့် ကိုယ်စားပြုရန် အဆင်ပြေသည်။ Bins discretization ကို အသုံးပြုပါက, state vector ၏ အရွယ်အစားကိုလည်း သိနိုင်ပြီး, 20x20x10x10x2 အရွယ်အစားရှိ array ဖြင့် state ကို ကိုယ်စားပြုနိုင်သည်။

သို့သော်, observation space ၏ တိကျသော အရွယ်အစားကို မသိနိုင်သောအခါလည်း ရှိသည်။ `discretize` function ၏ အမှုအတွက်, state သည် သတ်မှတ်ထားသော အကွာအဝေးအတွင်းသာ ရှိနေမည်ဟု သေချာမရနိုင်ပါ။ ထို့ကြောင့်, Q-Table ကို dictionary ဖြင့် ကိုယ်စားပြုမည်။

1. *(state, action)* ကို dictionary key အဖြစ် အသုံးပြုပြီး, value သည် Q-Table entry value ကို ကိုယ်စားပြုမည်။ (code block 9)

    ```python
    Q = {}
    actions = (0,1)
    
    def qvalues(state):
        return [Q.get((state,a),0) for a in actions]
    ```

    ဒီနေရာမှာ, `qvalues()` function ကိုလည်း သတ်မှတ်ထားပြီး, state တစ်ခုအတွက် Q-Table တန်ဖိုးများကို ပြန်ပေးသည်။

## Q-Learning စတင်ကြရအောင်

ယခု Peter ကို တိုင်ထိန်းသိမ်းရန် သင်ကြားရန် ပြင်ဆင်ပြီးပြီ!

1. ပထမဦးစွာ, အချို့သော hyperparameters ကို သတ်မှတ်ပါ - (code block 10)

    ```python
    # hyperparameters
    alpha = 0.3
    gamma = 0.9
    epsilon = 0.90
    ```

    ဒီနေရာမှာ, `alpha` သည် **learning rate** ဖြစ်ပြီး, Q-Table ၏ လက်ရှိတန်ဖိုးများကို ချိန်ညှိရန် အတိုင်းအတာကို သတ်မှတ်သည်။ `gamma` သည် **discount factor** ဖြစ်ပြီး, အနာဂတ် reward ကို ပိုမိုဦးစားပေးရန် သတ်မှတ်သည်။ `epsilon` သည် **exploration/exploitation factor** ဖြစ်ပြီး, random action နှင့် Q-Table အရ next action ကို ရွေးချယ်ရန် ရွေးချယ်မှုကို သတ်မှတ်သည်။

    ✅ Balancing အရ random action (exploration) သည် မှားယွင်းသော ဦးတည်ချက်သို့ random punch တစ်ခုအဖြစ် လုပ်ဆောင်မည်။

### Algorithm ကို တိုးတက်အောင် ပြုလုပ်ပါ

ယခင်သင်ခန်းစာမှ algorithm ကို အောက်ပါအတိုင်း တိုးတက်အောင် ပြုလုပ်နိုင်သည် -

- **ပျမ်းမျှ cumulative reward ကိုတွက်ချက်ပါ**: 5000 ကြိမ် iteration အတွင်း cumulative reward ကို ပျမ်းမျှတွက်ချက်ပြီး, 195 အထက်ရရှိပါက, ပြဿနာကို ဖြေရှင်းပြီးဖြစ်သည်ဟု သတ်မှတ်နိုင်သည်။

- **အများဆုံး cumulative result ကိုတွက်ချက်ပါ**: `Qmax` ကိုတွက်ချက်ပြီး, Q-Table ၏ အကောင်းဆုံးမော်ဒယ်ကို သိမ်းဆည်းပါ။

1. Simulation တစ်ခုစီတွင် cumulative rewards ကို `rewards` vector တွင် စုဆောင်းပါ - (code block 11)

    ```python
    def probs(v,eps=1e-4):
        v = v-v.min()+eps
        v = v/v.sum()
        return v
    
    Qmax = 0
    cum_rewards = []
    rewards = []
    for epoch in range(100000):
        obs = env.reset()
        done = False
        cum_reward=0
        # == do the simulation ==
        while not done:
            s = discretize(obs)
            if random.random()<epsilon:
                # exploitation - chose the action according to Q-Table probabilities
                v = probs(np.array(qvalues(s)))
                a = random.choices(actions,weights=v)[0]
            else:
                # exploration - randomly chose the action
                a = np.random.randint(env.action_space.n)
    
            obs, rew, done, info = env.step(a)
            cum_reward+=rew
            ns = discretize(obs)
            Q[(s,a)] = (1 - alpha) * Q.get((s,a),0) + alpha * (rew + gamma * max(qvalues(ns)))
        cum_rewards.append(cum_reward)
        rewards.append(cum_reward)
        # == Periodically print results and calculate average reward ==
        if epoch%5000==0:
            print(f"{epoch}: {np.average(cum_rewards)}, alpha={alpha}, epsilon={epsilon}")
            if np.average(cum_rewards) > Qmax:
                Qmax = np.average(cum_rewards)
                Qbest = Q
            cum_rewards=[]
    ```

ဒီရလဒ်များမှ သတိပြုနိုင်သောအရာများ -

- **ရည်မှန်းချက်အနီး**: 195 cumulative rewards ရရှိရန် အနီးကပ်ဖြစ်နေသည်။
  
- **Reward ကျဆင်းမှု**: Reward ကျဆင်းမှုသည် Q-Table ၏ ရှိပြီးသားတန်ဖိုးများကို ပျက်စီးစေနိုင်သည်။

ဒီအချက်ကို training progress ကို plot လုပ်ပါက ပိုမိုရှင်းလင်းစေမည်။

## Training Progress ကို Plot လုပ်ခြင်း

Training အတွင်း, cumulative reward တန်ဖိုးကို `rewards` vector တွင် စုဆောင်းထားသည်။ Iteration နံပါတ်နှင့်အတူ plot လုပ်ပါက -

```python
plt.plot(rewards)
```

![raw progress](../../../../8-Reinforcement/2-Gym/images/train_progress_raw.png)

ဒီ graph မှ အချက်အလက်မရနိုင်ပါ, stochastic training process ၏ သဘာဝကြောင့် ဖြစ်သည်။ 

Running average ကိုတွက်ချက်ပြီး, graph ကို ပိုမိုရှင်းလင်းစေရန် -

```python
def running_average(x,window):
    return np.convolve(x,np.ones(window)/window,mode='valid')

plt.plot(running_average(rewards,100))
```

![training progress](../../../../8-Reinforcement/2-Gym/images/train_progress_runav.png)

## Hyperparameters ကို ပြောင်းလဲခြင်း

Learning ကို ပိုမိုတည်ငြိမ်စေရန်, training အတွင်း hyperparameters အချို့ကို ပြောင်းလဲရန် make sense ဖြစ်သည်။ အထူးသဖြင့် -

- **Learning rate (`alpha`)**: 1 အနီးမှ စတင်ပြီး, အချိန်နှင့်အမျှ တဖြည်းဖြည်းလျော့ချနိုင်သည်။

- **Epsilon ကို တိုးချဲ့ပါ**: Exploration ကို လျော့ချပြီး, exploitation ကို ပိုမိုလုပ်ဆောင်ရန် `epsilon` ကို တဖြည်းဖြည်းတိုးချဲ့နိုင်သည်။
> **အလုပ် ၁**: ဟိုက်ပါပါမီတာတန်ဖိုးများကို ပြောင်းလဲကစားကြည့်ပြီး ပိုမိုမြင့်မားသော စုစုပေါင်းဆုရမှတ်ကို ရရှိနိုင်မလား ကြည့်ပါ။ သင် ၁၉၅ အထက်ရရှိနေတာရှိပါသလား?
> **Task 2**: ပြဿနာကို တရားဝင်ဖြေရှင်းနိုင်ရန် 100 ကြိမ်ဆက်တိုက် လည်ပတ်မှုများအတွက် 195 အလယ်အလတ်ဆုလာဘ်ရမှတ်ကို ရရှိရမည်။ လေ့ကျင့်မှုအတွင်း၌ ထိုရလဒ်ကိုတိုင်းတာပြီး ပြဿနာကို တရားဝင်ဖြေရှင်းနိုင်ကြောင်း သေချာစေပါ။

## ရလဒ်ကို လက်တွေ့ကြည့်ရှုခြင်း

လေ့ကျင့်ပြီးသော မော်ဒယ်၏ အပြုအမူကို လက်တွေ့ကြည့်ရှုရတာ စိတ်ဝင်စားဖွယ်ဖြစ်မည်။ စမ်းသပ်မှုကို လည်ပတ်ပြီး Q-Table အတွင်းရှိ probability distribution အတိုင်း လေ့ကျင့်မှုအတွင်း အသုံးပြုခဲ့သည့် အလားတူသော လုပ်ဆောင်မှုရွေးချယ်မှု မဟာဗျူဟာကို လိုက်နာကြည့်ပါ။ (code block 13)

```python
obs = env.reset()
done = False
while not done:
   s = discretize(obs)
   env.render()
   v = probs(np.array(qvalues(s)))
   a = random.choices(actions,weights=v)[0]
   obs,_,done,_ = env.step(a)
env.close()
```

သင်သည် အောက်ပါအတိုင်း တစ်စုံတစ်ရာကို မြင်ရမည်ဖြစ်သည်-

![a balancing cartpole](../../../../8-Reinforcement/2-Gym/images/cartpole-balance.gif)

---

## 🚀စိန်ခေါ်မှု

> **Task 3**: ဒီနေရာမှာတော့ Q-Table ၏ နောက်ဆုံးမိတ္တူကို အသုံးပြုခဲ့ပြီးဖြစ်သည်၊ ဒါပေမယ့် အကောင်းဆုံးမဟုတ်နိုင်ပါ။ သတိပြုပါ၊ အကောင်းဆုံးလုပ်ဆောင်မှုရ Q-Table ကို `Qbest` variable ထဲသို့ သိမ်းဆည်းထားပြီးဖြစ်သည်! `Qbest` ကို `Q` ထဲသို့ ကူးယူပြီး အကောင်းဆုံး Q-Table ဖြင့် ထိုနမူနာကို ထပ်မံစမ်းသပ်ကြည့်ပါ၊ ကွာခြားချက်ကို သတိပြုမိပါက မှတ်သားပါ။

> **Task 4**: ဒီနေရာမှာတော့ အဆင့်တိုင်းမှာ အကောင်းဆုံးလုပ်ဆောင်မှုကို ရွေးချယ်ခြင်းမဟုတ်ဘဲ၊ probability distribution နှင့် ကိုက်ညီသော လုပ်ဆောင်မှုကို ရွေးချယ်ခဲ့ပါသည်။ Q-Table ရဲ့ အမြင့်ဆုံးတန်ဖိုးကို ကိုယ်စားပြုသော လုပ်ဆောင်မှုကို အမြဲရွေးချယ်ခြင်းက ပို make sense ဖြစ်မည်မဟုတ်လား? ဒါကို `np.argmax` function ကို အသုံးပြု၍ Q-Table ရဲ့ အမြင့်ဆုံးတန်ဖိုးနှင့် ကိုက်ညီသော လုပ်ဆောင်မှုနံပါတ်ကို ရှာဖွေပြီး အကောင်အထည်ဖော်နိုင်သည်။ ဒီမဟာဗျူဟာကို အကောင်အထည်ဖော်ပြီး balancing ကို တိုးတက်စေမလား စမ်းသပ်ကြည့်ပါ။

## [Post-lecture quiz](https://ff-quizzes.netlify.app/en/ml/)

## လုပ်ငန်းတာဝန်
[Train a Mountain Car](assignment.md)

## နိဂုံးချုပ်

ယခုအခါတွင် သင်သည် ဆုလာဘ် function တစ်ခုကို ပေးခြင်းဖြင့်၊ ဂိမ်း၏ လိုအပ်သောအခြေအနေကို သတ်မှတ်ပေးခြင်းဖြင့်၊ ထိုအခြေအနေကို ရှာဖွေစူးစမ်းရန် အခွင့်အရေးပေးခြင်းဖြင့် အေးဂျင့်များကို ကောင်းမွန်သောရလဒ်များရရှိအောင် လေ့ကျင့်ပေးနိုင်သည်ကို သင်ယူပြီးဖြစ်သည်။ Q-Learning algorithm ကို discrete နှင့် continuous environment များတွင်၊ သို့သော် discrete actions များဖြင့် အောင်မြင်စွာ အသုံးပြုနိုင်ခဲ့ပါသည်။

သို့သော်လည်း၊ action state သည် continuous ဖြစ်သည့်အခြေအနေများနှင့် observation space သည် ပိုမိုရှုပ်ထွေးသော အခြေအနေများကိုလည်း လေ့လာရန် အရေးကြီးပါသည်၊ ဥပမာ Atari ဂိမ်း screen မှ ရရှိသော ပုံရိပ်များကဲ့သို့။ ထိုပြဿနာများတွင် ကောင်းမွန်သောရလဒ်များရရှိရန် အားကောင်းသော machine learning နည်းလမ်းများ၊ neural networks ကဲ့သို့သော နည်းလမ်းများကို အသုံးပြုရန် လိုအပ်ပါသည်။ ထို advanced အကြောင်းအရာများသည် ကျွန်ုပ်တို့၏ လာမည့် အဆင့်မြင့် AI သင်တန်း၏ အကြောင်းအရာများဖြစ်ပါသည်။

---

**ဝက်ဘ်ဆိုက်မှတ်ချက်**:  
ဤစာရွက်စာတမ်းကို AI ဘာသာပြန်ဝန်ဆောင်မှု [Co-op Translator](https://github.com/Azure/co-op-translator) ကို အသုံးပြု၍ ဘာသာပြန်ထားပါသည်။ ကျွန်ုပ်တို့သည် တိကျမှန်ကန်မှုအတွက် ကြိုးစားနေသော်လည်း၊ အလိုအလျောက်ဘာသာပြန်မှုများတွင် အမှားများ သို့မဟုတ် မတိကျမှုများ ပါဝင်နိုင်သည်ကို ကျေးဇူးပြု၍ သိရှိထားပါ။ မူရင်းဘာသာစကားဖြင့် ရေးသားထားသော စာရွက်စာတမ်းကို အာဏာတည်သော ရင်းမြစ်အဖြစ် သတ်မှတ်သင့်ပါသည်။ အရေးကြီးသော အချက်အလက်များအတွက် လူသားပညာရှင်များမှ ဘာသာပြန်ခြင်းကို အကြံပြုပါသည်။ ဤဘာသာပြန်မှုကို အသုံးပြုခြင်းမှ ဖြစ်ပေါ်လာသော နားလည်မှုမှားများ သို့မဟုတ် အဓိပ္ပာယ်မှားများအတွက် ကျွန်ုပ်တို့သည် တာဝန်မယူပါ။