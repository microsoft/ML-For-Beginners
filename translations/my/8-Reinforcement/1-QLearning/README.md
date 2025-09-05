<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "911efd5e595089000cb3c16fce1beab8",
  "translation_date": "2025-09-05T13:40:42+00:00",
  "source_file": "8-Reinforcement/1-QLearning/README.md",
  "language_code": "my"
}
-->
# Reinforcement Learning နှင့် Q-Learning အကြောင်းမိတ်ဆက်

![Machine Learning တွင် reinforcement အကျဉ်းချုပ်ကို sketchnote](../../../../sketchnotes/ml-reinforcement.png)
> Sketchnote by [Tomomi Imura](https://www.twitter.com/girlie_mac)

Reinforcement learning တွင် အရေးပါသော အချက် ၃ ခုရှိသည်။ အဲဒါတွေကတော့ agent, states တစ်ချို့, နှင့် state တစ်ခုစီအတွက် actions တွေဖြစ်သည်။ သတ်မှတ်ထားသော state မှာ action တစ်ခုကို လုပ်ဆောင်ခြင်းအားဖြင့် agent သည် reward ရရှိမည်။ Super Mario ကွန်ပျူတာဂိမ်းကို ထပ်မံစဉ်းစားပါ။ သင်သည် Mario ဖြစ်ပြီး၊ ဂိမ်းအဆင့်တစ်ခုတွင်၊ ချိုင့်စွန်းအနားမှာရပ်နေသည်။ သင့်အပေါ်မှာ coin တစ်ခုရှိသည်။ သင်သည် Mario ဖြစ်ပြီး၊ ဂိမ်းအဆင့်တစ်ခုတွင်၊ တိကျသောနေရာတွင် ရပ်နေသည်... ဒါက သင့် state ဖြစ်သည်။ ညာဘက်ကို တစ်ခြေလှမ်း (action) ရွှေ့ခြင်းသည် ချိုင့်စွန်းအောက်ကို ကျသွားစေပြီး၊ အနိမ့်သော ကိန်းဂဏန်း score ရရှိမည်။ သို့သော် jump ခလုတ်ကို နှိပ်ခြင်းဖြင့် သင် point ရရှိပြီး အသက်ရှင်နေမည်။ ဒါက အကောင်းဆုံးရလဒ်ဖြစ်ပြီး၊ သင့်ကို အကောင်းသော ကိန်းဂဏန်း score ပေးသင့်သည်။

Reinforcement learning နှင့် simulator (ဂိမ်း) ကို အသုံးပြုခြင်းအားဖြင့် သင်သည် အသက်ရှင်နေခြင်းနှင့် အများဆုံး point ရရှိရန် ဂိမ်းကို ကစားပုံကို သင်ယူနိုင်သည်။

[![Reinforcement Learning မိတ်ဆက်](https://img.youtube.com/vi/lDq_en8RNOo/0.jpg)](https://www.youtube.com/watch?v=lDq_en8RNOo)

> 🎥 အထက်ပါပုံကို နှိပ်ပြီး Dmitry ၏ Reinforcement Learning အကြောင်းဆွေးနွေးမှုကို ကြည့်ပါ

## [Pre-lecture quiz](https://ff-quizzes.netlify.app/en/ml/)

## အကြိုလိုအပ်ချက်များနှင့် Setup

ဒီသင်ခန်းစာမှာ Python မှာ code တွေကို စမ်းသပ်မည်ဖြစ်သည်။ သင်သည် Jupyter Notebook code ကို သင်ခန်းစာမှ သင်၏ကွန်ပျူတာတွင် သို့မဟုတ် cloud တစ်ခုခုတွင် run လို့ရနိုင်ရမည်။

[သင်ခန်းစာ notebook](https://github.com/microsoft/ML-For-Beginners/blob/main/8-Reinforcement/1-QLearning/notebook.ipynb) ကို ဖွင့်ပြီး ဒီသင်ခန်းစာကို လိုက်လျောညီထွေဖြင့် တည်ဆောက်ပါ။

> **Note:** သင်သည် ဒီ code ကို cloud မှ ဖွင့်နေပါက၊ notebook code တွင် အသုံးပြုထားသော [`rlboard.py`](https://github.com/microsoft/ML-For-Beginners/blob/main/8-Reinforcement/1-QLearning/rlboard.py) ဖိုင်ကိုလည်း fetch လုပ်ရန်လိုအပ်သည်။ notebook နှင့် တူညီသော directory တွင် ထည့်ပါ။

## မိတ်ဆက်

ဒီသင်ခန်းစာမှာ **[Peter and the Wolf](https://en.wikipedia.org/wiki/Peter_and_the_Wolf)** ၏ ကမ္ဘာကို ရုရှားတေးရေးဆရာ [Sergei Prokofiev](https://en.wikipedia.org/wiki/Sergei_Prokofiev) ၏ ဂီတပုံပြင်မှ အားပေးမှုဖြင့် ရှာဖွေမည်။ **Reinforcement Learning** ကို အသုံးပြုပြီး Peter ကို သူ့ပတ်ဝန်းကျင်ကို စူးစမ်းစေပြီး၊ အရသာရှိသောပန်းသီးများကို စုဆောင်းစေပြီး ဝံကို မတွေ့ရအောင်လုပ်မည်။

**Reinforcement Learning** (RL) သည် **agent** တစ်ခု၏ **environment** တစ်ခုတွင် အကောင်းဆုံးအပြုအမူကို သင်ယူရန် အတတ်ပညာဖြစ်သည်။ ဒီ environment တွင် agent တစ်ခုသည် **reward function** ဖြင့် သတ်မှတ်ထားသော **goal** တစ်ခုရှိရမည်။

## Environment

ရိုးရှင်းစွာပြောရမည်ဆိုပါက Peter ၏ ကမ္ဘာကို `width` x `height` အရွယ်ရှိသော စတုရန်း board အဖြစ် သတ်မှတ်ပါမည်၊ ဒီလိုပုံစံဖြစ်သည်။

![Peter ၏ Environment](../../../../8-Reinforcement/1-QLearning/images/environment.png)

ဒီ board ရဲ့ cell တစ်ခုစီမှာ အောက်ပါအတိုင်းဖြစ်နိုင်သည်-

* **ground**, Peter နှင့် အခြားသော သတ္တဝါများ လမ်းလျှောက်နိုင်သောနေရာ။
* **water**, လမ်းလျှောက်လို့မရသောနေရာ။
* **tree** သို့မဟုတ် **grass**, အနားယူနိုင်သောနေရာ။
* **apple**, Peter အတွက် အစားအသောက်အဖြစ် ရှာဖွေလိုသောနေရာ။
* **wolf**, အန္တရာယ်ရှိပြီး ရှောင်ရှားသင့်သောနေရာ။

Python module တစ်ခုဖြစ်သော [`rlboard.py`](https://github.com/microsoft/ML-For-Beginners/blob/main/8-Reinforcement/1-QLearning/rlboard.py) တွင် ဒီ environment နှင့် အလုပ်လုပ်ရန် code ပါဝင်သည်။ ဒီ code သည် ကျွန်ုပ်တို့၏ concepts ကို နားလည်ရန်အရေးကြီးမဟုတ်သောကြောင့် module ကို import လုပ်ပြီး sample board ကို ဖန်တီးရန် အသုံးပြုမည် (code block 1):

```python
from rlboard import *

width, height = 8,8
m = Board(width,height)
m.randomize(seed=13)
m.plot()
```

ဒီ code သည် အထက်ပါပုံနှင့် ဆင်တူသော environment ၏ ပုံကို print လုပ်သင့်သည်။

## Actions နှင့် Policy

ဒီဥပမာမှာ Peter ၏ ရည်မှန်းချက်မှာ ဝံနှင့် အခြားသော အတားအဆီးများကို ရှောင်ရှားပြီး ပန်းသီးကို ရှာဖွေခြင်းဖြစ်သည်။ ဒီအတွက် သူသည် ပန်းသီးကို ရှာဖွေမရအောင် လမ်းလျှောက်နိုင်သည်။

ထို့ကြောင့်၊ တည်နေရာတစ်ခုစီတွင် သူသည် အောက်ပါ action များအနက် တစ်ခုကို ရွေးချယ်နိုင်သည်- up, down, left နှင့် right။

action များကို dictionary အဖြစ် သတ်မှတ်ပြီး၊ coordinate changes နှင့် mapping လုပ်မည်။ ဥပမာအားဖြင့်, right (`R`) ကို `(1,0)` pair နှင့် ကိုက်ညီစေမည်။ (code block 2):

```python
actions = { "U" : (0,-1), "D" : (0,1), "L" : (-1,0), "R" : (1,0) }
action_idx = { a : i for i,a in enumerate(actions.keys()) }
```

အကျဉ်းချုပ်အားဖြင့်၊ ဒီ scenario ၏ strategy နှင့် goal သည် အောက်ပါအတိုင်းဖြစ်သည်-

- **Strategy**, agent (Peter) ၏ strategy ကို **policy** ဟုခေါ်သော function ဖြင့် သတ်မှတ်သည်။ Policy သည် တည်နေရာတစ်ခုစီတွင် action ကို ပြန်ပေးသည်။ ကျွန်ုပ်တို့၏ problem ၏ state သည် player ၏ လက်ရှိတည်နေရာအပါအဝင် board ကို ကိုယ်စားပြုသည်။

- **Goal**, reinforcement learning ၏ ရည်မှန်းချက်မှာ problem ကို ထိရောက်စွာ ဖြေရှင်းနိုင်ရန် အကောင်းဆုံး policy ကို သင်ယူရန်ဖြစ်သည်။ သို့သော် baseline အနေဖြင့် **random walk** ဟုခေါ်သော ရိုးရှင်းသော policy ကို စဉ်းစားပါမည်။

## Random Walk

အရင်ဆုံး random walk strategy ကို အသုံးပြု၍ ကျွန်ုပ်တို့၏ problem ကို ဖြေရှင်းပါမည်။ Random walk ဖြင့်၊ ကျွန်ုပ်တို့သည် ခွင့်ပြုထားသော actions များအနက်မှ နောက်တစ်ခုကို အလွတ်ရွေးချယ်ပြီး၊ ပန်းသီးကို ရောက်သည်အထိ ဆက်လုပ်မည် (code block 3)။

1. အောက်ပါ code ဖြင့် random walk ကို အကောင်အထည်ဖော်ပါ-

    ```python
    def random_policy(m):
        return random.choice(list(actions))
    
    def walk(m,policy,start_position=None):
        n = 0 # number of steps
        # set initial position
        if start_position:
            m.human = start_position 
        else:
            m.random_start()
        while True:
            if m.at() == Board.Cell.apple:
                return n # success!
            if m.at() in [Board.Cell.wolf, Board.Cell.water]:
                return -1 # eaten by wolf or drowned
            while True:
                a = actions[policy(m)]
                new_pos = m.move_pos(m.human,a)
                if m.is_valid(new_pos) and m.at(new_pos)!=Board.Cell.water:
                    m.move(a) # do the actual move
                    break
            n+=1
    
    walk(m,random_policy)
    ```

    `walk` ကို ခေါ်ဆိုခြင်းသည် သက်ဆိုင်ရာ လမ်းကြောင်း၏ အရှည်ကို ပြန်ပေးသင့်ပြီး၊ run တစ်ခုစီတွင် ကွဲပြားနိုင်သည်။

1. walk စမ်းသပ်မှုကို အကြိမ်ရေ (ဥပမာ 100) အများကြီး run လုပ်ပြီး၊ ရလဒ် statistics ကို print လုပ်ပါ (code block 4):

    ```python
    def print_statistics(policy):
        s,w,n = 0,0,0
        for _ in range(100):
            z = walk(m,policy)
            if z<0:
                w+=1
            else:
                s += z
                n += 1
        print(f"Average path length = {s/n}, eaten by wolf: {w} times")
    
    print_statistics(random_policy)
    ```

    လမ်းကြောင်း၏ အရှည်ပျမ်းမျှသည် 30-40 ခြေလှမ်းအနီးအနားရှိသည်ကို သတိပြုပါ၊ ပန်းသီးအနီးဆုံးအကွာအဝေးပျမ်းမျှသည် 5-6 ခြေလှမ်းသာရှိသည်။

    Random walk အတွင်း Peter ၏ လှုပ်ရှားမှုကိုလည်း ကြည့်နိုင်သည်-

    ![Peter ၏ Random Walk](../../../../8-Reinforcement/1-QLearning/images/random_walk.gif)

## Reward Function

Policy ကို ပိုပြီး ဉာဏ်ရည်ရှိအောင်လုပ်ရန်၊ ဘယ် moves တွေက "ပိုကောင်း" သောအရာများဖြစ်သည်ကို နားလည်ရန်လိုအပ်သည်။ ဒီအတွက် ကျွန်ုပ်တို့၏ ရည်မှန်းချက်ကို သတ်မှတ်ရန်လိုအပ်သည်။

ရည်မှန်းချက်ကို **reward function** ဖြင့် သတ်မှတ်နိုင်ပြီး၊ state တစ်ခုစီအတွက် score value တစ်ခုကို ပြန်ပေးမည်။ ကိန်းဂဏန်းမြင့်မည်လျှင် reward function ပိုကောင်းသည်။ (code block 5)

```python
move_reward = -0.1
goal_reward = 10
end_reward = -10

def reward(m,pos=None):
    pos = pos or m.human
    if not m.is_valid(pos):
        return end_reward
    x = m.at(pos)
    if x==Board.Cell.water or x == Board.Cell.wolf:
        return end_reward
    if x==Board.Cell.apple:
        return goal_reward
    return move_reward
```

Reward functions ၏ စိတ်ဝင်စားဖွယ်ကောင်းသောအချက်မှာ အများဆုံးအခြေအနေတွင် *ဂိမ်းအဆုံးတွင်သာ အရေးပါသော reward ရရှိသည်* ဖြစ်သည်။ ဒါကဆိုရင် ကျွန်ုပ်တို့၏ algorithm သည် positive reward ရရှိသောအခြေအနေများကို "မှတ်မိ" ရမည်။ ထို့နောက်၊ အကောင်းဆုံးရလဒ်ရရှိရန် "ကောင်းသော" လှုပ်ရှားမှုများ၏ အရေးပါမှုကို တိုးမြှင့်ရမည်။ ထို့နောက်၊ အဆိုးဆုံးရလဒ်ရရှိသော moves များကိုလည်း လျော့နည်းစေရမည်။

## Q-Learning

ဒီမှာ ဆွေးနွေးမည့် algorithm ကို **Q-Learning** ဟုခေါ်သည်။ ဒီ algorithm တွင် policy ကို **Q-Table** ဟုခေါ်သော function (သို့မဟုတ် data structure) ဖြင့် သတ်မှတ်သည်။ Q-Table သည် တည်နေရာတစ်ခုစီတွင် action များ၏ "ကောင်းမှု" ကို မှတ်တမ်းတင်သည်။

Q-Table ဟုခေါ်ရသည်မှာ table သို့မဟုတ် multi-dimensional array အဖြစ် ကိုယ်စားပြုရန် အဆင်ပြေသောကြောင့်ဖြစ်သည်။ ကျွန်ုပ်တို့၏ board တွင် `width` x `height` အတိုင်းအတာရှိသည့်အတွက် Q-Table ကို numpy array ဖြင့် `width` x `height` x `len(actions)` အရွယ်အစားဖြင့် ကိုယ်စားပြုနိုင်သည်။ (code block 6)

```python
Q = np.ones((width,height,len(actions)),dtype=np.float)*1.0/len(actions)
```

Q-Table ၏ value အားလုံးကို တူညီသောတန်ဖိုးဖြင့် initialize လုပ်သည်ကို သတိပြုပါ၊ ကျွန်ုပ်တို့၏အခြေအနေတွင် - 0.25 ဖြစ်သည်။ ဒါက "random walk" policy ကို ကိုယ်စားပြုသည်၊ အကြောင်းမူကား တစ်ခုစီတွင် moves အားလုံးသည် တူညီသောအကောင်းမှုရှိသည်။ Q-Table ကို board တွင် visualize လုပ်ရန် `plot` function ကို pass လုပ်နိုင်သည်- `m.plot(Q)`။

![Peter ၏ Environment](../../../../8-Reinforcement/1-QLearning/images/env_init.png)

Cell တစ်ခုစီ၏ အလယ်တွင် "arrow" တစ်ခုရှိပြီး၊ လှုပ်ရှားမှု၏ အကြိုက်ဆုံး direction ကို ဖော်ပြသည်။ Direction အားလုံးတူညီသောကြောင့် dot တစ်ခုကို ဖော်ပြသည်။

အခု simulation ကို run လုပ်ပြီး၊ environment ကို စူးစမ်းပြီး၊ Q-Table values များ၏ distribution ကို ပိုကောင်းစေပြီး၊ ပန်းသီးကို ရှာဖွေရာတွင် ပိုမြန်စေမည်။

## Q-Learning ၏ အဓိကအချက်: Bellman Equation

လှုပ်ရှားမှုကို စတင်ပြီးနောက်၊ action တစ်ခုစီတွင် သက်ဆိုင်ရာ reward ရရှိမည်၊ ဉပမာ immediate reward အမြင့်ဆုံးဖြင့် နောက်တစ်ခုကို ရွေးချယ်နိုင်သည်။ သို့သော်၊ အများဆုံး states တွင် move သည် ပန်းသီးကို ရောက်ရန် ရည်မှန်းချက်ကို မရောက်စေပါ၊ ထို့ကြောင့် ဘယ် direction က ပိုကောင်းသည်ကို ချက်ချင်းဆုံးဖြတ်လို့မရပါ။

> သတိပြုပါ၊ ချက်ချင်းရလဒ်သည် အရေးမကြီးပါ၊ အရေးကြီးသည် simulation ၏ အဆုံးတွင် ရရှိမည့် နောက်ဆုံးရလဒ်ဖြစ်သည်။

ဒီ delayed reward ကို ထည့်သွင်းစဉ်းစားရန် **[dynamic programming](https://en.wikipedia.org/wiki/Dynamic_programming)** ၏ မူကွဲများကို အသုံးပြုရမည်၊ အဲဒါက ကျွန်ုပ်တို့၏ပြဿနာကို recursive အနေနှင့် စဉ်းစားရန် ခွင့်ပြုသည်။

ယခု state *s* တွင်ရှိပြီး၊ နောက် state *s'* သို့ ရွှေ့လိုသည်ဟု ယူဆပါ။ ဒီလိုလုပ်ခြင်းအားဖြင့် immediate reward *r(s,a)* ကို reward function ဖြင့် သတ်မှတ်ပြီး၊ အနာဂတ် reward တစ်ခုရရှိမည်။ ကျွန်ုပ်တို့၏ Q-Table သည် action တစ်ခုစီ၏ "attractiveness" ကို မှန်ကန်စွာ ဖော်ပြသည်ဟု ယူဆပါက၊ state *s'* တွင် *Q(s',a')* ၏ maximum value ကို ကိုက်ညီသော action *a'* ကို ရွေးချယ်မည်။ ထို့ကြောင့် state *s* တွင် ရရှိနိုင်သော အကောင်းဆုံး အနာဂတ် reward ကို `max` *Q(s',a')* (state *s'* တွင် action *a'* များအားလုံးအပေါ်မှာ maximum ကိုတွက်ချက်သည်) ဖြင့် သတ်မှတ်နိုင်သည်။

ဒီအချက်သည် **Bellman formula** ကို ပေးသည်၊ action *a* ကို ရွေးချယ်သော state *s* တွင် Q-Table ၏ value ကိုတွက်ချက်ရန်:

## မူဝါဒကို စစ်ဆေးခြင်း

Q-Table သည် အခြေအနေတစ်ခုစီတွင် လုပ်ဆောင်မှုတစ်ခုစီ၏ "ဆွဲဆောင်မှု" ကို ဖော်ပြထားသောကြောင့် ကျွန်ုပ်တို့၏ကမ္ဘာတွင် ထိရောက်သော လမ်းကြောင်းရှာဖွေမှုကို သတ်မှတ်ရန် အလွယ်တကူ အသုံးပြုနိုင်သည်။ အလွယ်တကူဆုံးသောအခြေအနေတွင် Q-Table အတန်ဖိုးအမြင့်ဆုံးနှင့် ကိုက်ညီသော လုပ်ဆောင်မှုကို ရွေးချယ်နိုင်သည်။ (code block 9)

```python
def qpolicy_strict(m):
        x,y = m.human
        v = probs(Q[x,y])
        a = list(actions)[np.argmax(v)]
        return a

walk(m,qpolicy_strict)
```

> အထက်ပါ code ကို အကြိမ်ကြိမ် စမ်းကြည့်ပါက တစ်ခါတစ်ရံ "hang" ဖြစ်ပြီး notebook တွင် STOP ခလုတ်ကို နှိပ်၍ ရပ်တန့်ရန်လိုအပ်သည်ကို သတိထားမိနိုင်သည်။ ၎င်းသည် အခြေအနေနှစ်ခုသည် အကောင်းဆုံး Q-Value အရ တစ်ခုနှင့်တစ်ခုကို "ညွှန်ပြ" သည့်အခြေအနေများရှိနိုင်သောကြောင့် ဖြစ်ပြီး agent သည် အခြေအနေများအကြား အဆုံးမရှိလှုပ်ရှားနေမိသည်။

## 🚀စိန်ခေါ်မှု

> **Task 1:** `walk` function ကို ပြင်ဆင်ပြီး လမ်းကြောင်း၏ အရှည်ကို အတိအကျ အဆင့်အတန်း (ဥပမာ 100) ဖြင့် ကန့်သတ်ပါ၊ အထက်ပါ code သည် အချိန်အခါမရွေး ဤတန်ဖိုးကို ပြန်ပေးသည်ကို ကြည့်ပါ။

> **Task 2:** `walk` function ကို ပြင်ဆင်ပြီး ယခင်က ရောက်ရှိခဲ့သောနေရာများကို ပြန်မသွားအောင်လုပ်ပါ။ ၎င်းသည် `walk` ကို loop ဖြစ်ခြင်းမှ ကာကွယ်ပေးနိုင်သော်လည်း agent သည် ထွက်မရနိုင်သောနေရာတွင် "ပိတ်မိ" ဖြစ်နိုင်သည်။

## လမ်းကြောင်းရှာဖွေမှု

လမ်းကြောင်းရှာဖွေမှု မူဝါဒအကောင်းဆုံးသည် ကျွန်ုပ်တို့ သင်ကြားမှုအတွင်း အသုံးပြုခဲ့သော မူဝါဒဖြစ်ပြီး exploitation နှင့် exploration ကို ပေါင်းစပ်ထားသည်။ ဤမူဝါဒတွင် Q-Table တွင်ရှိသောတန်ဖိုးများနှင့် အချိုးကျသော probability ဖြင့် လုပ်ဆောင်မှုတစ်ခုစီကို ရွေးချယ်မည်ဖြစ်သည်။ ဤနည်းလမ်းသည် agent ကို ရှာဖွေပြီးသားနေရာသို့ ပြန်သွားစေမည့် အခွင့်အရေးရှိသော်လည်း အောက်ပါ code မှာ မြင်နိုင်သည့်အတိုင်း ရှိသည့်နေရာသို့ ရောက်ရန် အလွန်တိုသော လမ်းကြောင်းကို ရလဒ်ပေးသည်။ (သတိရပါ - `print_statistics` သည် simulation ကို 100 ကြိမ် ပြုလုပ်သည်): (code block 10)

```python
def qpolicy(m):
        x,y = m.human
        v = probs(Q[x,y])
        a = random.choices(list(actions),weights=v)[0]
        return a

print_statistics(qpolicy)
```

ဤ code ကို run ပြီးပါက အရင်ကထက် average path length အလွန်တိုသော 3-6 အတွင်း ရရှိသင့်သည်။

## သင်ယူမှုလုပ်ငန်းစဉ်ကို စုံစမ်းခြင်း

ကျွန်ုပ်တို့ ပြောခဲ့သည့်အတိုင်း သင်ယူမှုလုပ်ငန်းစဉ်သည် problem space ၏ ဖွဲ့စည်းမှုအပေါ် ရရှိထားသော အသိပညာကို ရှာဖွေခြင်းနှင့် အသုံးချခြင်းအကြား တစ်ခုတည်းသော လိုက်လျောမှုဖြစ်သည်။ သင်ယူမှုရလဒ်များ (agent ကို ရည်မှန်းချက်သို့ ရောက်ရန် အတိုသော လမ်းကြောင်းကို ရှာဖွေနိုင်စွမ်း) ကောင်းမွန်လာသည်ကို မြင်ရသော်လည်း သင်ယူမှုလုပ်ငန်းစဉ်အတွင်း average path length ၏ အပြောင်းအလဲကို ကြည့်ရှုခြင်းလည်း စိတ်ဝင်စားဖွယ်ကောင်းသည်။

## သင်ယူမှုများကို အကျဉ်းချုပ်နိုင်သည်။

- **Average path length တိုးလာသည်**။ အစပိုင်းတွင် average path length တိုးလာသည်ကို မြင်ရသည်။ ၎င်းသည် ပတ်ဝန်းကျင်အကြောင်း မသိသေးသောအခါတွင် အဆိုးဆုံး state များ (ရေ၊ ဝက်ဝံ) တွင် ပိတ်မိနိုင်သောကြောင့် ဖြစ်နိုင်သည်။ ပတ်ဝန်းကျင်အကြောင်းပိုမိုသိလာပြီး ဤအသိပညာကို အသုံးပြု၍ ပတ်ဝန်းကျင်ကို ရှာဖွေနိုင်သော်လည်း ပန်းသီးများရှိရာကို မသိသေးသောကြောင့် ဖြစ်နိုင်သည်။

- **Path length လျော့ကျလာသည်**။ သင်ယူမှုများလုံလောက်စွာ ရရှိလာသည့်အခါ agent အတွက် ရည်မှန်းချက်ကို ရောက်ရန် ပိုမိုလွယ်ကူလာပြီး path length လျော့ကျလာသည်။ သို့သော်လည်း agent သည် အကောင်းဆုံးလမ်းကြောင်းမှ ချော်၍ အခြားရွေးချယ်မှုများကို ရှာဖွေသည့်အခါ path length သည် အကောင်းဆုံးထက် ပိုမိုရှည်လျားလာနိုင်သည်။

- **Length တစ်ခါတစ်ရံ ရုတ်တရက် တိုးလာသည်**။ ဤ graph တွင် တစ်ချိန်ချိန်တွင် length ရုတ်တရက် တိုးလာသည်ကိုလည်း တွေ့ရသည်။ ၎င်းသည် လုပ်ငန်းစဉ်၏ stochastic nature ကို ဖော်ပြပြီး Q-Table coefficients များကို တန်ဖိုးအသစ်များဖြင့် ပြန်ရေးသားခြင်းကြောင့် ဖြစ်နိုင်သည်။ ၎င်းကို သင်ကြားမှုအဆုံးပိုင်းတွင် learning rate ကို လျော့ချခြင်းဖြင့် minimize လုပ်သင့်သည် (ဥပမာ Q-Table တန်ဖိုးများကို အနည်းငယ်သာ ပြင်ဆင်ခြင်း)။

စုစုပေါင်းအားဖြင့် သင်ယူမှုလုပ်ငန်းစဉ်၏ အောင်မြင်မှုနှင့် အရည်အသွေးသည် learning rate, learning rate decay, နှင့် discount factor ကဲ့သို့သော parameters များအပေါ် အလွန်အမင်း မှီခိုနေသည်ကို သတိထားရမည်။ ၎င်းတို့ကို **hyperparameters** ဟု ခေါ်ပြီး **parameters** (ဥပမာ Q-Table coefficients) နှင့် ခွဲခြားရန် သတ်မှတ်ထားသည်။ hyperparameter များ၏ အကောင်းဆုံးတန်ဖိုးများကို ရှာဖွေခြင်းလုပ်ငန်းစဉ်ကို **hyperparameter optimization** ဟု ခေါ်ပြီး ၎င်းသည် သီးခြားအကြောင်းအရာတစ်ခုအဖြစ် သတ်မှတ်ရန် တန်ဖိုးရှိသည်။

## [Post-lecture quiz](https://ff-quizzes.netlify.app/en/ml/)

## အလုပ်ပေးစာ 
[A More Realistic World](assignment.md)

---

**ဝက်ဘ်ဆိုက်မှတ်ချက်**:  
ဤစာရွက်စာတမ်းကို AI ဘာသာပြန်ဝန်ဆောင်မှု [Co-op Translator](https://github.com/Azure/co-op-translator) ကို အသုံးပြု၍ ဘာသာပြန်ထားပါသည်။ ကျွန်ုပ်တို့သည် တိကျမှန်ကန်မှုအတွက် ကြိုးစားနေပါသော်လည်း၊ အလိုအလျောက်ဘာသာပြန်ဆိုမှုများတွင် အမှားများ သို့မဟုတ် မတိကျမှုများ ပါဝင်နိုင်သည်ကို ကျေးဇူးပြု၍ သတိပြုပါ။ မူရင်းဘာသာစကားဖြင့် ရေးသားထားသော စာရွက်စာတမ်းကို အာဏာတည်သော ရင်းမြစ်အဖြစ် သတ်မှတ်သင့်ပါသည်။ အရေးကြီးသော အချက်အလက်များအတွက် လူက ဘာသာပြန်ဝန်ဆောင်မှုကို အသုံးပြုရန် အကြံပြုပါသည်။ ဤဘာသာပြန်ကို အသုံးပြုခြင်းမှ ဖြစ်ပေါ်လာသော နားလည်မှုမှားများ သို့မဟုတ် အဓိပ္ပာယ်မှားများအတွက် ကျွန်ုပ်တို့သည် တာဝန်မယူပါ။