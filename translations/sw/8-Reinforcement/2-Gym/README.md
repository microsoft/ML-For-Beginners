# Kuteleza kwa CartPole

Tatizo ambalo tumekuwa tukitatua katika somo lililopita linaweza kuonekana kama tatizo la mchezo, halitumiki sana kwa hali halisi za maisha. Hii siyo kesi, kwa sababu matatizo mengi halisi pia yanashirikiana hali hii - ikiwa ni pamoja na kucheza Chess au Go. Yanafanana, kwa sababu pia tuna bodi yenye sheria zilizowekwa na **hali ya diskrete**.

## [Jaribio kabla ya somo](https://ff-quizzes.netlify.app/en/ml/)

## Utangulizi

Katika somo hili tutatumia kanuni zile zile za Q-Learning kwa tatizo lenye **hali ya kuendelea**, yaani hali inayotolewa na nambari moja au zaidi halisi. Tutashughulikia tatizo lifuatalo:

> **Tatizo**: Ikiwa Peter anataka kutoroka mbwa mwitu, anahitaji kuweza kusogea kwa kasi zaidi. Tutaona jinsi Peter anavyoweza kujifunza kuteleza, hasa, kudumisha usawa, kwa kutumia Q-Learning.

![Kimbilio kubwa!](../../../../translated_images/sw/escape.18862db9930337e3.webp)

> Peter na marafiki zake wanatumia ubunifu kutoroka mbwa mwitu! Picha na [Jen Looper](https://twitter.com/jenlooper)

Tutatumia toleo lililorahisishwa la kusawazisha linalojulikana kama tatizo la **CartPole**. Katika dunia ya cartpole, tuna slide ya usawa inayoweza kusogea kushoto au kulia, na lengo ni kusawazisha nguzo wima juu ya slide.

<img alt="a cartpole" src="../../../../translated_images/sw/cartpole.b5609cc0494a14f7.webp" width="200"/>

## Mahitaji ya awali

Katika somo hili, tutatumia maktaba inayoitwa **OpenAI Gym** kuiga **mazingira** tofauti. Unaweza kuendesha msimbo wa somo hili ndani ya kompyuta (mfano kutoka Visual Studio Code), ambapo mizunguko itaonekana kwenye dirisha jipya. Unapochukua msimbo mtandaoni, unaweza kuhitaji kufanya marekebisho kidogo kwenye msimbo, kama ilivyoelezwa [hapa](https://towardsdatascience.com/rendering-openai-gym-envs-on-binder-and-google-colab-536f99391cc7).

## OpenAI Gym

Katika somo lililopita, sheria za mchezo na hali zilielezwa na darasa la `Board` tulilolifafanua wenyewe. Hapa tutatumia **mazingira ya kuiga** maalum, ambayo itaiga fizikia nyuma ya nguzo inayobadilika. Moja ya mazingira maarufu kwa mafunzo ya algoriti za kujifunza kwa nguvu inayoitwa [Gym](https://gym.openai.com/), inayoendeshwa na [OpenAI](https://openai.com/). Kwa kutumia gym hii tunaweza kuunda **mazingira** tofauti kutoka kwa kuiga cartpole hadi michezo ya Atari.

> **Kumbuka**: Unaweza kuona mazingira mengine yanayopatikana kutoka OpenAI Gym [hapa](https://gym.openai.com/envs/#classic_control). 

Kwanza, wacha tufunge gym na tulete maktaba zinazohitajika (kifungu cha msimbo 1):

```python
import sys
!{sys.executable} -m pip install gym 

import gym
import matplotlib.pyplot as plt
import numpy as np
import random
```

## Zoefu - anzisha mazingira ya cartpole

Ili kufanya kazi na tatizo la kusawazisha cartpole, tunahitaji kuanzisha mazingira yanayohusiana. Kila mazingira yanahusishwa na:

- **Eneo la uchunguzi** linalofafanua muundo wa taarifa tunazopokea kutoka kwa mazingira. Kwa tatizo la cartpole, tunapokea nafasi ya nguzo, kasi, na baadhi ya thamani nyingine.

- **Eneo la hatua** linalofafanua hatua zinazowezekana. Katika kesi yetu, eneo la hatua ni diskrete, na linajumuisha hatua mbili - **kushoto** na **kulia**. (kifungu cha msimbo 2)

1. Ili kuanzisha, andika msimbo ufuatao:

    ```python
    env = gym.make("CartPole-v1")
    print(env.action_space)
    print(env.observation_space)
    print(env.action_space.sample())
    ```

Ili kuona jinsi mazingira yanavyofanya kazi, tuchukue mizunguko mifupi ya majaribio kwa hatua 100. Kila hatua tunatoa moja ya hatua zinazopaswa kuchukuliwa - katika mizunguko hii tunachagua hatua kwa bahati nasibu kutoka `action_space`. 

1. Endesha msimbo hapa chini na uone kinacho tufikia.

✅ Kumbuka ni bora kuendesha msimbo huu kwenye usanikishaji wa Python wa ndani! (kifungu cha msimbo 3)

    ```python
    env.reset()
    
    for i in range(100):
       env.render()
       env.step(env.action_space.sample())
    env.close()
    ```

Unapaswa kuona kitu kinachofanana na picha hii:

![cartpole isiyosawazishwa](../../../../8-Reinforcement/2-Gym/images/cartpole-nobalance.gif)

1. Wakati wa mizunguko, tunahitaji kupata uchunguzi ili kuamua jinsi ya kuchukua hatua. Kazi ya hatua hurudisha uchunguzi wa sasa, thamani ya zawadi, na bendera ya kumaliza inayoonyesha kama kuna maana kuendelea na mzunguko au la: (kifungu cha msimbo 4)

    ```python
    env.reset()
    
    done = False
    while not done:
       env.render()
       obs, rew, done, info = env.step(env.action_space.sample())
       print(f"{obs} -> {rew}")
    env.close()
    ```

Utamaliza kuona kitu kama hiki katika matokeo ya daftari:

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

Vector ya uchunguzi inayorudishwa kila hatua ya mizunguko ina thamani zifuatazo:
- Nafasi ya gari
- Kasi ya gari
- Mwelekeo wa nguzo
- Kasi ya mzunguko wa nguzo

1. Pata thamani ndogo na kubwa kati ya nambari hizo: (kifungu cha msimbo 5)

    ```python
    print(env.observation_space.low)
    print(env.observation_space.high)
    ```

Unaweza pia kugundua kwamba thamani ya zawadi katika kila hatua ya mizunguko siku zote ni 1. Hii ni kwa sababu lengo letu ni kuishi kwa muda mrefu iwezekanavyo, yaani kudumisha nguzo katika mwelekeo wa wima kwa muda mrefu zaidi.

✅ Kwa kweli, kuiga CartPole kunachukuliwa kutatuliwa ikiwa tunaweza kupata wastani wa zawadi wa 195 kwa majaribio 100 mfululizo.

## Diskuretisha hali

Katika Q-Learning, tunahitaji kujenga Jedwali la Q linaloelezea nini cha kufanya kila hali. Ili kufanya hivyo, hali lazima iwe **diskrete**, hasa iwe na idadi ya kiwango cha thamani chenye kikomo. Hivyo, tunahitaji kwa namna fulani **kugawanya** maoni yetu, tukiyalinganisha na seti chache za hali.

Kuna njia chache tunaweza kufanya hivi:

- **Gawanya katika kontena**. Ikiwa tunajua awamu ya thamani fulani, tunaweza kugawa awamu hii katika kontena kadhaa, kisha kubadilisha thamani kwa nambari ya kontena inayohusiana. Hii inaweza kufanywa kwa kutumia njia ya numpy [`digitize`](https://numpy.org/doc/stable/reference/generated/numpy.digitize.html). Katika kesi hii, tutajua kwa usahihi ukubwa wa hali, kwa sababu utategemea idadi ya kontena tulizochagua kwa diditalization.
  
✅ Tunaweza pia kutumia interpoleshini ya mstari kuleta thamani katika awamu ndogo (mfano, kutoka -20 hadi 20), kisha kubadilisha nambari kwa integers kwa kuziround. Hii hutupa udhibiti mdogo juu ya ukubwa wa hali, hasa ikiwa hatujui safu sahihi ya thamani za ingizo. Kwa mfano, katika kesi yetu 2 kati ya thamani 4 haina mipaka ya juu/chini, ambayo inaweza kusababisha idadi isiyo na kikomo ya hali.

Katika mfano wetu, tutatumia mbinu ya pili. Kama utaona baadaye, licha ya mipaka isiyoelezwa, thamani hizi mara chache hupita katika awamu fulani za kikomo, hivyo hali hizo zenye thamani kubwa zitakuwa nadra sana.

1. Hapa ni kazi itakayochukua maoni kutoka kwa mfano wetu na kutoa tuple yenye thamani 4 za integer: (kifungu cha msimbo 6)

    ```python
    def discretize(x):
        return tuple((x/np.array([0.25, 0.25, 0.01, 0.1])).astype(np.int))
    ```

1. Pia tuchunguze njia nyingine ya diskuretisha kwa kutumia kontena: (kifungu cha msimbo 7)

    ```python
    def create_bins(i,num):
        return np.arange(num+1)*(i[1]-i[0])/num+i[0]
    
    print("Sample bins for interval (-5,5) with 10 bins\n",create_bins((-5,5),10))
    
    ints = [(-5,5),(-2,2),(-0.5,0.5),(-2,2)] # vipindi vya thamani kwa kila parameta
    nbins = [20,20,10,10] # idadi ya masanduku kwa kila parameta
    bins = [create_bins(ints[i],nbins[i]) for i in range(4)]
    
    def discretize_bins(x):
        return tuple(np.digitize(x[i],bins[i]) for i in range(4))
    ```

1. Sasa tutaendesha mizunguko mfupi na kuangalia thamani hizo za mazingira ya diskrete. Jaribu `discretize` na `discretize_bins` na uone kama kuna tofauti.

✅ discretize_bins hurudisha nambari ya kontena, ambayo huanzia 0. Hivyo kwa thamani za variable za ingizo karibu na 0 hurudisha nambari ya katikati ya awamu (10). Katika discretize, hatukujali juu ya safu ya thamani za output, tukiruhusu kuwa hasi, hivyo thamani za hali hazijasogezwa, na 0 ni sawa na 0. (kifungu cha msimbo 8)

    ```python
    env.reset()
    
    done = False
    while not done:
       #env.onyeshaji()
       obs, rew, done, info = env.step(env.action_space.sample())
       #chapisha(discretize_bins(obs))
       print(discretize(obs))
    env.close()
    ```

✅ Fungua mstari unaoanza na env.render ikiwa unataka kuona mazingira yanavyotekeleza. Vinginevyo unaweza kuyatekeleza nyuma, ambayo ni haraka zaidi. Tutatumia utekelezaji huu "usioonekana" wakati wa mchakato wa Q-Learning.

## Muundo wa Jedwali la Q

Katika somo letu lililopita, hali ilikuwa jozi rahisi ya nambari kutoka 0 hadi 8, hivyo ilikuwa rahisi kuwakilisha Jedwali la Q kwa tensor ya numpy yenye umbo 8x8x2. Ikiwa tutatumia diskuretisha kwa kontena, ukubwa wa vector ya hali pia unajulikana, hivyo tunaweza kutumia mbinu ile ile na kuwakilisha hali kwa safu yenye umbo 20x20x10x10x2 (hapa 2 ni urefu wa eneo la hatua, na vipimo vya kwanza ni idadi ya kontena tulizochagua kwa kila parameter katika eneo la uchunguzi).

Hata hivyo, mara nyingine vipimo kamili vya eneo la uchunguzi havijulikani. Katika kesi ya `discretize`, hatuwezi kamwe kuwa na uhakika kuwa hali yetu itabaki ndani ya mipaka fulani, kwa sababu baadhi ya thamani za asili hazina mipaka. Kwa hivyo, tutatumia mbinu tofauti kidogo na kuwakilisha Jedwali la Q kwa kamusi.

1. Tumia jozi *(state,action)* kama ufunguo wa kamusi, na thamani itawakilisha thamani ya entry ya Jedwali la Q. (kifungu cha msimbo 9)

    ```python
    Q = {}
    actions = (0,1)
    
    def qvalues(state):
        return [Q.get((state,a),0) for a in actions]
    ```

Hapa pia tunafafanua kazi `qvalues()`, inayorudisha orodha ya thamani za Jedwali la Q kwa hali fulani kuhusiana na hatua zote zinazowezekana. Ikiwa entry haina katika Jedwali la Q, tutarudisha 0 kama chaguo la msingi.

## Wacha tuanze Q-Learning

Sasa tuko tayari kumfundisha Peter kusawazisha!

1. Kwanza, wacha tuiweke mipangilio ya hyperparameters: (kifungu cha msimbo 10)

    ```python
    # vigezo vya juu
    alpha = 0.3
    gamma = 0.9
    epsilon = 0.90
    ```

Hapa, `alpha` ni **kasi ya kujifunza** inayofafanua hatua gani tunapaswa kurekebisha thamani za sasa za Jedwali la Q kila hatua. Katika somo lililopita tulianza na 1, kisha tuliipunguza `alpha` hadi thamani ndogo wakati wa mafunzo. Katika mfano huu tutaiweka thabiti kwa urahisi, na unaweza kujaribu kurekebisha thamani za `alpha` baadaye.

`gamma` ni **kigezo cha punguzo** kinachoonyesha kiasi gani tunapaswa kuweka kipaumbele zawadi za baadaye kuliko zawadi za sasa.

`epsilon` ni **kigezo cha uchunguzi/kutumia** kinachoamua kama tunapaswa kupendelea uchunguzi kuliko matumizi au kinyume chake. Katika algoriti yetu, katika asilimia ya `epsilon` tutachagua hatua ifuatayo kulingana na thamani za Jedwali la Q, na katika idadi iliyobaki tutatekeleza hatua ya bahati nasibu. Hii itaturuhusu kuchunguza maeneo ya utafutaji ambayo hatujawahi kuona.

✅ Kuhusu kusawazisha - kuchagua hatua ya bahati nasibu (uchunguzi) itakuwa kama pigo la bahati nasibu kwenda upande usiofaa, na nguzo itahitaji kujifunza jinsi ya kurejesha usawa kutokana na "makosa" hayo

### Boresha algoriti

Tunaweza pia kufanya maboresho mawili kwa algoriti yetu kutoka somo lililopita:

- **Hesabu wastani wa zawadi kwa pamoja**, kwa mzunguko kadhaa. Tutaweka maendeleo kila mizunguko 5000, na tutapima wastani wa zawadi kwa pamoja kwa kipindi hicho. Inamaanisha ikiwa tunaweza kupata zaidi ya alama 195 - tunaweza kuchukulia tatizo limetatuliwa, kwa ubora hata zaidi kuliko uliohitajika.
  
- **Hesabu matokeo makubwa ya wastani wa pamoja**, `Qmax`, na tutahifadhi Jedwali la Q linalohusiana na matokeo hayo. Utapoganda mafunzo utagundua mara nyingine matokeo ya wastani ya pamoja huanza kupungua, na tunataka kuhifadhi thamani za Jedwali la Q zinazolingana na mfano bora ulioonekana wakati wa mafunzo.

1. Kusanya zawadi zote kwa pamoja kila mzunguko katika vector `rewards` kwa ajili ya ploti ya baadaye. (kifungu cha msimbo  11)

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
        # == fanya upigaji wa majaribio ==
        while not done:
            s = discretize(obs)
            if random.random()<epsilon:
                # matumizi - chagua kitendo kulingana na uwezekano wa Jedwali la Q
                v = probs(np.array(qvalues(s)))
                a = random.choices(actions,weights=v)[0]
            else:
                # uchunguzi - chagua kitendo kwa bahati nasibu
                a = np.random.randint(env.action_space.n)
    
            obs, rew, done, info = env.step(a)
            cum_reward+=rew
            ns = discretize(obs)
            Q[(s,a)] = (1 - alpha) * Q.get((s,a),0) + alpha * (rew + gamma * max(qvalues(ns)))
        cum_rewards.append(cum_reward)
        rewards.append(cum_reward)
        # == Mara kwa mara chapisha matokeo na hesabu zawadi ya wastani ==
        if epoch%5000==0:
            print(f"{epoch}: {np.average(cum_rewards)}, alpha={alpha}, epsilon={epsilon}")
            if np.average(cum_rewards) > Qmax:
                Qmax = np.average(cum_rewards)
                Qbest = Q
            cum_rewards=[]
    ```

Unachoweza kugundua kutoka kwa matokeo hayo:

- **Karibu na lengo letu**. Tuko karibu sana kufanikisha lengo la kupata 195 za zawadi kwa mizunguko 100 mfululizo ya majaribio ya kuiga, au labda tumefanikisha! Hata tukipata nambari ndogo zaidi, bado hatujui, kwa sababu tunapima wastani kwa mizunguko 5000, na mizunguko 100 ndio inahitajika katika vigezo rasmi.
  
- **Zawadi inaanza kupungua**. Wakati mwingine zawadi huanza kupungua, maana yake tunaweza "kuharibu" thamani zilizojifunza katika Jedwali la Q na zile zinazofanya hali kuwa mbaya.

Uchunguzi huu unaonekana wazi zaidi ikiwa tuliweka maendeleo ya mafunzo kwenye mchoro.

## Kuchora Maendeleo ya Mafunzo

Wakati wa mafunzo, tuliokusanya thamani ya zawadi kwa pamoja kila mara katika vector ya `rewards`. Huu ndio muonekano wake tunapochora dhidi ya nambari ya mzunguko:

```python
plt.plot(rewards)
```

![maendeleo ghafi](../../../../translated_images/sw/train_progress_raw.2adfdf2daea09c59.webp)

Kutoka kwenye grafu hii, si rahisi kuelewa chochote kwa sababu kutokana na asili ya mchakato wa mafunzo wa kishtochasti urefu wa vikao vya mafunzo hutofautiana sana. Ili kufanya grafu hii iwe na maana zaidi, tunaweza kuhesabu **wastani unaoendelea** kwa mfululizo wa majaribio, tuseme 100. Hii inaweza kufanyika kwa urahisi kwa kutumia `np.convolve`: (kifungu cha msimbo 12)

```python
def running_average(x,window):
    return np.convolve(x,np.ones(window)/window,mode='valid')

plt.plot(running_average(rewards,100))
```

![maendeleo ya mafunzo](../../../../translated_images/sw/train_progress_runav.c71694a8fa9ab359.webp)

## Kubadilisha hyperparameters

Ili kufanya kujifunza kuwa thabiti zaidi, ni busara kurekebisha baadhi ya hyperparameters zetu wakati wa mafunzo. Hasa:

- **Kwa kasi ya kujifunza**, `alpha`, tunaweza kuanza na thamani karibu na 1, kisha tukae tukipunguza kidogo kidogo. Hivi ndivyo tutakuwa na thamani nzuri za uwezekano za Jedwali la Q, hivyo tunapaswa kuzipanga kidogo badala ya kuzibadilisha kabisa na thamani mpya.

- **Ongeza epsilon**. Tunaweza kuonekana kuongeza `epsilon` polepole, ili tukuelekeze kidogo kwenye uchunguzi na zaidi kwenye matumizi. Inaonekana busara kuanza na thamani ndogo ya `epsilon`, kisha kwenda hadi karibu 1.

> **Kazi 1**: Jiunge na thamani za hyperparameter na uone kama unaweza kupata zawadi ya juu zaidi kwa pamoja. Je, unapiga zaidi ya 195?


> **Kazi 2**: Ili kutatua tatizo rasmi, unahitaji kupata zawadi ya wastani ya 195 katika mizunguko 100 mfululizo. Pima hilo wakati wa mafunzo na hakikisha kuwa umetatua tatizo rasmi!

## Kuona matokeo kwa vitendo

Itakuwa ya kuvutia kuona jinsi mfano uliopita mafunzo unavyotenda. Hebu endesha simulizi na ufuate mkakati ule ule wa kuchagua vitendo kama wakati wa mafunzo, ukichagua kulingana na usambazaji wa uwezekano katika Jedwali la Q: (block ya nambari 13)

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

Unapaswa kuona kitu kama hiki:

![a balancing cartpole](../../../../8-Reinforcement/2-Gym/images/cartpole-balance.gif)

---

## 🚀Changamoto

> **Kazi 3**: Hapa, tulikuwa tunatumia nakala ya mwisho ya Jedwali la Q, ambayo huenda si ile bora zaidi. Kumbuka kuwa tumehifadhi Jedwali la Q linayofanya vizuri zaidi kwenye kigeuzi `Qbest`! Jaribu mfano ule ule ukitumia Jedwali la Q linayofanya vizuri zaidi kwa kunakili `Qbest` kwenda `Q` na angalia kama utaona tofauti.

> **Kazi 4**: Hapa hatukuchagua tendo bora kila hatua, bali tulikuwa tunachagua kulingana na usambazaji wa uwezekano unaolingana. Je, wangekuwa na maana zaidi kila wakati kuchagua tendo bora zaidi, lenye thamani kubwa zaidi katika Jedwali la Q? Hii inaweza kufanywa kwa kutumia kazi ya `np.argmax` kugundua nambari ya tendo linalolingana na thamani kubwa zaidi ya Jedwali la Q. Tekeleza mkakati huu na uone kama unaboresha usawa.

## [Mtihani baada ya mihadhara](https://ff-quizzes.netlify.app/en/ml/)

## Kazi ya nyumbani
[Funza Gari la Mlima](assignment.md)

## Hitimisho

Sasa tumefundishwa jinsi ya kufunza mawakala kufikia matokeo mazuri kwa kuwapa tu kazi ya zawadi inayobainisha hali inayotakiwa ya mchezo, na kwa kuwapa nafasi ya kuchunguza kwa akili nafasi ya utaftaji. Tumetumia kwa mafanikio algorithm ya Q-Learning katika mazingira ya tarakimu na yanayoendelea, lakini kwa vitendo vya tarakimu.

Ni muhimu pia kusomea hali ambapo hali ya tendo pia ni endelevu, na wakati nafasi ya uchunguzi ni tata zaidi, kama vile picha kutoka kwenye skrini ya mchezo wa Atari. Katika matatizo hayo mara nyingi tunahitaji kutumia mbinu zilizo na nguvu zaidi za ujifunzaji wa mashine, kama vile mitandao ya neva, ili kufikia matokeo mazuri. Mada hizo za juu zaidi ni somo la kozi yetu ya juu zaidi ya AI inayokuja.

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**Kionyozo**:
Hati hii imetafsiriwa kwa kutumia huduma ya tafsiri ya AI [Co-op Translator](https://github.com/Azure/co-op-translator). Ingawa tunajitahidi kupata usahihi, tafadhali fahamu kwamba tafsiri za kiotomatiki zinaweza kuwa na makosa au upungufu wa usahihi. Hati ya asili katika lugha yake halisi inapaswa kuchukuliwa kama chanzo cha mamlaka. Kwa taarifa muhimu, tafsiri ya kitaalamu inayofanywa na binadamu inapendekezwa. Hatutojibu kwa kuelewa vibaya au tafsiri potofu zinazotokea kutokana na matumizi ya tafsiri hii.
<!-- CO-OP TRANSLATOR DISCLAIMER END -->