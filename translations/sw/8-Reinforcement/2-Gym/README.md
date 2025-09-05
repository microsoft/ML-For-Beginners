<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "107d5bb29da8a562e7ae72262d251a75",
  "translation_date": "2025-09-05T16:44:35+00:00",
  "source_file": "8-Reinforcement/2-Gym/README.md",
  "language_code": "sw"
}
-->
## Mahitaji ya Awali

Katika somo hili, tutatumia maktaba inayoitwa **OpenAI Gym** kuiga mazingira tofauti. Unaweza kuendesha msimbo wa somo hili kwenye kompyuta yako (mfano, kutoka Visual Studio Code), ambapo simulizi itafunguka kwenye dirisha jipya. Unapokimbia msimbo mtandaoni, huenda ukahitaji kufanya marekebisho fulani kwenye msimbo, kama ilivyoelezwa [hapa](https://towardsdatascience.com/rendering-openai-gym-envs-on-binder-and-google-colab-536f99391cc7).

## OpenAI Gym

Katika somo lililopita, sheria za mchezo na hali zilikuwa zimetolewa na darasa la `Board` ambalo tulilifafanua sisi wenyewe. Hapa tutatumia mazingira maalum ya **simulizi**, ambayo yataiga fizikia ya fimbo inayosawazishwa. Mojawapo ya mazingira maarufu ya simulizi kwa mafunzo ya algoriti za kujifunza kwa kuimarisha inaitwa [Gym](https://gym.openai.com/), ambayo inadumishwa na [OpenAI](https://openai.com/). Kwa kutumia Gym hii tunaweza kuunda mazingira tofauti kutoka simulizi ya CartPole hadi michezo ya Atari.

> **Note**: Unaweza kuona mazingira mengine yanayopatikana kutoka OpenAI Gym [hapa](https://gym.openai.com/envs/#classic_control). 

Kwanza, wacha tusakinishe Gym na kuingiza maktaba zinazohitajika (msimbo wa block 1):

```python
import sys
!{sys.executable} -m pip install gym 

import gym
import matplotlib.pyplot as plt
import numpy as np
import random
```

## Zoezi - kuanzisha mazingira ya CartPole

Ili kufanya kazi na tatizo la kusawazisha CartPole, tunahitaji kuanzisha mazingira yanayohusiana. Kila mazingira yanahusishwa na:

- **Observation space** inayofafanua muundo wa taarifa tunazopokea kutoka kwa mazingira. Kwa tatizo la CartPole, tunapokea nafasi ya fimbo, kasi, na thamani nyingine.

- **Action space** inayofafanua hatua zinazowezekana. Katika kesi yetu, action space ni ya kidhahiri, na ina hatua mbili - **kushoto** na **kulia**. (msimbo wa block 2)

1. Ili kuanzisha, andika msimbo ufuatao:

    ```python
    env = gym.make("CartPole-v1")
    print(env.action_space)
    print(env.observation_space)
    print(env.action_space.sample())
    ```

Ili kuona jinsi mazingira yanavyofanya kazi, wacha tuendeshe simulizi fupi kwa hatua 100. Katika kila hatua, tunatoa moja ya hatua za kuchukuliwa - katika simulizi hii tunachagua hatua kwa nasibu kutoka `action_space`. 

1. Kimbia msimbo hapa chini na uone matokeo yake.

    ✅ Kumbuka kuwa inapendekezwa kuendesha msimbo huu kwenye usakinishaji wa Python wa ndani! (msimbo wa block 3)

    ```python
    env.reset()
    
    for i in range(100):
       env.render()
       env.step(env.action_space.sample())
    env.close()
    ```

    Unapaswa kuona kitu kinachofanana na picha hii:

    ![CartPole isiyosawazishwa](../../../../8-Reinforcement/2-Gym/images/cartpole-nobalance.gif)

1. Wakati wa simulizi, tunahitaji kupata uchunguzi ili kuamua jinsi ya kutenda. Kwa kweli, kazi ya hatua inarudisha uchunguzi wa sasa, kazi ya malipo, na bendera ya kumaliza inayonyesha kama ina maana kuendelea na simulizi au la: (msimbo wa block 4)

    ```python
    env.reset()
    
    done = False
    while not done:
       env.render()
       obs, rew, done, info = env.step(env.action_space.sample())
       print(f"{obs} -> {rew}")
    env.close()
    ```

    Mwisho wake utakuwa kitu kama hiki kwenye matokeo ya daftari:

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

    Vector ya uchunguzi inayorudishwa katika kila hatua ya simulizi ina thamani zifuatazo:
    - Nafasi ya gari
    - Kasi ya gari
    - Pembe ya fimbo
    - Kiwango cha mzunguko wa fimbo

1. Pata thamani ya chini na ya juu ya namba hizo: (msimbo wa block 5)

    ```python
    print(env.observation_space.low)
    print(env.observation_space.high)
    ```

    Unaweza pia kugundua kuwa thamani ya malipo katika kila hatua ya simulizi daima ni 1. Hii ni kwa sababu lengo letu ni kuishi kwa muda mrefu iwezekanavyo, yaani, kuweka fimbo katika nafasi ya wima kwa muda mrefu zaidi.

    ✅ Kwa kweli, simulizi ya CartPole inachukuliwa kuwa imetatuliwa ikiwa tunafanikiwa kupata wastani wa malipo ya 195 katika majaribio 100 mfululizo.

## Ubadilishaji wa Hali

Katika Q-Learning, tunahitaji kujenga Q-Table inayofafanua nini cha kufanya katika kila hali. Ili kufanya hivyo, tunahitaji hali iwe **kidhahiri**, haswa, inapaswa kuwa na idadi finyu ya thamani za kidhahiri. Kwa hivyo, tunahitaji kwa namna fulani **kubadilisha** uchunguzi wetu, kuupangilia kwenye seti finyu ya hali.

Kuna njia kadhaa tunaweza kufanya hivi:

- **Gawanya katika bins**. Ikiwa tunajua kipindi cha thamani fulani, tunaweza kugawanya kipindi hiki katika idadi ya **bins**, kisha kubadilisha thamani kwa namba ya bin ambayo inahusiana nayo. Hii inaweza kufanywa kwa kutumia njia ya numpy [`digitize`](https://numpy.org/doc/stable/reference/generated/numpy.digitize.html). Katika kesi hii, tutajua kwa usahihi ukubwa wa hali, kwa sababu itategemea idadi ya bins tunazochagua kwa digitalization.
  
✅ Tunaweza kutumia uingiliano wa mstari kuleta thamani kwenye kipindi finyu (sema, kutoka -20 hadi 20), kisha kubadilisha namba kuwa namba za kidhahiri kwa kuzirudisha. Hii inatupa udhibiti mdogo juu ya ukubwa wa hali, hasa ikiwa hatujui mipaka halisi ya thamani za pembejeo. Kwa mfano, katika kesi yetu 2 kati ya 4 za thamani hazina mipaka ya juu/chini, ambayo inaweza kusababisha idadi isiyo na mwisho ya hali.

Katika mfano wetu, tutatumia njia ya pili. Kama utakavyogundua baadaye, licha ya mipaka isiyofafanuliwa ya juu/chini, thamani hizo mara chache huchukua thamani nje ya vipindi finyu, kwa hivyo hali hizo zenye thamani za juu zitakuwa nadra sana.

1. Hapa kuna kazi itakayochukua uchunguzi kutoka kwa mfano wetu na kutoa tuple ya thamani 4 za kidhahiri: (msimbo wa block 6)

    ```python
    def discretize(x):
        return tuple((x/np.array([0.25, 0.25, 0.01, 0.1])).astype(np.int))
    ```

1. Wacha pia tuchunguze njia nyingine ya ubadilishaji kwa kutumia bins: (msimbo wa block 7)

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

1. Sasa wacha tuendeshe simulizi fupi na kuchunguza thamani za mazingira ya kidhahiri. Jisikie huru kujaribu `discretize` na `discretize_bins` na uone kama kuna tofauti.

    ✅ discretize_bins inarudisha namba ya bin, ambayo ni 0-based. Kwa hivyo kwa thamani za pembejeo karibu na 0 inarudisha namba kutoka katikati ya kipindi (10). Katika discretize, hatukujali kuhusu kipimo cha thamani za matokeo, tukiruhusu ziwe hasi, kwa hivyo thamani za hali hazijabadilishwa, na 0 inahusiana na 0. (msimbo wa block 8)

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

    ✅ Ondoa mstari unaoanza na env.render ikiwa unataka kuona jinsi mazingira yanavyotekelezwa. Vinginevyo unaweza kuyatekeleza kwa siri, ambayo ni haraka zaidi. Tutatumia utekelezaji huu "usioonekana" wakati wa mchakato wetu wa Q-Learning.

## Muundo wa Q-Table

Katika somo letu lililopita, hali ilikuwa jozi rahisi ya namba kutoka 0 hadi 8, na kwa hivyo ilikuwa rahisi kuwakilisha Q-Table kwa tensor ya numpy yenye umbo la 8x8x2. Ikiwa tunatumia bins discretization, ukubwa wa vector ya hali yetu pia unajulikana, kwa hivyo tunaweza kutumia mbinu sawa na kuwakilisha hali kwa safu yenye umbo la 20x20x10x10x2 (hapa 2 ni kipimo cha action space, na vipimo vya kwanza vinahusiana na idadi ya bins tulizochagua kutumia kwa kila parameter katika observation space).

Hata hivyo, wakati mwingine vipimo halisi vya observation space havijulikani. Katika kesi ya kazi ya `discretize`, hatuwezi kamwe kuwa na uhakika kwamba hali yetu inabaki ndani ya mipaka fulani, kwa sababu baadhi ya thamani za awali hazina mipaka. Kwa hivyo, tutatumia mbinu tofauti kidogo na kuwakilisha Q-Table kwa kamusi.

1. Tumia jozi *(state,action)* kama ufunguo wa kamusi, na thamani itahusiana na thamani ya Q-Table. (msimbo wa block 9)

    ```python
    Q = {}
    actions = (0,1)
    
    def qvalues(state):
        return [Q.get((state,a),0) for a in actions]
    ```

    Hapa pia tunafafanua kazi `qvalues()`, ambayo inarudisha orodha ya thamani za Q-Table kwa hali fulani inayohusiana na hatua zote zinazowezekana. Ikiwa kiingilio hakipo katika Q-Table, tutarudisha 0 kama chaguo-msingi.

## Wacha tuanze Q-Learning

Sasa tuko tayari kumfundisha Peter kusawazisha!

1. Kwanza, wacha tuweke baadhi ya hyperparameters: (msimbo wa block 10)

    ```python
    # hyperparameters
    alpha = 0.3
    gamma = 0.9
    epsilon = 0.90
    ```

    Hapa, `alpha` ni **learning rate** inayofafanua kiwango ambacho tunapaswa kurekebisha thamani za sasa za Q-Table katika kila hatua. Katika somo lililopita tulianza na 1, kisha tukapunguza `alpha` hadi thamani za chini wakati wa mafunzo. Katika mfano huu tutaiweka kuwa ya kudumu kwa urahisi, na unaweza kujaribu kurekebisha thamani za `alpha` baadaye.

    `gamma` ni **discount factor** inayonyesha kiwango ambacho tunapaswa kuzingatia malipo ya baadaye zaidi ya malipo ya sasa.

    `epsilon` ni **exploration/exploitation factor** inayodhamiria kama tunapaswa kupendelea uchunguzi au matumizi. Katika algoriti yetu, tutachagua hatua inayofuata kulingana na thamani za Q-Table kwa asilimia ya `epsilon`, na katika idadi iliyobaki ya kesi tutatekeleza hatua ya nasibu. Hii itaturuhusu kuchunguza maeneo ya nafasi ya utafutaji ambayo hatujawahi kuona hapo awali. 

    ✅ Kwa suala la kusawazisha - kuchagua hatua ya nasibu (uchunguzi) itakuwa kama pigo la nasibu katika mwelekeo usio sahihi, na fimbo italazimika kujifunza jinsi ya kurejesha usawa kutoka kwa "makosa" hayo.

### Boresha Algoriti

Tunaweza pia kufanya maboresho mawili kwenye algoriti yetu kutoka somo lililopita:

- **Hesabu wastani wa malipo ya jumla**, katika idadi ya simulizi. Tutachapisha maendeleo kila majaribio 5000, na tutapunguza wastani wa malipo yetu ya jumla katika kipindi hicho cha muda. Inamaanisha kwamba ikiwa tutapata zaidi ya pointi 195 - tunaweza kuzingatia tatizo limetatuliwa, kwa ubora wa juu zaidi kuliko inavyohitajika.
  
- **Hesabu matokeo ya juu ya wastani wa jumla**, `Qmax`, na tutahifadhi Q-Table inayohusiana na matokeo hayo. Unapokimbia mafunzo utagundua kwamba wakati mwingine matokeo ya wastani ya jumla yanaanza kushuka, na tunataka kuhifadhi thamani za Q-Table zinazohusiana na mfano bora uliotazamwa wakati wa mafunzo.

1. Kusanya malipo yote ya jumla katika kila simulizi kwenye vector ya `rewards` kwa ajili ya kuchora baadaye. (msimbo wa block 11)

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

Unachoweza kugundua kutoka kwa matokeo hayo:

- **Karibu na lengo letu**. Tuko karibu sana kufanikisha lengo la kupata malipo ya jumla ya 195 katika majaribio 100+ mfululizo ya simulizi, au tunaweza kuwa tumelifanikisha! Hata kama tunapata namba ndogo, bado hatujui, kwa sababu tunapunguza wastani wa majaribio 5000, na majaribio 100 tu yanahitajika katika vigezo rasmi.
  
- **Malipo yanaanza kushuka**. Wakati mwingine malipo yanaanza kushuka, ambayo inamaanisha kwamba tunaweza "kuharibu" thamani zilizojifunza tayari katika Q-Table na zile zinazofanya hali kuwa mbaya zaidi.

Uchunguzi huu unaonekana wazi zaidi ikiwa tunachora maendeleo ya mafunzo.

## Kuchora Maendeleo ya Mafunzo

Wakati wa mafunzo, tumekusanya thamani ya malipo ya jumla katika kila mojawapo ya majaribio kwenye vector ya `rewards`. Hivi ndivyo inavyoonekana tunapochora dhidi ya namba ya majaribio:

```python
plt.plot(rewards)
```

![Maendeleo ya awali](../../../../8-Reinforcement/2-Gym/images/train_progress_raw.png)

Kutoka kwenye grafu hii, haiwezekani kusema chochote, kwa sababu kutokana na asili ya mchakato wa mafunzo wa nasibu urefu wa vipindi vya mafunzo hutofautiana sana. Ili kufanya grafu hii iwe na maana zaidi, tunaweza kuhesabu **wastani wa kukimbia** katika mfululizo wa majaribio, tuseme 100. Hii inaweza kufanywa kwa urahisi kwa kutumia `np.convolve`: (msimbo wa block 12)

```python
def running_average(x,window):
    return np.convolve(x,np.ones(window)/window,mode='valid')

plt.plot(running_average(rewards,100))
```

![Maendeleo ya mafunzo](../../../../8-Reinforcement/2-Gym/images/train_progress_runav.png)

## Kubadilisha Hyperparameters

Ili kufanya mafunzo kuwa thabiti zaidi, ina maana kurekebisha baadhi ya hyperparameters zetu wakati wa mafunzo. Haswa:

- **Kwa learning rate**, `alpha`, tunaweza kuanza na thamani karibu na 1, kisha kuendelea kupunguza parameter. Kwa muda, tutakuwa tunapata thamani nzuri za uwezekano katika Q-Table, na kwa hivyo tunapaswa kuzirekebisha kidogo, na si kuandika upya kabisa na thamani mpya.

- **Ongeza epsilon**. Tunaweza kutaka kuongeza `epsilon` polepole, ili kuchunguza kidogo na kutumia zaidi. Inawezekana ina maana kuanza na thamani ya chini ya `epsilon`, na kuendelea hadi karibu na 1.
> **Kazi ya 1**: Cheza na thamani za hyperparameter na uone kama unaweza kufikia zawadi ya juu zaidi ya jumla. Je, unapata zaidi ya 195?
> **Kazi ya 2**: Ili kutatua tatizo rasmi, unahitaji kupata wastani wa zawadi ya 195 katika mizunguko 100 mfululizo. Pima hilo wakati wa mafunzo na hakikisha kuwa umetatua tatizo rasmi!

## Kuona matokeo kwa vitendo

Itakuwa ya kuvutia kuona jinsi mfano uliopata mafunzo unavyofanya kazi. Hebu tuendeshe simulizi na kufuata mkakati wa kuchagua hatua kama wakati wa mafunzo, tukichagua kulingana na mgawanyo wa uwezekano katika Q-Table: (sehemu ya msimbo 13)

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

![gari la kusawazisha pole](../../../../8-Reinforcement/2-Gym/images/cartpole-balance.gif)

---

## 🚀Changamoto

> **Kazi ya 3**: Hapa, tulikuwa tunatumia nakala ya mwisho ya Q-Table, ambayo huenda isiwe bora zaidi. Kumbuka kuwa tumetunza Q-Table inayofanya vizuri zaidi katika `Qbest`! Jaribu mfano huo huo ukitumia Q-Table inayofanya vizuri zaidi kwa kunakili `Qbest` kwenda `Q` na uone kama unagundua tofauti.

> **Kazi ya 4**: Hapa hatukuchagua hatua bora katika kila hatua, bali tulichagua kulingana na mgawanyo wa uwezekano unaolingana. Je, ingekuwa na maana zaidi kuchagua hatua bora kila wakati, yenye thamani ya juu zaidi katika Q-Table? Hili linaweza kufanyika kwa kutumia kazi ya `np.argmax` ili kupata namba ya hatua inayolingana na thamani ya juu zaidi ya Q-Table. Tekeleza mkakati huu na uone kama unaboreshwa usawazishaji.

## [Jaribio la baada ya somo](https://ff-quizzes.netlify.app/en/ml/)

## Kazi ya Nyumbani
[Fundisha Gari la Mlima](assignment.md)

## Hitimisho

Sasa tumejifunza jinsi ya kufundisha mawakala kufanikisha matokeo mazuri kwa kuwapa tu kazi ya zawadi inayofafanua hali inayotakiwa ya mchezo, na kwa kuwapa fursa ya kuchunguza kwa akili nafasi ya utafutaji. Tumetumia kwa mafanikio algoriti ya Q-Learning katika hali za mazingira ya kidijitali na endelevu, lakini kwa hatua za kidijitali.

Ni muhimu pia kusoma hali ambapo hali ya hatua ni endelevu, na wakati nafasi ya uchunguzi ni ngumu zaidi, kama picha kutoka skrini ya mchezo wa Atari. Katika matatizo hayo mara nyingi tunahitaji kutumia mbinu za kujifunza mashine zenye nguvu zaidi, kama mitandao ya neva, ili kufanikisha matokeo mazuri. Mada hizo za juu zaidi ni somo la kozi yetu ya AI ya juu inayokuja.

---

**Kanusho**:  
Hati hii imetafsiriwa kwa kutumia huduma ya kutafsiri ya AI [Co-op Translator](https://github.com/Azure/co-op-translator). Ingawa tunajitahidi kuhakikisha usahihi, tafadhali fahamu kuwa tafsiri za kiotomatiki zinaweza kuwa na makosa au kutokuwa sahihi. Hati asilia katika lugha yake ya awali inapaswa kuchukuliwa kama chanzo cha mamlaka. Kwa taarifa muhimu, tafsiri ya kitaalamu ya binadamu inapendekezwa. Hatutawajibika kwa kutoelewana au tafsiri zisizo sahihi zinazotokana na matumizi ya tafsiri hii.