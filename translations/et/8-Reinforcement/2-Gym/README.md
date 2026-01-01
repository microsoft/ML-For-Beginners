<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "107d5bb29da8a562e7ae72262d251a75",
  "translation_date": "2025-10-11T11:16:15+00:00",
  "source_file": "8-Reinforcement/2-Gym/README.md",
  "language_code": "et"
}
-->
# CartPole Uisutamine

Probleem, mida me eelmises tunnis lahendasime, võib tunduda mänguline ja mitte eriti elulähedane. Tegelikult see nii ei ole, sest paljud päriselulised probleemid jagavad sama stsenaariumi – näiteks malet või Go-d mängides. Need on sarnased, kuna meil on samuti mängulaud kindlate reeglitega ja **diskreetne olek**.

## [Eeltesti küsimustik](https://ff-quizzes.netlify.app/en/ml/)

## Sissejuhatus

Selles tunnis rakendame Q-õppe põhimõtteid probleemile, millel on **jätkuv olek**, st olek, mida kirjeldavad üks või mitu reaalarvu. Me tegeleme järgmise probleemiga:

> **Probleem**: Kui Peeter tahab hundi eest põgeneda, peab ta liikuma kiiremini. Me näeme, kuidas Peeter saab õppida uisutama, täpsemalt tasakaalu hoidma, kasutades Q-õpet.

![Suur põgenemine!](../../../../translated_images/escape.18862db9930337e3.et.png)

> Peeter ja tema sõbrad muutuvad loovaks, et hundi eest põgeneda! Pilt: [Jen Looper](https://twitter.com/jenlooper)

Kasutame tasakaalu lihtsustatud versiooni, mida tuntakse kui **CartPole** probleem. CartPole maailmas on meil horisontaalne liugur, mis saab liikuda vasakule või paremale, ja eesmärk on hoida vertikaalset posti liuguri peal tasakaalus.

<img alt="CartPole" src="../../../../translated_images/cartpole.b5609cc0494a14f7.et.png" width="200"/>

## Eeltingimused

Selles tunnis kasutame raamatukogu nimega **OpenAI Gym**, et simuleerida erinevaid **keskkondi**. Saate selle tunni koodi käivitada lokaalselt (nt Visual Studio Code'is), sel juhul avaneb simulatsioon uues aknas. Kui käivitate koodi veebis, peate võib-olla koodi veidi kohandama, nagu kirjeldatud [siin](https://towardsdatascience.com/rendering-openai-gym-envs-on-binder-and-google-colab-536f99391cc7).

## OpenAI Gym

Eelmises tunnis määras mängu reeglid ja oleku meie enda defineeritud `Board` klass. Siin kasutame spetsiaalset **simulatsioonikeskkonda**, mis simuleerib tasakaalu posti füüsikat. Üks populaarsemaid simulatsioonikeskkondi, mida kasutatakse tugevdusõppe algoritmide treenimiseks, on [Gym](https://gym.openai.com/), mida haldab [OpenAI](https://openai.com/). Selle Gymi abil saame luua erinevaid **keskkondi**, alates CartPole simulatsioonist kuni Atari mängudeni.

> **Märkus**: Teisi OpenAI Gymi keskkondi saate vaadata [siit](https://gym.openai.com/envs/#classic_control).

Kõigepealt installime Gymi ja impordime vajalikud teegid (koodiplokk 1):

```python
import sys
!{sys.executable} -m pip install gym 

import gym
import matplotlib.pyplot as plt
import numpy as np
import random
```

## Harjutus - CartPole keskkonna initsialiseerimine

CartPole tasakaalu probleemi lahendamiseks peame initsialiseerima vastava keskkonna. Iga keskkond on seotud:

- **Vaatlusruumiga**, mis määratleb struktuuri, mille kaudu saame keskkonnast teavet. CartPole probleemi puhul saame posti asukoha, kiiruse ja mõned muud väärtused.

- **Tegevusruumiga**, mis määratleb võimalikud tegevused. Meie puhul on tegevusruum diskreetne ja koosneb kahest tegevusest – **vasakule** ja **paremale**. (koodiplokk 2)

1. Initsialiseerimiseks sisestage järgmine kood:

    ```python
    env = gym.make("CartPole-v1")
    print(env.action_space)
    print(env.observation_space)
    print(env.action_space.sample())
    ```

Et näha, kuidas keskkond töötab, käivitame lühikese simulatsiooni 100 sammu jooksul. Igal sammul anname ühe tegevuse, mida tuleb teha – selles simulatsioonis valime juhuslikult tegevuse `action_space` hulgast.

1. Käivitage allolev kood ja vaadake, mis juhtub.

    ✅ Pidage meeles, et eelistatav on käivitada see kood lokaalses Python'i installatsioonis! (koodiplokk 3)

    ```python
    env.reset()
    
    for i in range(100):
       env.render()
       env.step(env.action_space.sample())
    env.close()
    ```

    Peaksite nägema midagi sarnast sellele pildile:

    ![tasakaalustamata CartPole](../../../../8-Reinforcement/2-Gym/images/cartpole-nobalance.gif)

1. Simulatsiooni ajal peame saama vaatlusi, et otsustada, kuidas tegutseda. Tegelikult tagastab `step` funktsioon praegused vaatlused, tasu funktsiooni ja `done` lipu, mis näitab, kas simulatsiooni jätkamine on mõistlik või mitte: (koodiplokk 4)

    ```python
    env.reset()
    
    done = False
    while not done:
       env.render()
       obs, rew, done, info = env.step(env.action_space.sample())
       print(f"{obs} -> {rew}")
    env.close()
    ```

    Näete midagi sellist oma märkmiku väljundis:

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

    Simulatsiooni iga sammu jooksul tagastatud vaatlusvektor sisaldab järgmisi väärtusi:
    - Käru asukoht
    - Käru kiirus
    - Posti nurk
    - Posti pöörlemiskiirus

1. Leidke nende numbrite minimaalne ja maksimaalne väärtus: (koodiplokk 5)

    ```python
    print(env.observation_space.low)
    print(env.observation_space.high)
    ```

    Samuti võite märgata, et tasu väärtus igal simulatsiooni sammul on alati 1. See on sellepärast, et meie eesmärk on ellu jääda nii kaua kui võimalik, st hoida post mõistlikult vertikaalses asendis võimalikult pika aja jooksul.

    ✅ Tegelikult loetakse CartPole simulatsioon lahendatuks, kui suudame saavutada keskmise tasu 195 üle 100 järjestikuse katse.

## Oleku diskretiseerimine

Q-õppes peame looma Q-tabeli, mis määratleb, mida teha igas olekus. Selleks peab olek olema **diskreetne**, täpsemalt, see peaks sisaldama piiratud arvu diskreetseid väärtusi. Seega peame kuidagi **diskretiseerima** oma vaatlused, kaardistades need piiratud olekute hulgale.

Selleks on mitu võimalust:

- **Jagamine vahemikeks**. Kui teame teatud väärtuse intervalli, saame selle intervalli jagada mitmeks **vahemikuks** ja seejärel asendada väärtuse vahemiku numbriga, kuhu see kuulub. Seda saab teha numpy [`digitize`](https://numpy.org/doc/stable/reference/generated/numpy.digitize.html) meetodiga. Sel juhul teame täpselt oleku suurust, kuna see sõltub valitud vahemike arvust.

✅ Võime kasutada lineaarset interpolatsiooni, et tuua väärtused teatud piiratud intervalli (näiteks -20 kuni 20), ja seejärel teisendada numbrid täisarvudeks, neid ümardades. See annab meile oleku suuruse üle vähem kontrolli, eriti kui me ei tea sisendväärtuste täpseid vahemikke. Näiteks meie puhul ei ole 2-st 4-st väärtusest ülemisi/alumisi piire, mis võib viia lõpmatu arvu olekuteni.

Meie näites kasutame teist lähenemist. Nagu hiljem märkate, hoolimata määramata ülemistest/alumistest piiridest, võtavad need väärtused harva teatud piiratud intervallidest väljapoole jäävaid väärtusi, seega on äärmuslike väärtustega olekud väga haruldased.

1. Siin on funktsioon, mis võtab meie mudelist vaatluse ja toodab 4 täisarvu tuple'i: (koodiplokk 6)

    ```python
    def discretize(x):
        return tuple((x/np.array([0.25, 0.25, 0.01, 0.1])).astype(np.int))
    ```

1. Uurime ka teist diskretiseerimismeetodit, kasutades vahemikke: (koodiplokk 7)

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

1. Käivitame nüüd lühikese simulatsiooni ja vaatleme neid diskreetseid keskkonna väärtusi. Proovige julgelt nii `discretize` kui ka `discretize_bins` ja vaadake, kas on erinevusi.

    ✅ `discretize_bins` tagastab vahemiku numbri, mis on 0-põhine. Seega sisendmuutuja väärtuste puhul, mis on umbes 0, tagastab see intervalli keskelt numbri (10). `discretize` puhul ei hoolinud me väljundväärtuste vahemikust, lubades neil olla negatiivsed, seega oleku väärtused ei ole nihutatud ja 0 vastab 0-le. (koodiplokk 8)

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

    ✅ Kommenteerige lahti rida, mis algab `env.render`, kui soovite näha, kuidas keskkond täitub. Vastasel juhul saate selle taustal täita, mis on kiirem. Kasutame seda "nähtamatut" täitmist Q-õppe protsessi ajal.

## Q-tabeli struktuur

Eelmises tunnis oli olek lihtne paar numbreid vahemikus 0 kuni 8, mistõttu oli mugav esitada Q-tabelit numpy tensorina kujuga 8x8x2. Kui kasutame vahemike diskretiseerimist, on meie olekuvektori suurus samuti teada, seega saame kasutada sama lähenemist ja esitada oleku massiivina kujuga 20x20x10x10x2 (siin 2 on tegevusruumi dimensioon ja esimesed dimensioonid vastavad vahemike arvule, mida oleme valinud iga vaatlusruumi parameetri jaoks).

Kuid mõnikord ei ole vaatlusruumi täpsed dimensioonid teada. `discretize` funktsiooni puhul ei pruugi me kunagi olla kindlad, et meie olek jääb teatud piiridesse, sest mõned algsed väärtused ei ole piiratud. Seetõttu kasutame veidi teistsugust lähenemist ja esitame Q-tabeli sõnastikuna.

1. Kasutage paari *(olek, tegevus)* sõnastiku võtmena ja väärtus vastaks Q-tabeli kirje väärtusele. (koodiplokk 9)

    ```python
    Q = {}
    actions = (0,1)
    
    def qvalues(state):
        return [Q.get((state,a),0) for a in actions]
    ```

    Siin määratleme ka funktsiooni `qvalues()`, mis tagastab loendi Q-tabeli väärtustest antud oleku jaoks, mis vastab kõigile võimalikele tegevustele. Kui kirje ei ole Q-tabelis olemas, tagastame vaikimisi 0.

## Alustame Q-õpet

Nüüd oleme valmis õpetama Peetrit tasakaalu hoidma!

1. Kõigepealt määrame mõned hüperparameetrid: (koodiplokk 10)

    ```python
    # hyperparameters
    alpha = 0.3
    gamma = 0.9
    epsilon = 0.90
    ```

    Siin on `alpha` **õppemäär**, mis määrab, mil määral peaksime Q-tabeli praeguseid väärtusi igal sammul kohandama. Eelmises tunnis alustasime väärtusega 1 ja vähendasime seejärel `alpha` väärtusi treeningu ajal. Selles näites hoiame selle lihtsuse huvides konstantsena ja saate hiljem katsetada `alpha` väärtuste kohandamist.

    `gamma` on **diskontomäär**, mis näitab, mil määral peaksime eelistama tulevast tasu praeguse tasu ees.

    `epsilon` on **uurimise/kasutamise tegur**, mis määrab, kas peaksime eelistama uurimist või kasutamist. Meie algoritmis valime `epsilon` protsendil juhtudest järgmise tegevuse vastavalt Q-tabeli väärtustele ja ülejäänud juhtudel teeme juhusliku tegevuse. See võimaldab meil uurida otsinguruumi piirkondi, mida me pole kunagi varem näinud.

    ✅ Tasakaalu osas – juhusliku tegevuse valimine (uurimine) toimiks juhusliku löögina vales suunas ja post peaks õppima, kuidas nendest "vigadest" tasakaalu taastada.

### Paranda algoritmi

Saame oma algoritmi eelmise tunni põhjal kahel viisil täiustada:

- **Arvuta keskmine kumulatiivne tasu** mitme simulatsiooni jooksul. Trükime progressi iga 5000 iteratsiooni järel ja keskmistame kumulatiivse tasu selle aja jooksul. See tähendab, et kui saame rohkem kui 195 punkti, võime probleemi lahendatuks pidada, isegi kõrgema kvaliteediga kui nõutud.

- **Arvuta maksimaalne keskmine kumulatiivne tulemus**, `Qmax`, ja salvestame Q-tabeli, mis vastab sellele tulemusele. Kui treeningu ajal märkate, et keskmine kumulatiivne tulemus hakkab langema, tahame säilitada Q-tabeli väärtused, mis vastavad treeningu parimale mudelile.

1. Koguge kõik kumulatiivsed tasud igal simulatsioonil `rewards` vektorisse edasiseks joonistamiseks. (koodiplokk 11)

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

Mida võite nendest tulemustest märgata:

- **Eesmärgile lähedal**. Oleme väga lähedal eesmärgi saavutamisele, saades 195 kumulatiivset tasu üle 100+ järjestikuse simulatsiooni või oleme selle tegelikult saavutanud! Isegi kui saame väiksemaid numbreid, ei tea me seda, sest keskmistame 5000 jooksu jooksul ja ametlikus kriteeriumis on nõutud ainult 100 jooksu.

- **Tasu hakkab langema**. Mõnikord hakkab tasu langema, mis tähendab, et võime "hävitada" juba õpitud väärtused Q-tabelis väärtustega, mis olukorda halvendavad.

See tähelepanek on selgem, kui joonistame treeningu progressi.

## Treeningu progressi joonistamine

Treeningu ajal kogusime kumulatiivse tasu väärtuse igal iteratsioonil `rewards` vektorisse. Siin on, kuidas see välja näeb, kui joonistame selle iteratsiooni numbri vastu:

```python
plt.plot(rewards)
```

![toores progress](../../../../translated_images/train_progress_raw.2adfdf2daea09c59.et.png)

Sellest graafikust ei ole võimalik midagi järeldada, sest stohhastilise treeningprotsessi olemuse tõttu varieerub treeningseansside pikkus suuresti. Selle graafiku mõistlikumaks muutmiseks saame arvutada **jooksva keskmise** mitme katse jooksul, näiteks 100. Seda saab mugavalt teha `np.convolve` abil: (koodiplokk 12)

```python
def running_average(x,window):
    return np.convolve(x,np.ones(window)/window,mode='valid')

plt.plot(running_average(rewards,100))
```

![treeningu progress](../../../../translated_images/train_progress_runav.c71694a8fa9ab359.et.png)

## Hüperparameetrite muutmine

Et õppimine oleks stabiilsem, on mõistlik treeningu ajal mõningaid hüperparameetreid kohandada. Eelkõige:

- **Õppemäära**, `alpha`, puhul võime alustada väärtustega, mis on lähedased 1-le, ja seejärel seda parameetrit järk-järgult vähendada. Aja jooksul saame Q-tabelis häid tõenäosusväärtusi ja seega peaksime neid veidi kohandama, mitte täielikult uute väärtustega üle kirjutama.

- **Suurenda epsilonit**. Võime soovida `epsilon` väärtust aeglaselt suurendada, et uurida vähem ja kasutada rohkem. Tõenäoliselt on mõistlik alustada madalama `epsilon` väärtusega ja liikuda peaaegu 1-ni.

> **Ülesanne 1**: Mängige hüperparameetrite väärtustega ja vaadake, kas suudate saavutada kõrgema kumulatiivse tasu. Kas jõuate üle 195?
> **Ülesanne 2**: Probleemi ametlikuks lahendamiseks peate saavutama 195 keskmise tasu 100 järjestikuse jooksu jooksul. Mõõtke seda treeningu ajal ja veenduge, et olete probleemi ametlikult lahendanud!

## Tulemust tegevuses näha

Oleks huvitav näha, kuidas treenitud mudel tegelikult käitub. Käivitame simulatsiooni ja järgime sama tegevuse valimise strateegiat nagu treeningu ajal, valides tegevusi vastavalt Q-tabeli tõenäosusjaotusele: (koodiplokk 13)

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

Peaksite nägema midagi sellist:

![tasakaalu hoidev cartpole](../../../../8-Reinforcement/2-Gym/images/cartpole-balance.gif)

---

## 🚀Väljakutse

> **Ülesanne 3**: Siin kasutasime Q-tabeli lõplikku koopiat, mis ei pruugi olla parim. Pidage meeles, et oleme salvestanud parima tulemusega Q-tabeli `Qbest` muutujasse! Proovige sama näidet parima tulemusega Q-tabeliga, kopeerides `Qbest` üle `Q`-sse ja vaadake, kas märkate erinevust.

> **Ülesanne 4**: Siin me ei valinud igal sammul parimat tegevust, vaid pigem valisime tegevusi vastavalt tõenäosusjaotusele. Kas oleks mõistlikum alati valida parim tegevus, millel on Q-tabelis kõrgeim väärtus? Seda saab teha, kasutades `np.argmax` funktsiooni, et leida tegevuse number, mis vastab kõrgeimale Q-tabeli väärtusele. Rakendage see strateegia ja vaadake, kas see parandab tasakaalu hoidmist.

## [Loengu järgne viktoriin](https://ff-quizzes.netlify.app/en/ml/)

## Ülesanne
[Treeni Mountain Car](assignment.md)

## Kokkuvõte

Oleme nüüd õppinud, kuidas treenida agente saavutama häid tulemusi, pakkudes neile ainult tasufunktsiooni, mis määratleb mängu soovitud oleku, ja andes neile võimaluse intelligentselt otsinguruumi uurida. Oleme edukalt rakendanud Q-õppe algoritmi nii diskreetsete kui ka pidevate keskkondade puhul, kuid diskreetsete tegevustega.

Oluline on uurida ka olukordi, kus tegevusruum on samuti pidev ja kui vaatlusruum on palju keerulisem, näiteks pilt Atari mängu ekraanilt. Selliste probleemide lahendamiseks on sageli vaja kasutada võimsamaid masinõppe tehnikaid, nagu näiteks tehisnärvivõrgud, et saavutada häid tulemusi. Need keerukamad teemad on meie tulevase edasijõudnute AI kursuse fookuses.

---

**Lahtiütlus**:  
See dokument on tõlgitud AI tõlketeenuse [Co-op Translator](https://github.com/Azure/co-op-translator) abil. Kuigi püüame tagada täpsust, palume arvestada, et automaatsed tõlked võivad sisaldada vigu või ebatäpsusi. Algne dokument selle algses keeles tuleks pidada autoriteetseks allikaks. Olulise teabe puhul soovitame kasutada professionaalset inimtõlget. Me ei vastuta selle tõlke kasutamisest tulenevate arusaamatuste või valesti tõlgenduste eest.