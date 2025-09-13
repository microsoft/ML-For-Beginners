<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "107d5bb29da8a562e7ae72262d251a75",
  "translation_date": "2025-09-05T16:46:17+00:00",
  "source_file": "8-Reinforcement/2-Gym/README.md",
  "language_code": "sk"
}
-->
## Predpoklady

V tejto lekcii budeme používať knižnicu **OpenAI Gym** na simuláciu rôznych **prostredí**. Kód z tejto lekcie môžete spustiť lokálne (napr. vo Visual Studio Code), v takom prípade sa simulácia otvorí v novom okne. Pri spúšťaní kódu online môže byť potrebné upraviť kód, ako je popísané [tu](https://towardsdatascience.com/rendering-openai-gym-envs-on-binder-and-google-colab-536f99391cc7).

## OpenAI Gym

V predchádzajúcej lekcii boli pravidlá hry a stav definované triedou `Board`, ktorú sme si sami vytvorili. Tu použijeme špeciálne **simulačné prostredie**, ktoré bude simulovať fyziku za balansujúcou tyčou. Jedným z najpopulárnejších simulačných prostredí na trénovanie algoritmov posilneného učenia je [Gym](https://gym.openai.com/), ktorý spravuje [OpenAI](https://openai.com/). Pomocou Gym môžeme vytvárať rôzne **prostredia**, od simulácie CartPole až po hry Atari.

> **Poznámka**: Ďalšie prostredia dostupné v OpenAI Gym si môžete pozrieť [tu](https://gym.openai.com/envs/#classic_control).

Najskôr nainštalujme Gym a importujme potrebné knižnice (kódový blok 1):

```python
import sys
!{sys.executable} -m pip install gym 

import gym
import matplotlib.pyplot as plt
import numpy as np
import random
```

## Cvičenie - inicializácia prostredia CartPole

Aby sme mohli pracovať s problémom balansovania CartPole, musíme inicializovať príslušné prostredie. Každé prostredie je spojené s:

- **Priestorom pozorovaní**, ktorý definuje štruktúru informácií, ktoré dostávame z prostredia. Pri probléme CartPole dostávame polohu tyče, rýchlosť a ďalšie hodnoty.

- **Priestorom akcií**, ktorý definuje možné akcie. V našom prípade je priestor akcií diskrétny a pozostáva z dvoch akcií - **vľavo** a **vpravo**. (kódový blok 2)

1. Na inicializáciu zadajte nasledujúci kód:

    ```python
    env = gym.make("CartPole-v1")
    print(env.action_space)
    print(env.observation_space)
    print(env.action_space.sample())
    ```

Aby sme videli, ako prostredie funguje, spustime krátku simuláciu na 100 krokov. Pri každom kroku poskytneme jednu z akcií, ktoré sa majú vykonať - v tejto simulácii náhodne vyberáme akciu z `action_space`.

1. Spustite nasledujúci kód a pozrite sa, čo sa stane.

    ✅ Pamätajte, že je preferované spustiť tento kód na lokálnej inštalácii Pythonu! (kódový blok 3)

    ```python
    env.reset()
    
    for i in range(100):
       env.render()
       env.step(env.action_space.sample())
    env.close()
    ```

    Mali by ste vidieť niečo podobné ako na tomto obrázku:

    ![nebalansujúci CartPole](../../../../8-Reinforcement/2-Gym/images/cartpole-nobalance.gif)

1. Počas simulácie musíme získavať pozorovania, aby sme sa rozhodli, ako konať. V skutočnosti funkcia `step` vracia aktuálne pozorovania, funkciu odmeny a príznak `done`, ktorý indikuje, či má zmysel pokračovať v simulácii alebo nie: (kódový blok 4)

    ```python
    env.reset()
    
    done = False
    while not done:
       env.render()
       obs, rew, done, info = env.step(env.action_space.sample())
       print(f"{obs} -> {rew}")
    env.close()
    ```

    V notebooku by ste mali vidieť niečo podobné ako toto:

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

    Vektor pozorovaní, ktorý sa vracia pri každom kroku simulácie, obsahuje nasledujúce hodnoty:
    - Poloha vozíka
    - Rýchlosť vozíka
    - Uhol tyče
    - Rýchlosť rotácie tyče

1. Získajte minimálnu a maximálnu hodnotu týchto čísel: (kódový blok 5)

    ```python
    print(env.observation_space.low)
    print(env.observation_space.high)
    ```

    Môžete si tiež všimnúť, že hodnota odmeny pri každom kroku simulácie je vždy 1. Je to preto, že naším cieľom je prežiť čo najdlhšie, t. j. udržať tyč v primerane vertikálnej polohe čo najdlhšie.

    ✅ V skutočnosti sa simulácia CartPole považuje za vyriešenú, ak sa nám podarí dosiahnuť priemernú odmenu 195 počas 100 po sebe nasledujúcich pokusov.

## Diskretizácia stavu

Pri Q-Learningu musíme vytvoriť Q-Tabuľku, ktorá definuje, čo robiť v každom stave. Aby sme to mohli urobiť, potrebujeme, aby bol stav **diskrétny**, presnejšie, aby obsahoval konečný počet diskrétnych hodnôt. Preto musíme nejako **diskretizovať** naše pozorovania, mapovať ich na konečnú množinu stavov.

Existuje niekoľko spôsobov, ako to urobiť:

- **Rozdelenie na intervaly**. Ak poznáme interval určitej hodnoty, môžeme tento interval rozdeliť na niekoľko **intervalov** a potom nahradiť hodnotu číslom intervalu, do ktorého patrí. To sa dá urobiť pomocou metódy numpy [`digitize`](https://numpy.org/doc/stable/reference/generated/numpy.digitize.html). V tomto prípade budeme presne vedieť veľkosť stavu, pretože bude závisieť od počtu intervalov, ktoré vyberieme na digitalizáciu.

✅ Môžeme použiť lineárnu interpoláciu na privedenie hodnôt do určitého konečného intervalu (napríklad od -20 do 20) a potom previesť čísla na celé čísla zaokrúhlením. To nám dáva o niečo menšiu kontrolu nad veľkosťou stavu, najmä ak nepoznáme presné rozsahy vstupných hodnôt. Napríklad v našom prípade 2 zo 4 hodnôt nemajú horné/dolné hranice svojich hodnôt, čo môže viesť k nekonečnému počtu stavov.

V našom príklade použijeme druhý prístup. Ako si neskôr všimnete, napriek nedefinovaným horným/dolným hraniciam tieto hodnoty zriedka nadobúdajú hodnoty mimo určitých konečných intervalov, takže stavy s extrémnymi hodnotami budú veľmi zriedkavé.

1. Tu je funkcia, ktorá vezme pozorovanie z nášho modelu a vytvorí z neho štvoricu 4 celých čísel: (kódový blok 6)

    ```python
    def discretize(x):
        return tuple((x/np.array([0.25, 0.25, 0.01, 0.1])).astype(np.int))
    ```

1. Preskúmajme aj ďalšiu metódu diskretizácie pomocou intervalov: (kódový blok 7)

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

1. Teraz spustime krátku simuláciu a pozorujme tieto diskrétne hodnoty prostredia. Skúste použiť `discretize` aj `discretize_bins` a zistite, či je medzi nimi rozdiel.

    ✅ `discretize_bins` vracia číslo intervalu, ktoré je 0-based. Takže pre hodnoty vstupnej premennej okolo 0 vracia číslo zo stredu intervalu (10). Pri `discretize` sme sa nestarali o rozsah výstupných hodnôt, umožnili sme im byť záporné, takže hodnoty stavu nie sú posunuté a 0 zodpovedá 0. (kódový blok 8)

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

    ✅ Odkomentujte riadok začínajúci `env.render`, ak chcete vidieť, ako sa prostredie vykonáva. Inak ho môžete spustiť na pozadí, čo je rýchlejšie. Tento "neviditeľný" spôsob vykonávania použijeme počas procesu Q-Learningu.

## Štruktúra Q-Tabuľky

V našej predchádzajúcej lekcii bol stav jednoduchou dvojicou čísel od 0 do 8, a preto bolo pohodlné reprezentovať Q-Tabuľku pomocou numpy tenzora s tvarom 8x8x2. Ak použijeme diskretizáciu pomocou intervalov, veľkosť nášho stavového vektora je tiež známa, takže môžeme použiť rovnaký prístup a reprezentovať stav pomocou poľa s tvarom 20x20x10x10x2 (tu 2 je dimenzia priestoru akcií a prvé dimenzie zodpovedajú počtu intervalov, ktoré sme vybrali na použitie pre každú z parametrov v priestore pozorovaní).

Avšak niekedy presné dimenzie priestoru pozorovaní nie sú známe. V prípade funkcie `discretize` si nikdy nemôžeme byť istí, že náš stav zostane v určitých hraniciach, pretože niektoré z pôvodných hodnôt nie sú ohraničené. Preto použijeme trochu iný prístup a reprezentujeme Q-Tabuľku pomocou slovníka.

1. Použite dvojicu *(state,action)* ako kľúč slovníka a hodnota by zodpovedala hodnote záznamu Q-Tabuľky. (kódový blok 9)

    ```python
    Q = {}
    actions = (0,1)
    
    def qvalues(state):
        return [Q.get((state,a),0) for a in actions]
    ```

    Tu tiež definujeme funkciu `qvalues()`, ktorá vracia zoznam hodnôt Q-Tabuľky pre daný stav, ktorý zodpovedá všetkým možným akciám. Ak záznam nie je prítomný v Q-Tabuľke, vrátime 0 ako predvolenú hodnotu.

## Začnime Q-Learning

Teraz sme pripravení naučiť Petra balansovať!

1. Najskôr nastavme niektoré hyperparametre: (kódový blok 10)

    ```python
    # hyperparameters
    alpha = 0.3
    gamma = 0.9
    epsilon = 0.90
    ```

    Tu `alpha` je **rýchlosť učenia**, ktorá určuje, do akej miery by sme mali upraviť aktuálne hodnoty Q-Tabuľky pri každom kroku. V predchádzajúcej lekcii sme začali s hodnotou 1 a potom sme znižovali `alpha` na nižšie hodnoty počas tréningu. V tomto príklade ju ponecháme konštantnú len pre jednoduchosť a môžete experimentovať s úpravou hodnôt `alpha` neskôr.

    `gamma` je **faktor diskontovania**, ktorý ukazuje, do akej miery by sme mali uprednostniť budúcu odmenu pred aktuálnou odmenou.

    `epsilon` je **faktor prieskumu/využitia**, ktorý určuje, či by sme mali uprednostniť prieskum pred využitím alebo naopak. V našom algoritme v `epsilon` percentách prípadov vyberieme ďalšiu akciu podľa hodnôt Q-Tabuľky a v zostávajúcom počte prípadov vykonáme náhodnú akciu. To nám umožní preskúmať oblasti vyhľadávacieho priestoru, ktoré sme nikdy predtým nevideli.

    ✅ Z hľadiska balansovania - výber náhodnej akcie (prieskum) by pôsobil ako náhodný úder nesprávnym smerom a tyč by sa musela naučiť, ako obnoviť rovnováhu z týchto "chýb".

### Zlepšenie algoritmu

Môžeme tiež urobiť dve vylepšenia nášho algoritmu z predchádzajúcej lekcie:

- **Vypočítať priemernú kumulatívnu odmenu** počas niekoľkých simulácií. Pokrok budeme tlačiť každých 5000 iterácií a priemernú kumulatívnu odmenu vypočítame za toto obdobie. To znamená, že ak získame viac ako 195 bodov, môžeme problém považovať za vyriešený, dokonca s vyššou kvalitou, než je požadovaná.

- **Vypočítať maximálny priemerný kumulatívny výsledok**, `Qmax`, a uložíme Q-Tabuľku zodpovedajúcu tomuto výsledku. Keď spustíte tréning, všimnete si, že niekedy priemerný kumulatívny výsledok začne klesať, a chceme si ponechať hodnoty Q-Tabuľky, ktoré zodpovedajú najlepšiemu modelu pozorovanému počas tréningu.

1. Zbierajte všetky kumulatívne odmeny pri každej simulácii do vektora `rewards` na ďalšie vykreslenie. (kódový blok 11)

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

Čo si môžete všimnúť z týchto výsledkov:

- **Blízko nášho cieľa**. Sme veľmi blízko k dosiahnutiu cieľa získania 195 kumulatívnych odmien počas 100+ po sebe nasledujúcich simulácií, alebo sme ho možno už dosiahli! Aj keď získame menšie čísla, stále to nevieme, pretože priemerujeme cez 5000 pokusov a formálne kritérium vyžaduje iba 100 pokusov.

- **Odmena začína klesať**. Niekedy odmena začne klesať, čo znamená, že môžeme "zničiť" už naučené hodnoty v Q-Tabuľke tými, ktoré situáciu zhoršujú.

Toto pozorovanie je jasnejšie viditeľné, ak vykreslíme pokrok tréningu.

## Vykreslenie pokroku tréningu

Počas tréningu sme zbierali hodnotu kumulatívnej odmeny pri každej iterácii do vektora `rewards`. Takto to vyzerá, keď to vykreslíme proti číslu iterácie:

```python
plt.plot(rewards)
```

![surový pokrok](../../../../8-Reinforcement/2-Gym/images/train_progress_raw.png)

Z tohto grafu nie je možné nič povedať, pretože kvôli povahe stochastického tréningového procesu sa dĺžka tréningových relácií veľmi líši. Aby sme tento graf urobili zrozumiteľnejším, môžeme vypočítať **bežiaci priemer** cez sériu experimentov, povedzme 100. To sa dá pohodlne urobiť pomocou `np.convolve`: (kódový blok 12)

```python
def running_average(x,window):
    return np.convolve(x,np.ones(window)/window,mode='valid')

plt.plot(running_average(rewards,100))
```

![pokrok tréningu](../../../../8-Reinforcement/2-Gym/images/train_progress_runav.png)

## Zmena hyperparametrov

Aby bolo učenie stabilnejšie, má zmysel upraviť niektoré z našich hyperparametrov počas tréningu. Konkrétne:

- **Pre rýchlosť učenia**, `alpha`, môžeme začať s hodnotami blízkymi 1 a potom postupne znižovať parameter. S časom budeme získavať dobré pravdepodobnostné hodnoty v Q-Tabuľke, a preto by sme ich mali upravovať mierne, a nie úplne prepisovať novými hodnotami.

- **Zvýšiť epsilon**. Môžeme chcieť pomaly zvyšovať `epsilon`, aby sme menej skúmali a viac využívali. Pravdepodobne má zmysel začať s nižšou hodnotou `epsilon` a postupne ju zvýšiť takmer na 1.
> **Úloha 1**: Skúste experimentovať s hodnotami hyperparametrov a zistite, či dokážete dosiahnuť vyššiu kumulatívnu odmenu. Dosahujete viac ako 195?
> **Úloha 2**: Na formálne vyriešenie problému je potrebné dosiahnuť priemernú odmenu 195 počas 100 po sebe idúcich spustení. Merajte to počas tréningu a uistite sa, že ste problém formálne vyriešili!

## Vidieť výsledok v akcii

Bolo by zaujímavé vidieť, ako sa vyškolený model správa. Spustime simuláciu a použime rovnakú stratégiu výberu akcií ako počas tréningu, pričom vzorkujeme podľa pravdepodobnostného rozdelenia v Q-Tabuľke: (blok kódu 13)

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

Mali by ste vidieť niečo takéto:

![balansujúci cartpole](../../../../8-Reinforcement/2-Gym/images/cartpole-balance.gif)

---

## 🚀Výzva

> **Úloha 3**: Tu sme používali finálnu verziu Q-Tabuľky, ktorá nemusí byť najlepšia. Pamätajte, že sme uložili najlepšie fungujúcu Q-Tabuľku do premennej `Qbest`! Skúste ten istý príklad s najlepšie fungujúcou Q-Tabuľkou tak, že skopírujete `Qbest` do `Q` a sledujte, či si všimnete rozdiel.

> **Úloha 4**: Tu sme nevyberali najlepšiu akciu v každom kroku, ale namiesto toho vzorkovali podľa zodpovedajúceho pravdepodobnostného rozdelenia. Malo by väčší zmysel vždy vybrať najlepšiu akciu s najvyššou hodnotou v Q-Tabuľke? To sa dá urobiť pomocou funkcie `np.argmax`, ktorá nájde číslo akcie zodpovedajúce najvyššej hodnote v Q-Tabuľke. Implementujte túto stratégiu a sledujte, či zlepší balansovanie.

## [Kvíz po prednáške](https://ff-quizzes.netlify.app/en/ml/)

## Zadanie
[Vytrénujte Mountain Car](assignment.md)

## Záver

Teraz sme sa naučili, ako trénovať agentov na dosiahnutie dobrých výsledkov len tým, že im poskytneme funkciu odmeny, ktorá definuje požadovaný stav hry, a dáme im príležitosť inteligentne preskúmať priestor možností. Úspešne sme aplikovali algoritmus Q-Learning v prípadoch diskrétnych a spojitých prostredí, ale s diskrétnymi akciami.

Je dôležité študovať aj situácie, kde je stav akcií spojitý a kde je priestor pozorovaní oveľa zložitejší, napríklad obrazovka z hry Atari. Pri takýchto problémoch často potrebujeme použiť výkonnejšie techniky strojového učenia, ako sú neurónové siete, aby sme dosiahli dobré výsledky. Tieto pokročilejšie témy sú predmetom nášho nadchádzajúceho pokročilého kurzu AI.

---

**Upozornenie**:  
Tento dokument bol preložený pomocou služby AI prekladu [Co-op Translator](https://github.com/Azure/co-op-translator). Hoci sa snažíme o presnosť, prosím, berte na vedomie, že automatizované preklady môžu obsahovať chyby alebo nepresnosti. Pôvodný dokument v jeho pôvodnom jazyku by mal byť považovaný za autoritatívny zdroj. Pre kritické informácie sa odporúča profesionálny ľudský preklad. Nie sme zodpovední za akékoľvek nedorozumenia alebo nesprávne interpretácie vyplývajúce z použitia tohto prekladu.