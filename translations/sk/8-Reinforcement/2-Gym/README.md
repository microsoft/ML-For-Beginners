# CartPole korčuľovanie

Problém, ktorý sme riešili v predchádzajúcej lekcii, sa môže javiť ako odľahčený problém, ktorý nie je naozaj použiteľný v reálnych situáciách. To však nie je pravda, pretože mnohé problémy zo skutočného sveta zdieľajú tento scenár - vrátane hrania šachu alebo Go. Sú podobné, pretože máme šachovnicu s danými pravidlami a **diskrétny stav**.

## [Prednáškový kvíz](https://ff-quizzes.netlify.app/en/ml/)

## Úvod

V tejto lekcii aplikujeme rovnaké princípy Q-Learningu na problém s **kontinuálnym stavom**, teda stavom, ktorý je určený jedným alebo viacerými reálnymi číslami. Budeme riešiť nasledujúci problém:

> **Problém**: Ak chce Peter uniknúť vlkovi, musí sa vedieť pohybovať rýchlejšie. Ukážeme si, ako sa Peter môže naučiť korčuľovať, konkrétne ako udržať rovnováhu, pomocou Q-Learningu.

![Veľký únik!](../../../../translated_images/sk/escape.18862db9930337e3.webp)

> Peter a jeho priatelia sú kreatívni, aby unikli vlkovi! Obrázok od [Jen Looper](https://twitter.com/jenlooper)

Použijeme zjednodušenú verziu balančného problému známeho ako **CartPole**. V svete cartpole máme horizontálny posuvník, ktorý sa môže pohybovať doľava alebo doprava, a cieľom je vyvážiť vertikálnu tyč na vrchu posuvníka.

<img alt="cartpole" src="../../../../translated_images/sk/cartpole.b5609cc0494a14f7.webp" width="200"/>

## Predpoklady

V tejto lekcii použijeme knižnicu s názvom **OpenAI Gym** na simuláciu rôznych **prostredí**. Môžete spustiť kód tejto lekcie lokálne (napr. z Visual Studio Code), v takom prípade sa simulácia otvorí v novom okne. Pri spúšťaní kódu online možno bude potrebné urobiť niektoré úpravy kódu, ako je popísané [tu](https://towardsdatascience.com/rendering-openai-gym-envs-on-binder-and-google-colab-536f99391cc7).

## OpenAI Gym

V predchádzajúcej lekcii nám pravidlá hry a stav definovala trieda `Board`, ktorú sme si definovali sami. Tu použijeme špeciálne **simulačné prostredie**, ktoré bude simulovať fyziku za balančnou tyčou. Jedným z najpopulárnejších simulačných prostredí na trénovanie algoritmov posilňovaného učenia je [Gym](https://gym.openai.com/), ktorý spravuje [OpenAI](https://openai.com/). Použitím tohto gym môžeme vytvárať rôzne **prostredia** od simulácie cartpole až po Atari hry.

> **Poznámka**: Ďalšie dostupné prostredia OpenAI Gym môžete vidieť [tu](https://gym.openai.com/envs/#classic_control). 

Najskôr si nainštalujme gym a naimportujme potrebné knižnice (kódový blok 1):

```python
import sys
!{sys.executable} -m pip install gym 

import gym
import matplotlib.pyplot as plt
import numpy as np
import random
```

## Cvičenie – inicializácia prostredia cartpole

Na prácu s problémom balančnej tyče musíme inicializovať zodpovedajúce prostredie. Každé prostredie je spojené s:

- **Prieskumný priestor** (Observation space), ktorý definuje štruktúru informácií, ktoré dostávame z prostredia. Pre cartpole problém dostávame polohu tyče, rýchlosť a niektoré ďalšie hodnoty.

- **Akčný priestor** (Action space), ktorý definuje možné akcie. V našom prípade je akčný priestor diskrétny a skladá sa z dvoch akcií – **doľava** a **doprava**. (kódový blok 2)

1. Na inicializáciu zadajte nasledujúci kód:

    ```python
    env = gym.make("CartPole-v1")
    print(env.action_space)
    print(env.observation_space)
    print(env.action_space.sample())
    ```

Aby sme videli, ako prostredie funguje, spustime krátku simuláciu na 100 krokov. Na každom kroku vykonáme jednu z akcií – v tejto simulácii náhodne vyberieme akciu z `action_space`. 

1. Spustite kód nižšie a pozrite sa, čo z toho vyjde.

    ✅ Pamätajte, že je vhodnejšie spustiť tento kód na lokálnej inštalácii Pythonu! (kódový blok 3)

    ```python
    env.reset()
    
    for i in range(100):
       env.render()
       env.step(env.action_space.sample())
    env.close()
    ```

    Mali by ste vidieť niečo podobné tomuto obrázku:

    ![cartpole bez rovnováhy](../../../../8-Reinforcement/2-Gym/images/cartpole-nobalance.gif)

1. Počas simulácie potrebujeme získať pozorovania, aby sme sa rozhodli, ako konať. Funkcia `step` vráti aktuálne pozorovanie, odmenu a príznak `done`, ktorý označuje, či má zmysel v simulácii pokračovať alebo nie: (kódový blok 4)

    ```python
    env.reset()
    
    done = False
    while not done:
       env.render()
       obs, rew, done, info = env.step(env.action_space.sample())
       print(f"{obs} -> {rew}")
    env.close()
    ```

    Výstup v notebooku bude vyzerať nejako takto:

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

    Pozorovací vektor, ktorý sa vracia na každom kroku simulácie, obsahuje nasledujúce hodnoty:
    - Poloha vozíka
    - Rýchlosť vozíka
    - Uhol tyče
    - Rýchlosť rotácie tyče

1. Zistite minimum a maximum týchto hodnôt: (kódový blok 5)

    ```python
    print(env.observation_space.low)
    print(env.observation_space.high)
    ```

    Tiež si môžete všimnúť, že hodnota odmeny pri každom kroku simulácie je vždy 1. Je to preto, že naším cieľom je prežiť čo najdlhšie, teda udržať tyč v primerane vertikálnej polohe čo najdlhšie.

    ✅ Simulácia CartPole je považovaná za vyriešenú, ak dosiahneme priemernú odmenu 195 za 100 po sebe nasledujúcich pokusov.

## Diskretizácia stavu

V Q-Learningu potrebujeme vytvoriť Q-tabulu, ktorá definuje, čo robiť v každom stave. Aby sme to mohli urobiť, potrebujeme, aby bol stav **diskrétny**, presnejšie, aby obsahoval konečný počet diskrétnych hodnôt. Preto musíme nejako **diskretizovať** naše pozorovania a namapovať ich na konečnú množinu stavov.

Existuje niekoľko spôsobov, ako to môžeme robiť:

- **Rozdeliť do intervalov (bins)**. Ak poznáme interval určitej hodnoty, môžeme tento interval rozdeliť na niekoľko **intervalov (bins)** a následne hodnotu nahradiť číslom intervalu, do ktorého patrí. To sa dá urobiť pomocou numpy metódy [`digitize`](https://numpy.org/doc/stable/reference/generated/numpy.digitize.html). V tomto prípade presne poznáme veľkosť stavu, pretože bude závisieť od počtu intervalov, ktoré vyberieme pre digitalizáciu.
  
✅ Môžeme použiť lineárnu interpoláciu na prenesenie hodnôt do nejakého konečného intervalu (napr. od -20 do 20) a potom čísla previesť na celé čísla zaokrúhlením. Toto nám dá trochu menej kontroly nad veľkosťou stavu, najmä ak nepoznáme presné rozsahy vstupných hodnôt. Napríklad v našom prípade 2 zo 4 hodnôt nemajú horné alebo dolné hranice, čo by mohlo viesť k nekonečnému počtu stavov.

V našom príklade použijeme druhý prístup. Ako si neskôr všimnete, napriek nedefinovaným horným a dolným hraniciam tieto hodnoty zriedkavo nadobúdajú hodnoty mimo určitých konečných intervalov, takže stavy s extrémnymi hodnotami budú veľmi zriedkavé.

1. Tu je funkcia, ktorá vezme pozorovanie nášho modelu a vráti 4-ticu celých čísel: (kódový blok 6)

    ```python
    def discretize(x):
        return tuple((x/np.array([0.25, 0.25, 0.01, 0.1])).astype(np.int))
    ```

1. Preskúmajme aj druhý spôsob diskretizácie pomocou intervalov: (kódový blok 7)

    ```python
    def create_bins(i,num):
        return np.arange(num+1)*(i[1]-i[0])/num+i[0]
    
    print("Sample bins for interval (-5,5) with 10 bins\n",create_bins((-5,5),10))
    
    ints = [(-5,5),(-2,2),(-0.5,0.5),(-2,2)] # intervaly hodnôt pre každý parameter
    nbins = [20,20,10,10] # počet priehradiek pre každý parameter
    bins = [create_bins(ints[i],nbins[i]) for i in range(4)]
    
    def discretize_bins(x):
        return tuple(np.digitize(x[i],bins[i]) for i in range(4))
    ```

1. Teraz spustime krátku simuláciu a pozorujme tieto diskrétne hodnoty prostredia. Kľudne vyskúšajte obe funkcie `discretize` a `discretize_bins` a uvidíte, či je rozdiel.

    ✅ `discretize_bins` vracia číslo intervalu, ktoré je indexované od 0. Preto pre hodnoty vstupnej premennej okolo 0 vracia číslo zo stredu intervalu (10). Vo `discretize` sme nerozlišovali rozsah výstupných hodnôt, ktoré mohli byť aj záporné, takže hodnoty stavu nie sú posunuté a 0 zodpovedá 0. (kódový blok 8)

    ```python
    env.reset()
    
    done = False
    while not done:
       #env.render()
       obs, rew, done, info = env.step(env.action_space.sample())
       #print(diskretizované_segmenty(obs))
       print(discretize(obs))
    env.close()
    ```

    ✅ Odinštalujte riadok začínajúci `env.render`, ak chcete vidieť, ako prostredie vykonáva simuláciu. Inak môžete simuláciu nechať bežať na pozadí, čo je rýchlejšie. Túto „neviditeľnú“ simuláciu použijeme počas Q-Learning procesu.

## Štruktúra Q-tabule

V predchádzajúcej lekcii bol stav jednoduchý pár čísel od 0 do 8, preto bolo pohodlné reprezentovať Q-tabulu ako numpy tenzor s tvarom 8x8x2. Ak použijeme diskretizáciu pomocou intervalov, veľkosť nášho vektorového stavu je tiež známa, takže môžeme použiť rovnaký prístup a reprezentovať stav ako pole tvaru 20x20x10x10x2 (tu je 2 rozmer akčného priestoru a prvé rozmery zodpovedajú počtu intervalov, ktoré sme vybrali pre každý parameter v pozorovacom priestore).

Niekedy však presné rozmery pozorovacieho priestoru nie sú známe. V prípade funkcie `discretize` nemusíme byť istí, že náš stav zostane v určitých hraniciach, pretože niektoré z pôvodných hodnôt nie sú ohraničené. Preto použijeme trochu iný prístup a budeme reprezentovať Q-tabulu ako slovník.

1. Použite dvojicu *(stav, akcia)* ako kľúč slovníka a hodnota bude zodpovedať hodnote položky v Q-tabuli. (kódový blok 9)

    ```python
    Q = {}
    actions = (0,1)
    
    def qvalues(state):
        return [Q.get((state,a),0) for a in actions]
    ```

    Tu tiež definujeme funkciu `qvalues()`, ktorá vráti zoznam hodnôt Q-tabule pre daný stav zodpovedajúci všetkým možným akciám. Ak položka v Q-tabuli neexistuje, vrátime 0 ako predvolenú hodnotu.

## Začnime Q-Learning

Teraz sme pripravení naučiť Petra, ako udržať rovnováhu!

1. Najprv nastavme niektoré hyperparametre: (kódový blok 10)

    ```python
    # hyperparametre
    alpha = 0.3
    gamma = 0.9
    epsilon = 0.90
    ```

    Tu je `alpha` **rýchlosť učenia**, ktorá definuje, do akej miery by sme mali upravovať aktuálne hodnoty Q-tabule na každom kroku. V predchádzajúcej lekcii sme začínali s hodnotou 1 a potom sme postupne znižovali `alpha` počas tréningu. V tomto príklade ju necháme konštantnú, pre jednoduchosť, a môžete si neskôr experimentovať s jej nastavením.

    `gamma` je **diskontný faktor**, ktorý ukazuje, do akej miery by sme mali uprednostňovať budúce odmeny oproti súčasným.

    `epsilon` je **faktor prieskumu/využitia**, ktorý určuje, či by sme mali uprednostniť prieskum pred využívaním alebo naopak. V našom algoritme v `epsilon` percentách prípadov vyberieme nasledujúcu akciu podľa hodnôt Q-tabule a v ostatných prípadoch vykonáme náhodnú akciu. To nám umožní preskúmať oblasti vyhľadávacieho priestoru, ktoré sme ešte nevideli. 

    ✅ Pokiaľ ide o balančné problémy - voľba náhodnej akcie (prieskum) by bola ako náhodný úder zlým smerom a tyč by sa musela naučiť, ako z týchto „chýb“ obnoviť rovnováhu.

### Vylepšiť algoritmus

Môžeme tiež urobiť dve vylepšenia nášho algoritmu z predchádzajúcej lekcie:

- **Vypočítať priemernú kumulatívnu odmenu** za určitý počet simulácií. Budeme tlačiť priebeh každých 5000 iterácií a za toto obdobie vyhladia kumulatívnu odmenu. To znamená, že ak dosiahneme viac ako 195 bodov, môžeme považovať problém za vyriešený, dokonca s ešte lepšou kvalitou.
  
- **Vypočítať maximálny priemerný kumulatívny výsledok**, `Qmax`, a uložíme Q-tabulu príslušnú k tomuto výsledku. Pri tréningu často uvidíte, že priemerný kumulatívny výsledok začne klesať a my chceme uchovať hodnoty Q-tabule, ktoré zodpovedajú najlepšiemu modelu pozorovanému počas tréningu.

1. Zhromaždite všetky kumulatívne odmeny pri každej simulácii do vektora `rewards` pre ďalšie znázornenie. (kódový blok 11)

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
        # == vykonaj simuláciu ==
        while not done:
            s = discretize(obs)
            if random.random()<epsilon:
                # vyťaženie - vyber akciu podľa pravdepodobností v Q-tabulke
                v = probs(np.array(qvalues(s)))
                a = random.choices(actions,weights=v)[0]
            else:
                # prieskum - náhodne vyber akciu
                a = np.random.randint(env.action_space.n)
    
            obs, rew, done, info = env.step(a)
            cum_reward+=rew
            ns = discretize(obs)
            Q[(s,a)] = (1 - alpha) * Q.get((s,a),0) + alpha * (rew + gamma * max(qvalues(ns)))
        cum_rewards.append(cum_reward)
        rewards.append(cum_reward)
        # == Periodicky vypíš výsledky a vypočítaj priemernú odmenu ==
        if epoch%5000==0:
            print(f"{epoch}: {np.average(cum_rewards)}, alpha={alpha}, epsilon={epsilon}")
            if np.average(cum_rewards) > Qmax:
                Qmax = np.average(cum_rewards)
                Qbest = Q
            cum_rewards=[]
    ```

Čo si môžete všimnúť z týchto výsledkov:

- **Blízko nášmu cieľu**. Sme veľmi blízko dosiahnutia ciela, ktorý je získať 195 kumulatívnych odmien za 100+ po sebe nasledujúcich spustení simulácie, alebo sme ho možno už dosiahli! Aj keď dostaneme menšie čísla, ešte to nevieme, pretože vyhladzujeme cez 5000 spustení a formálne je potrebných len 100.
  
- **Odmena začína klesať**. Niekedy odmena začne klesať, čo znamená, že môžeme „zničiť“ už naučené hodnoty v Q-tabuli novými, ktoré situáciu zhoršujú.

Toto pozorovanie je jasnejšie viditeľné, keď si zobrazíme priebeh tréningu.

## Grafický priebeh tréningu

Počas tréningu sme zhromaždili kumulatívnu odmenu v každej iterácii do vektora `rewards`. Takto to vyzerá, keď vyobrazíme túto hodnotu proti číslu iterácie:

```python
plt.plot(rewards)
```

![surový priebeh](../../../../translated_images/sk/train_progress_raw.2adfdf2daea09c59.webp)

Z tohto grafu nemožno nič vyčítať, pretože vzhľadom na povahu stochastického tréningového procesu sa dĺžka tréningových sekvencií veľmi líši. Aby to dávalo väčší zmysel, môžeme vypočítať **bežiaci priemer** cez sériu experimentov, povedzme 100. To sa dá pohodlne urobiť pomocou `np.convolve`: (kódový blok 12)

```python
def running_average(x,window):
    return np.convolve(x,np.ones(window)/window,mode='valid')

plt.plot(running_average(rewards,100))
```

![priebeh tréningu](../../../../translated_images/sk/train_progress_runav.c71694a8fa9ab359.webp)

## Menenie hyperparametrov

Pre stabilnejšie učenie je rozumné počas tréningu niektoré hyperparametre upravovať. Najmä:

- **Pre rýchlosť učenia** `alpha` môžeme začať s hodnotami blízkymi 1 a postupne parameter znižovať. Postupom času budeme mať dobré pravdepodobnostné hodnoty v Q-tabuli a preto ich budeme upravovať len mierne, nie úplne meniť na nové hodnoty.

- **Zvýšiť epsilon**. Môžeme postupne zvyšovať `epsilon`, aby sme menej preskúmavali a viac využívali. Pravdepodobne je rozumné začať s nízkou hodnotou `epsilon` a postupne ju zvyšovať takmer na 1.

> **Úloha 1**: Hrajte sa s hodnotami hyperparametrov a skúste dosiahnuť vyššiu kumulatívnu odmenu. Dosahujete hodnoty nad 195?


> **Úloha 2**: Na formálne vyriešenie problému potrebujete dosiahnuť priemernú odmenu 195 za 100 po sebe nasledujúcich behov. Merajte to počas tréningu a uistite sa, že ste problém formálne vyriešili!

## Vidieť výsledok v akcii

Bolo by zaujímavé skutočne vidieť, ako sa trénovaný model správa. Spustime simuláciu a použime rovnakú stratégiu výberu akcií ako počas tréningu, vzorkujúc podľa pravdepodobnostného rozdelenia v Q-tabulke: (kódový blok 13)

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

![a balancing cartpole](../../../../8-Reinforcement/2-Gym/images/cartpole-balance.gif)

---

## 🚀Výzva

> **Úloha 3**: Tu sme používali finálnu kópiu Q-tabulky, ktorá nemusí byť najlepšia. Pamätajte, že sme uložili najlepšie fungujúcu Q-tabulku do premennej `Qbest`! Skúste ten istý príklad s najlepšie fungujúcou Q-tabulkou tak, že prekopírujete `Qbest` do `Q` a uvidíte, či si všimnete rozdiel.

> **Úloha 4**: Tu sme nevyberali najlepšiu akciu v každom kroku, ale vzorkovali sme podľa zodpovedajúceho pravdepodobnostného rozdelenia. Dávalo by väčší zmysel vždy zvoliť najlepšiu akciu, s najvyššou hodnotou Q-tabulky? Toto sa dá urobiť použitím funkcie `np.argmax` na zistenie čísla akcie zodpovedajúcej najvyššej hodnote v Q-tabulke. Implementujte túto stratégiu a uvidíte, či to vylepší balansovanie.

## [Post-lekčný kvíz](https://ff-quizzes.netlify.app/en/ml/)

## Zadanie
[Natrénuj Mountain Car](assignment.md)

## Záver

Teraz sme sa naučili, ako trénovať agentov na dosahovanie dobrých výsledkov len tým, že im poskytneme odmeňovaciu funkciu, ktorá definuje požadovaný stav hry, a tým, že im umožníme inteligentne skúmať vyhľadávací priestor. Úspešne sme aplikovali algoritmus Q-learning v prípadoch diskrétnych a súvislých prostredí, ale s diskrétnymi akciami.

Je dôležité tiež študovať situácie, kde je stav akcie tiež súvislý, a keď je pozorovací priestor oveľa zložitejší, napríklad obraz z obrazovky hry Atari. V týchto problémoch často potrebujeme použiť výkonnejšie metódy strojového učenia, ako sú neurónové siete, aby sme dosiahli dobré výsledky. Témy s vyššou náročnosťou sú predmetom nášho pripravovaného pokročilejšieho kurzu AI.

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**Vyhlásenie o zodpovednosti**:
Tento dokument bol preložený pomocou AI prekladateľskej služby [Co-op Translator](https://github.com/Azure/co-op-translator). Hoci sa snažíme o presnosť, vezmite prosím na vedomie, že automatické preklady môžu obsahovať chyby alebo nepresnosti. Pôvodný dokument v jeho natívnom jazyku by mal byť považovaný za autoritatívny zdroj. Pre kritické informácie sa odporúča profesionálny ľudský preklad. Nie sme zodpovední za žiadne nedorozumenia alebo nesprávne interpretácie vyplývajúce z použitia tohto prekladu.
<!-- CO-OP TRANSLATOR DISCLAIMER END -->