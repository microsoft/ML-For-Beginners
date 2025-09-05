<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "107d5bb29da8a562e7ae72262d251a75",
  "translation_date": "2025-09-05T01:15:06+00:00",
  "source_file": "8-Reinforcement/2-Gym/README.md",
  "language_code": "cs"
}
-->
# CartPole Bruslení

Problém, který jsme řešili v předchozí lekci, se může zdát jako hračka, která nemá skutečné využití v reálných situacích. To však není pravda, protože mnoho reálných problémů má podobný scénář – například hraní šachů nebo Go. Jsou podobné, protože také máme hrací desku s danými pravidly a **diskrétní stav**.

## [Kvíz před lekcí](https://ff-quizzes.netlify.app/en/ml/)

## Úvod

V této lekci použijeme stejné principy Q-Learningu na problém s **kontinuálním stavem**, tj. stavem, který je definován jedním nebo více reálnými čísly. Budeme se zabývat následujícím problémem:

> **Problém**: Pokud chce Petr utéct vlkovi, musí se naučit pohybovat rychleji. Uvidíme, jak se Petr může naučit bruslit, konkrétně udržovat rovnováhu, pomocí Q-Learningu.

![Velký útěk!](../../../../8-Reinforcement/2-Gym/images/escape.png)

> Petr a jeho přátelé jsou kreativní, aby unikli vlkovi! Obrázek od [Jen Looper](https://twitter.com/jenlooper)

Použijeme zjednodušenou verzi udržování rovnováhy známou jako problém **CartPole**. Ve světě CartPole máme horizontální jezdec, který se může pohybovat doleva nebo doprava, a cílem je udržet vertikální tyč na vrcholu jezdce.

## Předpoklady

V této lekci budeme používat knihovnu **OpenAI Gym** k simulaci různých **prostředí**. Kód této lekce můžete spustit lokálně (např. z Visual Studio Code), v takovém případě se simulace otevře v novém okně. Při spuštění kódu online může být nutné provést některé úpravy kódu, jak je popsáno [zde](https://towardsdatascience.com/rendering-openai-gym-envs-on-binder-and-google-colab-536f99391cc7).

## OpenAI Gym

V předchozí lekci byly pravidla hry a stav definovány třídou `Board`, kterou jsme si sami vytvořili. Zde použijeme speciální **simulační prostředí**, které bude simulovat fyziku za udržováním rovnováhy tyče. Jedno z nejpopulárnějších simulačních prostředí pro trénování algoritmů posilovaného učení se nazývá [Gym](https://gym.openai.com/), které spravuje [OpenAI](https://openai.com/). Pomocí tohoto gymu můžeme vytvořit různá **prostředí** od simulace CartPole až po hry Atari.

> **Poznámka**: Další prostředí dostupná v OpenAI Gym si můžete prohlédnout [zde](https://gym.openai.com/envs/#classic_control).

Nejprve nainstalujeme gym a importujeme potřebné knihovny (blok kódu 1):

```python
import sys
!{sys.executable} -m pip install gym 

import gym
import matplotlib.pyplot as plt
import numpy as np
import random
```

## Cvičení – inicializace prostředí CartPole

Pro práci s problémem udržování rovnováhy CartPole musíme inicializovat odpovídající prostředí. Každé prostředí je spojeno s:

- **Prostor pozorování**, který definuje strukturu informací, které získáváme z prostředí. U problému CartPole získáváme polohu tyče, rychlost a některé další hodnoty.

- **Prostor akcí**, který definuje možné akce. V našem případě je prostor akcí diskrétní a skládá se ze dvou akcí – **doleva** a **doprava**. (blok kódu 2)

1. Pro inicializaci napište následující kód:

    ```python
    env = gym.make("CartPole-v1")
    print(env.action_space)
    print(env.observation_space)
    print(env.action_space.sample())
    ```

Abychom viděli, jak prostředí funguje, spusťme krátkou simulaci na 100 kroků. V každém kroku poskytujeme jednu z akcí, které mají být provedeny – v této simulaci náhodně vybíráme akci z `action_space`.

1. Spusťte níže uvedený kód a podívejte se, k čemu vede.

    ✅ Pamatujte, že je preferováno spustit tento kód na lokální instalaci Pythonu! (blok kódu 3)

    ```python
    env.reset()
    
    for i in range(100):
       env.render()
       env.step(env.action_space.sample())
    env.close()
    ```

    Měli byste vidět něco podobného tomuto obrázku:

    ![nevyvážený CartPole](../../../../8-Reinforcement/2-Gym/images/cartpole-nobalance.gif)

1. Během simulace potřebujeme získat pozorování, abychom rozhodli, jak jednat. Ve skutečnosti funkce `step` vrací aktuální pozorování, funkci odměny a příznak `done`, který označuje, zda má smysl pokračovat v simulaci nebo ne: (blok kódu 4)

    ```python
    env.reset()
    
    done = False
    while not done:
       env.render()
       obs, rew, done, info = env.step(env.action_space.sample())
       print(f"{obs} -> {rew}")
    env.close()
    ```

    V notebooku byste měli vidět něco podobného tomuto výstupu:

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

    Vektor pozorování, který je vrácen při každém kroku simulace, obsahuje následující hodnoty:
    - Poloha vozíku
    - Rychlost vozíku
    - Úhel tyče
    - Rychlost otáčení tyče

1. Získejte minimální a maximální hodnotu těchto čísel: (blok kódu 5)

    ```python
    print(env.observation_space.low)
    print(env.observation_space.high)
    ```

    Můžete si také všimnout, že hodnota odměny při každém kroku simulace je vždy 1. To je proto, že naším cílem je přežít co nejdéle, tj. udržet tyč v přiměřeně vertikální poloze po co nejdelší dobu.

    ✅ Ve skutečnosti je simulace CartPole považována za vyřešenou, pokud se nám podaří získat průměrnou odměnu 195 během 100 po sobě jdoucích pokusů.

## Diskretizace stavu

V Q-Learningu potřebujeme vytvořit Q-Tabulku, která definuje, co dělat v každém stavu. Abychom toho dosáhli, musí být stav **diskrétní**, přesněji řečeno, musí obsahovat konečný počet diskrétních hodnot. Proto musíme nějak **diskretizovat** naše pozorování, mapovat je na konečnou množinu stavů.

Existuje několik způsobů, jak to udělat:

- **Rozdělení na intervaly**. Pokud známe interval určité hodnoty, můžeme tento interval rozdělit na několik **intervalů** a poté nahradit hodnotu číslem intervalu, do kterého patří. To lze provést pomocí metody numpy [`digitize`](https://numpy.org/doc/stable/reference/generated/numpy.digitize.html). V tomto případě budeme přesně znát velikost stavu, protože bude záviset na počtu intervalů, které vybereme pro digitalizaci.

✅ Můžeme použít lineární interpolaci k přivedení hodnot na nějaký konečný interval (např. od -20 do 20) a poté převést čísla na celá čísla zaokrouhlením. To nám dává o něco menší kontrolu nad velikostí stavu, zejména pokud neznáme přesné rozsahy vstupních hodnot. Například v našem případě 2 ze 4 hodnot nemají horní/dolní hranice svých hodnot, což může vést k nekonečnému počtu stavů.

V našem příkladu použijeme druhý přístup. Jak si možná později všimnete, navzdory nedefinovaným horním/dolním hranicím tyto hodnoty zřídka nabývají hodnot mimo určité konečné intervaly, takže tyto stavy s extrémními hodnotami budou velmi vzácné.

1. Zde je funkce, která vezme pozorování z našeho modelu a vytvoří čtveřici 4 celých čísel: (blok kódu 6)

    ```python
    def discretize(x):
        return tuple((x/np.array([0.25, 0.25, 0.01, 0.1])).astype(np.int))
    ```

1. Prozkoumejme také jinou metodu diskretizace pomocí intervalů: (blok kódu 7)

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

1. Nyní spusťme krátkou simulaci a pozorujme tyto diskrétní hodnoty prostředí. Vyzkoušejte `discretize` i `discretize_bins` a podívejte se, zda je mezi nimi rozdíl.

    ✅ `discretize_bins` vrací číslo intervalu, které je 0-based. Takže pro hodnoty vstupní proměnné kolem 0 vrací číslo ze středu intervalu (10). U `discretize` jsme se nestarali o rozsah výstupních hodnot, což umožňuje, aby byly negativní, takže hodnoty stavu nejsou posunuté a 0 odpovídá 0. (blok kódu 8)

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

    ✅ Odkomentujte řádek začínající `env.render`, pokud chcete vidět, jak prostředí funguje. Jinak jej můžete spustit na pozadí, což je rychlejší. Tento "neviditelný" způsob provádění použijeme během procesu Q-Learningu.

## Struktura Q-Tabulky

V naší předchozí lekci byl stav jednoduchý pár čísel od 0 do 8, a proto bylo pohodlné reprezentovat Q-Tabulku pomocí numpy tenzoru s tvarem 8x8x2. Pokud použijeme diskretizaci pomocí intervalů, velikost našeho stavového vektoru je také známá, takže můžeme použít stejný přístup a reprezentovat stav pomocí pole tvaru 20x20x10x10x2 (zde 2 je dimenze prostoru akcí a první dimenze odpovídají počtu intervalů, které jsme vybrali pro každou z hodnot v prostoru pozorování).

Nicméně někdy přesné rozměry prostoru pozorování nejsou známé. V případě funkce `discretize` si nikdy nemůžeme být jisti, že náš stav zůstane v určitých mezích, protože některé z původních hodnot nejsou omezené. Proto použijeme mírně odlišný přístup a reprezentujeme Q-Tabulku pomocí slovníku.

1. Použijte dvojici *(state,action)* jako klíč slovníku a hodnota by odpovídala hodnotě položky Q-Tabulky. (blok kódu 9)

    ```python
    Q = {}
    actions = (0,1)
    
    def qvalues(state):
        return [Q.get((state,a),0) for a in actions]
    ```

    Zde také definujeme funkci `qvalues()`, která vrací seznam hodnot Q-Tabulky pro daný stav, který odpovídá všem možným akcím. Pokud položka není přítomna v Q-Tabulce, vrátíme 0 jako výchozí hodnotu.

## Začněme Q-Learning

Nyní jsme připraveni naučit Petra udržovat rovnováhu!

1. Nejprve nastavme některé hyperparametry: (blok kódu 10)

    ```python
    # hyperparameters
    alpha = 0.3
    gamma = 0.9
    epsilon = 0.90
    ```

    Zde `alpha` je **rychlost učení**, která určuje, do jaké míry bychom měli upravit aktuální hodnoty Q-Tabulky při každém kroku. V předchozí lekci jsme začali s hodnotou 1 a poté jsme `alpha` snížili na nižší hodnoty během tréninku. V tomto příkladu ji ponecháme konstantní jen pro jednoduchost, a můžete experimentovat s úpravou hodnot `alpha` později.

    `gamma` je **faktor diskontování**, který ukazuje, do jaké míry bychom měli upřednostňovat budoucí odměnu před aktuální odměnou.

    `epsilon` je **faktor průzkumu/využití**, který určuje, zda bychom měli preferovat průzkum před využitím nebo naopak. V našem algoritmu budeme v `epsilon` procentech případů vybírat další akci podle hodnot Q-Tabulky a ve zbývajícím počtu případů provedeme náhodnou akci. To nám umožní prozkoumat oblasti prostoru hledání, které jsme dosud neviděli.

    ✅ Z hlediska udržování rovnováhy – výběr náhodné akce (průzkum) by působil jako náhodný úder špatným směrem a tyč by se musela naučit, jak obnovit rovnováhu z těchto "chyb".

### Vylepšení algoritmu

Můžeme také provést dvě vylepšení našeho algoritmu z předchozí lekce:

- **Výpočet průměrné kumulativní odměny** během několika simulací. Pokrok budeme tisknout každých 5000 iterací a průměrnou kumulativní odměnu za toto období zprůměrujeme. To znamená, že pokud získáme více než 195 bodů, můžeme problém považovat za vyřešený, a to s ještě vyšší kvalitou, než je požadováno.

- **Výpočet maximálního průměrného kumulativního výsledku**, `Qmax`, a uložíme Q-Tabulku odpovídající tomuto výsledku. Když spustíte trénink, všimnete si, že někdy průměrný kumulativní výsledek začne klesat, a chceme si uchovat hodnoty Q-Tabulky, které odpovídají nejlepšímu modelu pozorovanému během tréninku.

1. Sbírejte všechny kumulativní odměny při každé simulaci do vektoru `rewards` pro další vykreslení. (blok kódu 11)

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

Co si můžete všimnout z těchto výsledků:

- **Blízko našeho cíle**. Jsme velmi blízko dosažení cíle získat 195 kumulativních odměn během 100+ po sobě jdoucích běhů simulace, nebo jsme toho možná již dosáhli! I když získáme menší čísla, stále to nevíme, protože průměrujeme přes 5000 běhů a pouze 100 běhů je požadováno v rámci formálních kritérií.

- **Odměna začíná klesat**. Někdy odměna začne klesat, což znamená, že můžeme "zničit" již naučené hodnoty v Q-Tabulce těmi, které situaci zhoršují.

Toto pozorování je jasněji viditelné, pokud vykreslíme průběh tréninku.

## Vykreslení průběhu tréninku

Během tréninku jsme sbírali hodnotu kumulativní odměny při každé iteraci do vektoru `rewards`. Takto to vypadá, když to vykreslíme proti číslu iterace:

```python
plt.plot(rewards)
```

![surový průběh](../../../../8-Reinforcement/2-Gym/images/train_progress_raw.png)

Z tohoto grafu není možné nic říct, protože kvůli povaze stochastického procesu tréninku se délka tréninkových sezení značně liší. Aby měl tento graf větší smysl, můžeme vypočítat **běžný průměr** přes sérii experimentů, řekněme 100. To lze pohodlně provést pomocí `np.convolve`: (blok kódu 12)

```python
def running_average(x,window):
    return np.convolve(x,np.ones(window)/window,mode='valid')

plt.plot(running_average(rewards,100))
```

![průběh tréninku](../../../../8-Reinforcement/2-Gym/images/train_progress_runav.png)

## Úprava hyperparametrů

Aby bylo učení stabilnější, má smysl upravit některé z našich hyperparametrů během tréninku. Konkrétně:

- **Pro rychlost učení**, `alpha`, můžeme začít s hodnotami blízkými 1 a poté tento parametr postupně snižovat. S časem budeme získávat dobré pravděpodobnostní hodnoty v Q-Tabulce, a proto bychom je měli upravovat mírně, a ne zcela přepisovat novými hodnotami.

- **Zvýšení epsilon**. Můžeme chtít `epsilon` pomalu zvyšovat, aby se méně prozkoumávalo a více využívalo. Pravděpodobně má smysl začít s nižší hodnotou `epsilon` a postupně ji zvýšit téměř na 1.
> **Úkol 1**: Experimentujte s hodnotami hyperparametrů a zjistěte, zda můžete dosáhnout vyššího kumulativního odměny. Dosahujete více než 195?
> **Úkol 2**: Aby bylo možné problém formálně vyřešit, je potřeba dosáhnout průměrné odměny 195 během 100 po sobě jdoucích běhů. Měřte to během tréninku a ujistěte se, že jste problém formálně vyřešili!

## Vidět výsledek v akci

Bylo by zajímavé skutečně vidět, jak se naučený model chová. Spusťme simulaci a použijme stejnou strategii výběru akcí jako během tréninku, tedy vzorkování podle pravděpodobnostního rozdělení v Q-Tabulce: (blok kódu 13)

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

Měli byste vidět něco podobného:

![balancující cartpole](../../../../8-Reinforcement/2-Gym/images/cartpole-balance.gif)

---

## 🚀Výzva

> **Úkol 3**: Zde jsme používali finální verzi Q-Tabulky, která nemusí být ta nejlepší. Pamatujte, že jsme uložili nejlépe fungující Q-Tabulku do proměnné `Qbest`! Vyzkoušejte stejný příklad s nejlépe fungující Q-Tabulkou tím, že zkopírujete `Qbest` do `Q`, a sledujte, zda zaznamenáte rozdíl.

> **Úkol 4**: Zde jsme na každém kroku nevybírali nejlepší akci, ale spíše vzorkovali podle odpovídajícího pravděpodobnostního rozdělení. Mělo by větší smysl vždy vybírat nejlepší akci s nejvyšší hodnotou v Q-Tabulce? To lze provést pomocí funkce `np.argmax`, která zjistí číslo akce odpovídající nejvyšší hodnotě v Q-Tabulce. Implementujte tuto strategii a sledujte, zda zlepší balancování.

## [Kvíz po přednášce](https://ff-quizzes.netlify.app/en/ml/)

## Zadání
[Vytrénujte Mountain Car](assignment.md)

## Závěr

Nyní jsme se naučili, jak trénovat agenty, aby dosáhli dobrých výsledků pouze tím, že jim poskytneme funkci odměny, která definuje požadovaný stav hry, a dáme jim příležitost inteligentně prozkoumat prostor hledání. Úspěšně jsme aplikovali algoritmus Q-Learning v případech diskrétních i spojitých prostředí, ale s diskrétními akcemi.

Je také důležité studovat situace, kdy je stav akcí spojitý a kdy je prostor pozorování mnohem složitější, například obraz z obrazovky hry Atari. V těchto problémech často potřebujeme použít výkonnější techniky strojového učení, jako jsou neuronové sítě, abychom dosáhli dobrých výsledků. Tyto pokročilejší témata jsou předmětem našeho nadcházejícího pokročilého kurzu AI.

---

**Prohlášení**:  
Tento dokument byl přeložen pomocí služby pro automatický překlad [Co-op Translator](https://github.com/Azure/co-op-translator). Ačkoli se snažíme o přesnost, mějte na paměti, že automatické překlady mohou obsahovat chyby nebo nepřesnosti. Původní dokument v jeho původním jazyce by měl být považován za autoritativní zdroj. Pro důležité informace doporučujeme profesionální lidský překlad. Neodpovídáme za žádná nedorozumění nebo nesprávné interpretace vyplývající z použití tohoto překladu.