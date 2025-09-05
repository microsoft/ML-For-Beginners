<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "911efd5e595089000cb3c16fce1beab8",
  "translation_date": "2025-09-05T01:06:51+00:00",
  "source_file": "8-Reinforcement/1-QLearning/README.md",
  "language_code": "cs"
}
-->
# Úvod do posilovaného učení a Q-Learningu

![Shrnutí posilovaného učení v oblasti strojového učení ve sketchnote](../../../../sketchnotes/ml-reinforcement.png)
> Sketchnote od [Tomomi Imura](https://www.twitter.com/girlie_mac)

Posilované učení zahrnuje tři důležité koncepty: agenta, stavy a sadu akcí pro každý stav. Prováděním akce ve specifickém stavu získává agent odměnu. Představte si opět počítačovou hru Super Mario. Vy jste Mario, nacházíte se v úrovni hry, stojíte vedle okraje útesu. Nad vámi je mince. Vy jako Mario, v herní úrovni, na konkrétní pozici... to je váš stav. Posun o krok doprava (akce) vás přivede přes okraj, což by vám přineslo nízké číselné skóre. Stisknutí tlačítka skoku by vám však umožnilo získat bod a zůstat naživu. To je pozitivní výsledek, který by vám měl přinést pozitivní číselné skóre.

Pomocí posilovaného učení a simulátoru (hry) se můžete naučit, jak hru hrát, abyste maximalizovali odměnu, což znamená zůstat naživu a získat co nejvíce bodů.

[![Úvod do posilovaného učení](https://img.youtube.com/vi/lDq_en8RNOo/0.jpg)](https://www.youtube.com/watch?v=lDq_en8RNOo)

> 🎥 Klikněte na obrázek výše a poslechněte si Dmitryho, jak diskutuje o posilovaném učení.

## [Kvíz před lekcí](https://ff-quizzes.netlify.app/en/ml/)

## Předpoklady a nastavení

V této lekci budeme experimentovat s kódem v Pythonu. Měli byste být schopni spustit kód v Jupyter Notebooku z této lekce, buď na svém počítači, nebo někde v cloudu.

Můžete otevřít [notebook lekce](https://github.com/microsoft/ML-For-Beginners/blob/main/8-Reinforcement/1-QLearning/notebook.ipynb) a projít si tuto lekci krok za krokem.

> **Poznámka:** Pokud otevíráte tento kód z cloudu, musíte také stáhnout soubor [`rlboard.py`](https://github.com/microsoft/ML-For-Beginners/blob/main/8-Reinforcement/1-QLearning/rlboard.py), který se používá v kódu notebooku. Přidejte jej do stejného adresáře jako notebook.

## Úvod

V této lekci prozkoumáme svět **[Petr a vlk](https://en.wikipedia.org/wiki/Peter_and_the_Wolf)**, inspirovaný hudební pohádkou ruského skladatele [Sergeje Prokofjeva](https://en.wikipedia.org/wiki/Sergei_Prokofiev). Použijeme **posilované učení**, aby Petr mohl prozkoumat své prostředí, sbírat chutná jablka a vyhnout se setkání s vlkem.

**Posilované učení** (RL) je technika učení, která nám umožňuje naučit se optimální chování **agenta** v nějakém **prostředí** prostřednictvím mnoha experimentů. Agent v tomto prostředí by měl mít nějaký **cíl**, definovaný pomocí **funkce odměny**.

## Prostředí

Pro jednoduchost si představme Petrov svět jako čtvercovou desku o velikosti `šířka` x `výška`, jako je tato:

![Petrovo prostředí](../../../../8-Reinforcement/1-QLearning/images/environment.png)

Každá buňka na této desce může být:

* **zem**, po které Petr a další bytosti mohou chodit.
* **voda**, po které samozřejmě nemůžete chodit.
* **strom** nebo **tráva**, místo, kde si můžete odpočinout.
* **jablko**, což představuje něco, co by Petr rád našel, aby se nakrmil.
* **vlk**, který je nebezpečný a měl by být vyhnut.

Existuje samostatný modul v Pythonu, [`rlboard.py`](https://github.com/microsoft/ML-For-Beginners/blob/main/8-Reinforcement/1-QLearning/rlboard.py), který obsahuje kód pro práci s tímto prostředím. Protože tento kód není důležitý pro pochopení našich konceptů, importujeme modul a použijeme jej k vytvoření vzorové desky (blok kódu 1):

```python
from rlboard import *

width, height = 8,8
m = Board(width,height)
m.randomize(seed=13)
m.plot()
```

Tento kód by měl vytisknout obrázek prostředí podobný tomu výše.

## Akce a politika

V našem příkladu by Petrovým cílem bylo najít jablko, zatímco se vyhýbá vlkovi a dalším překážkám. K tomu může v podstatě chodit, dokud nenajde jablko.

Proto může na jakékoli pozici zvolit jednu z následujících akcí: nahoru, dolů, doleva a doprava.

Tyto akce definujeme jako slovník a mapujeme je na dvojice odpovídajících změn souřadnic. Například pohyb doprava (`R`) by odpovídal dvojici `(1,0)`. (blok kódu 2):

```python
actions = { "U" : (0,-1), "D" : (0,1), "L" : (-1,0), "R" : (1,0) }
action_idx = { a : i for i,a in enumerate(actions.keys()) }
```

Shrneme-li, strategie a cíl tohoto scénáře jsou následující:

- **Strategie** našeho agenta (Petra) je definována tzv. **politikou**. Politika je funkce, která vrací akci v daném stavu. V našem případě je stav problému reprezentován deskou, včetně aktuální pozice hráče.

- **Cíl** posilovaného učení je nakonec naučit se dobrou politiku, která nám umožní problém efektivně vyřešit. Jako základ však zvažme nejjednodušší politiku nazvanou **náhodná chůze**.

## Náhodná chůze

Nejprve vyřešíme náš problém implementací strategie náhodné chůze. Při náhodné chůzi budeme náhodně vybírat další akci z povolených akcí, dokud nedosáhneme jablka (blok kódu 3).

1. Implementujte náhodnou chůzi pomocí níže uvedeného kódu:

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

    Volání `walk` by mělo vrátit délku odpovídající cesty, která se může lišit od jednoho spuštění k druhému.

1. Spusťte experiment chůze několikrát (řekněme 100krát) a vytiskněte výsledné statistiky (blok kódu 4):

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

    Všimněte si, že průměrná délka cesty je kolem 30-40 kroků, což je poměrně hodně, vzhledem k tomu, že průměrná vzdálenost k nejbližšímu jablku je kolem 5-6 kroků.

    Můžete také vidět, jak vypadá Petrov pohyb během náhodné chůze:

    ![Petrova náhodná chůze](../../../../8-Reinforcement/1-QLearning/images/random_walk.gif)

## Funkce odměny

Aby byla naše politika inteligentnější, musíme pochopit, které kroky jsou "lepší" než jiné. K tomu musíme definovat náš cíl.

Cíl může být definován pomocí **funkce odměny**, která vrátí nějakou hodnotu skóre pro každý stav. Čím vyšší číslo, tím lepší funkce odměny. (blok kódu 5)

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

Zajímavé na funkcích odměny je, že ve většině případů *dostáváme podstatnou odměnu až na konci hry*. To znamená, že náš algoritmus by měl nějakým způsobem zapamatovat "dobré" kroky, které vedou k pozitivní odměně na konci, a zvýšit jejich důležitost. Podobně by měly být odrazeny všechny kroky, které vedou k špatným výsledkům.

## Q-Learning

Algoritmus, který zde budeme diskutovat, se nazývá **Q-Learning**. V tomto algoritmu je politika definována funkcí (nebo datovou strukturou) nazvanou **Q-Tabulka**. Ta zaznamenává "kvalitu" každé akce v daném stavu.

Nazývá se Q-Tabulka, protože je často výhodné ji reprezentovat jako tabulku nebo vícerozměrné pole. Protože naše deska má rozměry `šířka` x `výška`, můžeme Q-Tabulku reprezentovat pomocí numpy pole s tvarem `šířka` x `výška` x `len(actions)`: (blok kódu 6)

```python
Q = np.ones((width,height,len(actions)),dtype=np.float)*1.0/len(actions)
```

Všimněte si, že inicializujeme všechny hodnoty Q-Tabulky stejnou hodnotou, v našem případě - 0.25. To odpovídá politice "náhodné chůze", protože všechny kroky v každém stavu jsou stejně dobré. Q-Tabulku můžeme předat funkci `plot`, abychom ji vizualizovali na desce: `m.plot(Q)`.

![Petrovo prostředí](../../../../8-Reinforcement/1-QLearning/images/env_init.png)

Uprostřed každé buňky je "šipka", která označuje preferovaný směr pohybu. Protože všechny směry jsou stejné, je zobrazen bod.

Nyní musíme spustit simulaci, prozkoumat naše prostředí a naučit se lepší rozložení hodnot Q-Tabulky, které nám umožní najít cestu k jablku mnohem rychleji.

## Podstata Q-Learningu: Bellmanova rovnice

Jakmile se začneme pohybovat, každá akce bude mít odpovídající odměnu, tj. teoreticky můžeme vybrat další akci na základě nejvyšší okamžité odměny. Ve většině stavů však krok nedosáhne našeho cíle dosáhnout jablka, a proto nemůžeme okamžitě rozhodnout, který směr je lepší.

> Pamatujte, že nezáleží na okamžitém výsledku, ale spíše na konečném výsledku, který získáme na konci simulace.

Abychom zohlednili tuto zpožděnou odměnu, musíme použít principy **[dynamického programování](https://en.wikipedia.org/wiki/Dynamic_programming)**, které nám umožňují přemýšlet o našem problému rekurzivně.

Předpokládejme, že se nyní nacházíme ve stavu *s* a chceme se přesunout do dalšího stavu *s'*. Tím získáme okamžitou odměnu *r(s,a)*, definovanou funkcí odměny, plus nějakou budoucí odměnu. Pokud předpokládáme, že naše Q-Tabulka správně odráží "atraktivitu" každé akce, pak ve stavu *s'* zvolíme akci *a*, která odpovídá maximální hodnotě *Q(s',a')*. Tím pádem nejlepší možná budoucí odměna, kterou bychom mohli získat ve stavu *s*, bude definována jako `max`

## Kontrola politiky

Protože Q-Tabulka uvádí "atraktivitu" každé akce v každém stavu, je poměrně snadné ji použít k definování efektivní navigace v našem světě. V nejjednodušším případě můžeme vybrat akci odpovídající nejvyšší hodnotě v Q-Tabulce: (kódový blok 9)

```python
def qpolicy_strict(m):
        x,y = m.human
        v = probs(Q[x,y])
        a = list(actions)[np.argmax(v)]
        return a

walk(m,qpolicy_strict)
```

> Pokud výše uvedený kód vyzkoušíte několikrát, můžete si všimnout, že se někdy "zasekne" a je třeba stisknout tlačítko STOP v notebooku, abyste ho přerušili. K tomu dochází, protože mohou nastat situace, kdy si dva stavy "ukazují" na sebe z hlediska optimální hodnoty Q, což vede k tomu, že agent se mezi těmito stavy pohybuje nekonečně.

## 🚀Výzva

> **Úkol 1:** Upravte funkci `walk` tak, aby omezila maximální délku cesty na určitý počet kroků (například 100), a sledujte, jak výše uvedený kód tuto hodnotu čas od času vrací.

> **Úkol 2:** Upravte funkci `walk` tak, aby se nevracela na místa, kde již byla. Tím se zabrání tomu, aby se `walk` opakovala, nicméně agent může stále skončit "uvězněný" na místě, odkud se nemůže dostat.

## Navigace

Lepší navigační politika by byla ta, kterou jsme použili během tréninku, která kombinuje využívání a zkoumání. V této politice budeme vybírat každou akci s určitou pravděpodobností, úměrnou hodnotám v Q-Tabulce. Tato strategie může stále vést k tomu, že se agent vrátí na pozici, kterou již prozkoumal, ale jak můžete vidět z níže uvedeného kódu, vede k velmi krátké průměrné cestě k požadovanému místu (pamatujte, že `print_statistics` spouští simulaci 100krát): (kódový blok 10)

```python
def qpolicy(m):
        x,y = m.human
        v = probs(Q[x,y])
        a = random.choices(list(actions),weights=v)[0]
        return a

print_statistics(qpolicy)
```

Po spuštění tohoto kódu byste měli získat mnohem kratší průměrnou délku cesty než dříve, v rozmezí 3-6.

## Zkoumání procesu učení

Jak jsme zmínili, proces učení je rovnováhou mezi zkoumáním a využíváním získaných znalostí o struktuře prostoru problému. Viděli jsme, že výsledky učení (schopnost pomoci agentovi najít krátkou cestu k cíli) se zlepšily, ale je také zajímavé sledovat, jak se průměrná délka cesty chová během procesu učení:

## Shrnutí poznatků:

- **Průměrná délka cesty se zvyšuje**. Na začátku vidíme, že průměrná délka cesty roste. Pravděpodobně je to způsobeno tím, že když o prostředí nic nevíme, máme tendenci uvíznout ve špatných stavech, jako je voda nebo vlk. Jak se dozvídáme více a začneme tyto znalosti využívat, můžeme prostředí prozkoumávat déle, ale stále nevíme, kde přesně jsou jablka.

- **Délka cesty se s učením zkracuje**. Jakmile se naučíme dostatečně, je pro agenta snazší dosáhnout cíle a délka cesty se začne zkracovat. Stále však zůstáváme otevření zkoumání, takže často odbočíme od nejlepší cesty a zkoumáme nové možnosti, což cestu prodlužuje nad optimální délku.

- **Délka se náhle zvýší**. Na grafu také pozorujeme, že v určitém bodě se délka náhle zvýší. To ukazuje na stochastickou povahu procesu a na to, že můžeme v určitém okamžiku "zkazit" koeficienty Q-Tabulky jejich přepsáním novými hodnotami. Ideálně by se tomu mělo zabránit snížením rychlosti učení (například ke konci tréninku upravujeme hodnoty Q-Tabulky pouze o malou hodnotu).

Celkově je důležité si uvědomit, že úspěch a kvalita procesu učení významně závisí na parametrech, jako je rychlost učení, pokles rychlosti učení a diskontní faktor. Tyto parametry se často nazývají **hyperparametry**, aby se odlišily od **parametrů**, které optimalizujeme během tréninku (například koeficienty Q-Tabulky). Proces hledání nejlepších hodnot hyperparametrů se nazývá **optimalizace hyperparametrů** a zaslouží si samostatné téma.

## [Kvíz po přednášce](https://ff-quizzes.netlify.app/en/ml/)

## Zadání 
[Realističtější svět](assignment.md)

---

**Prohlášení**:  
Tento dokument byl přeložen pomocí služby pro automatický překlad [Co-op Translator](https://github.com/Azure/co-op-translator). Ačkoli se snažíme o přesnost, mějte prosím na paměti, že automatické překlady mohou obsahovat chyby nebo nepřesnosti. Původní dokument v jeho původním jazyce by měl být považován za autoritativní zdroj. Pro důležité informace se doporučuje profesionální lidský překlad. Neodpovídáme za žádná nedorozumění nebo nesprávné interpretace vyplývající z použití tohoto překladu.