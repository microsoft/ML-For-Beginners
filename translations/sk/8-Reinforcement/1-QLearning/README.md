<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "911efd5e595089000cb3c16fce1beab8",
  "translation_date": "2025-09-05T16:40:14+00:00",
  "source_file": "8-Reinforcement/1-QLearning/README.md",
  "language_code": "sk"
}
-->
# Úvod do posilňovacieho učenia a Q-Learningu

![Zhrnutie posilňovacieho učenia v strojovom učení v sketchnote](../../../../sketchnotes/ml-reinforcement.png)
> Sketchnote od [Tomomi Imura](https://www.twitter.com/girlie_mac)

Posilňovacie učenie zahŕňa tri dôležité koncepty: agenta, určité stavy a súbor akcií pre každý stav. Vykonaním akcie v špecifikovanom stave dostane agent odmenu. Predstavte si počítačovú hru Super Mario. Vy ste Mario, nachádzate sa v úrovni hry, stojíte vedľa okraja útesu. Nad vami je minca. Vy ako Mario, v úrovni hry, na konkrétnej pozícii... to je váš stav. Pohyb o jeden krok doprava (akcia) vás zavedie cez okraj, čo by vám prinieslo nízke číselné skóre. Avšak stlačením tlačidla skoku by ste získali bod a zostali nažive. To je pozitívny výsledok, ktorý by vám mal priniesť pozitívne číselné skóre.

Pomocou posilňovacieho učenia a simulátora (hry) sa môžete naučiť, ako hrať hru tak, aby ste maximalizovali odmenu, čo znamená zostať nažive a získať čo najviac bodov.

[![Úvod do posilňovacieho učenia](https://img.youtube.com/vi/lDq_en8RNOo/0.jpg)](https://www.youtube.com/watch?v=lDq_en8RNOo)

> 🎥 Kliknite na obrázok vyššie a vypočujte si Dmitryho diskusiu o posilňovacom učení

## [Kvíz pred prednáškou](https://ff-quizzes.netlify.app/en/ml/)

## Predpoklady a nastavenie

V tejto lekcii budeme experimentovať s kódom v Pythone. Mali by ste byť schopní spustiť kód Jupyter Notebook z tejto lekcie, buď na svojom počítači alebo niekde v cloude.

Môžete otvoriť [notebook lekcie](https://github.com/microsoft/ML-For-Beginners/blob/main/8-Reinforcement/1-QLearning/notebook.ipynb) a prejsť touto lekciou, aby ste si ju osvojili.

> **Poznámka:** Ak otvárate tento kód z cloudu, musíte tiež stiahnuť súbor [`rlboard.py`](https://github.com/microsoft/ML-For-Beginners/blob/main/8-Reinforcement/1-QLearning/rlboard.py), ktorý sa používa v kóde notebooku. Pridajte ho do rovnakého adresára ako notebook.

## Úvod

V tejto lekcii preskúmame svet **[Peter a vlk](https://en.wikipedia.org/wiki/Peter_and_the_Wolf)**, inšpirovaný hudobnou rozprávkou od ruského skladateľa [Sergeja Prokofieva](https://en.wikipedia.org/wiki/Sergei_Prokofiev). Použijeme **posilňovacie učenie**, aby sme umožnili Petrovi preskúmať jeho prostredie, zbierať chutné jablká a vyhnúť sa stretnutiu s vlkom.

**Posilňovacie učenie** (RL) je technika učenia, ktorá nám umožňuje naučiť sa optimálne správanie **agenta** v určitom **prostredí** vykonávaním mnohých experimentov. Agent v tomto prostredí by mal mať nejaký **cieľ**, definovaný pomocou **funkcie odmeny**.

## Prostredie

Pre jednoduchosť si predstavme Petrov svet ako štvorcovú dosku veľkosti `šírka` x `výška`, ako je táto:

![Petrovo prostredie](../../../../8-Reinforcement/1-QLearning/images/environment.png)

Každá bunka na tejto doske môže byť:

* **zem**, po ktorej Peter a iné bytosti môžu chodiť.
* **voda**, po ktorej samozrejme nemôžete chodiť.
* **strom** alebo **tráva**, miesto, kde si môžete oddýchnuť.
* **jablko**, ktoré predstavuje niečo, čo by Peter rád našiel, aby sa nakŕmil.
* **vlk**, ktorý je nebezpečný a treba sa mu vyhnúť.

Existuje samostatný Python modul, [`rlboard.py`](https://github.com/microsoft/ML-For-Beginners/blob/main/8-Reinforcement/1-QLearning/rlboard.py), ktorý obsahuje kód na prácu s týmto prostredím. Keďže tento kód nie je dôležitý pre pochopenie našich konceptov, importujeme modul a použijeme ho na vytvorenie vzorovej dosky (blok kódu 1):

```python
from rlboard import *

width, height = 8,8
m = Board(width,height)
m.randomize(seed=13)
m.plot()
```

Tento kód by mal vytlačiť obrázok prostredia podobný tomu vyššie.

## Akcie a politika

V našom príklade by Petrovým cieľom bolo nájsť jablko, pričom sa vyhne vlkovi a iným prekážkam. Na to môže v podstate chodiť, kým nenájde jablko.

Preto si na akejkoľvek pozícii môže vybrať jednu z nasledujúcich akcií: hore, dole, doľava a doprava.

Tieto akcie definujeme ako slovník a mapujeme ich na dvojice zodpovedajúcich zmien súradníc. Napríklad pohyb doprava (`R`) by zodpovedal dvojici `(1,0)`. (blok kódu 2):

```python
actions = { "U" : (0,-1), "D" : (0,1), "L" : (-1,0), "R" : (1,0) }
action_idx = { a : i for i,a in enumerate(actions.keys()) }
```

Aby sme to zhrnuli, stratégia a cieľ tohto scenára sú nasledovné:

- **Stratégia** nášho agenta (Petra) je definovaná tzv. **politikou**. Politika je funkcia, ktorá vracia akciu v akomkoľvek danom stave. V našom prípade je stav problému reprezentovaný doskou vrátane aktuálnej pozície hráča.

- **Cieľ** posilňovacieho učenia je nakoniec naučiť sa dobrú politiku, ktorá nám umožní efektívne vyriešiť problém. Ako základ však zvážme najjednoduchšiu politiku nazývanú **náhodná chôdza**.

## Náhodná chôdza

Najprv vyriešme náš problém implementáciou stratégie náhodnej chôdze. Pri náhodnej chôdzi budeme náhodne vyberať ďalšiu akciu z povolených akcií, kým nedosiahneme jablko (blok kódu 3).

1. Implementujte náhodnú chôdzu pomocou nižšie uvedeného kódu:

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

    Volanie `walk` by malo vrátiť dĺžku zodpovedajúcej cesty, ktorá sa môže líšiť od jedného spustenia k druhému.

1. Spustite experiment chôdze niekoľkokrát (povedzme 100-krát) a vytlačte výsledné štatistiky (blok kódu 4):

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

    Všimnite si, že priemerná dĺžka cesty je okolo 30-40 krokov, čo je dosť veľa, vzhľadom na to, že priemerná vzdialenosť k najbližšiemu jablku je okolo 5-6 krokov.

    Môžete tiež vidieť, ako vyzerá Petrov pohyb počas náhodnej chôdze:

    ![Petrova náhodná chôdza](../../../../8-Reinforcement/1-QLearning/images/random_walk.gif)

## Funkcia odmeny

Aby bola naša politika inteligentnejšia, musíme pochopiť, ktoré pohyby sú "lepšie" ako ostatné. Na to musíme definovať náš cieľ.

Cieľ môže byť definovaný pomocou **funkcie odmeny**, ktorá vráti nejakú hodnotu skóre pre každý stav. Čím vyššie číslo, tým lepšia funkcia odmeny. (blok kódu 5)

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

Zaujímavé na funkciách odmeny je, že vo väčšine prípadov *dostaneme podstatnú odmenu až na konci hry*. To znamená, že náš algoritmus by mal nejako zapamätať "dobré" kroky, ktoré vedú k pozitívnej odmene na konci, a zvýšiť ich dôležitosť. Podobne by mali byť odradené všetky pohyby, ktoré vedú k zlým výsledkom.

## Q-Learning

Algoritmus, ktorý tu budeme diskutovať, sa nazýva **Q-Learning**. V tomto algoritme je politika definovaná funkciou (alebo dátovou štruktúrou) nazývanou **Q-Tabuľka**. Tá zaznamenáva "kvalitu" každej akcie v danom stave.

Nazýva sa Q-Tabuľka, pretože je často výhodné ju reprezentovať ako tabuľku alebo viacrozmerné pole. Keďže naša doska má rozmery `šírka` x `výška`, môžeme Q-Tabuľku reprezentovať pomocou numpy poľa s tvarom `šírka` x `výška` x `len(akcie)`: (blok kódu 6)

```python
Q = np.ones((width,height,len(actions)),dtype=np.float)*1.0/len(actions)
```

Všimnite si, že inicializujeme všetky hodnoty Q-Tabuľky rovnakou hodnotou, v našom prípade - 0.25. To zodpovedá politike "náhodnej chôdze", pretože všetky pohyby v každom stave sú rovnako dobré. Q-Tabuľku môžeme odovzdať funkcii `plot`, aby sme tabuľku vizualizovali na doske: `m.plot(Q)`.

![Petrovo prostredie](../../../../8-Reinforcement/1-QLearning/images/env_init.png)

V strede každej bunky je "šípka", ktorá označuje preferovaný smer pohybu. Keďže všetky smery sú rovnaké, zobrazí sa bodka.

Teraz musíme spustiť simuláciu, preskúmať naše prostredie a naučiť sa lepšie rozdelenie hodnôt Q-Tabuľky, ktoré nám umožní oveľa rýchlejšie nájsť cestu k jablku.

## Podstata Q-Learningu: Bellmanova rovnica

Keď sa začneme pohybovať, každá akcia bude mať zodpovedajúcu odmenu, t.j. teoreticky môžeme vybrať ďalšiu akciu na základe najvyššej okamžitej odmeny. Avšak vo väčšine stavov pohyb nedosiahne náš cieľ dosiahnuť jablko, a preto nemôžeme okamžite rozhodnúť, ktorý smer je lepší.

> Pamätajte, že nezáleží na okamžitom výsledku, ale skôr na konečnom výsledku, ktorý dosiahneme na konci simulácie.

Aby sme zohľadnili túto oneskorenú odmenu, musíme použiť princípy **[dynamického programovania](https://en.wikipedia.org/wiki/Dynamic_programming)**, ktoré nám umožňujú premýšľať o našom probléme rekurzívne.

Predpokladajme, že sa teraz nachádzame v stave *s*, a chceme sa presunúť do ďalšieho stavu *s'*. Týmto krokom získame okamžitú odmenu *r(s,a)*, definovanú funkciou odmeny, plus nejakú budúcu odmenu. Ak predpokladáme, že naša Q-Tabuľka správne odráža "atraktivitu" každej akcie, potom v stave *s'* si vyberieme akciu *a*, ktorá zodpovedá maximálnej hodnote *Q(s',a')*. Takže najlepšia možná budúca odmena, ktorú by sme mohli získať v stave *s*, bude definovaná ako `max`

## Kontrola politiky

Keďže Q-Tabuľka uvádza „atraktivitu“ každej akcie v každom stave, je pomerne jednoduché použiť ju na definovanie efektívnej navigácie v našom svete. V najjednoduchšom prípade môžeme vybrať akciu zodpovedajúcu najvyššej hodnote v Q-Tabuľke: (kódový blok 9)

```python
def qpolicy_strict(m):
        x,y = m.human
        v = probs(Q[x,y])
        a = list(actions)[np.argmax(v)]
        return a

walk(m,qpolicy_strict)
```

> Ak vyskúšate vyššie uvedený kód niekoľkokrát, môžete si všimnúť, že sa niekedy „zasekne“ a musíte stlačiť tlačidlo STOP v notebooku, aby ste ho prerušili. K tomu dochádza, pretože môžu existovať situácie, keď si dva stavy „ukazujú“ navzájom z hľadiska optimálnej hodnoty Q, v takom prípade agent skončí pohybom medzi týmito stavmi donekonečna.

## 🚀Výzva

> **Úloha 1:** Upraviť funkciu `walk` tak, aby obmedzila maximálnu dĺžku cesty na určitý počet krokov (napríklad 100) a pozorovať, ako vyššie uvedený kód občas vráti túto hodnotu.

> **Úloha 2:** Upraviť funkciu `walk` tak, aby sa nevracala na miesta, kde už predtým bola. Tým sa zabráni tomu, aby sa `walk` opakoval, avšak agent sa stále môže ocitnúť „uväznený“ na mieste, z ktorého sa nedokáže dostať.

## Navigácia

Lepšia navigačná politika by bola tá, ktorú sme použili počas tréningu, ktorá kombinuje využívanie a skúmanie. V tejto politike vyberieme každú akciu s určitou pravdepodobnosťou, úmernou hodnotám v Q-Tabuľke. Táto stratégia môže stále viesť k tomu, že sa agent vráti na pozíciu, ktorú už preskúmal, ale, ako môžete vidieť z nižšie uvedeného kódu, výsledkom je veľmi krátka priemerná cesta k požadovanému miestu (nezabudnite, že `print_statistics` spúšťa simuláciu 100-krát): (kódový blok 10)

```python
def qpolicy(m):
        x,y = m.human
        v = probs(Q[x,y])
        a = random.choices(list(actions),weights=v)[0]
        return a

print_statistics(qpolicy)
```

Po spustení tohto kódu by ste mali získať oveľa kratšiu priemernú dĺžku cesty ako predtým, v rozmedzí 3-6.

## Skúmanie procesu učenia

Ako sme spomenuli, proces učenia je rovnováhou medzi skúmaním a využívaním získaných znalostí o štruktúre problémového priestoru. Videli sme, že výsledky učenia (schopnosť pomôcť agentovi nájsť krátku cestu k cieľu) sa zlepšili, ale je tiež zaujímavé pozorovať, ako sa priemerná dĺžka cesty správa počas procesu učenia:

## Zhrnutie poznatkov:

- **Priemerná dĺžka cesty sa zvyšuje**. Na začiatku vidíme, že priemerná dĺžka cesty sa zvyšuje. Pravdepodobne je to spôsobené tým, že keď o prostredí nič nevieme, máme tendenciu uviaznuť v zlých stavoch, ako je voda alebo vlk. Keď sa dozvieme viac a začneme tieto znalosti využívať, môžeme prostredie skúmať dlhšie, ale stále nevieme presne, kde sú jablká.

- **Dĺžka cesty sa znižuje, ako sa učíme viac**. Keď sa naučíme dosť, agentovi sa ľahšie dosahuje cieľ a dĺžka cesty sa začne znižovať. Stále však skúmame nové možnosti, takže sa často odkloníme od najlepšej cesty a skúmame nové možnosti, čo predlžuje cestu nad optimálnu hodnotu.

- **Dĺžka sa náhle zvýši**. Na grafe tiež vidíme, že v určitom bode sa dĺžka náhle zvýšila. To poukazuje na stochastickú povahu procesu a na to, že môžeme v určitom bode „pokaziť“ koeficienty Q-Tabuľky tým, že ich prepíšeme novými hodnotami. Ideálne by sa tomu malo predísť znížením rýchlosti učenia (napríklad ku koncu tréningu upravujeme hodnoty Q-Tabuľky len o malú hodnotu).

Celkovo je dôležité pamätať na to, že úspech a kvalita procesu učenia významne závisí od parametrov, ako je rýchlosť učenia, pokles rýchlosti učenia a diskontný faktor. Tieto sa často nazývajú **hyperparametre**, aby sa odlíšili od **parametrov**, ktoré optimalizujeme počas tréningu (napríklad koeficienty Q-Tabuľky). Proces hľadania najlepších hodnôt hyperparametrov sa nazýva **optimalizácia hyperparametrov** a zaslúži si samostatnú tému.

## [Kvíz po prednáške](https://ff-quizzes.netlify.app/en/ml/)

## Zadanie 
[Realistickejší svet](assignment.md)

---

**Upozornenie**:  
Tento dokument bol preložený pomocou služby AI prekladu [Co-op Translator](https://github.com/Azure/co-op-translator). Hoci sa snažíme o presnosť, prosím, berte na vedomie, že automatizované preklady môžu obsahovať chyby alebo nepresnosti. Pôvodný dokument v jeho pôvodnom jazyku by mal byť považovaný za autoritatívny zdroj. Pre kritické informácie sa odporúča profesionálny ľudský preklad. Nie sme zodpovední za akékoľvek nedorozumenia alebo nesprávne interpretácie vyplývajúce z použitia tohto prekladu.