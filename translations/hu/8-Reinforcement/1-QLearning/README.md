<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "911efd5e595089000cb3c16fce1beab8",
  "translation_date": "2025-09-05T16:38:33+00:00",
  "source_file": "8-Reinforcement/1-QLearning/README.md",
  "language_code": "hu"
}
-->
# Bevezetés a megerősítéses tanulásba és a Q-tanulásba

![A gépi tanulás megerősítésének összefoglalása egy sketchnote-ban](../../../../sketchnotes/ml-reinforcement.png)
> Sketchnote: [Tomomi Imura](https://www.twitter.com/girlie_mac)

A megerősítéses tanulás három fontos fogalmat foglal magában: az ügynököt, az állapotokat és az állapotonkénti cselekvések halmazát. Egy adott állapotban végrehajtott cselekvésért az ügynök jutalmat kap. Képzeljük el újra a Super Mario számítógépes játékot. Te vagy Mario, egy pályán állsz egy szakadék szélén. Fölötted egy érme van. Az, hogy te Mario vagy, egy adott pályán, egy adott pozícióban... ez az állapotod. Ha egy lépést jobbra lépsz (egy cselekvés), leesel a szakadékba, és alacsony pontszámot kapsz. Ha viszont megnyomod az ugrás gombot, pontot szerzel, és életben maradsz. Ez egy pozitív kimenetel, amiért pozitív pontszámot kell kapnod.

A megerősítéses tanulás és egy szimulátor (a játék) segítségével megtanulhatod, hogyan játszd a játékot úgy, hogy maximalizáld a jutalmat, vagyis életben maradj, és minél több pontot szerezz.

[![Bevezetés a megerősítéses tanulásba](https://img.youtube.com/vi/lDq_en8RNOo/0.jpg)](https://www.youtube.com/watch?v=lDq_en8RNOo)

> 🎥 Kattints a fenti képre, hogy meghallgasd Dmitry előadását a megerősítéses tanulásról.

## [Előadás előtti kvíz](https://ff-quizzes.netlify.app/en/ml/)

## Előfeltételek és beállítás

Ebben a leckében Pythonban fogunk kódot kipróbálni. Képesnek kell lenned futtatni a Jupyter Notebook kódját, akár a saját számítógépeden, akár a felhőben.

Megnyithatod [a lecke notebookját](https://github.com/microsoft/ML-For-Beginners/blob/main/8-Reinforcement/1-QLearning/notebook.ipynb), és végigmehetsz a leckén, hogy felépítsd a példát.

> **Megjegyzés:** Ha a kódot a felhőből nyitod meg, le kell töltened az [`rlboard.py`](https://github.com/microsoft/ML-For-Beginners/blob/main/8-Reinforcement/1-QLearning/rlboard.py) fájlt is, amelyet a notebook kód használ. Helyezd el ugyanabban a könyvtárban, ahol a notebook található.

## Bevezetés

Ebben a leckében **[Péter és a farkas](https://hu.wikipedia.org/wiki/P%C3%A9ter_%C3%A9s_a_farkas)** világát fogjuk felfedezni, amelyet egy orosz zeneszerző, [Szergej Prokofjev](https://hu.wikipedia.org/wiki/Szergej_Prokofjev) zenés meséje ihletett. A **megerősítéses tanulás** segítségével Pétert irányítjuk, hogy felfedezze a környezetét, ízletes almákat gyűjtsön, és elkerülje a farkassal való találkozást.

A **megerősítéses tanulás** (RL) egy olyan tanulási technika, amely lehetővé teszi számunkra, hogy egy **ügynök** optimális viselkedését tanuljuk meg egy adott **környezetben**, számos kísérlet lefuttatásával. Az ügynöknek ebben a környezetben van egy **célja**, amelyet egy **jutalomfüggvény** határoz meg.

## A környezet

Egyszerűség kedvéért tekintsük Péter világát egy `szélesség` x `magasság` méretű négyzet alakú táblának, például így:

![Péter környezete](../../../../8-Reinforcement/1-QLearning/images/environment.png)

A tábla minden cellája lehet:

* **talaj**, amin Péter és más lények járhatnak.
* **víz**, amin nyilvánvalóan nem lehet járni.
* **fa** vagy **fű**, ahol pihenni lehet.
* egy **alma**, amit Péter örömmel találna meg, hogy táplálkozzon.
* egy **farkas**, ami veszélyes, és el kell kerülni.

Van egy külön Python modul, az [`rlboard.py`](https://github.com/microsoft/ML-For-Beginners/blob/main/8-Reinforcement/1-QLearning/rlboard.py), amely tartalmazza a kódot a környezettel való munkához. Mivel ez a kód nem fontos a fogalmak megértéséhez, importáljuk a modult, és használjuk a minta tábla létrehozásához (kódblokk 1):

```python
from rlboard import *

width, height = 8,8
m = Board(width,height)
m.randomize(seed=13)
m.plot()
```

Ez a kód egy, a fentiekhez hasonló környezetet fog megjeleníteni.

## Cselekvések és stratégia

Példánkban Péter célja az alma megtalálása, miközben elkerüli a farkast és más akadályokat. Ehhez lényegében addig sétálhat, amíg meg nem találja az almát.

Ezért bármely pozícióban választhat a következő cselekvések közül: fel, le, balra és jobbra.

Ezeket a cselekvéseket egy szótárként definiáljuk, és a megfelelő koordinátaváltozásokhoz rendeljük. Például a jobbra mozgás (`R`) egy `(1,0)` párnak felel meg. (kódblokk 2):

```python
actions = { "U" : (0,-1), "D" : (0,1), "L" : (-1,0), "R" : (1,0) }
action_idx = { a : i for i,a in enumerate(actions.keys()) }
```

Összefoglalva, a forgatókönyv stratégiája és célja a következő:

- **A stratégia**, az ügynökünk (Péter) stratégiáját egy úgynevezett **politika** határozza meg. A politika egy olyan függvény, amely bármely adott állapotban visszaadja a cselekvést. Esetünkben a probléma állapotát a tábla és a játékos aktuális pozíciója képviseli.

- **A cél**, a megerősítéses tanulás célja egy jó politika megtanulása, amely lehetővé teszi a probléma hatékony megoldását. Az alapként vegyük figyelembe a legegyszerűbb politikát, az úgynevezett **véletlenszerű sétát**.

## Véletlenszerű séta

Először oldjuk meg a problémát egy véletlenszerű séta stratégiával. A véletlenszerű séta során véletlenszerűen választjuk ki a következő cselekvést az engedélyezett cselekvések közül, amíg el nem érjük az almát (kódblokk 3).

1. Valósítsd meg a véletlenszerű sétát az alábbi kóddal:

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

    A `walk` hívás visszaadja a megfelelő útvonal hosszát, amely futtatásonként változhat.

1. Futtasd le a séta kísérletet többször (például 100-szor), és nyomtasd ki az eredményeket (kódblokk 4):

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

    Figyeld meg, hogy az útvonal átlagos hossza körülbelül 30-40 lépés, ami elég sok, tekintve, hogy az átlagos távolság a legközelebbi almáig körülbelül 5-6 lépés.

    Azt is láthatod, hogyan mozog Péter a véletlenszerű séta során:

    ![Péter véletlenszerű sétája](../../../../8-Reinforcement/1-QLearning/images/random_walk.gif)

## Jutalomfüggvény

Ahhoz, hogy a politikánk intelligensebb legyen, meg kell értenünk, mely lépések "jobbak", mint mások. Ehhez meg kell határoznunk a célunkat.

A cél egy **jutalomfüggvény** segítségével határozható meg, amely minden állapothoz visszaad egy pontszámot. Minél magasabb a szám, annál jobb a jutalomfüggvény. (kódblokk 5)

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

A jutalomfüggvények érdekessége, hogy a legtöbb esetben *csak a játék végén kapunk jelentős jutalmat*. Ez azt jelenti, hogy az algoritmusunknak valahogy emlékeznie kell a "jó" lépésekre, amelyek pozitív jutalomhoz vezettek a végén, és növelnie kell azok fontosságát. Hasonlóképpen, minden olyan lépést, amely rossz eredményhez vezet, el kell kerülni.

## Q-tanulás

Az algoritmus, amelyet itt tárgyalunk, a **Q-tanulás**. Ebben az algoritmusban a politika egy **Q-tábla** nevű függvénnyel (vagy adatszerkezettel) van meghatározva. Ez rögzíti az egyes cselekvések "jóságát" egy adott állapotban.

Q-táblának hívják, mert gyakran kényelmes táblázatként vagy többdimenziós tömbként ábrázolni. Mivel a táblánk mérete `szélesség` x `magasság`, a Q-táblát egy numpy tömbként ábrázolhatjuk, amelynek alakja `szélesség` x `magasság` x `len(cselekvések)`: (kódblokk 6)

```python
Q = np.ones((width,height,len(actions)),dtype=np.float)*1.0/len(actions)
```

Figyeld meg, hogy a Q-tábla összes értékét egyenlő értékkel inicializáljuk, esetünkben 0,25-tel. Ez megfelel a "véletlenszerű séta" politikának, mert minden lépés minden állapotban egyformán jó. A Q-táblát átadhatjuk a `plot` függvénynek, hogy vizualizáljuk a táblát a táblán: `m.plot(Q)`.

![Péter környezete](../../../../8-Reinforcement/1-QLearning/images/env_init.png)

A cellák közepén egy "nyíl" látható, amely a mozgás preferált irányát jelzi. Mivel minden irány egyenlő, egy pont jelenik meg.

Most el kell indítanunk a szimulációt, felfedeznünk a környezetet, és meg kell tanulnunk a Q-tábla értékeinek jobb eloszlását, amely lehetővé teszi számunkra, hogy sokkal gyorsabban megtaláljuk az utat az almához.

## A Q-tanulás lényege: Bellman-egyenlet

Amint elkezdünk mozogni, minden cselekvéshez tartozik egy megfelelő jutalom, azaz elméletileg kiválaszthatjuk a következő cselekvést a legmagasabb azonnali jutalom alapján. Azonban a legtöbb állapotban a lépés nem éri el a célunkat, hogy elérjük az almát, így nem tudjuk azonnal eldönteni, melyik irány a jobb.

> Ne feledd, hogy nem az azonnali eredmény számít, hanem a végső eredmény, amelyet a szimuláció végén kapunk.

Ahhoz, hogy figyelembe vegyük ezt a késleltetett jutalmat, a **[dinamikus programozás](https://hu.wikipedia.org/wiki/Dinamikus_programoz%C3%A1s)** elveit kell alkalmaznunk, amelyek lehetővé teszik, hogy rekurzívan gondolkodjunk a problémánkról.

Tegyük fel, hogy most az *s* állapotban vagyunk, és a következő állapotba, *s'*-be akarunk lépni. Ezzel megkapjuk az azonnali jutalmat, *r(s,a)*, amelyet a jutalomfüggvény határoz meg, plusz némi jövőbeli jutalmat. Ha feltételezzük, hogy a Q-táblánk helyesen tükrözi az egyes cselekvések "vonzerejét", akkor az *s'* állapotban egy olyan *a* cselekvést választunk, amely a *Q(s',a')* maximális értékének felel meg. Így az *s* állapotban elérhető legjobb jövőbeli jutalom a következőképpen lesz meghatározva: `max`

## A szabály ellenőrzése

Mivel a Q-tábla felsorolja az egyes állapotokban végrehajtható cselekvések "vonzerejét", könnyen használható hatékony navigáció meghatározására a világunkban. A legegyszerűbb esetben kiválaszthatjuk azt a cselekvést, amely a legmagasabb Q-tábla értékhez tartozik: (kód blokk 9)

```python
def qpolicy_strict(m):
        x,y = m.human
        v = probs(Q[x,y])
        a = list(actions)[np.argmax(v)]
        return a

walk(m,qpolicy_strict)
```

> Ha többször kipróbálod a fenti kódot, észreveheted, hogy néha "elakad", és a jegyzetfüzetben a STOP gombot kell megnyomnod a megszakításhoz. Ez azért történik, mert előfordulhatnak olyan helyzetek, amikor két állapot "mutat" egymásra az optimális Q-értékek alapján, ilyenkor az ügynök végtelenül mozoghat ezek között az állapotok között.

## 🚀Kihívás

> **Feladat 1:** Módosítsd a `walk` függvényt úgy, hogy korlátozza az út maximális hosszát egy bizonyos lépésszámra (például 100), és figyeld meg, hogy a fenti kód időnként visszaadja ezt az értéket.

> **Feladat 2:** Módosítsd a `walk` függvényt úgy, hogy ne térjen vissza olyan helyekre, ahol korábban már járt. Ez megakadályozza, hogy a `walk` ciklusba kerüljön, azonban az ügynök még mindig "csapdába" eshet egy olyan helyen, ahonnan nem tud kijutni.

## Navigáció

Egy jobb navigációs szabály az lenne, amit a tanulás során használtunk, amely kombinálja a kihasználást és a felfedezést. Ebben a szabályban minden cselekvést egy bizonyos valószínűséggel választunk ki, amely arányos a Q-tábla értékeivel. Ez a stratégia még mindig eredményezheti, hogy az ügynök visszatér egy már felfedezett helyre, de ahogy az alábbi kódból látható, ez nagyon rövid átlagos útvonalat eredményez a kívánt helyre (ne feledd, hogy a `print_statistics` 100-szor futtatja a szimulációt): (kód blokk 10)

```python
def qpolicy(m):
        x,y = m.human
        v = probs(Q[x,y])
        a = random.choices(list(actions),weights=v)[0]
        return a

print_statistics(qpolicy)
```

A kód futtatása után sokkal rövidebb átlagos útvonalhosszt kell kapnod, 3-6 lépés körül.

## A tanulási folyamat vizsgálata

Ahogy említettük, a tanulási folyamat egyensúlyozás a felfedezés és a problématér szerkezetéről szerzett tudás kihasználása között. Láttuk, hogy a tanulás eredményei (az ügynök képessége, hogy rövid utat találjon a célhoz) javultak, de az is érdekes, hogy megfigyeljük, hogyan viselkedik az átlagos útvonalhossz a tanulási folyamat során:

## A tanulságok összefoglalása:

- **Az átlagos útvonalhossz növekszik**. Amit itt látunk, az az, hogy eleinte az átlagos útvonalhossz növekszik. Ez valószínűleg azért van, mert amikor semmit sem tudunk a környezetről, hajlamosak vagyunk rossz állapotokba, vízbe vagy farkasok közé kerülni. Ahogy többet tanulunk és elkezdjük használni ezt a tudást, hosszabb ideig tudjuk felfedezni a környezetet, de még mindig nem tudjuk pontosan, hol vannak az almák.

- **Az útvonalhossz csökken, ahogy többet tanulunk**. Miután eleget tanultunk, az ügynök számára könnyebbé válik a cél elérése, és az útvonalhossz csökkenni kezd. Azonban még mindig nyitottak vagyunk a felfedezésre, így gyakran eltérünk az optimális úttól, és új lehetőségeket fedezünk fel, ami hosszabb utat eredményez.

- **A hossz hirtelen megnő**. Amit ezen a grafikonon még megfigyelhetünk, az az, hogy egy ponton az útvonalhossz hirtelen megnőtt. Ez a folyamat sztochasztikus természetét jelzi, és azt, hogy bizonyos pontokon "elronthatjuk" a Q-tábla együtthatóit azzal, hogy új értékekkel felülírjuk őket. Ezt ideálisan minimalizálni kellene a tanulási ráta csökkentésével (például a tanulás végéhez közeledve csak kis értékkel módosítjuk a Q-tábla értékeit).

Összességében fontos megjegyezni, hogy a tanulási folyamat sikere és minősége jelentősen függ a paraméterektől, mint például a tanulási ráta, a tanulási ráta csökkenése és a diszkontfaktor. Ezeket gyakran **hiperparamétereknek** nevezik, hogy megkülönböztessék őket a **paraméterektől**, amelyeket a tanulás során optimalizálunk (például a Q-tábla együtthatói). A legjobb hiperparaméter értékek megtalálásának folyamatát **hiperparaméter optimalizációnak** nevezzük, és ez egy külön témát érdemel.

## [Utó-előadás kvíz](https://ff-quizzes.netlify.app/en/ml/)

## Feladat 
[Egy reálisabb világ](assignment.md)

---

**Felelősség kizárása**:  
Ez a dokumentum az AI fordítási szolgáltatás, a [Co-op Translator](https://github.com/Azure/co-op-translator) segítségével lett lefordítva. Bár törekszünk a pontosságra, kérjük, vegye figyelembe, hogy az automatikus fordítások hibákat vagy pontatlanságokat tartalmazhatnak. Az eredeti dokumentum az eredeti nyelvén tekintendő hiteles forrásnak. Kritikus információk esetén javasolt professzionális emberi fordítást igénybe venni. Nem vállalunk felelősséget semmilyen félreértésért vagy téves értelmezésért, amely a fordítás használatából eredhet.