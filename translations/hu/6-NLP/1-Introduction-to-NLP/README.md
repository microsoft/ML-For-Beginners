<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "1c2ec40cf55c98a028a359c27ef7e45a",
  "translation_date": "2025-09-05T16:59:57+00:00",
  "source_file": "6-NLP/1-Introduction-to-NLP/README.md",
  "language_code": "hu"
}
-->
# Bevezetés a természetes nyelvfeldolgozásba

Ez a lecke a *természetes nyelvfeldolgozás* rövid történetét és fontos fogalmait tárgyalja, amely a *számítógépes nyelvészet* egyik részterülete.

## [Előadás előtti kvíz](https://ff-quizzes.netlify.app/en/ml/)

## Bevezetés

Az NLP, ahogy általában nevezik, az egyik legismertebb terület, ahol a gépi tanulást alkalmazták és használták a gyártási szoftverekben.

✅ Tudsz olyan szoftverre gondolni, amit naponta használsz, és valószínűleg van benne valamilyen NLP? Mi a helyzet a szövegszerkesztő programokkal vagy a rendszeresen használt mobilalkalmazásokkal?

A következőkről fogsz tanulni:

- **A nyelvek fogalma**. Hogyan fejlődtek a nyelvek, és mik voltak a főbb kutatási területek.
- **Definíciók és fogalmak**. Megismered a szöveg számítógépes feldolgozásának definícióit és fogalmait, beleértve a szintaktikai elemzést, a nyelvtant, valamint a főnevek és igék azonosítását. Ebben a leckében lesznek kódolási feladatok, és számos fontos fogalmat mutatunk be, amelyeket a következő leckékben kódolni is megtanulsz.

## Számítógépes nyelvészet

A számítógépes nyelvészet egy évtizedek óta tartó kutatási és fejlesztési terület, amely azt vizsgálja, hogyan tudnak a számítógépek nyelvekkel dolgozni, megérteni, fordítani és kommunikálni. A természetes nyelvfeldolgozás (NLP) egy kapcsolódó terület, amely arra összpontosít, hogy a számítógépek hogyan tudják feldolgozni a "természetes", azaz emberi nyelveket.

### Példa - telefonos diktálás

Ha valaha diktáltál a telefonodnak gépelés helyett, vagy kérdést tettél fel egy virtuális asszisztensnek, akkor a beszédedet szöveges formára alakították, majd feldolgozták vagy *szintaktikailag elemezték* az általad használt nyelvet. Az észlelt kulcsszavakat ezután olyan formátumba dolgozták fel, amelyet a telefon vagy az asszisztens megértett és végrehajtott.

![megértés](../../../../6-NLP/1-Introduction-to-NLP/images/comprehension.png)
> A valódi nyelvi megértés nehéz! Kép: [Jen Looper](https://twitter.com/jenlooper)

### Hogyan lehetséges ez a technológia?

Ez azért lehetséges, mert valaki írt egy számítógépes programot, hogy ezt megvalósítsa. Néhány évtizeddel ezelőtt néhány sci-fi író azt jósolta, hogy az emberek főként beszélni fognak a számítógépeikhez, és a számítógépek mindig pontosan megértik majd, mit akarnak mondani. Sajnos kiderült, hogy ez a probléma nehezebb, mint sokan gondolták, és bár ma már sokkal jobban értjük, jelentős kihívásokkal kell szembenézni a mondatok jelentésének "tökéletes" természetes nyelvfeldolgozása során. Ez különösen nehéz, ha a mondatokban a humor vagy az érzelmek, például az irónia felismeréséről van szó.

Ezen a ponton talán eszedbe jutnak az iskolai órák, ahol a tanár a mondatok nyelvtani részeit tárgyalta. Egyes országokban a diákok külön tantárgyként tanulják a nyelvtant és a nyelvészetet, de sok helyen ezek a témák a nyelvtanulás részeként szerepelnek: akár az első nyelv tanulásakor az általános iskolában (olvasás és írás tanulása), akár egy második nyelv tanulásakor a középiskolában. Ne aggódj, ha nem vagy szakértő a főnevek és igék vagy a határozószók és melléknevek megkülönböztetésében!

Ha nehézséget okoz a *jelen egyszerű* és a *jelen folyamatos* közötti különbség, nem vagy egyedül. Ez sok ember számára kihívást jelent, még egy nyelv anyanyelvi beszélőinek is. A jó hír az, hogy a számítógépek nagyon jók a formális szabályok alkalmazásában, és meg fogod tanulni, hogyan írj kódot, amely egy mondatot olyan jól tud *szintaktikailag elemezni*, mint egy ember. A nagyobb kihívás, amelyet később megvizsgálsz, a mondat *jelentésének* és *érzelmi töltetének* megértése.

## Előfeltételek

Ehhez a leckéhez a fő előfeltétel az, hogy képes legyél elolvasni és megérteni a lecke nyelvét. Nincsenek matematikai problémák vagy megoldandó egyenletek. Bár az eredeti szerző angolul írta ezt a leckét, más nyelvekre is lefordították, így lehet, hogy fordítást olvasol. Vannak példák, ahol különböző nyelveket használnak (a különböző nyelvtani szabályok összehasonlítására). Ezek *nem* kerülnek fordításra, de a magyarázó szöveg igen, így a jelentésnek érthetőnek kell lennie.

A kódolási feladatokhoz Python-t fogsz használni, és a példák Python 3.8-at használnak.

Ebben a szakaszban szükséged lesz:

- **Python 3 megértése**. A Python 3 programozási nyelv megértése, ez a lecke bemeneti adatokat, ciklusokat, fájlolvasást és tömböket használ.
- **Visual Studio Code + kiegészítő**. A Visual Studio Code-ot és annak Python kiegészítőjét fogjuk használni. Használhatsz más Python IDE-t is.
- **TextBlob**. A [TextBlob](https://github.com/sloria/TextBlob) egy egyszerűsített szövegfeldolgozó könyvtár Pythonhoz. Kövesd a TextBlob weboldalán található utasításokat a telepítéshez (telepítsd a korpuszokat is, ahogy az alábbiakban látható):

   ```bash
   pip install -U textblob
   python -m textblob.download_corpora
   ```

> 💡 Tipp: A Python közvetlenül futtatható a VS Code környezetekben. További információért nézd meg a [dokumentációt](https://code.visualstudio.com/docs/languages/python?WT.mc_id=academic-77952-leestott).

## Beszélgetés gépekkel

Az emberi nyelv számítógépes megértésének története évtizedekre nyúlik vissza, és az egyik legkorábbi tudós, aki a természetes nyelvfeldolgozást vizsgálta, *Alan Turing* volt.

### A 'Turing-teszt'

Amikor Turing az 1950-es években a *mesterséges intelligenciát* kutatta, azt vizsgálta, hogy egy beszélgetési tesztet lehetne-e adni egy embernek és egy számítógépnek (gépelés útján), ahol a beszélgetésben részt vevő ember nem biztos abban, hogy egy másik emberrel vagy egy számítógéppel beszélget.

Ha egy bizonyos hosszúságú beszélgetés után az ember nem tudja megállapítani, hogy a válaszok számítógéptől származnak-e vagy sem, akkor mondható-e, hogy a számítógép *gondolkodik*?

### Az inspiráció - 'az utánzás játéka'

Az ötlet egy *Az utánzás játéka* nevű társasjátékból származott, ahol egy kérdező egyedül van egy szobában, és meg kell határoznia, hogy a másik szobában lévő két ember közül ki férfi és ki nő. A kérdező üzeneteket küldhet, és olyan kérdéseket kell kitalálnia, amelyek írásos válaszai felfedik a rejtélyes személy nemét. Természetesen a másik szobában lévő játékosok megpróbálják megtéveszteni vagy összezavarni a kérdezőt azáltal, hogy olyan módon válaszolnak, amely félrevezető vagy zavaró, miközben úgy tűnik, hogy őszintén válaszolnak.

### Eliza fejlesztése

Az 1960-as években egy MIT tudós, *Joseph Weizenbaum* kifejlesztette [*Eliza*](https://wikipedia.org/wiki/ELIZA) nevű számítógépes "terapeutát", amely kérdéseket tett fel az embernek, és úgy tűnt, hogy megérti a válaszait. Azonban, bár Eliza képes volt egy mondatot szintaktikailag elemezni, bizonyos nyelvtani szerkezeteket és kulcsszavakat azonosítani, hogy ésszerű választ adjon, nem mondható, hogy *megértette* a mondatot. Ha Eliza egy olyan mondatot kapott, amely a "**Én vagyok** <u>szomorú</u>" formát követte, akkor átrendezhette és helyettesíthette a mondat szavait, hogy a válasz "Mióta **vagy** <u>szomorú</u>" legyen.

Ez azt a benyomást keltette, hogy Eliza megértette az állítást, és egy következő kérdést tett fel, míg valójában csak megváltoztatta az igeidőt és hozzáadott néhány szót. Ha Eliza nem tudott azonosítani egy kulcsszót, amelyre válasza volt, akkor véletlenszerű választ adott, amely sok különböző állításra alkalmazható lehetett. Eliza könnyen becsapható volt, például ha egy felhasználó azt írta, "**Te vagy** egy <u>bicikli</u>", akkor azt válaszolhatta, "Mióta **vagyok** egy <u>bicikli</u>?", ahelyett, hogy egy ésszerűbb választ adott volna.

[![Beszélgetés Elizával](https://img.youtube.com/vi/RMK9AphfLco/0.jpg)](https://youtu.be/RMK9AphfLco "Beszélgetés Elizával")

> 🎥 Kattints a fenti képre az eredeti ELIZA programról szóló videóért

> Megjegyzés: Az eredeti leírást [Elizáról](https://cacm.acm.org/magazines/1966/1/13317-elizaa-computer-program-for-the-study-of-natural-language-communication-between-man-and-machine/abstract), amelyet 1966-ban publikáltak, elolvashatod, ha van ACM fiókod. Alternatívaként olvass Elizáról a [wikipédián](https://wikipedia.org/wiki/ELIZA).

## Gyakorlat - egy alapvető beszélgető bot kódolása

Egy beszélgető bot, mint Eliza, egy olyan program, amely felhasználói bemenetet kér, és úgy tűnik, hogy intelligensen válaszol. Elizával ellentétben a botunknak nem lesz több szabálya, amelyek intelligens beszélgetés látszatát keltik. Ehelyett a botunknak csak egy képessége lesz: a beszélgetés folytatása véletlenszerű válaszokkal, amelyek szinte bármilyen triviális beszélgetésben működhetnek.

### A terv

A beszélgető bot építésének lépései:

1. Nyomtass utasításokat, amelyek tanácsot adnak a felhasználónak, hogyan lépjen kapcsolatba a bottal
2. Indíts egy ciklust
   1. Fogadj felhasználói bemenetet
   2. Ha a felhasználó kilépést kér, lépj ki
   3. Dolgozd fel a felhasználói bemenetet, és határozd meg a választ (ebben az esetben a válasz egy véletlenszerű választás a lehetséges általános válaszok listájából)
   4. Nyomtasd ki a választ
3. Térj vissza a 2. lépéshez

### A bot építése

Hozzuk létre a botot! Kezdjük néhány kifejezés definiálásával.

1. Hozd létre ezt a botot Pythonban a következő véletlenszerű válaszokkal:

    ```python
    random_responses = ["That is quite interesting, please tell me more.",
                        "I see. Do go on.",
                        "Why do you say that?",
                        "Funny weather we've been having, isn't it?",
                        "Let's change the subject.",
                        "Did you catch the game last night?"]
    ```

    Íme néhány minta kimenet, amely segíthet (a felhasználói bemenet a `>`-tal kezdődő sorokon van):

    ```output
    Hello, I am Marvin, the simple robot.
    You can end this conversation at any time by typing 'bye'
    After typing each answer, press 'enter'
    How are you today?
    > I am good thanks
    That is quite interesting, please tell me more.
    > today I went for a walk     
    Did you catch the game last night?
    > I did, but my team lost
    Funny weather we've been having, isn't it?
    > yes but I hope next week is better
    Let's change the subject.
    > ok, lets talk about music
    Why do you say that?
    > because I like music!
    Why do you say that?
    > bye
    It was nice talking to you, goodbye!
    ```

    Egy lehetséges megoldás a feladatra [itt található](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/1-Introduction-to-NLP/solution/bot.py)

    ✅ Állj meg és gondolkodj el

    1. Szerinted a véletlenszerű válaszok "becsapnák" valakit, hogy azt gondolja, a bot valóban megértette őt?
    2. Milyen funkciókra lenne szüksége a botnak, hogy hatékonyabb legyen?
    3. Ha egy bot valóban "megértené" egy mondat jelentését, szüksége lenne arra, hogy "emlékezzen" a beszélgetés korábbi mondatai jelentésére is?

---

## 🚀Kihívás

Válassz egyet a fenti "állj meg és gondolkodj el" elemek közül, és próbáld meg megvalósítani kódban, vagy írj egy megoldást papíron álpseudokóddal.

A következő leckében számos más megközelítést fogsz megismerni a természetes nyelv szintaktikai elemzésére és gépi tanulásra.

## [Előadás utáni kvíz](https://ff-quizzes.netlify.app/en/ml/)

## Áttekintés és önálló tanulás

Tekintsd meg az alábbi hivatkozásokat további olvasási lehetőségként.

### Hivatkozások

1. Schubert, Lenhart, "Computational Linguistics", *The Stanford Encyclopedia of Philosophy* (Spring 2020 Edition), Edward N. Zalta (ed.), URL = <https://plato.stanford.edu/archives/spr2020/entries/computational-linguistics/>.
2. Princeton University "About WordNet." [WordNet](https://wordnet.princeton.edu/). Princeton University. 2010. 

## Feladat 

[Keress egy botot](assignment.md)

---

**Felelősség kizárása**:  
Ez a dokumentum az AI fordítási szolgáltatás [Co-op Translator](https://github.com/Azure/co-op-translator) segítségével lett lefordítva. Bár törekszünk a pontosságra, kérjük, vegye figyelembe, hogy az automatikus fordítások hibákat vagy pontatlanságokat tartalmazhatnak. Az eredeti dokumentum az eredeti nyelvén tekintendő hiteles forrásnak. Fontos információk esetén javasolt professzionális emberi fordítást igénybe venni. Nem vállalunk felelősséget semmilyen félreértésért vagy téves értelmezésért, amely a fordítás használatából eredhet.