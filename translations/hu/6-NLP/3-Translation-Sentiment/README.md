<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "be03c8182982b87ced155e4e9d1438e8",
  "translation_date": "2025-09-05T17:03:02+00:00",
  "source_file": "6-NLP/3-Translation-Sentiment/README.md",
  "language_code": "hu"
}
-->
# Fordítás és érzelemelemzés gépi tanulással

Az előző leckékben megtanultad, hogyan készíts egy alap botot a `TextBlob` segítségével, amely egy olyan könyvtár, amely gépi tanulást alkalmaz a háttérben alapvető természetes nyelvi feldolgozási (NLP) feladatok, például főnévi kifejezések kinyerése érdekében. A számítógépes nyelvészet másik fontos kihívása egy mondat pontos _fordítása_ egyik beszélt vagy írott nyelvről a másikra.

## [Előadás előtti kvíz](https://ff-quizzes.netlify.app/en/ml/)

A fordítás egy nagyon nehéz probléma, amelyet tovább bonyolít az a tény, hogy több ezer nyelv létezik, és mindegyiknek nagyon eltérő nyelvtani szabályai lehetnek. Az egyik megközelítés az, hogy az egyik nyelv, például az angol formális nyelvtani szabályait egy nyelvtől független struktúrává alakítjuk, majd visszafordítjuk egy másik nyelvre. Ez a megközelítés a következő lépéseket jelenti:

1. **Azonosítás**. Azonosítsd vagy címkézd fel a bemeneti nyelv szavait főnevekként, igékként stb.
2. **Fordítás létrehozása**. Készíts közvetlen fordítást minden szóról a célnyelv formátumában.

### Példa mondat, angolról írre

Angol nyelven a _I feel happy_ mondat három szóból áll, a következő sorrendben:

- **alany** (I)
- **ige** (feel)
- **melléknév** (happy)

Az ír nyelvben azonban ugyanaz a mondat nagyon eltérő nyelvtani szerkezettel rendelkezik – az érzelmek, mint "*happy*" vagy "*sad*" úgy vannak kifejezve, mintha *rajtad lennének*.

Az angol `I feel happy` kifejezés írül `Tá athas orm`. Egy *szó szerinti* fordítás így hangzana: `Happy is upon me`.

Egy ír anyanyelvű, aki angolra fordít, azt mondaná, hogy `I feel happy`, nem pedig `Happy is upon me`, mert érti a mondat jelentését, még akkor is, ha a szavak és a mondatszerkezet eltérőek.

Az ír mondat formális sorrendje:

- **ige** (Tá vagy is)
- **melléknév** (athas, vagy happy)
- **alany** (orm, vagy upon me)

## Fordítás

Egy naiv fordítóprogram csak a szavakat fordítaná le, figyelmen kívül hagyva a mondatszerkezetet.

✅ Ha felnőttként tanultál második (vagy harmadik vagy több) nyelvet, valószínűleg úgy kezdtél, hogy anyanyelveden gondolkodtál, majd fejben szó szerint lefordítottad a fogalmat a második nyelvre, és kimondtad a fordítást. Ez hasonló ahhoz, amit a naiv fordítóprogramok csinálnak. Fontos túllépni ezen a fázison, hogy elérjük a folyékonyságot!

A naiv fordítás rossz (és néha mulatságos) félrefordításokhoz vezet: `I feel happy` szó szerint `Mise bhraitheann athas`-ként fordítódik írre. Ez azt jelenti (szó szerint), hogy `me feel happy`, és nem egy érvényes ír mondat. Annak ellenére, hogy az angol és az ír két szomszédos szigeten beszélt nyelv, nagyon különbözőek, eltérő nyelvtani szerkezettel.

> Nézhetsz néhány videót az ír nyelvi hagyományokról, például [ezt](https://www.youtube.com/watch?v=mRIaLSdRMMs)

### Gépi tanulási megközelítések

Eddig a természetes nyelvi feldolgozás formális szabályok szerinti megközelítéséről tanultál. Egy másik megközelítés az, hogy figyelmen kívül hagyjuk a szavak jelentését, és _helyette gépi tanulással mintákat észlelünk_. Ez működhet fordítás esetén, ha sok szöveg (egy *korpusz*) vagy szövegek (*korpuszok*) állnak rendelkezésre az eredeti és a célnyelven.

Például vegyük Jane Austen 1813-ban írt híres angol regényét, a *Büszkeség és balítélet*-et. Ha megnézed a könyvet angolul és annak emberi fordítását *franciául*, észlelhetsz olyan kifejezéseket, amelyek _idiomatikusan_ fordítódnak egyik nyelvről a másikra. Ezt hamarosan meg is teszed.

Például, amikor az angol `I have no money` kifejezés szó szerint franciára fordítódik, az `Je n'ai pas de monnaie` lesz. A "monnaie" egy trükkös francia 'hamis rokon', mivel a 'money' és a 'monnaie' nem szinonimák. Egy jobb fordítás, amelyet egy ember készítene, az `Je n'ai pas d'argent`, mert ez jobban közvetíti azt a jelentést, hogy nincs pénzed (nem pedig 'aprópénz', ami a 'monnaie' jelentése).

![monnaie](../../../../6-NLP/3-Translation-Sentiment/images/monnaie.png)

> Kép: [Jen Looper](https://twitter.com/jenlooper)

Ha egy gépi tanulási modellnek elegendő emberi fordítása van, amelyre modellt építhet, javíthatja a fordítások pontosságát azáltal, hogy azonosítja a korábban szakértő emberi beszélők által fordított szövegekben gyakori mintákat.

### Gyakorlat - fordítás

Használhatod a `TextBlob`-ot mondatok fordítására. Próbáld ki a **Büszkeség és balítélet** híres első mondatát:

```python
from textblob import TextBlob

blob = TextBlob(
    "It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want of a wife!"
)
print(blob.translate(to="fr"))

```

A `TextBlob` elég jól teljesít a fordításban: "C'est une vérité universellement reconnue, qu'un homme célibataire en possession d'une bonne fortune doit avoir besoin d'une femme!". 

Érvelhetünk azzal, hogy a TextBlob fordítása sokkal pontosabb, mint a könyv 1932-es francia fordítása V. Leconte és Ch. Pressoir által:

"C'est une vérité universelle qu'un célibataire pourvu d'une belle fortune doit avoir envie de se marier, et, si peu que l'on sache de son sentiment à cet egard, lorsqu'il arrive dans une nouvelle résidence, cette idée est si bien fixée dans l'esprit de ses voisins qu'ils le considèrent sur-le-champ comme la propriété légitime de l'une ou l'autre de leurs filles."

Ebben az esetben a gépi tanulással támogatott fordítás jobb munkát végez, mint az emberi fordító, aki szükségtelenül szavakat ad az eredeti szerző szájába a 'tisztázás' érdekében.

> Mi történik itt, és miért olyan jó a TextBlob a fordításban? Nos, a háttérben a Google Translate-et használja, amely egy kifinomult mesterséges intelligencia, amely képes több millió kifejezést elemezni, hogy előre jelezze a legjobb szövegeket az adott feladathoz. Itt semmi manuális nincs, és internetkapcsolatra van szükséged a `blob.translate` használatához.

✅ Próbálj ki néhány további mondatot. Melyik jobb, a gépi tanulás vagy az emberi fordítás? Milyen esetekben?

## Érzelemelemzés

Egy másik terület, ahol a gépi tanulás nagyon jól működhet, az érzelemelemzés. Egy nem gépi tanulási megközelítés az, hogy azonosítjuk a 'pozitív' és 'negatív' szavakat és kifejezéseket. Ezután egy új szöveg esetében kiszámítjuk a pozitív, negatív és semleges szavak összértékét, hogy meghatározzuk az általános érzelmet.

Ez a megközelítés könnyen becsapható, ahogy azt a Marvin feladatban láthattad – a mondat `Great, that was a wonderful waste of time, I'm glad we are lost on this dark road` egy szarkasztikus, negatív érzelmű mondat, de az egyszerű algoritmus a 'great', 'wonderful', 'glad' szavakat pozitívként, míg a 'waste', 'lost' és 'dark' szavakat negatívként érzékeli. Az összesített érzelem ezeknek az ellentmondásos szavaknak köszönhetően torzul.

✅ Állj meg egy pillanatra, és gondold át, hogyan közvetítjük a szarkazmust emberi beszélőként. A hanghordozás nagy szerepet játszik. Próbáld meg különböző módon kimondani a "Well, that film was awesome" mondatot, hogy felfedezd, hogyan közvetíti a hangod a jelentést.

### Gépi tanulási megközelítések

A gépi tanulási megközelítés az lenne, hogy manuálisan gyűjtünk negatív és pozitív szövegeket – tweeteket, filmkritikákat, vagy bármit, ahol az ember adott egy pontszámot *és* egy írott véleményt. Ezután NLP technikákat alkalmazhatunk a véleményekre és pontszámokra, hogy minták jelenjenek meg (például a pozitív filmkritikákban gyakrabban szerepel az 'Oscar worthy' kifejezés, mint a negatív kritikákban, vagy a pozitív étteremkritikákban gyakrabban szerepel a 'gourmet', mint a 'disgusting').

> ⚖️ **Példa**: Ha egy politikus irodájában dolgoznál, és egy új törvényt vitatnának meg, a választók támogató vagy ellenző e-maileket írhatnának az irodának az adott új törvénnyel kapcsolatban. Tegyük fel, hogy az a feladatod, hogy elolvasd az e-maileket, és két kupacba sorold őket, *támogató* és *ellenző*. Ha sok e-mail érkezne, túlterheltnek érezhetnéd magad, hogy mindet elolvasd. Nem lenne jó, ha egy bot elolvashatná őket helyetted, megértené, és megmondaná, melyik kupacba tartozik az egyes e-mailek? 
> 
> Egy módja ennek elérésére a gépi tanulás használata. A modellt az *ellenző* e-mailek egy részével és a *támogató* e-mailek egy részével képeznéd ki. A modell hajlamos lenne bizonyos kifejezéseket és szavakat az ellenző vagy támogató oldalhoz társítani, *de nem értené a tartalmat*, csak azt, hogy bizonyos szavak és minták nagyobb valószínűséggel jelennek meg egy *ellenző* vagy *támogató* e-mailben. Tesztelhetnéd néhány olyan e-maillel, amelyet nem használtál a modell képzésére, és megnézhetnéd, hogy ugyanarra a következtetésre jut-e, mint te. Ezután, ha elégedett lennél a modell pontosságával, feldolgozhatnád a jövőbeli e-maileket anélkül, hogy mindegyiket el kellene olvasnod.

✅ Ez a folyamat hasonlít olyan folyamatokra, amelyeket korábbi leckékben használtál?

## Gyakorlat - érzelmi mondatok

Az érzelmeket *polaritással* mérik -1-től 1-ig, ahol -1 a legnegatívabb érzelem, és 1 a legpozitívabb. Az érzelmeket továbbá 0 - 1 skálán mérik objektivitás (0) és szubjektivitás (1) szerint.

Nézd meg újra Jane Austen *Büszkeség és balítélet* című művét. A szöveg elérhető itt: [Project Gutenberg](https://www.gutenberg.org/files/1342/1342-h/1342-h.htm). Az alábbi minta egy rövid programot mutat be, amely elemzi a könyv első és utolsó mondatának érzelmi polaritását és szubjektivitás/objektivitás pontszámát.

A következő feladatban használd a fent leírt `TextBlob` könyvtárat az `érzelem` meghatározására (nem kell saját érzelemkalkulátort írnod).

```python
from textblob import TextBlob

quote1 = """It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want of a wife."""

quote2 = """Darcy, as well as Elizabeth, really loved them; and they were both ever sensible of the warmest gratitude towards the persons who, by bringing her into Derbyshire, had been the means of uniting them."""

sentiment1 = TextBlob(quote1).sentiment
sentiment2 = TextBlob(quote2).sentiment

print(quote1 + " has a sentiment of " + str(sentiment1))
print(quote2 + " has a sentiment of " + str(sentiment2))
```

A következő kimenetet látod:

```output
It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want # of a wife. has a sentiment of Sentiment(polarity=0.20952380952380953, subjectivity=0.27142857142857146)

Darcy, as well as Elizabeth, really loved them; and they were
     both ever sensible of the warmest gratitude towards the persons
      who, by bringing her into Derbyshire, had been the means of
      uniting them. has a sentiment of Sentiment(polarity=0.7, subjectivity=0.8)
```

## Kihívás - érzelmi polaritás ellenőrzése

A feladatod az, hogy érzelmi polaritás alapján meghatározd, hogy a *Büszkeség és balítélet* több abszolút pozitív mondatot tartalmaz-e, mint abszolút negatívat. Ehhez a feladathoz feltételezheted, hogy az 1 vagy -1 polaritás pontszám abszolút pozitív vagy negatív.

**Lépések:**

1. Töltsd le a [Büszkeség és balítélet](https://www.gutenberg.org/files/1342/1342-h/1342-h.htm) egy példányát a Project Gutenbergről .txt fájlként. Távolítsd el a metaadatokat a fájl elejéről és végéről, hogy csak az eredeti szöveg maradjon.
2. Nyisd meg a fájlt Pythonban, és sztringként olvasd ki a tartalmát.
3. Hozz létre egy TextBlob-ot a könyv sztringjéből.
4. Elemezd a könyv minden mondatát egy ciklusban.
   1. Ha a polaritás 1 vagy -1, tárold a mondatot egy pozitív vagy negatív üzeneteket tartalmazó tömbben vagy listában.
5. A végén külön-külön nyomtasd ki az összes pozitív és negatív mondatot, valamint azok számát.

Itt találsz egy mintát [megoldásként](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/3-Translation-Sentiment/solution/notebook.ipynb).

✅ Tudásellenőrzés

1. Az érzelem a mondatban használt szavakon alapul, de a kód *érti* a szavakat?
2. Gondolod, hogy az érzelmi polaritás pontos, vagy más szavakkal, *egyetértesz* a pontszámokkal?
   1. Különösen egyetértesz vagy nem értesz egyet az alábbi mondatok abszolút **pozitív** polaritásával?
      * “What an excellent father you have, girls!” said she, when the door was shut.
      * “Your examination of Mr. Darcy is over, I presume,” said Miss Bingley; “and pray what is the result?” “I am perfectly convinced by it that Mr. Darcy has no defect.
      * How wonderfully these sort of things occur!
      * I have the greatest dislike in the world to that sort of thing.
      * Charlotte is an excellent manager, I dare say.
      * “This is delightful indeed!
      * I am so happy!
      * Your idea of the ponies is delightful.
   2. A következő 3 mondat abszolút pozitív érzelmi pontszámot kapott, de alapos olvasás után nem pozitív mondatok. Miért gondolta az érzelemelemzés, hogy pozitív mondatok?
      * Happy shall I be, when his stay at Netherfield is over!” “I wish I could say anything to comfort you,” replied Elizabeth; “but it is wholly out of my power.
      * If I could but see you as happy!
      * Our distress, my dear Lizzy, is very great.
   3. Egyetértesz vagy nem értesz egyet az alábbi mondatok abszolút **negatív** polaritásával?
      - Everybody is disgusted with his pride.
      - “I should like to know how he behaves among strangers.” “You shall hear then—but prepare yourself for something very dreadful.
      - The pause was to Elizabeth’s feelings dreadful.
      - It would be dreadful!

✅ Jane Austen bármely rajongója megérti, hogy gyakran használja könyveit az angol regency társadalom nevetségesebb aspektusainak kritikájára. Elizabeth Bennett, a *Büszkeség és balítélet* főszereplője, éles társadalmi megfigyelő (mint az író), és nyelve gyakran erősen árnyalt. Még Mr. Darcy (a történet szerelmi érdeklődése) is megjegyzi Elizabeth játékos és szurkálódó nyelvhasználatát: "I have had the pleasure of your acquaintance long enough to know that you find great enjoyment in occasionally professing opinions which in fact are
Számos módja van annak, hogy szövegből érzelmeket vonjunk ki. Gondolj az üzleti alkalmazásokra, amelyek ezt a technikát használhatják. Gondolj arra is, hogyan sülhet el rosszul. Olvass többet kifinomult, vállalati szintű rendszerekről, amelyek érzelmeket elemeznek, például az [Azure Text Analysis](https://docs.microsoft.com/azure/cognitive-services/Text-Analytics/how-tos/text-analytics-how-to-sentiment-analysis?tabs=version-3-1?WT.mc_id=academic-77952-leestott) szolgáltatásról. Tesztelj néhány mondatot a Büszkeség és balítéletből, és nézd meg, hogy képes-e érzékelni a finom árnyalatokat.

## Feladat

[Poétikai szabadság](assignment.md)

---

**Felelősség kizárása**:  
Ez a dokumentum az AI fordítási szolgáltatás, a [Co-op Translator](https://github.com/Azure/co-op-translator) segítségével lett lefordítva. Bár törekszünk a pontosságra, kérjük, vegye figyelembe, hogy az automatikus fordítások hibákat vagy pontatlanságokat tartalmazhatnak. Az eredeti dokumentum az eredeti nyelvén tekintendő hiteles forrásnak. Kritikus információk esetén javasolt professzionális emberi fordítást igénybe venni. Nem vállalunk felelősséget semmilyen félreértésért vagy téves értelmezésért, amely a fordítás használatából eredhet.