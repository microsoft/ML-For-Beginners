<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "5f3cb462e3122e1afe7ab0050ccf2bd3",
  "translation_date": "2025-09-05T16:50:29+00:00",
  "source_file": "6-NLP/2-Tasks/README.md",
  "language_code": "hu"
}
-->
# Gyakori természetes nyelvfeldolgozási feladatok és technikák

A legtöbb *természetes nyelvfeldolgozási* feladat esetében a feldolgozandó szöveget fel kell bontani, meg kell vizsgálni, és az eredményeket el kell tárolni vagy össze kell vetni szabályokkal és adatbázisokkal. Ezek a feladatok lehetővé teszik a programozó számára, hogy a szöveg _jelentését_, _szándékát_ vagy csak a kifejezések és szavak _gyakoriságát_ megértse.

## [Előadás előtti kvíz](https://ff-quizzes.netlify.app/en/ml/)

Fedezzük fel a szövegfeldolgozásban használt gyakori technikákat. Ezek a technikák gépi tanulással kombinálva segítenek hatékonyan elemezni nagy mennyiségű szöveget. Mielőtt gépi tanulást alkalmaznánk ezekre a feladatokra, először értsük meg azokat a problémákat, amelyekkel egy NLP szakember szembesül.

## NLP-hez kapcsolódó feladatok

Számos módja van annak, hogy elemezzük a szöveget, amelyen dolgozunk. Vannak feladatok, amelyeket elvégezhetünk, és ezek révén megérthetjük a szöveget, valamint következtetéseket vonhatunk le. Ezeket a feladatokat általában egy meghatározott sorrendben hajtjuk végre.

### Tokenizáció

Valószínűleg az első dolog, amit a legtöbb NLP algoritmusnak el kell végeznie, az a szöveg tokenekre vagy szavakra bontása. Bár ez egyszerűnek tűnik, a különböző nyelvek írásjelei és mondathatárolói miatt bonyolult lehet. Különböző módszereket kell alkalmazni a határok meghatározásához.

![tokenizáció](../../../../6-NLP/2-Tasks/images/tokenization.png)
> Egy mondat tokenizálása a **Büszkeség és balítélet** című műből. Infografika: [Jen Looper](https://twitter.com/jenlooper)

### Beágyazások

[A szavak beágyazása](https://wikipedia.org/wiki/Word_embedding) egy módszer arra, hogy a szövegadatokat numerikus formába alakítsuk. A beágyazásokat úgy végezzük, hogy a hasonló jelentésű vagy együtt használt szavak csoportosuljanak.

![szavak beágyazása](../../../../6-NLP/2-Tasks/images/embedding.png)
> "A legnagyobb tisztelettel vagyok az idegeid iránt, ők a régi barátaim." - Szavak beágyazása egy mondatban a **Büszkeség és balítélet** című műből. Infografika: [Jen Looper](https://twitter.com/jenlooper)

✅ Próbáld ki [ezt az érdekes eszközt](https://projector.tensorflow.org/) a szavak beágyazásának kísérletezéséhez. Egy szó kiválasztásával hasonló szavak csoportjait láthatod: például a 'játék' csoportosul a 'disney', 'lego', 'playstation' és 'konzol' szavakkal.

### Elemzés és szófaji címkézés

Minden tokenizált szót szófajként lehet címkézni - például főnév, ige vagy melléknév. A mondat `a gyors vörös róka átugrott a lusta barna kutya felett` szófaji címkézése lehet például róka = főnév, ugrott = ige.

![elemzés](../../../../6-NLP/2-Tasks/images/parse.png)

> Egy mondat elemzése a **Büszkeség és balítélet** című műből. Infografika: [Jen Looper](https://twitter.com/jenlooper)

Az elemzés során felismerjük, hogy mely szavak kapcsolódnak egymáshoz egy mondatban - például `a gyors vörös róka ugrott` egy melléknév-főnév-ige sorozat, amely elkülönül a `lusta barna kutya` sorozattól.

### Szó- és kifejezésgyakoriságok

Egy nagy szövegtest elemzésekor hasznos lehet egy szótár létrehozása, amely tartalmazza az összes érdekes szót vagy kifejezést, valamint azok előfordulási gyakoriságát. A mondat `a gyors vörös róka átugrott a lusta barna kutya felett` szógyakorisága például 2 a 'a' esetében.

Nézzünk egy példaszöveget, ahol megszámoljuk a szavak gyakoriságát. Rudyard Kipling verse, A győztesek, tartalmazza a következő versszakot:

```output
What the moral? Who rides may read.
When the night is thick and the tracks are blind
A friend at a pinch is a friend, indeed,
But a fool to wait for the laggard behind.
Down to Gehenna or up to the Throne,
He travels the fastest who travels alone.
```

Mivel a kifejezésgyakoriság lehet kis- és nagybetűérzékeny, a `egy barát` kifejezés gyakorisága 2, a `a` gyakorisága 6, és a `utazik` gyakorisága 2.

### N-gramok

Egy szöveg felosztható meghatározott hosszúságú szósorozatokra: egy szó (unigram), két szó (bigram), három szó (trigram) vagy bármilyen számú szó (n-gram).

Például `a gyors vörös róka átugrott a lusta barna kutya felett` egy 2-es n-gram értékkel a következő n-gramokat eredményezi:

1. a gyors  
2. gyors vörös  
3. vörös róka  
4. róka ugrott  
5. ugrott át  
6. át a  
7. a lusta  
8. lusta barna  
9. barna kutya  

Könnyebb lehet ezt egy csúszó ablakként elképzelni a mondat felett. Íme egy 3 szavas n-gram példája, ahol az n-gram kiemelve látható:

1.   <u>**a gyors vörös**</u> róka átugrott a lusta barna kutya felett  
2.   a **<u>gyors vörös róka</u>** átugrott a lusta barna kutya felett  
3.   a gyors **<u>vörös róka ugrott</u>** át a lusta barna kutya felett  
4.   a gyors vörös **<u>róka ugrott át</u>** a lusta barna kutya felett  
5.   a gyors vörös róka **<u>ugrott át a</u>** lusta barna kutya felett  
6.   a gyors vörös róka ugrott **<u>át a lusta</u>** barna kutya felett  
7.   a gyors vörös róka ugrott át <u>**a lusta barna**</u> kutya felett  
8.   a gyors vörös róka ugrott át a **<u>lusta barna kutya</u>**

![n-gramok csúszó ablak](../../../../6-NLP/2-Tasks/images/n-grams.gif)

> N-gram érték 3: Infografika: [Jen Looper](https://twitter.com/jenlooper)

### Főnévi kifejezések kinyerése

A legtöbb mondatban van egy főnév, amely a mondat alanya vagy tárgya. Angol nyelvben gyakran azonosítható az 'a', 'an' vagy 'the' előtag alapján. A mondat alanyának vagy tárgyának azonosítása a 'főnévi kifejezés kinyerésével' gyakori feladat az NLP-ben, amikor a mondat jelentését próbáljuk megérteni.

✅ A mondatban "Nem tudom megmondani az órát, a helyet, a kinézetet vagy a szavakat, amelyek megalapozták. Túl régen volt. Már benne voltam, mielőtt tudtam volna, hogy elkezdtem." Fel tudod ismerni a főnévi kifejezéseket?

A mondatban `a gyors vörös róka átugrott a lusta barna kutya felett` 2 főnévi kifejezés van: **gyors vörös róka** és **lusta barna kutya**.

### Érzelemelemzés

Egy mondat vagy szöveg elemezhető az érzelmek szempontjából, hogy mennyire *pozitív* vagy *negatív*. Az érzelmeket *polaritás* és *objektivitás/szubjektivitás* alapján mérjük. A polaritás -1.0-tól 1.0-ig terjed (negatívtól pozitívig), az objektivitás pedig 0.0-tól 1.0-ig (legobjektívebbtől legszubjektívebbig).

✅ Később megtanulod, hogy különböző módokon lehet meghatározni az érzelmeket gépi tanulás segítségével, de az egyik módszer az, hogy egy emberi szakértő által pozitívnak vagy negatívnak kategorizált szavak és kifejezések listáját alkalmazzuk a szövegre, hogy kiszámítsuk a polaritási pontszámot. Látod, hogyan működhet ez bizonyos helyzetekben, és kevésbé jól másokban?

### Inflekció

Az inflekció lehetővé teszi, hogy egy szót átalakítsunk egyes vagy többes számú formájába.

### Lemmatizáció

A *lemma* egy szó gyökere vagy alapformája, például *repült*, *repülők*, *repülés* esetében a lemma az *repül* ige.

Hasznos adatbázisok is rendelkezésre állnak az NLP kutatók számára, különösen:

### WordNet

[WordNet](https://wordnet.princeton.edu/) egy adatbázis, amely szavakat, szinonimákat, ellentéteket és sok más részletet tartalmaz különböző nyelveken. Rendkívül hasznos fordítások, helyesírás-ellenőrzők vagy bármilyen nyelvi eszköz létrehozásakor.

## NLP könyvtárak

Szerencsére nem kell ezeket a technikákat magunknak felépíteni, mivel kiváló Python könyvtárak állnak rendelkezésre, amelyek sokkal hozzáférhetőbbé teszik azokat a fejlesztők számára, akik nem szakosodtak természetes nyelvfeldolgozásra vagy gépi tanulásra. A következő leckékben több példát is bemutatunk ezekre, de itt néhány hasznos példát találsz, amelyek segítenek a következő feladatban.

### Gyakorlat - `TextBlob` könyvtár használata

Használjunk egy TextBlob nevű könyvtárat, mivel hasznos API-kat tartalmaz az ilyen típusú feladatok megoldásához. A TextBlob "a [NLTK](https://nltk.org) és a [pattern](https://github.com/clips/pattern) óriási vállain áll, és jól működik mindkettővel." Jelentős mennyiségű gépi tanulás van beépítve az API-jába.

> Megjegyzés: Egy hasznos [Gyors kezdés](https://textblob.readthedocs.io/en/dev/quickstart.html#quickstart) útmutató elérhető a TextBlob számára, amelyet tapasztalt Python fejlesztőknek ajánlunk.

Amikor *főnévi kifejezéseket* próbálunk azonosítani, a TextBlob több lehetőséget kínál az ilyen kifejezések kinyerésére.

1. Nézd meg a `ConllExtractor`-t.

    ```python
    from textblob import TextBlob
    from textblob.np_extractors import ConllExtractor
    # import and create a Conll extractor to use later 
    extractor = ConllExtractor()
    
    # later when you need a noun phrase extractor:
    user_input = input("> ")
    user_input_blob = TextBlob(user_input, np_extractor=extractor)  # note non-default extractor specified
    np = user_input_blob.noun_phrases                                    
    ```

    > Mi történik itt? A [ConllExtractor](https://textblob.readthedocs.io/en/dev/api_reference.html?highlight=Conll#textblob.en.np_extractors.ConllExtractor) egy "főnévi kifejezés kinyerő, amely a ConLL-2000 tanulási korpusz alapján képzett chunk parsingot használ." A ConLL-2000 a 2000-es Számítógépes Természetes Nyelv Tanulási Konferencia. Minden évben a konferencia egy workshopot tartott egy nehéz NLP probléma megoldására, és 2000-ben ez a főnévi chunking volt. Egy modellt képeztek a Wall Street Journal alapján, "a 15-18. szakaszokat használva tanulási adatként (211727 token) és a 20. szakaszt tesztadatként (47377 token)". Az alkalmazott eljárásokat [itt](https://www.clips.uantwerpen.be/conll2000/chunking/) és az [eredményeket](https://ifarm.nl/erikt/research/np-chunking.html) megtekintheted.

### Kihívás - javítsd a botodat NLP segítségével

Az előző leckében egy nagyon egyszerű kérdés-válasz botot készítettél. Most Marvin-t egy kicsit szimpatikusabbá teszed azáltal, hogy elemzed a bemenetet érzelmek szempontjából, és ennek megfelelő választ adsz. Emellett azonosítanod kell egy `főnévi kifejezést`, és kérdezned kell róla.

A jobb beszélgető bot létrehozásának lépései:

1. Nyomtass utasításokat, amelyek tanácsot adnak a felhasználónak, hogyan lépjen kapcsolatba a bottal.
2. Indítsd el a ciklust:
   1. Fogadd el a felhasználói bemenetet.
   2. Ha a felhasználó kilépést kért, lépj ki.
   3. Dolgozd fel a felhasználói bemenetet, és határozd meg a megfelelő érzelmi választ.
   4. Ha főnévi kifejezést észlelsz az érzelemben, tedd többes számba, és kérj további bemenetet a témáról.
   5. Nyomtass választ.
3. Térj vissza a 2. lépéshez.

Íme a kódrészlet az érzelem meghatározásához a TextBlob segítségével. Figyeld meg, hogy csak négy *érzelmi gradiens* van (ha szeretnéd, lehet több is):

```python
if user_input_blob.polarity <= -0.5:
  response = "Oh dear, that sounds bad. "
elif user_input_blob.polarity <= 0:
  response = "Hmm, that's not great. "
elif user_input_blob.polarity <= 0.5:
  response = "Well, that sounds positive. "
elif user_input_blob.polarity <= 1:
  response = "Wow, that sounds great. "
```

Íme néhány minta kimenet, amely segíthet (a felhasználói bemenet a > jellel kezdődő sorokon található):

```output
Hello, I am Marvin, the friendly robot.
You can end this conversation at any time by typing 'bye'
After typing each answer, press 'enter'
How are you today?
> I am ok
Well, that sounds positive. Can you tell me more?
> I went for a walk and saw a lovely cat
Well, that sounds positive. Can you tell me more about lovely cats?
> cats are the best. But I also have a cool dog
Wow, that sounds great. Can you tell me more about cool dogs?
> I have an old hounddog but he is sick
Hmm, that's not great. Can you tell me more about old hounddogs?
> bye
It was nice talking to you, goodbye!
```

Egy lehetséges megoldás a feladatra [itt található](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/2-Tasks/solution/bot.py)

✅ Tudásellenőrzés

1. Szerinted a szimpatikus válaszok 'becsapnák' valakit, hogy azt higgye, a bot valóban megérti őt?
2. A főnévi kifejezés azonosítása hitelesebbé teszi a botot?
3. Miért lehet hasznos egy mondatból főnévi kifejezést kinyerni?

---

Valósítsd meg a botot az előző tudásellenőrzés alapján, és teszteld egy barátodon. Sikerül becsapnia őket? Tudod hitelesebbé tenni a botodat?

## 🚀Kihívás

Válassz egy feladatot az előző tudásellenőrzésből, és próbáld megvalósítani. Teszteld a botot egy barátodon. Sikerül becsapnia őket? Tudod hitelesebbé tenni a botodat?

## [Előadás utáni kvíz](https://ff-quizzes.netlify.app/en/ml/)

## Áttekintés és önálló tanulás

A következő néhány leckében többet fogsz tanulni az érzelemelemzésről. Kutass erről az érdekes technikáról olyan cikkekben, mint például ezek a [KDNuggets](https://www.kdnuggets.com/tag/nlp) oldalon.

## Feladat 

[Bot készítése, amely válaszol](assignment.md)

---

**Felelősség kizárása**:  
Ez a dokumentum az AI fordítási szolgáltatás, a [Co-op Translator](https://github.com/Azure/co-op-translator) segítségével lett lefordítva. Bár törekszünk a pontosságra, kérjük, vegye figyelembe, hogy az automatikus fordítások hibákat vagy pontatlanságokat tartalmazhatnak. Az eredeti dokumentum az eredeti nyelvén tekintendő hiteles forrásnak. Kritikus információk esetén javasolt professzionális emberi fordítást igénybe venni. Nem vállalunk felelősséget semmilyen félreértésért vagy téves értelmezésért, amely a fordítás használatából eredhet.