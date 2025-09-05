<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "9d91f3af3758fdd4569fb410575995ef",
  "translation_date": "2025-09-05T16:03:22+00:00",
  "source_file": "1-Introduction/4-techniques-of-ML/README.md",
  "language_code": "hu"
}
-->
# Gépi tanulás technikái

A gépi tanulási modellek és az általuk használt adatok létrehozása, használata és karbantartása nagyon eltérő folyamat, mint sok más fejlesztési munkafolyamat. Ebben a leckében eloszlatjuk a folyamat körüli homályt, és bemutatjuk azokat a fő technikákat, amelyeket ismerned kell. A következőket fogod megtanulni:

- Megérteni a gépi tanulás alapvető folyamatait.
- Felfedezni az alapfogalmakat, mint például a „modellek”, „előrejelzések” és „tanító adatok”.

## [Előzetes kvíz](https://ff-quizzes.netlify.app/en/ml/)

[![Gépi tanulás kezdőknek - Gépi tanulás technikái](https://img.youtube.com/vi/4NGM0U2ZSHU/0.jpg)](https://youtu.be/4NGM0U2ZSHU "Gépi tanulás kezdőknek - Gépi tanulás technikái")

> 🎥 Kattints a fenti képre egy rövid videóért, amely bemutatja ezt a leckét.

## Bevezetés

Magas szinten nézve a gépi tanulási (ML) folyamatok létrehozása több lépésből áll:

1. **Határozd meg a kérdést**. A legtöbb ML folyamat egy olyan kérdés feltevésével kezdődik, amelyet nem lehet egyszerű feltételes programmal vagy szabályalapú motorral megválaszolni. Ezek a kérdések gyakran az adatok gyűjteménye alapján történő előrejelzések körül forognak.
2. **Gyűjtsd össze és készítsd elő az adatokat**. Ahhoz, hogy megválaszolhasd a kérdésedet, adatokra van szükséged. Az adatok minősége és néha mennyisége határozza meg, hogy mennyire jól tudod megválaszolni az eredeti kérdésedet. Az adatok vizualizálása fontos része ennek a fázisnak. Ez a fázis magában foglalja az adatok tanító és tesztelő csoportokra való felosztását a modell építéséhez.
3. **Válassz egy tanítási módszert**. A kérdésed és az adataid jellege alapján ki kell választanod, hogyan szeretnéd tanítani a modellt, hogy az a legjobban tükrözze az adataidat, és pontos előrejelzéseket készítsen. Ez az ML folyamat azon része, amely specifikus szakértelmet igényel, és gyakran jelentős mennyiségű kísérletezést.
4. **Tanítsd a modellt**. A tanító adataidat használva különböző algoritmusok segítségével tanítasz egy modellt, hogy felismerje az adatokban rejlő mintázatokat. A modell belső súlyokat használhat, amelyeket úgy lehet beállítani, hogy bizonyos adatokat előnyben részesítsen másokkal szemben, hogy jobb modellt építsen.
5. **Értékeld a modellt**. Az összegyűjtött adatokból származó, korábban nem látott adatok (tesztelő adatok) segítségével ellenőrzöd, hogyan teljesít a modell.
6. **Paraméterek finomhangolása**. A modell teljesítménye alapján újra elvégezheted a folyamatot különböző paraméterek vagy változók használatával, amelyek az algoritmusok viselkedését szabályozzák.
7. **Előrejelzés**. Új bemenetek segítségével tesztelheted a modell pontosságát.

## Milyen kérdést tegyünk fel?

A számítógépek különösen ügyesek az adatokban rejtett mintázatok felfedezésében. Ez a képesség nagyon hasznos a kutatók számára, akik olyan kérdéseket tesznek fel egy adott területen, amelyeket nem lehet könnyen megválaszolni feltételes szabálymotor létrehozásával. Például egy aktuáriusi feladat esetén egy adatkutató képes lehet kézzel készített szabályokat alkotni a dohányosok és nem dohányosok halálozási arányáról.

Ha azonban sok más változót is figyelembe veszünk, egy ML modell hatékonyabb lehet a jövőbeli halálozási arányok előrejelzésére a korábbi egészségügyi előzmények alapján. Egy vidámabb példa lehet az áprilisi időjárás előrejelzése egy adott helyen olyan adatok alapján, mint szélesség, hosszúság, éghajlatváltozás, óceán közelsége, jet stream mintázatok és még sok más.

✅ Ez a [prezentáció](https://www2.cisl.ucar.edu/sites/default/files/2021-10/0900%20June%2024%20Haupt_0.pdf) az időjárási modellekről történelmi perspektívát nyújt az ML időjárás-elemzésben való használatáról.  

## Modellépítés előtti feladatok

Mielőtt elkezdenéd a modell építését, számos feladatot kell elvégezned. Ahhoz, hogy tesztelhesd a kérdésedet és hipotézist alkothass a modell előrejelzései alapján, azonosítanod és konfigurálnod kell néhány elemet.

### Adatok

Ahhoz, hogy bármilyen bizonyossággal megválaszolhasd a kérdésedet, megfelelő mennyiségű és típusú adatra van szükséged. Ezen a ponton két dolgot kell tenned:

- **Adatok gyűjtése**. Az előző leckében tárgyalt adatelemzési méltányosságot szem előtt tartva gyűjtsd össze az adataidat gondosan. Légy tisztában az adatok forrásaival, az esetleges benne rejlő torzításokkal, és dokumentáld az eredetüket.
- **Adatok előkészítése**. Az adatok előkészítési folyamatának több lépése van. Lehet, hogy össze kell gyűjtened és normalizálnod kell az adatokat, ha különböző forrásokból származnak. Az adatok minőségét és mennyiségét különböző módszerekkel javíthatod, például szöveges adatok számokká alakításával (ahogy a [Klaszterezés](../../5-Clustering/1-Visualize/README.md) során tesszük). Új adatokat is generálhatsz az eredeti alapján (ahogy a [Kategorizálás](../../4-Classification/1-Introduction/README.md) során tesszük). Az adatokat tisztíthatod és szerkesztheted (ahogy a [Webalkalmazás](../../3-Web-App/README.md) lecke előtt tesszük). Végül lehet, hogy véletlenszerűsítened és keverned kell az adatokat, az alkalmazott tanítási technikák függvényében.

✅ Miután összegyűjtötted és feldolgoztad az adatokat, szánj egy pillanatot arra, hogy megnézd, az adatok formája lehetővé teszi-e számodra a tervezett kérdés megválaszolását. Lehet, hogy az adatok nem teljesítenek jól az adott feladatban, ahogy azt a [Klaszterezés](../../5-Clustering/1-Visualize/README.md) leckékben felfedezzük!

### Jellemzők és cél

Egy [jellemző](https://www.datasciencecentral.com/profiles/blogs/an-introduction-to-variable-and-feature-selection) az adataid mérhető tulajdonsága. Sok adatállományban ez oszlopfejlécként jelenik meg, például „dátum”, „méret” vagy „szín”. A jellemző változók, amelyeket általában `X`-ként jelölünk a kódban, azokat a bemeneti változókat képviselik, amelyeket a modell tanítására használunk.

A cél az, amit megpróbálsz előre jelezni. A célt általában `y`-ként jelöljük a kódban, és az adatokkal kapcsolatos kérdésedre adott választ képviseli: decemberben milyen **színű** tökök lesznek a legolcsóbbak? San Franciscóban melyik környékeken lesz a legjobb az ingatlanok **ára**? Néha a célt címke attribútumnak is nevezik.

### Jellemző változó kiválasztása

🎓 **Jellemzők kiválasztása és kinyerése** Hogyan döntöd el, melyik változót válaszd ki a modell építésekor? Valószínűleg végig fogsz menni egy jellemzők kiválasztási vagy kinyerési folyamatán, hogy kiválaszd a legmegfelelőbb változókat a legjobb teljesítményű modellhez. Ezek azonban nem ugyanazok: „A jellemzők kinyerése új jellemzőket hoz létre az eredeti jellemzők függvényeiből, míg a jellemzők kiválasztása az eredeti jellemzők egy részhalmazát adja vissza.” ([forrás](https://wikipedia.org/wiki/Feature_selection))

### Adatok vizualizálása

Az adatkutató eszköztárának fontos része az adatok vizualizálásának képessége, amelyhez számos kiváló könyvtár, például Seaborn vagy MatPlotLib áll rendelkezésre. Az adatok vizuális ábrázolása lehetővé teheti, hogy rejtett összefüggéseket fedezz fel, amelyeket kihasználhatsz. A vizualizációk segíthetnek abban is, hogy torzítást vagy kiegyensúlyozatlan adatokat fedezz fel (ahogy azt a [Kategorizálás](../../4-Classification/2-Classifiers-1/README.md) során felfedezzük).

### Adatállomány felosztása

A tanítás előtt fel kell osztanod az adatállományodat két vagy több, egyenlőtlen méretű részre, amelyek még mindig jól reprezentálják az adatokat.

- **Tanítás**. Az adatállomány ezen része illeszkedik a modelledhez, hogy megtanítsa azt. Ez a rész az eredeti adatállomány többségét alkotja.
- **Tesztelés**. A tesztadatállomány az eredeti adatokból származó független adatok csoportja, amelyet a modell teljesítményének megerősítésére használsz.
- **Érvényesítés**. Az érvényesítési készlet egy kisebb független példák csoportja, amelyet a modell hiperparamétereinek vagy architektúrájának finomhangolására használsz, hogy javítsd a modellt. Az adatok méretétől és a kérdésedtől függően lehet, hogy nem szükséges ezt a harmadik készletet létrehozni (ahogy azt a [Idősor előrejelzés](../../7-TimeSeries/1-Introduction/README.md) leckében megjegyezzük).

## Modell építése

A tanító adataidat használva az a célod, hogy egy modellt, vagyis az adataid statisztikai reprezentációját építsd fel különböző algoritmusok segítségével, hogy **tanítsd** azt. A modell tanítása során az adatoknak való kitettség lehetővé teszi, hogy feltételezéseket tegyen az általa felfedezett mintázatokról, amelyeket érvényesít, elfogad vagy elutasít.

### Tanítási módszer kiválasztása

A kérdésed és az adataid jellege alapján választasz egy módszert a tanításhoz. A [Scikit-learn dokumentációjának](https://scikit-learn.org/stable/user_guide.html) átlépése során - amelyet ebben a kurzusban használunk - számos módot fedezhetsz fel a modell tanítására. Tapasztalatodtól függően lehet, hogy több különböző módszert kell kipróbálnod a legjobb modell felépítéséhez. Valószínűleg egy olyan folyamaton mész keresztül, amely során az adatkutatók értékelik a modell teljesítményét azáltal, hogy nem látott adatokat adnak neki, ellenőrzik a pontosságot, torzítást és más minőségromboló problémákat, és kiválasztják a legmegfelelőbb tanítási módszert az adott feladathoz.

### Modell tanítása

A tanító adataiddal felvértezve készen állsz arra, hogy „illeszd” azokat egy modell létrehozásához. Észre fogod venni, hogy sok ML könyvtárban megtalálható a „model.fit” kód - ekkor küldöd be a jellemző változót értékek tömbjeként (általában „X”) és egy célváltozót (általában „y”).

### Modell értékelése

Miután a tanítási folyamat befejeződött (egy nagy modell tanítása sok iterációt, vagy „epoch”-ot igényelhet), képes leszel értékelni a modell minőségét tesztadatok segítségével, hogy felmérd a teljesítményét. Ezek az adatok az eredeti adatok egy részhalmazát képezik, amelyeket a modell korábban nem elemzett. Kinyomtathatsz egy táblázatot a modell minőségéről szóló metrikákról.

🎓 **Modell illesztése**

A gépi tanulás kontextusában a modell illesztése arra utal, hogy a modell alapvető funkciója mennyire pontosan próbálja elemezni azokat az adatokat, amelyekkel nem ismerős.

🎓 **Alulillesztés** és **túlillesztés** gyakori problémák, amelyek rontják a modell minőségét, mivel a modell vagy nem elég jól, vagy túl jól illeszkedik. Ez azt okozza, hogy a modell vagy túl szorosan, vagy túl lazán igazodik a tanító adataihoz. Egy túlillesztett modell túl jól előrejelzi a tanító adatokat, mert túl jól megtanulta az adatok részleteit és zaját. Egy alulillesztett modell nem pontos, mivel sem a tanító adatait, sem azokat az adatokat, amelyeket még nem „látott”, nem tudja pontosan elemezni.

![túlillesztett modell](../../../../1-Introduction/4-techniques-of-ML/images/overfitting.png)
> Infografika: [Jen Looper](https://twitter.com/jenlooper)

## Paraméterek finomhangolása

Miután az első tanítás befejeződött, figyeld meg a modell minőségét, és fontold meg annak javítását a „hiperparaméterek” finomhangolásával. Olvass többet a folyamatról [a dokumentációban](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-tune-hyperparameters?WT.mc_id=academic-77952-leestott).

## Előrejelzés

Ez az a pillanat, amikor teljesen új adatokat használhatsz a modell pontosságának tesztelésére. Egy „alkalmazott” ML környezetben, ahol webes eszközöket építesz a modell használatához a gyakorlatban, ez a folyamat magában foglalhatja a felhasználói bemenetek (például egy gombnyomás) összegyűjtését egy változó beállításához, amelyet elküldesz a modellnek következtetésre vagy értékelésre.

Ezekben a leckékben felfedezed, hogyan használhatod ezeket a lépéseket az adatok előkészítésére, modellek építésére, tesztelésére, értékelésére és előrejelzésére - mindazokat a mozdulatokat, amelyeket egy adatkutató végez, és még többet, ahogy haladsz az úton, hogy „full stack” ML mérnökké válj.

---

## 🚀Kihívás

Rajzolj egy folyamatábrát, amely tükrözi egy ML szakember lépéseit. Hol látod magad jelenleg a folyamatban? Hol gondolod, hogy nehézségekbe ütközöl? Mi tűnik könnyűnek számodra?

## [Utólagos kvíz](https://ff-quizzes.netlify.app/en/ml/)

## Áttekintés és önálló tanulás

Keress online interjúkat adatkutatókkal, akik a napi munkájukról beszélnek. Itt van [egy](https://www.youtube.com/watch?v=Z3IjgbbCEfs).

## Feladat

[Interjú egy adatkutatóval](assignment.md)

---

**Felelősség kizárása**:  
Ez a dokumentum az AI fordítási szolgáltatás [Co-op Translator](https://github.com/Azure/co-op-translator) segítségével lett lefordítva. Bár törekszünk a pontosságra, kérjük, vegye figyelembe, hogy az automatikus fordítások hibákat vagy pontatlanságokat tartalmazhatnak. Az eredeti dokumentum az eredeti nyelvén tekintendő hiteles forrásnak. Kritikus információk esetén javasolt professzionális emberi fordítást igénybe venni. Nem vállalunk felelősséget semmilyen félreértésért vagy téves értelmezésért, amely a fordítás használatából eredhet.