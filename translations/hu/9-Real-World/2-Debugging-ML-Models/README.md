<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "df2b538e8fbb3e91cf0419ae2f858675",
  "translation_date": "2025-09-05T15:54:19+00:00",
  "source_file": "9-Real-World/2-Debugging-ML-Models/README.md",
  "language_code": "hu"
}
-->
# Utószó: Modellhibakeresés gépi tanulásban a Responsible AI dashboard komponenseivel

## [Előadás előtti kvíz](https://ff-quizzes.netlify.app/en/ml/)

## Bevezetés

A gépi tanulás hatással van mindennapi életünkre. Az AI egyre inkább megjelenik olyan rendszerekben, amelyek alapvetően befolyásolják az egyéneket és a társadalmat, például az egészségügyben, pénzügyekben, oktatásban és foglalkoztatásban. Például rendszerek és modellek vesznek részt napi döntéshozatali feladatokban, mint például egészségügyi diagnózisok vagy csalások észlelése. Az AI fejlődése és gyors elterjedése azonban új társadalmi elvárásokkal és növekvő szabályozással találkozik. Gyakran látjuk, hogy az AI rendszerek nem felelnek meg az elvárásoknak, új kihívásokat vetnek fel, és a kormányok elkezdik szabályozni az AI megoldásokat. Ezért fontos, hogy ezeket a modelleket elemezzük, hogy mindenki számára igazságos, megbízható, befogadó, átlátható és felelősségteljes eredményeket biztosítsanak.

Ebben a tananyagban gyakorlati eszközöket mutatunk be, amelyekkel megvizsgálható, hogy egy modell rendelkezik-e felelősségteljes AI problémákkal. A hagyományos gépi tanulási hibakeresési technikák általában kvantitatív számításokon alapulnak, mint például az összesített pontosság vagy az átlagos hibaveszteség. Gondoljunk bele, mi történik, ha az adatok, amelyeket a modellek építéséhez használunk, bizonyos demográfiai csoportokat nem tartalmaznak, például faji, nemi, politikai nézetek vagy vallási csoportokat, vagy aránytalanul képviselik ezeket. Mi történik akkor, ha a modell kimenete egyes demográfiai csoportokat előnyben részesít? Ez túl- vagy alulképviseletet eredményezhet az érzékeny jellemzőcsoportokban, ami igazságossági, befogadási vagy megbízhatósági problémákat okozhat. Továbbá, a gépi tanulási modelleket gyakran "fekete dobozként" kezelik, ami megnehezíti annak megértését és magyarázatát, hogy mi vezérli a modell előrejelzéseit. Ezek mind olyan kihívások, amelyekkel az adatkutatók és AI fejlesztők szembesülnek, ha nincsenek megfelelő eszközeik a modellek igazságosságának vagy megbízhatóságának hibakeresésére és értékelésére.

Ebben a leckében megtanulhatod, hogyan végezz hibakeresést a modelleken az alábbiak segítségével:

- **Hibaelemzés**: azonosítsd, hogy az adateloszlás mely részeinél magas a modell hibaaránya.
- **Modelláttekintés**: végezz összehasonlító elemzést különböző adatcsoportok között, hogy felfedezd a modell teljesítménymutatóiban lévő eltéréseket.
- **Adatelemzés**: vizsgáld meg, hol lehet túl- vagy alulképviselet az adataidban, ami a modellt egy demográfiai csoport előnyben részesítésére késztetheti egy másikkal szemben.
- **Jellemzők fontossága**: értsd meg, mely jellemzők befolyásolják a modell előrejelzéseit globális vagy lokális szinten.

## Előfeltétel

Előfeltételként kérjük, tekintsd át a [Felelősségteljes AI eszközök fejlesztőknek](https://www.microsoft.com/ai/ai-lab-responsible-ai-dashboard) című anyagot.

> ![Gif a felelősségteljes AI eszközökről](../../../../9-Real-World/2-Debugging-ML-Models/images/rai-overview.gif)

## Hibaelemzés

A hagyományos modellteljesítmény-mutatók, amelyeket a pontosság mérésére használnak, többnyire helyes és helytelen előrejelzések alapján végzett számítások. Például egy modell, amely 89%-ban pontos, és 0,001 hibaveszteséggel rendelkezik, jó teljesítményűnek tekinthető. Azonban a hibák gyakran nem oszlanak el egyenletesen az alapul szolgáló adathalmazban. Lehet, hogy 89%-os pontossági eredményt kapsz, de felfedezed, hogy az adatok bizonyos régióiban a modell 42%-ban hibázik. Az ilyen hibaminták következményei bizonyos adatcsoportokkal igazságossági vagy megbízhatósági problémákhoz vezethetnek. Fontos megérteni, hogy a modell hol teljesít jól vagy rosszul. Azok az adatrégiók, ahol a modell pontatlanságai magasak, fontos demográfiai csoportok lehetnek.

![Modellek hibáinak elemzése és hibakeresése](../../../../9-Real-World/2-Debugging-ML-Models/images/ea-error-distribution.png)

A RAI dashboard Hibaelemzés komponense megmutatja, hogyan oszlanak el a modellhibák különböző csoportok között egy fa vizualizáció segítségével. Ez hasznos annak azonosításában, hogy mely jellemzők vagy területek okoznak magas hibaarányt az adathalmazban. Azáltal, hogy látod, honnan származnak a modell pontatlanságai, elkezdheted vizsgálni a gyökérokokat. Adatcsoportokat is létrehozhatsz az elemzéshez. Ezek az adatcsoportok segítenek a hibakeresési folyamatban annak meghatározásában, hogy miért teljesít jól a modell az egyik csoportban, de hibázik a másikban.

![Hibaelemzés](../../../../9-Real-World/2-Debugging-ML-Models/images/ea-error-cohort.png)

A fa térkép vizuális jelzői segítenek gyorsabban megtalálni a problémás területeket. Például minél sötétebb piros színű egy fa csomópont, annál magasabb a hibaarány.

A hőtérkép egy másik vizualizációs funkció, amelyet a felhasználók használhatnak a hibaarány vizsgálatára egy vagy két jellemző alapján, hogy megtalálják a modellhibák hozzájáruló tényezőit az egész adathalmazban vagy csoportokban.

![Hibaelemzés hőtérkép](../../../../9-Real-World/2-Debugging-ML-Models/images/ea-heatmap.png)

Használj hibaelemzést, ha:

* Mélyebb megértést szeretnél szerezni arról, hogyan oszlanak el a modellhibák az adathalmazon és több bemeneti és jellemző dimenzión keresztül.
* Fel szeretnéd bontani az összesített teljesítménymutatókat, hogy automatikusan felfedezd a hibás csoportokat, és célzott enyhítési lépéseket tegyél.

## Modelláttekintés

Egy gépi tanulási modell teljesítményének értékelése átfogó megértést igényel a viselkedéséről. Ez több mutató, például hibaarány, pontosság, visszahívás, precizitás vagy MAE (átlagos abszolút hiba) áttekintésével érhető el, hogy feltárjuk a teljesítménymutatók közötti eltéréseket. Egy mutató lehet, hogy kiválóan néz ki, de egy másik mutatóban pontatlanságok derülhetnek ki. Ezenkívül a mutatók összehasonlítása az egész adathalmazon vagy csoportokon belül segít rávilágítani arra, hogy a modell hol teljesít jól vagy rosszul. Ez különösen fontos annak megértésében, hogy a modell hogyan teljesít érzékeny és nem érzékeny jellemzők között (pl. beteg faja, neme vagy életkora), hogy feltárjuk a modell esetleges igazságtalanságait. Például, ha felfedezzük, hogy a modell hibásabb egy érzékeny jellemzőket tartalmazó csoportban, az igazságtalanságot jelezhet.

A RAI dashboard Modelláttekintés komponense nemcsak az adatreprezentáció teljesítménymutatóinak elemzésében segít egy csoportban, hanem lehetőséget ad a modell viselkedésének összehasonlítására különböző csoportok között.

![Adathalmaz csoportok - modelláttekintés a RAI dashboardon](../../../../9-Real-World/2-Debugging-ML-Models/images/model-overview-dataset-cohorts.png)

A komponens jellemző-alapú elemzési funkciója lehetővé teszi a felhasználók számára, hogy szűkítsék az adatcsoportokat egy adott jellemzőn belül, hogy anomáliákat azonosítsanak részletes szinten. Például a dashboard beépített intelligenciával automatikusan generál csoportokat egy felhasználó által kiválasztott jellemző alapján (pl. *"time_in_hospital < 3"* vagy *"time_in_hospital >= 7"*). Ez lehetővé teszi a felhasználó számára, hogy egy adott jellemzőt elkülönítsen egy nagyobb adatcsoportból, hogy lássa, ez-e a kulcsfontosságú tényező a modell hibás eredményeiben.

![Jellemző csoportok - modelláttekintés a RAI dashboardon](../../../../9-Real-World/2-Debugging-ML-Models/images/model-overview-feature-cohorts.png)

A Modelláttekintés komponens két osztályú eltérési mutatót támogat:

**Eltérés a modell teljesítményében**: Ezek a mutatók kiszámítják az eltérést (különbséget) a kiválasztott teljesítménymutató értékei között az adatcsoportokban. Néhány példa:

* Pontossági arány eltérése
* Hibaarány eltérése
* Precizitás eltérése
* Visszahívás eltérése
* Átlagos abszolút hiba (MAE) eltérése

**Eltérés a kiválasztási arányban**: Ez a mutató tartalmazza a kiválasztási arány (kedvező előrejelzés) különbségét az adatcsoportok között. Példa erre a hiteljóváhagyási arány eltérése. A kiválasztási arány azt jelenti, hogy az egyes osztályok adatpontjainak hány százalékát osztályozzák 1-nek (bináris osztályozásban) vagy az előrejelzési értékek eloszlását (regresszióban).

## Adatelemzés

> "Ha elég sokáig kínozod az adatokat, bármit bevallanak" - Ronald Coase

Ez az állítás szélsőségesen hangzik, de igaz, hogy az adatok manipulálhatók bármilyen következtetés támogatására. Az ilyen manipuláció néha akaratlanul történik. Emberek vagyunk, és mindannyian rendelkezünk előítéletekkel, amelyeket gyakran nehéz tudatosan felismerni, amikor adatokat torzítunk. Az igazságosság biztosítása az AI-ban és a gépi tanulásban továbbra is összetett kihívás.

Az adatok nagy vakfoltot jelentenek a hagyományos modellteljesítmény-mutatók számára. Lehet, hogy magas pontossági eredményeket kapsz, de ez nem mindig tükrözi az adathalmazban lévő alapvető adatelfogultságot. Például, ha egy vállalat alkalmazottainak adathalmazában az ügyvezető pozíciókban 27% nő és 73% férfi van, egy álláshirdetési AI modell, amelyet ezen adatok alapján képeztek, valószínűleg főként férfi közönséget céloz meg vezetői pozíciókra. Az adatokban lévő egyensúlyhiány a modell előrejelzését egy nem előnyben részesítésére késztette. Ez igazságossági problémát tár fel, ahol nemi elfogultság van az AI modellben.

A RAI dashboard Adatelemzés komponense segít azonosítani azokat a területeket, ahol túl- vagy alulképviselet van az adathalmazban. Segít a felhasználóknak diagnosztizálni azokat a hibák és igazságossági problémák gyökérokait, amelyeket az adatok egyensúlyhiánya vagy egy adott adatcsoport hiánya okoz. Ez lehetőséget ad a felhasználóknak arra, hogy vizualizálják az adathalmazokat előrejelzett és valós eredmények, hibacsoportok és konkrét jellemzők alapján. Néha egy alulképviselt adatcsoport felfedezése azt is feltárhatja, hogy a modell nem tanul jól, ezért magas a pontatlanság. Egy adatelfogultsággal rendelkező modell nemcsak igazságossági problémát jelent, hanem azt is mutatja, hogy a modell nem befogadó vagy megbízható.

![Adatelemzés komponens a RAI dashboardon](../../../../9-Real-World/2-Debugging-ML-Models/images/dataanalysis-cover.png)

Használj adatelemzést, ha:

* Felfedezni szeretnéd az adathalmaz statisztikáit különböző szűrők kiválasztásával, hogy az adatokat különböző dimenziókra (más néven csoportokra) bontsd.
* Megérteni szeretnéd az adathalmaz eloszlását különböző csoportok és jellemzőcsoportok között.
* Meghatározni szeretnéd, hogy az igazságossággal, hibaelemzéssel és ok-okozati összefüggésekkel kapcsolatos megállapításaid (amelyeket más dashboard komponensekből származtattál) az adathalmaz eloszlásának eredményei-e.
* Eldönteni, hogy mely területeken gyűjts több adatot, hogy enyhítsd azokat a hibákat, amelyek reprezentációs problémákból, címkezajból, jellemzőzajból, címkeelfogultságból és hasonló tényezőkből származnak.

## Modellérthetőség

A gépi tanulási modellek gyakran "fekete dobozként" működnek. Nehéz megérteni, hogy mely kulcsfontosságú adatjellemzők vezérlik a modell előrejelzéseit. Fontos, hogy átláthatóságot biztosítsunk arra vonatkozóan, hogy miért hoz egy modell bizonyos előrejelzést. Például, ha egy AI rendszer azt jósolja, hogy egy cukorbeteg páciensnél fennáll a kockázata annak, hogy 30 napon belül visszakerül a kórházba, akkor képesnek kell lennie arra, hogy támogató adatokat nyújtson, amelyek az előrejelzéséhez vezettek. A támogató adatjelzők átláthatóságot biztosítanak, hogy segítsenek az orvosoknak vagy kórházaknak jól informált döntéseket hozni. Ezenkívül az, hogy megmagyarázható, miért hozott egy modell előrejelzést egy adott páciens esetében, lehetővé teszi az egészségügyi szabályozásokkal való megfelelést. Amikor gépi tanulási modelleket használsz olyan módon, amely hatással van az emberek életére, elengedhetetlen megérteni és megmagyarázni, mi befolyásolja a modell viselkedését. A modell magyarázhatósága és érthetősége segít választ adni az alábbi helyzetekben:

* Modellhibakeresés: Miért követte el a modell ezt a hibát? Hogyan javíthatom a modellemet?
* Ember-AI együttműködés: Hogyan érthetem meg és bízhatok a modell döntéseiben?
* Szabályozási megfelelés: Megfelel-e a modellem a jogi követelményeknek?

A RAI dashboard Jellemzők fontossága komponense segít hibakeresésben és átfogó megértést nyújt arról, hogyan hoz egy modell előrejelzéseket. Ez egy hasznos eszköz gépi tanulási szakemberek és döntéshozók számára, hogy megmagyarázzák és bizonyítékot mutassanak arra, hogy mely jellemzők befolyásolják a modell viselkedését a szabályozási megfelelés érdekében. A felhasználók globális és lokális magyarázatokat is felfedezhetnek, hogy érvényesítsék, mely jellemzők vezérlik a modell előrejelzéseit. A globális magyarázatok felsorolják azokat a legfontosabb j
- **Túl- vagy alulreprezentáció**. Az elképzelés az, hogy egy bizonyos csoport nem jelenik meg egy adott szakmában, és bármely szolgáltatás vagy funkció, amely ezt tovább erősíti, káros hatást gyakorol.

### Azure RAI dashboard

Az [Azure RAI dashboard](https://learn.microsoft.com/en-us/azure/machine-learning/concept-responsible-ai-dashboard?WT.mc_id=aiml-90525-ruyakubu) nyílt forráskódú eszközökre épül, amelyeket vezető akadémiai intézmények és szervezetek, köztük a Microsoft fejlesztettek ki. Ezek az eszközök segítik az adatkutatókat és AI fejlesztőket abban, hogy jobban megértsék a modellek viselkedését, és hogy felfedezzék és enyhítsék az AI modellek nem kívánt problémáit.

- Ismerd meg, hogyan használhatod a különböző komponenseket az RAI dashboard [dokumentációjának](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-responsible-ai-dashboard?WT.mc_id=aiml-90525-ruyakubu) átnézésével.

- Nézd meg néhány RAI dashboard [példa notebookot](https://github.com/Azure/RAI-vNext-Preview/tree/main/examples/notebooks), amelyek segítenek felelősségteljesebb AI forgatókönyvek hibakeresésében az Azure Machine Learning-ben.

---

## 🚀 Kihívás

Annak érdekében, hogy statisztikai vagy adatbeli torzítások már eleve ne kerüljenek bevezetésre, a következőket kell tennünk:

- biztosítsuk, hogy a rendszereken dolgozó emberek különböző háttérrel és nézőpontokkal rendelkezzenek  
- fektessünk be olyan adathalmazokba, amelyek tükrözik társadalmunk sokszínűségét  
- fejlesszünk jobb módszereket a torzítások észlelésére és kijavítására, amikor azok előfordulnak  

Gondolj valós életbeli helyzetekre, ahol az igazságtalanság nyilvánvaló a modellek építése és használata során. Mit kellene még figyelembe vennünk?

## [Előadás utáni kvíz](https://ff-quizzes.netlify.app/en/ml/)

## Áttekintés és önálló tanulás

Ebben a leckében megismerkedtél néhány gyakorlati eszközzel, amelyek segítenek a felelősségteljes AI beépítésében a gépi tanulásba.

Nézd meg ezt a workshopot, hogy mélyebben elmerülj a témákban:

- Responsible AI Dashboard: Egyablakos megoldás a felelősségteljes AI gyakorlati alkalmazásához, előadók: Besmira Nushi és Mehrnoosh Sameki

[![Responsible AI Dashboard: Egyablakos megoldás a felelősségteljes AI gyakorlati alkalmazásához](https://img.youtube.com/vi/f1oaDNl3djg/0.jpg)](https://www.youtube.com/watch?v=f1oaDNl3djg "Responsible AI Dashboard: Egyablakos megoldás a felelősségteljes AI gyakorlati alkalmazásához")

> 🎥 Kattints a fenti képre a videóért: Responsible AI Dashboard: Egyablakos megoldás a felelősségteljes AI gyakorlati alkalmazásához, előadók: Besmira Nushi és Mehrnoosh Sameki

Használd az alábbi anyagokat, hogy többet megtudj a felelősségteljes AI-ról és arról, hogyan építhetsz megbízhatóbb modelleket:

- Microsoft RAI dashboard eszközei ML modellek hibakereséséhez: [Responsible AI tools resources](https://aka.ms/rai-dashboard)

- Fedezd fel a Responsible AI eszköztárat: [Github](https://github.com/microsoft/responsible-ai-toolbox)

- Microsoft RAI erőforrásközpontja: [Responsible AI Resources – Microsoft AI](https://www.microsoft.com/ai/responsible-ai-resources?activetab=pivot1%3aprimaryr4)

- Microsoft FATE kutatócsoportja: [FATE: Fairness, Accountability, Transparency, and Ethics in AI - Microsoft Research](https://www.microsoft.com/research/theme/fate/)

## Feladat

[Ismerd meg az RAI dashboardot](assignment.md)

---

**Felelősség kizárása**:  
Ez a dokumentum az AI fordítási szolgáltatás [Co-op Translator](https://github.com/Azure/co-op-translator) segítségével lett lefordítva. Bár törekszünk a pontosságra, kérjük, vegye figyelembe, hogy az automatikus fordítások hibákat vagy pontatlanságokat tartalmazhatnak. Az eredeti dokumentum az eredeti nyelvén tekintendő hiteles forrásnak. Kritikus információk esetén javasolt professzionális emberi fordítást igénybe venni. Nem vállalunk felelősséget semmilyen félreértésért vagy téves értelmezésért, amely a fordítás használatából eredhet.