<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "9a6b702d1437c0467e3c5c28d763dac2",
  "translation_date": "2025-09-05T15:59:53+00:00",
  "source_file": "1-Introduction/3-fairness/README.md",
  "language_code": "hu"
}
-->
# Gépi tanulási megoldások építése felelős AI-val

![A felelős AI összefoglalása a gépi tanulásban egy sketchnote-ban](../../../../sketchnotes/ml-fairness.png)
> Sketchnote készítette: [Tomomi Imura](https://www.twitter.com/girlie_mac)

## [Előadás előtti kvíz](https://ff-quizzes.netlify.app/en/ml/)

## Bevezetés

Ebben a tananyagban elkezdjük felfedezni, hogyan hat a gépi tanulás a mindennapi életünkre. Már most is rendszerek és modellek vesznek részt napi döntéshozatali feladatokban, például egészségügyi diagnózisokban, hitelkérelmek jóváhagyásában vagy csalások észlelésében. Ezért fontos, hogy ezek a modellek megbízható eredményeket nyújtsanak. Ahogy bármely szoftveralkalmazás, az AI rendszerek is elmaradhatnak az elvárásoktól, vagy nemkívánatos eredményt hozhatnak. Ezért elengedhetetlen, hogy megértsük és magyarázni tudjuk egy AI modell viselkedését.

Képzeljük el, mi történik, ha az adatok, amelyeket ezeknek a modelleknek az építéséhez használunk, bizonyos demográfiai csoportokat nem tartalmaznak, például faji, nemi, politikai nézetek, vallás, vagy aránytalanul képviselik ezeket. Mi történik, ha a modell kimenete egyes demográfiai csoportokat előnyben részesít? Mi a következmény az alkalmazásra nézve? Továbbá, mi történik, ha a modell káros hatást gyakorol, és árt az embereknek? Ki felelős az AI rendszerek viselkedéséért? Ezeket a kérdéseket fogjuk megvizsgálni ebben a tananyagban.

Ebben a leckében:

- Felhívjuk a figyelmet a gépi tanulásban való méltányosság fontosságára és a méltányossággal kapcsolatos károkra.
- Megismerkedünk azzal a gyakorlattal, hogy a szélsőséges eseteket és szokatlan forgatókönyveket vizsgáljuk a megbízhatóság és biztonság érdekében.
- Megértjük, miért fontos mindenkit felhatalmazni inkluzív rendszerek tervezésével.
- Felfedezzük, milyen létfontosságú a személyes adatok és az emberek biztonságának védelme.
- Megértjük, miért fontos az "üvegdoboz" megközelítés az AI modellek viselkedésének magyarázatában.
- Tudatosítjuk, hogy az elszámoltathatóság elengedhetetlen az AI rendszerekbe vetett bizalom kiépítéséhez.

## Előfeltétel

Előfeltételként kérjük, végezze el a "Felelős AI alapelvei" tanulási útvonalat, és nézze meg az alábbi videót a témáról:

Tudjon meg többet a felelős AI-ról ezen a [tanulási útvonalon](https://docs.microsoft.com/learn/modules/responsible-ai-principles/?WT.mc_id=academic-77952-leestott)

[![Microsoft megközelítése a felelős AI-hoz](https://img.youtube.com/vi/dnC8-uUZXSc/0.jpg)](https://youtu.be/dnC8-uUZXSc "Microsoft megközelítése a felelős AI-hoz")

> 🎥 Kattintson a fenti képre a videóért: Microsoft megközelítése a felelős AI-hoz

## Méltányosság

Az AI rendszereknek mindenkit méltányosan kell kezelniük, és el kell kerülniük, hogy hasonló csoportokat különböző módon érintsenek. Például, amikor az AI rendszerek orvosi kezelési tanácsokat, hitelkérelmeket vagy foglalkoztatási ajánlásokat nyújtanak, ugyanazokat az ajánlásokat kell tenniük mindenki számára, akik hasonló tünetekkel, pénzügyi helyzettel vagy szakmai képesítéssel rendelkeznek. Mindannyian örökölt előítéleteket hordozunk magunkban, amelyek befolyásolják döntéseinket és cselekedeteinket. Ezek az előítéletek megjelenhetnek az adatokban, amelyeket az AI rendszerek képzéséhez használunk. Az ilyen manipuláció néha akaratlanul történik. Gyakran nehéz tudatosan felismerni, mikor vezetünk be előítéletet az adatokba.

**„Méltánytalanság”** olyan negatív hatásokat vagy „károkat” foglal magában, amelyek egy csoportot érintenek, például faji, nemi, életkori vagy fogyatékossági státusz alapján. A méltányossággal kapcsolatos főbb károk a következők:

- **Elosztás**, ha például egy nem vagy etnikum előnyben részesül egy másikkal szemben.
- **Szolgáltatás minősége**. Ha az adatokat egy konkrét forgatókönyvre képezzük, de a valóság sokkal összetettebb, az gyenge teljesítményű szolgáltatáshoz vezet. Például egy kézmosó adagoló, amely nem érzékeli a sötét bőrű embereket. [Referencia](https://gizmodo.com/why-cant-this-soap-dispenser-identify-dark-skin-1797931773)
- **Becsmérlés**. Valami vagy valaki igazságtalan kritizálása és címkézése. Például egy képfelismerő technológia hírhedten gorillának címkézte a sötét bőrű emberek képeit.
- **Túl- vagy alulreprezentáció**. Az a gondolat, hogy egy bizonyos csoportot nem látunk egy bizonyos szakmában, és minden szolgáltatás vagy funkció, amely ezt tovább erősíti, hozzájárul a kárhoz.
- **Sztereotipizálás**. Egy adott csoportot előre meghatározott attribútumokkal társítani. Például egy angol és török közötti nyelvi fordítórendszer pontatlanságokat mutathat a nemekhez kapcsolódó sztereotip asszociációk miatt.

![Fordítás törökre](../../../../1-Introduction/3-fairness/images/gender-bias-translate-en-tr.png)
> Fordítás törökre

![Fordítás vissza angolra](../../../../1-Introduction/3-fairness/images/gender-bias-translate-tr-en.png)
> Fordítás vissza angolra

Az AI rendszerek tervezése és tesztelése során biztosítanunk kell, hogy az AI méltányos legyen, és ne legyen programozva előítéletes vagy diszkriminatív döntések meghozatalára, amelyeket az emberek számára is tiltanak. Az AI és gépi tanulás méltányosságának garantálása továbbra is összetett társadalmi-technikai kihívás.

### Megbízhatóság és biztonság

Az AI rendszereknek megbízhatónak, biztonságosnak és következetesnek kell lenniük normál és váratlan körülmények között. Fontos tudni, hogyan viselkednek az AI rendszerek különböző helyzetekben, különösen szélsőséges esetekben. Az AI megoldások építésekor jelentős figyelmet kell fordítani arra, hogyan kezeljük az AI megoldások által tapasztalt különféle körülményeket. Például egy önvezető autónak az emberek biztonságát kell elsődleges prioritásként kezelnie. Ennek eredményeként az autót működtető AI-nak figyelembe kell vennie az összes lehetséges forgatókönyvet, amelyet az autó találhat, például éjszaka, viharok vagy hóviharok, gyerekek, akik átszaladnak az úton, háziállatok, útépítések stb. Az AI rendszer megbízható és biztonságos kezelése széles körülmények között tükrözi az adatkutató vagy AI fejlesztő által a rendszer tervezése vagy tesztelése során figyelembe vett előrelátás szintjét.

> [🎥 Kattintson ide a videóért: ](https://www.microsoft.com/videoplayer/embed/RE4vvIl)

### Inkluzivitás

Az AI rendszereket úgy kell megtervezni, hogy mindenkit bevonjanak és felhatalmazzanak. Az AI rendszerek tervezése és megvalósítása során az adatkutatók és AI fejlesztők azonosítják és kezelik a rendszerben lévő potenciális akadályokat, amelyek akaratlanul kizárhatnak embereket. Például világszerte 1 milliárd ember él fogyatékossággal. Az AI fejlődésével könnyebben hozzáférhetnek információkhoz és lehetőségekhez a mindennapi életükben. Az akadályok kezelésével lehetőséget teremtünk az innovációra és az AI termékek fejlesztésére, amelyek jobb élményeket nyújtanak mindenki számára.

> [🎥 Kattintson ide a videóért: inkluzivitás az AI-ban](https://www.microsoft.com/videoplayer/embed/RE4vl9v)

### Biztonság és adatvédelem

Az AI rendszereknek biztonságosnak kell lenniük, és tiszteletben kell tartaniuk az emberek magánéletét. Az emberek kevésbé bíznak azokban a rendszerekben, amelyek veszélyeztetik a magánéletüket, információikat vagy életüket. A gépi tanulási modellek képzésekor az adatokra támaszkodunk a legjobb eredmények elérése érdekében. Ennek során figyelembe kell venni az adatok eredetét és integritását. Például, az adatok felhasználói beküldésűek vagy nyilvánosan elérhetők voltak? Továbbá, az adatokkal való munka során elengedhetetlen olyan AI rendszerek fejlesztése, amelyek képesek megvédeni a bizalmas információkat és ellenállni a támadásoknak. Ahogy az AI egyre elterjedtebbé válik, a magánélet védelme és a fontos személyes és üzleti információk biztonságának megőrzése egyre kritikusabbá és összetettebbé válik. Az adatvédelem és adatbiztonság kérdései különösen nagy figyelmet igényelnek az AI esetében, mivel az adatokhoz való hozzáférés elengedhetetlen az AI rendszerek számára, hogy pontos és megalapozott előrejelzéseket és döntéseket hozzanak az emberekről.

> [🎥 Kattintson ide a videóért: biztonság az AI-ban](https://www.microsoft.com/videoplayer/embed/RE4voJF)

- Az iparág jelentős előrelépéseket tett az adatvédelem és biztonság terén, amelyet jelentősen ösztönöztek olyan szabályozások, mint a GDPR (Általános Adatvédelmi Rendelet).
- Az AI rendszerekkel azonban el kell ismernünk a feszültséget a személyes adatok szükségessége és a magánélet védelme között.
- Ahogy az internethez kapcsolt számítógépek születésével, az AI-val kapcsolatos biztonsági problémák száma is jelentősen megnőtt.
- Ugyanakkor az AI-t a biztonság javítására is használjuk. Például a legtöbb modern víruskereső szkennert AI-alapú heurisztikák vezérlik.
- Biztosítanunk kell, hogy az adatkutatási folyamataink harmonikusan illeszkedjenek a legújabb adatvédelmi és biztonsági gyakorlatokhoz.

### Átláthatóság

Az AI rendszereknek érthetőnek kell lenniük. Az átláthatóság kulcsfontosságú része az AI rendszerek és azok összetevőinek viselkedésének magyarázata. Az AI rendszerek megértésének javítása megköveteli, hogy az érintettek megértsék, hogyan és miért működnek, hogy azonosítani tudják a lehetséges teljesítményproblémákat, biztonsági és adatvédelmi aggályokat, előítéleteket, kizáró gyakorlatokat vagy nem szándékos eredményeket. Úgy gondoljuk, hogy azoknak, akik AI rendszereket használnak, őszintének és nyíltnak kell lenniük arról, hogy mikor, miért és hogyan döntenek azok alkalmazása mellett. Valamint a rendszerek korlátairól. Például, ha egy bank AI rendszert használ a fogyasztói hiteldöntések támogatására, fontos megvizsgálni az eredményeket, és megérteni, hogy mely adatok befolyásolják a rendszer ajánlásait. A kormányok elkezdték szabályozni az AI-t az iparágakban, így az adatkutatóknak és szervezeteknek magyarázatot kell adniuk arra, hogy az AI rendszer megfelel-e a szabályozási követelményeknek, különösen, ha nem kívánatos eredmény születik.

> [🎥 Kattintson ide a videóért: átláthatóság az AI-ban](https://www.microsoft.com/videoplayer/embed/RE4voJF)

- Mivel az AI rendszerek nagyon összetettek, nehéz megérteni, hogyan működnek és értelmezni az eredményeket.
- Ez a megértés hiánya befolyásolja, hogyan kezelik, üzemeltetik és dokumentálják ezeket a rendszereket.
- Ez a megértés hiánya még fontosabb módon befolyásolja azokat a döntéseket, amelyeket ezeknek a rendszereknek az eredményei alapján hoznak.

### Elszámoltathatóság

Azoknak, akik AI rendszereket terveznek és telepítenek, felelősséget kell vállalniuk rendszereik működéséért. Az elszámoltathatóság szükségessége különösen fontos az érzékeny technológiák, például az arcfelismerés esetében. Az utóbbi időben egyre nagyobb igény mutatkozik az arcfelismerő technológia iránt, különösen a bűnüldöző szervezetek részéről, akik látják a technológia lehetőségeit például eltűnt gyermekek megtalálásában. Azonban ezek a technológiák potenciálisan veszélyeztethetik az állampolgárok alapvető szabadságjogait, például az egyének folyamatos megfigyelésének lehetővé tételével. Ezért az adatkutatóknak és szervezeteknek felelősséget kell vállalniuk AI rendszerük egyénekre vagy társadalomra gyakorolt hatásáért.

[![Vezető AI kutató figyelmeztet a tömeges megfigyelés veszélyeire arcfelismeréssel](../../../../1-Introduction/3-fairness/images/accountability.png)](https://www.youtube.com/watch?v=Wldt8P5V6D0 "Microsoft megközelítése a felelős AI-hoz")

> 🎥 Kattintson a fenti képre a videóért: Figyelmeztetés a tömeges megfigyelés veszélyeire arcfelismeréssel

Végső soron az egyik legnagyobb kérdés generációnk számára, mint az első generáció, amely AI-t hoz a társadalomba, az, hogyan biztosíthatjuk, hogy a számítógépek továbbra is elszámoltathatók maradjanak az emberek számára, és hogyan biztosíthatjuk, hogy a számítógépeket tervező emberek elszámoltathatók maradjanak mindenki más számára.

## Hatásvizsgálat

Mielőtt gépi tanulási modellt képeznénk, fontos hatásvizsgálatot végezni, hogy megértsük az AI rendszer célját; mi a tervezett felhasználás; hol lesz telepítve; és kik fognak interakcióba lépni a rendszerrel. Ezek segítenek a rendszer értékelését végzőknek vagy tesztelőknek, hogy tudják, milyen tényezőket kell figyelembe venniük a lehetséges kockázatok és várható következmények azonosításakor.

A hatásvizsgálat során az alábbi területekre kell összpontosítani:

* **Kedvezőtlen hatás az egyénekre**. Fontos tudatában lenni minden korlátozásnak vagy követelménynek, nem támogatott használatnak vagy ismert korlátozásnak, amelyek akadályozhatják a rendszer teljesítményét, hogy biztosítsuk, hogy a rendszer ne okozzon kárt az egyéneknek.
* **Adatigények**. Az adatok felhasználásának módj
Nézd meg ezt a workshopot, hogy mélyebben elmerülj a témákban:

- A felelős mesterséges intelligencia nyomában: Elvek gyakorlati alkalmazása Besmira Nushi, Mehrnoosh Sameki és Amit Sharma előadásában

[![Responsible AI Toolbox: Nyílt forráskódú keretrendszer a felelős mesterséges intelligencia építéséhez](https://img.youtube.com/vi/tGgJCrA-MZU/0.jpg)](https://www.youtube.com/watch?v=tGgJCrA-MZU "RAI Toolbox: Nyílt forráskódú keretrendszer a felelős mesterséges intelligencia építéséhez")

> 🎥 Kattints a fenti képre a videóért: RAI Toolbox: Nyílt forráskódú keretrendszer a felelős mesterséges intelligencia építéséhez Besmira Nushi, Mehrnoosh Sameki és Amit Sharma előadásában

Olvasd el továbbá:

- Microsoft RAI erőforrásközpontja: [Responsible AI Resources – Microsoft AI](https://www.microsoft.com/ai/responsible-ai-resources?activetab=pivot1%3aprimaryr4)

- Microsoft FATE kutatócsoportja: [FATE: Fairness, Accountability, Transparency, and Ethics in AI - Microsoft Research](https://www.microsoft.com/research/theme/fate/)

RAI Toolbox:

- [Responsible AI Toolbox GitHub repository](https://github.com/microsoft/responsible-ai-toolbox)

Olvass az Azure Machine Learning eszközeiről, amelyek a méltányosság biztosítását szolgálják:

- [Azure Machine Learning](https://docs.microsoft.com/azure/machine-learning/concept-fairness-ml?WT.mc_id=academic-77952-leestott)

## Feladat

[Ismerd meg a RAI Toolboxot](assignment.md)

---

**Felelősség kizárása**:  
Ez a dokumentum az AI fordítási szolgáltatás, a [Co-op Translator](https://github.com/Azure/co-op-translator) segítségével lett lefordítva. Bár törekszünk a pontosságra, kérjük, vegye figyelembe, hogy az automatikus fordítások hibákat vagy pontatlanságokat tartalmazhatnak. Az eredeti dokumentum az eredeti nyelvén tekintendő hiteles forrásnak. Kritikus információk esetén javasolt professzionális emberi fordítást igénybe venni. Nem vállalunk felelősséget semmilyen félreértésért vagy téves értelmezésért, amely a fordítás használatából eredhet.