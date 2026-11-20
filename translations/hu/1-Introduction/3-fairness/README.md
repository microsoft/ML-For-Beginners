# Felelős MI-vel gépi tanulási megoldások építése
 
![Összefoglaló a felelős MI-ről gépi tanulásban egy sketchnote-on](../../../../translated_images/hu/ml-fairness.ef296ebec6afc98a.webp)
> Sketchnote készítette: [Tomomi Imura](https://www.twitter.com/girlie_mac)

## [Előadás előtti kvíz](https://ff-quizzes.netlify.app/en/ml/)
 
## Bevezetés

Ebben a tananyagsorozatban elkezded felfedezni, hogy a gépi tanulás hogyan és milyen hatással van a mindennapi életünkre. Már most is rendszerek és modellek vesznek részt napi döntéshozatali feladatokban, például egészségügyi diagnózisokban, hitelbírálatokban vagy csalásfelderítésben. Ezért fontos, hogy ezek a modellek jól működjenek, megbízható eredményeket nyújtva. Ahogy bármely szoftveralkalmazás, úgy a MI rendszerek sem mindig felelnek meg az elvárásoknak, vagy váratlan, nem kívánatos eredményt adhatnak. Ezért elengedhetetlen, hogy megértsük és elmagyarázzuk egy MI modell viselkedését.

Képzeld el, mi történhet, ha azok az adatok, amelyeket ezekhez a modellekhez használsz, hiányosak bizonyos demográfiai csoportokat illetően, mint például faj, nem, politikai nézet, vallás, vagy aránytalanul képviselnek egyes demográfiai csoportokat. Mi történik, ha a modell kimenetét úgy értelmezik, hogy egy demográfiai csoport javára szolgáljon? Mi ennek a következménye az alkalmazás szempontjából? Továbbá, mi történik, ha a modell káros eredményt hoz és árt az embereknek? Ki a felelős az MI rendszer viselkedéséért? Ezekre a kérdésekre keressük a választ ebben a tananyagban.

Ebben a leckében a következőket fogod megtanulni:

- Tudatosságod növelése a gépi tanulásban rejlő méltányosság fontosságáról és a méltányosságot érintő károkról.
- Ismerkedés a szokatlan esetek és kilógó értékek vizsgálatának gyakorlatával a megbízhatóság és biztonság biztosítása érdekében.
- Megértés arról, miért fontos mindenkit felhatalmazni befogadó rendszerek tervezésével.
- Felfedezés arról, milyen fontos az adat- és személyes adatok magánéletének és biztonságának védelme.
- Látni a "üveg doboz" megközelítés fontosságát az MI modellek viselkedésének magyarázatában.
- Tudatosság a felelősségvállalásról, amely elengedhetetlen a MI rendszerekbe vetett bizalom kiépítéséhez.

## Előfeltétel

Előfeltételként vedd végig a „Felelős MI alapelvek” tanulási útvonalat, és nézd meg az alábbi videót a témában:

Tudj meg többet a Felelős MI-ről ezen a [Tanulási Útvonalon](https://docs.microsoft.com/learn/modules/responsible-ai-principles/?WT.mc_id=academic-77952-leestott)

[![Microsoft megközelítése a Felelős MI-re](https://img.youtube.com/vi/dnC8-uUZXSc/0.jpg)](https://youtu.be/dnC8-uUZXSc "Microsoft megközelítése a Felelős MI-re")

> 🎥 Kattints a fenti képre a videóhoz: Microsoft megközelítése a Felelős MI-re

## Méltányosság

Az MI rendszereknek mindenkivel méltányosan kell bánniuk, és el kell kerülniük, hogy hasonló csoportokat eltérő módon érjenek hatások. Például amikor MI rendszerek adnak tanácsot egészségügyi kezelésekhez, hitelbírálathoz vagy foglalkoztatáshoz, ugyanazokat a javaslatokat kell megadniuk minden olyan embernek, akik hasonló tünetekkel, anyagi helyzettel vagy szakmai képesítéssel rendelkeznek. Mindannyian örökölt torzításokat hordozunk döntéseinkben és cselekedeteinkben. Ezek a torzítások megjelenhetnek az adatokban is, amelyeket a MI rendszerek tanítására használunk. Az ilyen manipulációk előfordulhatnak akaratlanul is. Gyakran nehéz tudatosan felismerni, mikor viszünk be torzítást az adatokba.

**„Mély méltánytalanság”** negatív hatásokat, vagy „károkat” jelent egy csoport számára, például faji, nemi, életkori vagy fogyatékossági státusz alapján meghatározott csoportok esetében. A főbb méltányossággal kapcsolatos károk a következők lehetnek:

- **Elosztás**, ha például egy nem vagy etnikum előnyben részesül egy másikkal szemben.
- **Szolgáltatás minősége**. Ha egy modellt egy adott szcenárióra tanítanak, de a valóság sokkal összetettebb, az rossz teljesítményhez vezet. Például egy kézmosó adagoló nem tudja érzékelni a sötétebb bőrű embereket. [Hivatkozás](https://gizmodo.com/why-cant-this-soap-dispenser-identify-dark-skin-1797931773)
- **Lealacsonyítás**. Valami vagy valaki méltánytalan kritikája és megbélyegzése. Például egy kép-felismerő technológia hírhedten gorillának címkézte fel a sötétebb bőrű emberek képeit.
- **Túlsúlyos vagy alulsúlyos képviselet**. Az az elképzelés, hogy egy bizonyos csoport nem jelenik meg adott szakmában, és minden szolgáltatás vagy funkció, amely ezt tovább erősíti, károsít.
- **Stereotipizálás**. Egy adott csoport összevonása előre hozzárendelt tulajdonságokkal. Például angol–török nyelvű fordító rendszer pontatlanságai, amelyek nemi sztereotipikus szóhasználatból erednek.

![fordítás törökre](../../../../translated_images/hu/gender-bias-translate-en-tr.f185fd8822c2d437.webp)
> fordítás törökre

![visszafordítás angolra](../../../../translated_images/hu/gender-bias-translate-tr-en.4eee7e3cecb8c70e.webp)
> visszafordítás angolra

Amikor MI rendszereket tervezünk és tesztelünk, biztosítanunk kell, hogy az MI méltányos legyen, és ne legyen programozva elfogult vagy diszkriminatív döntések meghozatalára, amit az embereknek is tiltott megtenni. A méltányosság garantálása az MI-ben és gépi tanulásban összetett szociotechnikai kihívás marad.

### Megbízhatóság és biztonság

A bizalom kiépítéséhez az MI rendszereknek megbízhatónak, biztonságosnak és kiszámíthatónak kell lenniük normál és váratlan helyzetekben egyaránt. Fontos tudni, hogyan viselkednek az MI rendszerek különböző helyzetekben, különösen akkor, ha kilógnak az átlagból. MI megoldások építésekor nagy hangsúlyt kell fektetni arra, hogyan kezelik az AI megoldások az általuk előforduló széles körű helyzeteket. Például egy önvezető autónak az emberek biztonsága a legfőbb szempont. Ennek megfelelően az autó MI-jének figyelembe kell vennie minden lehetséges helyzetet, amit az autó akár átélhet: éjszaka, vihar, hóvihar, utcán futó gyerekek, háziállatok, útépítés stb. Egy MI rendszer képessége arra, hogy megbízhatóan és biztonságosan kezelje a szélsőséges körülményeket, tükrözi a tervező vagy fejlesztő előrelátását a rendszer tervezése vagy tesztelése során.

> [🎥 Kattints ide egy videóért: Megbízhatóság és biztonság az MI-ben](https://www.microsoft.com/videoplayer/embed/RE4vvIl)

### Befogadás

Az MI rendszereket úgy kell tervezni, hogy mindenkit bevonjanak és felhatalmazzanak. Tervezés és bevezetés során az adattudósok és MI fejlesztők azonosítják és kezelik azokat az akadályokat, amelyek véletlenül kizárhatnak embereket. Például 1 milliárd fogyatékossággal élő ember van világszerte. Az MI fejlődésével ők könnyebben férhetnek hozzá széles körű információkhoz és lehetőségekhez a mindennapi életük során. Az akadályok feltárásával lehetőség nyílik arra, hogy jobb élményt kínáló MI termékeket alkossunk, amelyek mindenkinek hasznára válnak.

> [🎥 Kattints ide egy videóért: Befogadás az MI-ben](https://www.microsoft.com/videoplayer/embed/RE4vl9v)

### Biztonság és magánélet

Az MI rendszereknek biztonságosnak kell lenniük és tiszteletben kell tartaniuk az emberek magánéletét. Az emberek kevésbé bíznak olyan rendszerekben, amelyek veszélyeztetik magánéletüket, adataikat vagy életüket. Amikor gépi tanulási modelleket tanítunk, adatokat használunk a legjobb eredmények eléréséhez. Ezért figyelembe kell venni az adatok eredetét és integritását. Például az adatokat a felhasználó küldte be, vagy nyilvánosan elérhetőek? Ezután az adatokkal dolgozva létfontosságú olyan MI rendszereket fejleszteni, amelyek képesek megvédeni a bizalmas információkat és ellenállni a támadásoknak. Ahogy az MI egyre elterjedtebbé válik, egyre fontosabb és összetettebb lesz az adatvédelem és a személyes, valamint üzleti információk biztonsága. A magánélet és adatbiztonság különösen fontos az MI számára, mert az MI rendszereknek az adatokhoz való hozzáférésre van szükségük pontos és megalapozott előrejelzések és döntések meghozatalához az emberekről.

> [🎥 Kattints ide egy videóért: Biztonság az MI-ben](https://www.microsoft.com/videoplayer/embed/RE4voJF)

- Iparágként jelentős előrelépéseket tettünk az adatvédelem és a biztonság terén, jelentősen támaszkodva olyan szabályozásokra, mint a GDPR (Általános Adatvédelmi Rendelet).
- Ugyanakkor az MI rendszerek esetén el kell ismernünk a feszültséget az iránt, hogy több személyes adatra van szükség a személyesebb és hatékonyabb rendszerekhez – és a magánélet között.
- Akárcsak az internet megjelenésével járó összekapcsolt számítógépek esetén, az MI-vel kapcsolatos biztonsági kérdések száma is hirtelen megugrott.
- Egyidejűleg azt is láttuk, hogy az MI-t a biztonság javítására használják. Például a legtöbb modern vírusirtó scannert ma MI-heurisztikák működtetik.
- Biztosítanunk kell, hogy az adattudományi folyamataik összhangban legyenek a legújabb adatvédelmi és biztonsági gyakorlatokkal.


### Átláthatóság
Az MI rendszereknek érthetőnek kell lenniük. Az átláthatóság kulcsfontosságú része az MI rendszerek és alkotóelemeik viselkedésének magyarázata. Az MI rendszerek jobb megértéséhez szükséges, hogy az érdekelt felek megértsék hogyan és miért működnek ezek a rendszerek, hogy azonosítani tudják az esetleges teljesítménybeli problémákat, biztonsági és adatvédelmi aggályokat, elfogultságot, kizáró gyakorlatokat vagy nem várt eredményeket. Azt is gondoljuk, hogy az MI rendszereket használóknak őszintén és nyíltan kell beszámolniuk arról, mikor, miért és hogyan döntenek az alkalmazásukról. Illetve a használatuk korlátairól. Például, ha egy bank MI rendszert használ fogyasztói hiteldöntései támogatására, fontos megvizsgálni az eredményeket és megérteni, hogy milyen adatok befolyásolják a rendszer ajánlásait. A kormányok elkezdték szabályozni az MI-t az iparágakban, ezért az adattudósoknak és szervezeteknek meg kell tudniuk magyarázni, hogy egy MI rendszer megfelel-e a szabályozási követelményeknek, különösen nem kívánatos eredmény esetén.

> [🎥 Kattints ide egy videóért: Átláthatóság az MI-ben](https://www.microsoft.com/videoplayer/embed/RE4voJF)

- Az MI rendszerek annyira összetettek, hogy nehéz megérteni, hogyan működnek és értelmezni az eredményeket.
- Ez a megértés hiánya hatással van arra, hogyan kezelik, működtetik és dokumentálják ezeket a rendszereket.
- Ez a megértés hiánya még inkább befolyásolja a döntéseket, amelyeket a rendszerek eredményei alapján hoznak.

### Felelősségvállalás
 
Azoknak, akik MI rendszereket terveznek és bevezetnek, felelősséget kell vállalniuk rendszereik működéséért. A felelősségvállalás különösen fontos érzékeny technológiák, például arcfelismerés esetén. Az utóbbi időben egyre nagyobb igény mutatkozik az arcfelismerő technológiák iránt, különösen a rendvédelmi szerveknél, amelyek látják annak potenciálját olyan feladatokban, mint az eltűnt gyermekek felkutatása. Ugyanakkor ezek a technológiák potenciálisan arra is használhatók egy kormány részéről, hogy alapvető szabadságjogokat veszélyeztessenek, például folyamatos megfigyeléssel egyes egyének esetében. Ezért az adattudósoknak és szervezeteknek felelősen kell viszonyulniuk ahhoz, hogy MI rendszerük hogyan hat az egyénekre vagy a társadalomra.

[![Vezető MI kutató figyelmeztet az arcfelismerés által vezérelt tömeges megfigyelésre](../../../../translated_images/hu/accountability.41d8c0f4b85b6231.webp)](https://www.youtube.com/watch?v=Wldt8P5V6D0 "Microsoft megközelítése a Felelős MI-re")

> 🎥 Kattints a fenti képre a videóhoz: Figyelmeztetések az arcfelismerés által vezérelt tömeges megfigyelésről

Végül az egyik legfontosabb kérdés a mi generációnk számára, akik elsőként hozzuk az MI-t a társadalomba, hogy miként biztosítsuk, hogy a számítógépek felelősséggel tartozzanak az emberek irányában, és hogy a számítógépeket tervezők maguk is felelősségteljesek maradjanak mindenki felé.

## Hatásvizsgálat

Egy gépi tanulási modell tanítása előtt fontos hatásvizsgálatot végezni, hogy megértsük az MI rendszer célját; milyen használatra szánják; hol fog üzemelni; és ki fogja a rendszert használni. Ezek hasznosak a rendszer átvizsgálóinak vagy tesztelőinek, akik így tudják, milyen tényezőkre kell figyelni a lehetséges kockázatok és várható következmények azonosítása során.

A hatásvizsgálat során az alábbi területekre érdemes fókuszálni:

* **Káros hatás az egyénekre**. Tudatosság bármely korlátozásról vagy feltételről, támogatás nélküli használatról vagy ismert korlátozásokról, amelyek hátráltathatják a rendszer működését, létfontosságú, hogy a rendszert ne használják olyan módon, amely kárt okozhat az egyéneknek.
* **Adatkövetelmények**. A rendszer adatfelhasználásának megértése lehetővé teszi az átvizsgálók számára, hogy feltárják az esetleges adatkövetelményeket, amelyeket figyelembe kell venni (pl. GDPR vagy HIPAA adatvédelmi szabályozások). Emellett vizsgálják meg, hogy az adatforrás és mennyiség elegendő-e a képzéshez.
* **Hatás összefoglaló**. Gyűjts össze egy listát azokról a lehetséges károkról, amelyeket a rendszer használata okozhat. A gépi tanulási életciklus alatt folyamatosan vizsgáld, hogy az azonosított problémákat enyhítik vagy kezelik-e.
* **Alkalmazandó célok** a hat alappillér mindegyikére. Értékeld, hogy a pillérek céljait teljesítik-e, és vannak-e hiányosságok.


## Hibakeresés felelős MI-vel

Hasonlóan egy szoftveralkalmazás hibakereséséhez, az MI rendszer hibakeresése a rendszer problémáinak azonosítását és megoldását jelenti. Számos tényező befolyásolja, hogy egy modell nem teljesít az elvárt vagy felelős módon. A legtöbb hagyományos modell teljesítménymutató kvantitatív összesítője a modell teljesítményének, ami nem elegendő az MI alapelvek megsértésének elemzésére. Továbbá egy gépi tanulási modell egy "fekete doboz", amely megnehezíti, hogy megértsük, mi okozza az eredményt vagy magyarázatot adjunk hiba esetén. Később ebben a tanfolyamban megtanuljuk, hogyan használjuk a Felelős MI műszerfalat az MI rendszerek hibakereséséhez. A műszerfal átfogó eszközt nyújt adattudósoknak és fejlesztőknek a következőkhöz:

* **Hibaanalízis**. A modell hibáinak eloszlásának azonosítása, amelyek befolyásolhatják a rendszer méltányosságát vagy megbízhatóságát.
* **Modell áttekintés**. Annak felfedezése, hol vannak eltérések a modell teljesítményében az adat-alcsoportok között.
* **Adat elemzés**. Az adat-eloszlás megértése és az esetleges torzítások azonosítása az adatokban, amelyek méltányossági, befogadási és megbízhatósági problémákhoz vezethetnek.
* **Modell értelmezhetőség**. Megérteni, mi befolyásolja a modell előrejelzéseit. Ez segít a modell viselkedésének magyarázatában, ami fontos az átláthatóság és felelősségvállalás szempontjából.


## 🚀 Kihívás
 
Annak érdekében, hogy megakadályozzuk a károk kialakulását, a következőket kell tennünk:

- legyen sokféle háttérrel és nézőponttal rendelkező szakember a rendszereken dolgozók között
- fektessünk be olyan adatbázisokba, amelyek társadalmunk sokszínűségét tükrözik
- fejlesszünk jobb módszereket a gépi tanulás életciklusa során a felelőtlen MI felismerésére és kijavítására, amikor előfordul

Gondolkodj el valós példákon, amelyeknél egy modell megbízhatatlansága egyértelműen megmutatkozik az építés és használat során. Mit kell még figyelembe vennünk?

## [Előadás utáni kvíz](https://ff-quizzes.netlify.app/en/ml/)

## Összefoglalás és önálló tanulás
 
Ebben a leckében megismerted a méltányosság és méltánytalanság gépi tanulásban rejlő alapfogalmait.
 
Nézd meg ezt a workshopot, hogy mélyebben belemerülj a témákba:

- A felelős MI után: Az alapelvek gyakorlati alkalmazása Besmira Nushi, Mehrnoosh Sameki és Amit Sharma előadásában

[![Felelős MI eszköztár: Egy nyílt forráskódú keretrendszer a felelős MI fejlesztéséhez](https://img.youtube.com/vi/tGgJCrA-MZU/0.jpg)](https://www.youtube.com/watch?v=tGgJCrA-MZU "RAI Toolbox: Egy nyílt forráskódú keretrendszer a felelős MI fejlesztéséhez")

> 🎥 Kattints a fenti képére egy videóért: RAI Toolbox: Egy nyílt forráskódú keretrendszer a felelős MI fejlesztéséhez, írta Besmira Nushi, Mehrnoosh Sameki és Amit Sharma

Olvasd el továbbá: 

- A Microsoft RAI forrásközpontja: [Felelős MI források – Microsoft AI](https://www.microsoft.com/ai/responsible-ai-resources?activetab=pivot1%3aprimaryr4) 

- A Microsoft FATE kutatócsoportja: [FATE: Méltányosság, elszámoltathatóság, átláthatóság és etika az MI-ben - Microsoft Research](https://www.microsoft.com/research/theme/fate/) 

RAI eszköztár: 

- [Felelős MI Eszköztár GitHub tár](https://github.com/microsoft/responsible-ai-toolbox)

Olvass az Azure Machine Learning eszközeiről a méltányosság biztosításához:

- [Azure Machine Learning](https://docs.microsoft.com/azure/machine-learning/concept-fairness-ml?WT.mc_id=academic-77952-leestott) 

## Feladat

[Fedezd fel a RAI Eszköztárat](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**Jogi nyilatkozat**:
Ez a dokumentum az AI fordítási szolgáltatás, a [Co-op Translator](https://github.com/Azure/co-op-translator) segítségével készült. Bár az pontosságra törekszünk, kérjük, vegye figyelembe, hogy az automatikus fordítások hibákat vagy pontatlanságokat tartalmazhatnak. Az eredeti dokumentum az anyanyelvén tekintendő hiteles forrásnak. Fontos információk esetén professzionális emberi fordítást javasolunk. Nem vállalunk felelősséget semmilyen félreértésért vagy téves értelmezésért, amely ebből a fordításból ered.
<!-- CO-OP TRANSLATOR DISCLAIMER END -->