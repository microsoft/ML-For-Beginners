# Felelős MI alapú gépi tanulási megoldások építése
 
![Összefoglaló a felelős MI-ről a gépi tanulásban egy vázlatrajzon](../../../../translated_images/hu/ml-fairness.ef296ebec6afc98a.webp)
> Vázlatrajz: [Tomomi Imura](https://www.twitter.com/girlie_mac)

## [Előadás előtti kvíz](https://ff-quizzes.netlify.app/en/ml/)
 
## Bevezetés

Ebben a tananyagban elkezded felfedezni, hogyan befolyásolja a gépi tanulás a mindennapi életünket. Már most is rendszerek és modellek vesznek részt napi döntéshozatali folyamatokban, például egészségügyi diagnózisokban, kölcsönkérelmek elbírálásában vagy csalás felismerésében. Ezért fontos, hogy ezek a modellek jól működjenek, és megbízható eredményeket nyújtsanak. Ahogyan bármely szoftveralkalmazás, az MI rendszerek is elvárásokat nem teljesíthetnek vagy nem kívánt kimenetet produkálhatnak. Ezért elengedhetetlen megérteni és megmagyarázni egy MI modell viselkedését. 

Képzeld el, mi történhet, ha az általad használt adatok egyes demográfiai csoportokat, például fajt, nemeket, politikai nézeteket, vallást hiányosan vagy aránytalanul képviselnek a modellek építése során. Mi történik, ha a modell kimenetét úgy értelmezik, hogy egyes demográfiai csoportok javára torzít? Mi ennek a következménye az alkalmazásra nézve? Ráadásul mi történik, ha a modell káros végkimenetelt produkál és árt az embereknek? Ki felel az MI rendszer viselkedéséért? Ezek néhány kérdés, amelyeket ebben a tananyagban fogunk megvizsgálni. 

Ebben az órában: 

- Növelni fogod az igazságosság fontosságával kapcsolatos tudatosságodat a gépi tanulásban, valamint az igazságossággal kapcsolatos ártalmak terén.
- Megismerkedsz a szokatlan esetek és kiugró értékek feltárásának gyakorlatával a megbízhatóság és biztonság érdekében.
- Megérted, hogy miért fontos mindenkit képessé tenni a befogadó rendszerek megtervezésével.
- Feltárjuk, milyen létfontosságú az adatok és emberek magánéletének és biztonságának védelme.
- Meglátod, hogy miért fontos átlátható („üveg doboz”) megközelítés az MI modellek viselkedésének magyarázatához.
- Tudatos leszel arról, hogy a felelősségvállalás mennyire elengedhetetlen a bizalom megteremtéséhez az MI rendszerekben.

## Előfeltétel

Előfeltételként kérjük, vedd fel a "Felelős MI elvei" tanulási útvonalat, és nézd meg az alábbi videót a témában:

Tudj meg többet a Felelős MI-ről ezen a [Tanulási Útvonalon](https://docs.microsoft.com/learn/modules/responsible-ai-principles/?WT.mc_id=academic-77952-leestott)

[![Microsoft megközelítése a Felelős MI-hez](https://img.youtube.com/vi/dnC8-uUZXSc/0.jpg)](https://youtu.be/dnC8-uUZXSc "Microsoft megközelítése a Felelős MI-hez")

> 🎥 Kattints a képre a videóért: Microsoft megközelítése a Felelős MI-hez

## Igazságosság

Az MI rendszereknek mindenkit igazságosan kell kezelniük, és kerülnie kell, hogy hasonló csoportokat eltérően érintsenek. Például amikor MI rendszerek orvosi kezelésben, kölcsönkérelmek elbírálásában vagy foglalkoztatásban adnak útmutatást, akkor hasonló tünetekkel, anyagi háttérrel vagy szakmai képesítéssel rendelkező embereknek ugyanazokat a javaslatokat kell adniuk. Mindannyiunkban öröklött elfogultságok vannak, amelyek döntéseinket és cselekedeteinket befolyásolják. Ezek az elfogultságok a gépi tanulási modellek képzéséhez használt adatokban is megjelenhetnek. Az ilyen torzítás néha akaratlanul is előfordulhat. Sokszor tudatosan nehéz megállapítani, mikor vezetünk be elfogultságot az adatba. 

**„Igazságtalanság”** negatív hatásokat vagy „károkat” jelent egy adott csoport számára, például faji, nemi, életkori vagy fogyatékossági státusz alapján definiált embereket. Az igazságossággal kapcsolatos legfőbb károk az alábbi kategóriákba sorolhatók: 

- **Elosztás**, ha például egy nem vagy etnikum előnyben részesül a másikkal szemben.
- **Szolgáltatás minősége**. Ha egy specifikus szcenárióra tanítasz adatot, de a valóság sokkal összetettebb, az gyenge teljesítményű szolgáltatáshoz vezet. Például egy kézmosószappan adagoló látszólag nem érzékeli a sötét bőrű embereket. [Forrás](https://gizmodo.com/why-cant-this-soap-dispenser-identify-dark-skin-1797931773)
- **Megvetés**. Valami vagy valaki igazságtalan kritikája és bántó címkézése. Például egy képosztályozó technológia hírhedten sötét bőrű emberek képeit majmokként címkézte.
- **Túl- vagy alulreprezentálás**. Az az elképzelés, hogy egy adott csoport nem látható egy bizonyos szakmában, és minden olyan szolgáltatás vagy funkció, amely ezt fenntartja, káros hatással van.
- **Sztenderdizálás**. Egy adott csoportot előre meghatározott tulajdonságokkal társítanak. Például egy angol és török nyelv közötti fordító rendszer pontatlanságokkal küzdhet a nemi sztereotípiák miatt.

![fordítás törökre](../../../../translated_images/hu/gender-bias-translate-en-tr.f185fd8822c2d437.webp)
> fordítás törökre

![fordítás vissza angolra](../../../../translated_images/hu/gender-bias-translate-tr-en.4eee7e3cecb8c70e.webp)
> fordítás vissza angolra

MI rendszerek tervezése és tesztelése során biztosítani kell, hogy az MI igazságos legyen, és ne programozzanak be elfogult vagy diszkriminatív döntéseket, amit az embereknek is tilos tenni. Az igazságosság garantálása MI-ben és gépi tanulásban összetett szociotechnikai kihívás marad. 

### Megbízhatóság és biztonság

A bizalom kiépítéséhez az MI rendszereknek megbízhatónak, biztonságosnak és következetesnek kell lenniük normál és váratlan körülmények között egyaránt. Fontos tudni, hogyan viselkednek az MI rendszerek különféle helyzetekben, különösen, ha kiugró esetekről van szó. MI megoldások építésekor nagy figyelmet kell fordítani arra, hogyan kezeljük azokat a sokféle helyzetet, amivel az MI találkozhat. Például az önvezető autónak mindenekelőtt az emberek biztonságát kell előtérbe helyeznie. Ezért az autót működtető MI-nek minden lehetséges helyzetet figyelembe kell vennie, például éjszaka, viharban vagy hóviharban, gyerekek átfutása az úton, háziállatok, útlezárások stb. Az, hogy egy MI rendszer mennyire képes megbízhatóan és biztonságosan kezelni szélsőséges helyzeteket, azt tükrözi, mennyire előrelátó volt az adatkutató vagy MI fejlesztő a rendszer tervezése vagy tesztelése során.  

> [🎥 Kattints ide a videóért: ](https://www.microsoft.com/videoplayer/embed/RE4vvIl)

### Befogadás

Az MI rendszereket úgy kell tervezni, hogy mindenkit bevonjanak és képessé tegyenek. MI rendszerek tervezése és megvalósítása során az adatkutatók és MI fejlesztők azonosítják és kezelik az esetleges akadályokat, amelyek véletlenül kizárhatnak embereket. Például a világon 1 milliárd fogyatékkal élő ember él. Az MI előrehaladtával ezek az emberek könnyebben férhetnek hozzá széles körű információkhoz és lehetőségekhez mindennapi életük során. Az akadályok kezelése lehetőséget teremt innovációra és olyan MI termékek fejlesztésére, amelyek jobb élményt nyújtanak mindenki számára. 

> [🎥 Kattints ide a videóért: befogadás az MI-ben](https://www.microsoft.com/videoplayer/embed/RE4vl9v)

### Biztonság és adatvédelem 

Az MI rendszereknek biztonságosnak kell lenniük, és tiszteletben kell tartaniuk az emberek magánéletét. Az emberek kevésbé bíznak meg olyan rendszerekben, amelyek veszélyeztetik magánéletüket, adataikat vagy életüket. Gépi tanulási modellek képzésekor jó eredmények eléréséhez adatokra támaszkodunk. Ennek során figyelembe kell venni az adatok származását és sértetlenségét. Például az adatokat a felhasználó szolgáltatta vagy nyilvánosan hozzáférhetőek? Ezután az adatfeldolgozás során létfontosságú olyan MI rendszereket fejleszteni, amelyek képesek védelmezni az érzékeny információkat és ellenállni a támadásoknak. Mivel az MI egyre elterjedtebb, a magánélet védelme és fontos személyes vagy üzleti információk védelme egyre kritikusabbá és összetettebbé válik. A magánélet és adatbiztonság különösen nagy figyelmet igényel az MI-nél, mert az adatokhoz való hozzáférés elengedhetetlen az MI rendszerek számára, hogy pontos és megalapozott előrejelzéseket és döntéseket hozzanak az emberekről. 

> [🎥 Kattints ide a videóért: biztonság az MI-ben](https://www.microsoft.com/videoplayer/embed/RE4voJF)

- Iparágként jelentős előrelépéseket értünk el az adatvédelem és biztonság terén, amit nagymértékben a GDPR (Általános Adatvédelmi Rendelet) szabályozásai ösztönöznek. 
- Azonban az MI rendszereknél el kell ismernünk a feszültséget a személyes adatok növekvő szükséglete és a magánélet védelme között.
- Ahogy az internet megjelenésekor a hálózatba kötött számítógépek révén, úgy napjainkban is nagymértékben megnőtt az MI-hez kapcsolódó biztonsági problémák száma. 
- Ugyanakkor azt is láttuk, hogy az MI-t a biztonság javítására is használják. Például a legtöbb modern vírusirtó szoftvert ma MI heurisztikák vezérlik. 
- Biztosítani kell, hogy az adatkutatási folyamataink harmonikusan illeszkedjenek a legújabb adatvédelmi és biztonsági gyakorlatokhoz. 


### Átláthatóság
Az MI rendszereknek érthetőknek kell lenniük. Az átláthatóság fontos része az MI rendszerek és azok elemeinek viselkedésének magyarázata. Az MI rendszerek jobb megértése megköveteli, hogy az érintettek megértsék, hogyan és miért működnek, hogy felismerjék a potenciális teljesítményproblémákat, biztonsági és adatvédelmi aggályokat, elfogultságokat, kizáró gyakorlatokat vagy nem kívánt következményeket. Úgy véljük, hogy azoknak, akik MI rendszereket használnak, őszintén és nyíltan kell beszámolniuk arról, mikor, miért és hogyan döntöttek a bevetésükről, valamint a rendszerek korlátairól is. Például, ha egy bank egy MI rendszert használ fogyasztói hitel döntések támogatására, fontos megvizsgálni az eredményeket és megérteni, mely adatok befolyásolják a rendszer javaslatait. A kormányok elkezdték szabályozni az MI-t az iparágakban, ezért az adatkutatóknak és szervezeteknek meg kell tudniuk magyarázni, hogy egy MI rendszer megfelel-e a szabályozási előírásoknak, különösen, ha nem kívánt eredmény született. 

> [🎥 Kattints ide a videóért: átláthatóság az MI-ben](https://www.microsoft.com/videoplayer/embed/RE4voJF)

- Mivel az MI rendszerek annyira összetettek, nehéz megérteni működésüket és értelmezni az eredményeket. 
- Ez a megértés hiánya befolyásolja, hogy hogyan kezelik, működtetik és dokumentálják ezeket a rendszereket. 
- Ennél is fontosabb, hogy ez a megértés hiánya befolyásolja az ezeket a rendszereket használó döntéseket.

### Felelősségre vonhatóság 
 
Azoknak, akik MI rendszereket terveznek és alkalmaznak, felelősséget kell vállalniuk a rendszerek működéséért. A felelősség különösen fontos érzékeny technológiák, mint az arcfelismerés esetén. Az utóbbi időben nőtt az érdeklődés az arcfelismerő technológia iránt, különösen a rendvédelmi szervezetek részéről, akik ezt a technológiát olyan célokra látják hasznosnak, mint eltűnt gyermekek felkutatása. Ugyanakkor ezek a technológiák potenciálisan felhasználhatók a kormányok által az állampolgárok alapvető szabadságainak veszélyeztetésére, például az adott személyek folyamatos megfigyelésének engedélyezésére. Ezért az adatkutatóknak és szervezeteknek felelősséget kell vállalniuk az MI rendszerük hatásaiért az egyedekre vagy a társadalomra nézve.

[![Vezető MI kutató figyelmeztet az arcfelismerés által okozott tömeges megfigyelésre](../../../../translated_images/hu/accountability.41d8c0f4b85b6231.webp)](https://www.youtube.com/watch?v=Wldt8P5V6D0 "Microsoft megközelítése a Felelős MI-hez")

> 🎥 Kattints a képre a videóért: Figyelmeztetések az arcfelismerés által okozott tömeges megfigyelésre

Végső soron egyik legnagyobb kérdés a mi generációnknak, az elsőnek, amely az MI-t bevezeti a társadalomba, az, hogyan biztosítsuk, hogy a számítógépek felelősséggel tartozzanak az emberek felé, és hogyan biztosítsuk, hogy a számítógépeket tervező emberek mindenkihez felelősséggel tartozzanak.

## Hatásértékelés

Mielőtt gépi tanulási modellt képeznénk, fontos hatásértékelést végezni az MI rendszer céljának megértéséhez; annak tervezett használatához; hogy hol fogják alkalmazni; és kik fognak interakcióba lépni a rendszerrel. Ezek a jelöltek vagy tesztelők számára hasznosak a rendszer értékelésekor, hogy tudják, milyen tényezőket vegyenek figyelembe a potenciális kockázatok és várható következmények azonosításakor.

Az alábbi területekre kell fókuszálni a hatásértékelés során:

* **Káros hatás az egyénekre**. Fontos tisztában lenni minden korlátozással vagy előírással, a támogatott használaton kívüli alkalmazással vagy ismert korlátozásokkal, amelyek akadályozhatják a rendszer működését annak érdekében, hogy ne használják olyan módon, amely ártalmat okozhat az egyéneknek.
* **Adatigények**. Az, hogy megértsük, hogyan és hol használja a rendszer az adatokat, lehetővé teszi a felülvizsgálók számára, hogy felmérjék az esetleges adatvédelmi szabályozásokat, például GDPR vagy HIPAA előírásokat. Emellett vizsgálják meg, van-e elegendő adatforrás és mennyiség a képzéshez.
* **Hatás összefoglalása**. Gyűjtsünk listát a potenciális károkról, amelyek a rendszer használatából eredhetnek. A gépi tanulás életciklusa során vizsgáljuk felül, hogy a felmerült problémákat kezelik vagy mérséklik-e.
* **Alkalmazható célok** a hat fő elv mindegyikére. Értékeljük, hogy az egyes elvek céljai teljesülnek-e, és hogy vannak-e hiányosságok.


## Hibakeresés felelős MI-vel  

Hasonlóan a szoftveralkalmazások hibakereséséhez, az MI rendszer hibakeresése is szükséges a rendszerben lévő problémák azonosításához és megoldásához. Számos tényező befolyásolhatja, hogy egy modell nem teljesít elvárások szerint vagy nem felel meg a felelős MI elveknek. A hagyományos modell teljesítménymutatók többsége egy modell teljesítményének mennyiségi összegzése, amely nem elegendő annak elemzésére, hogyan sérti a modell a felelős MI elveit. Továbbá, a gépi tanulási modell egy fekete doboz, amely megnehezíti megérteni, mi befolyásolja a kimenetét, vagy magyarázatot adni, ha hibázik. A tanfolyam későbbi részében megmutatjuk, hogyan használható a Felelős MI műszerfal az MI rendszerek hibakeresésére. A műszerfal átfogó eszközt nyújt adatkutatóknak és MI fejlesztőknek az alábbiak elvégzéséhez:

* **Hibaanalízis**. A modell hibák eloszlásának azonosítása, ami befolyásolhatja a rendszer igazságosságát vagy megbízhatóságát.
* **Modelláttekintés**. Az eltérések felfedezése a modell teljesítményében az adatcsoportok között.
* **Adatok elemzése**. Az adateloszlás megértése és bármilyen potenciális elfogultság azonosítása az adatokban, amely igazságossági, befogadási és megbízhatósági problémákat okozhat.
* **Modell értelmezhetősége**. Megérteni, mi befolyásolja a modell előrejelzéseit, ami fontos az átláthatóság és felelősség szempontjából.


## 🚀 Kihívás 
 
Az ártalmak kialakulásának megakadályozása érdekében: 

- változatos háttérrel és nézőpontokkal rendelkező emberek dolgozzanak a rendszereken 
- olyan adatállományokba fektessünk be, amelyek tükrözik társadalmunk sokszínűségét 
- jobb módszereket dolgozzunk ki a gépi tanulás életciklusa során a felelős MI hibák felismerésére és javítására 

Gondolkodj el valós életbeli helyzeteken, ahol egy modell megbízhatatlansága egyértelműen megjelenik a modellépítés és felhasználás során. Mit kell még figyelembe vennünk? 

## [Előadás utáni kvíz](https://ff-quizzes.netlify.app/en/ml/)

## Összefoglaló & Önálló tanulás 
 

Ebben a leckében megismerted a gépi tanulásban a méltányosság és igazságtalanság alapfogalmait.  
 
Nézd meg ezt a workshopot, hogy mélyebben elmerülj a témakörökben: 

- Felelős MI nyomában: Elvek a gyakorlatban Besmira Nushi, Mehrnoosh Sameki és Amit Sharma előadásában

[![Felelős MI Eszköztár: Nyílt forráskódú keretrendszer felelős MI építéséhez](https://img.youtube.com/vi/tGgJCrA-MZU/0.jpg)](https://www.youtube.com/watch?v=tGgJCrA-MZU "RAI Toolbox: An open-source framework for building responsible AI")

> 🎥 Kattints a fenti képre a videóhoz: RAI Toolbox: Nyílt forráskódú keretrendszer felelős MI építéséhez Besmira Nushi, Mehrnoosh Sameki és Amit Sharma előadásában

Olvasd el még: 

- A Microsoft RAI forrásközpontja: [Responsible AI Resources – Microsoft AI](https://www.microsoft.com/ai/responsible-ai-resources?activetab=pivot1%3aprimaryr4) 

- A Microsoft FATE kutatócsoportja: [FATE: Fairness, Accountability, Transparency, and Ethics in AI - Microsoft Research](https://www.microsoft.com/research/theme/fate/) 

RAI Eszköztár: 

- [Responsible AI Toolbox GitHub tárhely](https://github.com/microsoft/responsible-ai-toolbox)

Ismerd meg az Azure Machine Learning eszközeit a méltányosság biztosításához:

- [Azure Machine Learning](https://docs.microsoft.com/azure/machine-learning/concept-fairness-ml?WT.mc_id=academic-77952-leestott) 

## Feladat

[Böngéssz a RAI Eszköztárban](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**Jogi nyilatkozat**:
Ez a dokumentum az AI fordítási szolgáltatás, a [Co-op Translator](https://github.com/Azure/co-op-translator) segítségével készült. Bár az pontosságra törekszünk, kérjük, vegye figyelembe, hogy az automatikus fordítások hibákat vagy pontatlanságokat tartalmazhatnak. Az eredeti dokumentum az anyanyelvén tekintendő hiteles forrásnak. Fontos információk esetén professzionális emberi fordítást javasolunk. Nem vállalunk felelősséget semmilyen félreértésért vagy téves értelmezésért, amely ebből a fordításból ered.
<!-- CO-OP TRANSLATOR DISCLAIMER END -->