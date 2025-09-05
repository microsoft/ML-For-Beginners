<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "83320d6b6994909e35d830cebf214039",
  "translation_date": "2025-09-05T15:50:14+00:00",
  "source_file": "9-Real-World/1-Applications/README.md",
  "language_code": "hu"
}
-->
# Utóirat: Gépi tanulás a való világban

![A gépi tanulás összefoglalása a való világban egy sketchnote-ban](../../../../sketchnotes/ml-realworld.png)
> Sketchnote készítette: [Tomomi Imura](https://www.twitter.com/girlie_mac)

Ebben a tananyagban számos módszert tanultál az adatok előkészítésére és gépi tanulási modellek létrehozására. Klasszikus regressziós, klaszterezési, osztályozási, természetes nyelvfeldolgozási és idősoros modellek sorozatát építetted fel. Gratulálunk! Most talán azon gondolkodsz, hogy mindez mire jó... milyen valós alkalmazásai vannak ezeknek a modelleknek?

Bár az iparban nagy érdeklődést váltott ki az AI, amely általában mély tanulást használ, a klasszikus gépi tanulási modelleknek továbbra is értékes alkalmazásai vannak. Lehet, hogy már ma is használod ezeket az alkalmazásokat! Ebben a leckében megvizsgáljuk, hogyan használja nyolc különböző iparág és szakterület ezeket a modelleket, hogy alkalmazásaik teljesítményét, megbízhatóságát, intelligenciáját és értékét növeljék a felhasználók számára.

## [Előadás előtti kvíz](https://ff-quizzes.netlify.app/en/ml/)

## 💰 Pénzügy

A pénzügyi szektor számos lehetőséget kínál a gépi tanulás számára. Sok probléma ebben a területen modellezhető és megoldható ML segítségével.

### Hitelkártya-csalás észlelése

Korábban tanultunk a [k-means klaszterezésről](../../5-Clustering/2-K-Means/README.md), de hogyan használható ez a hitelkártya-csalásokkal kapcsolatos problémák megoldására?

A k-means klaszterezés hasznos a hitelkártya-csalás észlelésének egyik technikájában, amelyet **szélsőértékek észlelésének** neveznek. A szélsőértékek, vagyis az adathalmaz megfigyeléseiben tapasztalható eltérések, megmutathatják, hogy egy hitelkártyát normál módon használnak-e, vagy valami szokatlan történik. Az alábbi tanulmány szerint a hitelkártya-adatokat k-means klaszterezési algoritmus segítségével lehet rendezni, és minden tranzakciót egy klaszterhez rendelni annak alapján, hogy mennyire tűnik szélsőértéknek. Ezután ki lehet értékelni a legkockázatosabb klasztereket, hogy megállapítsuk, csalásról vagy legitim tranzakcióról van-e szó.
[Hivatkozás](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.680.1195&rep=rep1&type=pdf)

### Vagyonkezelés

A vagyonkezelés során egy személy vagy cég kezeli ügyfelei befektetéseit. Feladatuk a vagyon hosszú távú fenntartása és növelése, ezért elengedhetetlen, hogy jól teljesítő befektetéseket válasszanak.

Egy befektetés teljesítményének értékelésére az egyik módszer a statisztikai regresszió. A [lineáris regresszió](../../2-Regression/1-Tools/README.md) értékes eszköz annak megértéséhez, hogy egy alap hogyan teljesít egy benchmarkhoz képest. Azt is megállapíthatjuk, hogy a regresszió eredményei statisztikailag szignifikánsak-e, vagy hogy mennyire befolyásolnák az ügyfél befektetéseit. Az elemzést tovább bővítheted többszörös regresszióval, ahol további kockázati tényezőket is figyelembe lehet venni. Egy konkrét alap teljesítményének értékelésére vonatkozó példát az alábbi tanulmányban találhatsz.
[Hivatkozás](http://www.brightwoodventures.com/evaluating-fund-performance-using-regression/)

## 🎓 Oktatás

Az oktatási szektor szintén nagyon érdekes terület, ahol az ML alkalmazható. Érdekes problémák merülhetnek fel, például a csalás észlelése teszteken vagy esszéken, illetve az elfogultság kezelése, akár szándékos, akár nem, a javítási folyamat során.

### Diákok viselkedésének előrejelzése

A [Coursera](https://coursera.com), egy online nyílt kurzusokat kínáló szolgáltató, nagyszerű technológiai blogot vezet, ahol számos mérnöki döntést megvitatnak. Ebben az esettanulmányban regressziós vonalat rajzoltak, hogy megvizsgálják, van-e összefüggés az alacsony NPS (Net Promoter Score) értékelés és a kurzus megtartása vagy lemorzsolódása között.
[Hivatkozás](https://medium.com/coursera-engineering/controlled-regression-quantifying-the-impact-of-course-quality-on-learner-retention-31f956bd592a)

### Elfogultság csökkentése

A [Grammarly](https://grammarly.com), egy írássegítő, amely helyesírási és nyelvtani hibákat ellenőriz, kifinomult [természetes nyelvfeldolgozási rendszereket](../../6-NLP/README.md) használ termékeiben. Érdekes esettanulmányt tettek közzé technológiai blogjukban arról, hogyan kezelték a nemi elfogultságot a gépi tanulásban, amelyről az [igazságosságról szóló bevezető leckénkben](../../1-Introduction/3-fairness/README.md) tanultál.
[Hivatkozás](https://www.grammarly.com/blog/engineering/mitigating-gender-bias-in-autocorrect/)

## 👜 Kereskedelem

A kereskedelmi szektor jelentős előnyöket élvezhet az ML használatából, kezdve a jobb vásárlói élmény megteremtésétől az optimális készletkezelésig.

### Vásárlói élmény személyre szabása

A Wayfair, egy otthoni termékeket, például bútorokat árusító cég, kiemelten fontosnak tartja, hogy segítsen vásárlóinak megtalálni az ízlésüknek és igényeiknek megfelelő termékeket. Ebben a cikkben a cég mérnökei leírják, hogyan használják az ML-t és az NLP-t, hogy "a megfelelő eredményeket kínálják a vásárlóknak". Különösen a Query Intent Engine-t építették ki, amely entitáskinyerést, osztályozó tanítást, eszköz- és véleménykinyerést, valamint érzelemcímkézést alkalmaz az ügyfélértékelésekben. Ez egy klasszikus példa arra, hogyan működik az NLP az online kereskedelemben.
[Hivatkozás](https://www.aboutwayfair.com/tech-innovation/how-we-use-machine-learning-and-natural-language-processing-to-empower-search)

### Készletkezelés

Az innovatív, rugalmas cégek, mint például a [StitchFix](https://stitchfix.com), egy dobozos szolgáltatás, amely ruhákat küld a fogyasztóknak, erősen támaszkodnak az ML-re az ajánlások és készletkezelés terén. Stíluscsapataik együttműködnek a merchandising csapataikkal: "egy adatkutatónk egy genetikus algoritmussal kísérletezett, és alkalmazta azt a ruházatra, hogy megjósolja, milyen sikeres ruhadarab lehet, amely ma még nem létezik. Ezt bemutattuk a merchandising csapatnak, és most már eszközként használhatják."
[Hivatkozás](https://www.zdnet.com/article/how-stitch-fix-uses-machine-learning-to-master-the-science-of-styling/)

## 🏥 Egészségügy

Az egészségügyi szektor az ML-t kutatási feladatok optimalizálására, valamint logisztikai problémák, például a betegek visszafogadása vagy a betegségek terjedésének megállítása érdekében használhatja.

### Klinikai vizsgálatok kezelése

A klinikai vizsgálatokban a toxicitás komoly aggodalomra ad okot a gyógyszergyártók számára. Mennyi toxicitás tolerálható? Ebben a tanulmányban különböző klinikai vizsgálati módszerek elemzése új megközelítést eredményezett a klinikai vizsgálati eredmények esélyeinek előrejelzésére. Különösen a random forest segítségével hoztak létre egy [osztályozót](../../4-Classification/README.md), amely képes megkülönböztetni a gyógyszercsoportokat.
[Hivatkozás](https://www.sciencedirect.com/science/article/pii/S2451945616302914)

### Kórházi visszafogadás kezelése

A kórházi ellátás költséges, különösen akkor, ha a betegeket vissza kell fogadni. Ez a tanulmány egy olyan céget tárgyal, amely ML-t használ a visszafogadás potenciáljának előrejelzésére [klaszterezési](../../5-Clustering/README.md) algoritmusok segítségével. Ezek a klaszterek segítenek az elemzőknek "felfedezni olyan visszafogadási csoportokat, amelyeknek közös oka lehet".
[Hivatkozás](https://healthmanagement.org/c/healthmanagement/issuearticle/hospital-readmissions-and-machine-learning)

### Betegségkezelés

A közelmúltbeli járvány rávilágított arra, hogy a gépi tanulás hogyan segíthet a betegségek terjedésének megállításában. Ebben a cikkben felismerheted az ARIMA, logisztikai görbék, lineáris regresszió és SARIMA használatát. "Ez a munka arra irányul, hogy kiszámítsa a vírus terjedési sebességét, és így előre jelezze a haláleseteket, gyógyulásokat és megerősített eseteket, hogy jobban felkészülhessünk és túlélhessünk."
[Hivatkozás](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7979218/)

## 🌲 Ökológia és zöld technológia

A természet és ökológia számos érzékeny rendszert foglal magában, ahol az állatok és a természet közötti kölcsönhatás kerül előtérbe. Fontos, hogy ezeket a rendszereket pontosan mérjük, és megfelelően cselekedjünk, ha valami történik, például erdőtűz vagy az állatpopuláció csökkenése.

### Erdőgazdálkodás

Korábbi leckékben tanultál a [megerősítéses tanulásról](../../8-Reinforcement/README.md). Ez nagyon hasznos lehet, amikor a természetben előforduló mintázatokat próbáljuk megjósolni. Különösen hasznos lehet ökológiai problémák, például erdőtüzek és invazív fajok terjedésének nyomon követésére. Kanadában egy kutatócsoport megerősítéses tanulást használt erdőtűz dinamikai modellek építésére műholdképek alapján. Egy innovatív "térbeli terjedési folyamat (SSP)" segítségével az erdőtüzet "ügynökként képzelték el a táj bármely cellájában". "Azok az akciók, amelyeket a tűz bármely helyszínen bármely időpontban megtehet, magukban foglalják az északra, délre, keletre vagy nyugatra való terjedést, vagy a nem terjedést."

Ez a megközelítés megfordítja a szokásos RL beállítást, mivel a megfelelő Markov döntési folyamat (MDP) dinamikája ismert funkció az azonnali erdőtűz terjedésére. Olvass többet az algoritmusokról az alábbi linken.
[Hivatkozás](https://www.frontiersin.org/articles/10.3389/fict.2018.00006/full)

### Állatok mozgásának érzékelése

Bár a mély tanulás forradalmat hozott az állatok mozgásának vizuális nyomon követésében (építhetsz saját [jegesmedve nyomkövetőt](https://docs.microsoft.com/learn/modules/build-ml-model-with-azure-stream-analytics/?WT.mc_id=academic-77952-leestott) itt), a klasszikus ML-nek még mindig van helye ebben a feladatban.

Az állatok mozgásának nyomon követésére szolgáló érzékelők és az IoT az ilyen típusú vizuális feldolgozást használják, de az alapvető ML technikák hasznosak az adatok előfeldolgozásában. Például ebben a tanulmányban a juhok testtartását különböző osztályozó algoritmusok segítségével figyelték meg és elemezték. Felismerheted az ROC görbét a 335. oldalon.
[Hivatkozás](https://druckhaus-hofmann.de/gallery/31-wj-feb-2020.pdf)

### ⚡️ Energia menedzsment

Az [idősoros előrejelzésről](../../7-TimeSeries/README.md) szóló leckéinkben megemlítettük az okos parkolóórák koncepcióját, amelyek segítenek bevételt generálni egy város számára a kereslet és kínálat megértése alapján. Ez a cikk részletesen tárgyalja, hogyan kombinálták a klaszterezést, regressziót és idősoros előrejelzést, hogy előre jelezzék Írország jövőbeli energiafelhasználását okos mérés alapján.
[Hivatkozás](https://www-cdn.knime.com/sites/default/files/inline-images/knime_bigdata_energy_timeseries_whitepaper.pdf)

## 💼 Biztosítás

A biztosítási szektor egy másik terület, amely az ML-t használja életképes pénzügyi és aktuáriusi modellek felépítésére és optimalizálására.

### Volatilitás kezelése

A MetLife, egy életbiztosítási szolgáltató, nyíltan beszél arról, hogyan elemzik és csökkentik a volatilitást pénzügyi modelljeikben. Ebben a cikkben bináris és ordinális osztályozási vizualizációkat találhatsz. Emellett előrejelzési vizualizációkat is felfedezhetsz.
[Hivatkozás](https://investments.metlife.com/content/dam/metlifecom/us/investments/insights/research-topics/macro-strategy/pdf/MetLifeInvestmentManagement_MachineLearnedRanking_070920.pdf)

## 🎨 Művészetek, kultúra és irodalom

A művészetekben, például az újságírásban, számos érdekes probléma merül fel. A hamis hírek észlelése komoly probléma, mivel bizonyítottan befolyásolja az emberek véleményét, sőt demokráciákat is megingathat. A múzeumok is profitálhatnak az ML használatából, például az artefaktumok közötti kapcsolatok megtalálásában vagy az erőforrások tervezésében.

### Hamis hírek észlelése

A hamis hírek észlelése macska-egér játékká vált a mai médiában. Ebben a cikkben a kutatók azt javasolják, hogy több ML technikát kombináló rendszert teszteljenek, és a legjobb modellt alkalmazzák: "Ez a rendszer természetes nyelvfeldolgozáson alapul, hogy adatokat nyerjen ki, majd ezek az adatok gépi tanulási osztályozók, például Naive Bayes, Support Vector Machine (SVM), Random Forest (RF), Stochastic Gradient Descent (SGD) és Logistic Regression (LR) tanítására szolgálnak."
[Hivatkozás](https://www.irjet.net/archives/V7/i6/IRJET-V7I6688.pdf)

Ez a cikk bemutatja, hogyan lehet különböző ML területeket kombinálni, hogy érdekes eredményeket érjünk el, amelyek segíthetnek megállítani a hamis hírek terjedését és valódi károkat okozását; ebben az esetben az indíték a COVID-kezelésekről szóló pletykák terjedése volt, amelyek tömeges erőszakot váltottak ki.

### Múzeumi ML

A múzeumok az AI forradalmának küszö
## [Előadás utáni kvíz](https://ff-quizzes.netlify.app/en/ml/)

## Áttekintés és önálló tanulás

A Wayfair adatkutató csapata több érdekes videót készített arról, hogyan használják a gépi tanulást a vállalatuknál. Érdemes [megnézni](https://www.youtube.com/channel/UCe2PjkQXqOuwkW1gw6Ameuw/videos)!

## Feladat

[ML kincsvadászat](assignment.md)

---

**Felelősség kizárása**:  
Ez a dokumentum az AI fordítási szolgáltatás, a [Co-op Translator](https://github.com/Azure/co-op-translator) segítségével lett lefordítva. Bár törekszünk a pontosságra, kérjük, vegye figyelembe, hogy az automatikus fordítások hibákat vagy pontatlanságokat tartalmazhatnak. Az eredeti dokumentum az eredeti nyelvén tekintendő hiteles forrásnak. Kritikus információk esetén javasolt professzionális emberi fordítást igénybe venni. Nem vállalunk felelősséget semmilyen félreértésért vagy téves értelmezésért, amely a fordítás használatából eredhet.