<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "0a6f4476a4f3934a4aa47c1bf47158bc",
  "translation_date": "2026-01-16T14:49:09+00:00",
  "source_file": "README.md",
  "language_code": "hu"
}
-->
[![GitHub license](https://img.shields.io/github/license/microsoft/ML-For-Beginners.svg)](https://github.com/microsoft/ML-For-Beginners/blob/master/LICENSE)
[![GitHub contributors](https://img.shields.io/github/contributors/microsoft/ML-For-Beginners.svg)](https://GitHub.com/microsoft/ML-For-Beginners/graphs/contributors/)
[![GitHub issues](https://img.shields.io/github/issues/microsoft/ML-For-Beginners.svg)](https://GitHub.com/microsoft/ML-For-Beginners/issues/)
[![GitHub pull-requests](https://img.shields.io/github/issues-pr/microsoft/ML-For-Beginners.svg)](https://GitHub.com/microsoft/ML-For-Beginners/pulls/)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](http://makeapullrequest.com)

[![GitHub watchers](https://img.shields.io/github/watchers/microsoft/ML-For-Beginners.svg?style=social&label=Watch)](https://GitHub.com/microsoft/ML-For-Beginners/watchers/)
[![GitHub forks](https://img.shields.io/github/forks/microsoft/ML-For-Beginners.svg?style=social&label=Fork)](https://GitHub.com/microsoft/ML-For-Beginners/network/)
[![GitHub stars](https://img.shields.io/github/stars/microsoft/ML-For-Beginners.svg?style=social&label=Star)](https://GitHub.com/microsoft/ML-For-Beginners/stargazers/)

### 🌐 Többnyelvű támogatás

#### GitHub Action segítségével (automatikus és mindig naprakész)

<!-- CO-OP TRANSLATOR LANGUAGES TABLE START -->
[Arab](../ar/README.md) | [Bengáli](../bn/README.md) | [Bolgár](../bg/README.md) | [Burmai (Myanmar)](../my/README.md) | [Kínai (egyszerűsített)](../zh/README.md) | [Kínai (hagyományos, Hong Kong)](../hk/README.md) | [Kínai (hagyományos, Makaó)](../mo/README.md) | [Kínai (hagyományos, Tajvan)](../tw/README.md) | [Horvát](../hr/README.md) | [Cseh](../cs/README.md) | [Dán](../da/README.md) | [Holland](../nl/README.md) | [Észt](../et/README.md) | [Finn](../fi/README.md) | [Francia](../fr/README.md) | [Német](../de/README.md) | [Görög](../el/README.md) | [Héber](../he/README.md) | [Hindi](../hi/README.md) | [Magyar](./README.md) | [Indonéz](../id/README.md) | [Olasz](../it/README.md) | [Japán](../ja/README.md) | [Kannada](../kn/README.md) | [Koreai](../ko/README.md) | [Litván](../lt/README.md) | [Maláj](../ms/README.md) | [Malayalam](../ml/README.md) | [Marathi](../mr/README.md) | [Nepáli](../ne/README.md) | [Nigériai pidgin](../pcm/README.md) | [Norvég](../no/README.md) | [Perzsa (Fárszi)](../fa/README.md) | [Lengyel](../pl/README.md) | [Portugál (Brazília)](../br/README.md) | [Portugál (Portugália)](../pt/README.md) | [Pandzsábi (Gurmuki)](../pa/README.md) | [Román](../ro/README.md) | [Orosz](../ru/README.md) | [Szerb (cirill)](../sr/README.md) | [Szlovák](../sk/README.md) | [Szlovén](../sl/README.md) | [Spanyol](../es/README.md) | [Szuahéli](../sw/README.md) | [Svéd](../sv/README.md) | [Tagalog (Filippínó)](../tl/README.md) | [Tamil](../ta/README.md) | [Telugu](../te/README.md) | [Thai](../th/README.md) | [Török](../tr/README.md) | [Ukrán](../uk/README.md) | [Urdu](../ur/README.md) | [Vietnami](../vi/README.md)

> **Inkább helyben klónoznád?**

> Ez a tárház több mint 50 nyelvi fordítást tartalmaz, amelyek jelentősen megnövelik a letöltési méretet. Fordítások nélküli klónozáshoz használja a sparse checkout-ot:
> ```bash
> git clone --filter=blob:none --sparse https://github.com/microsoft/ML-For-Beginners.git
> cd ML-For-Beginners
> git sparse-checkout set --no-cone '/*' '!translations' '!translated_images'
> ```
> Így minden megvan, ami a tanfolyam elvégzéséhez szükséges, sokkal gyorsabb letöltéssel.
<!-- CO-OP TRANSLATOR LANGUAGES TABLE END -->

#### Csatlakozzon közösségünkhöz

[![Microsoft Foundry Discord](https://dcbadge.limes.pink/api/server/nTYy5BXMWG)](https://discord.gg/nTYy5BXMWG)

Folyamatban van egy Discord "Tanulj az MI-vel" sorozatunk, további információkért és részvételért látogasson el a [Learn with AI Series](https://aka.ms/learnwithai/discord) oldalra 2025. szeptember 18. és 30. között. Tippeket és trükköket kaphat a GitHub Copilot használatához az Adattudományban.

![Tanulj az MI-vel sorozat](../../../../translated_images/hu/3.9b58fd8d6c373c20.webp)

# Gépi tanulás kezdőknek - Tananyag

> 🌍 Utazz velünk a világ körül, miközben a gépi tanulást a világ kultúrái által fedezzük fel 🌍

A Microsoft felhős képviselői örömmel kínálnak egy 12 hetes, 26 leckéből álló tananyagot, amely a **gépi tanulásról** szól. Ebben a tananyagban azt tanulod meg, amit néha **klasszikus gépi tanulásnak** neveznek, elsősorban a Scikit-learn könyvtárat használva, elkerülve a mélytanulást, amelyet a [MI kezdőknek tananyagunkban](https://aka.ms/ai4beginners) tárgyalunk. Párosítsd ezeket a leckéket a ['Adattudomány kezdőknek' tananyaggal](https://aka.ms/ds4beginners) is!

Utazz velünk a világ körül, miközben ezeket a klasszikus technikákat a világ sok területéről származó adatokra alkalmazzuk. Minden lecke tartalmaz elő- és utóleckés kvízeket, írásos útmutatót a lecke elvégzéséhez, megoldást, feladatot, és még többet. Projekt-alapú oktatásunk révén tanulhatsz miközben építesz, ami bevált módszer az új készségek tartós elsajátítására.

**✍️ Hálás köszönet szerzőinknek:** Jen Looper, Stephen Howell, Francesca Lazzeri, Tomomi Imura, Cassie Breviu, Dmitry Soshnikov, Chris Noring, Anirban Mukherjee, Ornella Altunyan, Ruth Yakubu és Amy Boyd

**🎨 Köszönet illusztrátorainknak is:** Tomomi Imura, Dasani Madipalli és Jen Looper

**🙏 Külön köszönet a Microsoft Student Ambassador szerzőknek, lektoroknak és tartalomközreműködőknek, különösen Rishit Dagli, Muhammad Sakib Khan Inan, Rohan Raj, Alexandru Petrescu, Abhishek Jaiswal, Nawrin Tabassum, Ioan Samuila és Snigdha Agarwal**

**🤩 További hálánk Microsoft Student Ambassadors Eric Wanjau, Jasleen Sondhi és Vidushi Gupta számára az R leckéinkért!**

# Kezdés

Kövesd az alábbi lépéseket:
1. **Furkálja le a tárházat**: Kattints a jobb felső sarokban a "Fork" gombra.
2. **Klónozd a tárházat**: `git clone https://github.com/microsoft/ML-For-Beginners.git`

> [az összes további erőforrást megtalálod a Microsoft Learn gyűjteményünkben](https://learn.microsoft.com/en-us/collections/qrqzamz1nn2wx3?WT.mc_id=academic-77952-bethanycheum)

> 🔧 **Segítségre van szükséged?** Nézd meg a [Hibakeresési útmutatónkat](TROUBLESHOOTING.md) az általános telepítési, beállítási és lecke futtatási problémák megoldásához.


**[Hallgatók](https://aka.ms/student-page)**, ahhoz, hogy használjátok ezt a tananyagot, forkold a teljes repót a saját GitHub fiókodra és végezd el a feladatokat egyénileg vagy csoportban:

- Kezdd egy előadás előtti kvízzel.
- Olvasd el az előadást és végezd el a tevékenységeket, minden tudásellenőrzésnél tarts szünetet és gondolkodj el.
- Próbáld meg a projekteket a leckék megértésével létrehozni, ne csak a megoldás-kód futtatásával; azonban ez a kód elérhető a /solution mappákban minden projekt-orientált leckében.
- Tedd le az utóleckés kvízt.
- Teljesítsd a kihívást.
- Végezd el a feladatot.
- Miután egy leckecsoportot befejeztél, látogass el a [Vita fórumra](https://github.com/microsoft/ML-For-Beginners/discussions) és tanulj hangosan azáltal, hogy kitöltöd a megfelelő PAT értékelőt. A 'PAT' egy haladás-értékelő eszköz, amely egy értékelőrendszer, amit kitöltesz, hogy előbbre juss a tanulásban. Más PAT-ekre is reagálhatsz, hogy együtt tanuljunk.

> További tanulmányozásra azt ajánljuk, hogy kövesd ezeket a [Microsoft Learn](https://docs.microsoft.com/en-us/users/jenlooper-2911/collections/k7o7tg1gp306q4?WT.mc_id=academic-77952-leestott) modulokat és tanulási útvonalakat.

**Tanárként**, [találsz néhány javaslatot](for-teachers.md) arra, hogyan használd ezt a tananyagot.

---

## Videó bemutatók

A leckék egy része rövid formátumú videóként is elérhető. Mindezeket megtalálod az leckékben inline, vagy a [ML for Beginners lejátszási listán a Microsoft Developer YouTube csatornán](https://aka.ms/ml-beginners-videos) a lenti képre kattintva.

[![ML for beginners banner](../../../../translated_images/hu/ml-for-beginners-video-banner.63f694a100034bc6.webp)](https://aka.ms/ml-beginners-videos)

---

## Ismerd meg a csapatot

[![Promo video](../../images/ml.gif)](https://youtu.be/Tj1XWrDSYJU)

**Gif készítője** [Mohit Jaisal](https://linkedin.com/in/mohitjaisal)

> 🎥 Kattints a fenti képre egy videóért a projektről és az alkotóiról!

---

## Oktatási elvek

Két pedagógiai elvet választottunk a tananyag elkészítésekor: azt, hogy gyakorlatorientált, **projekt-alapú** legyen, és hogy tartalmazzon **gyakori kvízeket**. Ezen felül a tananyagnak van egy közös **téma** is, hogy kohéziót adjon neki.

Azáltal, hogy a tartalom a projektekhez kapcsolódik, az egész folyamat élvezetesebb a tanulók számára, és a fogalmak megtartása is fokozódik. Ezenkívül egy alacsony tétű kvíz az óra előtt előkészíti a hallgató szándékát a téma megtanulására, míg az óra utáni második kvíz további megtartást biztosít. Ez a tananyag rugalmas és szórakoztató formában készült, egyben vagy részleteiben is végezhető. A projektek kicsiben kezdődnek, és egyre összetettebbé válnak a 12 hetes ciklus végére. A tananyag egy utószót is tartalmaz a gépi tanulás valódi alkalmazásairól, amely extra pontként vagy megbeszélés alapjaként használható.

> Tekintsd meg az [Etikai kódexünket](CODE_OF_CONDUCT.md), a [Közreműködés](CONTRIBUTING.md), [Fordítási](TRANSLATIONS.md) és [Hibakeresési](TROUBLESHOOTING.md) irányelveinket. Várjuk építő jellegű visszajelzéseidet!

## Minden lecke tartalmaz

- opcionális vázlatjegyzetet
- opcionális kiegészítő videót
- videós bemutatót (csak egyes leckéknél)
- [előadás előtti bemelegítő kvízt](https://ff-quizzes.netlify.app/en/ml/)
- írásos leckét
- projekt-alapú leckékhez lépésenkénti útmutatót a projekt elkészítéséhez
- tudásellenőrzéseket
- kihívást
- kiegészítő olvasmányt
- feladatot
- [előadás utáni kvízt](https://ff-quizzes.netlify.app/en/ml/)

> **Megjegyzés a nyelvekről**: Ezek a leckék elsősorban Pythonban íródtak, de sok elérhető R-ben is. Az R lecke elvégzéséhez menj a `/solution` mappába, és keress R leckéket. Ezek .rmd kiterjesztésű fájlok, amelyek egy **R Markdown** fájlt jelentenek, egyszerűen megfogalmazva egy olyan dokumentumot, ami `kódrészleteket` (R vagy más nyelvek) és egy `YAML fejlécet` (amely útmutatást ad a kimenetek, például PDF formázására) tartalmaz markdown dokumentumban. Így példamutató szerkesztési keretet nyújt az adattudományhoz, mivel lehetővé teszi, hogy kódodat, annak kimenetét és gondolataidat markdownban írd le. R Markdown dokumentumok PDF, HTML vagy Word kimeneti formátumokba is konvertálhatók.
> **Megjegyzés a kvízekhez**: Az összes kvíz megtalálható a [Quiz App mappában](../../quiz-app), összesen 52 darab, mindegyik három kérdéssel. Az órákból van linkelve, de a kvízalkalmazás helyben is futtatható; kövesd a `quiz-app` mappa utasításait a helyi hosztoláshoz vagy az Azure-ra történő telepítéshez.

| Óra száma |                             Téma                              |                   Óra csoportosítása                   | Tanulási célok                                                                                                             |                                                              Linked Lesson                                                               |                        Szerző                        |
| :-------: | :----------------------------------------------------------: | :----------------------------------------------------: | -------------------------------------------------------------------------------------------------------------------------- | :--------------------------------------------------------------------------------------------------------------------------------------: | :-------------------------------------------------: |
|    01     |                Bevezetés a gépi tanulásba                   |      [Bevezetés](1-Introduction/README.md)             | Ismerd meg a gépi tanulás alapfogalmait                                                                                   |                                             [Óra](1-Introduction/1-intro-to-ML/README.md)                                            |                       Muhammad                       |
|    02     |                A gépi tanulás története                     |      [Bevezetés](1-Introduction/README.md)             | Ismerd meg e terület történetét                                                                                           |                                            [Óra](1-Introduction/2-history-of-ML/README.md)                                             |                     Jen és Amy                       |
|    03     |                 Méltányosság és gépi tanulás                |      [Bevezetés](1-Introduction/README.md)             | Mik a fontos filozófiai kérdések a méltányossággal kapcsolatban, amelyeket a tanulóknak figyelembe kell venniük a gépi tanulási modellek építése és alkalmazása során? |                                              [Óra](1-Introduction/3-fairness/README.md)                                               |                        Tomomi                        |
|    04     |                Gépi tanulási technikák                      |      [Bevezetés](1-Introduction/README.md)             | Milyen technikákat használnak a gépi tanulás kutatói a modellek építéséhez?                                               |                                          [Óra](1-Introduction/4-techniques-of-ML/README.md)                                           |                    Chris és Jen                      |
|    05     |                   Bevezetés a regresszióba                   |        [Regresszió](2-Regression/README.md)             | Ismerkedj meg a Python és Scikit-learn használatával regressziós modellekhez                                             |         [Python](2-Regression/1-Tools/README.md) • [R](../../2-Regression/1-Tools/solution/R/lesson_1.html)         |      Jen • Eric Wanjau       |
|    06     |                Észak-amerikai tökárak 🎃                     |        [Regresszió](2-Regression/README.md)             | Adatok vizualizálása és tisztítása gépi tanulás előkészítéséhez                                                           |          [Python](2-Regression/2-Data/README.md) • [R](../../2-Regression/2-Data/solution/R/lesson_2.html)          |      Jen • Eric Wanjau       |
|    07     |                Észak-amerikai tökárak 🎃                     |        [Regresszió](2-Regression/README.md)             | Lineáris és polinomiális regressziós modellek építése                                                                     |        [Python](2-Regression/3-Linear/README.md) • [R](../../2-Regression/3-Linear/solution/R/lesson_3.html)        |      Jen és Dmitry • Eric Wanjau       |
|    08     |                Észak-amerikai tökárak 🎃                     |        [Regresszió](2-Regression/README.md)             | Logisztikus regressziós modell építése                                                                                     |     [Python](2-Regression/4-Logistic/README.md) • [R](../../2-Regression/4-Logistic/solution/R/lesson_4.html)      |      Jen • Eric Wanjau       |
|    09     |                          Webalkalmazás 🔌                    |           [Web App](3-Web-App/README.md)                | Webalkalmazás építése a betanított modell használatához                                                                   |                                                 [Python](3-Web-App/1-Web-App/README.md)                                                  |                         Jen                          |
|    10     |                 Bevezetés osztályozásba                      |    [Osztályozás](4-Classification/README.md)           | Tisztítsd, készítsd elő és vizualizáld az adatokat; bevezetés az osztályozásba                                              | [Python](4-Classification/1-Introduction/README.md) • [R](../../4-Classification/1-Introduction/solution/R/lesson_10.html)  | Jen és Cassie • Eric Wanjau |
|    11     |             Finom ázsiai és indiai konyhák 🍜                 |    [Osztályozás](4-Classification/README.md)           | Bevezetés az osztályozókba                                                                                                 | [Python](4-Classification/2-Classifiers-1/README.md) • [R](../../4-Classification/2-Classifiers-1/solution/R/lesson_11.html) | Jen és Cassie • Eric Wanjau |
|    12     |             Finom ázsiai és indiai konyhák 🍜                 |    [Osztályozás](4-Classification/README.md)           | Több osztályozó                                                                                                            | [Python](4-Classification/3-Classifiers-2/README.md) • [R](../../4-Classification/3-Classifiers-2/solution/R/lesson_12.html) | Jen és Cassie • Eric Wanjau |
|    13     |             Finom ázsiai és indiai konyhák 🍜                 |    [Osztályozás](4-Classification/README.md)           | Ajánló webalkalmazás építése a modelled használatával                                                                      |                                              [Python](4-Classification/4-Applied/README.md)                                              |                         Jen                          |
|    14     |                   Bevezetés a klaszterezésbe                  |        [Klaszterezés](5-Clustering/README.md)           | Tisztítsd, készítsd elő és vizualizáld az adatokat; bevezetés a klaszterezésbe                                             |         [Python](5-Clustering/1-Visualize/README.md) • [R](../../5-Clustering/1-Visualize/solution/R/lesson_14.html)         |      Jen • Eric Wanjau       |
|    15     |              Felfedezés a nigériai zenei ízlésekben 🎧        |        [Klaszterezés](5-Clustering/README.md)           | Fedezd fel a K-közép klaszterezési módszert                                                                                 |           [Python](5-Clustering/2-K-Means/README.md) • [R](../../5-Clustering/2-K-Means/solution/R/lesson_15.html)           |      Jen • Eric Wanjau       |
|    16     |        Bevezetés a természetes nyelvfeldolgozásba ☕️         |   [Természetes nyelvfeldolgozás](6-NLP/README.md)        | Ismerd meg az NLP alapjait egyszerű bot építésével                                                                         |                                             [Python](6-NLP/1-Introduction-to-NLP/README.md)                                              |                       Stephen                        |
|    17     |                      Gyakori NLP feladatok ☕️                |   [Természetes nyelvfeldolgozás](6-NLP/README.md)        | Mélyítsd el NLP tudásod, értsd meg a nyelvi struktúrákkal kapcsolatos gyakori feladatokat                                     |                                                    [Python](6-NLP/2-Tasks/README.md)                                                     |                       Stephen                        |
|    18     |             Fordítás és érzelemelemzés ♥️                     |   [Természetes nyelvfeldolgozás](6-NLP/README.md)        | Fordítás és érzelemelemzés Jane Austen műveivel                                                                            |                                            [Python](6-NLP/3-Translation-Sentiment/README.md)                                             |                       Stephen                        |
|    19     |                  Romantikus európai szállodák ♥️              |   [Természetes nyelvfeldolgozás](6-NLP/README.md)        | Érzelemelemzés szállodai véleményekkel 1                                                                                   |                                               [Python](6-NLP/4-Hotel-Reviews-1/README.md)                                                |                       Stephen                        |
|    20     |                  Romantikus európai szállodák ♥️              |   [Természetes nyelvfeldolgozás](6-NLP/README.md)        | Érzelemelemzés szállodai véleményekkel 2                                                                                   |                                               [Python](6-NLP/5-Hotel-Reviews-2/README.md)                                                |                       Stephen                        |
|    21     |            Bevezetés az idősor-előrejelzésbe                  |        [Idősor](7-TimeSeries/README.md)                   | Bevezetés az idősor előrejelzésébe                                                                                         |                                             [Python](7-TimeSeries/1-Introduction/README.md)                                              |                      Francesca                       |
|    22     | ⚡️ Világ energiafelhasználás ⚡️ - idősor előrejelzés ARIMA-val |        [Idősor](7-TimeSeries/README.md)                   | Idősor előrejelzés ARIMA-val                                                                                                |                                                 [Python](7-TimeSeries/2-ARIMA/README.md)                                                 |                      Francesca                       |
|    23     |  ⚡️ Világ energiafelhasználás ⚡️ - idősor előrejelzés SVR-rel  |        [Idősor](7-TimeSeries/README.md)                   | Idősor előrejelzés Támogató Vektorgépes regresszorral (SVR)                                                                |                                                  [Python](7-TimeSeries/3-SVR/README.md)                                                  |                       Anirban                        |
|    24     |             Bevezetés a megerősítéses tanulásba               | [Megerősítéses tanulás](8-Reinforcement/README.md)        | Bevezetés a megerősítéses tanulásba Q-learning használatával                                                               |                                             [Python](8-Reinforcement/1-QLearning/README.md)                                              |                        Dmitry                        |
|    25     |                 Segíts Peternek elkerülni a farkast! 🐺        | [Megerősítéses tanulás](8-Reinforcement/README.md)        | Megerősítéses tanulás Gym keretrendszerrel                                                                                  |                                                [Python](8-Reinforcement/2-Gym/README.md)                                                 |                        Dmitry                        |
| Utóirat   |            Valós világ gépi tanulási forgatókönyvek és alkalmazások |      [Gépi tanulás a gyakorlatban](9-Real-World/README.md) | Érdekes és tanulságos valós példák a klasszikus gépi tanulás alkalmazásaira                                                 |                                             [Óra](9-Real-World/1-Applications/README.md)                                              |                         Csapat                        |
| Utóirat   |            Modell hibakeresése gépi tanulásban az RAI irányítópult segítségével |      [Gépi tanulás a gyakorlatban](9-Real-World/README.md) | Modell hibakeresése gépi tanulásban a Responsible AI irányítópult komponenseinek segítségével                                 |                                             [Óra](9-Real-World/2-Debugging-ML-Models/README.md)                                              |                         Ruth Yakubu                       |

> [keresd meg az összes további erőforrást ehhez a tanfolyamhoz a Microsoft Learn gyűjteményünkben](https://learn.microsoft.com/en-us/collections/qrqzamz1nn2wx3?WT.mc_id=academic-77952-bethanycheum)

## Offline hozzáférés

A dokumentációt offline is futtathatod a [Docsify](https://docsify.js.org/#/) segítségével. Forkold ezt a repót, [telepítsd a Docsify-t](https://docsify.js.org/#/quickstart) a helyi gépeden, majd a repo gyökérmappájában írd be a `docsify serve` parancsot. Az oldal a localhost 3000-es portján lesz elérhető: `localhost:3000`.

## PDF-ek

A tananyag pdf formátumban, linkekkel [itt található](https://microsoft.github.io/ML-For-Beginners/pdf/readme.pdf).


## 🎒 Egyéb tanfolyamok

Csapatunk más tanfolyamokat is készít! Nézd meg:

<!-- CO-OP TRANSLATOR OTHER COURSES START -->
### LangChain
[![LangChain4j kezdőknek](https://img.shields.io/badge/LangChain4j%20for%20Beginners-22C55E?style=for-the-badge&&labelColor=E5E7EB&color=0553D6)](https://aka.ms/langchain4j-for-beginners)
[![LangChain.js kezdőknek](https://img.shields.io/badge/LangChain.js%20for%20Beginners-22C55E?style=for-the-badge&labelColor=E5E7EB&color=0553D6)](https://aka.ms/langchainjs-for-beginners?WT.mc_id=m365-94501-dwahlin)

---

### Azure / Edge / MCP / Ügynökök
[![AZD kezdőknek](https://img.shields.io/badge/AZD%20for%20Beginners-0078D4?style=for-the-badge&labelColor=E5E7EB&color=0078D4)](https://github.com/microsoft/AZD-for-beginners?WT.mc_id=academic-105485-koreyst)
[![Edge AI kezdőknek](https://img.shields.io/badge/Edge%20AI%20for%20Beginners-00B8E4?style=for-the-badge&labelColor=E5E7EB&color=00B8E4)](https://github.com/microsoft/edgeai-for-beginners?WT.mc_id=academic-105485-koreyst)
[![MCP kezdőknek](https://img.shields.io/badge/MCP%20for%20Beginners-009688?style=for-the-badge&labelColor=E5E7EB&color=009688)](https://github.com/microsoft/mcp-for-beginners?WT.mc_id=academic-105485-koreyst)
[![AI ügynökök kezdőknek](https://img.shields.io/badge/AI%20Agents%20for%20Beginners-00C49A?style=for-the-badge&labelColor=E5E7EB&color=00C49A)](https://github.com/microsoft/ai-agents-for-beginners?WT.mc_id=academic-105485-koreyst)

---
 
### Generatív MI sorozat
[![Generatív MI kezdőknek](https://img.shields.io/badge/Generative%20AI%20for%20Beginners-8B5CF6?style=for-the-badge&labelColor=E5E7EB&color=8B5CF6)](https://github.com/microsoft/generative-ai-for-beginners?WT.mc_id=academic-105485-koreyst)
[![Generatív MI (.NET)](https://img.shields.io/badge/Generative%20AI%20(.NET)-9333EA?style=for-the-badge&labelColor=E5E7EB&color=9333EA)](https://github.com/microsoft/Generative-AI-for-beginners-dotnet?WT.mc_id=academic-105485-koreyst)
[![Generatív MI (Java)](https://img.shields.io/badge/Generative%20AI%20(Java)-C084FC?style=for-the-badge&labelColor=E5E7EB&color=C084FC)](https://github.com/microsoft/generative-ai-for-beginners-java?WT.mc_id=academic-105485-koreyst)
[![Generatív MI (JavaScript)](https://img.shields.io/badge/Generative%20AI%20(JavaScript)-E879F9?style=for-the-badge&labelColor=E5E7EB&color=E879F9)](https://github.com/microsoft/generative-ai-with-javascript?WT.mc_id=academic-105485-koreyst)

---
 
### Alapvető tanulás
[![ML kezdőknek](https://img.shields.io/badge/ML%20for%20Beginners-22C55E?style=for-the-badge&labelColor=E5E7EB&color=22C55E)](https://aka.ms/ml-beginners?WT.mc_id=academic-105485-koreyst)
[![Adattudomány kezdőknek](https://img.shields.io/badge/Data%20Science%20for%20Beginners-84CC16?style=for-the-badge&labelColor=E5E7EB&color=84CC16)](https://aka.ms/datascience-beginners?WT.mc_id=academic-105485-koreyst)
[![MI kezdőknek](https://img.shields.io/badge/AI%20for%20Beginners-A3E635?style=for-the-badge&labelColor=E5E7EB&color=A3E635)](https://aka.ms/ai-beginners?WT.mc_id=academic-105485-koreyst)
[![Kiberbiztonság kezdőknek](https://img.shields.io/badge/Cybersecurity%20for%20Beginners-F97316?style=for-the-badge&labelColor=E5E7EB&color=F97316)](https://github.com/microsoft/Security-101?WT.mc_id=academic-96948-sayoung)
[![Webfejlesztés kezdőknek](https://img.shields.io/badge/Web%20Dev%20for%20Beginners-EC4899?style=for-the-badge&labelColor=E5E7EB&color=EC4899)](https://aka.ms/webdev-beginners?WT.mc_id=academic-105485-koreyst)
[![IoT kezdőknek](https://img.shields.io/badge/IoT%20for%20Beginners-14B8A6?style=for-the-badge&labelColor=E5E7EB&color=14B8A6)](https://aka.ms/iot-beginners?WT.mc_id=academic-105485-koreyst)
[![XR fejlesztés kezdőknek](https://img.shields.io/badge/XR%20Development%20for%20Beginners-38BDF8?style=for-the-badge&labelColor=E5E7EB&color=38BDF8)](https://github.com/microsoft/xr-development-for-beginners?WT.mc_id=academic-105485-koreyst)

---
 
### Copilot sorozat
[![Copilot AI páros programozáshoz](https://img.shields.io/badge/Copilot%20for%20AI%20Paired%20Programming-FACC15?style=for-the-badge&labelColor=E5E7EB&color=FACC15)](https://aka.ms/GitHubCopilotAI?WT.mc_id=academic-105485-koreyst)
[![Copilot C#/.NET-hez](https://img.shields.io/badge/Copilot%20for%20C%23/.NET-FBBF24?style=for-the-badge&labelColor=E5E7EB&color=FBBF24)](https://github.com/microsoft/mastering-github-copilot-for-dotnet-csharp-developers?WT.mc_id=academic-105485-koreyst)
[![Copilot kaland](https://img.shields.io/badge/Copilot%20Adventure-FDE68A?style=for-the-badge&labelColor=E5E7EB&color=FDE68A)](https://github.com/microsoft/CopilotAdventures?WT.mc_id=academic-105485-koreyst)
<!-- CO-OP TRANSLATOR OTHER COURSES END -->

## Segítség kérése

Ha elakad vagy kérdése van az MI-alapú alkalmazások fejlesztésével kapcsolatban, csatlakozzon a tanulótársakhoz és tapasztalt fejlesztőkhöz az MCP vitáiban. Ez egy támogató közösség, ahol a kérdések szívesen látottak és a tudás szabadon megosztott.

[![Microsoft Foundry Discord](https://dcbadge.limes.pink/api/server/nTYy5BXMWG)](https://discord.gg/nTYy5BXMWG)

Ha termék-visszajelzése vagy hibája van a fejlesztés során, látogasson el ide:

[![Microsoft Foundry fejlesztői fórum](https://img.shields.io/badge/GitHub-Microsoft_Foundry_Developer_Forum-blue?style=for-the-badge&logo=github&color=000000&logoColor=fff)](https://aka.ms/foundry/forum)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**Jogi nyilatkozat**:
Ez a dokumentum az AI fordító szolgáltatás [Co-op Translator](https://github.com/Azure/co-op-translator) segítségével készült. Bár a pontosságra törekszünk, kérjük, vegye figyelembe, hogy az automatikus fordítás hibákat vagy pontatlanságokat tartalmazhat. Az eredeti dokumentum az anyanyelvén tekinthető hiteles forrásnak. Kritikus információk esetén professzionális emberi fordítást javaslunk. Nem vállalunk felelősséget az ebből a fordításból eredő félreértésekért vagy téves értelmezésekért.
<!-- CO-OP TRANSLATOR DISCLAIMER END -->