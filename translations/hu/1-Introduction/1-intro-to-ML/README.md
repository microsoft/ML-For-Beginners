<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "69389392fa6346e0dfa30f664b7b6fec",
  "translation_date": "2025-09-05T16:06:36+00:00",
  "source_file": "1-Introduction/1-intro-to-ML/README.md",
  "language_code": "hu"
}
-->
# Bevezetés a gépi tanulásba

## [Előadás előtti kvíz](https://ff-quizzes.netlify.app/en/ml/)

---

[![Gépi tanulás kezdőknek - Bevezetés a gépi tanulásba](https://img.youtube.com/vi/6mSx_KJxcHI/0.jpg)](https://youtu.be/6mSx_KJxcHI "Gépi tanulás kezdőknek - Bevezetés a gépi tanulásba")

> 🎥 Kattints a fenti képre egy rövid videóért, amely bemutatja ezt a leckét.

Üdvözlünk ezen a kezdőknek szóló klasszikus gépi tanulás kurzuson! Akár teljesen új vagy a témában, akár tapasztalt ML szakemberként szeretnéd felfrissíteni tudásodat, örülünk, hogy csatlakoztál hozzánk! Célunk, hogy barátságos kiindulópontot biztosítsunk a gépi tanulás tanulmányozásához, és szívesen fogadjuk, értékeljük, valamint beépítjük [visszajelzéseidet](https://github.com/microsoft/ML-For-Beginners/discussions).

[![Bevezetés a gépi tanulásba](https://img.youtube.com/vi/h0e2HAPTGF4/0.jpg)](https://youtu.be/h0e2HAPTGF4 "Bevezetés a gépi tanulásba")

> 🎥 Kattints a fenti képre egy videóért: MIT John Guttag bemutatja a gépi tanulást

---
## Első lépések a gépi tanulásban

Mielőtt elkezdenéd ezt a tananyagot, győződj meg róla, hogy számítógéped készen áll a notebookok helyi futtatására.

- **Állítsd be a gépedet ezekkel a videókkal**. Használd az alábbi linkeket, hogy megtanuld [hogyan telepítsd a Python-t](https://youtu.be/CXZYvNRIAKM) a rendszeredre, és [hogyan állítsd be egy szövegszerkesztőt](https://youtu.be/EU8eayHWoZg) a fejlesztéshez.
- **Tanuld meg a Python alapjait**. Ajánlott, hogy legyen alapvető ismereted a [Pythonról](https://docs.microsoft.com/learn/paths/python-language/?WT.mc_id=academic-77952-leestott), egy programozási nyelvről, amely hasznos az adatkutatók számára, és amelyet ebben a kurzusban használunk.
- **Tanuld meg a Node.js-t és a JavaScriptet**. A kurzus során néhány alkalommal használjuk a JavaScriptet webalkalmazások készítéséhez, ezért szükséged lesz [node](https://nodejs.org) és [npm](https://www.npmjs.com/) telepítésére, valamint [Visual Studio Code](https://code.visualstudio.com/) használatára Python és JavaScript fejlesztéshez.
- **Hozz létre egy GitHub fiókot**. Mivel itt találtál ránk a [GitHubon](https://github.com), lehet, hogy már van fiókod, de ha nincs, hozz létre egyet, majd forkolj meg ezt a tananyagot, hogy saját magad használhasd. (Ne felejts el csillagot adni nekünk 😊)
- **Ismerkedj meg a Scikit-learnnel**. Ismerd meg a [Scikit-learn](https://scikit-learn.org/stable/user_guide.html) könyvtárat, amelyet ezekben a leckékben hivatkozunk.

---
## Mi az a gépi tanulás?

A 'gépi tanulás' kifejezés napjaink egyik legnépszerűbb és leggyakrabban használt fogalma. Nem kizárt, hogy legalább egyszer hallottad már ezt a kifejezést, ha valamilyen szinten ismered a technológiát, függetlenül attól, hogy milyen területen dolgozol. A gépi tanulás mechanikája azonban a legtöbb ember számára rejtély. Egy gépi tanulás kezdő számára a téma néha túlterhelőnek tűnhet. Ezért fontos megérteni, hogy valójában mi is a gépi tanulás, és lépésről lépésre, gyakorlati példákon keresztül tanulni róla.

---
## A hype görbe

![ml hype curve](../../../../1-Introduction/1-intro-to-ML/images/hype.png)

> A Google Trends mutatja a 'gépi tanulás' kifejezés legutóbbi hype görbéjét

---
## Egy rejtélyes univerzum

Egy lenyűgöző rejtélyekkel teli univerzumban élünk. Nagy tudósok, mint Stephen Hawking, Albert Einstein és sokan mások, életüket annak szentelték, hogy értelmes információkat találjanak, amelyek feltárják a körülöttünk lévő világ rejtélyeit. Ez az emberi tanulás feltétele: egy emberi gyermek új dolgokat tanul, és évről évre felfedezi világának szerkezetét, ahogy felnőtté válik.

---
## A gyermek agya

Egy gyermek agya és érzékei érzékelik környezetük tényeit, és fokozatosan megtanulják az élet rejtett mintázatait, amelyek segítenek logikai szabályokat alkotni a tanult minták azonosításához. Az emberi agy tanulási folyamata teszi az embereket a világ legkifinomultabb élőlényévé. Azáltal, hogy folyamatosan tanulunk, felfedezzük a rejtett mintákat, majd innoválunk ezek alapján, képesek vagyunk egyre jobbak lenni életünk során. Ez a tanulási képesség és fejlődési kapacitás összefüggésben áll egy [agy plaszticitásának](https://www.simplypsychology.org/brain-plasticity.html) nevezett fogalommal. Felületesen nézve motivációs hasonlóságokat vonhatunk az emberi agy tanulási folyamata és a gépi tanulás fogalmai között.

---
## Az emberi agy

Az [emberi agy](https://www.livescience.com/29365-human-brain.html) érzékeli a valós világ dolgait, feldolgozza az érzékelt információkat, racionális döntéseket hoz, és bizonyos körülmények alapján cselekszik. Ezt nevezzük intelligens viselkedésnek. Amikor egy intelligens viselkedési folyamatot programozunk egy gépbe, azt mesterséges intelligenciának (AI) nevezzük.

---
## Néhány terminológia

Bár a fogalmak összekeverhetők, a gépi tanulás (ML) a mesterséges intelligencia fontos részhalmaza. **Az ML arra összpontosít, hogy speciális algoritmusokat használjon értelmes információk feltárására és rejtett minták megtalálására az érzékelt adatokból, hogy támogassa a racionális döntéshozatali folyamatot**.

---
## AI, ML, Mélytanulás

![AI, ML, deep learning, data science](../../../../1-Introduction/1-intro-to-ML/images/ai-ml-ds.png)

> Egy diagram, amely bemutatja az AI, ML, mélytanulás és adatkutatás közötti kapcsolatokat. Infografika [Jen Looper](https://twitter.com/jenlooper) által, amelyet [ez a grafika](https://softwareengineering.stackexchange.com/questions/366996/distinction-between-ai-ml-neural-networks-deep-learning-and-data-mining) inspirált.

---
## Lefedendő fogalmak

Ebben a tananyagban csak a gépi tanulás alapvető fogalmait fogjuk lefedni, amelyeket egy kezdőnek ismernie kell. Elsősorban a 'klasszikus gépi tanulást' tárgyaljuk, főként a Scikit-learn használatával, amely egy kiváló könyvtár, amit sok diák használ az alapok elsajátításához. Ahhoz, hogy megértsük a mesterséges intelligencia vagy mélytanulás szélesebb körű fogalmait, elengedhetetlen a gépi tanulás erős alapvető ismerete, és ezt szeretnénk itt biztosítani.

---
## Ebben a kurzusban megtanulod:

- a gépi tanulás alapfogalmait
- az ML történetét
- az ML és az igazságosság kapcsolatát
- regressziós ML technikákat
- osztályozási ML technikákat
- klaszterezési ML technikákat
- természetes nyelvfeldolgozási ML technikákat
- időbeli előrejelzési ML technikákat
- megerősítéses tanulást
- az ML valós alkalmazásait

---
## Amit nem fogunk lefedni

- mélytanulás
- neurális hálózatok
- mesterséges intelligencia

A jobb tanulási élmény érdekében elkerüljük a neurális hálózatok, a 'mélytanulás' - többrétegű modellépítés neurális hálózatokkal - és az AI komplexitásait, amelyeket egy másik tananyagban fogunk tárgyalni. Emellett egy közelgő adatkutatási tananyagot is kínálunk, amely erre a nagyobb területre összpontosít.

---
## Miért érdemes gépi tanulást tanulni?

A gépi tanulás rendszerszempontból úgy definiálható, mint automatizált rendszerek létrehozása, amelyek képesek rejtett mintákat tanulni az adatokból, hogy segítsenek intelligens döntések meghozatalában.

Ez a motiváció lazán inspirálódik abból, ahogyan az emberi agy bizonyos dolgokat tanul az érzékelt adatok alapján.

✅ Gondolkodj el egy percre azon, hogy egy vállalkozás miért választaná a gépi tanulási stratégiákat egy keményen kódolt szabályalapú motor létrehozása helyett.

---
## A gépi tanulás alkalmazásai

A gépi tanulás alkalmazásai ma már szinte mindenhol jelen vannak, és olyan elterjedtek, mint az adatok, amelyek társadalmainkban áramlanak, okostelefonjaink, csatlakoztatott eszközeink és más rendszereink által generálva. Figyelembe véve a legmodernebb gépi tanulási algoritmusok hatalmas potenciálját, a kutatók vizsgálják azok képességét, hogy multidimenziós és multidiszciplináris valós problémákat oldjanak meg nagy pozitív eredményekkel.

---
## Alkalmazott ML példák

**A gépi tanulást számos módon használhatod**:

- Betegség valószínűségének előrejelzésére egy beteg kórtörténete vagy jelentései alapján.
- Időjárási adatok felhasználásával időjárási események előrejelzésére.
- Szöveg érzelmi tartalmának megértésére.
- Hamis hírek és propaganda terjedésének megállítására.

A pénzügyek, közgazdaságtan, földtudomány, űrkutatás, biomedikai mérnökség, kognitív tudomány és még a humán tudományok területei is alkalmazzák a gépi tanulást, hogy megoldják saját területük nehéz, adatfeldolgozás-igényes problémáit.

---
## Összegzés

A gépi tanulás automatizálja a mintázat-felfedezés folyamatát azáltal, hogy értelmes betekintéseket talál a valós vagy generált adatokból. Bizonyította értékét az üzleti, egészségügyi és pénzügyi alkalmazásokban, többek között.

A közeljövőben a gépi tanulás alapjainak megértése elengedhetetlen lesz minden területen dolgozó emberek számára, tekintettel annak széles körű elterjedésére.

---
# 🚀 Kihívás

Rajzolj papíron vagy egy online alkalmazás, például [Excalidraw](https://excalidraw.com/) segítségével egy vázlatot arról, hogyan érted az AI, ML, mélytanulás és adatkutatás közötti különbségeket. Adj hozzá néhány ötletet arról, hogy milyen problémák megoldására alkalmasak ezek a technikák.

# [Előadás utáni kvíz](https://ff-quizzes.netlify.app/en/ml/)

---
# Áttekintés és önálló tanulás

Ha többet szeretnél megtudni arról, hogyan dolgozhatsz ML algoritmusokkal a felhőben, kövesd ezt a [tanulási útvonalat](https://docs.microsoft.com/learn/paths/create-no-code-predictive-models-azure-machine-learning/?WT.mc_id=academic-77952-leestott).

Vegyél részt egy [tanulási útvonalon](https://docs.microsoft.com/learn/modules/introduction-to-machine-learning/?WT.mc_id=academic-77952-leestott), amely az ML alapjairól szól.

---
# Feladat

[Indulj el](assignment.md)

---

**Felelősség kizárása**:  
Ez a dokumentum az AI fordítási szolgáltatás, a [Co-op Translator](https://github.com/Azure/co-op-translator) segítségével lett lefordítva. Bár törekszünk a pontosságra, kérjük, vegye figyelembe, hogy az automatikus fordítások hibákat vagy pontatlanságokat tartalmazhatnak. Az eredeti dokumentum az eredeti nyelvén tekintendő hiteles forrásnak. Kritikus információk esetén javasolt professzionális emberi fordítást igénybe venni. Nem vállalunk felelősséget semmilyen félreértésért vagy téves értelmezésért, amely a fordítás használatából eredhet.