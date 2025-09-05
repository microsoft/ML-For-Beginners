<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "6a05fec147e734c3e6bfa54505648e2b",
  "translation_date": "2025-09-05T16:09:18+00:00",
  "source_file": "1-Introduction/2-history-of-ML/README.md",
  "language_code": "hu"
}
-->
# A gépi tanulás története

![A gépi tanulás történetének összefoglalása sketchnote-ban](../../../../sketchnotes/ml-history.png)
> Sketchnote készítette: [Tomomi Imura](https://www.twitter.com/girlie_mac)

## [Előadás előtti kvíz](https://ff-quizzes.netlify.app/en/ml/)

---

[![Gépi tanulás kezdőknek - A gépi tanulás története](https://img.youtube.com/vi/N6wxM4wZ7V0/0.jpg)](https://youtu.be/N6wxM4wZ7V0 "Gépi tanulás kezdőknek - A gépi tanulás története")

> 🎥 Kattints a fenti képre egy rövid videóért, amely bemutatja ezt a leckét.

Ebben a leckében végigjárjuk a gépi tanulás és mesterséges intelligencia történetének főbb mérföldköveit.

A mesterséges intelligencia (MI) mint terület története szorosan összefonódik a gépi tanulás történetével, mivel a gépi tanulás alapját képező algoritmusok és számítástechnikai fejlődések hozzájárultak az MI fejlődéséhez. Érdemes megjegyezni, hogy bár ezek a területek mint különálló kutatási irányok az 1950-es években kezdtek körvonalazódni, fontos [algoritmikus, statisztikai, matematikai, számítástechnikai és technikai felfedezések](https://wikipedia.org/wiki/Timeline_of_machine_learning) már korábban is történtek, és átfedték ezt az időszakot. Valójában az emberek már [évszázadok óta](https://wikipedia.org/wiki/History_of_artificial_intelligence) foglalkoznak ezekkel a kérdésekkel: ez a cikk a "gondolkodó gép" ötletének történelmi szellemi alapjait tárgyalja.

---
## Figyelemre méltó felfedezések

- 1763, 1812 [Bayes-tétel](https://wikipedia.org/wiki/Bayes%27_theorem) és elődei. Ez a tétel és alkalmazásai az események bekövetkezésének valószínűségét írják le korábbi ismeretek alapján.
- 1805 [Legkisebb négyzetek módszere](https://wikipedia.org/wiki/Least_squares) Adrien-Marie Legendre francia matematikustól. Ez az elmélet, amelyet a regresszióról szóló egységünkben tanulni fogsz, segít az adatok illesztésében.
- 1913 [Markov-láncok](https://wikipedia.org/wiki/Markov_chain), amelyeket Andrey Markov orosz matematikusról neveztek el, egy eseménysorozatot írnak le, amely az előző állapoton alapul.
- 1957 [Perceptron](https://wikipedia.org/wiki/Perceptron), egy lineáris osztályozó típusa, amelyet Frank Rosenblatt amerikai pszichológus talált fel, és amely a mélytanulás fejlődésének alapját képezi.

---

- 1967 [Legközelebbi szomszéd](https://wikipedia.org/wiki/Nearest_neighbor) algoritmus, amelyet eredetileg útvonalak feltérképezésére terveztek. Gépi tanulásban mintázatok felismerésére használják.
- 1970 [Visszaterjesztés](https://wikipedia.org/wiki/Backpropagation), amelyet [előrecsatolt neurális hálók](https://wikipedia.org/wiki/Feedforward_neural_network) tanítására használnak.
- 1982 [Rekurzív neurális hálók](https://wikipedia.org/wiki/Recurrent_neural_network), amelyek az előrecsatolt neurális hálókból származnak, és időbeli gráfokat hoznak létre.

✅ Végezz egy kis kutatást. Mely más dátumok emelkednek ki a gépi tanulás és MI történetében?

---
## 1950: Gondolkodó gépek

Alan Turing, egy igazán figyelemre méltó személy, akit [2019-ben a közönség](https://wikipedia.org/wiki/Icons:_The_Greatest_Person_of_the_20th_Century) a 20. század legnagyobb tudósának választott, segített lefektetni a "gondolkodó gép" koncepciójának alapjait. Turing a kétkedőkkel és saját empirikus bizonyítékok iránti igényével küzdött, részben azáltal, hogy megalkotta a [Turing-tesztet](https://www.bbc.com/news/technology-18475646), amelyet a NLP leckéinkben fogsz megvizsgálni.

---
## 1956: Dartmouth nyári kutatási projekt

"A Dartmouth nyári kutatási projekt a mesterséges intelligenciáról egy alapvető esemény volt a mesterséges intelligencia mint terület számára," és itt alkották meg a 'mesterséges intelligencia' kifejezést ([forrás](https://250.dartmouth.edu/highlights/artificial-intelligence-ai-coined-dartmouth)).

> A tanulás vagy az intelligencia bármely más jellemzője elvileg olyan pontosan leírható, hogy egy gép képes legyen szimulálni azt.

---

A vezető kutató, John McCarthy matematikaprofesszor remélte, hogy "a tanulás vagy az intelligencia bármely más jellemzője elvileg olyan pontosan leírható, hogy egy gép képes legyen szimulálni azt." A résztvevők között volt egy másik kiemelkedő személyiség, Marvin Minsky.

A workshopot annak tulajdonítják, hogy számos vitát kezdeményezett és ösztönzött, beleértve "a szimbolikus módszerek felemelkedését, a korlátozott területekre összpontosító rendszereket (korai szakértői rendszerek), valamint a deduktív rendszerek és az induktív rendszerek közötti különbségeket." ([forrás](https://wikipedia.org/wiki/Dartmouth_workshop)).

---
## 1956 - 1974: "Az aranyévek"

Az 1950-es évektől a '70-es évek közepéig nagy optimizmus uralkodott abban a reményben, hogy az MI számos problémát megoldhat. 1967-ben Marvin Minsky magabiztosan kijelentette, hogy "Egy generáción belül ... az 'mesterséges intelligencia' létrehozásának problémája lényegében megoldódik." (Minsky, Marvin (1967), Computation: Finite and Infinite Machines, Englewood Cliffs, N.J.: Prentice-Hall)

A természetes nyelvfeldolgozás kutatása virágzott, a keresés finomodott és hatékonyabbá vált, és létrejött a 'mikrovilágok' koncepciója, ahol egyszerű feladatokat lehetett elvégezni egyszerű nyelvi utasításokkal.

---

A kutatást jól finanszírozták kormányzati ügynökségek, előrelépések történtek a számítástechnikában és algoritmusokban, és intelligens gépek prototípusait építették. Néhány ilyen gép:

* [Shakey robot](https://wikipedia.org/wiki/Shakey_the_robot), amely képes volt manőverezni és 'intelligensen' dönteni a feladatok elvégzéséről.

    ![Shakey, egy intelligens robot](../../../../1-Introduction/2-history-of-ML/images/shakey.jpg)
    > Shakey 1972-ben

---

* Eliza, egy korai 'beszélgetőbot', képes volt emberekkel beszélgetni és primitív 'terapeutaként' működni. Az NLP leckékben többet fogsz tanulni Elizáról.

    ![Eliza, egy bot](../../../../1-Introduction/2-history-of-ML/images/eliza.png)
    > Eliza egy verziója, egy chatbot

---

* "Blocks world" egy mikrovilág példája volt, ahol blokkokat lehetett egymásra rakni és rendezni, és kísérleteket lehetett végezni a gépek döntéshozatalának tanításával. Az olyan könyvtárakkal, mint [SHRDLU](https://wikipedia.org/wiki/SHRDLU), végzett fejlesztések elősegítették a nyelvfeldolgozás fejlődését.

    [![blocks world SHRDLU-val](https://img.youtube.com/vi/QAJz4YKUwqw/0.jpg)](https://www.youtube.com/watch?v=QAJz4YKUwqw "blocks world SHRDLU-val")

    > 🎥 Kattints a fenti képre egy videóért: Blocks world SHRDLU-val

---
## 1974 - 1980: "AI tél"

Az 1970-es évek közepére nyilvánvalóvá vált, hogy az 'intelligens gépek' létrehozásának bonyolultságát alábecsülték, és az ígéreteket, tekintettel a rendelkezésre álló számítástechnikai kapacitásra, túlértékelték. A finanszírozás megszűnt, és a terület iránti bizalom csökkent. Néhány probléma, amely befolyásolta a bizalmat:

---
- **Korlátok**. A számítástechnikai kapacitás túl korlátozott volt.
- **Kombinatorikus robbanás**. Az edzéshez szükséges paraméterek száma exponenciálisan nőtt, ahogy egyre többet vártak el a számítógépektől, anélkül, hogy a számítástechnikai kapacitás és képesség párhuzamosan fejlődött volna.
- **Adathiány**. Az adatok hiánya akadályozta az algoritmusok tesztelését, fejlesztését és finomítását.
- **A megfelelő kérdéseket tesszük fel?**. Az éppen feltett kérdéseket is elkezdték megkérdőjelezni. A kutatók kritikákat kaptak a megközelítéseikkel kapcsolatban:
  - A Turing-teszteket megkérdőjelezték többek között a 'kínai szoba elmélet' révén, amely azt állította, hogy "egy digitális számítógép programozása látszólag megértést mutathat, de nem képes valódi megértést produkálni." ([forrás](https://plato.stanford.edu/entries/chinese-room/))
  - Az olyan mesterséges intelligenciák, mint a "terapeuta" ELIZA társadalomba való bevezetésének etikáját megkérdőjelezték.

---

Ezzel egy időben különböző MI iskolák kezdtek kialakulni. Egy dichotómia jött létre ["scruffy" vs. "neat AI"](https://wikipedia.org/wiki/Neats_and_scruffies) gyakorlatok között. _Scruffy_ laborok órákig finomították a programokat, amíg el nem érték a kívánt eredményeket. _Neat_ laborok "a logikára és a formális problémamegoldásra" összpontosítottak. ELIZA és SHRDLU jól ismert _scruffy_ rendszerek voltak. Az 1980-as években, amikor igény mutatkozott a gépi tanulási rendszerek reprodukálhatóságára, a _neat_ megközelítés fokozatosan előtérbe került, mivel eredményei jobban magyarázhatók.

---
## 1980-as évek: Szakértői rendszerek

Ahogy a terület fejlődött, egyre világosabbá vált az üzleti haszna, és az 1980-as években elterjedtek a 'szakértői rendszerek'. "A szakértői rendszerek az első igazán sikeres mesterséges intelligencia (MI) szoftverformák közé tartoztak." ([forrás](https://wikipedia.org/wiki/Expert_system)).

Ez a rendszer valójában _hibrid_, részben egy szabálymotorból áll, amely meghatározza az üzleti követelményeket, és egy következtetési motorból, amely a szabályrendszert használja új tények levonására.

Ebben az időszakban a neurális hálók iránti figyelem is növekedett.

---
## 1987 - 1993: AI 'lehűlés'

A specializált szakértői rendszerek hardverének elterjedése sajnos túlságosan specializálttá vált. A személyi számítógépek térnyerése versenyre kelt ezekkel a nagy, specializált, központosított rendszerekkel. Elkezdődött a számítástechnika demokratizálása, amely végül utat nyitott a modern big data robbanásának.

---
## 1993 - 2011

Ez az időszak új korszakot hozott a gépi tanulás és MI számára, hogy megoldja azokat a problémákat, amelyeket korábban az adatok és számítástechnikai kapacitás hiánya okozott. Az adatok mennyisége gyorsan növekedni kezdett és szélesebb körben elérhetővé vált, jó és rossz értelemben egyaránt, különösen a 2007 körüli okostelefon megjelenésével. A számítástechnikai kapacitás exponenciálisan bővült, és az algoritmusok is fejlődtek. A terület kezdett éretté válni, ahogy a múlt szabad szellemű napjai egy valódi tudományággá kristályosodtak.

---
## Most

Ma a gépi tanulás és MI szinte minden részét érinti az életünknek. Ez a korszak gondos megértést igényel az algoritmusok emberi életre gyakorolt kockázatairól és potenciális hatásairól. Ahogy Brad Smith, a Microsoft egyik vezetője kijelentette: "Az információs technológia olyan kérdéseket vet fel, amelyek alapvető emberi jogi védelmek, például a magánélet és a véleménynyilvánítás szabadsága szívéhez vezetnek. Ezek a kérdések fokozzák a felelősséget a technológiai cégek számára, amelyek ezeket a termékeket létrehozzák. Véleményünk szerint ezek átgondolt kormányzati szabályozást és normák kidolgozását is igénylik az elfogadható felhasználások körül" ([forrás](https://www.technologyreview.com/2019/12/18/102365/the-future-of-ais-impact-on-society/)).

---

Még nem tudjuk, mit tartogat a jövő, de fontos megérteni ezeket a számítógépes rendszereket, valamint a szoftvereket és algoritmusokat, amelyeket futtatnak. Reméljük, hogy ez a tananyag segít jobban megérteni, hogy saját magad dönthess.

[![A mélytanulás története](https://img.youtube.com/vi/mTtDfKgLm54/0.jpg)](https://www.youtube.com/watch?v=mTtDfKgLm54 "A mélytanulás története")
> 🎥 Kattints a fenti képre egy videóért: Yann LeCun a mélytanulás történetéről beszél ebben az előadásban

---
## 🚀Kihívás

Merülj el az egyik történelmi pillanatban, és tudj meg többet az emberekről, akik mögötte állnak. Érdekes személyiségek vannak, és egyetlen tudományos felfedezés sem született kulturális vákuumban. Mit fedezel fel?

## [Előadás utáni kvíz](https://ff-quizzes.netlify.app/en/ml/)

---
## Áttekintés és önálló tanulás

Íme néhány néznivaló és hallgatnivaló:

[Ez a podcast, amelyben Amy Boyd az MI fejlődéséről beszél](http://runasradio.com/Shows/Show/739)

[![Az MI története Amy Boyd által](https://img.youtube.com/vi/EJt3_bFYKss/0.jpg)](https://www.youtube.com/watch?v=EJt3_bFYKss "Az MI története Amy Boyd által")

---

## Feladat

[Hozz létre egy idővonalat](assignment.md)

---

**Felelősség kizárása**:  
Ez a dokumentum az AI fordítási szolgáltatás, a [Co-op Translator](https://github.com/Azure/co-op-translator) segítségével lett lefordítva. Bár törekszünk a pontosságra, kérjük, vegye figyelembe, hogy az automatikus fordítások hibákat vagy pontatlanságokat tartalmazhatnak. Az eredeti dokumentum az eredeti nyelvén tekintendő hiteles forrásnak. Kritikus információk esetén javasolt professzionális emberi fordítást igénybe venni. Nem vállalunk felelősséget semmilyen félreértésért vagy téves értelmezésért, amely a fordítás használatából eredhet.