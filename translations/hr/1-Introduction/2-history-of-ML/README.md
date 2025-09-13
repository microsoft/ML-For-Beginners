<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "6a05fec147e734c3e6bfa54505648e2b",
  "translation_date": "2025-09-05T12:52:20+00:00",
  "source_file": "1-Introduction/2-history-of-ML/README.md",
  "language_code": "hr"
}
-->
# Povijest strojnog učenja

![Sažetak povijesti strojnog učenja u sketchnoteu](../../../../sketchnotes/ml-history.png)
> Sketchnote autorice [Tomomi Imura](https://www.twitter.com/girlie_mac)

## [Kviz prije predavanja](https://ff-quizzes.netlify.app/en/ml/)

---

[![Strojno učenje za početnike - Povijest strojnog učenja](https://img.youtube.com/vi/N6wxM4wZ7V0/0.jpg)](https://youtu.be/N6wxM4wZ7V0 "Strojno učenje za početnike - Povijest strojnog učenja")

> 🎥 Kliknite na sliku iznad za kratki video koji obrađuje ovu lekciju.

U ovoj lekciji proći ćemo kroz ključne prekretnice u povijesti strojnog učenja i umjetne inteligencije.

Povijest umjetne inteligencije (AI) kao područja usko je povezana s poviješću strojnog učenja, budući da su algoritmi i računalni napredak koji čine temelj ML-a pridonijeli razvoju AI-a. Važno je zapamtiti da, iako su se ova područja kao zasebne discipline počela kristalizirati 1950-ih, važna [algoritamska, statistička, matematička, računalna i tehnička otkrića](https://wikipedia.org/wiki/Timeline_of_machine_learning) prethodila su tom razdoblju i preklapala se s njim. Zapravo, ljudi razmišljaju o ovim pitanjima već [stotinama godina](https://wikipedia.org/wiki/History_of_artificial_intelligence): ovaj članak raspravlja o povijesnim intelektualnim temeljima ideje o 'stroju koji razmišlja'.

---
## Značajna otkrića

- 1763, 1812 [Bayesov teorem](https://wikipedia.org/wiki/Bayes%27_theorem) i njegovi prethodnici. Ovaj teorem i njegove primjene čine temelj zaključivanja, opisujući vjerojatnost događaja na temelju prethodnog znanja.
- 1805 [Teorija najmanjih kvadrata](https://wikipedia.org/wiki/Least_squares) francuskog matematičara Adriena-Marie Legendrea. Ova teorija, o kojoj ćete učiti u našoj jedinici o regresiji, pomaže u prilagodbi podataka.
- 1913 [Markovljevi lanci](https://wikipedia.org/wiki/Markov_chain), nazvani po ruskom matematičaru Andreju Markovu, koriste se za opisivanje slijeda mogućih događaja na temelju prethodnog stanja.
- 1957 [Perceptron](https://wikipedia.org/wiki/Perceptron) je vrsta linearnog klasifikatora koju je izumio američki psiholog Frank Rosenblatt, a koja čini temelj napretka u dubokom učenju.

---

- 1967 [Najbliži susjed](https://wikipedia.org/wiki/Nearest_neighbor) je algoritam izvorno dizajniran za mapiranje ruta. U kontekstu ML-a koristi se za otkrivanje uzoraka.
- 1970 [Backpropagation](https://wikipedia.org/wiki/Backpropagation) koristi se za treniranje [feedforward neuronskih mreža](https://wikipedia.org/wiki/Feedforward_neural_network).
- 1982 [Rekurentne neuronske mreže](https://wikipedia.org/wiki/Recurrent_neural_network) su umjetne neuronske mreže izvedene iz feedforward neuronskih mreža koje stvaraju vremenske grafove.

✅ Istražite malo. Koji drugi datumi se ističu kao ključni u povijesti ML-a i AI-a?

---
## 1950: Strojevi koji razmišljaju

Alan Turing, doista izvanredna osoba koju je [javnost 2019. godine](https://wikipedia.org/wiki/Icons:_The_Greatest_Person_of_the_20th_Century) proglasila najvećim znanstvenikom 20. stoljeća, zaslužan je za postavljanje temelja koncepta 'stroja koji može razmišljati'. Suočavao se s kritičarima i vlastitom potrebom za empirijskim dokazima ovog koncepta djelomično stvaranjem [Turingovog testa](https://www.bbc.com/news/technology-18475646), koji ćete istražiti u našim lekcijama o NLP-u.

---
## 1956: Ljetni istraživački projekt na Dartmouthu

"Ljetni istraživački projekt na Dartmouthu o umjetnoj inteligenciji bio je ključni događaj za umjetnu inteligenciju kao područje," i ovdje je skovan termin 'umjetna inteligencija' ([izvor](https://250.dartmouth.edu/highlights/artificial-intelligence-ai-coined-dartmouth)).

> Svaki aspekt učenja ili bilo koja druga značajka inteligencije može se u načelu tako precizno opisati da se stroj može napraviti da ga simulira.

---

Glavni istraživač, profesor matematike John McCarthy, nadao se "nastaviti na temelju pretpostavke da se svaki aspekt učenja ili bilo koja druga značajka inteligencije može u načelu tako precizno opisati da se stroj može napraviti da ga simulira." Sudionici su uključivali još jednu istaknutu osobu u ovom području, Marvina Minskyja.

Radionica je zaslužna za pokretanje i poticanje nekoliko rasprava, uključujući "uspon simboličkih metoda, sustava usmjerenih na ograničene domene (rani ekspertni sustavi) i deduktivnih sustava naspram induktivnih sustava." ([izvor](https://wikipedia.org/wiki/Dartmouth_workshop)).

---
## 1956 - 1974: "Zlatne godine"

Od 1950-ih do sredine '70-ih, optimizam je bio velik u nadi da AI može riješiti mnoge probleme. Godine 1967. Marvin Minsky je samouvjereno izjavio: "Unutar jedne generacije ... problem stvaranja 'umjetne inteligencije' bit će u velikoj mjeri riješen." (Minsky, Marvin (1967), Computation: Finite and Infinite Machines, Englewood Cliffs, N.J.: Prentice-Hall)

Istraživanje obrade prirodnog jezika je cvjetalo, pretraživanje je usavršeno i postalo moćnije, a stvoren je koncept 'mikro-svjetova', gdje su jednostavni zadaci izvršavani koristeći upute u običnom jeziku.

---

Istraživanje je bilo dobro financirano od strane vladinih agencija, postignut je napredak u računalstvu i algoritmima, a prototipovi inteligentnih strojeva su izgrađeni. Neki od tih strojeva uključuju:

* [Shakey robot](https://wikipedia.org/wiki/Shakey_the_robot), koji se mogao kretati i odlučivati kako inteligentno obavljati zadatke.

    ![Shakey, inteligentni robot](../../../../1-Introduction/2-history-of-ML/images/shakey.jpg)
    > Shakey 1972. godine

---

* Eliza, rani 'chatterbot', mogla je razgovarati s ljudima i djelovati kao primitivni 'terapeut'. Više o Elizi ćete naučiti u lekcijama o NLP-u.

    ![Eliza, bot](../../../../1-Introduction/2-history-of-ML/images/eliza.png)
    > Verzija Elize, chatbot

---

* "Blocks world" bio je primjer mikro-svijeta gdje su se blokovi mogli slagati i sortirati, a eksperimenti u podučavanju strojeva donošenju odluka mogli su se testirati. Napredak postignut s bibliotekama poput [SHRDLU](https://wikipedia.org/wiki/SHRDLU) pomogao je u napretku obrade jezika.

    [![blocks world sa SHRDLU](https://img.youtube.com/vi/QAJz4YKUwqw/0.jpg)](https://www.youtube.com/watch?v=QAJz4YKUwqw "blocks world sa SHRDLU")

    > 🎥 Kliknite na sliku iznad za video: Blocks world sa SHRDLU

---
## 1974 - 1980: "Zima AI-a"

Do sredine 1970-ih postalo je jasno da je složenost stvaranja 'inteligentnih strojeva' bila podcijenjena i da je njezino obećanje, s obzirom na dostupnu računalnu snagu, bilo precijenjeno. Financiranje je presušilo, a povjerenje u ovo područje usporilo. Neki problemi koji su utjecali na povjerenje uključuju:
---
- **Ograničenja**. Računalna snaga bila je previše ograničena.
- **Kombinatorna eksplozija**. Broj parametara koje je trebalo trenirati eksponencijalno je rastao kako se od računala tražilo više, bez paralelnog razvoja računalne snage i sposobnosti.
- **Nedostatak podataka**. Nedostatak podataka otežavao je proces testiranja, razvoja i usavršavanja algoritama.
- **Postavljamo li prava pitanja?**. Sama pitanja koja su se postavljala počela su se dovoditi u pitanje. Istraživači su se suočavali s kritikama svojih pristupa:
  - Turingovi testovi dovedeni su u pitanje, između ostalog, teorijom 'kineske sobe' koja je tvrdila da "programiranje digitalnog računala može učiniti da izgleda kao da razumije jezik, ali ne može proizvesti stvarno razumijevanje." ([izvor](https://plato.stanford.edu/entries/chinese-room/))
  - Etika uvođenja umjetnih inteligencija poput "terapeuta" ELIZE u društvo bila je izazvana.

---

Istovremeno, počele su se formirati različite škole mišljenja o AI-u. Uspostavljena je dihotomija između praksi ["neurednog" i "urednog AI-a"](https://wikipedia.org/wiki/Neats_and_scruffies). _Neuredni_ laboratoriji su satima prilagođavali programe dok nisu postigli željene rezultate. _Uredni_ laboratoriji "fokusirali su se na logiku i formalno rješavanje problema". ELIZA i SHRDLU bili su poznati _neuredni_ sustavi. U 1980-ima, kako se pojavila potreba za reproducibilnošću ML sustava, _uredni_ pristup postupno je preuzeo primat jer su njegovi rezultati bili objašnjiviji.

---
## 1980-e Ekspertni sustavi

Kako je područje raslo, njegova korist za poslovanje postala je jasnija, a 1980-ih došlo je do proliferacije 'ekspertnih sustava'. "Ekspertni sustavi bili su među prvim zaista uspješnim oblicima softvera umjetne inteligencije (AI)." ([izvor](https://wikipedia.org/wiki/Expert_system)).

Ova vrsta sustava zapravo je _hibridna_, djelomično se sastoji od sustava pravila koji definira poslovne zahtjeve i sustava zaključivanja koji koristi sustav pravila za izvođenje novih činjenica.

Ovo razdoblje također je donijelo sve veći interes za neuronske mreže.

---
## 1987 - 1993: AI 'hladnoća'

Proliferacija specijaliziranog hardvera za ekspertne sustave imala je nesretan učinak prevelike specijalizacije. Pojava osobnih računala također je konkurirala ovim velikim, specijaliziranim, centraliziranim sustavima. Demokratizacija računalstva je započela i na kraju otvorila put za modernu eksploziju velikih podataka.

---
## 1993 - 2011

Ovo razdoblje donijelo je novu eru za ML i AI kako bi mogli riješiti neke probleme uzrokovane ranije nedostatkom podataka i računalne snage. Količina podataka počela je brzo rasti i postajati dostupnija, na bolje i na gore, posebno s pojavom pametnog telefona oko 2007. godine. Računalna snaga eksponencijalno se povećala, a algoritmi su se razvijali paralelno. Područje je počelo sazrijevati kako su slobodniji dani prošlosti počeli kristalizirati u pravu disciplinu.

---
## Sada

Danas strojno učenje i AI dodiruju gotovo svaki dio našeg života. Ovo razdoblje zahtijeva pažljivo razumijevanje rizika i potencijalnih učinaka ovih algoritama na ljudske živote. Kako je izjavio Brad Smith iz Microsofta: "Informacijska tehnologija postavlja pitanja koja se tiču temeljnih zaštita ljudskih prava poput privatnosti i slobode izražavanja. Ova pitanja povećavaju odgovornost tehnoloških tvrtki koje stvaraju ove proizvode. Po našem mišljenju, također pozivaju na promišljenu regulaciju vlade i razvoj normi oko prihvatljivih upotreba" ([izvor](https://www.technologyreview.com/2019/12/18/102365/the-future-of-ais-impact-on-society/)).

---

Ostaje vidjeti što budućnost donosi, ali važno je razumjeti ove računalne sustave i softver te algoritme koje pokreću. Nadamo se da će vam ovaj kurikulum pomoći da steknete bolje razumijevanje kako biste sami mogli donijeti odluke.

[![Povijest dubokog učenja](https://img.youtube.com/vi/mTtDfKgLm54/0.jpg)](https://www.youtube.com/watch?v=mTtDfKgLm54 "Povijest dubokog učenja")
> 🎥 Kliknite na sliku iznad za video: Yann LeCun raspravlja o povijesti dubokog učenja u ovom predavanju

---
## 🚀Izazov

Zaronite u jedan od ovih povijesnih trenutaka i saznajte više o ljudima koji stoje iza njih. Postoje fascinantni likovi, a nijedno znanstveno otkriće nikada nije nastalo u kulturnom vakuumu. Što otkrivate?

## [Kviz nakon predavanja](https://ff-quizzes.netlify.app/en/ml/)

---
## Pregled i samostalno učenje

Evo stavki za gledanje i slušanje:

[Ovaj podcast u kojem Amy Boyd raspravlja o evoluciji AI-a](http://runasradio.com/Shows/Show/739)

[![Povijest AI-a od Amy Boyd](https://img.youtube.com/vi/EJt3_bFYKss/0.jpg)](https://www.youtube.com/watch?v=EJt3_bFYKss "Povijest AI-a od Amy Boyd")

---

## Zadatak

[Izradite vremensku crtu](assignment.md)

---

**Odricanje od odgovornosti**:  
Ovaj dokument je preveden korištenjem AI usluge za prevođenje [Co-op Translator](https://github.com/Azure/co-op-translator). Iako nastojimo osigurati točnost, imajte na umu da automatski prijevodi mogu sadržavati pogreške ili netočnosti. Izvorni dokument na izvornom jeziku treba smatrati mjerodavnim izvorom. Za ključne informacije preporučuje se profesionalni prijevod od strane stručnjaka. Ne preuzimamo odgovornost za bilo kakva nesporazuma ili pogrešna tumačenja koja mogu proizaći iz korištenja ovog prijevoda.