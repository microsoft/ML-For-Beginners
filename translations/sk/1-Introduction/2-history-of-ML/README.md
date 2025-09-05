<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "6a05fec147e734c3e6bfa54505648e2b",
  "translation_date": "2025-09-05T16:10:01+00:00",
  "source_file": "1-Introduction/2-history-of-ML/README.md",
  "language_code": "sk"
}
-->
# História strojového učenia

![Zhrnutie histórie strojového učenia v sketchnote](../../../../sketchnotes/ml-history.png)  
> Sketchnote od [Tomomi Imura](https://www.twitter.com/girlie_mac)

## [Kvíz pred prednáškou](https://ff-quizzes.netlify.app/en/ml/)

---

[![ML pre začiatočníkov - História strojového učenia](https://img.youtube.com/vi/N6wxM4wZ7V0/0.jpg)](https://youtu.be/N6wxM4wZ7V0 "ML pre začiatočníkov - História strojového učenia")

> 🎥 Kliknite na obrázok vyššie pre krátke video k tejto lekcii.

V tejto lekcii si prejdeme hlavné míľniky v histórii strojového učenia a umelej inteligencie.

História umelej inteligencie (AI) ako odboru je úzko prepojená s históriou strojového učenia, pretože algoritmy a výpočtové pokroky, ktoré sú základom strojového učenia, prispeli k rozvoju umelej inteligencie. Je užitočné si uvedomiť, že hoci sa tieto oblasti ako samostatné disciplíny začali formovať v 50. rokoch 20. storočia, dôležité [algoritmické, štatistické, matematické, výpočtové a technické objavy](https://wikipedia.org/wiki/Timeline_of_machine_learning) predchádzali a prekrývali toto obdobie. V skutočnosti sa ľudia zaoberali týmito otázkami už [stovky rokov](https://wikipedia.org/wiki/History_of_artificial_intelligence): tento článok sa zaoberá historickými intelektuálnymi základmi myšlienky „mysliaceho stroja“.

---
## Významné objavy

- 1763, 1812 [Bayesova veta](https://wikipedia.org/wiki/Bayes%27_theorem) a jej predchodcovia. Táto veta a jej aplikácie sú základom inferencie, opisujú pravdepodobnosť výskytu udalosti na základe predchádzajúcich znalostí.
- 1805 [Metóda najmenších štvorcov](https://wikipedia.org/wiki/Least_squares) od francúzskeho matematika Adriena-Marieho Legendra. Táto metóda, o ktorej sa dozviete v našej jednotke o regresii, pomáha pri prispôsobovaní údajov.
- 1913 [Markovove reťazce](https://wikipedia.org/wiki/Markov_chain), pomenované po ruskom matematikovi Andrejovi Markovovi, sa používajú na opis sekvencie možných udalostí na základe predchádzajúceho stavu.
- 1957 [Perceptron](https://wikipedia.org/wiki/Perceptron) je typ lineárneho klasifikátora, ktorý vynašiel americký psychológ Frank Rosenblatt a ktorý je základom pokrokov v hlbokom učení.

---

- 1967 [Najbližší sused](https://wikipedia.org/wiki/Nearest_neighbor) je algoritmus pôvodne navrhnutý na mapovanie trás. V kontexte strojového učenia sa používa na detekciu vzorov.
- 1970 [Backpropagation](https://wikipedia.org/wiki/Backpropagation) sa používa na trénovanie [dopredných neurónových sietí](https://wikipedia.org/wiki/Feedforward_neural_network).
- 1982 [Rekurentné neurónové siete](https://wikipedia.org/wiki/Recurrent_neural_network) sú umelé neurónové siete odvodené od dopredných neurónových sietí, ktoré vytvárajú časové grafy.

✅ Urobte si malý prieskum. Ktoré ďalšie dátumy považujete za kľúčové v histórii strojového učenia a umelej inteligencie?

---
## 1950: Stroje, ktoré myslia

Alan Turing, skutočne výnimočný človek, ktorý bol [verejnosťou v roku 2019](https://wikipedia.org/wiki/Icons:_The_Greatest_Person_of_the_20th_Century) zvolený za najväčšieho vedca 20. storočia, je považovaný za osobu, ktorá pomohla položiť základy konceptu „stroja, ktorý dokáže myslieť“. Zaoberal sa skeptikmi a potrebou empirických dôkazov o tomto koncepte, okrem iného vytvorením [Turingovho testu](https://www.bbc.com/news/technology-18475646), ktorý preskúmate v našich lekciách o spracovaní prirodzeného jazyka.

---
## 1956: Letný výskumný projekt v Dartmouthe

"Letný výskumný projekt o umelej inteligencii v Dartmouthe bol pre umelú inteligenciu ako odbor zásadnou udalosťou," a práve tu bol zavedený pojem „umelá inteligencia“ ([zdroj](https://250.dartmouth.edu/highlights/artificial-intelligence-ai-coined-dartmouth)).

> Každý aspekt učenia alebo akejkoľvek inej črty inteligencie môže byť v princípe tak presne opísaný, že stroj môže byť vytvorený na jeho simuláciu.

---

Vedúci výskumník, profesor matematiky John McCarthy, dúfal, že "bude pokračovať na základe hypotézy, že každý aspekt učenia alebo akejkoľvek inej črty inteligencie môže byť v princípe tak presne opísaný, že stroj môže byť vytvorený na jeho simuláciu." Medzi účastníkmi bol aj ďalší významný predstaviteľ odboru, Marvin Minsky.

Workshop je považovaný za iniciátora a podporovateľa viacerých diskusií vrátane "vzostupu symbolických metód, systémov zameraných na obmedzené oblasti (rané expertné systémy) a deduktívnych systémov oproti induktívnym systémom." ([zdroj](https://wikipedia.org/wiki/Dartmouth_workshop)).

---
## 1956 - 1974: "Zlaté roky"

Od 50. rokov do polovice 70. rokov vládol optimizmus, že AI dokáže vyriešiť mnoho problémov. V roku 1967 Marvin Minsky sebavedome vyhlásil: "Do jednej generácie ... bude problém vytvorenia 'umelej inteligencie' podstatne vyriešený." (Minsky, Marvin (1967), Computation: Finite and Infinite Machines, Englewood Cliffs, N.J.: Prentice-Hall)

Výskum spracovania prirodzeného jazyka prekvital, vyhľadávanie bolo zdokonalené a zefektívnené a vznikol koncept „mikrosvetov“, kde sa jednoduché úlohy vykonávali pomocou jednoduchých jazykových inštrukcií.

---

Výskum bol dobre financovaný vládnymi agentúrami, dosiahli sa pokroky vo výpočtoch a algoritmoch a boli postavené prototypy inteligentných strojov. Niektoré z týchto strojov zahŕňajú:

* [Shakey robot](https://wikipedia.org/wiki/Shakey_the_robot), ktorý sa dokázal pohybovať a rozhodovať, ako vykonávať úlohy „inteligentne“.

    ![Shakey, inteligentný robot](../../../../1-Introduction/2-history-of-ML/images/shakey.jpg)  
    > Shakey v roku 1972

---

* Eliza, raný „chatterbot“, dokázala komunikovať s ľuďmi a pôsobiť ako primitívny „terapeut“. O Elize sa dozviete viac v lekciách o spracovaní prirodzeného jazyka.

    ![Eliza, bot](../../../../1-Introduction/2-history-of-ML/images/eliza.png)  
    > Verzia Elizy, chatbot

---

* „Blocks world“ bol príklad mikrosveta, kde sa mohli bloky stohovať a triediť a mohli sa testovať experimenty v učení strojov rozhodovať. Pokroky dosiahnuté s knižnicami ako [SHRDLU](https://wikipedia.org/wiki/SHRDLU) pomohli posunúť spracovanie jazyka vpred.

    [![blocks world so SHRDLU](https://img.youtube.com/vi/QAJz4YKUwqw/0.jpg)](https://www.youtube.com/watch?v=QAJz4YKUwqw "blocks world so SHRDLU")

    > 🎥 Kliknite na obrázok vyššie pre video: Blocks world so SHRDLU

---
## 1974 - 1980: "Zima AI"

Do polovice 70. rokov sa ukázalo, že zložitosť vytvárania „inteligentných strojov“ bola podcenená a jej sľuby, vzhľadom na dostupný výpočtový výkon, boli prehnané. Financovanie vyschlo a dôvera v odbor sa spomalila. Niektoré problémy, ktoré ovplyvnili dôveru, zahŕňali:  
---
- **Obmedzenia**. Výpočtový výkon bol príliš obmedzený.  
- **Kombinatorická explózia**. Počet parametrov potrebných na trénovanie rástol exponenciálne, keď sa od počítačov žiadalo viac, bez paralelného vývoja výpočtového výkonu a schopností.  
- **Nedostatok údajov**. Nedostatok údajov bránil procesu testovania, vývoja a zdokonaľovania algoritmov.  
- **Pýtame sa správne otázky?**. Samotné otázky, ktoré boli kladené, sa začali spochybňovať. Výskumníci čelili kritike svojich prístupov:  
  - Turingove testy boli spochybnené napríklad teóriou „čínskej miestnosti“, ktorá tvrdila, že „naprogramovanie digitálneho počítača môže vytvoriť zdanie porozumenia jazyka, ale nemôže vytvoriť skutočné porozumenie.“ ([zdroj](https://plato.stanford.edu/entries/chinese-room/))  
  - Etika zavádzania umelej inteligencie, ako napríklad „terapeuta“ ELIZA, do spoločnosti bola spochybnená.  

---

Zároveň sa začali formovať rôzne školy myslenia v oblasti umelej inteligencie. Vznikla dichotómia medzi prístupmi ["scruffy" a "neat AI"](https://wikipedia.org/wiki/Neats_and_scruffies). _Scruffy_ laboratóriá dolaďovali programy, kým nedosiahli požadované výsledky. _Neat_ laboratóriá sa zameriavali na logiku a formálne riešenie problémov. ELIZA a SHRDLU boli známe _scruffy_ systémy. V 80. rokoch, keď vznikol dopyt po reprodukovateľnosti systémov strojového učenia, sa _neat_ prístup postupne dostal do popredia, pretože jeho výsledky sú vysvetliteľnejšie.

---
## 1980s Expertné systémy

Ako odbor rástol, jeho prínos pre podnikanie sa stal zreteľnejším, a v 80. rokoch sa rozšírili „expertné systémy“. „Expertné systémy patrili medzi prvé skutočne úspešné formy softvéru umelej inteligencie (AI).“ ([zdroj](https://wikipedia.org/wiki/Expert_system)).

Tento typ systému je vlastne _hybridný_, pozostávajúci čiastočne z pravidlového enginu definujúceho obchodné požiadavky a inferenčného enginu, ktorý využíval pravidlový systém na odvodenie nových faktov.

Toto obdobie tiež zaznamenalo rastúcu pozornosť venovanú neurónovým sieťam.

---
## 1987 - 1993: AI 'Chill'

Rozšírenie špecializovaného hardvéru expertných systémov malo nešťastný efekt, že sa stalo príliš špecializovaným. Vzostup osobných počítačov tiež konkuroval týmto veľkým, špecializovaným, centralizovaným systémom. Začala sa demokratizácia výpočtovej techniky, ktorá nakoniec pripravila cestu pre moderný výbuch veľkých dát.

---
## 1993 - 2011

Toto obdobie prinieslo novú éru pre strojové učenie a umelú inteligenciu, aby mohli vyriešiť niektoré problémy spôsobené nedostatkom údajov a výpočtového výkonu v minulosti. Množstvo údajov začalo rýchlo narastať a stávať sa dostupnejším, či už v dobrom alebo zlom, najmä s príchodom smartfónov okolo roku 2007. Výpočtový výkon exponenciálne rástol a algoritmy sa vyvíjali spolu s ním. Odbor začal dosahovať zrelosť, keď sa voľné dni minulosti začali kryštalizovať do skutočnej disciplíny.

---
## Dnes

Dnes strojové učenie a umelá inteligencia zasahujú takmer do každej časti nášho života. Toto obdobie si vyžaduje dôkladné pochopenie rizík a potenciálnych dopadov týchto algoritmov na ľudské životy. Ako uviedol Brad Smith z Microsoftu: „Informačné technológie otvárajú otázky, ktoré sa dotýkajú samotného jadra ochrany základných ľudských práv, ako je súkromie a sloboda prejavu. Tieto otázky zvyšujú zodpovednosť technologických spoločností, ktoré tieto produkty vytvárajú. Podľa nás si tiež vyžadujú premyslenú vládnu reguláciu a vývoj noriem o prijateľnom používaní“ ([zdroj](https://www.technologyreview.com/2019/12/18/102365/the-future-of-ais-impact-on-society/)).

---

Zostáva vidieť, čo prinesie budúcnosť, ale je dôležité pochopiť tieto počítačové systémy a softvér a algoritmy, ktoré používajú. Dúfame, že tento kurz vám pomôže získať lepšie pochopenie, aby ste si mohli vytvoriť vlastný názor.

[![História hlbokého učenia](https://img.youtube.com/vi/mTtDfKgLm54/0.jpg)](https://www.youtube.com/watch?v=mTtDfKgLm54 "História hlbokého učenia")  
> 🎥 Kliknite na obrázok vyššie pre video: Yann LeCun hovorí o histórii hlbokého učenia v tejto prednáške

---
## 🚀Výzva

Ponorte sa do jedného z týchto historických momentov a zistite viac o ľuďoch, ktorí za nimi stoja. Sú to fascinujúce osobnosti a žiadny vedecký objav nevznikol v kultúrnom vákuu. Čo objavíte?

## [Kvíz po prednáške](https://ff-quizzes.netlify.app/en/ml/)

---
## Prehľad a samoštúdium

Tu sú položky na sledovanie a počúvanie:

[Tento podcast, kde Amy Boyd diskutuje o vývoji umelej inteligencie](http://runasradio.com/Shows/Show/739)

[![História AI od Amy Boyd](https://img.youtube.com/vi/EJt3_bFYKss/0.jpg)](https://www.youtube.com/watch?v=EJt3_bFYKss "História AI od Amy Boyd")

---

## Zadanie

[Vytvorte časovú os](assignment.md)

---

**Upozornenie**:  
Tento dokument bol preložený pomocou služby AI prekladu [Co-op Translator](https://github.com/Azure/co-op-translator). Aj keď sa snažíme o presnosť, prosím, berte na vedomie, že automatizované preklady môžu obsahovať chyby alebo nepresnosti. Pôvodný dokument v jeho pôvodnom jazyku by mal byť považovaný za autoritatívny zdroj. Pre kritické informácie sa odporúča profesionálny ľudský preklad. Nie sme zodpovední za akékoľvek nedorozumenia alebo nesprávne interpretácie vyplývajúce z použitia tohto prekladu.