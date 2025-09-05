<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "6a05fec147e734c3e6bfa54505648e2b",
  "translation_date": "2025-09-05T12:53:15+00:00",
  "source_file": "1-Introduction/2-history-of-ML/README.md",
  "language_code": "sl"
}
-->
# Zgodovina strojnega učenja

![Povzetek zgodovine strojnega učenja v sketchnote](../../../../sketchnotes/ml-history.png)
> Sketchnote avtorja [Tomomi Imura](https://www.twitter.com/girlie_mac)

## [Predhodni kviz](https://ff-quizzes.netlify.app/en/ml/)

---

[![ML za začetnike - Zgodovina strojnega učenja](https://img.youtube.com/vi/N6wxM4wZ7V0/0.jpg)](https://youtu.be/N6wxM4wZ7V0 "ML za začetnike - Zgodovina strojnega učenja")

> 🎥 Kliknite zgornjo sliko za kratek video, ki obravnava to lekcijo.

V tej lekciji bomo pregledali glavne mejnike v zgodovini strojnega učenja in umetne inteligence.

Zgodovina umetne inteligence (UI) kot področja je tesno povezana z zgodovino strojnega učenja, saj so algoritmi in računalniški napredki, ki podpirajo strojno učenje, prispevali k razvoju UI. Pomembno je vedeti, da so se ta področja kot ločeni raziskovalni smeri začela oblikovati v 50. letih prejšnjega stoletja, vendar so pomembna [algoritmična, statistična, matematična, računalniška in tehnična odkritja](https://wikipedia.org/wiki/Timeline_of_machine_learning) predhodila in prekrivala to obdobje. Pravzaprav ljudje razmišljajo o teh vprašanjih že [stoletja](https://wikipedia.org/wiki/History_of_artificial_intelligence): ta članek obravnava zgodovinske intelektualne temelje ideje o 'mislečem stroju.'

---
## Pomembna odkritja

- 1763, 1812 [Bayesov izrek](https://wikipedia.org/wiki/Bayes%27_theorem) in njegovi predhodniki. Ta izrek in njegove aplikacije so osnova za sklepanje, saj opisujejo verjetnost dogodka na podlagi predhodnega znanja.
- 1805 [Teorija najmanjših kvadratov](https://wikipedia.org/wiki/Least_squares) francoskega matematika Adriena-Marieja Legendra. Ta teorija, o kateri se boste učili v enoti o regresiji, pomaga pri prilagajanju podatkov.
- 1913 [Markovske verige](https://wikipedia.org/wiki/Markov_chain), poimenovane po ruskem matematiku Andreju Markovu, opisujejo zaporedje možnih dogodkov na podlagi prejšnjega stanja.
- 1957 [Perceptron](https://wikipedia.org/wiki/Perceptron) je vrsta linearnega klasifikatorja, ki ga je izumil ameriški psiholog Frank Rosenblatt in je osnova za napredek v globokem učenju.

---

- 1967 [Najbližji sosed](https://wikipedia.org/wiki/Nearest_neighbor) je algoritem, prvotno zasnovan za načrtovanje poti. V kontekstu strojnega učenja se uporablja za prepoznavanje vzorcev.
- 1970 [Povratno širjenje](https://wikipedia.org/wiki/Backpropagation) se uporablja za učenje [feedforward nevronskih mrež](https://wikipedia.org/wiki/Feedforward_neural_network).
- 1982 [Rekurentne nevronske mreže](https://wikipedia.org/wiki/Recurrent_neural_network) so umetne nevronske mreže, izpeljane iz feedforward mrež, ki ustvarjajo časovne grafe.

✅ Raziščite. Kateri drugi datumi so po vašem mnenju ključni v zgodovini strojnega učenja in umetne inteligence?

---
## 1950: Stroji, ki mislijo

Alan Turing, resnično izjemna oseba, ki je bil [leta 2019](https://wikipedia.org/wiki/Icons:_The_Greatest_Person_of_the_20th_Century) izbran za največjega znanstvenika 20. stoletja, je zaslužen za postavitev temeljev koncepta 'stroja, ki lahko misli.' Soočen je bil s skeptiki in lastno potrebo po empiričnih dokazih tega koncepta, deloma z ustvarjanjem [Turingovega testa](https://www.bbc.com/news/technology-18475646), ki ga boste raziskali v lekcijah o obdelavi naravnega jezika.

---
## 1956: Poletni raziskovalni projekt na Dartmouthu

"Poletni raziskovalni projekt o umetni inteligenci na Dartmouthu je bil ključen dogodek za umetno inteligenco kot področje," in prav tukaj je bil skovan izraz 'umetna inteligenca' ([vir](https://250.dartmouth.edu/highlights/artificial-intelligence-ai-coined-dartmouth)).

> Vsak vidik učenja ali katera koli druga značilnost inteligence je načeloma mogoče tako natančno opisati, da jo lahko stroj simulira.

---

Vodja raziskave, profesor matematike John McCarthy, je upal, "da bo mogoče nadaljevati na podlagi domneve, da je vsak vidik učenja ali katera koli druga značilnost inteligence načeloma mogoče tako natančno opisati, da jo lahko stroj simulira." Med udeleženci je bil tudi drug pomemben raziskovalec na tem področju, Marvin Minsky.

Delavnica je zaslužna za spodbujanje več razprav, vključno z "vzponom simboličnih metod, sistemov, osredotočenih na omejena področja (zgodnji ekspertni sistemi), in deduktivnih sistemov v primerjavi z induktivnimi sistemi." ([vir](https://wikipedia.org/wiki/Dartmouth_workshop)).

---
## 1956 - 1974: "Zlata leta"

Od 50. let do sredine 70. let je vladal optimizem, da bi UI lahko rešila številne težave. Leta 1967 je Marvin Minsky samozavestno izjavil: "V eni generaciji ... bo problem ustvarjanja 'umetne inteligence' v veliki meri rešen." (Minsky, Marvin (1967), Computation: Finite and Infinite Machines, Englewood Cliffs, N.J.: Prentice-Hall)

Raziskave obdelave naravnega jezika so cvetele, iskanje je postalo bolj izpopolnjeno in zmogljivo, ter ustvarjen je bil koncept 'mikro-svetov', kjer so bile preproste naloge izvedene z uporabo navodil v preprostem jeziku.

---

Raziskave so bile dobro financirane s strani vladnih agencij, napredek je bil dosežen na področju računalništva in algoritmov, ter prototipi inteligentnih strojev so bili zgrajeni. Nekateri od teh strojev vključujejo:

* [Robot Shakey](https://wikipedia.org/wiki/Shakey_the_robot), ki se je lahko premikal in 'inteligentno' odločal, kako opraviti naloge.

    ![Shakey, inteligentni robot](../../../../1-Introduction/2-history-of-ML/images/shakey.jpg)
    > Shakey leta 1972

---

* Eliza, zgodnji 'klepetalni robot', je lahko komunicirala z ljudmi in delovala kot primitivni 'terapevt'. Več o Elizi boste izvedeli v lekcijah o obdelavi naravnega jezika.

    ![Eliza, bot](../../../../1-Introduction/2-history-of-ML/images/eliza.png)
    > Različica Elize, klepetalnega robota

---

* "Svet blokov" je bil primer mikro-sveta, kjer so se bloki lahko zlagali in razvrščali, ter so se izvajali eksperimenti pri učenju strojev za sprejemanje odločitev. Napredki, doseženi z knjižnicami, kot je [SHRDLU](https://wikipedia.org/wiki/SHRDLU), so pomagali pri razvoju obdelave jezika.

    [![svet blokov s SHRDLU](https://img.youtube.com/vi/QAJz4YKUwqw/0.jpg)](https://www.youtube.com/watch?v=QAJz4YKUwqw "svet blokov s SHRDLU")

    > 🎥 Kliknite zgornjo sliko za video: Svet blokov s SHRDLU

---
## 1974 - 1980: "Zima UI"

Do sredine 70. let je postalo jasno, da je bila kompleksnost ustvarjanja 'inteligentnih strojev' podcenjena in da so bile obljube, glede na razpoložljivo računalniško moč, pretirane. Financiranje se je zmanjšalo, zaupanje v področje pa je upadlo. Nekateri problemi, ki so vplivali na zaupanje, vključujejo:
---
- **Omejitve**. Računalniška moč je bila prešibka.
- **Kombinatorna eksplozija**. Število parametrov, ki jih je bilo treba naučiti, je eksponentno naraščalo, brez vzporednega razvoja računalniške moči in zmogljivosti.
- **Pomanjkanje podatkov**. Pomanjkanje podatkov je oviralo proces testiranja, razvoja in izboljševanja algoritmov.
- **Ali postavljamo prava vprašanja?**. Začela so se postavljati vprašanja o samih vprašanjih, ki so jih raziskovalci zastavljali:
  - Turingovi testi so bili postavljeni pod vprašaj, med drugim tudi s teorijo 'kitajske sobe', ki je trdila, da "programiranje digitalnega računalnika lahko ustvari videz razumevanja jezika, vendar ne more ustvariti pravega razumevanja." ([vir](https://plato.stanford.edu/entries/chinese-room/))
  - Etika uvajanja umetnih inteligenc, kot je "terapevt" ELIZA, v družbo je bila izzvana.

---

Hkrati so se začele oblikovati različne šole misli o UI. Ustanovila se je dihotomija med ["neurejeno" in "urejeno UI"](https://wikipedia.org/wiki/Neats_and_scruffies). _Neurejeni_ laboratoriji so urejali programe, dokler niso dosegli želenih rezultatov. _Urejeni_ laboratoriji so se osredotočali na logiko in formalno reševanje problemov. ELIZA in SHRDLU sta bila znana _neurejena_ sistema. V 80. letih, ko se je pojavilo povpraševanje po reproducibilnosti sistemov strojnega učenja, je _urejen_ pristop postopoma prevladal, saj so njegovi rezultati bolj razložljivi.

---
## 1980: Ekspertni sistemi

Z rastjo področja je postala njegova korist za poslovanje bolj očitna, v 80. letih pa se je razširila uporaba 'ekspertnih sistemov'. "Ekspertni sistemi so bili med prvimi resnično uspešnimi oblikami programske opreme umetne inteligence (UI)." ([vir](https://wikipedia.org/wiki/Expert_system)).

Ta vrsta sistema je pravzaprav _hibridna_, saj delno vključuje pravila, ki določajo poslovne zahteve, in sklepni mehanizem, ki uporablja sistem pravil za sklepanje novih dejstev.

To obdobje je prineslo tudi večjo pozornost nevronskim mrežam.

---
## 1987 - 1993: Ohladitev UI

Razširitev specializirane strojne opreme za ekspertne sisteme je imela nesrečen učinek, da je postala preveč specializirana. Pojav osebnih računalnikov je prav tako tekmoval s temi velikimi, specializiranimi, centraliziranimi sistemi. Začela se je demokratizacija računalništva, ki je sčasoma tlakovala pot za sodobno eksplozijo velikih podatkov.

---
## 1993 - 2011

To obdobje je prineslo novo ero za strojno učenje in umetno inteligenco, da bi lahko rešila nekatere težave, ki so jih povzročili pomanjkanje podatkov in računalniške moči. Količina podatkov se je začela hitro povečevati in postajati bolj dostopna, tako v dobrem kot slabem, še posebej z uvedbo pametnega telefona okoli leta 2007. Računalniška moč se je eksponentno povečala, algoritmi pa so se razvijali vzporedno. Področje je začelo dosegati zrelost, saj so se svobodomiselni dnevi preteklosti začeli oblikovati v pravo disciplino.

---
## Danes

Danes strojno učenje in umetna inteligenca vplivata na skoraj vsak del našega življenja. To obdobje zahteva skrbno razumevanje tveganj in možnih učinkov teh algoritmov na človeška življenja. Kot je dejal Brad Smith iz Microsofta: "Informacijska tehnologija odpira vprašanja, ki segajo v samo srčiko temeljnih zaščit človekovih pravic, kot sta zasebnost in svoboda izražanja. Ta vprašanja povečujejo odgovornost tehnoloških podjetij, ki ustvarjajo te izdelke. Po našem mnenju zahtevajo tudi premišljeno vladno regulacijo in razvoj norm glede sprejemljive uporabe" ([vir](https://www.technologyreview.com/2019/12/18/102365/the-future-of-ais-impact-on-society/)).

---

Kaj prinaša prihodnost, ostaja neznano, vendar je pomembno razumeti te računalniške sisteme ter programsko opremo in algoritme, ki jih poganjajo. Upamo, da vam bo ta učni načrt pomagal pridobiti boljše razumevanje, da boste lahko sami presodili.

[![Zgodovina globokega učenja](https://img.youtube.com/vi/mTtDfKgLm54/0.jpg)](https://www.youtube.com/watch?v=mTtDfKgLm54 "Zgodovina globokega učenja")
> 🎥 Kliknite zgornjo sliko za video: Yann LeCun razpravlja o zgodovini globokega učenja v tem predavanju

---
## 🚀Izziv

Poglobite se v enega od teh zgodovinskih trenutkov in izvedite več o ljudeh, ki stojijo za njimi. Obstajajo fascinantni liki, in nobeno znanstveno odkritje ni bilo ustvarjeno v kulturnem vakuumu. Kaj odkrijete?

## [Kviz po predavanju](https://ff-quizzes.netlify.app/en/ml/)

---
## Pregled in samostojno učenje

Tukaj so predmeti za ogled in poslušanje:

[Ta podcast, kjer Amy Boyd razpravlja o razvoju UI](http://runasradio.com/Shows/Show/739)

[![Zgodovina UI avtorice Amy Boyd](https://img.youtube.com/vi/EJt3_bFYKss/0.jpg)](https://www.youtube.com/watch?v=EJt3_bFYKss "Zgodovina UI avtorice Amy Boyd")

---

## Naloga

[Ustvarite časovnico](assignment.md)

---

**Omejitev odgovornosti**:  
Ta dokument je bil preveden z uporabo storitve za strojno prevajanje [Co-op Translator](https://github.com/Azure/co-op-translator). Čeprav si prizadevamo za natančnost, vas prosimo, da upoštevate, da lahko avtomatizirani prevodi vsebujejo napake ali netočnosti. Izvirni dokument v njegovem izvirnem jeziku je treba obravnavati kot avtoritativni vir. Za ključne informacije priporočamo strokovno človeško prevajanje. Ne prevzemamo odgovornosti za morebitna nesporazumevanja ali napačne razlage, ki izhajajo iz uporabe tega prevoda.