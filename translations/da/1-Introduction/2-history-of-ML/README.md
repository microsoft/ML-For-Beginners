<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "6a05fec147e734c3e6bfa54505648e2b",
  "translation_date": "2025-09-05T00:33:52+00:00",
  "source_file": "1-Introduction/2-history-of-ML/README.md",
  "language_code": "da"
}
-->
# Historien om maskinlæring

![Oversigt over historien om maskinlæring i en sketchnote](../../../../sketchnotes/ml-history.png)
> Sketchnote af [Tomomi Imura](https://www.twitter.com/girlie_mac)

## [Quiz før lektionen](https://ff-quizzes.netlify.app/en/ml/)

---

[![ML for begyndere - Historien om maskinlæring](https://img.youtube.com/vi/N6wxM4wZ7V0/0.jpg)](https://youtu.be/N6wxM4wZ7V0 "ML for begyndere - Historien om maskinlæring")

> 🎥 Klik på billedet ovenfor for en kort video, der gennemgår denne lektion.

I denne lektion vil vi gennemgå de vigtigste milepæle i historien om maskinlæring og kunstig intelligens.

Historien om kunstig intelligens (AI) som et felt er tæt forbundet med historien om maskinlæring, da de algoritmer og beregningsmæssige fremskridt, der ligger til grund for ML, har bidraget til udviklingen af AI. Det er nyttigt at huske, at selvom disse felter som særskilte forskningsområder begyndte at tage form i 1950'erne, så fandt vigtige [algoritmiske, statistiske, matematiske, beregningsmæssige og tekniske opdagelser](https://wikipedia.org/wiki/Timeline_of_machine_learning) sted før og overlappede denne periode. Faktisk har mennesker tænkt over disse spørgsmål i [hundredevis af år](https://wikipedia.org/wiki/History_of_artificial_intelligence): denne artikel diskuterer de historiske intellektuelle fundamenter for ideen om en 'tænkende maskine.'

---
## Bemærkelsesværdige opdagelser

- 1763, 1812 [Bayes' teorem](https://wikipedia.org/wiki/Bayes%27_theorem) og dets forgængere. Dette teorem og dets anvendelser ligger til grund for inferens og beskriver sandsynligheden for, at en begivenhed indtræffer baseret på tidligere viden.
- 1805 [Mindste kvadraters metode](https://wikipedia.org/wiki/Least_squares) af den franske matematiker Adrien-Marie Legendre. Denne metode, som du vil lære om i vores Regression-enhed, hjælper med dataanpasning.
- 1913 [Markov-kæder](https://wikipedia.org/wiki/Markov_chain), opkaldt efter den russiske matematiker Andrey Markov, bruges til at beskrive en sekvens af mulige begivenheder baseret på en tidligere tilstand.
- 1957 [Perceptron](https://wikipedia.org/wiki/Perceptron) er en type lineær klassifikator opfundet af den amerikanske psykolog Frank Rosenblatt, som ligger til grund for fremskridt inden for dyb læring.

---

- 1967 [Nærmeste nabo](https://wikipedia.org/wiki/Nearest_neighbor) er en algoritme oprindeligt designet til at kortlægge ruter. I en ML-kontekst bruges den til at opdage mønstre.
- 1970 [Backpropagation](https://wikipedia.org/wiki/Backpropagation) bruges til at træne [feedforward neurale netværk](https://wikipedia.org/wiki/Feedforward_neural_network).
- 1982 [Recurrent Neural Networks](https://wikipedia.org/wiki/Recurrent_neural_network) er kunstige neurale netværk afledt af feedforward neurale netværk, der skaber tidsmæssige grafer.

✅ Lav lidt research. Hvilke andre datoer skiller sig ud som afgørende i historien om ML og AI?

---
## 1950: Maskiner, der tænker

Alan Turing, en virkelig bemærkelsesværdig person, der blev kåret [af offentligheden i 2019](https://wikipedia.org/wiki/Icons:_The_Greatest_Person_of_the_20th_Century) som det 20. århundredes største videnskabsmand, krediteres for at have hjulpet med at lægge fundamentet for konceptet om en 'maskine, der kan tænke.' Han kæmpede med skeptikere og sit eget behov for empirisk evidens for dette koncept, blandt andet ved at skabe [Turing-testen](https://www.bbc.com/news/technology-18475646), som du vil udforske i vores NLP-lektioner.

---
## 1956: Dartmouth Summer Research Project

"Dartmouth Summer Research Project on artificial intelligence var en skelsættende begivenhed for kunstig intelligens som et felt," og det var her, begrebet 'kunstig intelligens' blev opfundet ([kilde](https://250.dartmouth.edu/highlights/artificial-intelligence-ai-coined-dartmouth)).

> Hver eneste aspekt af læring eller enhver anden egenskab ved intelligens kan i princippet beskrives så præcist, at en maskine kan laves til at simulere det.

---

Den ledende forsker, matematikprofessor John McCarthy, håbede "at kunne gå videre på baggrund af antagelsen om, at hver eneste aspekt af læring eller enhver anden egenskab ved intelligens kan i princippet beskrives så præcist, at en maskine kan laves til at simulere det." Deltagerne inkluderede en anden fremtrædende figur inden for feltet, Marvin Minsky.

Workshoppen krediteres for at have initieret og opmuntret flere diskussioner, herunder "fremkomsten af symbolske metoder, systemer fokuseret på begrænsede domæner (tidlige ekspertsystemer) og deduktive systemer versus induktive systemer." ([kilde](https://wikipedia.org/wiki/Dartmouth_workshop)).

---
## 1956 - 1974: "De gyldne år"

Fra 1950'erne til midten af 70'erne var der stor optimisme omkring håbet om, at AI kunne løse mange problemer. I 1967 udtalte Marvin Minsky selvsikkert, "Inden for en generation ... vil problemet med at skabe 'kunstig intelligens' i væsentlig grad være løst." (Minsky, Marvin (1967), Computation: Finite and Infinite Machines, Englewood Cliffs, N.J.: Prentice-Hall)

Forskning i naturlig sprogbehandling blomstrede, søgning blev raffineret og gjort mere kraftfuld, og konceptet 'mikroverdener' blev skabt, hvor simple opgaver blev udført ved hjælp af almindelige sprogkommandoer.

---

Forskning blev godt finansieret af statslige organer, fremskridt blev gjort inden for beregning og algoritmer, og prototyper af intelligente maskiner blev bygget. Nogle af disse maskiner inkluderer:

* [Shakey-robotten](https://wikipedia.org/wiki/Shakey_the_robot), som kunne manøvrere og beslutte, hvordan opgaver skulle udføres 'intelligent'.

    ![Shakey, en intelligent robot](../../../../1-Introduction/2-history-of-ML/images/shakey.jpg)
    > Shakey i 1972

---

* Eliza, en tidlig 'chatterbot', kunne samtale med mennesker og fungere som en primitiv 'terapeut'. Du vil lære mere om Eliza i NLP-lektionerne.

    ![Eliza, en bot](../../../../1-Introduction/2-history-of-ML/images/eliza.png)
    > En version af Eliza, en chatbot

---

* "Blocks world" var et eksempel på en mikroverden, hvor blokke kunne stables og sorteres, og eksperimenter med at lære maskiner at træffe beslutninger kunne testes. Fremskridt bygget med biblioteker som [SHRDLU](https://wikipedia.org/wiki/SHRDLU) hjalp med at fremme sprogbehandling.

    [![blocks world med SHRDLU](https://img.youtube.com/vi/QAJz4YKUwqw/0.jpg)](https://www.youtube.com/watch?v=QAJz4YKUwqw "blocks world med SHRDLU")

    > 🎥 Klik på billedet ovenfor for en video: Blocks world med SHRDLU

---
## 1974 - 1980: "AI-vinter"

I midten af 1970'erne blev det klart, at kompleksiteten ved at skabe 'intelligente maskiner' var blevet undervurderet, og at løfterne, givet den tilgængelige beregningskraft, var blevet overdrevet. Finansiering tørrede ud, og tilliden til feltet aftog. Nogle problemer, der påvirkede tilliden, inkluderede:
---
- **Begrænsninger**. Beregningskraften var for begrænset.
- **Kombinatorisk eksplosion**. Antallet af parametre, der skulle trænes, voksede eksponentielt, efterhånden som der blev stillet flere krav til computere, uden en parallel udvikling af beregningskraft og kapacitet.
- **Mangel på data**. Der var en mangel på data, der hæmmede processen med at teste, udvikle og forfine algoritmer.
- **Stiller vi de rigtige spørgsmål?**. De spørgsmål, der blev stillet, begyndte at blive stillet spørgsmålstegn ved. Forskere begyndte at møde kritik af deres tilgange:
  - Turing-testen blev sat i tvivl, blandt andet gennem 'den kinesiske rum-teori', som hævdede, at "programmering af en digital computer kan få den til at fremstå som om den forstår sprog, men kan ikke producere reel forståelse." ([kilde](https://plato.stanford.edu/entries/chinese-room/))
  - Etikken ved at introducere kunstige intelligenser som "terapeuten" ELIZA i samfundet blev udfordret.

---

Samtidig begyndte forskellige AI-skoler at tage form. En dikotomi blev etableret mellem ["scruffy" vs. "neat AI"](https://wikipedia.org/wiki/Neats_and_scruffies) praksisser. _Scruffy_-laboratorier finjusterede programmer i timevis, indtil de opnåede de ønskede resultater. _Neat_-laboratorier "fokuserede på logik og formel problemløsning". ELIZA og SHRDLU var velkendte _scruffy_-systemer. I 1980'erne, da der opstod krav om at gøre ML-systemer reproducerbare, tog _neat_-tilgangen gradvist føringen, da dens resultater er mere forklarlige.

---
## 1980'ernes ekspertsystemer

Efterhånden som feltet voksede, blev dets fordele for erhvervslivet tydeligere, og i 1980'erne skete der en udbredelse af 'ekspertsystemer'. "Ekspertsystemer var blandt de første virkelig succesfulde former for kunstig intelligens (AI)-software." ([kilde](https://wikipedia.org/wiki/Expert_system)).

Denne type system er faktisk _hybrid_, bestående delvist af en regelmotor, der definerer forretningskrav, og en inferensmotor, der udnytter regelsystemet til at udlede nye fakta.

Denne æra så også stigende opmærksomhed på neurale netværk.

---
## 1987 - 1993: AI 'Chill'

Udbredelsen af specialiseret ekspertsystem-hardware havde den uheldige effekt at blive for specialiseret. Fremkomsten af personlige computere konkurrerede også med disse store, specialiserede, centraliserede systemer. Demokratiseringen af computing var begyndt, og det banede til sidst vejen for den moderne eksplosion af big data.

---
## 1993 - 2011

Denne epoke markerede en ny æra for ML og AI, hvor nogle af de problemer, der tidligere var forårsaget af mangel på data og beregningskraft, kunne løses. Mængden af data begyndte hurtigt at stige og blive mere bredt tilgængelig, på godt og ondt, især med fremkomsten af smartphones omkring 2007. Beregningskraften udvidede sig eksponentielt, og algoritmer udviklede sig sideløbende. Feltet begyndte at opnå modenhed, da de tidligere frie og eksperimenterende dage begyndte at krystallisere sig til en egentlig disciplin.

---
## Nu

I dag berører maskinlæring og AI næsten alle dele af vores liv. Denne æra kræver en omhyggelig forståelse af risiciene og de potentielle effekter af disse algoritmer på menneskeliv. Som Microsofts Brad Smith har udtalt: "Informationsteknologi rejser spørgsmål, der går til kernen af fundamentale menneskerettighedsbeskyttelser som privatliv og ytringsfrihed. Disse spørgsmål øger ansvaret for teknologivirksomheder, der skaber disse produkter. Efter vores mening kræver de også en gennemtænkt regeringsregulering og udvikling af normer omkring acceptable anvendelser" ([kilde](https://www.technologyreview.com/2019/12/18/102365/the-future-of-ais-impact-on-society/)).

---

Det er endnu uvist, hvad fremtiden bringer, men det er vigtigt at forstå disse computersystemer og den software og de algoritmer, de kører. Vi håber, at dette pensum vil hjælpe dig med at opnå en bedre forståelse, så du selv kan tage stilling.

[![Historien om dyb læring](https://img.youtube.com/vi/mTtDfKgLm54/0.jpg)](https://www.youtube.com/watch?v=mTtDfKgLm54 "Historien om dyb læring")
> 🎥 Klik på billedet ovenfor for en video: Yann LeCun diskuterer historien om dyb læring i denne forelæsning

---
## 🚀Udfordring

Dyk ned i et af disse historiske øjeblikke og lær mere om personerne bag dem. Der er fascinerende karakterer, og ingen videnskabelig opdagelse blev nogensinde skabt i et kulturelt vakuum. Hvad opdager du?

## [Quiz efter lektionen](https://ff-quizzes.netlify.app/en/ml/)

---
## Gennemgang & Selvstudie

Her er ting, du kan se og lytte til:

[Denne podcast, hvor Amy Boyd diskuterer AI's udvikling](http://runasradio.com/Shows/Show/739)

[![Historien om AI af Amy Boyd](https://img.youtube.com/vi/EJt3_bFYKss/0.jpg)](https://www.youtube.com/watch?v=EJt3_bFYKss "Historien om AI af Amy Boyd")

---

## Opgave

[Opret en tidslinje](assignment.md)

---

**Ansvarsfraskrivelse**:  
Dette dokument er blevet oversat ved hjælp af AI-oversættelsestjenesten [Co-op Translator](https://github.com/Azure/co-op-translator). Selvom vi bestræber os på nøjagtighed, skal du være opmærksom på, at automatiserede oversættelser kan indeholde fejl eller unøjagtigheder. Det originale dokument på dets oprindelige sprog bør betragtes som den autoritative kilde. For kritisk information anbefales professionel menneskelig oversættelse. Vi er ikke ansvarlige for eventuelle misforståelser eller fejltolkninger, der opstår som følge af brugen af denne oversættelse.