<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "6a05fec147e734c3e6bfa54505648e2b",
  "translation_date": "2025-09-05T21:45:17+00:00",
  "source_file": "1-Introduction/2-history-of-ML/README.md",
  "language_code": "no"
}
-->
# Historien om maskinlæring

![Oppsummering av historien om maskinlæring i en sketchnote](../../../../sketchnotes/ml-history.png)
> Sketchnote av [Tomomi Imura](https://www.twitter.com/girlie_mac)

## [Quiz før forelesning](https://ff-quizzes.netlify.app/en/ml/)

---

[![ML for nybegynnere - Historien om maskinlæring](https://img.youtube.com/vi/N6wxM4wZ7V0/0.jpg)](https://youtu.be/N6wxM4wZ7V0 "ML for nybegynnere - Historien om maskinlæring")

> 🎥 Klikk på bildet over for en kort video som går gjennom denne leksjonen.

I denne leksjonen skal vi gå gjennom de viktigste milepælene i historien om maskinlæring og kunstig intelligens.

Historien om kunstig intelligens (AI) som et fagfelt er nært knyttet til historien om maskinlæring, ettersom algoritmene og de teknologiske fremskrittene som ligger til grunn for ML har bidratt til utviklingen av AI. Det er nyttig å huske at selv om disse feltene som separate forskningsområder begynte å ta form på 1950-tallet, så fantes det viktige [algoritmiske, statistiske, matematiske, teknologiske og tekniske oppdagelser](https://wikipedia.org/wiki/Timeline_of_machine_learning) som både gikk forut for og overlappet denne perioden. Faktisk har mennesker tenkt på disse spørsmålene i [hundrevis av år](https://wikipedia.org/wiki/History_of_artificial_intelligence): denne artikkelen diskuterer de historiske intellektuelle grunnlagene for ideen om en 'tenkende maskin.'

---
## Viktige oppdagelser

- 1763, 1812 [Bayes' teorem](https://wikipedia.org/wiki/Bayes%27_theorem) og dets forgjengere. Dette teoremet og dets anvendelser ligger til grunn for inferens og beskriver sannsynligheten for at en hendelse inntreffer basert på tidligere kunnskap.
- 1805 [Minste kvadraters metode](https://wikipedia.org/wiki/Least_squares) av den franske matematikeren Adrien-Marie Legendre. Denne teorien, som du vil lære om i vår enhet om regresjon, hjelper med datafitting.
- 1913 [Markov-kjeder](https://wikipedia.org/wiki/Markov_chain), oppkalt etter den russiske matematikeren Andrey Markov, brukes til å beskrive en sekvens av mulige hendelser basert på en tidligere tilstand.
- 1957 [Perceptron](https://wikipedia.org/wiki/Perceptron) er en type lineær klassifikator oppfunnet av den amerikanske psykologen Frank Rosenblatt som ligger til grunn for fremskritt innen dyp læring.

---

- 1967 [Nærmeste nabo](https://wikipedia.org/wiki/Nearest_neighbor) er en algoritme opprinnelig designet for å kartlegge ruter. I en ML-kontekst brukes den til å oppdage mønstre.
- 1970 [Backpropagation](https://wikipedia.org/wiki/Backpropagation) brukes til å trene [feedforward-nevrale nettverk](https://wikipedia.org/wiki/Feedforward_neural_network).
- 1982 [Rekurrente nevrale nettverk](https://wikipedia.org/wiki/Recurrent_neural_network) er kunstige nevrale nettverk avledet fra feedforward-nevrale nettverk som skaper temporale grafer.

✅ Gjør litt research. Hvilke andre datoer skiller seg ut som avgjørende i historien om ML og AI?

---
## 1950: Maskiner som tenker

Alan Turing, en virkelig bemerkelsesverdig person som ble kåret [av publikum i 2019](https://wikipedia.org/wiki/Icons:_The_Greatest_Person_of_the_20th_Century) til den største vitenskapsmannen i det 20. århundre, er kreditert for å ha bidratt til å legge grunnlaget for konseptet om en 'maskin som kan tenke.' Han tok opp kampen med skeptikere og sitt eget behov for empirisk bevis for dette konseptet, blant annet ved å skape [Turing-testen](https://www.bbc.com/news/technology-18475646), som du vil utforske i våre NLP-leksjoner.

---
## 1956: Dartmouth Summer Research Project

"Dartmouth Summer Research Project on artificial intelligence var en banebrytende begivenhet for kunstig intelligens som et fagfelt," og det var her begrepet 'kunstig intelligens' ble introdusert ([kilde](https://250.dartmouth.edu/highlights/artificial-intelligence-ai-coined-dartmouth)).

> Hvert aspekt av læring eller enhver annen egenskap ved intelligens kan i prinsippet beskrives så presist at en maskin kan lages for å simulere det.

---

Lederforskeren, matematikkprofessor John McCarthy, håpet "å gå videre basert på antagelsen om at hvert aspekt av læring eller enhver annen egenskap ved intelligens kan i prinsippet beskrives så presist at en maskin kan lages for å simulere det." Deltakerne inkluderte en annen pioner innen feltet, Marvin Minsky.

Workshoppen er kreditert for å ha initiert og oppmuntret til flere diskusjoner, inkludert "fremveksten av symbolske metoder, systemer fokusert på begrensede domener (tidlige ekspert-systemer), og deduktive systemer versus induktive systemer." ([kilde](https://wikipedia.org/wiki/Dartmouth_workshop)).

---
## 1956 - 1974: "De gylne årene"

Fra 1950-tallet til midten av 70-tallet var optimismen høy med håp om at AI kunne løse mange problemer. I 1967 uttalte Marvin Minsky selvsikkert at "Innen en generasjon ... vil problemet med å skape 'kunstig intelligens' i stor grad være løst." (Minsky, Marvin (1967), Computation: Finite and Infinite Machines, Englewood Cliffs, N.J.: Prentice-Hall)

Forskning på naturlig språkprosessering blomstret, søk ble raffinert og gjort mer kraftfullt, og konseptet med 'mikroverdener' ble skapt, hvor enkle oppgaver ble utført ved hjelp av instruksjoner i vanlig språk.

---

Forskning ble godt finansiert av statlige organer, fremskritt ble gjort innen beregning og algoritmer, og prototyper av intelligente maskiner ble bygget. Noen av disse maskinene inkluderer:

* [Shakey-roboten](https://wikipedia.org/wiki/Shakey_the_robot), som kunne manøvrere og bestemme hvordan oppgaver skulle utføres 'intelligent'.

    ![Shakey, en intelligent robot](../../../../1-Introduction/2-history-of-ML/images/shakey.jpg)
    > Shakey i 1972

---

* Eliza, en tidlig 'chatterbot', kunne samtale med mennesker og fungere som en primitiv 'terapeut'. Du vil lære mer om Eliza i NLP-leksjonene.

    ![Eliza, en bot](../../../../1-Introduction/2-history-of-ML/images/eliza.png)
    > En versjon av Eliza, en chatbot

---

* "Blocks world" var et eksempel på en mikroverden hvor blokker kunne stables og sorteres, og eksperimenter med å lære maskiner å ta beslutninger kunne testes. Fremskritt bygget med biblioteker som [SHRDLU](https://wikipedia.org/wiki/SHRDLU) hjalp til med å drive språkprosessering fremover.

    [![blocks world med SHRDLU](https://img.youtube.com/vi/QAJz4YKUwqw/0.jpg)](https://www.youtube.com/watch?v=QAJz4YKUwqw "blocks world med SHRDLU")

    > 🎥 Klikk på bildet over for en video: Blocks world med SHRDLU

---
## 1974 - 1980: "AI-vinter"

På midten av 1970-tallet ble det klart at kompleksiteten ved å lage 'intelligente maskiner' hadde blitt undervurdert og at løftene, gitt den tilgjengelige beregningskraften, hadde blitt overdrevet. Finansiering tørket opp, og tilliten til feltet avtok. Noen problemer som påvirket tilliten inkluderte:
---
- **Begrensninger**. Beregningskraften var for begrenset.
- **Kombinatorisk eksplosjon**. Antallet parametere som måtte trenes vokste eksponentielt etter hvert som mer ble forventet av datamaskiner, uten en parallell utvikling av beregningskraft og kapasitet.
- **Mangel på data**. Det var mangel på data som hindret prosessen med å teste, utvikle og forbedre algoritmer.
- **Spør vi de riktige spørsmålene?**. Selve spørsmålene som ble stilt begynte å bli stilt spørsmål ved. Forskere begynte å møte kritikk for sine tilnærminger:
  - Turing-tester ble utfordret, blant annet gjennom 'kinesisk rom-teorien' som hevdet at "programmering av en digital datamaskin kan få den til å fremstå som om den forstår språk, men kunne ikke produsere ekte forståelse." ([kilde](https://plato.stanford.edu/entries/chinese-room/))
  - Etikken ved å introdusere kunstige intelligenser som "terapeuten" ELIZA i samfunnet ble utfordret.

---

Samtidig begynte ulike AI-skoler å dannes. En dikotomi ble etablert mellom ["scruffy" vs. "neat AI"](https://wikipedia.org/wiki/Neats_and_scruffies)-praksiser. _Scruffy_-laboratorier justerte programmer i timevis til de oppnådde ønskede resultater. _Neat_-laboratorier "fokuserte på logikk og formell problemløsning". ELIZA og SHRDLU var velkjente _scruffy_-systemer. På 1980-tallet, da etterspørselen etter å gjøre ML-systemer reproduserbare økte, tok _neat_-tilnærmingen gradvis ledelsen ettersom dens resultater er mer forklarbare.

---
## 1980-tallet: Ekspertsystemer

Etter hvert som feltet vokste, ble dets nytte for næringslivet tydeligere, og på 1980-tallet økte også spredningen av 'ekspertsystemer'. "Ekspertsystemer var blant de første virkelig vellykkede formene for kunstig intelligens (AI)-programvare." ([kilde](https://wikipedia.org/wiki/Expert_system)).

Denne typen system er faktisk _hybrid_, bestående delvis av en regelmotor som definerer forretningskrav, og en inferensmotor som utnyttet reglesystemet for å utlede nye fakta.

Denne perioden så også økende oppmerksomhet rettet mot nevrale nettverk.

---
## 1987 - 1993: AI 'Chill'

Spredningen av spesialisert maskinvare for ekspertsystemer hadde den uheldige effekten av å bli for spesialisert. Fremveksten av personlige datamaskiner konkurrerte også med disse store, spesialiserte, sentraliserte systemene. Demokratifiseringen av databehandling hadde begynt, og den banet til slutt vei for den moderne eksplosjonen av big data.

---
## 1993 - 2011

Denne epoken markerte en ny æra for ML og AI, hvor noen av problemene som tidligere hadde vært forårsaket av mangel på data og beregningskraft kunne løses. Mengden data begynte å øke raskt og bli mer tilgjengelig, på godt og vondt, spesielt med introduksjonen av smarttelefonen rundt 2007. Beregningskraften utvidet seg eksponentielt, og algoritmer utviklet seg parallelt. Feltet begynte å modnes etter hvert som de frie og eksperimentelle dagene fra fortiden begynte å krystallisere seg til en ekte disiplin.

---
## Nå

I dag berører maskinlæring og AI nesten alle deler av livene våre. Denne perioden krever nøye forståelse av risikoene og de potensielle effektene av disse algoritmene på menneskeliv. Som Microsofts Brad Smith har uttalt: "Informasjonsteknologi reiser spørsmål som går til kjernen av grunnleggende menneskerettighetsbeskyttelser som personvern og ytringsfrihet. Disse spørsmålene øker ansvaret for teknologiselskaper som skaper disse produktene. Etter vår mening krever de også gjennomtenkt statlig regulering og utvikling av normer rundt akseptabel bruk" ([kilde](https://www.technologyreview.com/2019/12/18/102365/the-future-of-ais-impact-on-society/)).

---

Det gjenstår å se hva fremtiden bringer, men det er viktig å forstå disse datasystemene og programvaren og algoritmene de kjører. Vi håper at dette pensumet vil hjelpe deg med å få en bedre forståelse slik at du kan ta dine egne beslutninger.

[![Historien om dyp læring](https://img.youtube.com/vi/mTtDfKgLm54/0.jpg)](https://www.youtube.com/watch?v=mTtDfKgLm54 "Historien om dyp læring")
> 🎥 Klikk på bildet over for en video: Yann LeCun diskuterer historien om dyp læring i denne forelesningen

---
## 🚀Utfordring

Fordyp deg i et av disse historiske øyeblikkene og lær mer om menneskene bak dem. Det finnes fascinerende karakterer, og ingen vitenskapelig oppdagelse ble noen gang skapt i et kulturelt vakuum. Hva oppdager du?

## [Quiz etter forelesning](https://ff-quizzes.netlify.app/en/ml/)

---
## Gjennomgang og selvstudium

Her er ting du kan se og lytte til:

[Denne podcasten hvor Amy Boyd diskuterer utviklingen av AI](http://runasradio.com/Shows/Show/739)

[![Historien om AI av Amy Boyd](https://img.youtube.com/vi/EJt3_bFYKss/0.jpg)](https://www.youtube.com/watch?v=EJt3_bFYKss "Historien om AI av Amy Boyd")

---

## Oppgave

[Lag en tidslinje](assignment.md)

---

**Ansvarsfraskrivelse**:  
Dette dokumentet er oversatt ved hjelp av AI-oversettelsestjenesten [Co-op Translator](https://github.com/Azure/co-op-translator). Selv om vi streber etter nøyaktighet, vær oppmerksom på at automatiserte oversettelser kan inneholde feil eller unøyaktigheter. Det originale dokumentet på sitt opprinnelige språk bør anses som den autoritative kilden. For kritisk informasjon anbefales profesjonell menneskelig oversettelse. Vi er ikke ansvarlige for eventuelle misforståelser eller feiltolkninger som oppstår ved bruk av denne oversettelsen.