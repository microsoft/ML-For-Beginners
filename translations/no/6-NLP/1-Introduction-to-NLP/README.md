<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "1c2ec40cf55c98a028a359c27ef7e45a",
  "translation_date": "2025-09-05T22:22:25+00:00",
  "source_file": "6-NLP/1-Introduction-to-NLP/README.md",
  "language_code": "no"
}
-->
# Introduksjon til naturlig språkbehandling

Denne leksjonen dekker en kort historie og viktige konsepter innen *naturlig språkbehandling*, et underfelt av *datamaskinlingvistikk*.

## [Quiz før leksjonen](https://ff-quizzes.netlify.app/en/ml/)

## Introduksjon

NLP, som det ofte kalles, er et av de mest kjente områdene hvor maskinlæring har blitt anvendt og brukt i produksjonsprogramvare.

✅ Kan du tenke på programvare du bruker hver dag som sannsynligvis har noe NLP innebygd? Hva med tekstbehandlingsprogrammer eller mobilapper du bruker regelmessig?

Du vil lære om:

- **Ideen om språk**. Hvordan språk utviklet seg og hva de viktigste studieområdene har vært.
- **Definisjon og konsepter**. Du vil også lære definisjoner og konsepter om hvordan datamaskiner behandler tekst, inkludert parsing, grammatikk og identifisering av substantiver og verb. Det er noen kodingsoppgaver i denne leksjonen, og flere viktige konsepter introduseres som du vil lære å kode senere i de neste leksjonene.

## Datamaskinlingvistikk

Datamaskinlingvistikk er et forsknings- og utviklingsområde gjennom mange tiår som studerer hvordan datamaskiner kan arbeide med, og til og med forstå, oversette og kommunisere med språk. Naturlig språkbehandling (NLP) er et relatert felt som fokuserer på hvordan datamaskiner kan behandle 'naturlige', eller menneskelige, språk.

### Eksempel - diktering på telefon

Hvis du noen gang har diktert til telefonen din i stedet for å skrive eller spurt en virtuell assistent et spørsmål, ble talen din konvertert til tekstform og deretter behandlet eller *parset* fra språket du snakket. De oppdagede nøkkelordene ble deretter behandlet til et format som telefonen eller assistenten kunne forstå og handle på.

![forståelse](../../../../6-NLP/1-Introduction-to-NLP/images/comprehension.png)
> Ekte språklig forståelse er vanskelig! Bilde av [Jen Looper](https://twitter.com/jenlooper)

### Hvordan er denne teknologien mulig?

Dette er mulig fordi noen har skrevet et dataprogram for å gjøre dette. For noen tiår siden forutså noen science fiction-forfattere at folk stort sett ville snakke med datamaskinene sine, og at datamaskinene alltid ville forstå nøyaktig hva de mente. Dessverre viste det seg å være et vanskeligere problem enn mange forestilte seg, og selv om det er et mye bedre forstått problem i dag, er det betydelige utfordringer med å oppnå 'perfekt' naturlig språkbehandling når det gjelder å forstå meningen med en setning. Dette er spesielt vanskelig når det gjelder å forstå humor eller oppdage følelser som sarkasme i en setning.

På dette tidspunktet husker du kanskje skoletimer der læreren dekket grammatiske deler av en setning. I noen land blir elever undervist i grammatikk og lingvistikk som et eget fag, men i mange land er disse temaene inkludert som en del av språkopplæringen: enten ditt førstespråk i barneskolen (lære å lese og skrive) og kanskje et andrespråk i ungdomsskolen eller videregående. Ikke bekymre deg hvis du ikke er ekspert på å skille substantiver fra verb eller adverb fra adjektiver!

Hvis du sliter med forskjellen mellom *presens* og *presens progressiv*, er du ikke alene. Dette er utfordrende for mange mennesker, selv de som har språket som morsmål. Den gode nyheten er at datamaskiner er veldig gode til å anvende formelle regler, og du vil lære å skrive kode som kan *parse* en setning like godt som et menneske. Den større utfordringen du vil undersøke senere er å forstå *meningen* og *følelsen* av en setning.

## Forutsetninger

For denne leksjonen er den viktigste forutsetningen at du kan lese og forstå språket i denne leksjonen. Det er ingen matematiske problemer eller ligninger å løse. Selv om den opprinnelige forfatteren skrev denne leksjonen på engelsk, er den også oversatt til andre språk, så du kan lese en oversettelse. Det er eksempler der flere forskjellige språk brukes (for å sammenligne de forskjellige grammatikkreglene for ulike språk). Disse er *ikke* oversatt, men den forklarende teksten er det, så meningen bør være klar.

For kodingsoppgavene vil du bruke Python, og eksemplene bruker Python 3.8.

I denne delen vil du trenge og bruke:

- **Python 3 forståelse**. Forståelse av programmeringsspråket Python 3, denne leksjonen bruker input, løkker, fillesing, arrays.
- **Visual Studio Code + utvidelse**. Vi vil bruke Visual Studio Code og dens Python-utvidelse. Du kan også bruke en Python IDE etter eget valg.
- **TextBlob**. [TextBlob](https://github.com/sloria/TextBlob) er et forenklet tekstbehandlingsbibliotek for Python. Følg instruksjonene på TextBlob-nettstedet for å installere det på systemet ditt (installer også corpora, som vist nedenfor):

   ```bash
   pip install -U textblob
   python -m textblob.download_corpora
   ```

> 💡 Tips: Du kan kjøre Python direkte i VS Code-miljøer. Sjekk [dokumentasjonen](https://code.visualstudio.com/docs/languages/python?WT.mc_id=academic-77952-leestott) for mer informasjon.

## Å snakke med maskiner

Historien om å prøve å få datamaskiner til å forstå menneskelig språk går flere tiår tilbake, og en av de tidligste forskerne som vurderte naturlig språkbehandling var *Alan Turing*.

### 'Turing-testen'

Da Turing forsket på *kunstig intelligens* på 1950-tallet, vurderte han om en samtaletest kunne gis til et menneske og en datamaskin (via skriftlig korrespondanse) der mennesket i samtalen ikke var sikker på om de kommuniserte med et annet menneske eller en datamaskin.

Hvis, etter en viss lengde på samtalen, mennesket ikke kunne avgjøre om svarene kom fra en datamaskin eller ikke, kunne datamaskinen da sies å *tenke*?

### Inspirasjonen - 'imitasjonsspillet'

Ideen til dette kom fra et selskapslek kalt *Imitasjonsspillet* der en utspørrer er alene i et rom og har som oppgave å avgjøre hvem av to personer (i et annet rom) som er henholdsvis mann og kvinne. Utspørreren kan sende notater og må prøve å tenke på spørsmål der de skriftlige svarene avslører kjønnet til den mystiske personen. Selvfølgelig prøver spillerne i det andre rommet å lure utspørreren ved å svare på spørsmål på en måte som villeder eller forvirrer utspørreren, samtidig som de gir inntrykk av å svare ærlig.

### Utviklingen av Eliza

På 1960-tallet utviklet en MIT-forsker ved navn *Joseph Weizenbaum* [*Eliza*](https://wikipedia.org/wiki/ELIZA), en datamaskin-'terapeut' som ville stille mennesket spørsmål og gi inntrykk av å forstå svarene deres. Men selv om Eliza kunne parse en setning og identifisere visse grammatiske konstruksjoner og nøkkelord for å gi et rimelig svar, kunne det ikke sies å *forstå* setningen. Hvis Eliza ble presentert med en setning som følger formatet "**Jeg er** <u>trist</u>", kunne den omorganisere og erstatte ord i setningen for å danne svaret "Hvor lenge har **du vært** <u>trist</u>". 

Dette ga inntrykk av at Eliza forsto utsagnet og stilte et oppfølgingsspørsmål, mens den i virkeligheten endret tid og la til noen ord. Hvis Eliza ikke kunne identifisere et nøkkelord som den hadde et svar for, ville den i stedet gi et tilfeldig svar som kunne være anvendelig for mange forskjellige utsagn. Eliza kunne lett bli lurt, for eksempel hvis en bruker skrev "**Du er** en <u>sykkel</u>", kunne den svare med "Hvor lenge har **jeg vært** en <u>sykkel</u>?", i stedet for et mer fornuftig svar.

[![Chatte med Eliza](https://img.youtube.com/vi/RMK9AphfLco/0.jpg)](https://youtu.be/RMK9AphfLco "Chatte med Eliza")

> 🎥 Klikk på bildet over for en video om det originale ELIZA-programmet

> Merk: Du kan lese den originale beskrivelsen av [Eliza](https://cacm.acm.org/magazines/1966/1/13317-elizaa-computer-program-for-the-study-of-natural-language-communication-between-man-and-machine/abstract) publisert i 1966 hvis du har en ACM-konto. Alternativt kan du lese om Eliza på [wikipedia](https://wikipedia.org/wiki/ELIZA)

## Øvelse - kode en enkel samtalebot

En samtalebot, som Eliza, er et program som henter brukerinput og ser ut til å forstå og svare intelligent. I motsetning til Eliza vil vår bot ikke ha flere regler som gir den inntrykk av å ha en intelligent samtale. I stedet vil vår bot ha én eneste evne, nemlig å holde samtalen i gang med tilfeldige svar som kan fungere i nesten enhver triviell samtale.

### Planen

Dine steg når du bygger en samtalebot:

1. Skriv ut instruksjoner som informerer brukeren om hvordan de skal interagere med boten
2. Start en løkke
   1. Aksepter brukerinput
   2. Hvis brukeren har bedt om å avslutte, avslutt
   3. Behandle brukerinput og bestem svar (i dette tilfellet er svaret et tilfeldig valg fra en liste over mulige generiske svar)
   4. Skriv ut svar
3. Gå tilbake til steg 2

### Bygge boten

La oss lage boten nå. Vi starter med å definere noen fraser.

1. Lag denne boten selv i Python med følgende tilfeldige svar:

    ```python
    random_responses = ["That is quite interesting, please tell me more.",
                        "I see. Do go on.",
                        "Why do you say that?",
                        "Funny weather we've been having, isn't it?",
                        "Let's change the subject.",
                        "Did you catch the game last night?"]
    ```

    Her er noen eksempler på output for å veilede deg (brukerinput er på linjene som starter med `>`):

    ```output
    Hello, I am Marvin, the simple robot.
    You can end this conversation at any time by typing 'bye'
    After typing each answer, press 'enter'
    How are you today?
    > I am good thanks
    That is quite interesting, please tell me more.
    > today I went for a walk     
    Did you catch the game last night?
    > I did, but my team lost
    Funny weather we've been having, isn't it?
    > yes but I hope next week is better
    Let's change the subject.
    > ok, lets talk about music
    Why do you say that?
    > because I like music!
    Why do you say that?
    > bye
    It was nice talking to you, goodbye!
    ```

    En mulig løsning på oppgaven er [her](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/1-Introduction-to-NLP/solution/bot.py)

    ✅ Stopp og vurder

    1. Tror du de tilfeldige svarene ville 'lure' noen til å tro at boten faktisk forsto dem?
    2. Hvilke funksjoner ville boten trenge for å være mer effektiv?
    3. Hvis en bot virkelig kunne 'forstå' meningen med en setning, ville den trenge å 'huske' meningen med tidligere setninger i en samtale også?

---

## 🚀Utfordring

Velg ett av "stopp og vurder"-elementene ovenfor og prøv enten å implementere dem i kode eller skriv en løsning på papir ved hjelp av pseudokode.

I neste leksjon vil du lære om en rekke andre tilnærminger til parsing av naturlig språk og maskinlæring.

## [Quiz etter leksjonen](https://ff-quizzes.netlify.app/en/ml/)

## Gjennomgang & Selvstudium

Ta en titt på referansene nedenfor som videre lesemuligheter.

### Referanser

1. Schubert, Lenhart, "Computational Linguistics", *The Stanford Encyclopedia of Philosophy* (Spring 2020 Edition), Edward N. Zalta (ed.), URL = <https://plato.stanford.edu/archives/spr2020/entries/computational-linguistics/>.
2. Princeton University "About WordNet." [WordNet](https://wordnet.princeton.edu/). Princeton University. 2010. 

## Oppgave 

[Søk etter en bot](assignment.md)

---

**Ansvarsfraskrivelse**:  
Dette dokumentet er oversatt ved hjelp av AI-oversettelsestjenesten [Co-op Translator](https://github.com/Azure/co-op-translator). Selv om vi tilstreber nøyaktighet, vær oppmerksom på at automatiske oversettelser kan inneholde feil eller unøyaktigheter. Det originale dokumentet på sitt opprinnelige språk bør anses som den autoritative kilden. For kritisk informasjon anbefales profesjonell menneskelig oversettelse. Vi er ikke ansvarlige for eventuelle misforståelser eller feiltolkninger som oppstår ved bruk av denne oversettelsen.