<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "1c2ec40cf55c98a028a359c27ef7e45a",
  "translation_date": "2025-09-05T01:34:50+00:00",
  "source_file": "6-NLP/1-Introduction-to-NLP/README.md",
  "language_code": "da"
}
-->
# Introduktion til naturlig sprogbehandling

Denne lektion dækker en kort historie og vigtige begreber inden for *naturlig sprogbehandling*, et underfelt af *computational linguistics*.

## [Quiz før lektionen](https://ff-quizzes.netlify.app/en/ml/)

## Introduktion

NLP, som det ofte kaldes, er et af de mest kendte områder, hvor maskinlæring er blevet anvendt og brugt i produktionssoftware.

✅ Kan du tænke på software, som du bruger hver dag, der sandsynligvis har noget NLP indbygget? Hvad med dine tekstbehandlingsprogrammer eller mobilapps, som du bruger regelmæssigt?

Du vil lære om:

- **Idéen om sprog**. Hvordan sprog udviklede sig, og hvad de vigtigste studieområder har været.
- **Definition og begreber**. Du vil også lære definitioner og begreber om, hvordan computere behandler tekst, herunder parsing, grammatik og identifikation af navneord og udsagnsord. Der er nogle kodningsopgaver i denne lektion, og flere vigtige begreber introduceres, som du senere vil lære at kode i de næste lektioner.

## Computational linguistics

Computational linguistics er et forsknings- og udviklingsområde gennem mange årtier, der studerer, hvordan computere kan arbejde med, og endda forstå, oversætte og kommunikere med sprog. Naturlig sprogbehandling (NLP) er et relateret felt, der fokuserer på, hvordan computere kan behandle 'naturlige', eller menneskelige, sprog.

### Eksempel - telefon-diktering

Hvis du nogensinde har dikteret til din telefon i stedet for at skrive eller stillet en virtuel assistent et spørgsmål, blev din tale konverteret til tekstform og derefter behandlet eller *parset* fra det sprog, du talte. De detekterede nøgleord blev derefter behandlet i et format, som telefonen eller assistenten kunne forstå og handle på.

![forståelse](../../../../6-NLP/1-Introduction-to-NLP/images/comprehension.png)
> Ægte lingvistisk forståelse er svært! Billede af [Jen Looper](https://twitter.com/jenlooper)

### Hvordan er denne teknologi mulig?

Dette er muligt, fordi nogen har skrevet et computerprogram til at gøre det. For nogle årtier siden forudsagde nogle science fiction-forfattere, at folk primært ville tale med deres computere, og at computere altid ville forstå præcis, hvad de mente. Desværre viste det sig at være et sværere problem, end mange forestillede sig, og selvom det er et meget bedre forstået problem i dag, er der betydelige udfordringer med at opnå 'perfekt' naturlig sprogbehandling, når det kommer til at forstå betydningen af en sætning. Dette er et særligt vanskeligt problem, når det kommer til at forstå humor eller opdage følelser som sarkasme i en sætning.

På dette tidspunkt husker du måske skoleklasser, hvor læreren gennemgik grammatiske dele af en sætning. I nogle lande undervises elever i grammatik og lingvistik som et dedikeret fag, men i mange er disse emner inkluderet som en del af at lære et sprog: enten dit første sprog i folkeskolen (at lære at læse og skrive) og måske et andet sprog i gymnasiet. Bare rolig, hvis du ikke er ekspert i at skelne mellem navneord og udsagnsord eller adverbier og adjektiver!

Hvis du har svært ved forskellen mellem *simpel nutid* og *nutid progressiv*, er du ikke alene. Dette er en udfordrende ting for mange mennesker, selv for modersmålstalere af et sprog. Den gode nyhed er, at computere er rigtig gode til at anvende formelle regler, og du vil lære at skrive kode, der kan *parse* en sætning lige så godt som et menneske. Den større udfordring, du vil undersøge senere, er at forstå *betydningen* og *følelsen* af en sætning.

## Forudsætninger

For denne lektion er den vigtigste forudsætning, at du kan læse og forstå sproget i denne lektion. Der er ingen matematiske problemer eller ligninger, der skal løses. Mens den oprindelige forfatter skrev denne lektion på engelsk, er den også oversat til andre sprog, så du kunne læse en oversættelse. Der er eksempler, hvor en række forskellige sprog bruges (for at sammenligne de forskellige grammatiske regler for forskellige sprog). Disse er *ikke* oversat, men den forklarende tekst er, så betydningen bør være klar.

For kodningsopgaverne vil du bruge Python, og eksemplerne bruger Python 3.8.

I denne sektion vil du have brug for og bruge:

- **Python 3 forståelse**. Forståelse af programmeringssproget Python 3, denne lektion bruger input, loops, fil-læsning, arrays.
- **Visual Studio Code + udvidelse**. Vi vil bruge Visual Studio Code og dets Python-udvidelse. Du kan også bruge en Python IDE efter eget valg.
- **TextBlob**. [TextBlob](https://github.com/sloria/TextBlob) er et forenklet tekstbehandlingsbibliotek til Python. Følg instruktionerne på TextBlob-siden for at installere det på dit system (installer også corpora som vist nedenfor):

   ```bash
   pip install -U textblob
   python -m textblob.download_corpora
   ```

> 💡 Tip: Du kan køre Python direkte i VS Code-miljøer. Tjek [dokumentationen](https://code.visualstudio.com/docs/languages/python?WT.mc_id=academic-77952-leestott) for mere information.

## At tale med maskiner

Historien om at forsøge at få computere til at forstå menneskeligt sprog går årtier tilbage, og en af de tidligste videnskabsfolk, der overvejede naturlig sprogbehandling, var *Alan Turing*.

### 'Turing-testen'

Da Turing forskede i *kunstig intelligens* i 1950'erne, overvejede han, om en samtaletest kunne gives til et menneske og en computer (via skriftlig korrespondance), hvor mennesket i samtalen ikke var sikker på, om de kommunikerede med et andet menneske eller en computer.

Hvis mennesket efter en vis længde af samtalen ikke kunne afgøre, om svarene kom fra en computer eller ej, kunne computeren så siges at *tænke*?

### Inspirationen - 'imitationsspillet'

Idéen til dette kom fra et selskabsspil kaldet *Imitationsspillet*, hvor en forhører er alene i et rum og har til opgave at afgøre, hvem af to personer (i et andet rum) er henholdsvis mand og kvinde. Forhøreren kan sende noter og skal forsøge at finde på spørgsmål, hvor de skriftlige svar afslører kønnet på den mystiske person. Selvfølgelig forsøger spillerne i det andet rum at narre forhøreren ved at besvare spørgsmål på en måde, der vildleder eller forvirrer forhøreren, samtidig med at de giver indtryk af at svare ærligt.

### Udvikling af Eliza

I 1960'erne udviklede en MIT-videnskabsmand ved navn *Joseph Weizenbaum* [*Eliza*](https://wikipedia.org/wiki/ELIZA), en computer-'terapeut', der ville stille mennesket spørgsmål og give indtryk af at forstå deres svar. Men selvom Eliza kunne parse en sætning og identificere visse grammatiske konstruktioner og nøgleord for at give et rimeligt svar, kunne den ikke siges at *forstå* sætningen. Hvis Eliza blev præsenteret for en sætning med formatet "**Jeg er** <u>ked af det</u>", kunne den omarrangere og erstatte ord i sætningen for at danne svaret "Hvor længe har **du været** <u>ked af det</u>". 

Dette gav indtryk af, at Eliza forstod udsagnet og stillede et opfølgende spørgsmål, mens den i virkeligheden ændrede tiden og tilføjede nogle ord. Hvis Eliza ikke kunne identificere et nøgleord, som den havde et svar på, ville den i stedet give et tilfældigt svar, der kunne være anvendeligt til mange forskellige udsagn. Eliza kunne nemt narres, for eksempel hvis en bruger skrev "**Du er** en <u>cykel</u>", kunne den svare med "Hvor længe har **jeg været** en <u>cykel</u>?", i stedet for et mere fornuftigt svar.

[![Chat med Eliza](https://img.youtube.com/vi/RMK9AphfLco/0.jpg)](https://youtu.be/RMK9AphfLco "Chat med Eliza")

> 🎥 Klik på billedet ovenfor for en video om det originale ELIZA-program

> Note: Du kan læse den originale beskrivelse af [Eliza](https://cacm.acm.org/magazines/1966/1/13317-elizaa-computer-program-for-the-study-of-natural-language-communication-between-man-and-machine/abstract) udgivet i 1966, hvis du har en ACM-konto. Alternativt kan du læse om Eliza på [wikipedia](https://wikipedia.org/wiki/ELIZA)

## Øvelse - kodning af en grundlæggende samtalebot

En samtalebot, som Eliza, er et program, der indhenter brugerinput og ser ud til at forstå og svare intelligent. I modsætning til Eliza vil vores bot ikke have flere regler, der giver den indtryk af at have en intelligent samtale. I stedet vil vores bot kun have én evne, nemlig at holde samtalen i gang med tilfældige svar, der måske fungerer i næsten enhver triviel samtale.

### Planen

Dine trin, når du bygger en samtalebot:

1. Udskriv instruktioner, der rådgiver brugeren om, hvordan man interagerer med botten
2. Start en løkke
   1. Accepter brugerinput
   2. Hvis brugeren har bedt om at afslutte, så afslut
   3. Behandl brugerinput og bestem svar (i dette tilfælde er svaret et tilfældigt valg fra en liste over mulige generiske svar)
   4. Udskriv svar
3. Gå tilbage til trin 2

### Bygning af botten

Lad os oprette botten nu. Vi starter med at definere nogle sætninger.

1. Opret denne bot selv i Python med følgende tilfældige svar:

    ```python
    random_responses = ["That is quite interesting, please tell me more.",
                        "I see. Do go on.",
                        "Why do you say that?",
                        "Funny weather we've been having, isn't it?",
                        "Let's change the subject.",
                        "Did you catch the game last night?"]
    ```

    Her er noget eksempeloutput til vejledning (brugerinput er på linjerne, der starter med `>`):

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

    En mulig løsning på opgaven er [her](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/1-Introduction-to-NLP/solution/bot.py)

    ✅ Stop og overvej

    1. Tror du, at de tilfældige svar ville 'narre' nogen til at tro, at botten faktisk forstod dem?
    2. Hvilke funktioner ville botten have brug for for at være mere effektiv?
    3. Hvis en bot virkelig kunne 'forstå' betydningen af en sætning, ville den så også have brug for at 'huske' betydningen af tidligere sætninger i en samtale?

---

## 🚀Udfordring

Vælg et af de "stop og overvej"-elementer ovenfor og prøv enten at implementere dem i kode eller skriv en løsning på papir ved hjælp af pseudokode.

I den næste lektion vil du lære om en række andre tilgange til parsing af naturligt sprog og maskinlæring.

## [Quiz efter lektionen](https://ff-quizzes.netlify.app/en/ml/)

## Gennemgang & Selvstudie

Tag et kig på referencerne nedenfor som yderligere læsemuligheder.

### Referencer

1. Schubert, Lenhart, "Computational Linguistics", *The Stanford Encyclopedia of Philosophy* (Spring 2020 Edition), Edward N. Zalta (ed.), URL = <https://plato.stanford.edu/archives/spr2020/entries/computational-linguistics/>.
2. Princeton University "About WordNet." [WordNet](https://wordnet.princeton.edu/). Princeton University. 2010. 

## Opgave 

[Søg efter en bot](assignment.md)

---

**Ansvarsfraskrivelse**:  
Dette dokument er blevet oversat ved hjælp af AI-oversættelsestjenesten [Co-op Translator](https://github.com/Azure/co-op-translator). Selvom vi bestræber os på nøjagtighed, skal du være opmærksom på, at automatiserede oversættelser kan indeholde fejl eller unøjagtigheder. Det originale dokument på dets oprindelige sprog bør betragtes som den autoritative kilde. For kritisk information anbefales professionel menneskelig oversættelse. Vi påtager os ikke ansvar for eventuelle misforståelser eller fejltolkninger, der måtte opstå som følge af brugen af denne oversættelse.