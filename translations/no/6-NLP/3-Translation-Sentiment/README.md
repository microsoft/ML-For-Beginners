<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "be03c8182982b87ced155e4e9d1438e8",
  "translation_date": "2025-09-05T22:24:54+00:00",
  "source_file": "6-NLP/3-Translation-Sentiment/README.md",
  "language_code": "no"
}
-->
# Oversettelse og sentimentanalyse med maskinlæring

I de forrige leksjonene lærte du hvordan du bygger en enkel bot ved hjelp av `TextBlob`, et bibliotek som bruker maskinlæring i bakgrunnen for å utføre grunnleggende NLP-oppgaver som å trekke ut substantivfraser. En annen viktig utfordring innen datalingvistikk er nøyaktig _oversettelse_ av en setning fra ett muntlig eller skriftlig språk til et annet.

## [Quiz før forelesning](https://ff-quizzes.netlify.app/en/ml/)

Oversettelse er et svært vanskelig problem, forsterket av det faktum at det finnes tusenvis av språk, hver med svært forskjellige grammatikkregler. En tilnærming er å konvertere de formelle grammatikkreglene for ett språk, som engelsk, til en struktur som ikke er språkavhengig, og deretter oversette det ved å konvertere tilbake til et annet språk. Denne tilnærmingen innebærer følgende trinn:

1. **Identifikasjon**. Identifiser eller merk ordene i inngangsspråket som substantiv, verb osv.
2. **Lag oversettelse**. Produser en direkte oversettelse av hvert ord i målspråkets format.

### Eksempelsentence, engelsk til irsk

På 'engelsk' er setningen _I feel happy_ tre ord i rekkefølgen:

- **subjekt** (I)
- **verb** (feel)
- **adjektiv** (happy)

Men på 'irsk' har den samme setningen en helt annen grammatisk struktur – følelser som "*happy*" eller "*sad*" uttrykkes som å være *på* deg.

Den engelske frasen `I feel happy` på irsk ville være `Tá athas orm`. En *bokstavelig* oversettelse ville være `Happy is upon me`.

En irsktalende som oversetter til engelsk ville si `I feel happy`, ikke `Happy is upon me`, fordi de forstår meningen med setningen, selv om ordene og setningsstrukturen er forskjellige.

Den formelle rekkefølgen for setningen på irsk er:

- **verb** (Tá eller is)
- **adjektiv** (athas, eller happy)
- **subjekt** (orm, eller upon me)

## Oversettelse

Et naivt oversettelsesprogram kan oversette ord kun, uten å ta hensyn til setningsstrukturen.

✅ Hvis du har lært et andre (eller tredje eller flere) språk som voksen, har du kanskje startet med å tenke på ditt morsmål, oversette et konsept ord for ord i hodet til det andre språket, og deretter si oversettelsen høyt. Dette ligner på hva naive oversettelsesprogrammer for datamaskiner gjør. Det er viktig å komme forbi denne fasen for å oppnå flyt!

Naiv oversettelse fører til dårlige (og noen ganger morsomme) feiltolkninger: `I feel happy` oversettes bokstavelig til `Mise bhraitheann athas` på irsk. Det betyr (bokstavelig talt) `me feel happy` og er ikke en gyldig irsk setning. Selv om engelsk og irsk er språk som snakkes på to nærliggende øyer, er de svært forskjellige språk med ulike grammatikkstrukturer.

> Du kan se noen videoer om irske språktradisjoner, som [denne](https://www.youtube.com/watch?v=mRIaLSdRMMs)

### Tilnærminger med maskinlæring

Så langt har du lært om tilnærmingen med formelle regler for naturlig språkbehandling. En annen tilnærming er å ignorere betydningen av ordene og _i stedet bruke maskinlæring til å oppdage mønstre_. Dette kan fungere i oversettelse hvis du har mye tekst (et *korpus*) eller tekster (*korpora*) på både kilde- og målspråket.

For eksempel, vurder tilfellet med *Pride and Prejudice*, en kjent engelsk roman skrevet av Jane Austen i 1813. Hvis du konsulterer boken på engelsk og en menneskelig oversettelse av boken til *fransk*, kan du oppdage fraser i den ene som er _idiomatisk_ oversatt til den andre. Det skal du gjøre om et øyeblikk.

For eksempel, når en engelsk frase som `I have no money` oversettes bokstavelig til fransk, kan det bli `Je n'ai pas de monnaie`. "Monnaie" er et vanskelig fransk 'falskt kognat', ettersom 'money' og 'monnaie' ikke er synonyme. En bedre oversettelse som et menneske kan gjøre, ville være `Je n'ai pas d'argent`, fordi det bedre formidler betydningen av at du ikke har penger (i stedet for 'småpenger', som er betydningen av 'monnaie').

![monnaie](../../../../6-NLP/3-Translation-Sentiment/images/monnaie.png)

> Bilde av [Jen Looper](https://twitter.com/jenlooper)

Hvis en ML-modell har nok menneskelige oversettelser å bygge en modell på, kan den forbedre nøyaktigheten av oversettelser ved å identifisere vanlige mønstre i tekster som tidligere har blitt oversatt av eksperter som snakker begge språk.

### Øvelse - oversettelse

Du kan bruke `TextBlob` til å oversette setninger. Prøv den berømte første linjen i **Pride and Prejudice**:

```python
from textblob import TextBlob

blob = TextBlob(
    "It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want of a wife!"
)
print(blob.translate(to="fr"))

```

`TextBlob` gjør en ganske god jobb med oversettelsen: "C'est une vérité universellement reconnue, qu'un homme célibataire en possession d'une bonne fortune doit avoir besoin d'une femme!". 

Det kan argumenteres for at TextBlobs oversettelse faktisk er langt mer presis enn den franske oversettelsen fra 1932 av boken av V. Leconte og Ch. Pressoir:

"C'est une vérité universelle qu'un célibataire pourvu d'une belle fortune doit avoir envie de se marier, et, si peu que l'on sache de son sentiment à cet egard, lorsqu'il arrive dans une nouvelle résidence, cette idée est si bien fixée dans l'esprit de ses voisins qu'ils le considèrent sur-le-champ comme la propriété légitime de l'une ou l'autre de leurs filles."

I dette tilfellet gjør oversettelsen informert av ML en bedre jobb enn den menneskelige oversetteren som unødvendig legger ord i den opprinnelige forfatterens munn for 'klarhet'.

> Hva skjer her? Og hvorfor er TextBlob så god på oversettelse? Vel, i bakgrunnen bruker den Google Translate, en sofistikert AI som kan analysere millioner av fraser for å forutsi de beste strengene for oppgaven. Det er ingenting manuelt som skjer her, og du trenger en internettforbindelse for å bruke `blob.translate`.

✅ Prøv noen flere setninger. Hva er bedre, ML eller menneskelig oversettelse? I hvilke tilfeller?

## Sentimentanalyse

Et annet område hvor maskinlæring kan fungere svært godt, er sentimentanalyse. En ikke-ML-tilnærming til sentiment er å identifisere ord og fraser som er 'positive' og 'negative'. Deretter, gitt en ny tekst, beregne den totale verdien av de positive, negative og nøytrale ordene for å identifisere den overordnede stemningen. 

Denne tilnærmingen kan lett lures, som du kanskje har sett i Marvin-oppgaven – setningen `Great, that was a wonderful waste of time, I'm glad we are lost on this dark road` er en sarkastisk, negativ setning, men den enkle algoritmen oppdager 'great', 'wonderful', 'glad' som positive og 'waste', 'lost' og 'dark' som negative. Den overordnede stemningen påvirkes av disse motstridende ordene.

✅ Stopp et øyeblikk og tenk på hvordan vi formidler sarkasme som mennesker. Tonefall spiller en stor rolle. Prøv å si frasen "Well, that film was awesome" på forskjellige måter for å oppdage hvordan stemmen din formidler mening.

### ML-tilnærminger

ML-tilnærmingen ville være å manuelt samle negative og positive tekstkropper – tweets, eller filmomtaler, eller hva som helst hvor mennesket har gitt en score *og* en skriftlig mening. Deretter kan NLP-teknikker brukes på meninger og scorer, slik at mønstre dukker opp (f.eks. positive filmomtaler har en tendens til å inneholde frasen 'Oscar worthy' mer enn negative filmomtaler, eller positive restaurantomtaler sier 'gourmet' mye mer enn 'disgusting').

> ⚖️ **Eksempel**: Hvis du jobbet på kontoret til en politiker og det var en ny lov som ble diskutert, kunne velgere skrive til kontoret med e-poster som støtter eller er imot den aktuelle nye loven. La oss si at du fikk i oppgave å lese e-postene og sortere dem i 2 bunker, *for* og *imot*. Hvis det var mange e-poster, kunne du bli overveldet av å forsøke å lese dem alle. Ville det ikke vært fint om en bot kunne lese dem alle for deg, forstå dem og fortelle deg i hvilken bunke hver e-post hørte hjemme? 
> 
> En måte å oppnå dette på er å bruke maskinlæring. Du ville trene modellen med en del av *imot*-e-postene og en del av *for*-e-postene. Modellen ville ha en tendens til å assosiere fraser og ord med imot-siden og for-siden, *men den ville ikke forstå noe av innholdet*, bare at visse ord og mønstre var mer sannsynlige å dukke opp i en *imot*- eller en *for*-e-post. Du kunne teste den med noen e-poster som du ikke hadde brukt til å trene modellen, og se om den kom til samme konklusjon som deg. Deretter, når du var fornøyd med modellens nøyaktighet, kunne du behandle fremtidige e-poster uten å måtte lese hver enkelt.

✅ Høres denne prosessen ut som prosesser du har brukt i tidligere leksjoner?

## Øvelse - sentimentale setninger

Sentiment måles med en *polarity* fra -1 til 1, der -1 er den mest negative stemningen, og 1 er den mest positive. Sentiment måles også med en score fra 0 til 1 for objektivitet (0) og subjektivitet (1).

Ta en ny titt på Jane Austens *Pride and Prejudice*. Teksten er tilgjengelig her på [Project Gutenberg](https://www.gutenberg.org/files/1342/1342-h/1342-h.htm). Eksemplet nedenfor viser et kort program som analyserer sentimentet i første og siste setning fra boken og viser dens sentimentpolarity og subjektivitets-/objektivitets-score.

Du bør bruke `TextBlob`-biblioteket (beskrevet ovenfor) for å bestemme `sentiment` (du trenger ikke skrive din egen sentimentkalkulator) i følgende oppgave.

```python
from textblob import TextBlob

quote1 = """It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want of a wife."""

quote2 = """Darcy, as well as Elizabeth, really loved them; and they were both ever sensible of the warmest gratitude towards the persons who, by bringing her into Derbyshire, had been the means of uniting them."""

sentiment1 = TextBlob(quote1).sentiment
sentiment2 = TextBlob(quote2).sentiment

print(quote1 + " has a sentiment of " + str(sentiment1))
print(quote2 + " has a sentiment of " + str(sentiment2))
```

Du ser følgende utdata:

```output
It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want # of a wife. has a sentiment of Sentiment(polarity=0.20952380952380953, subjectivity=0.27142857142857146)

Darcy, as well as Elizabeth, really loved them; and they were
     both ever sensible of the warmest gratitude towards the persons
      who, by bringing her into Derbyshire, had been the means of
      uniting them. has a sentiment of Sentiment(polarity=0.7, subjectivity=0.8)
```

## Utfordring - sjekk sentimentpolarity

Din oppgave er å avgjøre, ved hjelp av sentimentpolarity, om *Pride and Prejudice* har flere absolutt positive setninger enn absolutt negative. For denne oppgaven kan du anta at en polarity-score på 1 eller -1 er henholdsvis absolutt positiv eller negativ.

**Trinn:**

1. Last ned en [kopi av Pride and Prejudice](https://www.gutenberg.org/files/1342/1342-h/1342-h.htm) fra Project Gutenberg som en .txt-fil. Fjern metadataene i starten og slutten av filen, slik at bare originalteksten gjenstår.
2. Åpne filen i Python og trekk ut innholdet som en streng.
3. Lag en TextBlob ved hjelp av bokstrengen.
4. Analyser hver setning i boken i en løkke.
   1. Hvis polariteten er 1 eller -1, lagre setningen i en liste over positive eller negative meldinger.
5. Til slutt, skriv ut alle de positive og negative setningene (separat) og antallet av hver.

Her er en [eksempelløsning](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/3-Translation-Sentiment/solution/notebook.ipynb).

✅ Kunnskapssjekk

1. Sentimentet er basert på ordene som brukes i setningen, men forstår koden *ordene*?
2. Synes du sentimentpolarityen er nøyaktig, eller med andre ord, er du *enig* i scorene?
   1. Spesielt, er du enig eller uenig i den absolutte **positive** polariteten til følgende setninger?
      * “What an excellent father you have, girls!” said she, when the door was shut.
      * “Your examination of Mr. Darcy is over, I presume,” said Miss Bingley; “and pray what is the result?” “I am perfectly convinced by it that Mr. Darcy has no defect.
      * How wonderfully these sort of things occur!
      * I have the greatest dislike in the world to that sort of thing.
      * Charlotte is an excellent manager, I dare say.
      * “This is delightful indeed!
      * I am so happy!
      * Your idea of the ponies is delightful.
   2. De neste 3 setningene ble vurdert med en absolutt positiv sentiment, men ved nærmere lesing er de ikke positive setninger. Hvorfor trodde sentimentanalysen at de var positive setninger?
      * Happy shall I be, when his stay at Netherfield is over!” “I wish I could say anything to comfort you,” replied Elizabeth; “but it is wholly out of my power.
      * If I could but see you as happy!
      * Our distress, my dear Lizzy, is very great.
   3. Er du enig eller uenig i den absolutte **negative** polariteten til følgende setninger?
      - Everybody is disgusted with his pride.
      - “I should like to know how he behaves among strangers.” “You shall hear then—but prepare yourself for something very dreadful.
      - The pause was to Elizabeth’s feelings dreadful.
      - It would be dreadful!

✅ Enhver kjenner av Jane Austen vil forstå at hun ofte bruker bøkene sine til å kritisere de mer latterlige aspektene ved det engelske regentsamfunnet. Elizabeth Bennett, hovedpersonen i *Pride and Prejudice*, er en skarp sosial observatør (som forfatteren), og språket hennes er ofte sterkt nyansert. Selv Mr. Darcy (kjærlighetsinteressen i historien) bemerker Elizabeths lekne og ertende bruk av språk: "I have had the pleasure of your acquaintance long enough to know that you find great enjoyment in occasionally professing opinions which in fact are not your own."

---

## 🚀Utfordring

Kan du gjøre Marvin enda bedre ved å trekke ut andre funksjoner fra brukerens input?

## [Quiz etter forelesning](https://ff-quizzes.netlify.app/en/ml/)

## Gjennomgang og selvstudium
Det finnes mange måter å trekke ut sentiment fra tekst på. Tenk på forretningsapplikasjoner som kan dra nytte av denne teknikken. Tenk også på hvordan det kan gå galt. Les mer om sofistikerte, bedriftsklare systemer som analyserer sentiment, som for eksempel [Azure Text Analysis](https://docs.microsoft.com/azure/cognitive-services/Text-Analytics/how-tos/text-analytics-how-to-sentiment-analysis?tabs=version-3-1?WT.mc_id=academic-77952-leestott). Test noen av setningene fra Stolthet og fordom ovenfor, og se om det kan oppdage nyanser.

## Oppgave

[Poetisk frihet](assignment.md)

---

**Ansvarsfraskrivelse**:  
Dette dokumentet er oversatt ved hjelp av AI-oversettelsestjenesten [Co-op Translator](https://github.com/Azure/co-op-translator). Selv om vi tilstreber nøyaktighet, vennligst vær oppmerksom på at automatiske oversettelser kan inneholde feil eller unøyaktigheter. Det originale dokumentet på sitt opprinnelige språk bør anses som den autoritative kilden. For kritisk informasjon anbefales profesjonell menneskelig oversettelse. Vi er ikke ansvarlige for eventuelle misforståelser eller feiltolkninger som oppstår ved bruk av denne oversettelsen.