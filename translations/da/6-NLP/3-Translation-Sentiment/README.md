<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "be03c8182982b87ced155e4e9d1438e8",
  "translation_date": "2025-09-05T01:38:40+00:00",
  "source_file": "6-NLP/3-Translation-Sentiment/README.md",
  "language_code": "da"
}
-->
# Oversættelse og sentimentanalyse med ML

I de tidligere lektioner lærte du, hvordan man bygger en grundlæggende bot ved hjælp af `TextBlob`, et bibliotek, der integrerer ML bag kulisserne for at udføre grundlæggende NLP-opgaver som udtrækning af navneordssætninger. En anden vigtig udfordring inden for computerlingvistik er præcis _oversættelse_ af en sætning fra et talesprog eller skriftsprog til et andet.

## [Quiz før lektionen](https://ff-quizzes.netlify.app/en/ml/)

Oversættelse er et meget vanskeligt problem, som forværres af, at der findes tusindvis af sprog, og hvert sprog kan have meget forskellige grammatiske regler. En tilgang er at konvertere de formelle grammatiske regler for et sprog, som f.eks. engelsk, til en struktur, der er uafhængig af sprog, og derefter oversætte det ved at konvertere tilbage til et andet sprog. Denne tilgang indebærer følgende trin:

1. **Identifikation**. Identificer eller tag ordene i indgangssproget som navneord, verber osv.
2. **Skab oversættelse**. Lav en direkte oversættelse af hvert ord i målsprogets format.

### Eksempelsætning, engelsk til irsk

På 'engelsk' er sætningen _I feel happy_ tre ord i rækkefølgen:

- **subjekt** (I)
- **verbum** (feel)
- **adjektiv** (happy)

Men på 'irsk' har den samme sætning en meget anderledes grammatisk struktur - følelser som "*happy*" eller "*sad*" udtrykkes som værende *på* dig.

Den engelske sætning `I feel happy` på irsk ville være `Tá athas orm`. En *bogstavelig* oversættelse ville være `Happy is upon me`.

En irsktalende, der oversætter til engelsk, ville sige `I feel happy`, ikke `Happy is upon me`, fordi de forstår meningen med sætningen, selvom ordene og sætningsstrukturen er forskellige.

Den formelle rækkefølge for sætningen på irsk er:

- **verbum** (Tá eller is)
- **adjektiv** (athas, eller happy)
- **subjekt** (orm, eller upon me)

## Oversættelse

Et naivt oversættelsesprogram kunne oversætte ord alene og ignorere sætningsstrukturen.

✅ Hvis du har lært et andet (eller tredje eller flere) sprog som voksen, har du måske startet med at tænke på dit modersmål, oversætte et begreb ord for ord i dit hoved til det andet sprog og derefter tale din oversættelse. Dette ligner, hvad naive oversættelsesprogrammer gør. Det er vigtigt at komme forbi denne fase for at opnå flydende sprogkundskaber!

Naiv oversættelse fører til dårlige (og nogle gange morsomme) fejltolkninger: `I feel happy` oversættes bogstaveligt til `Mise bhraitheann athas` på irsk. Det betyder (bogstaveligt) `me feel happy` og er ikke en gyldig irsk sætning. Selvom engelsk og irsk er sprog, der tales på to nærtliggende øer, er de meget forskellige sprog med forskellige grammatiske strukturer.

> Du kan se nogle videoer om irske sproglige traditioner, såsom [denne](https://www.youtube.com/watch?v=mRIaLSdRMMs)

### Maskinlæringsmetoder

Indtil videre har du lært om den formelle regeltilgang til naturlig sprogbehandling. En anden tilgang er at ignorere ordenes betydning og _i stedet bruge maskinlæring til at opdage mønstre_. Dette kan fungere i oversættelse, hvis du har masser af tekst (et *corpus*) eller tekster (*corpora*) på både oprindelses- og målsproget.

For eksempel, overvej tilfældet med *Pride and Prejudice*, en velkendt engelsk roman skrevet af Jane Austen i 1813. Hvis du konsulterer bogen på engelsk og en menneskelig oversættelse af bogen på *fransk*, kunne du opdage fraser i den ene, der er _idiomatisk_ oversat til den anden. Det vil du gøre om lidt.

For eksempel, når en engelsk frase som `I have no money` oversættes bogstaveligt til fransk, kunne det blive `Je n'ai pas de monnaie`. "Monnaie" er et tricky fransk 'falsk cognate', da 'money' og 'monnaie' ikke er synonyme. En bedre oversættelse, som en menneskelig oversætter kunne lave, ville være `Je n'ai pas d'argent`, fordi det bedre formidler betydningen af, at du ikke har penge (i stedet for 'småpenge', som er betydningen af 'monnaie').

![monnaie](../../../../6-NLP/3-Translation-Sentiment/images/monnaie.png)

> Billede af [Jen Looper](https://twitter.com/jenlooper)

Hvis en ML-model har nok menneskelige oversættelser til at bygge en model på, kan den forbedre nøjagtigheden af oversættelser ved at identificere almindelige mønstre i tekster, der tidligere er blevet oversat af eksperttalere af begge sprog.

### Øvelse - oversættelse

Du kan bruge `TextBlob` til at oversætte sætninger. Prøv den berømte første linje fra **Pride and Prejudice**:

```python
from textblob import TextBlob

blob = TextBlob(
    "It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want of a wife!"
)
print(blob.translate(to="fr"))

```

`TextBlob` gør et ret godt stykke arbejde med oversættelsen: "C'est une vérité universellement reconnue, qu'un homme célibataire en possession d'une bonne fortune doit avoir besoin d'une femme!". 

Det kan argumenteres, at TextBlobs oversættelse er langt mere præcis, faktisk, end den franske oversættelse af bogen fra 1932 af V. Leconte og Ch. Pressoir:

"C'est une vérité universelle qu'un célibataire pourvu d'une belle fortune doit avoir envie de se marier, et, si peu que l'on sache de son sentiment à cet egard, lorsqu'il arrive dans une nouvelle résidence, cette idée est si bien fixée dans l'esprit de ses voisins qu'ils le considèrent sur-le-champ comme la propriété légitime de l'une ou l'autre de leurs filles."

I dette tilfælde gør oversættelsen informeret af ML et bedre arbejde end den menneskelige oversætter, der unødvendigt lægger ord i den oprindelige forfatters mund for 'klarhed'.

> Hvad sker der her? Og hvorfor er TextBlob så god til oversættelse? Bag kulisserne bruger det Google Translate, en sofistikeret AI, der er i stand til at analysere millioner af fraser for at forudsige de bedste strenge til den aktuelle opgave. Der foregår intet manuelt her, og du skal have en internetforbindelse for at bruge `blob.translate`.

✅ Prøv nogle flere sætninger. Hvilken er bedre, ML eller menneskelig oversættelse? I hvilke tilfælde?

## Sentimentanalyse

Et andet område, hvor maskinlæring kan fungere meget godt, er sentimentanalyse. En ikke-ML tilgang til sentiment er at identificere ord og fraser, der er 'positive' og 'negative'. Derefter, givet et nyt stykke tekst, beregne den samlede værdi af de positive, negative og neutrale ord for at identificere den overordnede sentiment. 

Denne tilgang kan nemt narres, som du måske har set i Marvin-opgaven - sætningen `Great, that was a wonderful waste of time, I'm glad we are lost on this dark road` er en sarkastisk, negativ sentiment sætning, men den simple algoritme registrerer 'great', 'wonderful', 'glad' som positive og 'waste', 'lost' og 'dark' som negative. Den overordnede sentiment påvirkes af disse modstridende ord.

✅ Stop et øjeblik og tænk over, hvordan vi som menneskelige talere formidler sarkasme. Tonefald spiller en stor rolle. Prøv at sige sætningen "Well, that film was awesome" på forskellige måder for at opdage, hvordan din stemme formidler mening.

### ML-tilgange

ML-tilgangen ville være manuelt at samle negative og positive tekstkroppe - tweets, eller filmanmeldelser, eller hvad som helst, hvor mennesket har givet en score *og* en skriftlig mening. Derefter kan NLP-teknikker anvendes på meninger og scores, så mønstre opstår (f.eks. positive filmanmeldelser har tendens til at indeholde frasen 'Oscar worthy' mere end negative filmanmeldelser, eller positive restaurantanmeldelser siger 'gourmet' meget mere end 'disgusting').

> ⚖️ **Eksempel**: Hvis du arbejdede i en politikers kontor, og der var en ny lov, der blev debatteret, kunne vælgere skrive til kontoret med e-mails, der støtter eller er imod den pågældende nye lov. Lad os sige, at du får til opgave at læse e-mails og sortere dem i 2 bunker, *for* og *imod*. Hvis der var mange e-mails, kunne du blive overvældet af at forsøge at læse dem alle. Ville det ikke være rart, hvis en bot kunne læse dem alle for dig, forstå dem og fortælle dig, hvilken bunke hver e-mail hørte til? 
> 
> En måde at opnå dette på er at bruge maskinlæring. Du ville træne modellen med en del af de *imod* e-mails og en del af de *for* e-mails. Modellen ville have tendens til at associere fraser og ord med imod-siden og for-siden, *men den ville ikke forstå noget af indholdet*, kun at visse ord og mønstre var mere tilbøjelige til at optræde i en *imod* eller en *for* e-mail. Du kunne teste det med nogle e-mails, som du ikke havde brugt til at træne modellen, og se, om den kom til samme konklusion som dig. Derefter, når du var tilfreds med modellens nøjagtighed, kunne du behandle fremtidige e-mails uden at skulle læse hver enkelt.

✅ Lyder denne proces som processer, du har brugt i tidligere lektioner?

## Øvelse - sentimentale sætninger

Sentiment måles med en *polarity* fra -1 til 1, hvilket betyder, at -1 er den mest negative sentiment, og 1 er den mest positive. Sentiment måles også med en score fra 0 - 1 for objektivitet (0) og subjektivitet (1).

Tag et nyt kig på Jane Austens *Pride and Prejudice*. Teksten er tilgængelig her på [Project Gutenberg](https://www.gutenberg.org/files/1342/1342-h/1342-h.htm). Eksemplet nedenfor viser et kort program, der analyserer sentimentet af første og sidste sætning fra bogen og viser dens sentimentpolarity og subjektivitets-/objektivitets-score.

Du skal bruge `TextBlob`-biblioteket (beskrevet ovenfor) til at bestemme `sentiment` (du behøver ikke at skrive din egen sentimentberegner) i den følgende opgave.

```python
from textblob import TextBlob

quote1 = """It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want of a wife."""

quote2 = """Darcy, as well as Elizabeth, really loved them; and they were both ever sensible of the warmest gratitude towards the persons who, by bringing her into Derbyshire, had been the means of uniting them."""

sentiment1 = TextBlob(quote1).sentiment
sentiment2 = TextBlob(quote2).sentiment

print(quote1 + " has a sentiment of " + str(sentiment1))
print(quote2 + " has a sentiment of " + str(sentiment2))
```

Du ser følgende output:

```output
It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want # of a wife. has a sentiment of Sentiment(polarity=0.20952380952380953, subjectivity=0.27142857142857146)

Darcy, as well as Elizabeth, really loved them; and they were
     both ever sensible of the warmest gratitude towards the persons
      who, by bringing her into Derbyshire, had been the means of
      uniting them. has a sentiment of Sentiment(polarity=0.7, subjectivity=0.8)
```

## Udfordring - tjek sentimentpolarity

Din opgave er at afgøre, ved hjælp af sentimentpolarity, om *Pride and Prejudice* har flere absolut positive sætninger end absolut negative. Til denne opgave kan du antage, at en polarity-score på 1 eller -1 er absolut positiv eller negativ henholdsvis.

**Trin:**

1. Download en [kopi af Pride and Prejudice](https://www.gutenberg.org/files/1342/1342-h/1342-h.htm) fra Project Gutenberg som en .txt-fil. Fjern metadataene i starten og slutningen af filen, så kun den originale tekst er tilbage.
2. Åbn filen i Python og udtræk indholdet som en streng.
3. Opret en TextBlob ved hjælp af bogstrengen.
4. Analyser hver sætning i bogen i en løkke.
   1. Hvis polariteten er 1 eller -1, gem sætningen i en array eller liste over positive eller negative beskeder.
5. Til sidst, udskriv alle de positive sætninger og negative sætninger (separat) og antallet af hver.

Her er en [eksempelløsning](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/3-Translation-Sentiment/solution/notebook.ipynb).

✅ Videnstjek

1. Sentimentet er baseret på ord, der bruges i sætningen, men forstår koden *ordene*?
2. Synes du, at sentimentpolarity er præcis, eller med andre ord, er du *enig* i scores?
   1. Især, er du enig eller uenig i den absolutte **positive** polaritet af følgende sætninger?
      * “What an excellent father you have, girls!” said she, when the door was shut.
      * “Your examination of Mr. Darcy is over, I presume,” said Miss Bingley; “and pray what is the result?” “I am perfectly convinced by it that Mr. Darcy has no defect.
      * How wonderfully these sort of things occur!
      * I have the greatest dislike in the world to that sort of thing.
      * Charlotte is an excellent manager, I dare say.
      * “This is delightful indeed!
      * I am so happy!
      * Your idea of the ponies is delightful.
   2. De næste 3 sætninger blev scoret med en absolut positiv sentiment, men ved nærmere læsning er de ikke positive sætninger. Hvorfor troede sentimentanalysen, at de var positive sætninger?
      * Happy shall I be, when his stay at Netherfield is over!” “I wish I could say anything to comfort you,” replied Elizabeth; “but it is wholly out of my power.
      * If I could but see you as happy!
      * Our distress, my dear Lizzy, is very great.
   3. Er du enig eller uenig i den absolutte **negative** polaritet af følgende sætninger?
      - Everybody is disgusted with his pride.
      - “I should like to know how he behaves among strangers.” “You shall hear then—but prepare yourself for something very dreadful.
      - The pause was to Elizabeth’s feelings dreadful.
      - It would be dreadful!

✅ Enhver Jane Austen-entusiast vil forstå, at hun ofte bruger sine bøger til at kritisere de mere latterlige aspekter af det engelske regency-samfund. Elizabeth Bennett, hovedpersonen i *Pride and Prejudice*, er en skarp social observatør (ligesom forfatteren), og hendes sprog er ofte stærkt nuanceret. Selv Mr. Darcy (kærlighedsinteressen i historien) bemærker Elizabeths legende og drilske brug af sprog: "I have had the pleasure of your acquaintance long enough to know that you find great enjoyment in occasionally professing opinions which in fact are not your own."

---

## 🚀Udfordring

Kan du gøre Marvin endnu bedre ved at udtrække andre funktioner fra brugerinput?

## [Quiz efter lektionen](https://ff-quizzes.netlify.app/en/ml/)

## Gennemgang & Selvstudie
Der er mange måder at udtrække sentiment fra tekst. Tænk på de forretningsapplikationer, der kunne gøre brug af denne teknik. Tænk over, hvordan det kan gå galt. Læs mere om avancerede, virksomhedsparate systemer, der analyserer sentiment, såsom [Azure Text Analysis](https://docs.microsoft.com/azure/cognitive-services/Text-Analytics/how-tos/text-analytics-how-to-sentiment-analysis?tabs=version-3-1?WT.mc_id=academic-77952-leestott). Test nogle af sætningerne fra "Stolthed og fordom" ovenfor og se, om det kan opfange nuancer.

## Opgave

[Digterisk frihed](assignment.md)

---

**Ansvarsfraskrivelse**:  
Dette dokument er blevet oversat ved hjælp af AI-oversættelsestjenesten [Co-op Translator](https://github.com/Azure/co-op-translator). Selvom vi bestræber os på nøjagtighed, skal du være opmærksom på, at automatiserede oversættelser kan indeholde fejl eller unøjagtigheder. Det originale dokument på dets oprindelige sprog bør betragtes som den autoritative kilde. For kritisk information anbefales professionel menneskelig oversættelse. Vi påtager os ikke ansvar for eventuelle misforståelser eller fejltolkninger, der opstår som følge af brugen af denne oversættelse.