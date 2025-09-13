<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "be03c8182982b87ced155e4e9d1438e8",
  "translation_date": "2025-09-05T20:38:21+00:00",
  "source_file": "6-NLP/3-Translation-Sentiment/README.md",
  "language_code": "nl"
}
-->
# Vertaling en sentimentanalyse met ML

In de vorige lessen heb je geleerd hoe je een eenvoudige bot kunt bouwen met `TextBlob`, een bibliotheek die machine learning achter de schermen gebruikt om basis NLP-taken uit te voeren, zoals het extraheren van zelfstandige naamwoordgroepen. Een andere belangrijke uitdaging in de computationele taalkunde is het nauwkeurig _vertalen_ van een zin van de ene gesproken of geschreven taal naar de andere.

## [Pre-lecture quiz](https://ff-quizzes.netlify.app/en/ml/)

Vertalen is een zeer complex probleem, mede door het feit dat er duizenden talen zijn, elk met zeer verschillende grammaticaregels. Een benadering is om de formele grammaticaregels van een taal, zoals Engels, om te zetten in een structuur die niet afhankelijk is van een specifieke taal, en deze vervolgens te vertalen door terug te converteren naar een andere taal. Deze aanpak omvat de volgende stappen:

1. **Identificatie**. Identificeer of label de woorden in de invoertaal als zelfstandige naamwoorden, werkwoorden, enz.
2. **Maak een vertaling**. Maak een directe vertaling van elk woord in het formaat van de doeltaal.

### Voorbeeldzin, Engels naar Iers

In het 'Engels' bestaat de zin _I feel happy_ uit drie woorden in de volgorde:

- **onderwerp** (I)
- **werkwoord** (feel)
- **bijvoeglijk naamwoord** (happy)

In de 'Ierse' taal heeft dezelfde zin echter een heel andere grammaticale structuur - emoties zoals "*happy*" of "*sad*" worden uitgedrukt als iets dat *op* jou is.

De Engelse zin `I feel happy` zou in het Iers worden vertaald als `Tá athas orm`. Een *letterlijke* vertaling zou zijn `Happy is upon me`.

Een Iers spreker die naar het Engels vertaalt, zou zeggen `I feel happy`, niet `Happy is upon me`, omdat ze de betekenis van de zin begrijpen, zelfs als de woorden en zinsstructuur anders zijn.

De formele volgorde voor de zin in het Iers is:

- **werkwoord** (Tá of is)
- **bijvoeglijk naamwoord** (athas, of happy)
- **onderwerp** (orm, of upon me)

## Vertaling

Een naïef vertaalprogramma zou alleen woorden vertalen en de zinsstructuur negeren.

✅ Als je als volwassene een tweede (of derde of meer) taal hebt geleerd, ben je misschien begonnen met denken in je moedertaal, waarbij je een concept woord voor woord in je hoofd naar de tweede taal vertaalt en vervolgens je vertaling uitspreekt. Dit lijkt op wat naïeve vertaalprogramma's doen. Het is belangrijk om voorbij deze fase te komen om vloeiendheid te bereiken!

Naïeve vertaling leidt tot slechte (en soms hilarische) mistranslaties: `I feel happy` wordt letterlijk vertaald naar `Mise bhraitheann athas` in het Iers. Dat betekent (letterlijk) `me feel happy` en is geen geldige Ierse zin. Hoewel Engels en Iers talen zijn die worden gesproken op twee dicht bij elkaar gelegen eilanden, zijn het zeer verschillende talen met verschillende grammaticale structuren.

> Je kunt enkele video's bekijken over Ierse taalkundige tradities, zoals [deze](https://www.youtube.com/watch?v=mRIaLSdRMMs)

### Machine learning benaderingen

Tot nu toe heb je geleerd over de formele regelsbenadering van natuurlijke taalverwerking. Een andere benadering is om de betekenis van de woorden te negeren en _in plaats daarvan machine learning te gebruiken om patronen te detecteren_. Dit kan werken bij vertaling als je veel tekst (een *corpus*) of teksten (*corpora*) hebt in zowel de oorspronkelijke als de doeltaal.

Bijvoorbeeld, neem het geval van *Pride and Prejudice*, een bekende Engelse roman geschreven door Jane Austen in 1813. Als je het boek in het Engels en een menselijke vertaling van het boek in het *Frans* raadpleegt, kun je zinnen in de ene taal detecteren die _idiomatisch_ zijn vertaald naar de andere. Dat ga je zo doen.

Bijvoorbeeld, wanneer een Engelse zin zoals `I have no money` letterlijk wordt vertaald naar het Frans, kan het worden `Je n'ai pas de monnaie`. "Monnaie" is een lastig Frans 'false cognate', omdat 'money' en 'monnaie' niet synoniem zijn. Een betere vertaling die een mens zou maken, zou zijn `Je n'ai pas d'argent`, omdat dit beter de betekenis overbrengt dat je geen geld hebt (in plaats van 'kleingeld', wat de betekenis van 'monnaie' is).

![monnaie](../../../../6-NLP/3-Translation-Sentiment/images/monnaie.png)

> Afbeelding door [Jen Looper](https://twitter.com/jenlooper)

Als een ML-model genoeg menselijke vertalingen heeft om een model op te bouwen, kan het de nauwkeurigheid van vertalingen verbeteren door veelvoorkomende patronen te identificeren in teksten die eerder zijn vertaald door deskundige menselijke sprekers van beide talen.

### Oefening - vertaling

Je kunt `TextBlob` gebruiken om zinnen te vertalen. Probeer de beroemde eerste zin van **Pride and Prejudice**:

```python
from textblob import TextBlob

blob = TextBlob(
    "It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want of a wife!"
)
print(blob.translate(to="fr"))

```

`TextBlob` doet een behoorlijk goede vertaling: "C'est une vérité universellement reconnue, qu'un homme célibataire en possession d'une bonne fortune doit avoir besoin d'une femme!". 

Er kan worden gesteld dat de vertaling van TextBlob veel nauwkeuriger is dan de Franse vertaling van het boek uit 1932 door V. Leconte en Ch. Pressoir:

"C'est une vérité universelle qu'un célibataire pourvu d'une belle fortune doit avoir envie de se marier, et, si peu que l'on sache de son sentiment à cet egard, lorsqu'il arrive dans une nouvelle résidence, cette idée est si bien fixée dans l'esprit de ses voisins qu'ils le considèrent sur-le-champ comme la propriété légitime de l'une ou l'autre de leurs filles."

In dit geval doet de door ML geïnformeerde vertaling het beter dan de menselijke vertaler, die onnodig woorden in de mond van de oorspronkelijke auteur legt voor 'duidelijkheid'.

> Wat gebeurt hier? En waarom is TextBlob zo goed in vertalen? Wel, achter de schermen gebruikt het Google Translate, een geavanceerde AI die miljoenen zinnen kan analyseren om de beste strings voor de taak te voorspellen. Er gebeurt hier niets handmatig en je hebt een internetverbinding nodig om `blob.translate` te gebruiken.

✅ Probeer enkele andere zinnen. Welke is beter, ML of menselijke vertaling? In welke gevallen?

## Sentimentanalyse

Een ander gebied waar machine learning zeer goed kan werken, is sentimentanalyse. Een niet-ML-benadering van sentiment is het identificeren van woorden en zinnen die 'positief' en 'negatief' zijn. Vervolgens, gegeven een nieuw stuk tekst, bereken je de totale waarde van de positieve, negatieve en neutrale woorden om het algemene sentiment te identificeren.

Deze aanpak is gemakkelijk te misleiden, zoals je misschien hebt gezien in de Marvin-taak - de zin `Great, that was a wonderful waste of time, I'm glad we are lost on this dark road` is een sarcastische, negatieve zin, maar het eenvoudige algoritme detecteert 'great', 'wonderful', 'glad' als positief en 'waste', 'lost' en 'dark' als negatief. Het algemene sentiment wordt beïnvloed door deze tegenstrijdige woorden.

✅ Stop even en denk na over hoe we sarcasme overbrengen als menselijke sprekers. Toonhoogte speelt een grote rol. Probeer de zin "Well, that film was awesome" op verschillende manieren te zeggen om te ontdekken hoe je stem betekenis overbrengt.

### ML-benaderingen

De ML-benadering zou zijn om handmatig negatieve en positieve tekstverzamelingen te verzamelen - tweets, of filmrecensies, of alles waarbij de mens een score *en* een geschreven mening heeft gegeven. Vervolgens kunnen NLP-technieken worden toegepast op meningen en scores, zodat patronen ontstaan (bijvoorbeeld, positieve filmrecensies bevatten vaker de zin 'Oscar worthy' dan negatieve filmrecensies, of positieve restaurantrecensies zeggen 'gourmet' veel vaker dan 'disgusting').

> ⚖️ **Voorbeeld**: Stel dat je in het kantoor van een politicus werkt en er wordt een nieuwe wet besproken. Kiezers kunnen het kantoor e-mails sturen ter ondersteuning of tegen de nieuwe wet. Stel dat je de taak krijgt om de e-mails te lezen en ze in 2 stapels te sorteren, *voor* en *tegen*. Als er veel e-mails zijn, kun je overweldigd raken door ze allemaal te lezen. Zou het niet fijn zijn als een bot ze allemaal voor je kon lezen, begrijpen en je kon vertellen in welke stapel elke e-mail thuishoort? 
> 
> Een manier om dat te bereiken is door Machine Learning te gebruiken. Je zou het model trainen met een deel van de *tegen* e-mails en een deel van de *voor* e-mails. Het model zou neigen naar het associëren van zinnen en woorden met de tegenkant en de voorkant, *maar het zou geen enkele inhoud begrijpen*, alleen dat bepaalde woorden en patronen vaker voorkomen in een *tegen* of een *voor* e-mail. Je zou het kunnen testen met enkele e-mails die je niet hebt gebruikt om het model te trainen, en kijken of het tot dezelfde conclusie komt als jij. Zodra je tevreden bent met de nauwkeurigheid van het model, kun je toekomstige e-mails verwerken zonder ze allemaal te hoeven lezen.

✅ Klinkt dit proces als processen die je in eerdere lessen hebt gebruikt?

## Oefening - sentimentele zinnen

Sentiment wordt gemeten met een *polariteit* van -1 tot 1, waarbij -1 het meest negatieve sentiment is en 1 het meest positieve. Sentiment wordt ook gemeten met een score van 0 - 1 voor objectiviteit (0) en subjectiviteit (1).

Bekijk Jane Austen's *Pride and Prejudice* opnieuw. De tekst is hier beschikbaar op [Project Gutenberg](https://www.gutenberg.org/files/1342/1342-h/1342-h.htm). Het onderstaande voorbeeld toont een kort programma dat het sentiment van de eerste en laatste zinnen van het boek analyseert en de polariteit en subjectiviteit/objectiviteitsscore weergeeft.

Je moet de `TextBlob`-bibliotheek (hierboven beschreven) gebruiken om `sentiment` te bepalen (je hoeft geen eigen sentimentcalculator te schrijven) in de volgende taak.

```python
from textblob import TextBlob

quote1 = """It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want of a wife."""

quote2 = """Darcy, as well as Elizabeth, really loved them; and they were both ever sensible of the warmest gratitude towards the persons who, by bringing her into Derbyshire, had been the means of uniting them."""

sentiment1 = TextBlob(quote1).sentiment
sentiment2 = TextBlob(quote2).sentiment

print(quote1 + " has a sentiment of " + str(sentiment1))
print(quote2 + " has a sentiment of " + str(sentiment2))
```

Je ziet de volgende uitvoer:

```output
It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want # of a wife. has a sentiment of Sentiment(polarity=0.20952380952380953, subjectivity=0.27142857142857146)

Darcy, as well as Elizabeth, really loved them; and they were
     both ever sensible of the warmest gratitude towards the persons
      who, by bringing her into Derbyshire, had been the means of
      uniting them. has a sentiment of Sentiment(polarity=0.7, subjectivity=0.8)
```

## Uitdaging - controleer sentimentpolariteit

Je taak is om, met behulp van sentimentpolariteit, te bepalen of *Pride and Prejudice* meer absoluut positieve zinnen heeft dan absoluut negatieve. Voor deze taak kun je aannemen dat een polariteitsscore van 1 of -1 absoluut positief of negatief is.

**Stappen:**

1. Download een [kopie van Pride and Prejudice](https://www.gutenberg.org/files/1342/1342-h/1342-h.htm) van Project Gutenberg als een .txt-bestand. Verwijder de metadata aan het begin en einde van het bestand, zodat alleen de originele tekst overblijft.
2. Open het bestand in Python en haal de inhoud op als een string.
3. Maak een TextBlob met de boekstring.
4. Analyseer elke zin in het boek in een lus.
   1. Als de polariteit 1 of -1 is, sla de zin op in een array of lijst van positieve of negatieve berichten.
5. Print aan het einde alle positieve en negatieve zinnen (apart) en het aantal van elk.

Hier is een voorbeeld [oplossing](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/3-Translation-Sentiment/solution/notebook.ipynb).

✅ Kenniscontrole

1. Het sentiment is gebaseerd op woorden die in de zin worden gebruikt, maar begrijpt de code *de woorden*?
2. Denk je dat de sentimentpolariteit nauwkeurig is, of met andere woorden, ben je het *eens* met de scores?
   1. Ben je het in het bijzonder eens of oneens met de absolute **positieve** polariteit van de volgende zinnen?
      * “What an excellent father you have, girls!” said she, when the door was shut.
      * “Your examination of Mr. Darcy is over, I presume,” said Miss Bingley; “and pray what is the result?” “I am perfectly convinced by it that Mr. Darcy has no defect.
      * How wonderfully these sort of things occur!
      * I have the greatest dislike in the world to that sort of thing.
      * Charlotte is an excellent manager, I dare say.
      * “This is delightful indeed!
      * I am so happy!
      * Your idea of the ponies is delightful.
   2. De volgende 3 zinnen werden beoordeeld met een absolute positieve sentiment, maar bij nader inzien zijn het geen positieve zinnen. Waarom dacht de sentimentanalyse dat ze positieve zinnen waren?
      * Happy shall I be, when his stay at Netherfield is over!” “I wish I could say anything to comfort you,” replied Elizabeth; “but it is wholly out of my power.
      * If I could but see you as happy!
      * Our distress, my dear Lizzy, is very great.
   3. Ben je het eens of oneens met de absolute **negatieve** polariteit van de volgende zinnen?
      - Everybody is disgusted with his pride.
      - “I should like to know how he behaves among strangers.” “You shall hear then—but prepare yourself for something very dreadful.
      - The pause was to Elizabeth’s feelings dreadful.
      - It would be dreadful!

✅ Elke liefhebber van Jane Austen zal begrijpen dat ze haar boeken vaak gebruikt om de meer belachelijke aspecten van de Engelse Regency-samenleving te bekritiseren. Elizabeth Bennett, de hoofdpersoon in *Pride and Prejudice*, is een scherpe sociale waarnemer (zoals de auteur) en haar taalgebruik is vaak sterk genuanceerd. Zelfs Mr. Darcy (de liefde in het verhaal) merkt Elizabeth's speelse en plagerige gebruik van taal op: "I have had the pleasure of your acquaintance long enough to know that you find great enjoyment in occasionally professing opinions which in fact are not your own."

---

## 🚀Uitdaging

Kun je Marvin nog beter maken door andere kenmerken uit de gebruikersinvoer te extraheren?

## [Post-lecture quiz](https://ff-quizzes.netlify.app/en/ml/)

## Review & Zelfstudie
Er zijn veel manieren om sentiment uit tekst te halen. Denk aan de zakelijke toepassingen die gebruik kunnen maken van deze techniek. Denk ook na over hoe het mis kan gaan. Lees meer over geavanceerde, bedrijfsgerichte systemen die sentiment analyseren, zoals [Azure Text Analysis](https://docs.microsoft.com/azure/cognitive-services/Text-Analytics/how-tos/text-analytics-how-to-sentiment-analysis?tabs=version-3-1?WT.mc_id=academic-77952-leestott). Test enkele van de zinnen uit Pride and Prejudice hierboven en kijk of het nuances kan detecteren.

## Opdracht

[Poëtische vrijheid](assignment.md)

---

**Disclaimer**:  
Dit document is vertaald met behulp van de AI-vertalingsservice [Co-op Translator](https://github.com/Azure/co-op-translator). Hoewel we streven naar nauwkeurigheid, dient u zich ervan bewust te zijn dat geautomatiseerde vertalingen fouten of onnauwkeurigheden kunnen bevatten. Het originele document in zijn oorspronkelijke taal moet worden beschouwd als de gezaghebbende bron. Voor cruciale informatie wordt professionele menselijke vertaling aanbevolen. Wij zijn niet aansprakelijk voor eventuele misverstanden of verkeerde interpretaties die voortvloeien uit het gebruik van deze vertaling.