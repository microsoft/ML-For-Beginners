<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "5f3cb462e3122e1afe7ab0050ccf2bd3",
  "translation_date": "2025-09-05T20:25:03+00:00",
  "source_file": "6-NLP/2-Tasks/README.md",
  "language_code": "nl"
}
-->
# Veelvoorkomende taken en technieken in natuurlijke taalverwerking

Voor de meeste *natuurlijke taalverwerking*-taken moet de te verwerken tekst worden opgesplitst, geanalyseerd en de resultaten worden opgeslagen of vergeleken met regels en datasets. Deze taken stellen de programmeur in staat om de _betekenis_ of _intentie_ of alleen de _frequentie_ van termen en woorden in een tekst af te leiden.

## [Quiz voorafgaand aan de les](https://ff-quizzes.netlify.app/en/ml/)

Laten we de veelvoorkomende technieken ontdekken die worden gebruikt bij het verwerken van tekst. Gecombineerd met machine learning helpen deze technieken je om grote hoeveelheden tekst efficiënt te analyseren. Voordat je ML toepast op deze taken, is het echter belangrijk om de problemen te begrijpen waarmee een NLP-specialist te maken krijgt.

## Veelvoorkomende NLP-taken

Er zijn verschillende manieren om een tekst te analyseren waarmee je werkt. Er zijn taken die je kunt uitvoeren en door deze taken uit te voeren kun je een begrip van de tekst krijgen en conclusies trekken. Meestal voer je deze taken in een bepaalde volgorde uit.

### Tokenisatie

Waarschijnlijk is het eerste wat de meeste NLP-algoritmen moeten doen, het splitsen van de tekst in tokens of woorden. Hoewel dit eenvoudig klinkt, kan het lastig zijn om rekening te houden met interpunctie en de verschillende woord- en zinsgrenzen van talen. Je moet mogelijk verschillende methoden gebruiken om de scheidingen te bepalen.

![tokenisatie](../../../../6-NLP/2-Tasks/images/tokenization.png)
> Tokeniseren van een zin uit **Pride and Prejudice**. Infographic door [Jen Looper](https://twitter.com/jenlooper)

### Embeddings

[Woordembeddings](https://wikipedia.org/wiki/Word_embedding) zijn een manier om je tekstgegevens numeriek te converteren. Embeddings worden zo uitgevoerd dat woorden met een vergelijkbare betekenis of woorden die samen worden gebruikt, bij elkaar clusteren.

![woordembeddings](../../../../6-NLP/2-Tasks/images/embedding.png)
> "I have the highest respect for your nerves, they are my old friends." - Woordembeddings voor een zin uit **Pride and Prejudice**. Infographic door [Jen Looper](https://twitter.com/jenlooper)

✅ Probeer [deze interessante tool](https://projector.tensorflow.org/) om te experimenteren met woordembeddings. Door op een woord te klikken, worden clusters van vergelijkbare woorden weergegeven: 'toy' clustert met 'disney', 'lego', 'playstation' en 'console'.

### Parsing & Part-of-speech Tagging

Elk woord dat is getokeniseerd kan worden getagd als een deel van de spraak - een zelfstandig naamwoord, werkwoord of bijvoeglijk naamwoord. De zin `the quick red fox jumped over the lazy brown dog` kan bijvoorbeeld worden getagd als fox = zelfstandig naamwoord, jumped = werkwoord.

![parsing](../../../../6-NLP/2-Tasks/images/parse.png)

> Parsing van een zin uit **Pride and Prejudice**. Infographic door [Jen Looper](https://twitter.com/jenlooper)

Parsing houdt in dat je herkent welke woorden in een zin met elkaar verbonden zijn - bijvoorbeeld `the quick red fox jumped` is een bijvoeglijk naamwoord-zelfstandig naamwoord-werkwoordreeks die losstaat van de `lazy brown dog`-reeks.  

### Woord- en zinsfrequenties

Een nuttige procedure bij het analyseren van een grote hoeveelheid tekst is het opbouwen van een woordenboek van elk woord of elke zin van belang en hoe vaak deze voorkomt. De zin `the quick red fox jumped over the lazy brown dog` heeft bijvoorbeeld een woordfrequentie van 2 voor het woord 'the'.

Laten we een voorbeeldtekst bekijken waarin we de frequentie van woorden tellen. Rudyard Kipling's gedicht The Winners bevat het volgende vers:

```output
What the moral? Who rides may read.
When the night is thick and the tracks are blind
A friend at a pinch is a friend, indeed,
But a fool to wait for the laggard behind.
Down to Gehenna or up to the Throne,
He travels the fastest who travels alone.
```

Omdat zinsfrequenties hoofdlettergevoelig of hoofdletterongevoelig kunnen zijn, heeft de zin `a friend` een frequentie van 2, `the` een frequentie van 6 en `travels` een frequentie van 2.

### N-grams

Een tekst kan worden opgesplitst in reeksen van woorden van een bepaalde lengte: een enkel woord (unigram), twee woorden (bigrammen), drie woorden (trigrammen) of een willekeurig aantal woorden (n-grams).

Bijvoorbeeld, `the quick red fox jumped over the lazy brown dog` met een n-gram score van 2 produceert de volgende n-grams:

1. the quick 
2. quick red 
3. red fox
4. fox jumped 
5. jumped over 
6. over the 
7. the lazy 
8. lazy brown 
9. brown dog

Het kan gemakkelijker zijn om het te visualiseren als een schuivend venster over de zin. Hier is het voor n-grams van 3 woorden, waarbij het n-gram vetgedrukt is in elke zin:

1.   <u>**the quick red**</u> fox jumped over the lazy brown dog
2.   the **<u>quick red fox</u>** jumped over the lazy brown dog
3.   the quick **<u>red fox jumped</u>** over the lazy brown dog
4.   the quick red **<u>fox jumped over</u>** the lazy brown dog
5.   the quick red fox **<u>jumped over the</u>** lazy brown dog
6.   the quick red fox jumped **<u>over the lazy</u>** brown dog
7.   the quick red fox jumped over <u>**the lazy brown**</u> dog
8.   the quick red fox jumped over the **<u>lazy brown dog</u>**

![n-grams schuivend venster](../../../../6-NLP/2-Tasks/images/n-grams.gif)

> N-gram waarde van 3: Infographic door [Jen Looper](https://twitter.com/jenlooper)

### Zelfstandige naamwoordzinnen extractie

In de meeste zinnen is er een zelfstandig naamwoord dat het onderwerp of object van de zin is. In het Engels is dit vaak te herkennen aan 'a', 'an' of 'the' ervoor. Het identificeren van het onderwerp of object van een zin door 'de zelfstandig naamwoordzin te extraheren' is een veelvoorkomende taak in NLP bij het proberen de betekenis van een zin te begrijpen.

✅ In de zin "I cannot fix on the hour, or the spot, or the look or the words, which laid the foundation. It is too long ago. I was in the middle before I knew that I had begun.", kun je de zelfstandig naamwoordzinnen identificeren?

In de zin `the quick red fox jumped over the lazy brown dog` zijn er 2 zelfstandig naamwoordzinnen: **quick red fox** en **lazy brown dog**.

### Sentimentanalyse

Een zin of tekst kan worden geanalyseerd op sentiment, of hoe *positief* of *negatief* het is. Sentiment wordt gemeten in *polariteit* en *objectiviteit/subjectiviteit*. Polariteit wordt gemeten van -1.0 tot 1.0 (negatief tot positief) en 0.0 tot 1.0 (meest objectief tot meest subjectief).

✅ Later leer je dat er verschillende manieren zijn om sentiment te bepalen met behulp van machine learning, maar een manier is om een lijst van woorden en zinnen te hebben die door een menselijke expert als positief of negatief zijn gecategoriseerd en dat model toe te passen op tekst om een polariteitsscore te berekenen. Kun je zien hoe dit in sommige gevallen zou werken en in andere minder goed?

### Verbuiging

Verbuiging stelt je in staat om een woord te nemen en de enkelvoudsvorm of meervoudsvorm van het woord te krijgen.

### Lemmatisatie

Een *lemma* is de stam of het hoofdwoord voor een reeks woorden, bijvoorbeeld *flew*, *flies*, *flying* hebben een lemma van het werkwoord *fly*.

Er zijn ook nuttige databases beschikbaar voor de NLP-onderzoeker, met name:

### WordNet

[WordNet](https://wordnet.princeton.edu/) is een database van woorden, synoniemen, antoniemen en vele andere details voor elk woord in veel verschillende talen. Het is ongelooflijk nuttig bij het proberen vertalingen, spellingscontrole of taaltools van welke aard dan ook te bouwen.

## NLP-bibliotheken

Gelukkig hoef je niet al deze technieken zelf te bouwen, want er zijn uitstekende Python-bibliotheken beschikbaar die het veel toegankelijker maken voor ontwikkelaars die niet gespecialiseerd zijn in natuurlijke taalverwerking of machine learning. De volgende lessen bevatten meer voorbeelden hiervan, maar hier leer je enkele nuttige voorbeelden om je te helpen bij de volgende taak.

### Oefening - gebruik van de `TextBlob`-bibliotheek

Laten we een bibliotheek genaamd TextBlob gebruiken, omdat deze handige API's bevat voor het aanpakken van dit soort taken. TextBlob "staat op de schouders van giganten zoals [NLTK](https://nltk.org) en [pattern](https://github.com/clips/pattern), en werkt goed samen met beide." Het heeft een aanzienlijke hoeveelheid ML ingebouwd in zijn API.

> Opmerking: Een nuttige [Quick Start](https://textblob.readthedocs.io/en/dev/quickstart.html#quickstart) gids is beschikbaar voor TextBlob en wordt aanbevolen voor ervaren Python-ontwikkelaars.

Bij het proberen *zelfstandige naamwoordzinnen* te identificeren, biedt TextBlob verschillende opties van extractors om zelfstandig naamwoordzinnen te vinden.

1. Bekijk `ConllExtractor`.

    ```python
    from textblob import TextBlob
    from textblob.np_extractors import ConllExtractor
    # import and create a Conll extractor to use later 
    extractor = ConllExtractor()
    
    # later when you need a noun phrase extractor:
    user_input = input("> ")
    user_input_blob = TextBlob(user_input, np_extractor=extractor)  # note non-default extractor specified
    np = user_input_blob.noun_phrases                                    
    ```

    > Wat gebeurt hier? [ConllExtractor](https://textblob.readthedocs.io/en/dev/api_reference.html?highlight=Conll#textblob.en.np_extractors.ConllExtractor) is "Een zelfstandig naamwoordzin-extractor die chunk parsing gebruikt, getraind met de ConLL-2000 trainingscorpus." ConLL-2000 verwijst naar de 2000 Conference on Computational Natural Language Learning. Elk jaar organiseerde de conferentie een workshop om een lastig NLP-probleem aan te pakken, en in 2000 was dat zelfstandig naamwoordchunking. Een model werd getraind op de Wall Street Journal, met "secties 15-18 als trainingsdata (211727 tokens) en sectie 20 als testdata (47377 tokens)". Je kunt de gebruikte procedures bekijken [hier](https://www.clips.uantwerpen.be/conll2000/chunking/) en de [resultaten](https://ifarm.nl/erikt/research/np-chunking.html).

### Uitdaging - je bot verbeteren met NLP

In de vorige les heb je een zeer eenvoudige Q&A-bot gebouwd. Nu maak je Marvin wat sympathieker door je invoer te analyseren op sentiment en een reactie te geven die past bij het sentiment. Je moet ook een `noun_phrase` identificeren en ernaar vragen.

Je stappen bij het bouwen van een betere conversatiebot:

1. Print instructies waarin de gebruiker wordt geadviseerd hoe met de bot te communiceren.
2. Start een loop:
   1. Accepteer gebruikersinvoer.
   2. Als de gebruiker heeft gevraagd om te stoppen, beëindig dan.
   3. Verwerk de gebruikersinvoer en bepaal een passende sentimentreactie.
   4. Als een zelfstandig naamwoordzin wordt gedetecteerd in het sentiment, maak het meervoud en vraag om meer input over dat onderwerp.
   5. Print de reactie.
3. Ga terug naar stap 2.

Hier is de codefragment om sentiment te bepalen met TextBlob. Merk op dat er slechts vier *gradaties* van sentimentreactie zijn (je kunt er meer toevoegen als je wilt):

```python
if user_input_blob.polarity <= -0.5:
  response = "Oh dear, that sounds bad. "
elif user_input_blob.polarity <= 0:
  response = "Hmm, that's not great. "
elif user_input_blob.polarity <= 0.5:
  response = "Well, that sounds positive. "
elif user_input_blob.polarity <= 1:
  response = "Wow, that sounds great. "
```

Hier is een voorbeeldoutput om je te begeleiden (gebruikersinvoer staat op de regels die beginnen met >):

```output
Hello, I am Marvin, the friendly robot.
You can end this conversation at any time by typing 'bye'
After typing each answer, press 'enter'
How are you today?
> I am ok
Well, that sounds positive. Can you tell me more?
> I went for a walk and saw a lovely cat
Well, that sounds positive. Can you tell me more about lovely cats?
> cats are the best. But I also have a cool dog
Wow, that sounds great. Can you tell me more about cool dogs?
> I have an old hounddog but he is sick
Hmm, that's not great. Can you tell me more about old hounddogs?
> bye
It was nice talking to you, goodbye!
```

Een mogelijke oplossing voor de taak is [hier](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/2-Tasks/solution/bot.py)

✅ Kenniscontrole

1. Denk je dat de sympathieke reacties iemand zouden kunnen 'misleiden' om te denken dat de bot hen daadwerkelijk begrijpt?
2. Maakt het identificeren van de zelfstandig naamwoordzin de bot geloofwaardiger?
3. Waarom zou het extraheren van een 'zelfstandig naamwoordzin' uit een zin een nuttige taak zijn?

---

Implementeer de bot in de bovenstaande kenniscontrole en test deze op een vriend. Kan het hen misleiden? Kun je je bot geloofwaardiger maken?

## 🚀Uitdaging

Neem een taak uit de bovenstaande kenniscontrole en probeer deze te implementeren. Test de bot op een vriend. Kan het hen misleiden? Kun je je bot geloofwaardiger maken?

## [Quiz na de les](https://ff-quizzes.netlify.app/en/ml/)

## Review & Zelfstudie

In de komende lessen leer je meer over sentimentanalyse. Onderzoek deze interessante techniek in artikelen zoals deze op [KDNuggets](https://www.kdnuggets.com/tag/nlp)

## Opdracht 

[Laat een bot terugpraten](assignment.md)

---

**Disclaimer**:  
Dit document is vertaald met behulp van de AI-vertalingsservice [Co-op Translator](https://github.com/Azure/co-op-translator). Hoewel we streven naar nauwkeurigheid, dient u zich ervan bewust te zijn dat geautomatiseerde vertalingen fouten of onnauwkeurigheden kunnen bevatten. Het originele document in zijn oorspronkelijke taal moet worden beschouwd als de gezaghebbende bron. Voor cruciale informatie wordt professionele menselijke vertaling aanbevolen. Wij zijn niet aansprakelijk voor misverstanden of verkeerde interpretaties die voortvloeien uit het gebruik van deze vertaling.