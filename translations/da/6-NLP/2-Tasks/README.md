<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "5f3cb462e3122e1afe7ab0050ccf2bd3",
  "translation_date": "2025-09-05T01:21:54+00:00",
  "source_file": "6-NLP/2-Tasks/README.md",
  "language_code": "da"
}
-->
# Almindelige opgaver og teknikker inden for naturlig sprogbehandling

For de fleste *naturlig sprogbehandling*-opgaver skal teksten, der skal behandles, opdeles, analyseres, og resultaterne gemmes eller krydsrefereres med regler og datasæt. Disse opgaver gør det muligt for programmøren at udlede _betydningen_ eller _intentionen_ eller blot _frekvensen_ af termer og ord i en tekst.

## [Quiz før forelæsning](https://ff-quizzes.netlify.app/en/ml/)

Lad os udforske almindelige teknikker, der bruges til at behandle tekst. Kombineret med maskinlæring hjælper disse teknikker dig med at analysere store mængder tekst effektivt. Før du anvender ML på disse opgaver, skal vi dog forstå de problemer, som en NLP-specialist støder på.

## Almindelige opgaver inden for NLP

Der er forskellige måder at analysere en tekst, du arbejder med. Der er opgaver, du kan udføre, og gennem disse opgaver kan du opnå en forståelse af teksten og drage konklusioner. Du udfører normalt disse opgaver i en sekvens.

### Tokenisering

Det første, de fleste NLP-algoritmer skal gøre, er sandsynligvis at opdele teksten i tokens eller ord. Selvom det lyder enkelt, kan det være vanskeligt at tage højde for tegnsætning og forskellige sprogs ord- og sætningsafgrænsninger. Du kan være nødt til at bruge forskellige metoder til at bestemme afgrænsninger.

![tokenisering](../../../../6-NLP/2-Tasks/images/tokenization.png)
> Tokenisering af en sætning fra **Pride and Prejudice**. Infografik af [Jen Looper](https://twitter.com/jenlooper)

### Embeddings

[Word embeddings](https://wikipedia.org/wiki/Word_embedding) er en måde at konvertere dine tekstdata til numeriske værdier. Embeddings udføres på en måde, så ord med lignende betydning eller ord, der bruges sammen, grupperes.

![word embeddings](../../../../6-NLP/2-Tasks/images/embedding.png)
> "I have the highest respect for your nerves, they are my old friends." - Word embeddings for en sætning i **Pride and Prejudice**. Infografik af [Jen Looper](https://twitter.com/jenlooper)

✅ Prøv [dette interessante værktøj](https://projector.tensorflow.org/) til at eksperimentere med word embeddings. Ved at klikke på et ord vises grupper af lignende ord: 'toy' grupperes med 'disney', 'lego', 'playstation' og 'console'.

### Parsing & Part-of-speech Tagging

Hvert ord, der er blevet tokeniseret, kan tagges som en del af talen - et substantiv, verbum eller adjektiv. Sætningen `the quick red fox jumped over the lazy brown dog` kan f.eks. POS-tagges som fox = substantiv, jumped = verbum.

![parsing](../../../../6-NLP/2-Tasks/images/parse.png)

> Parsing af en sætning fra **Pride and Prejudice**. Infografik af [Jen Looper](https://twitter.com/jenlooper)

Parsing handler om at genkende, hvilke ord der er relateret til hinanden i en sætning - for eksempel `the quick red fox jumped` er en adjektiv-substantiv-verbum-sekvens, der er adskilt fra sekvensen `lazy brown dog`.

### Ord- og frasefrekvenser

En nyttig procedure, når man analyserer en stor mængde tekst, er at opbygge en ordbog over hvert ord eller hver frase af interesse og hvor ofte det forekommer. Frasen `the quick red fox jumped over the lazy brown dog` har en ordfrekvens på 2 for the.

Lad os se på et eksempeltekst, hvor vi tæller frekvensen af ord. Rudyard Kiplings digt The Winners indeholder følgende vers:

```output
What the moral? Who rides may read.
When the night is thick and the tracks are blind
A friend at a pinch is a friend, indeed,
But a fool to wait for the laggard behind.
Down to Gehenna or up to the Throne,
He travels the fastest who travels alone.
```

Da frasefrekvenser kan være case-insensitive eller case-sensitive efter behov, har frasen `a friend` en frekvens på 2, `the` har en frekvens på 6, og `travels` er 2.

### N-grams

En tekst kan opdeles i sekvenser af ord med en fast længde, et enkelt ord (unigram), to ord (bigram), tre ord (trigram) eller et vilkårligt antal ord (n-grams).

For eksempel `the quick red fox jumped over the lazy brown dog` med en n-gram score på 2 producerer følgende n-grams:

1. the quick 
2. quick red 
3. red fox
4. fox jumped 
5. jumped over 
6. over the 
7. the lazy 
8. lazy brown 
9. brown dog

Det kan være lettere at visualisere det som en glidende boks over sætningen. Her er det for n-grams af 3 ord, hvor n-grammet er fremhævet i hver sætning:

1.   <u>**the quick red**</u> fox jumped over the lazy brown dog
2.   the **<u>quick red fox</u>** jumped over the lazy brown dog
3.   the quick **<u>red fox jumped</u>** over the lazy brown dog
4.   the quick red **<u>fox jumped over</u>** the lazy brown dog
5.   the quick red fox **<u>jumped over the</u>** lazy brown dog
6.   the quick red fox jumped **<u>over the lazy</u>** brown dog
7.   the quick red fox jumped over <u>**the lazy brown**</u> dog
8.   the quick red fox jumped over the **<u>lazy brown dog</u>**

![n-grams glidende vindue](../../../../6-NLP/2-Tasks/images/n-grams.gif)

> N-gram værdi på 3: Infografik af [Jen Looper](https://twitter.com/jenlooper)

### Substantivfrase-ekstraktion

I de fleste sætninger er der et substantiv, der er subjektet eller objektet i sætningen. På engelsk kan det ofte identificeres ved at have 'a', 'an' eller 'the' foran. At identificere subjektet eller objektet i en sætning ved at 'ekstrahere substantivfrasen' er en almindelig opgave i NLP, når man forsøger at forstå betydningen af en sætning.

✅ I sætningen "I cannot fix on the hour, or the spot, or the look or the words, which laid the foundation. It is too long ago. I was in the middle before I knew that I had begun.", kan du identificere substantivfraserne?

I sætningen `the quick red fox jumped over the lazy brown dog` er der 2 substantivfraser: **quick red fox** og **lazy brown dog**.

### Sentimentanalyse

En sætning eller tekst kan analyseres for sentiment, eller hvor *positiv* eller *negativ* den er. Sentiment måles i *polarity* og *objektivitet/subjektivitet*. Polarity måles fra -1.0 til 1.0 (negativ til positiv) og 0.0 til 1.0 (mest objektiv til mest subjektiv).

✅ Senere vil du lære, at der er forskellige måder at bestemme sentiment ved hjælp af maskinlæring, men en måde er at have en liste over ord og fraser, der er kategoriseret som positive eller negative af en menneskelig ekspert og anvende den model på tekst for at beregne en polaritetsscore. Kan du se, hvordan dette ville fungere i nogle tilfælde og mindre godt i andre?

### Bøjning

Bøjning gør det muligt at tage et ord og finde dets ental eller flertal.

### Lemmatization

En *lemma* er grundformen eller hovedordet for et sæt ord, for eksempel *flew*, *flies*, *flying* har en lemma af verbet *fly*.

Der findes også nyttige databaser til NLP-forskere, især:

### WordNet

[WordNet](https://wordnet.princeton.edu/) er en database over ord, synonymer, antonymer og mange andre detaljer for hvert ord på mange forskellige sprog. Det er utrolig nyttigt, når man forsøger at bygge oversættelser, stavekontroller eller sprogværktøjer af enhver art.

## NLP-biblioteker

Heldigvis behøver du ikke selv at bygge alle disse teknikker, da der findes fremragende Python-biblioteker, der gør det meget mere tilgængeligt for udviklere, der ikke er specialiserede i naturlig sprogbehandling eller maskinlæring. De næste lektioner inkluderer flere eksempler på disse, men her vil du lære nogle nyttige eksempler, der kan hjælpe dig med den næste opgave.

### Øvelse - brug af `TextBlob` biblioteket

Lad os bruge et bibliotek kaldet TextBlob, da det indeholder nyttige API'er til at tackle disse typer opgaver. TextBlob "står på de gigantiske skuldre af [NLTK](https://nltk.org) og [pattern](https://github.com/clips/pattern), og fungerer godt med begge." Det har en betydelig mængde ML indlejret i sin API.

> Bemærk: En nyttig [Quick Start](https://textblob.readthedocs.io/en/dev/quickstart.html#quickstart) guide er tilgængelig for TextBlob, som anbefales til erfarne Python-udviklere.

Når man forsøger at identificere *substantivfraser*, tilbyder TextBlob flere muligheder for ekstraktorer til at finde substantivfraser.

1. Tag et kig på `ConllExtractor`.

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

    > Hvad sker der her? [ConllExtractor](https://textblob.readthedocs.io/en/dev/api_reference.html?highlight=Conll#textblob.en.np_extractors.ConllExtractor) er "En substantivfrase-ekstraktor, der bruger chunk parsing trænet med ConLL-2000 træningskorpus." ConLL-2000 refererer til 2000-konferencen om Computational Natural Language Learning. Hvert år afholdt konferencen en workshop for at tackle et vanskeligt NLP-problem, og i 2000 var det substantiv chunking. En model blev trænet på Wall Street Journal med "sektioner 15-18 som træningsdata (211727 tokens) og sektion 20 som testdata (47377 tokens)". Du kan se de anvendte procedurer [her](https://www.clips.uantwerpen.be/conll2000/chunking/) og [resultaterne](https://ifarm.nl/erikt/research/np-chunking.html).

### Udfordring - forbedring af din bot med NLP

I den forrige lektion byggede du en meget simpel Q&A bot. Nu vil du gøre Marvin lidt mere sympatisk ved at analysere din input for sentiment og udskrive et svar, der matcher sentimentet. Du skal også identificere en `noun_phrase` og spørge om den.

Dine trin, når du bygger en bedre samtalebot:

1. Udskriv instruktioner, der rådgiver brugeren om, hvordan man interagerer med botten
2. Start loop 
   1. Accepter brugerinput
   2. Hvis brugeren har bedt om at afslutte, så afslut
   3. Behandl brugerinput og bestem passende sentimentrespons
   4. Hvis en substantivfrase detekteres i sentimentet, pluraliser den og spørg om mere input om det emne
   5. Udskriv svar
3. loop tilbage til trin 2

Her er kodeudsnittet til at bestemme sentiment ved hjælp af TextBlob. Bemærk, at der kun er fire *gradienter* af sentimentrespons (du kan have flere, hvis du vil):

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

Her er noget eksempeloutput til vejledning (brugerinput er på linjer, der starter med >):

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

En mulig løsning på opgaven findes [her](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/2-Tasks/solution/bot.py)

✅ Videnskontrol

1. Tror du, at de sympatiske svar ville 'narre' nogen til at tro, at botten faktisk forstod dem?
2. Gør identifikationen af substantivfrasen botten mere 'troværdig'?
3. Hvorfor ville det være nyttigt at ekstrahere en 'substantivfrase' fra en sætning?

---

Implementer botten i den tidligere videnskontrol og test den på en ven. Kan den narre dem? Kan du gøre din bot mere 'troværdig'?

## 🚀Udfordring

Tag en opgave fra den tidligere videnskontrol og prøv at implementere den. Test botten på en ven. Kan den narre dem? Kan du gøre din bot mere 'troværdig'?

## [Quiz efter forelæsning](https://ff-quizzes.netlify.app/en/ml/)

## Gennemgang & Selvstudie

I de næste par lektioner vil du lære mere om sentimentanalyse. Undersøg denne interessante teknik i artikler som disse på [KDNuggets](https://www.kdnuggets.com/tag/nlp)

## Opgave 

[Få en bot til at svare](assignment.md)

---

**Ansvarsfraskrivelse**:  
Dette dokument er blevet oversat ved hjælp af AI-oversættelsestjenesten [Co-op Translator](https://github.com/Azure/co-op-translator). Selvom vi bestræber os på nøjagtighed, skal du være opmærksom på, at automatiserede oversættelser kan indeholde fejl eller unøjagtigheder. Det originale dokument på dets oprindelige sprog bør betragtes som den autoritative kilde. For kritisk information anbefales professionel menneskelig oversættelse. Vi påtager os ikke ansvar for eventuelle misforståelser eller fejltolkninger, der opstår som følge af brugen af denne oversættelse.