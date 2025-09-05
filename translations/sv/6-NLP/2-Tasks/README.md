<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "5f3cb462e3122e1afe7ab0050ccf2bd3",
  "translation_date": "2025-09-05T22:13:18+00:00",
  "source_file": "6-NLP/2-Tasks/README.md",
  "language_code": "sv"
}
-->
# Vanliga uppgifter och tekniker inom naturlig språkbehandling

För de flesta *uppgifter inom naturlig språkbehandling* måste texten som ska bearbetas brytas ner, analyseras och resultaten lagras eller jämföras med regler och dataset. Dessa uppgifter gör det möjligt för programmeraren att härleda _meningen_ eller _avsikten_ eller bara _frekvensen_ av termer och ord i en text.

## [Quiz före föreläsningen](https://ff-quizzes.netlify.app/en/ml/)

Låt oss utforska vanliga tekniker som används för att bearbeta text. Kombinerat med maskininlärning hjälper dessa tekniker dig att analysera stora mängder text effektivt. Innan du tillämpar ML på dessa uppgifter, låt oss förstå de problem som en NLP-specialist stöter på.

## Vanliga uppgifter inom NLP

Det finns olika sätt att analysera en text du arbetar med. Det finns uppgifter du kan utföra, och genom dessa uppgifter kan du få en förståelse för texten och dra slutsatser. Du utför vanligtvis dessa uppgifter i en sekvens.

### Tokenisering

Förmodligen det första de flesta NLP-algoritmer måste göra är att dela upp texten i tokens, eller ord. Även om detta låter enkelt kan det bli knepigt att ta hänsyn till skiljetecken och olika språks ord- och meningsavgränsare. Du kan behöva använda olika metoder för att bestämma avgränsningar.

![tokenisering](../../../../6-NLP/2-Tasks/images/tokenization.png)
> Tokenisering av en mening från **Stolthet och fördom**. Infografik av [Jen Looper](https://twitter.com/jenlooper)

### Embeddingar

[Word embeddings](https://wikipedia.org/wiki/Word_embedding) är ett sätt att konvertera din textdata till numeriska värden. Embeddingar görs på ett sätt så att ord med liknande betydelse eller ord som används tillsammans grupperas.

![word embeddings](../../../../6-NLP/2-Tasks/images/embedding.png)
> "Jag har den största respekt för dina nerver, de är mina gamla vänner." - Word embeddings för en mening i **Stolthet och fördom**. Infografik av [Jen Looper](https://twitter.com/jenlooper)

✅ Prova [detta intressanta verktyg](https://projector.tensorflow.org/) för att experimentera med word embeddings. Genom att klicka på ett ord visas kluster av liknande ord: 'leksak' grupperas med 'disney', 'lego', 'playstation' och 'konsol'.

### Parsing & Part-of-speech Tagging

Varje ord som har tokeniserats kan taggas som en del av talet - ett substantiv, verb eller adjektiv. Meningen `den snabba röda räven hoppade över den lata bruna hunden` kan POS-taggas som räv = substantiv, hoppade = verb.

![parsing](../../../../6-NLP/2-Tasks/images/parse.png)

> Parsing av en mening från **Stolthet och fördom**. Infografik av [Jen Looper](https://twitter.com/jenlooper)

Parsing innebär att känna igen vilka ord som är relaterade till varandra i en mening - till exempel `den snabba röda räven hoppade` är en adjektiv-substantiv-verb-sekvens som är separat från sekvensen `den lata bruna hunden`.

### Ord- och frasfrekvenser

En användbar procedur vid analys av en stor textmassa är att bygga en ordlista över varje ord eller fras av intresse och hur ofta det förekommer. Frasen `den snabba röda räven hoppade över den lata bruna hunden` har en ordfrekvens på 2 för ordet "den".

Låt oss titta på ett exempel där vi räknar frekvensen av ord. Rudyard Kiplings dikt "The Winners" innehåller följande vers:

```output
What the moral? Who rides may read.
When the night is thick and the tracks are blind
A friend at a pinch is a friend, indeed,
But a fool to wait for the laggard behind.
Down to Gehenna or up to the Throne,
He travels the fastest who travels alone.
```

Eftersom frasfrekvenser kan vara skiftlägeskänsliga eller skiftlägesokänsliga efter behov, har frasen `en vän` en frekvens på 2 och `den` har en frekvens på 6, och `reser` är 2.

### N-grams

En text kan delas upp i sekvenser av ord med en viss längd, ett enda ord (unigram), två ord (bigram), tre ord (trigram) eller valfritt antal ord (n-grams).

Till exempel `den snabba röda räven hoppade över den lata bruna hunden` med ett n-gram-värde på 2 producerar följande n-grams:

1. den snabba  
2. snabba röda  
3. röda räven  
4. räven hoppade  
5. hoppade över  
6. över den  
7. den lata  
8. lata bruna  
9. bruna hunden  

Det kan vara lättare att visualisera det som en glidande ruta över meningen. Här är det för n-grams med 3 ord, n-grammet är fetstil i varje mening:

1.   <u>**den snabba röda**</u> räven hoppade över den lata bruna hunden  
2.   den **<u>snabba röda räven</u>** hoppade över den lata bruna hunden  
3.   den snabba **<u>röda räven hoppade</u>** över den lata bruna hunden  
4.   den snabba röda **<u>räven hoppade över</u>** den lata bruna hunden  
5.   den snabba röda räven **<u>hoppade över den</u>** lata bruna hunden  
6.   den snabba röda räven hoppade **<u>över den lata</u>** bruna hunden  
7.   den snabba röda räven hoppade över <u>**den lata bruna**</u> hunden  
8.   den snabba röda räven hoppade över den **<u>lata bruna hunden</u>**

![n-grams glidande ruta](../../../../6-NLP/2-Tasks/images/n-grams.gif)

> N-gram-värde på 3: Infografik av [Jen Looper](https://twitter.com/jenlooper)

### Substantivfrasextraktion

I de flesta meningar finns det ett substantiv som är subjekt eller objekt i meningen. På engelska kan det ofta identifieras genom att det föregås av 'a', 'an' eller 'the'. Att identifiera subjektet eller objektet i en mening genom att 'extrahera substantivfrasen' är en vanlig uppgift inom NLP när man försöker förstå meningen i en mening.

✅ I meningen "Jag kan inte bestämma mig för timmen, eller platsen, eller utseendet eller orden, som lade grunden. Det är för länge sedan. Jag var mitt i det innan jag visste att jag hade börjat.", kan du identifiera substantivfraserna?

I meningen `den snabba röda räven hoppade över den lata bruna hunden` finns det 2 substantivfraser: **snabba röda räven** och **lata bruna hunden**.

### Sentimentanalys

En mening eller text kan analyseras för sentiment, eller hur *positiv* eller *negativ* den är. Sentiment mäts i *polaritet* och *objektivitet/subjektivitet*. Polaritet mäts från -1,0 till 1,0 (negativ till positiv) och 0,0 till 1,0 (mest objektiv till mest subjektiv).

✅ Senare kommer du att lära dig att det finns olika sätt att bestämma sentiment med hjälp av maskininlärning, men ett sätt är att ha en lista med ord och fraser som kategoriseras som positiva eller negativa av en mänsklig expert och tillämpa den modellen på text för att beräkna ett polaritetsvärde. Kan du se hur detta skulle fungera i vissa fall och mindre bra i andra?

### Böjning

Böjning gör det möjligt att ta ett ord och få dess singular eller plural.

### Lemmatization

En *lemma* är grundformen eller huvudordet för en uppsättning ord, till exempel *flög*, *flyger*, *flygande* har en lemma av verbet *flyga*.

Det finns också användbara databaser tillgängliga för NLP-forskare, särskilt:

### WordNet

[WordNet](https://wordnet.princeton.edu/) är en databas med ord, synonymer, antonymer och många andra detaljer för varje ord på många olika språk. Det är otroligt användbart när man försöker bygga översättningar, stavningskontroller eller språkliga verktyg av alla slag.

## NLP-bibliotek

Som tur är behöver du inte bygga alla dessa tekniker själv, eftersom det finns utmärkta Python-bibliotek som gör det mycket mer tillgängligt för utvecklare som inte är specialiserade på naturlig språkbehandling eller maskininlärning. De kommande lektionerna innehåller fler exempel på dessa, men här kommer du att lära dig några användbara exempel för att hjälpa dig med nästa uppgift.

### Övning - använda biblioteket `TextBlob`

Låt oss använda ett bibliotek som heter TextBlob eftersom det innehåller användbara API:er för att hantera dessa typer av uppgifter. TextBlob "står på de gigantiska axlarna av [NLTK](https://nltk.org) och [pattern](https://github.com/clips/pattern), och fungerar bra med båda." Det har en betydande mängd ML inbäddad i sitt API.

> Obs: En användbar [Quick Start](https://textblob.readthedocs.io/en/dev/quickstart.html#quickstart) guide är tillgänglig för TextBlob som rekommenderas för erfarna Python-utvecklare.

När du försöker identifiera *substantivfraser* erbjuder TextBlob flera alternativ för att hitta substantivfraser.

1. Ta en titt på `ConllExtractor`.

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

    > Vad händer här? [ConllExtractor](https://textblob.readthedocs.io/en/dev/api_reference.html?highlight=Conll#textblob.en.np_extractors.ConllExtractor) är "En substantivfrasextraktor som använder chunk parsing tränad med ConLL-2000 träningskorpus." ConLL-2000 hänvisar till 2000 års konferens om Computational Natural Language Learning. Varje år höll konferensen en workshop för att hantera ett svårt NLP-problem, och 2000 var det substantivchunking. En modell tränades på Wall Street Journal, med "sektionerna 15-18 som träningsdata (211727 tokens) och sektion 20 som testdata (47377 tokens)". Du kan titta på de procedurer som användes [här](https://www.clips.uantwerpen.be/conll2000/chunking/) och [resultaten](https://ifarm.nl/erikt/research/np-chunking.html).

### Utmaning - förbättra din bot med NLP

I den föregående lektionen byggde du en mycket enkel Q&A-bot. Nu ska du göra Marvin lite mer sympatisk genom att analysera din input för sentiment och skriva ut ett svar som matchar sentimentet. Du måste också identifiera en `substantivfras` och fråga om den.

Dina steg när du bygger en bättre konversationsbot:

1. Skriv instruktioner som informerar användaren om hur man interagerar med boten  
2. Starta loop  
   1. Acceptera användarens input  
   2. Om användaren har bett om att avsluta, avsluta  
   3. Bearbeta användarens input och bestäm lämpligt sentiment-svar  
   4. Om en substantivfras upptäcks i sentimentet, pluralisera den och fråga om mer input om det ämnet  
   5. Skriv ut svar  
3. Gå tillbaka till steg 2  

Här är kodsnutten för att bestämma sentiment med TextBlob. Observera att det bara finns fyra *graderingar* av sentiment-svar (du kan ha fler om du vill):

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

Här är ett exempel på output för att guida dig (användarens input är på rader som börjar med >):

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

En möjlig lösning på uppgiften finns [här](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/2-Tasks/solution/bot.py)

✅ Kunskapskontroll

1. Tror du att de sympatiska svaren skulle 'lura' någon att tro att boten faktiskt förstod dem?  
2. Gör identifieringen av substantivfrasen boten mer 'trovärdig'?  
3. Varför skulle det vara användbart att extrahera en 'substantivfras' från en mening?  

---

Implementera boten i den tidigare kunskapskontrollen och testa den på en vän. Kan den lura dem? Kan du göra din bot mer 'trovärdig'?

## 🚀Utmaning

Ta en uppgift från den tidigare kunskapskontrollen och försök implementera den. Testa boten på en vän. Kan den lura dem? Kan du göra din bot mer 'trovärdig'?

## [Quiz efter föreläsningen](https://ff-quizzes.netlify.app/en/ml/)

## Granskning & Självstudier

I de kommande lektionerna kommer du att lära dig mer om sentimentanalys. Undersök denna intressanta teknik i artiklar som dessa på [KDNuggets](https://www.kdnuggets.com/tag/nlp)

## Uppgift 

[Få en bot att svara](assignment.md)

---

**Ansvarsfriskrivning**:  
Detta dokument har översatts med hjälp av AI-översättningstjänsten [Co-op Translator](https://github.com/Azure/co-op-translator). Även om vi strävar efter noggrannhet, bör du vara medveten om att automatiska översättningar kan innehålla fel eller felaktigheter. Det ursprungliga dokumentet på dess ursprungliga språk bör betraktas som den auktoritativa källan. För kritisk information rekommenderas professionell mänsklig översättning. Vi ansvarar inte för eventuella missförstånd eller feltolkningar som uppstår vid användning av denna översättning.