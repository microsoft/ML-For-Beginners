<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "5f3cb462e3122e1afe7ab0050ccf2bd3",
  "translation_date": "2025-09-04T22:06:09+00:00",
  "source_file": "6-NLP/2-Tasks/README.md",
  "language_code": "de"
}
-->
# Häufige Aufgaben und Techniken der Verarbeitung natürlicher Sprache

Für die meisten Aufgaben der *Verarbeitung natürlicher Sprache* muss der zu verarbeitende Text in kleinere Einheiten zerlegt, analysiert und die Ergebnisse gespeichert oder mit Regeln und Datensätzen abgeglichen werden. Diese Aufgaben ermöglichen es dem Programmierer, die _Bedeutung_ oder _Absicht_ oder lediglich die _Häufigkeit_ von Begriffen und Wörtern in einem Text abzuleiten.

## [Quiz vor der Vorlesung](https://ff-quizzes.netlify.app/en/ml/)

Lassen Sie uns gängige Techniken zur Verarbeitung von Texten entdecken. In Kombination mit maschinellem Lernen helfen diese Techniken, große Mengen an Text effizient zu analysieren. Bevor wir jedoch ML auf diese Aufgaben anwenden, sollten wir die Probleme verstehen, mit denen ein NLP-Spezialist konfrontiert ist.

## Häufige Aufgaben in der NLP

Es gibt verschiedene Möglichkeiten, einen Text zu analysieren, an dem Sie arbeiten. Es gibt Aufgaben, die Sie durchführen können, und durch diese Aufgaben können Sie ein Verständnis für den Text gewinnen und Schlussfolgerungen ziehen. Diese Aufgaben werden in der Regel in einer bestimmten Reihenfolge durchgeführt.

### Tokenisierung

Wahrscheinlich ist das Erste, was die meisten NLP-Algorithmen tun müssen, den Text in Token oder Wörter zu zerlegen. Obwohl dies einfach klingt, kann es durch die Berücksichtigung von Satzzeichen und unterschiedlichen Wort- und Satzgrenzen in verschiedenen Sprachen kompliziert werden. Es kann erforderlich sein, verschiedene Methoden anzuwenden, um die Abgrenzungen zu bestimmen.

![Tokenisierung](../../../../6-NLP/2-Tasks/images/tokenization.png)
> Tokenisierung eines Satzes aus **Stolz und Vorurteil**. Infografik von [Jen Looper](https://twitter.com/jenlooper)

### Einbettungen

[Wort-Einbettungen](https://wikipedia.org/wiki/Word_embedding) sind eine Möglichkeit, Ihre Textdaten numerisch darzustellen. Einbettungen werden so durchgeführt, dass Wörter mit ähnlicher Bedeutung oder Wörter, die zusammen verwendet werden, in Clustern gruppiert werden.

![Wort-Einbettungen](../../../../6-NLP/2-Tasks/images/embedding.png)
> "Ich habe den größten Respekt vor Ihren Nerven, sie sind meine alten Freunde." - Wort-Einbettungen für einen Satz aus **Stolz und Vorurteil**. Infografik von [Jen Looper](https://twitter.com/jenlooper)

✅ Probieren Sie [dieses interessante Tool](https://projector.tensorflow.org/) aus, um mit Wort-Einbettungen zu experimentieren. Wenn Sie auf ein Wort klicken, werden Cluster ähnlicher Wörter angezeigt: 'Spielzeug' gruppiert sich mit 'Disney', 'Lego', 'Playstation' und 'Konsole'.

### Parsing & Part-of-Speech-Tagging

Jedes tokenisierte Wort kann als Wortart (Part of Speech) markiert werden – z. B. als Substantiv, Verb oder Adjektiv. Der Satz `der schnelle rote Fuchs sprang über den faulen braunen Hund` könnte beispielsweise wie folgt markiert werden: Fuchs = Substantiv, sprang = Verb.

![Parsing](../../../../6-NLP/2-Tasks/images/parse.png)

> Parsing eines Satzes aus **Stolz und Vorurteil**. Infografik von [Jen Looper](https://twitter.com/jenlooper)

Parsing bedeutet, zu erkennen, welche Wörter in einem Satz miteinander in Beziehung stehen – zum Beispiel ist `der schnelle rote Fuchs sprang` eine Adjektiv-Substantiv-Verb-Sequenz, die von der Sequenz `fauler brauner Hund` getrennt ist.

### Wort- und Phrasenhäufigkeiten

Ein nützliches Verfahren bei der Analyse eines großen Textkorpus ist der Aufbau eines Wörterbuchs mit allen interessanten Wörtern oder Phrasen und deren Häufigkeit. Die Phrase `der schnelle rote Fuchs sprang über den faulen braunen Hund` hat beispielsweise eine Wortfrequenz von 2 für das Wort "der".

Schauen wir uns einen Beispieltext an, in dem wir die Häufigkeit von Wörtern zählen. Rudyard Kiplings Gedicht "The Winners" enthält die folgende Strophe:

```output
What the moral? Who rides may read.
When the night is thick and the tracks are blind
A friend at a pinch is a friend, indeed,
But a fool to wait for the laggard behind.
Down to Gehenna or up to the Throne,
He travels the fastest who travels alone.
```

Da die Häufigkeit von Phrasen je nach Bedarf groß- und kleinschreibungsempfindlich sein kann, hat die Phrase `a friend` eine Häufigkeit von 2, `the` eine Häufigkeit von 6 und `travels` eine Häufigkeit von 2.

### N-Gramme

Ein Text kann in Sequenzen von Wörtern einer bestimmten Länge aufgeteilt werden: ein einzelnes Wort (Unigramm), zwei Wörter (Bigramme), drei Wörter (Trigramme) oder eine beliebige Anzahl von Wörtern (N-Gramme).

Zum Beispiel ergibt `der schnelle rote Fuchs sprang über den faulen braunen Hund` mit einem N-Gramm-Wert von 2 die folgenden N-Gramme:

1. der schnelle  
2. schnelle rote  
3. rote Fuchs  
4. Fuchs sprang  
5. sprang über  
6. über den  
7. den faulen  
8. faulen braunen  
9. braunen Hund  

Es könnte einfacher sein, dies als ein gleitendes Fenster über den Satz zu visualisieren. Hier ist es für N-Gramme mit 3 Wörtern, wobei das N-Gramm in jedem Satz hervorgehoben ist:

1.   <u>**der schnelle rote**</u> Fuchs sprang über den faulen braunen Hund  
2.   der **<u>schnelle rote Fuchs</u>** sprang über den faulen braunen Hund  
3.   der schnelle **<u>rote Fuchs sprang</u>** über den faulen braunen Hund  
4.   der schnelle rote **<u>Fuchs sprang über</u>** den faulen braunen Hund  
5.   der schnelle rote Fuchs **<u>sprang über den</u>** faulen braunen Hund  
6.   der schnelle rote Fuchs sprang **<u>über den faulen</u>** braunen Hund  
7.   der schnelle rote Fuchs sprang über <u>**den faulen braunen**</u> Hund  
8.   der schnelle rote Fuchs sprang über den **<u>faulen braunen Hund</u>**  

![N-Gramme gleitendes Fenster](../../../../6-NLP/2-Tasks/images/n-grams.gif)

> N-Gramm-Wert von 3: Infografik von [Jen Looper](https://twitter.com/jenlooper)

### Extraktion von Nominalphrasen

In den meisten Sätzen gibt es ein Substantiv, das das Subjekt oder Objekt des Satzes ist. Im Englischen ist es oft daran erkennbar, dass ihm 'a', 'an' oder 'the' vorangestellt ist. Das Identifizieren des Subjekts oder Objekts eines Satzes durch die 'Extraktion der Nominalphrase' ist eine gängige Aufgabe in der NLP, wenn versucht wird, die Bedeutung eines Satzes zu verstehen.

✅ Im Satz "I cannot fix on the hour, or the spot, or the look or the words, which laid the foundation. It is too long ago. I was in the middle before I knew that I had begun.", können Sie die Nominalphrasen identifizieren?

Im Satz `der schnelle rote Fuchs sprang über den faulen braunen Hund` gibt es 2 Nominalphrasen: **schneller roter Fuchs** und **fauler brauner Hund**.

### Sentiment-Analyse

Ein Satz oder Text kann auf seine Stimmung analysiert werden, also wie *positiv* oder *negativ* er ist. Die Stimmung wird in *Polarität* und *Objektivität/Subjektivität* gemessen. Die Polarität wird von -1,0 bis 1,0 (negativ bis positiv) und die Objektivität von 0,0 bis 1,0 (am objektivsten bis am subjektivsten) gemessen.

✅ Später lernen Sie, dass es verschiedene Möglichkeiten gibt, die Stimmung mithilfe von maschinellem Lernen zu bestimmen. Eine Möglichkeit besteht jedoch darin, eine Liste von Wörtern und Phrasen zu haben, die von einem menschlichen Experten als positiv oder negativ kategorisiert wurden, und dieses Modell auf Text anzuwenden, um einen Polaritätswert zu berechnen. Können Sie sehen, wie dies in einigen Fällen funktioniert und in anderen weniger gut?

### Flexion

Die Flexion ermöglicht es Ihnen, ein Wort zu nehmen und die Einzahl oder Mehrzahl des Wortes zu erhalten.

### Lemmatisierung

Ein *Lemma* ist die Grundform oder das Stammwort für eine Gruppe von Wörtern, zum Beispiel haben *flog*, *fliegt*, *fliegend* das Lemma des Verbs *fliegen*.

Es gibt auch nützliche Datenbanken für NLP-Forscher, insbesondere:

### WordNet

[WordNet](https://wordnet.princeton.edu/) ist eine Datenbank von Wörtern, Synonymen, Antonymen und vielen anderen Details für jedes Wort in vielen verschiedenen Sprachen. Es ist unglaublich nützlich beim Erstellen von Übersetzungen, Rechtschreibprüfungen oder Sprachwerkzeugen jeglicher Art.

## NLP-Bibliotheken

Glücklicherweise müssen Sie nicht alle diese Techniken selbst entwickeln, da es ausgezeichnete Python-Bibliotheken gibt, die Entwicklern, die nicht auf die Verarbeitung natürlicher Sprache oder maschinelles Lernen spezialisiert sind, den Zugang erleichtern. In den nächsten Lektionen werden Sie mehr Beispiele dazu sehen, aber hier lernen Sie einige nützliche Beispiele kennen, die Ihnen bei der nächsten Aufgabe helfen.

### Übung - Verwendung der `TextBlob`-Bibliothek

Lassen Sie uns eine Bibliothek namens TextBlob verwenden, da sie hilfreiche APIs für diese Arten von Aufgaben enthält. TextBlob "steht auf den Schultern von [NLTK](https://nltk.org) und [pattern](https://github.com/clips/pattern) und arbeitet gut mit beiden zusammen." Es enthält eine beträchtliche Menge an ML in seiner API.

> Hinweis: Ein nützlicher [Quick Start](https://textblob.readthedocs.io/en/dev/quickstart.html#quickstart)-Leitfaden ist für TextBlob verfügbar und wird erfahrenen Python-Entwicklern empfohlen.

Wenn Sie versuchen, *Nominalphrasen* zu identifizieren, bietet TextBlob mehrere Optionen für Extraktoren, um Nominalphrasen zu finden.

1. Schauen Sie sich `ConllExtractor` an.

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

    > Was passiert hier? [ConllExtractor](https://textblob.readthedocs.io/en/dev/api_reference.html?highlight=Conll#textblob.en.np_extractors.ConllExtractor) ist "Ein Nominalphrasen-Extraktor, der Chunk-Parsing verwendet, das mit dem ConLL-2000-Trainingskorpus trainiert wurde." ConLL-2000 bezieht sich auf die Konferenz zur computergestützten Verarbeitung natürlicher Sprache im Jahr 2000. Jedes Jahr veranstaltete die Konferenz einen Workshop, um ein schwieriges NLP-Problem zu lösen, und 2000 war es das Chunking von Nominalphrasen. Ein Modell wurde auf dem Wall Street Journal trainiert, mit "Abschnitten 15-18 als Trainingsdaten (211727 Token) und Abschnitt 20 als Testdaten (47377 Token)". Sie können die verwendeten Verfahren [hier](https://www.clips.uantwerpen.be/conll2000/chunking/) und die [Ergebnisse](https://ifarm.nl/erikt/research/np-chunking.html) einsehen.

### Herausforderung - Verbesserung Ihres Bots mit NLP

In der vorherigen Lektion haben Sie einen sehr einfachen Q&A-Bot erstellt. Jetzt machen Sie Marvin etwas einfühlsamer, indem Sie Ihre Eingabe auf Stimmung analysieren und eine passende Antwort ausgeben. Sie müssen auch eine `noun_phrase` identifizieren und dazu weitere Eingaben anfordern.

Ihre Schritte beim Erstellen eines besseren Konversationsbots:

1. Anweisungen ausgeben, die den Benutzer darüber informieren, wie er mit dem Bot interagieren soll.
2. Schleife starten  
   1. Benutzereingabe akzeptieren  
   2. Wenn der Benutzer um Beendigung bittet, dann beenden  
   3. Benutzereingabe verarbeiten und passende Stimmungsantwort bestimmen  
   4. Wenn eine Nominalphrase in der Stimmung erkannt wird, diese pluralisieren und weitere Eingaben zu diesem Thema anfordern  
   5. Antwort ausgeben  
3. Zurück zu Schritt 2  

Hier ist der Codeausschnitt, um die Stimmung mit TextBlob zu bestimmen. Beachten Sie, dass es nur vier *Abstufungen* der Stimmungsantwort gibt (Sie könnten mehr hinzufügen, wenn Sie möchten):

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

Hier ist eine Beispielausgabe zur Orientierung (Benutzereingaben beginnen mit >):

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

Eine mögliche Lösung für die Aufgabe finden Sie [hier](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/2-Tasks/solution/bot.py)

✅ Wissensüberprüfung

1. Glauben Sie, dass die einfühlsamen Antworten jemanden dazu bringen könnten, zu denken, dass der Bot sie tatsächlich versteht?  
2. Macht das Identifizieren der Nominalphrase den Bot glaubwürdiger?  
3. Warum könnte das Extrahieren einer 'Nominalphrase' aus einem Satz nützlich sein?

---

Implementieren Sie den Bot aus der vorherigen Wissensüberprüfung und testen Sie ihn mit einem Freund. Kann er sie täuschen? Können Sie Ihren Bot glaubwürdiger machen?

## 🚀Herausforderung

Nehmen Sie eine Aufgabe aus der vorherigen Wissensüberprüfung und versuchen Sie, sie zu implementieren. Testen Sie den Bot mit einem Freund. Kann er sie täuschen? Können Sie Ihren Bot glaubwürdiger machen?

## [Quiz nach der Vorlesung](https://ff-quizzes.netlify.app/en/ml/)

## Rückblick & Selbststudium

In den nächsten Lektionen werden Sie mehr über Sentiment-Analyse erfahren. Recherchieren Sie diese interessante Technik in Artikeln wie diesen auf [KDNuggets](https://www.kdnuggets.com/tag/nlp).

## Aufgabe 

[Bringen Sie einen Bot zum Antworten](assignment.md)

---

**Haftungsausschluss**:  
Dieses Dokument wurde mit dem KI-Übersetzungsdienst [Co-op Translator](https://github.com/Azure/co-op-translator) übersetzt. Obwohl wir uns um Genauigkeit bemühen, weisen wir darauf hin, dass automatisierte Übersetzungen Fehler oder Ungenauigkeiten enthalten können. Das Originaldokument in seiner ursprünglichen Sprache sollte als maßgebliche Quelle betrachtet werden. Für kritische Informationen wird eine professionelle menschliche Übersetzung empfohlen. Wir übernehmen keine Haftung für Missverständnisse oder Fehlinterpretationen, die sich aus der Nutzung dieser Übersetzung ergeben.