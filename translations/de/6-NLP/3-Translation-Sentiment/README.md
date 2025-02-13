# Übersetzung und Sentimentanalyse mit ML

In den vorherigen Lektionen hast du gelernt, wie man einen einfachen Bot mit `TextBlob` erstellt, einer Bibliothek, die im Hintergrund ML integriert, um grundlegende NLP-Aufgaben wie die Extraktion von Nomenphrasen durchzuführen. Eine weitere wichtige Herausforderung in der computerlinguistischen Forschung ist die präzise _Übersetzung_ eines Satzes von einer gesprochenen oder geschriebenen Sprache in eine andere.

## [Vorlesungsquiz](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/35/)

Übersetzung ist ein sehr schwieriges Problem, das durch die Tatsache verstärkt wird, dass es Tausende von Sprachen gibt, die jeweils sehr unterschiedliche Grammatikregeln haben können. Ein Ansatz besteht darin, die formalen Grammatikregeln einer Sprache, wie z.B. Englisch, in eine nicht sprachabhängige Struktur zu konvertieren und sie dann durch Rückübersetzung in eine andere Sprache zu übersetzen. Dieser Ansatz bedeutet, dass du die folgenden Schritte unternehmen würdest:

1. **Identifikation**. Identifiziere oder tagge die Wörter in der Eingabesprache als Nomen, Verben usw.
2. **Übersetzung erstellen**. Produziere eine direkte Übersetzung jedes Wortes im Format der Zielsprache.

### Beispielsatz, Englisch zu Irisch

Im 'Englischen' besteht der Satz _I feel happy_ aus drei Wörtern in der Reihenfolge:

- **Subjekt** (I)
- **Verb** (feel)
- **Adjektiv** (happy)

Im 'Irischen' hat derselbe Satz jedoch eine ganz andere grammatikalische Struktur - Emotionen wie "*happy*" oder "*sad*" werden als *auf* dir ausgedrückt.

Die englische Phrase `I feel happy` würde im Irischen `Tá athas orm` sein. Eine *wörtliche* Übersetzung wäre `Happy is upon me`.

Ein Irischsprecher, der ins Englische übersetzt, würde `I feel happy` sagen, nicht `Happy is upon me`, weil er die Bedeutung des Satzes versteht, auch wenn die Wörter und die Satzstruktur unterschiedlich sind.

Die formale Reihenfolge für den Satz im Irischen ist:

- **Verb** (Tá oder is)
- **Adjektiv** (athas, oder happy)
- **Subjekt** (orm, oder upon me)

## Übersetzung

Ein naives Übersetzungsprogramm könnte nur Wörter übersetzen und dabei die Satzstruktur ignorieren.

✅ Wenn du als Erwachsener eine zweite (oder dritte oder mehr) Sprache gelernt hast, hast du vielleicht damit begonnen, in deiner Muttersprache zu denken, ein Konzept Wort für Wort in deinem Kopf in die zweite Sprache zu übersetzen und dann deine Übersetzung laut auszusprechen. Das ähnelt dem, was naive Übersetzungscomputerprogramme tun. Es ist wichtig, diese Phase zu überwinden, um fließend zu werden!

Naive Übersetzungen führen zu schlechten (und manchmal lustigen) Fehlübersetzungen: `I feel happy` wird wörtlich zu `Mise bhraitheann athas` im Irischen übersetzt. Das bedeutet (wörtlich) `me feel happy` und ist kein gültiger irischer Satz. Obwohl Englisch und Irisch Sprachen sind, die auf zwei benachbarten Inseln gesprochen werden, sind sie sehr unterschiedliche Sprachen mit unterschiedlichen Grammatikstrukturen.

> Du kannst dir einige Videos über irische Sprachtraditionen ansehen, wie [dieses](https://www.youtube.com/watch?v=mRIaLSdRMMs)

### Ansätze des maschinellen Lernens

Bisher hast du über den Ansatz der formalen Regeln in der Verarbeitung natürlicher Sprache gelernt. Ein anderer Ansatz besteht darin, die Bedeutung der Wörter zu ignorieren und _stattdessen maschinelles Lernen zu verwenden, um Muster zu erkennen_. Dies kann bei der Übersetzung funktionieren, wenn du viele Texte (ein *Korpus*) oder Texte (*Korpora*) in beiden Ausgangs- und Zielsprache hast.

Betrachte zum Beispiel den Fall von *Stolz und Vorurteil*, einem bekannten englischen Roman, der 1813 von Jane Austen geschrieben wurde. Wenn du das Buch auf Englisch und eine menschliche Übersetzung des Buches auf *Französisch* konsultierst, könntest du Phrasen erkennen, die in einer Sprache _idiomatisch_ in die andere übersetzt werden. Das wirst du gleich tun.

Wenn eine englische Phrase wie `I have no money` wörtlich ins Französische übersetzt wird, könnte sie `Je n'ai pas de monnaie` werden. "Monnaie" ist ein kniffliges französisches 'falsches Kognat', da 'money' und 'monnaie' nicht synonym sind. Eine bessere Übersetzung, die ein Mensch machen könnte, wäre `Je n'ai pas d'argent`, da sie besser die Bedeutung vermittelt, dass du kein Geld hast (im Gegensatz zu 'Kleingeld', was die Bedeutung von 'monnaie' ist).

![monnaie](../../../../translated_images/monnaie.606c5fa8369d5c3b3031ef0713e2069485c87985dd475cd9056bdf4c76c1f4b8.de.png)

> Bild von [Jen Looper](https://twitter.com/jenlooper)

Wenn ein ML-Modell genügend menschliche Übersetzungen hat, um ein Modell zu erstellen, kann es die Genauigkeit der Übersetzungen verbessern, indem es gemeinsame Muster in Texten identifiziert, die zuvor von erfahrenen menschlichen Sprechern beider Sprachen übersetzt wurden.

### Übung - Übersetzung

Du kannst `TextBlob` verwenden, um Sätze zu übersetzen. Probiere die berühmte erste Zeile von **Stolz und Vorurteil**:

```python
from textblob import TextBlob

blob = TextBlob(
    "It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want of a wife!"
)
print(blob.translate(to="fr"))

```

`TextBlob` macht bei der Übersetzung einen ziemlich guten Job: "C'est une vérité universellement reconnue, qu'un homme célibataire en possession d'une bonne fortune doit avoir besoin d'une femme!".

Es könnte argumentiert werden, dass die Übersetzung von TextBlob in der Tat viel genauer ist als die französische Übersetzung des Buches von 1932 durch V. Leconte und Ch. Pressoir:

"C'est une vérité universelle qu'un célibataire pourvu d'une belle fortune doit avoir envie de se marier, et, si peu que l'on sache de son sentiment à cet egard, lorsqu'il arrive dans une nouvelle résidence, cette idée est si bien fixée dans l'esprit de ses voisins qu'ils le considèrent sur-le-champ comme la propriété légitime de l'une ou l'autre de leurs filles."

In diesem Fall macht die durch ML informierte Übersetzung einen besseren Job als der menschliche Übersetzer, der unnötig Worte in den Mund des ursprünglichen Autors legt, um 'Klarheit' zu schaffen.

> Was passiert hier? Und warum ist TextBlob so gut bei der Übersetzung? Nun, im Hintergrund verwendet es Google Translate, eine ausgeklügelte KI, die in der Lage ist, Millionen von Phrasen zu analysieren, um die besten Strings für die jeweilige Aufgabe vorherzusagen. Hier läuft nichts manuell ab und du benötigst eine Internetverbindung, um `blob.translate`.

✅ Try some more sentences. Which is better, ML or human translation? In which cases?

## Sentiment analysis

Another area where machine learning can work very well is sentiment analysis. A non-ML approach to sentiment is to identify words and phrases which are 'positive' and 'negative'. Then, given a new piece of text, calculate the total value of the positive, negative and neutral words to identify the overall sentiment. 

This approach is easily tricked as you may have seen in the Marvin task - the sentence `Great, that was a wonderful waste of time, I'm glad we are lost on this dark road` ist ein sarkastischer, negativer Satz, aber der einfache Algorithmus erkennt 'great', 'wonderful', 'glad' als positiv und 'waste', 'lost' und 'dark' als negativ. Das Gesamtsentiment wird von diesen widersprüchlichen Wörtern beeinflusst.

✅ Halte einen Moment inne und denke darüber nach, wie wir Sarkasmus als menschliche Sprecher vermitteln. Tonfall spielt eine große Rolle. Versuche, den Satz "Nun, dieser Film war großartig" auf verschiedene Arten zu sagen, um zu entdecken, wie deine Stimme Bedeutung vermittelt.

### ML-Ansätze

Der ML-Ansatz würde darin bestehen, manuell negative und positive Textkörper zu sammeln - Tweets, Filmkritiken oder alles, wo der Mensch eine Bewertung *und* eine schriftliche Meinung abgegeben hat. Dann können NLP-Techniken auf Meinungen und Bewertungen angewendet werden, sodass Muster entstehen (z.B. positive Filmkritiken enthalten tendenziell häufiger die Phrase 'Oscar würdig' als negative Filmkritiken, oder positive Restaurantbewertungen sagen 'gourmet' viel häufiger als 'ekelhaft').

> ⚖️ **Beispiel**: Wenn du in einem Büro eines Politikers arbeitest und ein neues Gesetz diskutiert wird, könnten Wähler an das Büro schreiben mit E-Mails, die das bestimmte neue Gesetz unterstützen oder dagegen sind. Angenommen, du bist damit beauftragt, die E-Mails zu lesen und sie in zwei Stapel zu sortieren, *dafür* und *dagegen*. Wenn es viele E-Mails gibt, könntest du überfordert sein, wenn du versuchst, sie alle zu lesen. Wäre es nicht schön, wenn ein Bot sie alle für dich lesen könnte, sie versteht und dir sagt, in welchen Stapel jede E-Mail gehört? 
> 
> Eine Möglichkeit, dies zu erreichen, ist die Verwendung von maschinellem Lernen. Du würdest das Modell mit einem Teil der *dagegen* E-Mails und einem Teil der *dafür* E-Mails trainieren. Das Modell würde dazu tendieren, Phrasen und Wörter mit der Gegenseite und der Befürworterseite zu assoziieren, *aber es würde keinen der Inhalte verstehen*, nur dass bestimmte Wörter und Muster mit einer *dagegen* oder *dafür* E-Mail eher erscheinen würden. Du könntest es mit einigen E-Mails testen, die du nicht verwendet hast, um das Modell zu trainieren, und sehen, ob es zu demselben Schluss kommt wie du. Sobald du mit der Genauigkeit des Modells zufrieden bist, könntest du zukünftige E-Mails verarbeiten, ohne jede einzeln lesen zu müssen.

✅ Klingt dieser Prozess nach Prozessen, die du in früheren Lektionen verwendet hast?

## Übung - sentimentale Sätze

Das Sentiment wird mit einer *Polarität* von -1 bis 1 gemessen, wobei -1 das negativste Sentiment und 1 das positivste ist. Das Sentiment wird auch mit einem 0 - 1 Score für Objektivität (0) und Subjektivität (1) gemessen.

Sieh dir noch einmal Jane Austens *Stolz und Vorurteil* an. Der Text ist hier verfügbar bei [Project Gutenberg](https://www.gutenberg.org/files/1342/1342-h/1342-h.htm). Das folgende Beispiel zeigt ein kurzes Programm, das das Sentiment der ersten und letzten Sätze des Buches analysiert und seine Sentimentpolarität sowie den Subjektivitäts-/Objektivitäts-Score anzeigt.

Du solltest die `TextBlob` Bibliothek (oben beschrieben) verwenden, um `sentiment` zu bestimmen (du musst keinen eigenen Sentimentrechner schreiben) in der folgenden Aufgabe.

```python
from textblob import TextBlob

quote1 = """It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want of a wife."""

quote2 = """Darcy, as well as Elizabeth, really loved them; and they were both ever sensible of the warmest gratitude towards the persons who, by bringing her into Derbyshire, had been the means of uniting them."""

sentiment1 = TextBlob(quote1).sentiment
sentiment2 = TextBlob(quote2).sentiment

print(quote1 + " has a sentiment of " + str(sentiment1))
print(quote2 + " has a sentiment of " + str(sentiment2))
```

Du siehst die folgende Ausgabe:

```output
It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want # of a wife. has a sentiment of Sentiment(polarity=0.20952380952380953, subjectivity=0.27142857142857146)

Darcy, as well as Elizabeth, really loved them; and they were
     both ever sensible of the warmest gratitude towards the persons
      who, by bringing her into Derbyshire, had been the means of
      uniting them. has a sentiment of Sentiment(polarity=0.7, subjectivity=0.8)
```

## Herausforderung - Überprüfe die Sentimentpolarität

Deine Aufgabe ist es, anhand der Sentimentpolarität zu bestimmen, ob *Stolz und Vorurteil* mehr absolut positive Sätze als absolut negative hat. Für diese Aufgabe kannst du davon ausgehen, dass ein Polaritätswert von 1 oder -1 absolut positiv oder negativ ist.

**Schritte:**

1. Lade eine [Kopie von Stolz und Vorurteil](https://www.gutenberg.org/files/1342/1342-h/1342-h.htm) von Project Gutenberg als .txt-Datei herunter. Entferne die Metadaten am Anfang und Ende der Datei, sodass nur der ursprüngliche Text bleibt.
2. Öffne die Datei in Python und extrahiere den Inhalt als String.
3. Erstelle einen TextBlob aus dem Buchstring.
4. Analysiere jeden Satz im Buch in einer Schleife.
   1. Wenn die Polarität 1 oder -1 ist, speichere den Satz in einem Array oder einer Liste positiver oder negativer Nachrichten.
5. Am Ende drucke alle positiven Sätze und negativen Sätze (separat) sowie die Anzahl jedes Typs aus.

Hier ist eine Beispiel-[Lösung](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/3-Translation-Sentiment/solution/notebook.ipynb).

✅ Wissensüberprüfung

1. Das Sentiment basiert auf den in dem Satz verwendeten Wörtern, aber versteht der Code die Wörter?
2. Glaubst du, dass die Sentimentpolarität genau ist, oder anders gesagt, stimmst du mit den Bewertungen überein?
   1. Insbesondere, stimmst du mit der absoluten **positiven** Polarität der folgenden Sätze überein?
      * “Was für ein ausgezeichneter Vater du hast, Mädchen!” sagte sie, als die Tür geschlossen war.
      * “Ihre Untersuchung von Mr. Darcy ist vorbei, nehme ich an,” sagte Miss Bingley; “und was ist das Ergebnis?” “Ich bin davon vollkommen überzeugt, dass Mr. Darcy keinen Mangel hat.”
      * Wie wunderbar solche Dinge vorkommen!
      * Ich habe die größte Abneigung gegen solche Dinge.
      * Charlotte ist eine ausgezeichnete Managerin, das wage ich zu sagen.
      * “Das ist in der Tat erfreulich!”
      * Ich bin so glücklich!
      * Deine Idee von den Ponys ist erfreulich.
   2. Die nächsten 3 Sätze wurden mit einem absoluten positiven Sentiment bewertet, sind aber bei genauerem Hinsehen keine positiven Sätze. Warum hat die Sentimentanalyse gedacht, dass sie positive Sätze waren?
      * Glücklich werde ich sein, wenn sein Aufenthalt in Netherfield vorbei ist!” “Ich wünschte, ich könnte etwas sagen, um dich zu trösten,” antwortete Elizabeth; “aber es liegt ganz außerhalb meiner Macht.
      * Wenn ich dich nur glücklich sehen könnte!
      * Unser Leid, meine liebe Lizzy, ist sehr groß.
   3. Stimmst du mit der absoluten **negativen** Polarität der folgenden Sätze überein?
      - Jeder ist von seinem Stolz angewidert.
      - “Ich würde gerne wissen, wie er sich unter Fremden verhält.” “Du wirst dann hören—aber bereite dich auf etwas sehr Schreckliches vor.
      - Die Pause war für Elizabeths Gefühle schrecklich.
      - Es wäre schrecklich!

✅ Jeder Jane-Austen-Liebhaber wird verstehen, dass sie oft ihre Bücher nutzt, um die lächerlicheren Aspekte der englischen Regency-Gesellschaft zu kritisieren. Elizabeth Bennett, die Hauptfigur in *Stolz und Vorurteil*, ist eine scharfsinnige soziale Beobachterin (wie die Autorin) und ihre Sprache ist oft stark nuanciert. Sogar Mr. Darcy (der Liebesinteresse in der Geschichte) bemerkt Elizabeths verspielte und neckende Sprachverwendung: "Ich hatte das Vergnügen, deine Bekanntschaft lange genug zu machen, um zu wissen, dass du große Freude daran findest, gelegentlich Meinungen zu vertreten, die in der Tat nicht deine eigenen sind."

---

## 🚀Herausforderung

Kannst du Marvin noch besser machen, indem du andere Merkmale aus der Benutzereingabe extrahierst?

## [Nachlesungsquiz](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/36/)

## Überprüfung & Selbststudium

Es gibt viele Möglichkeiten, Sentiment aus Text zu extrahieren. Denke an die Geschäftsanwendungen, die diese Technik nutzen könnten. Denke darüber nach, wie es schiefgehen kann. Lies mehr über ausgeklügelte, unternehmensbereite Systeme, die Sentiment analysieren, wie [Azure Text Analysis](https://docs.microsoft.com/azure/cognitive-services/Text-Analytics/how-tos/text-analytics-how-to-sentiment-analysis?tabs=version-3-1?WT.mc_id=academic-77952-leestott). Teste einige der oben genannten Sätze aus Stolz und Vorurteil und sieh, ob es Nuancen erkennen kann.

## Aufgabe 

[Poetische Lizenz](assignment.md)

**Haftungsausschluss**:  
Dieses Dokument wurde mit maschinellen KI-Übersetzungsdiensten übersetzt. Obwohl wir uns um Genauigkeit bemühen, beachten Sie bitte, dass automatisierte Übersetzungen Fehler oder Ungenauigkeiten enthalten können. Das Originaldokument in seiner ursprünglichen Sprache sollte als maßgebliche Quelle betrachtet werden. Für kritische Informationen wird eine professionelle menschliche Übersetzung empfohlen. Wir übernehmen keine Haftung für Missverständnisse oder Fehlinterpretationen, die aus der Verwendung dieser Übersetzung entstehen.