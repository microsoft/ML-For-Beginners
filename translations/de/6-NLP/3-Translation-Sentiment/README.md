<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "6396d5d8617572cd2ac1de74fb0deb22",
  "translation_date": "2025-09-03T22:03:30+00:00",
  "source_file": "6-NLP/3-Translation-Sentiment/README.md",
  "language_code": "de"
}
-->
# Übersetzung und Sentiment-Analyse mit ML

In den vorherigen Lektionen hast du gelernt, wie man einen einfachen Bot mit `TextBlob` erstellt, einer Bibliothek, die maschinelles Lernen im Hintergrund einsetzt, um grundlegende NLP-Aufgaben wie die Extraktion von Nominalphrasen durchzuführen. Eine weitere wichtige Herausforderung in der Computerlinguistik ist die präzise _Übersetzung_ eines Satzes von einer gesprochenen oder geschriebenen Sprache in eine andere.

## [Quiz vor der Vorlesung](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/35/)

Übersetzung ist ein sehr schwieriges Problem, das durch die Tatsache erschwert wird, dass es Tausende von Sprachen gibt, die jeweils sehr unterschiedliche Grammatikregeln haben können. Ein Ansatz besteht darin, die formalen Grammatikregeln einer Sprache, wie Englisch, in eine sprachunabhängige Struktur umzuwandeln und diese dann durch Rückübersetzung in eine andere Sprache zu übersetzen. Dieser Ansatz umfasst die folgenden Schritte:

1. **Identifikation**. Identifiziere oder markiere die Wörter in der Eingabesprache als Substantive, Verben usw.
2. **Übersetzung erstellen**. Erstelle eine direkte Übersetzung jedes Wortes im Format der Zielsprache.

### Beispielsatz, Englisch zu Irisch

Im 'Englischen' lautet der Satz _I feel happy_ aus drei Wörtern in der Reihenfolge:

- **Subjekt** (I)
- **Verb** (feel)
- **Adjektiv** (happy)

Im 'Irischen' hat derselbe Satz jedoch eine ganz andere grammatikalische Struktur – Emotionen wie "*happy*" oder "*sad*" werden als etwas ausgedrückt, das *auf* dir liegt.

Die englische Phrase `I feel happy` würde im Irischen `Tá athas orm` lauten. Eine *wörtliche* Übersetzung wäre `Happy is upon me`.

Ein irischer Sprecher, der ins Englische übersetzt, würde sagen `I feel happy` und nicht `Happy is upon me`, weil er die Bedeutung des Satzes versteht, auch wenn die Wörter und die Satzstruktur unterschiedlich sind.

Die formale Reihenfolge des Satzes im Irischen lautet:

- **Verb** (Tá oder ist)
- **Adjektiv** (athas, oder happy)
- **Subjekt** (orm, oder upon me)

## Übersetzung

Ein naives Übersetzungsprogramm könnte nur Wörter übersetzen und die Satzstruktur ignorieren.

✅ Wenn du als Erwachsener eine zweite (oder dritte oder weitere) Sprache gelernt hast, hast du vielleicht damit begonnen, in deiner Muttersprache zu denken, ein Konzept Wort für Wort im Kopf in die zweite Sprache zu übersetzen und dann deine Übersetzung auszusprechen. Dies ähnelt dem, was naive Übersetzungsprogramme tun. Es ist wichtig, diese Phase zu überwinden, um fließend zu werden!

Naive Übersetzungen führen zu schlechten (und manchmal urkomischen) Fehlübersetzungen: `I feel happy` wird wörtlich zu `Mise bhraitheann athas` ins Irische übersetzt. Das bedeutet (wörtlich) `me feel happy` und ist kein gültiger irischer Satz. Obwohl Englisch und Irisch Sprachen sind, die auf zwei eng benachbarten Inseln gesprochen werden, sind sie sehr unterschiedliche Sprachen mit unterschiedlichen Grammatikstrukturen.

> Du kannst dir einige Videos über irische Sprachtraditionen ansehen, wie [dieses hier](https://www.youtube.com/watch?v=mRIaLSdRMMs)

### Ansätze mit maschinellem Lernen

Bisher hast du den Ansatz der formalen Regeln für die Verarbeitung natürlicher Sprache kennengelernt. Ein anderer Ansatz besteht darin, die Bedeutung der Wörter zu ignorieren und _stattdessen maschinelles Lernen zu verwenden, um Muster zu erkennen_. Dies kann bei der Übersetzung funktionieren, wenn du viele Texte (ein *Korpus*) oder Texte (*Korpora*) in der Ausgangs- und Zielsprache hast.

Betrachte beispielsweise den Fall von *Stolz und Vorurteil*, einem bekannten englischen Roman, der 1813 von Jane Austen geschrieben wurde. Wenn du das Buch auf Englisch und eine menschliche Übersetzung des Buches ins *Französische* konsultierst, könntest du Phrasen erkennen, die _idiomatisch_ in die andere Sprache übersetzt wurden. Das wirst du gleich ausprobieren.

Wenn beispielsweise eine englische Phrase wie `I have no money` wörtlich ins Französische übersetzt wird, könnte sie `Je n'ai pas de monnaie` werden. "Monnaie" ist ein schwieriges französisches 'falsches Kognat', da 'money' und 'monnaie' nicht synonym sind. Eine bessere Übersetzung, die ein Mensch machen könnte, wäre `Je n'ai pas d'argent`, da sie besser vermittelt, dass du kein Geld hast (anstatt 'Kleingeld', was die Bedeutung von 'monnaie' ist).

![monnaie](../../../../translated_images/monnaie.606c5fa8369d5c3b3031ef0713e2069485c87985dd475cd9056bdf4c76c1f4b8.de.png)

> Bild von [Jen Looper](https://twitter.com/jenlooper)

Wenn ein ML-Modell genügend menschliche Übersetzungen hat, um ein Modell darauf aufzubauen, kann es die Genauigkeit von Übersetzungen verbessern, indem es häufige Muster in Texten identifiziert, die zuvor von Experten, die beide Sprachen sprechen, übersetzt wurden.

### Übung - Übersetzung

Du kannst `TextBlob` verwenden, um Sätze zu übersetzen. Probiere die berühmte erste Zeile von **Stolz und Vorurteil**:

```python
from textblob import TextBlob

blob = TextBlob(
    "It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want of a wife!"
)
print(blob.translate(to="fr"))

```

`TextBlob` macht einen ziemlich guten Job bei der Übersetzung: "C'est une vérité universellement reconnue, qu'un homme célibataire en possession d'une bonne fortune doit avoir besoin d'une femme!". 

Man könnte argumentieren, dass die Übersetzung von TextBlob tatsächlich viel genauer ist als die französische Übersetzung des Buches von 1932 durch V. Leconte und Ch. Pressoir:

"C'est une vérité universelle qu'un célibataire pourvu d'une belle fortune doit avoir envie de se marier, et, si peu que l'on sache de son sentiment à cet egard, lorsqu'il arrive dans une nouvelle résidence, cette idée est si bien fixée dans l'esprit de ses voisins qu'ils le considèrent sur-le-champ comme la propriété légitime de l'une ou l'autre de leurs filles."

In diesem Fall leistet die durch ML informierte Übersetzung bessere Arbeit als der menschliche Übersetzer, der unnötigerweise Worte in den Mund der ursprünglichen Autorin legt, um 'Klarheit' zu schaffen.

> Was passiert hier? Und warum ist TextBlob so gut bei der Übersetzung? Nun, im Hintergrund verwendet es Google Translate, eine ausgeklügelte KI, die Millionen von Phrasen analysieren kann, um die besten Strings für die jeweilige Aufgabe vorherzusagen. Hier passiert nichts manuell, und du benötigst eine Internetverbindung, um `blob.translate` zu verwenden.

✅ Probiere einige weitere Sätze aus. Was ist besser, ML oder menschliche Übersetzung? In welchen Fällen?

## Sentiment-Analyse

Ein weiteres Gebiet, in dem maschinelles Lernen sehr gut funktionieren kann, ist die Sentiment-Analyse. Ein nicht-ML-Ansatz für Sentiment besteht darin, Wörter und Phrasen zu identifizieren, die 'positiv' und 'negativ' sind. Dann wird bei einem neuen Textstück der Gesamtwert der positiven, negativen und neutralen Wörter berechnet, um das Gesamtsentiment zu identifizieren. 

Dieser Ansatz lässt sich leicht täuschen, wie du vielleicht in der Marvin-Aufgabe gesehen hast – der Satz `Great, that was a wonderful waste of time, I'm glad we are lost on this dark road` ist ein sarkastischer, negativer Sentiment-Satz, aber der einfache Algorithmus erkennt 'great', 'wonderful', 'glad' als positiv und 'waste', 'lost' und 'dark' als negativ. Das Gesamtsentiment wird durch diese widersprüchlichen Wörter beeinflusst.

✅ Halte einen Moment inne und denke darüber nach, wie wir als menschliche Sprecher Sarkasmus vermitteln. Tonfall spielt eine große Rolle. Versuche, den Satz "Well, that film was awesome" auf verschiedene Arten zu sagen, um herauszufinden, wie deine Stimme Bedeutung vermittelt.

### ML-Ansätze

Der ML-Ansatz würde darin bestehen, manuell negative und positive Textkörper zu sammeln – Tweets, oder Filmkritiken, oder alles, bei dem der Mensch eine Bewertung *und* eine schriftliche Meinung abgegeben hat. Dann können NLP-Techniken auf Meinungen und Bewertungen angewendet werden, sodass Muster entstehen (z. B. positive Filmkritiken enthalten häufiger die Phrase 'Oscar worthy' als negative Filmkritiken, oder positive Restaurantkritiken sagen 'gourmet' viel häufiger als 'disgusting').

> ⚖️ **Beispiel**: Wenn du in einem Büro eines Politikers arbeiten würdest und ein neues Gesetz diskutiert würde, könnten Bürger E-Mails schreiben, die das Gesetz unterstützen oder ablehnen. Angenommen, du bist damit beauftragt, die E-Mails zu lesen und in zwei Stapel zu sortieren, *dafür* und *dagegen*. Wenn es viele E-Mails gäbe, könntest du überfordert sein, sie alle zu lesen. Wäre es nicht schön, wenn ein Bot sie alle für dich lesen, verstehen und dir sagen könnte, in welchen Stapel jede E-Mail gehört? 
> 
> Eine Möglichkeit, dies zu erreichen, ist die Verwendung von maschinellem Lernen. Du würdest das Modell mit einem Teil der *dagegen*-E-Mails und einem Teil der *dafür*-E-Mails trainieren. Das Modell würde dazu neigen, Phrasen und Wörter mit der dagegen-Seite und der dafür-Seite zu assoziieren, *aber es würde keinen der Inhalte verstehen*, sondern nur, dass bestimmte Wörter und Muster eher in einer *dagegen*- oder einer *dafür*-E-Mail erscheinen. Du könntest es mit einigen E-Mails testen, die du nicht zum Trainieren des Modells verwendet hast, und sehen, ob es zu denselben Schlussfolgerungen kommt wie du. Sobald du mit der Genauigkeit des Modells zufrieden bist, könntest du zukünftige E-Mails verarbeiten, ohne jede einzelne lesen zu müssen.

✅ Klingt dieser Prozess wie Prozesse, die du in vorherigen Lektionen verwendet hast?

## Übung - sentimentale Sätze

Sentiment wird mit einer *Polarität* von -1 bis 1 gemessen, wobei -1 das negativste Sentiment und 1 das positivste ist. Sentiment wird auch mit einem Wert von 0 - 1 für Objektivität (0) und Subjektivität (1) gemessen.

Betrachte erneut Jane Austens *Stolz und Vorurteil*. Der Text ist hier bei [Project Gutenberg](https://www.gutenberg.org/files/1342/1342-h/1342-h.htm) verfügbar. Das folgende Beispiel zeigt ein kurzes Programm, das das Sentiment der ersten und letzten Sätze des Buches analysiert und dessen Polarität und Subjektivität/Objektivität anzeigt.

Du solltest die `TextBlob`-Bibliothek (oben beschrieben) verwenden, um `sentiment` zu bestimmen (du musst keinen eigenen Sentiment-Rechner schreiben) in der folgenden Aufgabe.

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

## Herausforderung - Sentiment-Polarität überprüfen

Deine Aufgabe ist es, mithilfe der Sentiment-Polarität zu bestimmen, ob *Stolz und Vorurteil* mehr absolut positive Sätze als absolut negative hat. Für diese Aufgabe kannst du davon ausgehen, dass ein Polaritätswert von 1 oder -1 absolut positiv bzw. negativ ist.

**Schritte:**

1. Lade eine [Kopie von Stolz und Vorurteil](https://www.gutenberg.org/files/1342/1342-h/1342-h.htm) von Project Gutenberg als .txt-Datei herunter. Entferne die Metadaten am Anfang und Ende der Datei, sodass nur der Originaltext übrig bleibt.
2. Öffne die Datei in Python und extrahiere den Inhalt als String.
3. Erstelle ein TextBlob mit dem Buch-String.
4. Analysiere jeden Satz im Buch in einer Schleife.
   1. Wenn die Polarität 1 oder -1 ist, speichere den Satz in einer Liste oder einem Array von positiven oder negativen Nachrichten.
5. Am Ende gib alle positiven und negativen Sätze (separat) und die Anzahl von jedem aus.

Hier ist eine [Beispiel-Lösung](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/3-Translation-Sentiment/solution/notebook.ipynb).

✅ Wissensüberprüfung

1. Das Sentiment basiert auf den im Satz verwendeten Wörtern, aber versteht der Code *die Wörter*?
2. Denkst du, dass die Sentiment-Polarität genau ist, oder mit anderen Worten, stimmst du den Bewertungen zu?
   1. Insbesondere, stimmst du der absoluten **positiven** Polarität der folgenden Sätze zu?
      * „Was für ein ausgezeichneter Vater ihr habt, Mädchen!“ sagte sie, als die Tür geschlossen war.
      * „Ihre Untersuchung von Mr. Darcy ist abgeschlossen, nehme ich an“, sagte Miss Bingley; „und was ist das Ergebnis?“ „Ich bin vollkommen überzeugt davon, dass Mr. Darcy keinen Makel hat.“
      * Wie wunderbar solche Dinge geschehen!
      * Ich habe die größte Abneigung der Welt gegen solche Dinge.
      * Charlotte ist eine ausgezeichnete Managerin, das wage ich zu sagen.
      * „Das ist wirklich entzückend!
      * Ich bin so glücklich!
      * Deine Idee mit den Ponys ist entzückend.
   2. Die nächsten 3 Sätze wurden mit einer absoluten positiven Polarität bewertet, aber bei genauer Betrachtung sind sie keine positiven Sätze. Warum dachte die Sentiment-Analyse, dass sie positive Sätze seien?
      * Glücklich werde ich sein, wenn sein Aufenthalt in Netherfield vorbei ist!“ „Ich wünschte, ich könnte etwas sagen, um dich zu trösten“, antwortete Elizabeth; „aber es liegt völlig außerhalb meiner Macht.
      * Wenn ich dich nur glücklich sehen könnte!
      * Unsere Not, meine liebe Lizzy, ist sehr groß.
   3. Stimmst du der absoluten **negativen** Polarität der folgenden Sätze zu?
      - Jeder ist angewidert von seinem Stolz.
      - „Ich würde gerne wissen, wie er sich unter Fremden verhält.“ „Dann wirst du es hören – aber bereite dich auf etwas sehr Schreckliches vor.“
      - Die Pause war für Elizabeths Gefühle schrecklich.
      - Es wäre schrecklich!

✅ Jeder Kenner von Jane Austen wird verstehen, dass sie ihre Bücher oft nutzt, um die lächerlicheren Aspekte der englischen Regency-Gesellschaft zu kritisieren. Elizabeth Bennett, die Hauptfigur in *Stolz und Vorurteil*, ist eine scharfsinnige soziale Beobachterin (wie die Autorin), und ihre Sprache ist oft stark nuanciert. Selbst Mr. Darcy (der Liebesinteressent in der Geschichte) bemerkt Elizabeths spielerischen und neckenden Sprachgebrauch: "Ich habe das Vergnügen, Sie lange genug zu kennen, um zu wissen, dass Sie große Freude daran finden, gelegentlich Meinungen zu äußern, die tatsächlich nicht Ihre eigenen sind."

---

## 🚀Herausforderung

Kannst du Marvin noch besser machen, indem du andere Merkmale aus der Benutzereingabe extrahierst?

## [Quiz nach der Vorlesung](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/36/)

## Rückblick & Selbststudium
Es gibt viele Möglichkeiten, um die Stimmung aus Texten zu extrahieren. Überlege, welche geschäftlichen Anwendungen von dieser Technik profitieren könnten. Denke auch darüber nach, wie sie schiefgehen kann. Lies mehr über ausgefeilte, unternehmensgerechte Systeme zur Stimmungsanalyse, wie zum Beispiel [Azure Text Analysis](https://docs.microsoft.com/azure/cognitive-services/Text-Analytics/how-tos/text-analytics-how-to-sentiment-analysis?tabs=version-3-1?WT.mc_id=academic-77952-leestott). Teste einige der oben genannten Sätze aus "Stolz und Vorurteil" und prüfe, ob sie Nuancen erkennen können.

## Aufgabe

[Poetische Freiheit](assignment.md)

---

**Haftungsausschluss**:  
Dieses Dokument wurde mit dem KI-Übersetzungsdienst [Co-op Translator](https://github.com/Azure/co-op-translator) übersetzt. Obwohl wir uns um Genauigkeit bemühen, beachten Sie bitte, dass automatisierte Übersetzungen Fehler oder Ungenauigkeiten enthalten können. Das Originaldokument in seiner ursprünglichen Sprache sollte als maßgebliche Quelle betrachtet werden. Für kritische Informationen wird eine professionelle menschliche Übersetzung empfohlen. Wir übernehmen keine Haftung für Missverständnisse oder Fehlinterpretationen, die sich aus der Nutzung dieser Übersetzung ergeben.