<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "89e923cf3e8bdff9662536e8bf9516e6",
  "translation_date": "2025-09-03T22:02:58+00:00",
  "source_file": "6-NLP/1-Introduction-to-NLP/README.md",
  "language_code": "de"
}
-->
# Einführung in die Verarbeitung natürlicher Sprache

Diese Lektion behandelt eine kurze Geschichte und wichtige Konzepte der *Verarbeitung natürlicher Sprache*, einem Teilgebiet der *Computerlinguistik*.

## [Quiz vor der Vorlesung](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/31/)

## Einführung

NLP, wie es allgemein bekannt ist, gehört zu den bekanntesten Bereichen, in denen maschinelles Lernen angewendet und in Produktionssoftware eingesetzt wurde.

✅ Kannst du dir Software vorstellen, die du täglich nutzt und die wahrscheinlich NLP integriert hat? Was ist mit deinen Textverarbeitungsprogrammen oder mobilen Apps, die du regelmäßig verwendest?

Du wirst lernen:

- **Die Idee von Sprachen**. Wie Sprachen entstanden sind und welche Hauptbereiche der Forschung es gibt.
- **Definition und Konzepte**. Du wirst auch Definitionen und Konzepte darüber lernen, wie Computer Text verarbeiten, einschließlich Parsing, Grammatik und der Identifizierung von Substantiven und Verben. Es gibt einige Programmieraufgaben in dieser Lektion, und mehrere wichtige Konzepte werden eingeführt, die du später in den nächsten Lektionen programmieren wirst.

## Computerlinguistik

Die Computerlinguistik ist ein Forschungs- und Entwicklungsbereich, der sich über viele Jahrzehnte erstreckt und untersucht, wie Computer mit Sprachen arbeiten, sie verstehen, übersetzen und sogar kommunizieren können. Die Verarbeitung natürlicher Sprache (NLP) ist ein verwandtes Feld, das sich darauf konzentriert, wie Computer 'natürliche', also menschliche, Sprachen verarbeiten können.

### Beispiel - Telefon-Diktat

Wenn du jemals deinem Telefon etwas diktiert hast, anstatt zu tippen, oder einen virtuellen Assistenten eine Frage gestellt hast, wurde deine Sprache in eine Textform umgewandelt und dann verarbeitet oder *geparst*. Die erkannten Schlüsselwörter wurden dann in ein Format umgewandelt, das das Telefon oder der Assistent verstehen und darauf reagieren konnte.

![Verständnis](../../../../translated_images/comprehension.619708fc5959b0f6a24ebffba2ad7b0625391a476141df65b43b59de24e45c6f.de.png)
> Echtes sprachliches Verständnis ist schwierig! Bild von [Jen Looper](https://twitter.com/jenlooper)

### Wie wird diese Technologie möglich gemacht?

Das ist möglich, weil jemand ein Computerprogramm geschrieben hat, um dies zu tun. Vor einigen Jahrzehnten haben einige Science-Fiction-Autoren vorausgesagt, dass Menschen hauptsächlich mit ihren Computern sprechen würden und die Computer immer genau verstehen würden, was sie meinten. Leider stellte sich heraus, dass dies ein schwierigeres Problem war, als viele sich vorgestellt hatten. Obwohl es heute viel besser verstanden wird, gibt es erhebliche Herausforderungen, um ein 'perfektes' Verständnis der natürlichen Sprache zu erreichen, insbesondere wenn es darum geht, den Sinn eines Satzes zu verstehen. Dies ist besonders schwierig, wenn es darum geht, Humor oder Emotionen wie Sarkasmus in einem Satz zu erkennen.

Vielleicht erinnerst du dich jetzt an Schulstunden, in denen der Lehrer die Teile der Grammatik in einem Satz behandelt hat. In einigen Ländern wird Grammatik und Linguistik als eigenständiges Fach unterrichtet, in vielen anderen sind diese Themen jedoch Teil des Sprachenlernens: entweder deiner Erstsprache in der Grundschule (Lesen und Schreiben lernen) und möglicherweise einer Zweitsprache in der weiterführenden Schule. Mach dir keine Sorgen, wenn du kein Experte darin bist, Substantive von Verben oder Adverbien von Adjektiven zu unterscheiden!

Wenn du Schwierigkeiten hast, den Unterschied zwischen dem *einfachen Präsens* und dem *Präsens Progressiv* zu erkennen, bist du nicht allein. Dies ist für viele Menschen eine Herausforderung, selbst für Muttersprachler einer Sprache. Die gute Nachricht ist, dass Computer wirklich gut darin sind, formale Regeln anzuwenden, und du wirst lernen, Code zu schreiben, der einen Satz genauso gut wie ein Mensch *parsen* kann. Die größere Herausforderung, die du später untersuchen wirst, besteht darin, den *Sinn* und die *Stimmung* eines Satzes zu verstehen.

## Voraussetzungen

Für diese Lektion ist die Hauptvoraussetzung, die Sprache dieser Lektion lesen und verstehen zu können. Es gibt keine mathematischen Probleme oder Gleichungen zu lösen. Während der ursprüngliche Autor diese Lektion auf Englisch geschrieben hat, ist sie auch in andere Sprachen übersetzt, sodass du möglicherweise eine Übersetzung liest. Es gibt Beispiele, in denen verschiedene Sprachen verwendet werden (um die unterschiedlichen Grammatikregeln verschiedener Sprachen zu vergleichen). Diese werden *nicht* übersetzt, aber der erläuternde Text wird übersetzt, sodass die Bedeutung klar sein sollte.

Für die Programmieraufgaben wirst du Python verwenden, und die Beispiele nutzen Python 3.8.

In diesem Abschnitt wirst du benötigen und verwenden:

- **Python 3 Verständnis**. Verständnis der Programmiersprache Python 3, diese Lektion verwendet Eingaben, Schleifen, Dateilesen, Arrays.
- **Visual Studio Code + Erweiterung**. Wir werden Visual Studio Code und dessen Python-Erweiterung verwenden. Du kannst auch eine Python-IDE deiner Wahl nutzen.
- **TextBlob**. [TextBlob](https://github.com/sloria/TextBlob) ist eine vereinfachte Textverarbeitungsbibliothek für Python. Folge den Anweisungen auf der TextBlob-Seite, um es auf deinem System zu installieren (installiere auch die Corpora, wie unten gezeigt):

   ```bash
   pip install -U textblob
   python -m textblob.download_corpora
   ```

> 💡 Tipp: Du kannst Python direkt in VS Code-Umgebungen ausführen. Sieh dir die [Dokumentation](https://code.visualstudio.com/docs/languages/python?WT.mc_id=academic-77952-leestott) für weitere Informationen an.

## Mit Maschinen sprechen

Die Geschichte, wie versucht wurde, Computer dazu zu bringen, menschliche Sprache zu verstehen, reicht Jahrzehnte zurück. Einer der frühesten Wissenschaftler, der sich mit der Verarbeitung natürlicher Sprache beschäftigte, war *Alan Turing*.

### Der 'Turing-Test'

Als Turing in den 1950er Jahren *künstliche Intelligenz* erforschte, überlegte er, ob ein Konversationstest durchgeführt werden könnte, bei dem ein Mensch und ein Computer (über schriftliche Korrespondenz) miteinander kommunizieren, und der Mensch in der Konversation nicht sicher ist, ob er mit einem anderen Menschen oder einem Computer spricht.

Wenn der Mensch nach einer bestimmten Länge der Konversation nicht feststellen konnte, ob die Antworten von einem Computer stammen oder nicht, könnte man dann sagen, dass der Computer *denkt*?

### Die Inspiration - 'Das Imitationsspiel'

Die Idee dazu kam von einem Partyspiel namens *Das Imitationsspiel*, bei dem ein Fragesteller allein in einem Raum ist und herausfinden soll, welche von zwei Personen (in einem anderen Raum) männlich und weiblich sind. Der Fragesteller kann Notizen senden und muss versuchen, Fragen zu stellen, bei denen die schriftlichen Antworten das Geschlecht der mysteriösen Person offenbaren. Natürlich versuchen die Spieler im anderen Raum, den Fragesteller zu täuschen, indem sie die Fragen so beantworten, dass sie den Fragesteller in die Irre führen oder verwirren, während sie gleichzeitig den Anschein erwecken, ehrlich zu antworten.

### Entwicklung von Eliza

In den 1960er Jahren entwickelte ein MIT-Wissenschaftler namens *Joseph Weizenbaum* [*Eliza*](https://wikipedia.org/wiki/ELIZA), eine Computer-'Therapeutin', die dem Menschen Fragen stellte und den Eindruck erweckte, seine Antworten zu verstehen. Allerdings konnte Eliza zwar einen Satz parsen und bestimmte grammatikalische Konstrukte und Schlüsselwörter identifizieren, um eine vernünftige Antwort zu geben, aber man konnte nicht sagen, dass sie den Satz *verstand*. Wenn Eliza beispielsweise ein Satz im Format "**Ich bin** <u>traurig</u>" präsentiert wurde, könnte sie die Wörter im Satz umstellen und ersetzen, um die Antwort "Wie lange bist **du** <u>traurig</u>" zu bilden.

Dies erweckte den Eindruck, dass Eliza die Aussage verstand und eine Anschlussfrage stellte, während sie in Wirklichkeit die Zeitform änderte und einige Wörter hinzufügte. Wenn Eliza ein Schlüsselwort nicht identifizieren konnte, für das sie eine Antwort hatte, gab sie stattdessen eine zufällige Antwort, die auf viele verschiedene Aussagen anwendbar sein könnte. Eliza konnte leicht getäuscht werden, zum Beispiel wenn ein Benutzer schrieb "**Du bist** ein <u>Fahrrad</u>", könnte sie mit "Wie lange bin **ich** ein <u>Fahrrad</u>?" antworten, anstatt mit einer vernünftigeren Antwort.

[![Chatten mit Eliza](https://img.youtube.com/vi/RMK9AphfLco/0.jpg)](https://youtu.be/RMK9AphfLco "Chatten mit Eliza")

> 🎥 Klicke auf das Bild oben für ein Video über das ursprüngliche ELIZA-Programm

> Hinweis: Du kannst die ursprüngliche Beschreibung von [Eliza](https://cacm.acm.org/magazines/1966/1/13317-elizaa-computer-program-for-the-study-of-natural-language-communication-between-man-and-machine/abstract) lesen, die 1966 veröffentlicht wurde, wenn du ein ACM-Konto hast. Alternativ kannst du über Eliza auf [Wikipedia](https://wikipedia.org/wiki/ELIZA) lesen.

## Übung - Programmieren eines einfachen Konversationsbots

Ein Konversationsbot, wie Eliza, ist ein Programm, das Benutzereingaben entgegennimmt und den Eindruck erweckt, intelligent zu verstehen und zu antworten. Im Gegensatz zu Eliza wird unser Bot keine Vielzahl von Regeln haben, die den Eindruck einer intelligenten Konversation erwecken. Stattdessen wird unser Bot nur eine Fähigkeit haben: die Konversation mit zufälligen Antworten fortzusetzen, die in fast jeder trivialen Konversation funktionieren könnten.

### Der Plan

Deine Schritte beim Erstellen eines Konversationsbots:

1. Drucke Anweisungen, die den Benutzer darüber informieren, wie er mit dem Bot interagieren soll
2. Starte eine Schleife
   1. Akzeptiere Benutzereingaben
   2. Wenn der Benutzer um Beenden bittet, dann beenden
   3. Verarbeite die Benutzereingaben und bestimme die Antwort (in diesem Fall ist die Antwort eine zufällige Auswahl aus einer Liste möglicher allgemeiner Antworten)
   4. Drucke die Antwort
3. Kehre zu Schritt 2 zurück

### Den Bot erstellen

Lass uns den Bot erstellen. Wir beginnen damit, einige Phrasen zu definieren.

1. Erstelle diesen Bot selbst in Python mit den folgenden zufälligen Antworten:

    ```python
    random_responses = ["That is quite interesting, please tell me more.",
                        "I see. Do go on.",
                        "Why do you say that?",
                        "Funny weather we've been having, isn't it?",
                        "Let's change the subject.",
                        "Did you catch the game last night?"]
    ```

    Hier ist eine Beispielausgabe zur Orientierung (Benutzereingaben beginnen mit `>`):

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

    Eine mögliche Lösung für die Aufgabe findest du [hier](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/1-Introduction-to-NLP/solution/bot.py)

    ✅ Halte inne und überlege

    1. Glaubst du, dass die zufälligen Antworten jemanden dazu bringen könnten, zu denken, dass der Bot sie tatsächlich verstanden hat?
    2. Welche Funktionen müsste der Bot haben, um effektiver zu sein?
    3. Wenn ein Bot wirklich den Sinn eines Satzes verstehen könnte, müsste er dann auch den Sinn vorheriger Sätze in einer Konversation 'merken'?

---

## 🚀 Herausforderung

Wähle eines der oben genannten "Halte inne und überlege"-Elemente aus und versuche entweder, es in Code umzusetzen, oder schreibe eine Lösung auf Papier mit Pseudocode.

In der nächsten Lektion wirst du eine Reihe anderer Ansätze zur Verarbeitung natürlicher Sprache und maschinellem Lernen kennenlernen.

## [Quiz nach der Vorlesung](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/32/)

## Rückblick & Selbststudium

Sieh dir die unten stehenden Referenzen als weitere Lesemöglichkeiten an.

### Referenzen

1. Schubert, Lenhart, "Computational Linguistics", *The Stanford Encyclopedia of Philosophy* (Spring 2020 Edition), Edward N. Zalta (ed.), URL = <https://plato.stanford.edu/archives/spr2020/entries/computational-linguistics/>.
2. Princeton University "About WordNet." [WordNet](https://wordnet.princeton.edu/). Princeton University. 2010. 

## Aufgabe 

[Suche nach einem Bot](assignment.md)

---

**Haftungsausschluss**:  
Dieses Dokument wurde mit dem KI-Übersetzungsdienst [Co-op Translator](https://github.com/Azure/co-op-translator) übersetzt. Obwohl wir uns um Genauigkeit bemühen, beachten Sie bitte, dass automatisierte Übersetzungen Fehler oder Ungenauigkeiten enthalten können. Das Originaldokument in seiner ursprünglichen Sprache sollte als maßgebliche Quelle betrachtet werden. Für kritische Informationen wird eine professionelle menschliche Übersetzung empfohlen. Wir übernehmen keine Haftung für Missverständnisse oder Fehlinterpretationen, die sich aus der Nutzung dieser Übersetzung ergeben.