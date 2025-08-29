# Einführung in die Verarbeitung natürlicher Sprache

Diese Lektion behandelt eine kurze Geschichte und wichtige Konzepte der *Verarbeitung natürlicher Sprache*, einem Teilgebiet der *rechnergestützten Linguistik*.

## [Vorlesungsquiz](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/31/)

## Einführung

NLP, wie es allgemein bekannt ist, ist eines der bekanntesten Bereiche, in denen maschinelles Lernen angewendet und in Produktionssoftware genutzt wird.

✅ Können Sie an Software denken, die Sie jeden Tag verwenden und die wahrscheinlich einige NLP-Elemente enthält? Was ist mit Ihren Textverarbeitungsprogrammen oder mobilen Apps, die Sie regelmäßig nutzen?

Sie werden lernen über:

- **Die Idee von Sprachen**. Wie sich Sprachen entwickelt haben und was die Hauptstudienbereiche waren.
- **Definitionen und Konzepte**. Sie werden auch Definitionen und Konzepte darüber lernen, wie Computer Text verarbeiten, einschließlich Parsing, Grammatik und Identifizierung von Nomen und Verben. In dieser Lektion gibt es einige Programmieraufgaben, und mehrere wichtige Konzepte werden eingeführt, die Sie später in den nächsten Lektionen lernen werden.

## Rechnergestützte Linguistik

Rechnergestützte Linguistik ist ein Forschungs- und Entwicklungsbereich, der über viele Jahrzehnte untersucht, wie Computer mit Sprachen arbeiten, sie verstehen, übersetzen und kommunizieren können. Die Verarbeitung natürlicher Sprache (NLP) ist ein verwandtes Feld, das sich darauf konzentriert, wie Computer 'natürliche', oder menschliche, Sprachen verarbeiten können.

### Beispiel - Telefon-Diktat

Wenn Sie jemals Ihrem Telefon diktiert haben, anstatt zu tippen, oder einen virtuellen Assistenten eine Frage gestellt haben, wurde Ihre Sprache in eine Textform umgewandelt und dann aus der Sprache, die Sie gesprochen haben, *geparst*. Die erkannten Schlüsselwörter wurden dann in ein Format verarbeitet, das das Telefon oder der Assistent verstehen und darauf reagieren konnte.

![comprehension](../../../../translated_images/comprehension.619708fc5959b0f6a24ebffba2ad7b0625391a476141df65b43b59de24e45c6f.de.png)
> Echte sprachliche Verständlichkeit ist schwierig! Bild von [Jen Looper](https://twitter.com/jenlooper)

### Wie wird diese Technologie möglich?

Das ist möglich, weil jemand ein Computerprogramm geschrieben hat, um dies zu tun. Vor einigen Jahrzehnten sagten einige Science-Fiction-Autoren voraus, dass die Menschen hauptsächlich mit ihren Computern sprechen würden und die Computer immer genau wüssten, was sie meinten. Leider stellte sich heraus, dass dies ein schwierigeres Problem war, als viele dachten, und obwohl es heute viel besser verstanden wird, gibt es erhebliche Herausforderungen, 'perfekte' Verarbeitung natürlicher Sprache zu erreichen, wenn es darum geht, die Bedeutung eines Satzes zu verstehen. Dies ist ein besonders schwieriges Problem, wenn es darum geht, Humor oder Emotionen wie Sarkasmus in einem Satz zu erkennen.

An diesem Punkt erinnern Sie sich vielleicht an Schulstunden, in denen der Lehrer die Teile der Grammatik in einem Satz behandelte. In einigen Ländern wird den Schülern Grammatik und Linguistik als eigenständiges Fach beigebracht, aber in vielen Ländern sind diese Themen Teil des Sprachenlernens: entweder Ihre Muttersprache in der Grundschule (lesen und schreiben lernen) und vielleicht eine zweite Sprache in der weiterführenden Schule. Machen Sie sich keine Sorgen, wenn Sie kein Experte darin sind, Nomen von Verben oder Adverbien von Adjektiven zu unterscheiden!

Wenn Sie Schwierigkeiten mit dem Unterschied zwischen dem *Präsens* und dem *Verlaufsform Präsens* haben, sind Sie nicht allein. Das ist für viele Menschen, sogar für Muttersprachler, eine Herausforderung. Die gute Nachricht ist, dass Computer sehr gut darin sind, formale Regeln anzuwenden, und Sie werden lernen, Code zu schreiben, der einen Satz so *parsen* kann wie ein Mensch. Die größere Herausforderung, die Sie später untersuchen werden, ist das Verständnis der *Bedeutung* und des *Gefühls* eines Satzes.

## Voraussetzungen

Für diese Lektion ist die Hauptvoraussetzung, die Sprache dieser Lektion lesen und verstehen zu können. Es gibt keine Mathematikprobleme oder Gleichungen zu lösen. Während der ursprüngliche Autor diese Lektion in Englisch verfasst hat, ist sie auch in andere Sprachen übersetzt, sodass Sie möglicherweise eine Übersetzung lesen. Es gibt Beispiele, in denen eine Reihe von verschiedenen Sprachen verwendet wird (um die unterschiedlichen Grammatikregeln verschiedener Sprachen zu vergleichen). Diese sind *nicht* übersetzt, aber der erläuternde Text ist es, sodass die Bedeutung klar sein sollte.

Für die Programmieraufgaben werden Sie Python verwenden, und die Beispiele verwenden Python 3.8.

In diesem Abschnitt benötigen Sie und verwenden Sie:

- **Python 3 Verständnis**. Programmierverständnis in Python 3, diese Lektion verwendet Eingaben, Schleifen, Datei lesen, Arrays.
- **Visual Studio Code + Erweiterung**. Wir werden Visual Studio Code und seine Python-Erweiterung verwenden. Sie können auch eine Python-IDE Ihrer Wahl verwenden.
- **TextBlob**. [TextBlob](https://github.com/sloria/TextBlob) ist eine vereinfachte Textverarbeitungsbibliothek für Python. Befolgen Sie die Anweisungen auf der TextBlob-Website, um es auf Ihrem System zu installieren (installieren Sie auch die Korpora, wie unten gezeigt):

   ```bash
   pip install -U textblob
   python -m textblob.download_corpora
   ```

> 💡 Tipp: Sie können Python direkt in VS Code-Umgebungen ausführen. Überprüfen Sie die [Dokumentation](https://code.visualstudio.com/docs/languages/python?WT.mc_id=academic-77952-leestott) für weitere Informationen.

## Mit Maschinen sprechen

Die Geschichte des Versuchs, Computer menschliche Sprache verstehen zu lassen, reicht Jahrzehnte zurück, und einer der frühesten Wissenschaftler, der sich mit der Verarbeitung natürlicher Sprache beschäftigte, war *Alan Turing*.

### Der 'Turing-Test'

Als Turing in den 1950er Jahren *künstliche Intelligenz* erforschte, überlegte er, ob ein Konversationstest einem Menschen und einem Computer (über getippte Korrespondenz) gegeben werden könnte, bei dem der Mensch im Gespräch sich nicht sicher war, ob er mit einem anderen Menschen oder einem Computer sprach.

Wenn der Mensch nach einer bestimmten Gesprächsdauer nicht bestimmen konnte, ob die Antworten von einem Computer kamen oder nicht, könnte man dann sagen, dass der Computer *denkt*?

### Die Inspiration - 'das Nachahmungsspiel'

Die Idee dazu stammt von einem Partyspiel namens *Das Nachahmungsspiel*, bei dem ein Befrager allein in einem Raum ist und die Aufgabe hat, herauszufinden, welche von zwei Personen (in einem anderen Raum) männlich und weiblich sind. Der Befrager kann Notizen senden und muss versuchen, Fragen zu stellen, bei denen die schriftlichen Antworten das Geschlecht der geheimnisvollen Person enthüllen. Natürlich versuchen die Spieler im anderen Raum, den Befrager hereinzulegen, indem sie Fragen so beantworten, dass sie den Befrager in die Irre führen oder verwirren, während sie auch den Anschein erwecken, ehrlich zu antworten.

### Entwicklung von Eliza

In den 1960er Jahren entwickelte ein MIT-Wissenschaftler namens *Joseph Weizenbaum* [*Eliza*](https://wikipedia.org/wiki/ELIZA), einen Computer-'Therapeuten', der dem Menschen Fragen stellte und den Anschein erweckte, ihre Antworten zu verstehen. Allerdings konnte Eliza zwar einen Satz parsen und bestimmte grammatikalische Konstrukte und Schlüsselwörter identifizieren, um eine angemessene Antwort zu geben, aber man konnte nicht sagen, dass sie den Satz *verstanden* hat. Wenn Eliza mit einem Satz im Format "**Ich bin** <u>traurig</u>" konfrontiert wurde, könnte sie die Wörter im Satz umstellen und ersetzen, um die Antwort "Wie lange bist **du** <u>traurig</u>?" zu bilden.

Dies erweckte den Eindruck, dass Eliza die Aussage verstand und eine Folgefrage stellte, während sie in Wirklichkeit nur die Zeitform änderte und einige Wörter hinzufügte. Wenn Eliza ein Schlüsselwort nicht identifizieren konnte, für das sie eine Antwort hatte, gab sie stattdessen eine zufällige Antwort, die auf viele verschiedene Aussagen anwendbar sein konnte. Eliza konnte leicht hereingelegt werden; wenn ein Benutzer beispielsweise schrieb "**Du bist** ein <u>Fahrrad</u>", könnte sie mit "Wie lange bin **ich** ein <u>Fahrrad</u>?" antworten, anstatt mit einer überlegteren Antwort.

[![Chatten mit Eliza](https://img.youtube.com/vi/RMK9AphfLco/0.jpg)](https://youtu.be/RMK9AphfLco "Chatten mit Eliza")

> 🎥 Klicken Sie auf das Bild oben für ein Video über das ursprüngliche ELIZA-Programm

> Hinweis: Sie können die ursprüngliche Beschreibung von [Eliza](https://cacm.acm.org/magazines/1966/1/13317-elizaa-computer-program-for-the-study-of-natural-language-communication-between-man-and-machine/abstract) lesen, die 1966 veröffentlicht wurde, wenn Sie ein ACM-Konto haben. Alternativ können Sie über Eliza auf [Wikipedia](https://wikipedia.org/wiki/ELIZA) lesen.

## Übung - Programmierung eines einfachen Konversationsbots

Ein Konversationsbot, wie Eliza, ist ein Programm, das Benutzereingaben anfordert und den Anschein erweckt, intelligent zu verstehen und zu antworten. Im Gegensatz zu Eliza wird unser Bot nicht mehrere Regeln haben, die ihm den Anschein eines intelligenten Gesprächs verleihen. Stattdessen wird unser Bot nur eine Fähigkeit haben, das Gespräch mit zufälligen Antworten aufrechtzuerhalten, die in fast jedem trivialen Gespräch funktionieren könnten.

### Der Plan

Ihre Schritte beim Erstellen eines Konversationsbots:

1. Drucken Sie Anweisungen aus, die den Benutzer beraten, wie er mit dem Bot interagieren kann
2. Starten Sie eine Schleife
   1. Akzeptieren Sie die Benutzereingabe
   2. Wenn der Benutzer um einen Ausstieg gebeten hat, dann aussteigen
   3. Verarbeiten Sie die Benutzereingabe und bestimmen Sie die Antwort (in diesem Fall ist die Antwort eine zufällige Auswahl aus einer Liste möglicher allgemeiner Antworten)
   4. Drucken Sie die Antwort aus
3. Schleife zurück zu Schritt 2

### Den Bot erstellen

Lassen Sie uns als Nächstes den Bot erstellen. Wir beginnen damit, einige Phrasen zu definieren.

1. Erstellen Sie diesen Bot selbst in Python mit den folgenden zufälligen Antworten:

    ```python
    random_responses = ["That is quite interesting, please tell me more.",
                        "I see. Do go on.",
                        "Why do you say that?",
                        "Funny weather we've been having, isn't it?",
                        "Let's change the subject.",
                        "Did you catch the game last night?"]
    ```

    Hier ist eine Beispielausgabe, die Ihnen als Leitfaden dient (Benutzereingabe steht in den Zeilen, die mit `>` beginnen):

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

    Eine mögliche Lösung für die Aufgabe finden Sie [hier](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/1-Introduction-to-NLP/solution/bot.py).

    ✅ Stoppen und nachdenken

    1. Glauben Sie, dass die zufälligen Antworten jemanden 'täuschen' würden, dass der Bot sie tatsächlich verstand?
    2. Welche Funktionen müsste der Bot haben, um effektiver zu sein?
    3. Wenn ein Bot wirklich die Bedeutung eines Satzes 'verstehen' könnte, müsste er dann auch die Bedeutung vorheriger Sätze in einem Gespräch 'erinnern'?

---

## 🚀Herausforderung

Wählen Sie eines der oben genannten "Stoppen und nachdenken"-Elemente und versuchen Sie, es in Code umzusetzen oder eine Lösung auf Papier mit Pseudocode zu schreiben.

In der nächsten Lektion lernen Sie eine Reihe anderer Ansätze zum Parsen natürlicher Sprache und maschinellem Lernen kennen.

## [Nachlese-Quiz](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/32/)

## Überprüfung & Selbststudium

Schauen Sie sich die untenstehenden Referenzen als weitere Lesegelegenheiten an.

### Referenzen

1. Schubert, Lenhart, "Rechnergestützte Linguistik", *Die Stanford-Enzyklopädie der Philosophie* (Frühjahr 2020 Ausgabe), Edward N. Zalta (Hrsg.), URL = <https://plato.stanford.edu/archives/spr2020/entries/computational-linguistics/>.
2. Princeton University "Über WordNet." [WordNet](https://wordnet.princeton.edu/). Princeton University. 2010. 

## Aufgabe 

[Nach einem Bot suchen](assignment.md)

**Haftungsausschluss**:  
Dieses Dokument wurde mithilfe von KI-Übersetzungsdiensten übersetzt. Obwohl wir um Genauigkeit bemüht sind, beachten Sie bitte, dass automatisierte Übersetzungen Fehler oder Ungenauigkeiten enthalten können. Das Originaldokument in seiner ursprünglichen Sprache sollte als die maßgebliche Quelle angesehen werden. Für wichtige Informationen wird eine professionelle menschliche Übersetzung empfohlen. Wir übernehmen keine Verantwortung für Missverständnisse oder Fehlinterpretationen, die aus der Verwendung dieser Übersetzung entstehen.