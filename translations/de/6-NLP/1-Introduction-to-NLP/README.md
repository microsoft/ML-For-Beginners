<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "1c2ec40cf55c98a028a359c27ef7e45a",
  "translation_date": "2025-09-04T22:08:09+00:00",
  "source_file": "6-NLP/1-Introduction-to-NLP/README.md",
  "language_code": "de"
}
-->
# Einführung in die Verarbeitung natürlicher Sprache

Diese Lektion behandelt eine kurze Geschichte und wichtige Konzepte der *Verarbeitung natürlicher Sprache*, einem Teilbereich der *Computerlinguistik*.

## [Quiz vor der Vorlesung](https://ff-quizzes.netlify.app/en/ml/)

## Einführung

NLP, wie es allgemein bekannt ist, ist eines der bekanntesten Gebiete, in denen maschinelles Lernen angewendet und in Produktionssoftware eingesetzt wurde.

✅ Können Sie sich Software vorstellen, die Sie täglich nutzen und die wahrscheinlich NLP integriert hat? Was ist mit Ihren Textverarbeitungsprogrammen oder mobilen Apps, die Sie regelmäßig verwenden?

Sie werden lernen:

- **Die Idee von Sprachen**. Wie Sprachen entstanden sind und welche Hauptbereiche untersucht wurden.
- **Definition und Konzepte**. Sie werden auch Definitionen und Konzepte darüber lernen, wie Computer Text verarbeiten, einschließlich Parsing, Grammatik und der Identifizierung von Nomen und Verben. In dieser Lektion gibt es einige Programmieraufgaben, und es werden mehrere wichtige Konzepte eingeführt, die Sie später in den nächsten Lektionen programmieren lernen werden.

## Computerlinguistik

Die Computerlinguistik ist ein Forschungs- und Entwicklungsbereich, der sich über viele Jahrzehnte erstreckt und untersucht, wie Computer mit Sprachen arbeiten, sie verstehen, übersetzen und sogar kommunizieren können. Die Verarbeitung natürlicher Sprache (NLP) ist ein verwandtes Gebiet, das sich darauf konzentriert, wie Computer 'natürliche', also menschliche, Sprachen verarbeiten können.

### Beispiel - Telefon-Diktat

Wenn Sie jemals Ihrem Telefon etwas diktiert haben, anstatt zu tippen, oder einem virtuellen Assistenten eine Frage gestellt haben, wurde Ihre Sprache in eine Textform umgewandelt und dann verarbeitet oder *geparst* aus der Sprache, die Sie gesprochen haben. Die erkannten Schlüsselwörter wurden dann in ein Format verarbeitet, das das Telefon oder der Assistent verstehen und darauf reagieren konnte.

![Verständnis](../../../../6-NLP/1-Introduction-to-NLP/images/comprehension.png)
> Echtes sprachliches Verständnis ist schwierig! Bild von [Jen Looper](https://twitter.com/jenlooper)

### Wie wird diese Technologie möglich gemacht?

Dies ist möglich, weil jemand ein Computerprogramm geschrieben hat, um dies zu tun. Vor einigen Jahrzehnten haben einige Science-Fiction-Autoren vorausgesagt, dass Menschen hauptsächlich mit ihren Computern sprechen würden und die Computer immer genau verstehen würden, was sie meinten. Leider stellte sich heraus, dass dies ein schwierigeres Problem war, als viele sich vorgestellt hatten, und obwohl es heute viel besser verstanden wird, gibt es erhebliche Herausforderungen, um ein 'perfektes' Verständnis der natürlichen Sprache zu erreichen, insbesondere wenn es darum geht, den Sinn eines Satzes zu verstehen. Dies ist ein besonders schwieriges Problem, wenn es darum geht, Humor zu verstehen oder Emotionen wie Sarkasmus in einem Satz zu erkennen.

Vielleicht erinnern Sie sich jetzt an Schulstunden, in denen der Lehrer die Teile der Grammatik in einem Satz behandelt hat. In einigen Ländern wird Grammatik und Linguistik als eigenständiges Fach unterrichtet, in vielen anderen sind diese Themen jedoch Teil des Sprachenlernens: entweder Ihrer Erstsprache in der Grundschule (Lesen und Schreiben lernen) und möglicherweise einer Zweitsprache in der weiterführenden Schule. Machen Sie sich keine Sorgen, wenn Sie kein Experte darin sind, Nomen von Verben oder Adverbien von Adjektiven zu unterscheiden!

Wenn Sie Schwierigkeiten haben, den Unterschied zwischen dem *einfachen Präsens* und dem *Präsens Progressiv* zu erkennen, sind Sie nicht allein. Dies ist für viele Menschen eine Herausforderung, selbst für Muttersprachler einer Sprache. Die gute Nachricht ist, dass Computer sehr gut darin sind, formale Regeln anzuwenden, und Sie werden lernen, Code zu schreiben, der einen Satz genauso gut wie ein Mensch *parsen* kann. Die größere Herausforderung, die Sie später untersuchen werden, besteht darin, die *Bedeutung* und *Stimmung* eines Satzes zu verstehen.

## Voraussetzungen

Für diese Lektion ist die Hauptvoraussetzung, die Sprache dieser Lektion lesen und verstehen zu können. Es gibt keine mathematischen Probleme oder Gleichungen zu lösen. Während der ursprüngliche Autor diese Lektion auf Englisch geschrieben hat, ist sie auch in andere Sprachen übersetzt, sodass Sie möglicherweise eine Übersetzung lesen. Es gibt Beispiele, in denen eine Reihe verschiedener Sprachen verwendet wird (um die unterschiedlichen Grammatikregeln verschiedener Sprachen zu vergleichen). Diese werden *nicht* übersetzt, aber der erläuternde Text wird übersetzt, sodass die Bedeutung klar sein sollte.

Für die Programmieraufgaben verwenden Sie Python, und die Beispiele basieren auf Python 3.8.

In diesem Abschnitt benötigen und verwenden Sie:

- **Python 3 Verständnis**. Verständnis der Programmiersprache Python 3, diese Lektion verwendet Eingaben, Schleifen, Dateilesen, Arrays.
- **Visual Studio Code + Erweiterung**. Wir verwenden Visual Studio Code und dessen Python-Erweiterung. Sie können auch eine Python-IDE Ihrer Wahl verwenden.
- **TextBlob**. [TextBlob](https://github.com/sloria/TextBlob) ist eine vereinfachte Textverarbeitungsbibliothek für Python. Folgen Sie den Anweisungen auf der TextBlob-Website, um es auf Ihrem System zu installieren (installieren Sie auch die Corpora, wie unten gezeigt):

   ```bash
   pip install -U textblob
   python -m textblob.download_corpora
   ```

> 💡 Tipp: Sie können Python direkt in VS Code-Umgebungen ausführen. Weitere Informationen finden Sie in den [Dokumentationen](https://code.visualstudio.com/docs/languages/python?WT.mc_id=academic-77952-leestott).

## Mit Maschinen sprechen

Die Geschichte, Computer dazu zu bringen, menschliche Sprache zu verstehen, reicht Jahrzehnte zurück, und einer der frühesten Wissenschaftler, der sich mit der Verarbeitung natürlicher Sprache beschäftigte, war *Alan Turing*.

### Der 'Turing-Test'

Als Turing in den 1950er Jahren *künstliche Intelligenz* erforschte, überlegte er, ob ein Gesprächstest durchgeführt werden könnte, bei dem ein Mensch und ein Computer (über schriftliche Korrespondenz) miteinander kommunizieren, und der Mensch im Gespräch nicht sicher ist, ob er mit einem anderen Menschen oder einem Computer kommuniziert.

Wenn der Mensch nach einer bestimmten Gesprächsdauer nicht feststellen konnte, ob die Antworten von einem Computer stammen oder nicht, könnte man dann sagen, dass der Computer *denkt*?

### Die Inspiration - 'Das Imitationsspiel'

Die Idee dazu kam von einem Partyspiel namens *Das Imitationsspiel*, bei dem ein Fragesteller allein in einem Raum ist und herausfinden soll, welche der beiden Personen (in einem anderen Raum) männlich und weiblich sind. Der Fragesteller kann Notizen senden und muss versuchen, Fragen zu stellen, bei denen die schriftlichen Antworten das Geschlecht der mysteriösen Person offenbaren. Natürlich versuchen die Spieler im anderen Raum, den Fragesteller zu täuschen, indem sie Fragen so beantworten, dass sie den Fragesteller in die Irre führen oder verwirren, während sie gleichzeitig den Anschein erwecken, ehrlich zu antworten.

### Entwicklung von Eliza

In den 1960er Jahren entwickelte ein MIT-Wissenschaftler namens *Joseph Weizenbaum* [*Eliza*](https://wikipedia.org/wiki/ELIZA), eine Computer-'Therapeutin', die dem Menschen Fragen stellte und den Eindruck erweckte, seine Antworten zu verstehen. Während Eliza einen Satz parsen und bestimmte grammatikalische Konstrukte und Schlüsselwörter identifizieren konnte, um eine vernünftige Antwort zu geben, konnte man nicht sagen, dass sie den Satz *verstand*. Wenn Eliza beispielsweise ein Satz im Format "**Ich bin** <u>traurig</u>" präsentiert wurde, könnte sie Wörter im Satz umstellen und ersetzen, um die Antwort "Wie lange sind **Sie** <u>traurig</u>" zu bilden.

Dies erweckte den Eindruck, dass Eliza die Aussage verstand und eine Anschlussfrage stellte, während sie in Wirklichkeit die Zeitform änderte und einige Wörter hinzufügte. Wenn Eliza ein Schlüsselwort nicht identifizieren konnte, für das sie eine Antwort hatte, gab sie stattdessen eine zufällige Antwort, die auf viele verschiedene Aussagen anwendbar sein könnte. Eliza konnte leicht ausgetrickst werden, zum Beispiel wenn ein Benutzer schrieb "**Du bist** ein <u>Fahrrad</u>", könnte sie mit "Wie lange bin **ich** ein <u>Fahrrad</u>?" antworten, anstatt mit einer vernünftigeren Antwort.

[![Chatten mit Eliza](https://img.youtube.com/vi/RMK9AphfLco/0.jpg)](https://youtu.be/RMK9AphfLco "Chatten mit Eliza")

> 🎥 Klicken Sie auf das Bild oben für ein Video über das ursprüngliche ELIZA-Programm

> Hinweis: Sie können die ursprüngliche Beschreibung von [Eliza](https://cacm.acm.org/magazines/1966/1/13317-elizaa-computer-program-for-the-study-of-natural-language-communication-between-man-and-machine/abstract) lesen, die 1966 veröffentlicht wurde, wenn Sie ein ACM-Konto haben. Alternativ können Sie über Eliza auf [Wikipedia](https://wikipedia.org/wiki/ELIZA) lesen.

## Übung - Programmieren eines einfachen Konversationsbots

Ein Konversationsbot wie Eliza ist ein Programm, das Benutzereingaben entgegennimmt und scheinbar intelligent darauf reagiert. Im Gegensatz zu Eliza wird unser Bot keine Vielzahl von Regeln haben, die den Eindruck einer intelligenten Konversation erwecken. Stattdessen wird unser Bot nur eine Fähigkeit haben: die Konversation mit zufälligen Antworten fortzusetzen, die in fast jeder trivialen Unterhaltung funktionieren könnten.

### Der Plan

Ihre Schritte beim Erstellen eines Konversationsbots:

1. Drucken Sie Anweisungen, die den Benutzer darüber informieren, wie er mit dem Bot interagieren soll.
2. Starten Sie eine Schleife:
   1. Akzeptieren Sie Benutzereingaben.
   2. Wenn der Benutzer den Wunsch äußert, zu beenden, dann beenden Sie.
   3. Verarbeiten Sie die Benutzereingaben und bestimmen Sie die Antwort (in diesem Fall ist die Antwort eine zufällige Auswahl aus einer Liste möglicher allgemeiner Antworten).
   4. Drucken Sie die Antwort.
3. Kehren Sie zu Schritt 2 zurück.

### Den Bot erstellen

Lassen Sie uns den Bot erstellen. Wir beginnen mit der Definition einiger Phrasen.

1. Erstellen Sie diesen Bot selbst in Python mit den folgenden zufälligen Antworten:

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

    Eine mögliche Lösung für die Aufgabe finden Sie [hier](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/1-Introduction-to-NLP/solution/bot.py)

    ✅ Halten Sie inne und überlegen Sie:

    1. Glauben Sie, dass die zufälligen Antworten jemanden dazu bringen könnten, zu denken, dass der Bot sie tatsächlich versteht?
    2. Welche Funktionen müsste der Bot haben, um effektiver zu sein?
    3. Wenn ein Bot wirklich die *Bedeutung* eines Satzes verstehen könnte, müsste er dann auch die Bedeutung vorheriger Sätze in einem Gespräch *merken*?

---

## 🚀 Herausforderung

Wählen Sie eines der oben genannten "Halten Sie inne und überlegen Sie"-Elemente aus und versuchen Sie entweder, es in Code umzusetzen, oder schreiben Sie eine Lösung auf Papier mit Pseudocode.

In der nächsten Lektion lernen Sie eine Reihe anderer Ansätze zur Verarbeitung natürlicher Sprache und maschinellem Lernen kennen.

## [Quiz nach der Vorlesung](https://ff-quizzes.netlify.app/en/ml/)

## Überprüfung & Selbststudium

Sehen Sie sich die unten stehenden Referenzen als weitere Lesemöglichkeiten an.

### Referenzen

1. Schubert, Lenhart, "Computational Linguistics", *The Stanford Encyclopedia of Philosophy* (Spring 2020 Edition), Edward N. Zalta (ed.), URL = <https://plato.stanford.edu/archives/spr2020/entries/computational-linguistics/>.
2. Princeton University "About WordNet." [WordNet](https://wordnet.princeton.edu/). Princeton University. 2010. 

## Aufgabe 

[Suche nach einem Bot](assignment.md)

---

**Haftungsausschluss**:  
Dieses Dokument wurde mit dem KI-Übersetzungsdienst [Co-op Translator](https://github.com/Azure/co-op-translator) übersetzt. Obwohl wir uns um Genauigkeit bemühen, weisen wir darauf hin, dass automatisierte Übersetzungen Fehler oder Ungenauigkeiten enthalten können. Das Originaldokument in seiner ursprünglichen Sprache sollte als maßgebliche Quelle betrachtet werden. Für kritische Informationen wird eine professionelle menschliche Übersetzung empfohlen. Wir übernehmen keine Haftung für Missverständnisse oder Fehlinterpretationen, die aus der Nutzung dieser Übersetzung entstehen.