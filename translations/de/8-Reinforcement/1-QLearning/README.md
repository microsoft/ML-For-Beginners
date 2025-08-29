## Überprüfung der Strategie

Da die Q-Tabelle die "Attraktivität" jeder Aktion in jedem Zustand auflistet, ist es ziemlich einfach, sie zu verwenden, um die effiziente Navigation in unserer Welt zu definieren. Im einfachsten Fall können wir die Aktion auswählen, die dem höchsten Q-Tabelle-Wert entspricht: (Codeblock 9)

```python
def qpolicy_strict(m):
        x,y = m.human
        v = probs(Q[x,y])
        a = list(actions)[np.argmax(v)]
        return a

walk(m,qpolicy_strict)
```

> Wenn Sie den obigen Code mehrmals ausprobieren, könnten Sie feststellen, dass er manchmal "hängt" und Sie die STOP-Taste im Notizbuch drücken müssen, um ihn zu unterbrechen. Dies geschieht, weil es Situationen geben könnte, in denen zwei Zustände in Bezug auf den optimalen Q-Wert "aufeinander zeigen", in diesem Fall bewegt sich der Agent unendlich zwischen diesen Zuständen hin und her.

## 🚀Herausforderung

> **Aufgabe 1:** Ändern Sie die `walk` function to limit the maximum length of path by a certain number of steps (say, 100), and watch the code above return this value from time to time.

> **Task 2:** Modify the `walk` function so that it does not go back to the places where it has already been previously. This will prevent `walk` from looping, however, the agent can still end up being "trapped" in a location from which it is unable to escape.

## Navigation

A better navigation policy would be the one that we used during training, which combines exploitation and exploration. In this policy, we will select each action with a certain probability, proportional to the values in the Q-Table. This strategy may still result in the agent returning back to a position it has already explored, but, as you can see from the code below, it results in a very short average path to the desired location (remember that `print_statistics`, die die Simulation 100 Mal ausführt: (Codeblock 10)

```python
def qpolicy(m):
        x,y = m.human
        v = probs(Q[x,y])
        a = random.choices(list(actions),weights=v)[0]
        return a

print_statistics(qpolicy)
```

Nach dem Ausführen dieses Codes sollten Sie eine viel kürzere durchschnittliche Pfadlänge als zuvor erhalten, im Bereich von 3-6.

## Untersuchung des Lernprozesses

Wie bereits erwähnt, ist der Lernprozess ein Gleichgewicht zwischen Exploration und der Erkundung des erlangten Wissens über die Struktur des Problembereichs. Wir haben gesehen, dass sich die Ergebnisse des Lernens (die Fähigkeit, einem Agenten zu helfen, einen kurzen Weg zum Ziel zu finden) verbessert haben, aber es ist auch interessant zu beobachten, wie sich die durchschnittliche Pfadlänge während des Lernprozesses verhält:

Die Erkenntnisse können zusammengefasst werden als:

- **Durchschnittliche Pfadlänge steigt**. Was wir hier sehen, ist, dass die durchschnittliche Pfadlänge zunächst zunimmt. Dies liegt wahrscheinlich daran, dass wir, wenn wir nichts über die Umgebung wissen, wahrscheinlich in schlechten Zuständen, Wasser oder dem Wolf, gefangen werden. Während wir mehr lernen und dieses Wissen nutzen, können wir die Umgebung länger erkunden, aber wir wissen immer noch nicht genau, wo die Äpfel sind.

- **Pfadlänge verringert sich, während wir mehr lernen**. Sobald wir genug gelernt haben, wird es für den Agenten einfacher, das Ziel zu erreichen, und die Pfadlänge beginnt zu sinken. Wir sind jedoch weiterhin offen für Erkundungen, sodass wir oft vom besten Pfad abweichen und neue Optionen erkunden, was den Pfad länger macht als optimal.

- **Längensteigerung abrupt**. Was wir auch in diesem Diagramm beobachten, ist, dass die Länge an einem Punkt abrupt anstieg. Dies zeigt die stochastische Natur des Prozesses an und dass wir zu einem bestimmten Zeitpunkt die Q-Tabellen-Koeffizienten "verderben" können, indem wir sie mit neuen Werten überschreiben. Dies sollte idealerweise minimiert werden, indem die Lernrate verringert wird (zum Beispiel passen wir gegen Ende des Trainings die Q-Tabellen-Werte nur um einen kleinen Wert an).

Insgesamt ist es wichtig, sich daran zu erinnern, dass der Erfolg und die Qualität des Lernprozesses erheblich von Parametern wie Lernrate, Lernratenverringerung und Abzinsungsfaktor abhängen. Diese werden oft als **Hyperparameter** bezeichnet, um sie von **Parametern** zu unterscheiden, die wir während des Trainings optimieren (zum Beispiel die Q-Tabellen-Koeffizienten). Der Prozess, die besten Hyperparameterwerte zu finden, wird als **Hyperparameter-Optimierung** bezeichnet und verdient ein eigenes Thema.

## [Nachlese-Quiz](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/46/)

## Aufgabe 
[Eine realistischere Welt](assignment.md)

**Haftungsausschluss**:  
Dieses Dokument wurde mithilfe von KI-Übersetzungsdiensten übersetzt. Obwohl wir uns um Genauigkeit bemühen, beachten Sie bitte, dass automatisierte Übersetzungen Fehler oder Ungenauigkeiten enthalten können. Das Originaldokument in seiner Ausgangssprache sollte als die maßgebliche Quelle betrachtet werden. Für wichtige Informationen wird eine professionelle menschliche Übersetzung empfohlen. Wir übernehmen keine Haftung für Missverständnisse oder Fehlinterpretationen, die aus der Verwendung dieser Übersetzung entstehen.