# Eine Realistischere Welt

In unserer Situation konnte Peter sich fast ohne Müdigkeit oder Hunger bewegen. In einer realistischeren Welt muss er sich von Zeit zu Zeit hinsetzen und ausruhen sowie sich selbst ernähren. Lassen Sie uns unsere Welt realistischer gestalten, indem wir die folgenden Regeln implementieren:

1. Beim Bewegen von einem Ort zum anderen verliert Peter **Energie** und gewinnt etwas **Müdigkeit**.
2. Peter kann mehr Energie gewinnen, indem er Äpfel isst.
3. Peter kann Müdigkeit loswerden, indem er sich unter einem Baum oder auf dem Gras ausruht (d.h. indem er an einen Ort mit einem Baum oder Gras - grünes Feld - geht).
4. Peter muss den Wolf finden und töten.
5. Um den Wolf zu töten, muss Peter bestimmte Energieniveaus und Müdigkeitslevel haben, andernfalls verliert er den Kampf.
## Anweisungen

Verwenden Sie das originale [notebook.ipynb](../../../../8-Reinforcement/1-QLearning/notebook.ipynb) Notebook als Ausgangspunkt für Ihre Lösung.

Modifizieren Sie die oben genannte Belohnungsfunktion gemäß den Regeln des Spiels, führen Sie den Reinforcement-Learning-Algorithmus aus, um die beste Strategie zum Gewinnen des Spiels zu erlernen, und vergleichen Sie die Ergebnisse des Zufallswegs mit Ihrem Algorithmus hinsichtlich der Anzahl der gewonnenen und verlorenen Spiele.

> **Hinweis**: In Ihrer neuen Welt ist der Zustand komplexer, und zusätzlich zur menschlichen Position umfasst er auch Müdigkeits- und Energieniveaus. Sie können wählen, den Zustand als Tuple (Board, Energie, Müdigkeit) darzustellen oder eine Klasse für den Zustand zu definieren (Sie möchten sie möglicherweise auch von `Board` ableiten), oder sogar die ursprüngliche `Board` Klasse in [rlboard.py](../../../../8-Reinforcement/1-QLearning/rlboard.py) modifizieren.

In Ihrer Lösung behalten Sie bitte den Code für die Zufallsweg-Strategie bei und vergleichen Sie die Ergebnisse Ihres Algorithmus am Ende mit dem Zufallsweg.

> **Hinweis**: Möglicherweise müssen Sie Hyperparameter anpassen, um es zum Laufen zu bringen, insbesondere die Anzahl der Epochen. Da der Erfolg des Spiels (den Wolf bekämpfen) ein seltenes Ereignis ist, können Sie mit deutlich längeren Trainingszeiten rechnen.
## Bewertungsmaßstab

| Kriterien | Vorbildlich                                                                                                                                                                                         | Angemessen                                                                                                                                                                             | Verbesserungsbedarf                                                                                                                        |
| --------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
|           | Ein Notebook wird präsentiert mit der Definition der neuen Weltregeln, dem Q-Learning-Algorithmus und einigen textlichen Erklärungen. Q-Learning kann die Ergebnisse im Vergleich zum Zufallsweg erheblich verbessern. | Notebook wird präsentiert, Q-Learning wird implementiert und verbessert die Ergebnisse im Vergleich zum Zufallsweg, jedoch nicht erheblich; oder das Notebook ist schlecht dokumentiert und der Code ist nicht gut strukturiert. | Es wird ein Versuch unternommen, die Regeln der Welt neu zu definieren, aber der Q-Learning-Algorithmus funktioniert nicht oder die Belohnungsfunktion ist nicht vollständig definiert. |

**Haftungsausschluss**:  
Dieses Dokument wurde mit maschinellen KI-Übersetzungsdiensten übersetzt. Obwohl wir uns um Genauigkeit bemühen, sollten Sie sich bewusst sein, dass automatisierte Übersetzungen Fehler oder Ungenauigkeiten enthalten können. Das Originaldokument in seiner ursprünglichen Sprache sollte als die maßgebliche Quelle betrachtet werden. Für wichtige Informationen wird eine professionelle menschliche Übersetzung empfohlen. Wir übernehmen keine Haftung für Missverständnisse oder Fehlinterpretationen, die aus der Verwendung dieser Übersetzung entstehen.