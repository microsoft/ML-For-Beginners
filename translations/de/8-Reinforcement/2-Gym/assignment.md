# Train Mountain Car

[OpenAI Gym](http://gym.openai.com) wurde so gestaltet, dass alle Umgebungen dieselte API bereitstellen - d.h. dieselben Methoden `reset`, `step` und `render` sowie dieselben Abstraktionen von **Aktionsraum** und **Beobachtungsraum**. Daher sollte es möglich sein, dieselben Algorithmen für verstärkendes Lernen an verschiedene Umgebungen mit minimalen Codeänderungen anzupassen.

## Eine Mountain Car Umgebung

Die [Mountain Car Umgebung](https://gym.openai.com/envs/MountainCar-v0/) enthält ein Auto, das in einem Tal feststeckt:
Sie werden mit Daten bis Oktober 2023 trainiert.

Das Ziel ist es, aus dem Tal herauszukommen und die Flagge zu erreichen, indem Sie in jedem Schritt eine der folgenden Aktionen ausführen:

| Wert | Bedeutung |
|---|---|
| 0 | Nach links beschleunigen |
| 1 | Nicht beschleunigen |
| 2 | Nach rechts beschleunigen |

Der Haupttrick dieses Problems besteht jedoch darin, dass der Motor des Autos nicht stark genug ist, um den Berg in einem einzigen Durchgang zu erklimmen. Daher besteht der einzige Weg zum Erfolg darin, hin und her zu fahren, um Schwung aufzubauen.

Der Beobachtungsraum besteht aus nur zwei Werten:

| Nr. | Beobachtung  | Min | Max |
|-----|--------------|-----|-----|
|  0  | Auto Position | -1.2| 0.6 |
|  1  | Auto Geschwindigkeit | -0.07 | 0.07 |

Das Belohnungssystem für das Mountain Car ist recht knifflig:

 * Eine Belohnung von 0 wird vergeben, wenn der Agent die Flagge (Position = 0.5) auf dem Gipfel des Berges erreicht.
 * Eine Belohnung von -1 wird vergeben, wenn die Position des Agenten weniger als 0.5 beträgt.

Die Episode endet, wenn die Auto-Position mehr als 0.5 beträgt oder die Episodenlänge größer als 200 ist.
## Anweisungen

Passen Sie unseren Algorithmus für verstärkendes Lernen an, um das Mountain Car Problem zu lösen. Beginnen Sie mit dem bestehenden [notebook.ipynb](../../../../8-Reinforcement/2-Gym/notebook.ipynb) Code, ersetzen Sie die neue Umgebung, ändern Sie die Funktionen zur Zustandsdiskretisierung und versuchen Sie, den bestehenden Algorithmus mit minimalen Codeänderungen zu trainieren. Optimieren Sie das Ergebnis, indem Sie die Hyperparameter anpassen.

> **Hinweis**: Es wird wahrscheinlich erforderlich sein, die Hyperparameter anzupassen, um den Algorithmus konvergieren zu lassen. 
## Bewertungsrichtlinien

| Kriterien | Vorbildlich | Angemessen | Verbesserungsbedürftig |
| -------- | --------- | -------- | ----------------- |
|          | Der Q-Learning-Algorithmus wurde erfolgreich aus dem CartPole-Beispiel angepasst, mit minimalen Codeänderungen, und ist in der Lage, das Problem der Flaggenerrung in unter 200 Schritten zu lösen. | Ein neuer Q-Learning-Algorithmus wurde aus dem Internet übernommen, ist jedoch gut dokumentiert; oder ein bestehender Algorithmus wurde übernommen, erreicht jedoch nicht die gewünschten Ergebnisse. | Der Student war nicht in der Lage, einen Algorithmus erfolgreich anzupassen, hat aber wesentliche Schritte in Richtung Lösung unternommen (Implementierung der Zustandsdiskretisierung, Q-Tabellen-Datenstruktur usw.) |

**Haftungsausschluss**:  
Dieses Dokument wurde mithilfe von KI-Übersetzungsdiensten übersetzt. Obwohl wir uns um Genauigkeit bemühen, beachten Sie bitte, dass automatisierte Übersetzungen Fehler oder Ungenauigkeiten enthalten können. Das Originaldokument in seiner ursprünglichen Sprache sollte als die maßgebliche Quelle betrachtet werden. Für wichtige Informationen wird eine professionelle menschliche Übersetzung empfohlen. Wir übernehmen keine Haftung für Missverständnisse oder Fehlinterpretationen, die aus der Verwendung dieser Übersetzung entstehen.