<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "0ffe994d1cc881bdeb49226a064116e5",
  "translation_date": "2025-09-03T21:58:59+00:00",
  "source_file": "8-Reinforcement/1-QLearning/README.md",
  "language_code": "de"
}
-->
# Einführung in Reinforcement Learning und Q-Learning

![Zusammenfassung von Reinforcement Learning in maschinellem Lernen in einer Sketchnote](../../../../translated_images/ml-reinforcement.94024374d63348dbb3571c343ca7ddabef72adac0b8086d47164b769ba3a8a1d.de.png)
> Sketchnote von [Tomomi Imura](https://www.twitter.com/girlie_mac)

Reinforcement Learning umfasst drei wichtige Konzepte: den Agenten, einige Zustände und eine Menge von Aktionen pro Zustand. Indem der Agent in einem bestimmten Zustand eine Aktion ausführt, erhält er eine Belohnung. Stellen Sie sich erneut das Computerspiel Super Mario vor. Sie sind Mario, befinden sich in einem Level und stehen am Rand einer Klippe. Über Ihnen ist eine Münze. Sie, als Mario, in einem Level, an einer bestimmten Position ... das ist Ihr Zustand. Wenn Sie einen Schritt nach rechts machen (eine Aktion), fallen Sie über die Klippe, was Ihnen eine niedrige Punktzahl einbringt. Wenn Sie jedoch die Sprungtaste drücken, erzielen Sie einen Punkt und bleiben am Leben. Das ist ein positives Ergebnis und sollte Ihnen eine positive Punktzahl einbringen.

Mit Hilfe von Reinforcement Learning und einem Simulator (dem Spiel) können Sie lernen, wie Sie das Spiel spielen, um die Belohnung zu maximieren, also am Leben zu bleiben und so viele Punkte wie möglich zu sammeln.

[![Einführung in Reinforcement Learning](https://img.youtube.com/vi/lDq_en8RNOo/0.jpg)](https://www.youtube.com/watch?v=lDq_en8RNOo)

> 🎥 Klicken Sie auf das Bild oben, um Dmitry über Reinforcement Learning sprechen zu hören.

## [Quiz vor der Vorlesung](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/45/)

## Voraussetzungen und Einrichtung

In dieser Lektion werden wir mit etwas Code in Python experimentieren. Sie sollten in der Lage sein, den Jupyter Notebook-Code aus dieser Lektion entweder auf Ihrem Computer oder in der Cloud auszuführen.

Sie können [das Notebook der Lektion](https://github.com/microsoft/ML-For-Beginners/blob/main/8-Reinforcement/1-QLearning/notebook.ipynb) öffnen und diese Lektion durchgehen, um es zu erstellen.

> **Hinweis:** Wenn Sie diesen Code aus der Cloud öffnen, müssen Sie auch die Datei [`rlboard.py`](https://github.com/microsoft/ML-For-Beginners/blob/main/8-Reinforcement/1-QLearning/rlboard.py) abrufen, die im Notebook-Code verwendet wird. Fügen Sie sie in dasselbe Verzeichnis wie das Notebook ein.

## Einführung

In dieser Lektion erkunden wir die Welt von **[Peter und der Wolf](https://de.wikipedia.org/wiki/Peter_und_der_Wolf)**, inspiriert von einem musikalischen Märchen des russischen Komponisten [Sergei Prokofjew](https://de.wikipedia.org/wiki/Sergei_Prokofjew). Wir verwenden **Reinforcement Learning**, um Peter seine Umgebung erkunden zu lassen, leckere Äpfel zu sammeln und dem Wolf auszuweichen.

**Reinforcement Learning** (RL) ist eine Lerntechnik, die es uns ermöglicht, ein optimales Verhalten eines **Agenten** in einer **Umgebung** durch viele Experimente zu erlernen. Ein Agent in dieser Umgebung sollte ein **Ziel** haben, das durch eine **Belohnungsfunktion** definiert ist.

## Die Umgebung

Der Einfachheit halber betrachten wir Peters Welt als ein quadratisches Spielfeld der Größe `width` x `height`, wie dieses:

![Peters Umgebung](../../../../translated_images/environment.40ba3cb66256c93fa7e92f6f7214e1d1f588aafa97d266c11d108c5c5d101b6c.de.png)

Jede Zelle auf diesem Spielfeld kann entweder sein:

* **Boden**, auf dem Peter und andere Kreaturen laufen können.
* **Wasser**, auf dem man offensichtlich nicht laufen kann.
* ein **Baum** oder **Gras**, ein Ort, an dem man sich ausruhen kann.
* ein **Apfel**, den Peter gerne finden würde, um sich zu ernähren.
* ein **Wolf**, der gefährlich ist und dem man ausweichen sollte.

Es gibt ein separates Python-Modul, [`rlboard.py`](https://github.com/microsoft/ML-For-Beginners/blob/main/8-Reinforcement/1-QLearning/rlboard.py), das den Code enthält, um mit dieser Umgebung zu arbeiten. Da dieser Code nicht wichtig für das Verständnis unserer Konzepte ist, importieren wir das Modul und verwenden es, um das Beispielbrett zu erstellen (Codeblock 1):

```python
from rlboard import *

width, height = 8,8
m = Board(width,height)
m.randomize(seed=13)
m.plot()
```

Dieser Code sollte ein Bild der Umgebung ähnlich dem oben gezeigten ausgeben.

## Aktionen und Strategie

In unserem Beispiel besteht Peters Ziel darin, einen Apfel zu finden, während er dem Wolf und anderen Hindernissen ausweicht. Dazu kann er im Wesentlichen umherlaufen, bis er einen Apfel findet.

An jeder Position kann er zwischen den folgenden Aktionen wählen: hoch, runter, links und rechts.

Wir definieren diese Aktionen als ein Dictionary und ordnen sie Paaren von entsprechenden Koordinatenänderungen zu. Zum Beispiel würde die Bewegung nach rechts (`R`) einem Paar `(1,0)` entsprechen. (Codeblock 2):

```python
actions = { "U" : (0,-1), "D" : (0,1), "L" : (-1,0), "R" : (1,0) }
action_idx = { a : i for i,a in enumerate(actions.keys()) }
```

Zusammengefasst sind die Strategie und das Ziel dieses Szenarios wie folgt:

- **Die Strategie** unseres Agenten (Peter) wird durch eine sogenannte **Policy** definiert. Eine Policy ist eine Funktion, die die Aktion in einem gegebenen Zustand zurückgibt. In unserem Fall wird der Zustand des Problems durch das Spielfeld einschließlich der aktuellen Position des Spielers dargestellt.

- **Das Ziel** des Reinforcement Learnings ist es, letztendlich eine gute Policy zu erlernen, die es uns ermöglicht, das Problem effizient zu lösen. Als Ausgangspunkt betrachten wir jedoch die einfachste Policy, die als **Random Walk** bezeichnet wird.

## Random Walk

Lassen Sie uns zunächst unser Problem lösen, indem wir eine Random-Walk-Strategie implementieren. Beim Random Walk wählen wir zufällig die nächste Aktion aus den erlaubten Aktionen, bis wir den Apfel erreichen (Codeblock 3).

1. Implementieren Sie den Random Walk mit dem folgenden Code:

    ```python
    def random_policy(m):
        return random.choice(list(actions))
    
    def walk(m,policy,start_position=None):
        n = 0 # number of steps
        # set initial position
        if start_position:
            m.human = start_position 
        else:
            m.random_start()
        while True:
            if m.at() == Board.Cell.apple:
                return n # success!
            if m.at() in [Board.Cell.wolf, Board.Cell.water]:
                return -1 # eaten by wolf or drowned
            while True:
                a = actions[policy(m)]
                new_pos = m.move_pos(m.human,a)
                if m.is_valid(new_pos) and m.at(new_pos)!=Board.Cell.water:
                    m.move(a) # do the actual move
                    break
            n+=1
    
    walk(m,random_policy)
    ```

    Der Aufruf von `walk` sollte die Länge des entsprechenden Pfades zurückgeben, die von einem Lauf zum anderen variieren kann.

1. Führen Sie das Walk-Experiment mehrmals durch (z. B. 100 Mal) und geben Sie die resultierenden Statistiken aus (Codeblock 4):

    ```python
    def print_statistics(policy):
        s,w,n = 0,0,0
        for _ in range(100):
            z = walk(m,policy)
            if z<0:
                w+=1
            else:
                s += z
                n += 1
        print(f"Average path length = {s/n}, eaten by wolf: {w} times")
    
    print_statistics(random_policy)
    ```

    Beachten Sie, dass die durchschnittliche Länge eines Pfades etwa 30-40 Schritte beträgt, was ziemlich viel ist, wenn man bedenkt, dass die durchschnittliche Entfernung zum nächsten Apfel etwa 5-6 Schritte beträgt.

    Sie können auch sehen, wie Peters Bewegung während des Random Walk aussieht:

    ![Peters Random Walk](../../../../8-Reinforcement/1-QLearning/images/random_walk.gif)

## Belohnungsfunktion

Um unsere Policy intelligenter zu machen, müssen wir verstehen, welche Züge "besser" sind als andere. Dazu müssen wir unser Ziel definieren.

Das Ziel kann in Form einer **Belohnungsfunktion** definiert werden, die für jeden Zustand einen Punktwert zurückgibt. Je höher die Zahl, desto besser die Belohnungsfunktion. (Codeblock 5)

```python
move_reward = -0.1
goal_reward = 10
end_reward = -10

def reward(m,pos=None):
    pos = pos or m.human
    if not m.is_valid(pos):
        return end_reward
    x = m.at(pos)
    if x==Board.Cell.water or x == Board.Cell.wolf:
        return end_reward
    if x==Board.Cell.apple:
        return goal_reward
    return move_reward
```

Eine interessante Sache an Belohnungsfunktionen ist, dass in den meisten Fällen *wir nur am Ende des Spiels eine wesentliche Belohnung erhalten*. Das bedeutet, dass unser Algorithmus irgendwie "gute" Schritte, die zu einer positiven Belohnung am Ende führen, speichern und deren Bedeutung erhöhen sollte. Ebenso sollten alle Züge, die zu schlechten Ergebnissen führen, entmutigt werden.

## Q-Learning

Ein Algorithmus, den wir hier besprechen werden, heißt **Q-Learning**. In diesem Algorithmus wird die Policy durch eine Funktion (oder eine Datenstruktur) namens **Q-Tabelle** definiert. Sie zeichnet die "Qualität" jeder Aktion in einem gegebenen Zustand auf.

Es wird als Q-Tabelle bezeichnet, weil es oft praktisch ist, sie als Tabelle oder mehrdimensionales Array darzustellen. Da unser Spielfeld die Dimensionen `width` x `height` hat, können wir die Q-Tabelle mit einem numpy-Array der Form `width` x `height` x `len(actions)` darstellen: (Codeblock 6)

```python
Q = np.ones((width,height,len(actions)),dtype=np.float)*1.0/len(actions)
```

Beachten Sie, dass wir alle Werte der Q-Tabelle mit einem gleichen Wert initialisieren, in unserem Fall - 0.25. Dies entspricht der "Random Walk"-Policy, da alle Züge in jedem Zustand gleich gut sind. Wir können die Q-Tabelle an die `plot`-Funktion übergeben, um die Tabelle auf dem Spielfeld zu visualisieren: `m.plot(Q)`.

![Peters Umgebung](../../../../translated_images/env_init.04e8f26d2d60089e128f21d22e5fef57d580e559f0d5937b06c689e5e7cdd438.de.png)

In der Mitte jeder Zelle befindet sich ein "Pfeil", der die bevorzugte Bewegungsrichtung anzeigt. Da alle Richtungen gleich sind, wird ein Punkt angezeigt.

Jetzt müssen wir die Simulation ausführen, unsere Umgebung erkunden und eine bessere Verteilung der Q-Tabelle-Werte erlernen, die es uns ermöglicht, den Weg zum Apfel viel schneller zu finden.

## Essenz des Q-Learnings: Bellman-Gleichung

Sobald wir uns bewegen, hat jede Aktion eine entsprechende Belohnung, d. h. wir könnten theoretisch die nächste Aktion basierend auf der höchsten unmittelbaren Belohnung auswählen. In den meisten Zuständen wird der Zug jedoch nicht unser Ziel erreichen, den Apfel zu finden, und wir können daher nicht sofort entscheiden, welche Richtung besser ist.

> Denken Sie daran, dass nicht das unmittelbare Ergebnis zählt, sondern das Endergebnis, das wir am Ende der Simulation erhalten.

Um diese verzögerte Belohnung zu berücksichtigen, müssen wir die Prinzipien der **[dynamischen Programmierung](https://de.wikipedia.org/wiki/Dynamische_Programmierung)** verwenden, die es uns ermöglichen, unser Problem rekursiv zu betrachten.

Angenommen, wir befinden uns jetzt im Zustand *s* und möchten zum nächsten Zustand *s'* wechseln. Indem wir dies tun, erhalten wir die unmittelbare Belohnung *r(s,a)*, definiert durch die Belohnungsfunktion, plus eine zukünftige Belohnung. Wenn wir annehmen, dass unsere Q-Tabelle die "Attraktivität" jeder Aktion korrekt widerspiegelt, wählen wir im Zustand *s'* eine Aktion *a*, die dem maximalen Wert von *Q(s',a')* entspricht. Somit wird die bestmögliche zukünftige Belohnung, die wir im Zustand *s* erhalten könnten, durch `max`

## Überprüfung der Richtlinie

Da die Q-Tabelle die "Attraktivität" jeder Aktion in jedem Zustand auflistet, ist es recht einfach, sie zu nutzen, um eine effiziente Navigation in unserer Welt zu definieren. Im einfachsten Fall können wir die Aktion auswählen, die dem höchsten Wert in der Q-Tabelle entspricht: (Codeblock 9)

```python
def qpolicy_strict(m):
        x,y = m.human
        v = probs(Q[x,y])
        a = list(actions)[np.argmax(v)]
        return a

walk(m,qpolicy_strict)
```

> Wenn Sie den obigen Code mehrmals ausführen, werden Sie möglicherweise feststellen, dass er manchmal "hängt" und Sie die STOP-Taste im Notebook drücken müssen, um ihn zu unterbrechen. Dies geschieht, weil es Situationen geben kann, in denen zwei Zustände sich gegenseitig in Bezug auf den optimalen Q-Wert "zeigen", wodurch der Agent zwischen diesen Zuständen endlos hin- und herwechselt.

## 🚀Herausforderung

> **Aufgabe 1:** Ändern Sie die `walk`-Funktion so, dass die maximale Pfadlänge auf eine bestimmte Anzahl von Schritten (z. B. 100) begrenzt wird, und beobachten Sie, wie der obige Code diesen Wert von Zeit zu Zeit zurückgibt.

> **Aufgabe 2:** Ändern Sie die `walk`-Funktion so, dass sie nicht an Orte zurückkehrt, an denen sie bereits zuvor war. Dies verhindert, dass `walk` in einer Schleife hängen bleibt, allerdings kann der Agent dennoch in einer Position "gefangen" sein, aus der er nicht entkommen kann.

## Navigation

Eine bessere Navigationsstrategie wäre diejenige, die wir während des Trainings verwendet haben, die eine Kombination aus Ausnutzung und Erkundung darstellt. Bei dieser Strategie wählen wir jede Aktion mit einer bestimmten Wahrscheinlichkeit aus, die proportional zu den Werten in der Q-Tabelle ist. Diese Strategie kann zwar immer noch dazu führen, dass der Agent zu einer bereits erkundeten Position zurückkehrt, aber wie Sie im folgenden Code sehen können, führt sie zu einem sehr kurzen durchschnittlichen Pfad zur gewünschten Position (denken Sie daran, dass `print_statistics` die Simulation 100 Mal ausführt): (Codeblock 10)

```python
def qpolicy(m):
        x,y = m.human
        v = probs(Q[x,y])
        a = random.choices(list(actions),weights=v)[0]
        return a

print_statistics(qpolicy)
```

Nach dem Ausführen dieses Codes sollten Sie eine deutlich kürzere durchschnittliche Pfadlänge als zuvor erhalten, im Bereich von 3-6.

## Untersuchung des Lernprozesses

Wie bereits erwähnt, ist der Lernprozess ein Gleichgewicht zwischen der Erkundung und der Nutzung des gewonnenen Wissens über die Struktur des Problemraums. Wir haben gesehen, dass die Ergebnisse des Lernens (die Fähigkeit, einem Agenten zu helfen, einen kurzen Weg zum Ziel zu finden) sich verbessert haben, aber es ist auch interessant zu beobachten, wie sich die durchschnittliche Pfadlänge während des Lernprozesses verhält:

Die Erkenntnisse lassen sich wie folgt zusammenfassen:

- **Durchschnittliche Pfadlänge nimmt zu**. Was wir hier sehen, ist, dass die durchschnittliche Pfadlänge zunächst zunimmt. Dies liegt wahrscheinlich daran, dass wir, wenn wir nichts über die Umgebung wissen, dazu neigen, in schlechten Zuständen, wie Wasser oder bei einem Wolf, stecken zu bleiben. Wenn wir mehr lernen und dieses Wissen nutzen, können wir die Umgebung länger erkunden, wissen aber immer noch nicht genau, wo sich die Äpfel befinden.

- **Pfadlänge nimmt ab, je mehr wir lernen**. Sobald wir genug gelernt haben, wird es für den Agenten einfacher, das Ziel zu erreichen, und die Pfadlänge beginnt abzunehmen. Allerdings sind wir weiterhin offen für Erkundungen, sodass wir oft vom besten Weg abweichen und neue Optionen ausprobieren, was den Pfad länger als optimal macht.

- **Länge nimmt abrupt zu**. Was wir auf diesem Diagramm ebenfalls beobachten, ist, dass die Länge an einem Punkt abrupt zunimmt. Dies deutet auf die stochastische Natur des Prozesses hin und darauf, dass wir die Q-Tabellen-Koeffizienten durch das Überschreiben mit neuen Werten "verschlechtern" können. Dies sollte idealerweise minimiert werden, indem die Lernrate verringert wird (zum Beispiel passen wir gegen Ende des Trainings die Q-Tabellen-Werte nur noch geringfügig an).

Insgesamt ist es wichtig zu beachten, dass der Erfolg und die Qualität des Lernprozesses stark von Parametern wie der Lernrate, dem Abfall der Lernrate und dem Diskontierungsfaktor abhängen. Diese werden oft als **Hyperparameter** bezeichnet, um sie von **Parametern** zu unterscheiden, die wir während des Trainings optimieren (zum Beispiel die Q-Tabellen-Koeffizienten). Der Prozess, die besten Werte für die Hyperparameter zu finden, wird als **Hyperparameter-Optimierung** bezeichnet und verdient ein eigenes Thema.

## [Quiz nach der Vorlesung](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/46/)

## Aufgabe 
[Eine realistischere Welt](assignment.md)

---

**Haftungsausschluss**:  
Dieses Dokument wurde mit dem KI-Übersetzungsdienst [Co-op Translator](https://github.com/Azure/co-op-translator) übersetzt. Obwohl wir uns um Genauigkeit bemühen, beachten Sie bitte, dass automatisierte Übersetzungen Fehler oder Ungenauigkeiten enthalten können. Das Originaldokument in seiner ursprünglichen Sprache sollte als maßgebliche Quelle betrachtet werden. Für kritische Informationen wird eine professionelle menschliche Übersetzung empfohlen. Wir übernehmen keine Haftung für Missverständnisse oder Fehlinterpretationen, die sich aus der Nutzung dieser Übersetzung ergeben.