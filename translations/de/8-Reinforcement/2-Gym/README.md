<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "107d5bb29da8a562e7ae72262d251a75",
  "translation_date": "2025-09-04T22:05:24+00:00",
  "source_file": "8-Reinforcement/2-Gym/README.md",
  "language_code": "de"
}
-->
## Voraussetzungen

In dieser Lektion verwenden wir eine Bibliothek namens **OpenAI Gym**, um verschiedene **Umgebungen** zu simulieren. Du kannst den Code dieser Lektion lokal ausführen (z. B. in Visual Studio Code), wobei die Simulation in einem neuen Fenster geöffnet wird. Wenn du den Code online ausführst, musst du möglicherweise einige Anpassungen vornehmen, wie [hier](https://towardsdatascience.com/rendering-openai-gym-envs-on-binder-and-google-colab-536f99391cc7) beschrieben.

## OpenAI Gym

In der vorherigen Lektion wurden die Spielregeln und der Zustand durch die `Board`-Klasse definiert, die wir selbst erstellt haben. Hier verwenden wir eine spezielle **Simulationsumgebung**, die die Physik hinter der balancierenden Stange simuliert. Eine der beliebtesten Simulationsumgebungen für das Training von Reinforcement-Learning-Algorithmen ist das [Gym](https://gym.openai.com/), das von [OpenAI](https://openai.com/) gepflegt wird. Mit diesem Gym können wir verschiedene **Umgebungen** erstellen, von einer CartPole-Simulation bis hin zu Atari-Spielen.

> **Hinweis**: Weitere Umgebungen von OpenAI Gym findest du [hier](https://gym.openai.com/envs/#classic_control).

Zuerst installieren wir das Gym und importieren die benötigten Bibliotheken (Codeblock 1):

```python
import sys
!{sys.executable} -m pip install gym 

import gym
import matplotlib.pyplot as plt
import numpy as np
import random
```

## Übung - Initialisiere eine CartPole-Umgebung

Um mit dem CartPole-Balancierproblem zu arbeiten, müssen wir die entsprechende Umgebung initialisieren. Jede Umgebung ist mit einem:

- **Beobachtungsraum** verbunden, der die Struktur der Informationen definiert, die wir von der Umgebung erhalten. Beim CartPole-Problem erhalten wir die Position der Stange, die Geschwindigkeit und einige andere Werte.

- **Aktionsraum**, der mögliche Aktionen definiert. In unserem Fall ist der Aktionsraum diskret und besteht aus zwei Aktionen - **links** und **rechts**. (Codeblock 2)

1. Um zu initialisieren, gib den folgenden Code ein:

    ```python
    env = gym.make("CartPole-v1")
    print(env.action_space)
    print(env.observation_space)
    print(env.action_space.sample())
    ```

Um zu sehen, wie die Umgebung funktioniert, führen wir eine kurze Simulation für 100 Schritte durch. Bei jedem Schritt geben wir eine der Aktionen vor - in dieser Simulation wählen wir zufällig eine Aktion aus dem `action_space`.

1. Führe den folgenden Code aus und sieh dir an, was passiert.

    ✅ Denke daran, dass es bevorzugt wird, diesen Code in einer lokalen Python-Installation auszuführen! (Codeblock 3)

    ```python
    env.reset()
    
    for i in range(100):
       env.render()
       env.step(env.action_space.sample())
    env.close()
    ```

    Du solltest etwas Ähnliches wie dieses Bild sehen:

    ![nicht balancierendes CartPole](../../../../8-Reinforcement/2-Gym/images/cartpole-nobalance.gif)

1. Während der Simulation müssen wir Beobachtungen erhalten, um zu entscheiden, wie wir handeln sollen. Tatsächlich gibt die `step`-Funktion aktuelle Beobachtungen, eine Belohnungsfunktion und ein `done`-Flag zurück, das anzeigt, ob es sinnvoll ist, die Simulation fortzusetzen oder nicht: (Codeblock 4)

    ```python
    env.reset()
    
    done = False
    while not done:
       env.render()
       obs, rew, done, info = env.step(env.action_space.sample())
       print(f"{obs} -> {rew}")
    env.close()
    ```

    Du wirst etwas Ähnliches wie dies in der Notebook-Ausgabe sehen:

    ```text
    [ 0.03403272 -0.24301182  0.02669811  0.2895829 ] -> 1.0
    [ 0.02917248 -0.04828055  0.03248977  0.00543839] -> 1.0
    [ 0.02820687  0.14636075  0.03259854 -0.27681916] -> 1.0
    [ 0.03113408  0.34100283  0.02706215 -0.55904489] -> 1.0
    [ 0.03795414  0.53573468  0.01588125 -0.84308041] -> 1.0
    ...
    [ 0.17299878  0.15868546 -0.20754175 -0.55975453] -> 1.0
    [ 0.17617249  0.35602306 -0.21873684 -0.90998894] -> 1.0
    ```

    Der Beobachtungsvektor, der bei jedem Schritt der Simulation zurückgegeben wird, enthält die folgenden Werte:
    - Position des Wagens
    - Geschwindigkeit des Wagens
    - Winkel der Stange
    - Rotationsrate der Stange

1. Ermittle den minimalen und maximalen Wert dieser Zahlen: (Codeblock 5)

    ```python
    print(env.observation_space.low)
    print(env.observation_space.high)
    ```

    Du wirst auch feststellen, dass der Belohnungswert bei jedem Simulationsschritt immer 1 ist. Das liegt daran, dass unser Ziel darin besteht, so lange wie möglich zu überleben, d. h. die Stange so lange wie möglich in einer einigermaßen vertikalen Position zu halten.

    ✅ Tatsächlich gilt die CartPole-Simulation als gelöst, wenn wir es schaffen, eine durchschnittliche Belohnung von 195 über 100 aufeinanderfolgende Versuche zu erreichen.

## Zustandsdiskretisierung

Beim Q-Learning müssen wir eine Q-Tabelle erstellen, die definiert, was in jedem Zustand zu tun ist. Um dies tun zu können, muss der Zustand **diskret** sein, genauer gesagt, er sollte eine endliche Anzahl diskreter Werte enthalten. Daher müssen wir unsere Beobachtungen irgendwie **diskretisieren**, indem wir sie auf eine endliche Menge von Zuständen abbilden.

Es gibt einige Möglichkeiten, dies zu tun:

- **In Bins aufteilen**. Wenn wir das Intervall eines bestimmten Wertes kennen, können wir dieses Intervall in eine Anzahl von **Bins** aufteilen und dann den Wert durch die Bin-Nummer ersetzen, zu der er gehört. Dies kann mit der numpy-Methode [`digitize`](https://numpy.org/doc/stable/reference/generated/numpy.digitize.html) durchgeführt werden. In diesem Fall kennen wir die Zustandsgröße genau, da sie von der Anzahl der Bins abhängt, die wir für die Digitalisierung auswählen.

✅ Wir können lineare Interpolation verwenden, um Werte auf ein endliches Intervall (z. B. von -20 bis 20) zu bringen, und dann Zahlen durch Runden in ganze Zahlen umwandeln. Dies gibt uns etwas weniger Kontrolle über die Größe des Zustands, insbesondere wenn wir die genauen Bereiche der Eingabewerte nicht kennen. Zum Beispiel haben in unserem Fall 2 von 4 Werten keine oberen/unteren Grenzen, was zu einer unendlichen Anzahl von Zuständen führen kann.

In unserem Beispiel verwenden wir den zweiten Ansatz. Wie du später feststellen wirst, nehmen diese Werte trotz undefinierter oberer/unterer Grenzen selten Werte außerhalb bestimmter endlicher Intervalle an, sodass Zustände mit extremen Werten sehr selten sein werden.

1. Hier ist die Funktion, die die Beobachtung aus unserem Modell nimmt und ein Tupel aus 4 ganzzahligen Werten erzeugt: (Codeblock 6)

    ```python
    def discretize(x):
        return tuple((x/np.array([0.25, 0.25, 0.01, 0.1])).astype(np.int))
    ```

1. Lass uns auch eine andere Diskretisierungsmethode mit Bins erkunden: (Codeblock 7)

    ```python
    def create_bins(i,num):
        return np.arange(num+1)*(i[1]-i[0])/num+i[0]
    
    print("Sample bins for interval (-5,5) with 10 bins\n",create_bins((-5,5),10))
    
    ints = [(-5,5),(-2,2),(-0.5,0.5),(-2,2)] # intervals of values for each parameter
    nbins = [20,20,10,10] # number of bins for each parameter
    bins = [create_bins(ints[i],nbins[i]) for i in range(4)]
    
    def discretize_bins(x):
        return tuple(np.digitize(x[i],bins[i]) for i in range(4))
    ```

1. Lass uns nun eine kurze Simulation durchführen und diese diskreten Umgebungswerte beobachten. Probiere gerne sowohl `discretize` als auch `discretize_bins` aus und sieh, ob es einen Unterschied gibt.

    ✅ `discretize_bins` gibt die Bin-Nummer zurück, die bei 0 beginnt. Für Eingabewerte um 0 gibt es daher die Zahl aus der Mitte des Intervalls (10) zurück. Bei `discretize` haben wir uns nicht um den Bereich der Ausgabewerte gekümmert, sodass sie negativ sein können, und 0 entspricht 0. (Codeblock 8)

    ```python
    env.reset()
    
    done = False
    while not done:
       #env.render()
       obs, rew, done, info = env.step(env.action_space.sample())
       #print(discretize_bins(obs))
       print(discretize(obs))
    env.close()
    ```

    ✅ Kommentiere die Zeile, die mit `env.render` beginnt, aus, wenn du sehen möchtest, wie die Umgebung ausgeführt wird. Andernfalls kannst du sie im Hintergrund ausführen, was schneller ist. Wir werden diese "unsichtbare" Ausführung während unseres Q-Learning-Prozesses verwenden.

## Die Struktur der Q-Tabelle

In unserer vorherigen Lektion war der Zustand ein einfaches Zahlenpaar von 0 bis 8, und daher war es praktisch, die Q-Tabelle durch einen numpy-Tensor mit einer Form von 8x8x2 darzustellen. Wenn wir die Bins-Diskretisierung verwenden, ist die Größe unseres Zustandsvektors ebenfalls bekannt, sodass wir denselben Ansatz verwenden und den Zustand durch ein Array der Form 20x20x10x10x2 darstellen können (hier ist 2 die Dimension des Aktionsraums, und die ersten Dimensionen entsprechen der Anzahl der Bins, die wir für jeden der Parameter im Beobachtungsraum ausgewählt haben).

Manchmal sind jedoch die genauen Dimensionen des Beobachtungsraums nicht bekannt. Im Fall der `discretize`-Funktion können wir uns nie sicher sein, dass unser Zustand innerhalb bestimmter Grenzen bleibt, da einige der ursprünglichen Werte nicht begrenzt sind. Daher verwenden wir einen etwas anderen Ansatz und stellen die Q-Tabelle durch ein Wörterbuch dar.

1. Verwende das Paar *(state, action)* als Schlüssel des Wörterbuchs, und der Wert würde dem Eintrag der Q-Tabelle entsprechen. (Codeblock 9)

    ```python
    Q = {}
    actions = (0,1)
    
    def qvalues(state):
        return [Q.get((state,a),0) for a in actions]
    ```

    Hier definieren wir auch eine Funktion `qvalues()`, die eine Liste von Q-Tabellenwerten für einen gegebenen Zustand zurückgibt, die allen möglichen Aktionen entspricht. Wenn der Eintrag nicht in der Q-Tabelle vorhanden ist, geben wir standardmäßig 0 zurück.

## Lass uns mit Q-Learning beginnen

Jetzt sind wir bereit, Peter das Balancieren beizubringen!

1. Zuerst setzen wir einige Hyperparameter: (Codeblock 10)

    ```python
    # hyperparameters
    alpha = 0.3
    gamma = 0.9
    epsilon = 0.90
    ```

    Hier ist `alpha` die **Lernrate**, die definiert, in welchem Ausmaß wir die aktuellen Werte der Q-Tabelle bei jedem Schritt anpassen sollten. In der vorherigen Lektion haben wir mit 1 begonnen und dann `alpha` während des Trainings auf niedrigere Werte reduziert. In diesem Beispiel halten wir es der Einfachheit halber konstant, und du kannst später mit der Anpassung der `alpha`-Werte experimentieren.

    `gamma` ist der **Abzinsungsfaktor**, der zeigt, in welchem Ausmaß wir zukünftige Belohnungen gegenüber aktuellen Belohnungen priorisieren sollten.

    `epsilon` ist der **Explorations-/Exploiterungsfaktor**, der bestimmt, ob wir Exploration der Exploitation vorziehen sollten oder umgekehrt. In unserem Algorithmus wählen wir in `epsilon` Prozent der Fälle die nächste Aktion entsprechend den Q-Tabellenwerten aus, und in den verbleibenden Fällen führen wir eine zufällige Aktion aus. Dies ermöglicht es uns, Bereiche des Suchraums zu erkunden, die wir noch nie zuvor gesehen haben.

    ✅ In Bezug auf das Balancieren - das Auswählen einer zufälligen Aktion (Exploration) würde wie ein zufälliger Stoß in die falsche Richtung wirken, und die Stange müsste lernen, wie sie das Gleichgewicht aus diesen "Fehlern" wiederherstellt.

### Den Algorithmus verbessern

Wir können auch zwei Verbesserungen an unserem Algorithmus aus der vorherigen Lektion vornehmen:

- **Durchschnittliche kumulative Belohnung berechnen**, über eine Anzahl von Simulationen. Wir drucken den Fortschritt alle 5000 Iterationen aus und mitteln unsere kumulative Belohnung über diesen Zeitraum. Das bedeutet, dass wir, wenn wir mehr als 195 Punkte erreichen, das Problem als gelöst betrachten können, und zwar mit einer noch höheren Qualität als erforderlich.

- **Maximales durchschnittliches kumulatives Ergebnis berechnen**, `Qmax`, und wir speichern die Q-Tabelle, die diesem Ergebnis entspricht. Wenn du das Training ausführst, wirst du feststellen, dass manchmal das durchschnittliche kumulative Ergebnis zu sinken beginnt, und wir möchten die Werte der Q-Tabelle beibehalten, die dem besten während des Trainings beobachteten Modell entsprechen.

1. Sammle alle kumulativen Belohnungen bei jeder Simulation im `rewards`-Vektor für eine spätere Darstellung. (Codeblock 11)

    ```python
    def probs(v,eps=1e-4):
        v = v-v.min()+eps
        v = v/v.sum()
        return v
    
    Qmax = 0
    cum_rewards = []
    rewards = []
    for epoch in range(100000):
        obs = env.reset()
        done = False
        cum_reward=0
        # == do the simulation ==
        while not done:
            s = discretize(obs)
            if random.random()<epsilon:
                # exploitation - chose the action according to Q-Table probabilities
                v = probs(np.array(qvalues(s)))
                a = random.choices(actions,weights=v)[0]
            else:
                # exploration - randomly chose the action
                a = np.random.randint(env.action_space.n)
    
            obs, rew, done, info = env.step(a)
            cum_reward+=rew
            ns = discretize(obs)
            Q[(s,a)] = (1 - alpha) * Q.get((s,a),0) + alpha * (rew + gamma * max(qvalues(ns)))
        cum_rewards.append(cum_reward)
        rewards.append(cum_reward)
        # == Periodically print results and calculate average reward ==
        if epoch%5000==0:
            print(f"{epoch}: {np.average(cum_rewards)}, alpha={alpha}, epsilon={epsilon}")
            if np.average(cum_rewards) > Qmax:
                Qmax = np.average(cum_rewards)
                Qbest = Q
            cum_rewards=[]
    ```

Was du aus diesen Ergebnissen feststellen kannst:

- **Nahe an unserem Ziel**. Wir sind dem Ziel, 195 kumulative Belohnungen über 100+ aufeinanderfolgende Simulationen zu erreichen, sehr nahe oder haben es möglicherweise sogar erreicht! Selbst wenn wir kleinere Zahlen erhalten, wissen wir es nicht genau, da wir über 5000 Läufe mitteln, und nur 100 Läufe sind im formalen Kriterium erforderlich.

- **Belohnung beginnt zu sinken**. Manchmal beginnt die Belohnung zu sinken, was bedeutet, dass wir bereits gelernte Werte in der Q-Tabelle durch solche ersetzen können, die die Situation verschlechtern.

Diese Beobachtung wird deutlicher, wenn wir den Trainingsfortschritt grafisch darstellen.

## Trainingsfortschritt darstellen

Während des Trainings haben wir den kumulativen Belohnungswert bei jeder Iteration im `rewards`-Vektor gesammelt. So sieht es aus, wenn wir es gegen die Iterationsnummer plotten:

```python
plt.plot(rewards)
```

![roher Fortschritt](../../../../8-Reinforcement/2-Gym/images/train_progress_raw.png)

Aus diesem Diagramm lässt sich nichts ableiten, da aufgrund der stochastischen Natur des Trainingsprozesses die Länge der Trainingseinheiten stark variiert. Um dieses Diagramm sinnvoller zu machen, können wir den **gleitenden Durchschnitt** über eine Reihe von Experimenten berechnen, sagen wir 100. Dies kann bequem mit `np.convolve` durchgeführt werden: (Codeblock 12)

```python
def running_average(x,window):
    return np.convolve(x,np.ones(window)/window,mode='valid')

plt.plot(running_average(rewards,100))
```

![Trainingsfortschritt](../../../../8-Reinforcement/2-Gym/images/train_progress_runav.png)

## Hyperparameter variieren

Um das Lernen stabiler zu machen, macht es Sinn, einige unserer Hyperparameter während des Trainings anzupassen. Insbesondere:

- **Für die Lernrate**, `alpha`, können wir mit Werten nahe 1 beginnen und dann den Parameter schrittweise verringern. Mit der Zeit erhalten wir gute Wahrscheinlichkeitswerte in der Q-Tabelle, und daher sollten wir sie nur leicht anpassen und nicht vollständig mit neuen Werten überschreiben.

- **Epsilon erhöhen**. Wir könnten `epsilon` langsam erhöhen, um weniger zu explorieren und mehr zu exploiten. Es macht wahrscheinlich Sinn, mit einem niedrigeren Wert von `epsilon` zu beginnen und ihn auf fast 1 zu steigern.
> **Aufgabe 1**: Experimentiere mit den Hyperparametern und überprüfe, ob du eine höhere kumulative Belohnung erzielen kannst. Erreichst du mehr als 195?
> **Aufgabe 2**: Um das Problem formal zu lösen, musst du einen durchschnittlichen Reward von 195 über 100 aufeinanderfolgende Durchläufe erreichen. Messe dies während des Trainings und stelle sicher, dass du das Problem formal gelöst hast!

## Das Ergebnis in Aktion sehen

Es wäre interessant zu sehen, wie sich das trainierte Modell tatsächlich verhält. Lass uns die Simulation ausführen und dieselbe Aktionsauswahlstrategie wie während des Trainings anwenden, indem wir entsprechend der Wahrscheinlichkeitsverteilung in der Q-Tabelle sampeln: (Codeblock 13)

```python
obs = env.reset()
done = False
while not done:
   s = discretize(obs)
   env.render()
   v = probs(np.array(qvalues(s)))
   a = random.choices(actions,weights=v)[0]
   obs,_,done,_ = env.step(a)
env.close()
```

Du solltest etwas Ähnliches sehen wie hier:

![ein balancierender Cartpole](../../../../8-Reinforcement/2-Gym/images/cartpole-balance.gif)

---

## 🚀Herausforderung

> **Aufgabe 3**: Hier haben wir die finale Version der Q-Tabelle verwendet, die möglicherweise nicht die beste ist. Denke daran, dass wir die am besten performende Q-Tabelle in der Variable `Qbest` gespeichert haben! Probiere dasselbe Beispiel mit der am besten performenden Q-Tabelle aus, indem du `Qbest` in `Q` kopierst, und schau, ob du einen Unterschied bemerkst.

> **Aufgabe 4**: Hier haben wir nicht bei jedem Schritt die beste Aktion ausgewählt, sondern stattdessen entsprechend der Wahrscheinlichkeitsverteilung gesampelt. Wäre es sinnvoller, immer die beste Aktion mit dem höchsten Q-Tabelle-Wert auszuwählen? Das kann mit der Funktion `np.argmax` umgesetzt werden, um die Aktionsnummer mit dem höchsten Q-Tabelle-Wert zu ermitteln. Implementiere diese Strategie und überprüfe, ob sie das Balancieren verbessert.

## [Quiz nach der Vorlesung](https://ff-quizzes.netlify.app/en/ml/)

## Aufgabe
[Trainiere ein Mountain Car](assignment.md)

## Fazit

Wir haben nun gelernt, wie man Agenten trainiert, um gute Ergebnisse zu erzielen, indem man ihnen lediglich eine Reward-Funktion bereitstellt, die den gewünschten Zustand des Spiels definiert, und ihnen die Möglichkeit gibt, den Suchraum intelligent zu erkunden. Wir haben den Q-Learning-Algorithmus erfolgreich in Fällen mit diskreten und kontinuierlichen Umgebungen angewendet, jedoch mit diskreten Aktionen.

Es ist auch wichtig, Situationen zu untersuchen, in denen der Aktionsraum ebenfalls kontinuierlich ist und der Beobachtungsraum viel komplexer wird, wie beispielsweise ein Bild vom Bildschirm eines Atari-Spiels. In solchen Problemen müssen oft leistungsstärkere Machine-Learning-Techniken wie neuronale Netze eingesetzt werden, um gute Ergebnisse zu erzielen. Diese fortgeschritteneren Themen sind Gegenstand unseres kommenden, weiterführenden KI-Kurses.

---

**Haftungsausschluss**:  
Dieses Dokument wurde mit dem KI-Übersetzungsdienst [Co-op Translator](https://github.com/Azure/co-op-translator) übersetzt. Obwohl wir uns um Genauigkeit bemühen, weisen wir darauf hin, dass automatisierte Übersetzungen Fehler oder Ungenauigkeiten enthalten können. Das Originaldokument in seiner ursprünglichen Sprache sollte als maßgebliche Quelle betrachtet werden. Für kritische Informationen wird eine professionelle menschliche Übersetzung empfohlen. Wir übernehmen keine Haftung für Missverständnisse oder Fehlinterpretationen, die sich aus der Nutzung dieser Übersetzung ergeben.