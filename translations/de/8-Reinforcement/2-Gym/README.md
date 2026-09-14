# CartPole Skating

Das Problem, das wir in der vorherigen Lektion gelöst haben, mag wie ein Spielzeugproblem erscheinen, das in realen Szenarien nicht wirklich anwendbar ist. Das ist jedoch nicht der Fall, da viele reale Probleme dieses Szenario teilen – darunter das Spielen von Schach oder Go. Sie sind ähnlich, weil wir auch ein Brett mit bestimmten Regeln und einen **diskreten Zustand** haben.

## [Vorlesungsvorquiz](https://ff-quizzes.netlify.app/en/ml/)

## Einführung

In dieser Lektion wenden wir die gleichen Prinzipien des Q-Learning auf ein Problem mit **kontinuierlichem Zustand** an, d.h. einem Zustand, der durch eine oder mehrere reelle Zahlen gegeben ist. Wir beschäftigen uns mit folgendem Problem:

> **Problem**: Wenn Peter vor dem Wolf fliehen will, muss er sich schneller bewegen können. Wir werden sehen, wie Peter mit Q-Learning das Skaten lernen kann, insbesondere, wie man das Gleichgewicht hält.

![Die große Flucht!](../../../../translated_images/de/escape.18862db9930337e3.webp)

> Peter und seine Freunde sind kreativ, um vor dem Wolf zu entkommen! Bild von [Jen Looper](https://twitter.com/jenlooper)

Wir verwenden eine vereinfachte Version des Balancierens, bekannt als **CartPole**-Problem. In der Cartpole-Welt haben wir einen horizontalen Schlitten, der sich nach links oder rechts bewegen kann, und das Ziel ist es, einen vertikalen Stab auf dem Schlitten zu balancieren.

<img alt="ein Cartpole" src="../../../../translated_images/de/cartpole.b5609cc0494a14f7.webp" width="200"/>

## Voraussetzungen

In dieser Lektion verwenden wir eine Bibliothek namens **OpenAI Gym**, um verschiedene **Umgebungen** zu simulieren. Du kannst den Code dieser Lektion lokal ausführen (z.B. aus Visual Studio Code), in diesem Fall öffnet sich die Simulation in einem neuen Fenster. Wenn du den Code online ausführst, sind eventuell einige Anpassungen erforderlich, wie [hier](https://towardsdatascience.com/rendering-openai-gym-envs-on-binder-and-google-colab-536f99391cc7) beschrieben.

## OpenAI Gym

In der vorherigen Lektion wurden die Spielregeln und der Zustand von der selbst definierten `Board`-Klasse angegeben. Hier verwenden wir eine spezielle **Simulationsumgebung**, die die Physik hinter dem balancierenden Stab simuliert. Eine der beliebtesten Simulationsumgebungen zum Trainieren von Reinforcement-Learning-Algorithmen heißt [Gym](https://gym.openai.com/), das von [OpenAI](https://openai.com/) gepflegt wird. Mit diesem Gym können wir verschiedene **Umgebungen** erstellen – von einer Cartpole-Simulation bis zu Atari-Spielen.

> **Hinweis**: Andere verfügbare OpenAI Gym-Umgebungen findest du [hier](https://gym.openai.com/envs/#classic_control).

Zuerst installieren wir das Gym und importieren die benötigten Bibliotheken (Codeblock 1):

```python
import sys
!{sys.executable} -m pip install gym 

import gym
import matplotlib.pyplot as plt
import numpy as np
import random
```

## Übung – Initialisiere eine Cartpole-Umgebung

Um mit dem Cartpole-Balancierproblem zu arbeiten, müssen wir die entsprechende Umgebung initialisieren. Jede Umgebung ist mit einem:

- **Beobachtungsraum** verknüpft, der die Struktur der Informationen definiert, die wir von der Umgebung erhalten. Für das Cartpole-Problem erhalten wir Position des Stabs, Geschwindigkeit und einige andere Werte.

- **Aktionsraum**, der mögliche Aktionen definiert. In unserem Fall ist der Aktionsraum diskret und besteht aus zwei Aktionen – **links** und **rechts**. (Codeblock 2)

1. Um zu initialisieren, gib den folgenden Code ein:

    ```python
    env = gym.make("CartPole-v1")
    print(env.action_space)
    print(env.observation_space)
    print(env.action_space.sample())
    ```

Um zu sehen, wie die Umgebung funktioniert, führen wir eine kurze Simulation für 100 Schritte durch. Bei jedem Schritt geben wir eine der Aktionen vor – in dieser Simulation wählen wir die Aktion zufällig aus dem `action_space` aus.

1. Führe den folgenden Code aus und schau, was passiert.

    ✅ Es ist empfehlenswert, diesen Code in einer lokalen Python-Installation auszuführen! (Codeblock 3)

    ```python
    env.reset()
    
    for i in range(100):
       env.render()
       env.step(env.action_space.sample())
    env.close()
    ```

    Du solltest etwas Ähnliches wie dieses Bild sehen:

    ![nicht balancierender Cartpole](../../../../8-Reinforcement/2-Gym/images/cartpole-nobalance.gif)

1. Während der Simulation müssen wir Beobachtungen erhalten, um zu entscheiden, wie wir handeln. Die Schrittfunktion gibt tatsächlich aktuelle Beobachtungen, eine Belohnungsfunktion und das Done-Flag zurück, das anzeigt, ob es sinnvoll ist, die Simulation fortzusetzen oder nicht: (Codeblock 4)

    ```python
    env.reset()
    
    done = False
    while not done:
       env.render()
       obs, rew, done, info = env.step(env.action_space.sample())
       print(f"{obs} -> {rew}")
    env.close()
    ```

    Am Ende siehst du etwas Ähnliches in der Ausgabe des Notebooks:

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

    Der Beobachtungsvektor, der bei jedem Simulationsschritt zurückgegeben wird, enthält die folgenden Werte:
    - Position des Schlittens
    - Geschwindigkeit des Schlittens
    - Winkel des Stabs
    - Rotationsrate des Stabs

1. Erhalte Min- und Max-Werte dieser Zahlen: (Codeblock 5)

    ```python
    print(env.observation_space.low)
    print(env.observation_space.high)
    ```

    Du wirst auch bemerken, dass der Belohnungswert bei jedem Simulationsschritt immer 1 ist. Das liegt daran, dass unser Ziel ist, so lange wie möglich zu überleben, d.h. den Stab für die längste Zeit in einer einigermaßen vertikalen Position zu halten.

    ✅ Tatsächlich gilt die CartPole-Simulation als gelöst, wenn wir eine durchschnittliche Belohnung von 195 über 100 aufeinanderfolgende Versuche erreichen.

## Zustandsdiskretisierung

Im Q-Learning müssen wir eine Q-Tabelle erstellen, die definiert, was in jedem Zustand zu tun ist. Um dies tun zu können, muss der Zustand **diskret** sein, genauer gesagt, er sollte eine endliche Anzahl diskreter Werte enthalten. Deshalb müssen wir unsere Beobachtungen irgendwie **diskretisieren**, indem wir sie auf eine endliche Menge von Zuständen abbilden.

Es gibt einige Methoden, wie wir das tun können:

- **In Bins unterteilen**. Wenn wir das Intervall eines bestimmten Werts kennen, können wir dieses Intervall in mehrere **Bins** unterteilen und dann den Wert durch die Bin-Nummer ersetzen, zu der er gehört. Dies kann mit der numpy-Methode [`digitize`](https://numpy.org/doc/stable/reference/generated/numpy.digitize.html) erfolgen. So wissen wir genau die Größe des Zustands, da sie von der Anzahl der Bins abhängt, die wir für die Digitalisierung auswählen.
  
✅ Wir können lineare Interpolation verwenden, um Werte auf ein endliches Intervall zu bringen (z.B. von -20 bis 20) und dann die Zahlen durch Runden in Ganzzahlen umwandeln. Das gibt uns weniger Kontrolle über die Größe des Zustands, insbesondere wenn wir die genauen Bereiche der Eingabewerte nicht kennen. Zum Beispiel haben 2 von 4 Werten in unserem Fall keine obere/untere Grenze, was zu einer unendlichen Anzahl von Zuständen führen kann.

In unserem Beispiel wählen wir den zweiten Ansatz. Wie du später bemerken wirst, trotz undefinierter Ober-/Untergrenzen nehmen diese Werte nur selten Werte außerhalb bestimmter endlicher Intervalle an, sodass Zustände mit extremen Werten sehr selten sind.

1. Hier ist die Funktion, die die Beobachtung unseres Modells entgegennimmt und ein Tupel mit 4 ganzzahligen Werten zurückgibt: (Codeblock 6)

    ```python
    def discretize(x):
        return tuple((x/np.array([0.25, 0.25, 0.01, 0.1])).astype(np.int))
    ```

1. Lass uns auch eine andere Diskretisierungsmethode mit Bins erkunden: (Codeblock 7)

    ```python
    def create_bins(i,num):
        return np.arange(num+1)*(i[1]-i[0])/num+i[0]
    
    print("Sample bins for interval (-5,5) with 10 bins\n",create_bins((-5,5),10))
    
    ints = [(-5,5),(-2,2),(-0.5,0.5),(-2,2)] # Werteintervalle für jeden Parameter
    nbins = [20,20,10,10] # Anzahl der Klassen für jeden Parameter
    bins = [create_bins(ints[i],nbins[i]) for i in range(4)]
    
    def discretize_bins(x):
        return tuple(np.digitize(x[i],bins[i]) for i in range(4))
    ```

1. Lass uns nun eine kurze Simulation durchführen und diese diskreten Umweltwerte beobachten. Probiere gerne beide `discretize` und `discretize_bins` aus und schau, ob es Unterschiede gibt.

    ✅ `discretize_bins` gibt die Bin-Nummer zurück, die Null-basiert ist. Für Werte der Eingangsvariablen um 0 gibt es also die Zahl aus der Mitte des Intervalls (10) zurück. In `discretize` haben wir uns nicht um den Bereich der Ausgabewerte gekümmert, sodass sie negativ sein können, der Zustandswert ist nicht verschoben, und 0 entspricht 0. (Codeblock 8)

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

    ✅ Kommentiere die Zeile, die mit `env.render` beginnt, aus, wenn du sehen möchtest, wie die Umgebung ausgeführt wird. Ansonsten kannst du sie im Hintergrund ausführen lassen, was schneller ist. Diese „unsichtbare“ Ausführung verwenden wir während unseres Q-Learning-Prozesses.

## Die Struktur der Q-Tabelle

In unserer vorherigen Lektion war der Zustand ein einfaches Zahlenpaar von 0 bis 8, somit war es praktisch, die Q-Tabelle als numpy-Tensor mit der Form 8x8x2 darzustellen. Wenn wir Bins-Diskretisierung verwenden, kennen wir auch die Größe unseres Zustandsvektors, sodass wir denselben Ansatz verwenden können und den Zustand als Array mit der Form 20x20x10x10x2 darstellen (hier ist 2 die Dimension des Aktionsraums, und die ersten Dimensionen entsprechen der Anzahl der Bins, die wir für jeden Parameter im Beobachtungsraum ausgewählt haben).

Manchmal sind die genauen Dimensionen des Beobachtungsraums jedoch nicht bekannt. Im Fall der Funktion `discretize` können wir nie sicher sein, dass unser Zustand innerhalb bestimmter Grenzen bleibt, weil einige der ursprünglichen Werte nicht begrenzt sind. Deshalb verwenden wir einen etwas anderen Ansatz und stellen die Q-Tabelle als Dictionary dar.

1. Verwende das Paar *(Zustand, Aktion)* als Schlüssel im Dictionary, und der Wert entspricht dem Eintrag in der Q-Tabelle. (Codeblock 9)

    ```python
    Q = {}
    actions = (0,1)
    
    def qvalues(state):
        return [Q.get((state,a),0) for a in actions]
    ```

    Hier definieren wir auch eine Funktion `qvalues()`, die eine Liste der Q-Tabellenwerte für einen gegebenen Zustand zurückgibt, die allen möglichen Aktionen entsprechen. Wenn der Eintrag nicht in der Q-Tabelle vorhanden ist, geben wir als Standardwert 0 zurück.

## Starten wir mit Q-Learning

Nun sind wir bereit, Peter das Balancieren beizubringen!

1. Zuerst setzen wir einige Hyperparameter: (Codeblock 10)

    ```python
    # Hyperparameter
    alpha = 0.3
    gamma = 0.9
    epsilon = 0.90
    ```

    Hier ist `alpha` die **Lernrate**, die angibt, in welchem Ausmaß wir die aktuellen Werte der Q-Tabelle bei jedem Schritt anpassen sollten. In der vorherigen Lektion haben wir mit 1 begonnen und dann `alpha` während des Trainings verringert. In diesem Beispiel halten wir sie der Einfachheit halber konstant, du kannst aber später experimentieren und `alpha`-Werte anpassen.

    `gamma` ist der **Diskontierungsfaktor**, der zeigt, inwieweit zukünftige Belohnungen gegenüber aktuellen belohnt werden.

    `epsilon` ist der **Explorations-/Exploitation-Faktor**, der bestimmt, ob wir Exploration der Exploitation vorziehen oder umgekehrt. In unserem Algorithmus wählen wir in `epsilon` Prozent der Fälle die nächste Aktion basierend auf den Q-Tabellenwerten, und in den restlichen Fällen führen wir eine zufällige Aktion aus. So erkunden wir Bereiche des Suchraums, die wir noch nicht gesehen haben.

    ✅ Beim Balancieren entspricht die Wahl einer zufälligen Aktion (Exploration) einem zufälligen Schlag in die falsche Richtung, und der Stab muss lernen, das Gleichgewicht aus diesen „Fehlern“ wiederherzustellen.

### Verbesserung des Algorithmus

Wir können unseren Algorithmus aus der vorherigen Lektion auch in zwei Punkten verbessern:

- **Berechne die durchschnittliche kumulative Belohnung**, über eine Anzahl von Simulationen. Wir geben den Fortschritt alle 5000 Iterationen aus und mitteln unsere kumulative Belohnung über diesen Zeitraum. Das bedeutet, wenn wir mehr als 195 Punkte erreichen, können wir das Problem als gelöst betrachten, sogar mit höherer Qualität als erforderlich.
  
- **Berechne den maximalen durchschnittlichen kumulativen Wert**, `Qmax`, und speichere die Q-Tabelle, die diesem Ergebnis entspricht. Während des Trainings wirst du bemerken, dass der durchschnittliche kumulative Wert manchmal zu sinken beginnt. Wir möchten die Q-Tabellenwerte speichern, die dem besten während des Trainings beobachteten Modell entsprechen.

1. Sammle alle kumulativen Belohnungen jeder Simulation im `rewards`-Vektor für weitere Darstellungen. (Codeblock 11)

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
        # == führe die Simulation durch ==
        while not done:
            s = discretize(obs)
            if random.random()<epsilon:
                # Ausbeutung - wähle die Aktion entsprechend der Wahrscheinlichkeiten der Q-Tabelle
                v = probs(np.array(qvalues(s)))
                a = random.choices(actions,weights=v)[0]
            else:
                # Erforschung - wähle die Aktion zufällig
                a = np.random.randint(env.action_space.n)
    
            obs, rew, done, info = env.step(a)
            cum_reward+=rew
            ns = discretize(obs)
            Q[(s,a)] = (1 - alpha) * Q.get((s,a),0) + alpha * (rew + gamma * max(qvalues(ns)))
        cum_rewards.append(cum_reward)
        rewards.append(cum_reward)
        # == Drucke periodisch die Ergebnisse und berechne die durchschnittliche Belohnung ==
        if epoch%5000==0:
            print(f"{epoch}: {np.average(cum_rewards)}, alpha={alpha}, epsilon={epsilon}")
            if np.average(cum_rewards) > Qmax:
                Qmax = np.average(cum_rewards)
                Qbest = Q
            cum_rewards=[]
    ```

Was du aus diesen Ergebnissen erkennen kannst:

- **Nahe an unserem Ziel**. Wir sind sehr nahe daran, das Ziel von 195 kumulativen Belohnungen über 100+ aufeinanderfolgende Durchläufe der Simulation zu erreichen, oder haben es möglicherweise sogar erreicht! Auch wenn wir kleinere Zahlen bekommen, wissen wir es noch nicht sicher, weil wir über 5000 Durchläufe mitteln, und nur 100 Durchläufe für das formale Kriterium erforderlich sind.
  
- **Belohnung beginnt zu fallen**. Manchmal beginnt die Belohnung zu fallen, was bedeutet, dass wir bereits gelernte Werte in der Q-Tabelle durch Werte überschreiben, die die Situation verschlimmern.

Diese Beobachtung wird deutlicher, wenn wir den Trainingsfortschritt darstellen.

## Darstellung des Trainingsfortschritts

Während des Trainings haben wir den kumulativen Belohnungswert bei jeder Iteration in den `rewards`-Vektor gesammelt. So sieht es aus, wenn wir ihn gegen die Iterationsnummer darstellen:

```python
plt.plot(rewards)
```

![roher Fortschritt](../../../../translated_images/de/train_progress_raw.2adfdf2daea09c59.webp)

Aus diesem Diagramm lässt sich nicht viel ablesen, da die Länge der Trainingssessions aufgrund der stochastischen Natur des Trainingsprozesses stark variiert. Um dieses Diagramm besser zu interpretieren, können wir den **laufenden Durchschnitt** über eine Serie von Experimenten, sagen wir 100, berechnen. Das geht bequem mit `np.convolve`: (Codeblock 12)

```python
def running_average(x,window):
    return np.convolve(x,np.ones(window)/window,mode='valid')

plt.plot(running_average(rewards,100))
```

![Trainingsfortschritt](../../../../translated_images/de/train_progress_runav.c71694a8fa9ab359.webp)

## Variation der Hyperparameter

Um das Lernen stabiler zu machen, macht es Sinn, einige unserer Hyperparameter während des Trainings anzupassen. Insbesondere:

- **Für die Lernrate**, `alpha`, können wir mit Werten nahe 1 beginnen und den Parameter dann langsam verringern. Mit der Zeit erhalten wir gute Wahrscheinlichkeitswerte in der Q-Tabelle und sollten diese nur noch leicht anpassen und nicht mehr komplett überschreiben.

- **Erhöhe epsilon**. Wir könnten `epsilon` langsam erhöhen, um weniger zu explorieren und mehr zu exploitieren. Wahrscheinlich macht es Sinn, mit einem niedrigen Wert für `epsilon` zu starten und ihn bis fast 1 zu erhöhen.

> **Aufgabe 1**: Spiele mit den Hyperparameterwerten und schaue, ob du eine höhere kumulative Belohnung erzielen kannst. Kommst du über 195?


> **Aufgabe 2**: Um das Problem formell zu lösen, müssen Sie eine durchschnittliche Belohnung von 195 über 100 aufeinanderfolgende Durchläufe erreichen. Messen Sie das während des Trainings und stellen Sie sicher, dass Sie das Problem formell gelöst haben!

## Das Ergebnis in Aktion sehen

Es wäre interessant, tatsächlich zu sehen, wie sich das trainierte Modell verhält. Lassen Sie uns die Simulation laufen und dieselbe Aktionsauswahlstrategie wie beim Training verfolgen, indem wir entsprechend der Wahrscheinlichkeitsverteilung in der Q-Tabelle sampeln: (Codeblock 13)

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

Sie sollten so etwas sehen:

![ein balancierender Cartpole](../../../../8-Reinforcement/2-Gym/images/cartpole-balance.gif)

---

## 🚀Herausforderung

> **Aufgabe 3**: Hier haben wir die finale Kopie der Q-Tabelle verwendet, die möglicherweise nicht die beste ist. Denken Sie daran, dass wir die bestperformende Q-Tabelle in der Variable `Qbest` gespeichert haben! Versuchen Sie dasselbe Beispiel mit der bestperformenden Q-Tabelle, indem Sie `Qbest` auf `Q` kopieren, und sehen Sie, ob Sie einen Unterschied bemerken.

> **Aufgabe 4**: Hier haben wir nicht bei jedem Schritt die beste Aktion ausgewählt, sondern stattdessen entsprechend der Wahrscheinlichkeitsverteilung gesampelt. Würde es mehr Sinn machen, immer die beste Aktion mit dem höchsten Q-Tabellenwert auszuwählen? Das kann durch die Verwendung der Funktion `np.argmax` geschehen, um die Aktionsnummer zu finden, die dem höchsten Q-Tabellenwert entspricht. Implementieren Sie diese Strategie und prüfen Sie, ob sich das Balancieren verbessert.

## [Quiz nach der Vorlesung](https://ff-quizzes.netlify.app/en/ml/)

## Aufgabe
[Trainieren eines Mountain Car](assignment.md)

## Fazit

Wir haben nun gelernt, wie man Agenten trainiert, um gute Ergebnisse zu erzielen, indem man ihnen nur eine Belohnungsfunktion gibt, die den gewünschten Zustand des Spiels definiert, und indem man ihnen die Möglichkeit gibt, den Suchraum intelligent zu erkunden. Wir haben den Q-Learning-Algorithmus erfolgreich sowohl bei diskreten als auch bei kontinuierlichen Umgebungen angewendet, allerdings mit diskreten Aktionen.

Es ist wichtig, auch Situationen zu untersuchen, bei denen die Aktionsdimension ebenfalls kontinuierlich ist und der Beobachtungsraum viel komplexer ist, zum Beispiel ein Bild vom Atari-Spielbildschirm. In solchen Problemen müssen wir oft leistungsfähigere maschinelle Lernmethoden wie neuronale Netze verwenden, um gute Ergebnisse zu erzielen. Diese fortgeschritteneren Themen sind Gegenstand unseres bevorstehenden, fortgeschrittenen AI-Kurses.

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**Haftungsausschluss**:
Dieses Dokument wurde mit dem KI-Übersetzungsdienst [Co-op Translator](https://github.com/Azure/co-op-translator) übersetzt. Obwohl wir uns um Genauigkeit bemühen, beachten Sie bitte, dass automatisierte Übersetzungen Fehler oder Ungenauigkeiten enthalten können. Das Originaldokument in seiner Ursprungssprache gilt als maßgebliche Quelle. Bei kritischen Informationen wird eine professionelle menschliche Übersetzung empfohlen. Wir übernehmen keine Haftung für Missverständnisse oder Fehlinterpretationen, die aus der Verwendung dieser Übersetzung entstehen.
<!-- CO-OP TRANSLATOR DISCLAIMER END -->