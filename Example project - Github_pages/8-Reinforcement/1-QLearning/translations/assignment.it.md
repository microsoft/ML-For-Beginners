# Un mondo più realistico

Nella situazione citata, Pierino riusciva a muoversi quasi senza stancarsi o avere fame. In un mondo più realistico, ci si deve sedere e riposare di tanto in tanto, e anche nutrirsi. Si rende questo mondo più realistico, implementando le seguenti regole:

1. Spostandosi da un luogo all'altro, Pierino perde **energia** e accumula un po' di **fatica**.
2. Pierino può guadagnare più energia mangiando mele.
3. Pierino può recuperare energie riposando sotto l'albero o sull'erba (cioè camminando in una posizione nella tabola di gioco con un un albero o un prato - campo verde)
4. Pierino ha bisogno di trovare e uccidere il lupo
5. Per uccidere il lupo, Pierino deve avere determinati livelli di energia e fatica, altrimenti perde la battaglia.

## Istruzioni

Usare il notebook originale [notebook.ipynb](../notebook.ipynb) come punto di partenza per la propria soluzione.

Modificare la funzione di ricompensa in base alle regole del gioco, eseguire l'algoritmo di reinforcement learning per apprendere la migliore strategia per vincere la partita e confrontare i risultati della passeggiata aleatoria con il proprio algoritmo in termini di numero di partite vinte e perse.

> **Nota**: in questo nuovo mondo, lo stato è più complesso e oltre alla posizione umana include anche la fatica e i livelli di energia. Si può scegliere di rappresentare lo stato come una tupla (Board,energy,fatigue) - (Tavola, Energia, Fatica), o definire una classe per lo stato (si potrebbe anche volerla derivare da `Board`), o anche modificare la classe `Board` originale all'interno di [rlboard.py](../rlboard.py).

Nella propria soluzione, mantenere il codice responsabile della strategia di passeggiata aleatoria e confrontare i risultati del proprio algoritmo con la passeggiata aleatoria alla fine.

> **Nota**: potrebbe essere necessario regolare gli iperparametri per farlo funzionare, in particolare il numero di epoche. Poiché il successo del gioco (lotta contro il lupo) è un evento raro, ci si può aspettare un tempo di allenamento molto più lungo.

## Rubrica

| Criteri | Ottimo | Adeguato | Necessita miglioramento |
| -------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
|          | Viene presentato un notebook con la definizione delle nuove regole del mondo, l'algoritmo di Q-Learning e alcune spiegazioni testuali. Q-Learning è in grado di migliorare significativamente i risultati rispetto a random walk. | Viene presentato il notebook, viene implementato Q-Learning e migliora i risultati rispetto a random walk, ma non in modo significativo; o il notebook è scarsamente documentato e il codice non è ben strutturato | Vengono fatti alcuni tentativi di ridefinire le regole del mondo, ma l'algoritmo Q-Learning non funziona o la funzione di ricompensa non è completamente definita |
