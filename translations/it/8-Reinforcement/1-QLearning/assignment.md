# Un Mondo Più Realistico

Nella nostra situazione, Peter era in grado di muoversi quasi senza stancarsi o avere fame. In un mondo più realistico, deve sedersi e riposare di tanto in tanto, e anche nutrirsi. Rendiamo il nostro mondo più realistico, implementando le seguenti regole:

1. Spostandosi da un luogo all'altro, Peter perde **energia** e guadagna un po' di **fatica**.
2. Peter può guadagnare più energia mangiando mele.
3. Peter può liberarsi della fatica riposando sotto l'albero o sull'erba (cioè camminando in una posizione della tavola con un albero o erba - campo verde)
4. Peter deve trovare e uccidere il lupo.
5. Per uccidere il lupo, Peter deve avere certi livelli di energia e fatica, altrimenti perde la battaglia.

## Istruzioni

Usa il [notebook.ipynb](../../../../8-Reinforcement/1-QLearning/notebook.ipynb) originale come punto di partenza per la tua soluzione.

Modifica la funzione di ricompensa sopra secondo le regole del gioco, esegui l'algoritmo di apprendimento per rinforzo per imparare la migliore strategia per vincere il gioco, e confronta i risultati del cammino casuale con il tuo algoritmo in termini di numero di giochi vinti e persi.

> **Note**: Nel tuo nuovo mondo, lo stato è più complesso, e oltre alla posizione umana include anche i livelli di fatica e energia. Puoi scegliere di rappresentare lo stato come una tupla (Board,energy,fatigue), o definire una classe per lo stato (puoi anche voler derivarla da `Board`), o anche modificare la classe originale `Board` all'interno di [rlboard.py](../../../../8-Reinforcement/1-QLearning/rlboard.py).

Nella tua soluzione, per favore mantieni il codice responsabile della strategia del cammino casuale, e confronta i risultati del tuo algoritmo con il cammino casuale alla fine.

> **Note**: Potrebbe essere necessario regolare gli iperparametri per farlo funzionare, specialmente il numero di epoche. Poiché il successo del gioco (combattere il lupo) è un evento raro, puoi aspettarti tempi di allenamento molto più lunghi.

## Rubrica

| Criteri  | Esemplare                                                                                                                                                                                             | Adeguato                                                                                                                                                                                | Bisogno di Miglioramento                                                                                                                    |
| -------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
|          | Un notebook è presentato con la definizione delle nuove regole del mondo, algoritmo Q-Learning e alcune spiegazioni testuali. Q-Learning è in grado di migliorare significativamente i risultati rispetto al cammino casuale. | Il notebook è presentato, Q-Learning è implementato e migliora i risultati rispetto al cammino casuale, ma non significativamente; o il notebook è scarsamente documentato e il codice non è ben strutturato | È stato fatto qualche tentativo di ridefinire le regole del mondo, ma l'algoritmo Q-Learning non funziona, o la funzione di ricompensa non è completamente definita |

**Disclaimer**:
Questo documento è stato tradotto utilizzando servizi di traduzione automatica basati su intelligenza artificiale. Sebbene ci impegniamo per garantire l'accuratezza, si prega di essere consapevoli che le traduzioni automatiche possono contenere errori o imprecisioni. Il documento originale nella sua lingua nativa dovrebbe essere considerato la fonte autorevole. Per informazioni critiche, si raccomanda una traduzione umana professionale. Non siamo responsabili per eventuali incomprensioni o interpretazioni errate derivanti dall'uso di questa traduzione.