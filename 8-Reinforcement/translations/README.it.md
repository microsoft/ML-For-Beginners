# Introduzione al reinforcement learning

Il reinforcement learning (apprendimento per rinforzo), RL, è visto come uno dei paradigmi di base di machine learning, accanto all'apprendimento supervisionato e all'apprendimento non supervisionato. RL è tutta una questione di decisioni: fornire le decisioni giuste o almeno imparare da esse.

Si immagini di avere un ambiente simulato come il mercato azionario. Cosa succede se si impone un determinato regolamento. Ha un effetto positivo o negativo? Se accade qualcosa di negativo, si deve accettare questo _rinforzo negativo_, imparare da esso e cambiare rotta. Se è un risultato positivo, si deve costruire su quel _rinforzo positivo_.

![Pierino e il lupo](../images/peter.png)

> Pierino e i suoi amici devono sfuggire al lupo affamato! Immagine di [Jen Looper](https://twitter.com/jenlooper)

## Tema regionale: Pierino e il lupo (Russia)

[Pierino e il Lupo](https://it.wikipedia.org/wiki/Pierino_e_il_lupo) è una fiaba musicale scritta dal compositore russo [Sergei Prokofiev](https://it.wikipedia.org/wiki/Sergei_Prokofiev). È la storia del giovane pioniere Pierino, che coraggiosamente esce di casa per inseguire il lupo nella radura della foresta . In questa sezione, si addestreranno algoritmi di machine learning che aiuteranno Pierino a:

- **Esplorare** l'area circostante e costruire una mappa di navigazione ottimale
- **Imparare** a usare uno skateboard e bilanciarsi su di esso, per muoversi più velocemente.

[![Pierino e il lupo](https://img.youtube.com/vi/Fmi5zHg4QSM/0.jpg)](https://www.youtube.com/watch?v=Fmi5zHg4QSM)

> 🎥 Cliccare sull'immagine sopra per ascoltare Pierino e il Lupo di Prokofiev

## Reinforcement learning

Nelle sezioni precedenti, si sono visti due esempi di problemi di machine learning:

- **Supervisionato**, dove si ha un insieme di dati che suggeriscono soluzioni campione al problema da risolvere. [La classificazione](../../4-Classification/translations/README.it.md) e la [regressione](../../2-Regression/translations/README.it.md) sono attività di apprendimento supervisionato.
- **Non** supervisionato, in cui non si dispone di dati di allenamento etichettati. L'esempio principale di apprendimento non supervisionato è il [Clustering](../../5-Clustering/translations/README.it.md).

In questa sezione, viene presentato un nuovo tipo di problemi di apprendimento che non richiede dati di addestramento etichettati. Esistono diversi tipi di tali problemi:

- **[Apprendimento semi-supervisionato](https://wikipedia.org/wiki/Semi-supervised_learning)**, in cui si dispone di molti dati non etichettati che possono essere utilizzati per pre-addestrare il modello.
- **[Apprendimento per rinforzo](https://it.wikipedia.org/wiki/Apprendimento_per_rinforzo)**, in cui un agente impara come comportarsi eseguendo esperimenti in un ambiente simulato.

### Esempio: gioco per computer

Si supponga di voler insegnare a un computer a giocare a un gioco, come gli scacchi o [Super Mario](https://it.wikipedia.org/wiki/Mario_(serie_di_videogiochi)). Affinché il computer possa giocare, occorre prevedere quale mossa fare in ciascuno degli stati di gioco. Anche se questo può sembrare un problema di classificazione, non lo è, perché non si dispone di un insieme di dati con stati e azioni corrispondenti. Sebbene si potrebbero avere alcuni dati come partite di scacchi esistenti o registrazioni di giocatori che giocano a Super Mario, è probabile che tali dati non coprano a sufficienza un numero adeguato di possibili stati.

Invece di cercare dati di gioco esistenti, **Reinforcement Learning** (RL) si basa sull'idea di *far giocare il computer* molte volte e osservare il risultato. Quindi, per applicare il Reinforcement Learning, servono due cose:

- **Un ambiente** e **un simulatore** che permettono di giocare molte volte un gioco. Questo simulatore definirebbe tutte le regole del gioco, nonché possibili stati e azioni.

- **Una funzione di ricompensa**, che informi di quanto bene si è fatto durante ogni mossa o partita.

La differenza principale tra altri tipi di machine learning e RL è che in RL in genere non si sa se si vince o si perde finchè non si finisce il gioco. Pertanto, non è possibile dire se una determinata mossa da sola sia buona o meno: si riceve una ricompensa solo alla fine del gioco. L'obiettivo è progettare algoritmi che consentano di addestrare un modello in condizioni incerte. Si imparerà a conoscere un algoritmo RL chiamato **Q-learning**.

## Lezioni

1. [Introduzione a reinforcement learning e al Q-Learning](../1-QLearning/translations/README.it.md)
2. [Utilizzo di un ambiente di simulazione in palestra](../2-Gym/translations/README.it.md)

## Crediti

"Introduzione al Reinforcement Learning" è stato scritto con ♥️ da [Dmitry Soshnikov](http://soshnikov.com)
