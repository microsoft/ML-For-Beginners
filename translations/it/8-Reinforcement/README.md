# Introduzione all'apprendimento per rinforzo

L'apprendimento per rinforzo, RL, è considerato uno dei paradigmi fondamentali del machine learning, accanto all'apprendimento supervisionato e non supervisionato. L'RL riguarda le decisioni: prendere le decisioni giuste o almeno imparare da esse.

Immagina di avere un ambiente simulato come il mercato azionario. Cosa succede se imponi una determinata regolamentazione? Ha un effetto positivo o negativo? Se succede qualcosa di negativo, devi prendere questo _rinforzo negativo_, imparare da esso e cambiare rotta. Se l'esito è positivo, devi costruire su quel _rinforzo positivo_.

![peter e il lupo](../../../translated_images/peter.779730f9ba3a8a8d9290600dcf55f2e491c0640c785af7ac0d64f583c49b8864.it.png)

> Peter e i suoi amici devono sfuggire al lupo affamato! Immagine di [Jen Looper](https://twitter.com/jenlooper)

## Argomento regionale: Peter e il Lupo (Russia)

[Peter e il Lupo](https://en.wikipedia.org/wiki/Peter_and_the_Wolf) è una fiaba musicale scritta dal compositore russo [Sergei Prokofiev](https://en.wikipedia.org/wiki/Sergei_Prokofiev). È una storia che parla del giovane pioniere Peter, che coraggiosamente esce di casa per inseguire il lupo nella radura della foresta. In questa sezione, addestreremo algoritmi di machine learning che aiuteranno Peter a:

- **Esplorare** l'area circostante e costruire una mappa di navigazione ottimale
- **Imparare** a usare uno skateboard e a bilanciarsi su di esso, per muoversi più velocemente.

[![Peter e il Lupo](https://img.youtube.com/vi/Fmi5zHg4QSM/0.jpg)](https://www.youtube.com/watch?v=Fmi5zHg4QSM)

> 🎥 Clicca sull'immagine sopra per ascoltare Peter e il Lupo di Prokofiev

## Apprendimento per rinforzo

Nelle sezioni precedenti, hai visto due esempi di problemi di machine learning:

- **Supervisionato**, dove abbiamo dataset che suggeriscono soluzioni campione al problema che vogliamo risolvere. [Classificazione](../4-Classification/README.md) e [regressione](../2-Regression/README.md) sono compiti di apprendimento supervisionato.
- **Non supervisionato**, in cui non abbiamo dati di addestramento etichettati. L'esempio principale di apprendimento non supervisionato è il [Clustering](../5-Clustering/README.md).

In questa sezione, ti introdurremo a un nuovo tipo di problema di apprendimento che non richiede dati di addestramento etichettati. Esistono diversi tipi di tali problemi:

- **[Apprendimento semi-supervisionato](https://wikipedia.org/wiki/Semi-supervised_learning)**, dove abbiamo molti dati non etichettati che possono essere utilizzati per pre-addestrare il modello.
- **[Apprendimento per rinforzo](https://wikipedia.org/wiki/Reinforcement_learning)**, in cui un agente impara a comportarsi eseguendo esperimenti in un ambiente simulato.

### Esempio - gioco per computer

Supponiamo di voler insegnare a un computer a giocare a un gioco, come gli scacchi o [Super Mario](https://wikipedia.org/wiki/Super_Mario). Per far giocare il computer, dobbiamo fargli prevedere quale mossa fare in ciascuno degli stati del gioco. Anche se potrebbe sembrare un problema di classificazione, non lo è - perché non abbiamo un dataset con stati e azioni corrispondenti. Anche se potremmo avere alcuni dati come partite di scacchi esistenti o registrazioni di giocatori che giocano a Super Mario, è probabile che quei dati non coprano sufficientemente un numero sufficiente di stati possibili.

Invece di cercare dati di gioco esistenti, **l'Apprendimento per Rinforzo** (RL) si basa sull'idea di *far giocare il computer* molte volte e osservare il risultato. Pertanto, per applicare l'Apprendimento per Rinforzo, abbiamo bisogno di due cose:

- **Un ambiente** e **un simulatore** che ci permettano di giocare molte volte. Questo simulatore definirebbe tutte le regole del gioco, nonché gli stati e le azioni possibili.

- **Una funzione di ricompensa**, che ci dica quanto bene abbiamo fatto durante ogni mossa o partita.

La principale differenza tra gli altri tipi di machine learning e l'RL è che nell'RL tipicamente non sappiamo se vinciamo o perdiamo fino a quando non finiamo il gioco. Pertanto, non possiamo dire se una certa mossa da sola sia buona o no - riceviamo una ricompensa solo alla fine del gioco. E il nostro obiettivo è progettare algoritmi che ci permettano di addestrare un modello in condizioni di incertezza. Impareremo un algoritmo di RL chiamato **Q-learning**.

## Lezioni

1. [Introduzione all'apprendimento per rinforzo e Q-Learning](1-QLearning/README.md)
2. [Utilizzo di un ambiente di simulazione gym](2-Gym/README.md)

## Crediti

"L'Introduzione all'Apprendimento per Rinforzo" è stata scritta con ♥️ da [Dmitry Soshnikov](http://soshnikov.com)

**Avvertenza**:
Questo documento è stato tradotto utilizzando servizi di traduzione automatizzati basati su intelligenza artificiale. Sebbene ci sforziamo di garantire l'accuratezza, si prega di notare che le traduzioni automatizzate possono contenere errori o imprecisioni. Il documento originale nella sua lingua madre dovrebbe essere considerato la fonte autorevole. Per informazioni critiche, si raccomanda una traduzione professionale umana. Non siamo responsabili per eventuali malintesi o interpretazioni errate derivanti dall'uso di questa traduzione.