<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "20ca019012b1725de956681d036d8b18",
  "translation_date": "2025-08-29T22:02:46+00:00",
  "source_file": "8-Reinforcement/README.md",
  "language_code": "it"
}
-->
# Introduzione al reinforcement learning

Il reinforcement learning, RL, è considerato uno dei paradigmi fondamentali del machine learning, accanto al supervised learning e all'unsupervised learning. RL riguarda le decisioni: prendere le decisioni giuste o almeno imparare da esse.

Immagina di avere un ambiente simulato, come il mercato azionario. Cosa succede se imponi una determinata regolamentazione? Ha un effetto positivo o negativo? Se accade qualcosa di negativo, devi prendere questo _rinforzo negativo_, imparare da esso e cambiare rotta. Se invece l'esito è positivo, devi costruire su quel _rinforzo positivo_.

![peter and the wolf](../../../translated_images/peter.779730f9ba3a8a8d9290600dcf55f2e491c0640c785af7ac0d64f583c49b8864.it.png)

> Peter e i suoi amici devono scappare dal lupo affamato! Immagine di [Jen Looper](https://twitter.com/jenlooper)

## Argomento regionale: Pierino e il lupo (Russia)

[Pierino e il lupo](https://it.wikipedia.org/wiki/Pierino_e_il_lupo) è una fiaba musicale scritta dal compositore russo [Sergei Prokofiev](https://it.wikipedia.org/wiki/Sergej_Prokof%27ev). È la storia del giovane pioniere Pierino, che coraggiosamente esce di casa per andare nella radura della foresta a caccia del lupo. In questa sezione, addestreremo algoritmi di machine learning che aiuteranno Pierino a:

- **Esplorare** l'area circostante e costruire una mappa di navigazione ottimale.
- **Imparare** a usare uno skateboard e a mantenere l'equilibrio, per spostarsi più velocemente.

[![Pierino e il lupo](https://img.youtube.com/vi/Fmi5zHg4QSM/0.jpg)](https://www.youtube.com/watch?v=Fmi5zHg4QSM)

> 🎥 Clicca sull'immagine sopra per ascoltare Pierino e il lupo di Prokofiev

## Reinforcement learning

Nelle sezioni precedenti, hai visto due esempi di problemi di machine learning:

- **Supervised**, in cui abbiamo dataset che suggeriscono soluzioni campione al problema che vogliamo risolvere. [Classificazione](../4-Classification/README.md) e [regressione](../2-Regression/README.md) sono compiti di supervised learning.
- **Unsupervised**, in cui non abbiamo dati di addestramento etichettati. L'esempio principale di unsupervised learning è il [Clustering](../5-Clustering/README.md).

In questa sezione, ti introdurremo a un nuovo tipo di problema di apprendimento che non richiede dati di addestramento etichettati. Esistono diversi tipi di tali problemi:

- **[Apprendimento semi-supervisionato](https://it.wikipedia.org/wiki/Apprendimento_semi-supervisionato)**, in cui abbiamo molti dati non etichettati che possono essere utilizzati per pre-addestrare il modello.
- **[Reinforcement learning](https://it.wikipedia.org/wiki/Apprendimento_per_ricompensa)**, in cui un agente impara come comportarsi eseguendo esperimenti in un ambiente simulato.

### Esempio - videogioco

Supponiamo di voler insegnare a un computer a giocare a un videogioco, come gli scacchi o [Super Mario](https://it.wikipedia.org/wiki/Super_Mario). Per far giocare il computer, dobbiamo fargli prevedere quale mossa fare in ciascuno degli stati del gioco. Anche se potrebbe sembrare un problema di classificazione, non lo è - perché non abbiamo un dataset con stati e azioni corrispondenti. Anche se potremmo avere alcuni dati, come partite di scacchi esistenti o registrazioni di giocatori che giocano a Super Mario, è probabile che tali dati non coprano sufficientemente un numero abbastanza grande di stati possibili.

Invece di cercare dati di gioco esistenti, il **Reinforcement Learning** (RL) si basa sull'idea di *far giocare il computer* molte volte e osservare il risultato. Pertanto, per applicare il Reinforcement Learning, abbiamo bisogno di due cose:

- **Un ambiente** e **un simulatore** che ci permettano di giocare molte volte. Questo simulatore definirebbe tutte le regole del gioco, così come gli stati e le azioni possibili.

- **Una funzione di ricompensa**, che ci dica quanto bene abbiamo fatto durante ogni mossa o partita.

La principale differenza tra altri tipi di machine learning e RL è che in RL tipicamente non sappiamo se vinciamo o perdiamo fino a quando non terminiamo la partita. Pertanto, non possiamo dire se una certa mossa da sola sia buona o meno - riceviamo una ricompensa solo alla fine della partita. Il nostro obiettivo è progettare algoritmi che ci permettano di addestrare un modello in condizioni di incertezza. Impareremo un algoritmo di RL chiamato **Q-learning**.

## Lezioni

1. [Introduzione al reinforcement learning e al Q-Learning](1-QLearning/README.md)
2. [Utilizzo di un ambiente di simulazione gym](2-Gym/README.md)

## Crediti

"L'introduzione al Reinforcement Learning" è stata scritta con ♥️ da [Dmitry Soshnikov](http://soshnikov.com)

---

**Disclaimer**:  
Questo documento è stato tradotto utilizzando il servizio di traduzione automatica [Co-op Translator](https://github.com/Azure/co-op-translator). Sebbene ci impegniamo per garantire l'accuratezza, si prega di notare che le traduzioni automatiche possono contenere errori o imprecisioni. Il documento originale nella sua lingua nativa dovrebbe essere considerato la fonte autorevole. Per informazioni critiche, si raccomanda una traduzione professionale effettuata da un traduttore umano. Non siamo responsabili per eventuali incomprensioni o interpretazioni errate derivanti dall'uso di questa traduzione.