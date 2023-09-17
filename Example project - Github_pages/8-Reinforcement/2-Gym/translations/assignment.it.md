# Addestrare un'Auto di Montagna

[OpenAI Gym](http://gym.openai.com) è stato progettato in modo tale che tutti gli ambienti forniscano la stessa API, ovvero gli stessi metodi di `reset`, `step` e `render` e le stesse astrazioni dello **spazio di azione** e dello **spazio di osservazione**. Pertanto dovrebbe essere possibile adattare gli stessi algoritmi di reinforcement learning a diversi ambienti con modifiche minime al codice.

## Un ambiente automobilistico di montagna

[L'ambiente Mountain Car](https://gym.openai.com/envs/MountainCar-v0/) contiene un'auto bloccata in una valle:

<img src="../images/mountaincar.png" width="300"/>

L'obiettivo è uscire dalla valle e catturare la bandiera, compiendo ad ogni passaggio una delle seguenti azioni:

| Valore | Significato |
|---|---|
| 0 | Accelerare a sinistra |
| 1 | Non accelerare |
| 2 | Accelerare a destra |

Il trucco principale di questo problema è, tuttavia, che il motore dell'auto non è abbastanza forte per scalare la montagna in un solo passaggio. Pertanto, l'unico modo per avere successo è andare avanti e indietro per aumentare lo slancio.

Lo spazio di osservazione è costituito da due soli valori:

| Num | Osservazione | Min | Max |
|-----|--------------|-----|-----|
| 0 | Posizione dell'auto | -1,2 | 0.6 |
| 1 | Velocità dell'auto | -0.07% | 0,07 |

Il sistema di ricompensa per l'auto di montagna è piuttosto complicato:

* La ricompensa di 0 viene assegnata se l'agente ha raggiunto la bandiera (posizione = 0,5) in cima alla montagna.
* La ricompensa di -1 viene assegnata se la posizione dell'agente è inferiore a 0,5.

L'episodio termina se la posizione dell'auto è maggiore di 0,5 o la durata dell'episodio è maggiore di 200.

## Istruzioni

Adattare l'algoritmo di reinforcement learning per risolvere il problema della macchina di montagna. Iniziare con il codice nel [notebook.ipynb](notebook.ipynb) esistente, sostituire il nuovo ambiente, modificare le funzioni di discretizzazione dello stato e provare ad addestrare l'algoritmo esistente con modifiche minime del codice. Ottimizzare il risultato regolando gli iperparametri.

> **Nota**: è probabile che sia necessaria la regolazione degli iperparametri per far convergere l'algoritmo.
## Rubrica

| Criteri | Ottimo | Adeguato | Necessita miglioramento |
| -------- | --------- | -------- | ----------------- |
|          | L'algoritmo di Q-Learning è stato adattato con successo dall'esempio CartPole, con modifiche minime al codice, che è in grado di risolvere il problema di catturare la bandiera in meno di 200 passaggi. | Un nuovo algoritmo di Q-Learning è stato adottato da Internet, ma è ben documentato; oppure è stato adottato l'algoritmo esistente, ma non raggiunge i risultati desiderati | Lo studente non è stato in grado di adottare con successo alcun algoritmo, ma ha compiuto passi sostanziali verso la soluzione (discretizzazione dello stato implementata, struttura dati Q-Table, ecc.) |
