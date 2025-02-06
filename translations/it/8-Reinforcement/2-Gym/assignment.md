# Allenare l'Auto di Montagna

[OpenAI Gym](http://gym.openai.com) è stato progettato in modo tale che tutti gli ambienti forniscano la stessa API - cioè gli stessi metodi `reset`, `step` e `render`, e le stesse astrazioni di **spazio delle azioni** e **spazio delle osservazioni**. Pertanto, dovrebbe essere possibile adattare gli stessi algoritmi di apprendimento per rinforzo a diversi ambienti con minime modifiche al codice.

## Un Ambiente di Auto di Montagna

L'[ambiente dell'Auto di Montagna](https://gym.openai.com/envs/MountainCar-v0/) contiene un'auto bloccata in una valle:
L'obiettivo è uscire dalla valle e catturare la bandiera, compiendo ad ogni passo una delle seguenti azioni:

| Valore | Significato |
|---|---|
| 0 | Accelerare a sinistra |
| 1 | Non accelerare |
| 2 | Accelerare a destra |

Il trucco principale di questo problema è, tuttavia, che il motore dell'auto non è abbastanza potente da scalare la montagna in un solo passaggio. Pertanto, l'unico modo per avere successo è guidare avanti e indietro per accumulare slancio.

Lo spazio delle osservazioni consiste di soli due valori:

| Num | Osservazione  | Min | Max |
|-----|--------------|-----|-----|
|  0  | Posizione dell'Auto | -1.2| 0.6 |
|  1  | Velocità dell'Auto | -0.07 | 0.07 |

Il sistema di ricompensa per l'auto di montagna è piuttosto complicato:

 * Una ricompensa di 0 viene assegnata se l'agente ha raggiunto la bandiera (posizione = 0.5) in cima alla montagna.
 * Una ricompensa di -1 viene assegnata se la posizione dell'agente è inferiore a 0.5.

L'episodio termina se la posizione dell'auto è superiore a 0.5, o se la durata dell'episodio è superiore a 200.
## Istruzioni

Adatta il nostro algoritmo di apprendimento per rinforzo per risolvere il problema dell'auto di montagna. Inizia con il codice esistente nel [notebook.ipynb](../../../../8-Reinforcement/2-Gym/notebook.ipynb), sostituisci il nuovo ambiente, cambia le funzioni di discretizzazione dello stato e cerca di far allenare l'algoritmo esistente con minime modifiche al codice. Ottimizza il risultato regolando gli iperparametri.

> **Nota**: È probabile che sia necessario regolare gli iperparametri per far convergere l'algoritmo. 
## Rubrica

| Criteri | Esemplare | Adeguato | Bisogno di Miglioramento |
| -------- | --------- | -------- | ----------------- |
|          | L'algoritmo di Q-Learning è stato adattato con successo dall'esempio di CartPole, con minime modifiche al codice, ed è in grado di risolvere il problema di catturare la bandiera in meno di 200 passi. | È stato adottato un nuovo algoritmo di Q-Learning da Internet, ma è ben documentato; oppure l'algoritmo esistente è stato adottato, ma non raggiunge i risultati desiderati | Lo studente non è stato in grado di adottare con successo alcun algoritmo, ma ha fatto passi sostanziali verso la soluzione (implementazione della discretizzazione dello stato, struttura dati della Q-Table, ecc.) |

**Disclaimer**:
Questo documento è stato tradotto utilizzando servizi di traduzione basati su intelligenza artificiale. Sebbene ci impegniamo per garantire l'accuratezza, si prega di essere consapevoli che le traduzioni automatiche possono contenere errori o imprecisioni. Il documento originale nella sua lingua nativa dovrebbe essere considerato la fonte autorevole. Per informazioni critiche, si raccomanda una traduzione professionale umana. Non siamo responsabili per eventuali malintesi o interpretazioni errate derivanti dall'uso di questa traduzione.