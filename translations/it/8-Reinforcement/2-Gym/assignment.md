<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "1f2b7441745eb52e25745423b247016b",
  "translation_date": "2025-08-29T22:16:40+00:00",
  "source_file": "8-Reinforcement/2-Gym/assignment.md",
  "language_code": "it"
}
-->
# Allenare Mountain Car

[OpenAI Gym](http://gym.openai.com) è stato progettato in modo tale che tutti gli ambienti forniscano la stessa API - ovvero gli stessi metodi `reset`, `step` e `render`, e le stesse astrazioni di **spazio delle azioni** e **spazio delle osservazioni**. Pertanto, dovrebbe essere possibile adattare gli stessi algoritmi di apprendimento per rinforzo a diversi ambienti con modifiche minime al codice.

## Un ambiente Mountain Car

[L'ambiente Mountain Car](https://gym.openai.com/envs/MountainCar-v0/) contiene un'auto bloccata in una valle:

L'obiettivo è uscire dalla valle e catturare la bandiera, compiendo ad ogni passo una delle seguenti azioni:

| Valore | Significato |
|---|---|
| 0 | Accelerare a sinistra |
| 1 | Non accelerare |
| 2 | Accelerare a destra |

Il principale trucco di questo problema, tuttavia, è che il motore dell'auto non è abbastanza potente da scalare la montagna in un unico passaggio. Pertanto, l'unico modo per riuscire è guidare avanti e indietro per accumulare slancio.

Lo spazio delle osservazioni consiste in soli due valori:

| Num | Osservazione  | Min | Max |
|-----|--------------|-----|-----|
|  0  | Posizione dell'auto | -1.2| 0.6 |
|  1  | Velocità dell'auto | -0.07 | 0.07 |

Il sistema di ricompensa per Mountain Car è piuttosto complicato:

 * Una ricompensa di 0 viene assegnata se l'agente raggiunge la bandiera (posizione = 0.5) in cima alla montagna.
 * Una ricompensa di -1 viene assegnata se la posizione dell'agente è inferiore a 0.5.

L'episodio termina se la posizione dell'auto supera 0.5, o se la lunghezza dell'episodio è maggiore di 200.
## Istruzioni

Adatta il nostro algoritmo di apprendimento per rinforzo per risolvere il problema di Mountain Car. Parti dal codice esistente [notebook.ipynb](notebook.ipynb), sostituisci il nuovo ambiente, modifica le funzioni di discretizzazione dello stato e cerca di far allenare l'algoritmo esistente con modifiche minime al codice. Ottimizza il risultato regolando gli iperparametri.

> **Nota**: È probabile che sia necessario regolare gli iperparametri per far convergere l'algoritmo. 
## Valutazione

| Criteri | Esemplare | Adeguato | Da migliorare |
| -------- | --------- | -------- | ----------------- |
|          | L'algoritmo Q-Learning è stato adattato con successo dall'esempio CartPole, con modifiche minime al codice, ed è in grado di risolvere il problema di catturare la bandiera in meno di 200 passi. | Un nuovo algoritmo Q-Learning è stato adottato da Internet, ma è ben documentato; oppure l'algoritmo esistente è stato adottato, ma non raggiunge i risultati desiderati. | Lo studente non è stato in grado di adottare con successo alcun algoritmo, ma ha compiuto passi significativi verso la soluzione (implementazione della discretizzazione dello stato, struttura dati Q-Table, ecc.) |

---

**Disclaimer**:  
Questo documento è stato tradotto utilizzando il servizio di traduzione automatica [Co-op Translator](https://github.com/Azure/co-op-translator). Sebbene ci impegniamo per garantire l'accuratezza, si prega di notare che le traduzioni automatiche possono contenere errori o imprecisioni. Il documento originale nella sua lingua nativa dovrebbe essere considerato la fonte autorevole. Per informazioni critiche, si raccomanda una traduzione professionale effettuata da un traduttore umano. Non siamo responsabili per eventuali incomprensioni o interpretazioni errate derivanti dall'uso di questa traduzione.