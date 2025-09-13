<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "68394b2102d3503882e5e914bd0ff5c1",
  "translation_date": "2025-08-29T22:11:12+00:00",
  "source_file": "8-Reinforcement/1-QLearning/assignment.md",
  "language_code": "it"
}
-->
# Un Mondo Più Realistico

Nella nostra situazione, Peter era in grado di muoversi quasi senza stancarsi o avere fame. In un mondo più realistico, deve sedersi e riposarsi di tanto in tanto, e anche nutrirsi. Rendiamo il nostro mondo più realistico implementando le seguenti regole:

1. Spostandosi da un luogo all'altro, Peter perde **energia** e accumula **fatica**.
2. Peter può recuperare energia mangiando mele.
3. Peter può eliminare la fatica riposandosi sotto un albero o sull'erba (cioè camminando in una posizione della mappa con un albero o dell'erba - campo verde).
4. Peter deve trovare e uccidere il lupo.
5. Per uccidere il lupo, Peter deve avere determinati livelli di energia e fatica, altrimenti perderà la battaglia.

## Istruzioni

Usa il [notebook.ipynb](notebook.ipynb) originale come punto di partenza per la tua soluzione.

Modifica la funzione di ricompensa sopra in base alle regole del gioco, esegui l'algoritmo di apprendimento per rinforzo per apprendere la migliore strategia per vincere il gioco e confronta i risultati della camminata casuale con il tuo algoritmo in termini di numero di partite vinte e perse.

> **Nota**: Nel tuo nuovo mondo, lo stato è più complesso e, oltre alla posizione di Peter, include anche i livelli di fatica ed energia. Puoi scegliere di rappresentare lo stato come una tupla (Mappa, energia, fatica), oppure definire una classe per lo stato (potresti anche volerla derivare da `Board`), o persino modificare la classe originale `Board` all'interno di [rlboard.py](../../../../8-Reinforcement/1-QLearning/rlboard.py).

Nella tua soluzione, mantieni il codice responsabile della strategia di camminata casuale e confronta i risultati del tuo algoritmo con la camminata casuale alla fine.

> **Nota**: Potresti dover regolare gli iperparametri per far funzionare il tutto, specialmente il numero di epoche. Poiché il successo del gioco (combattere il lupo) è un evento raro, puoi aspettarti tempi di addestramento molto più lunghi.

## Valutazione

| Criteri  | Esemplare                                                                                                                                                                                             | Adeguato                                                                                                                                                                                | Da Migliorare                                                                                                                              |
| -------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
|          | Viene presentato un notebook con la definizione delle nuove regole del mondo, l'algoritmo di Q-Learning e alcune spiegazioni testuali. Il Q-Learning è in grado di migliorare significativamente i risultati rispetto alla camminata casuale. | Viene presentato un notebook, il Q-Learning è implementato e migliora i risultati rispetto alla camminata casuale, ma non in modo significativo; oppure il notebook è scarsamente documentato e il codice non è ben strutturato. | Viene fatto qualche tentativo di ridefinire le regole del mondo, ma l'algoritmo di Q-Learning non funziona, o la funzione di ricompensa non è completamente definita. |

---

**Disclaimer**:  
Questo documento è stato tradotto utilizzando il servizio di traduzione automatica [Co-op Translator](https://github.com/Azure/co-op-translator). Sebbene ci impegniamo per garantire l'accuratezza, si prega di notare che le traduzioni automatiche possono contenere errori o imprecisioni. Il documento originale nella sua lingua nativa dovrebbe essere considerato la fonte autorevole. Per informazioni critiche, si raccomanda una traduzione professionale effettuata da un traduttore umano. Non siamo responsabili per eventuali incomprensioni o interpretazioni errate derivanti dall'uso di questa traduzione.