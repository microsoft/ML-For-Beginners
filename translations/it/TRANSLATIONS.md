# Contribuire traducendo le lezioni

Siamo lieti di accogliere traduzioni per le lezioni di questo curriculum!
## Linee guida

Ci sono cartelle in ogni cartella di lezione e cartella di introduzione alla lezione che contengono i file markdown tradotti.

> Nota, per favore non tradurre alcun codice nei file di esempio di codice; le uniche cose da tradurre sono README, compiti e quiz. Grazie!

I file tradotti dovrebbero seguire questa convenzione di denominazione:

**README._[language]_.md**

dove _[language]_ è un'abbreviazione di due lettere della lingua secondo lo standard ISO 639-1 (ad esempio `README.es.md` per lo spagnolo e `README.nl.md` per l'olandese).

**assignment._[language]_.md**

Simile ai Readme, per favore traduci anche i compiti.

> Importante: quando traduci il testo in questo repository, assicurati di non utilizzare la traduzione automatica. Verificheremo le traduzioni tramite la comunità, quindi per favore offriti volontario per le traduzioni solo nelle lingue in cui sei competente.

**Quiz**

1. Aggiungi la tua traduzione all'app quiz aggiungendo un file qui: https://github.com/microsoft/ML-For-Beginners/tree/main/quiz-app/src/assets/translations, con la corretta convenzione di denominazione (en.json, fr.json). **Per favore non localizzare le parole 'true' o 'false' comunque. grazie!**

2. Aggiungi il codice della tua lingua al menu a tendina nel file App.vue dell'app quiz.

3. Modifica il [file index.js delle traduzioni](https://github.com/microsoft/ML-For-Beginners/blob/main/quiz-app/src/assets/translations/index.js) dell'app quiz per aggiungere la tua lingua.

4. Infine, modifica TUTTI i link dei quiz nei tuoi file README.md tradotti per puntare direttamente al tuo quiz tradotto: https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/1 diventa https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/1?loc=id

**GRAZIE**

Apprezziamo davvero i tuoi sforzi!

**Disclaimer**: 
Questo documento è stato tradotto utilizzando servizi di traduzione basati su intelligenza artificiale. Sebbene ci impegniamo per l'accuratezza, si prega di notare che le traduzioni automatiche possono contenere errori o imprecisioni. Il documento originale nella sua lingua madre dovrebbe essere considerato la fonte autorevole. Per informazioni critiche, si raccomanda una traduzione professionale umana. Non siamo responsabili per eventuali malintesi o interpretazioni errate derivanti dall'uso di questa traduzione.