<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "69389392fa6346e0dfa30f664b7b6fec",
  "translation_date": "2025-09-06T07:34:02+00:00",
  "source_file": "1-Introduction/1-intro-to-ML/README.md",
  "language_code": "it"
}
-->
# Introduzione al machine learning

## [Quiz pre-lezione](https://ff-quizzes.netlify.app/en/ml/)

---

[![ML per principianti - Introduzione al Machine Learning per principianti](https://img.youtube.com/vi/6mSx_KJxcHI/0.jpg)](https://youtu.be/6mSx_KJxcHI "ML per principianti - Introduzione al Machine Learning per principianti")

> 🎥 Clicca sull'immagine sopra per un breve video che illustra questa lezione.

Benvenuto in questo corso sul machine learning classico per principianti! Che tu sia completamente nuovo a questo argomento o un esperto di ML che desidera ripassare un'area, siamo felici di averti con noi! Vogliamo creare un punto di partenza amichevole per il tuo studio di ML e saremmo felici di valutare, rispondere e incorporare il tuo [feedback](https://github.com/microsoft/ML-For-Beginners/discussions).

[![Introduzione al ML](https://img.youtube.com/vi/h0e2HAPTGF4/0.jpg)](https://youtu.be/h0e2HAPTGF4 "Introduzione al ML")

> 🎥 Clicca sull'immagine sopra per un video: John Guttag del MIT introduce il machine learning

---
## Iniziare con il machine learning

Prima di iniziare con questo curriculum, è necessario configurare il tuo computer e prepararlo per eseguire notebook localmente.

- **Configura il tuo computer con questi video**. Usa i seguenti link per imparare [come installare Python](https://youtu.be/CXZYvNRIAKM) sul tuo sistema e [configurare un editor di testo](https://youtu.be/EU8eayHWoZg) per lo sviluppo.
- **Impara Python**. È anche consigliato avere una conoscenza di base di [Python](https://docs.microsoft.com/learn/paths/python-language/?WT.mc_id=academic-77952-leestott), un linguaggio di programmazione utile per i data scientist che utilizziamo in questo corso.
- **Impara Node.js e JavaScript**. Utilizziamo anche JavaScript alcune volte in questo corso per costruire app web, quindi sarà necessario avere [node](https://nodejs.org) e [npm](https://www.npmjs.com/) installati, oltre a [Visual Studio Code](https://code.visualstudio.com/) disponibile sia per lo sviluppo in Python che in JavaScript.
- **Crea un account GitHub**. Dato che ci hai trovato qui su [GitHub](https://github.com), potresti già avere un account, ma se non lo hai, creane uno e poi fai un fork di questo curriculum per usarlo da solo. (Sentiti libero di darci una stella, 😊)
- **Esplora Scikit-learn**. Familiarizza con [Scikit-learn](https://scikit-learn.org/stable/user_guide.html), un insieme di librerie ML che utilizziamo in queste lezioni.

---
## Cos'è il machine learning?

Il termine 'machine learning' è uno dei più popolari e frequentemente utilizzati oggi. È molto probabile che tu abbia sentito questo termine almeno una volta se hai una certa familiarità con la tecnologia, indipendentemente dal settore in cui lavori. Tuttavia, la meccanica del machine learning è un mistero per la maggior parte delle persone. Per un principiante del machine learning, l'argomento può talvolta sembrare opprimente. Pertanto, è importante capire cosa sia realmente il machine learning e impararlo passo dopo passo, attraverso esempi pratici.

---
## La curva dell'hype

![ml hype curve](../../../../1-Introduction/1-intro-to-ML/images/hype.png)

> Google Trends mostra la recente 'curva dell'hype' del termine 'machine learning'

---
## Un universo misterioso

Viviamo in un universo pieno di misteri affascinanti. Grandi scienziati come Stephen Hawking, Albert Einstein e molti altri hanno dedicato la loro vita alla ricerca di informazioni significative che svelano i misteri del mondo che ci circonda. Questa è la condizione umana dell'apprendimento: un bambino umano impara cose nuove e scopre la struttura del suo mondo anno dopo anno mentre cresce fino all'età adulta.

---
## Il cervello di un bambino

Il cervello e i sensi di un bambino percepiscono i fatti del loro ambiente e gradualmente apprendono i modelli nascosti della vita che aiutano il bambino a creare regole logiche per identificare i modelli appresi. Il processo di apprendimento del cervello umano rende gli esseri umani la creatura vivente più sofisticata di questo mondo. Apprendere continuamente scoprendo modelli nascosti e poi innovando su quei modelli ci consente di migliorarci continuamente nel corso della nostra vita. Questa capacità di apprendimento e capacità di evoluzione è legata a un concetto chiamato [plasticità cerebrale](https://www.simplypsychology.org/brain-plasticity.html). Superficialmente, possiamo tracciare alcune somiglianze motivazionali tra il processo di apprendimento del cervello umano e i concetti di machine learning.

---
## Il cervello umano

Il [cervello umano](https://www.livescience.com/29365-human-brain.html) percepisce cose dal mondo reale, elabora le informazioni percepite, prende decisioni razionali e compie determinate azioni in base alle circostanze. Questo è ciò che chiamiamo comportarsi in modo intelligente. Quando programmiamo una replica del processo comportamentale intelligente in una macchina, si chiama intelligenza artificiale (AI).

---
## Alcuni termini

Sebbene i termini possano essere confusi, il machine learning (ML) è un importante sottoinsieme dell'intelligenza artificiale. **ML si occupa di utilizzare algoritmi specializzati per scoprire informazioni significative e trovare modelli nascosti dai dati percepiti per corroborare il processo decisionale razionale**.

---
## AI, ML, Deep Learning

![AI, ML, deep learning, data science](../../../../1-Introduction/1-intro-to-ML/images/ai-ml-ds.png)

> Un diagramma che mostra le relazioni tra AI, ML, deep learning e data science. Infografica di [Jen Looper](https://twitter.com/jenlooper) ispirata a [questa grafica](https://softwareengineering.stackexchange.com/questions/366996/distinction-between-ai-ml-neural-networks-deep-learning-and-data-mining)

---
## Concetti da trattare

In questo curriculum, tratteremo solo i concetti fondamentali del machine learning che un principiante deve conoscere. Tratteremo ciò che chiamiamo 'machine learning classico' utilizzando principalmente Scikit-learn, una libreria eccellente che molti studenti usano per imparare le basi. Per comprendere concetti più ampi di intelligenza artificiale o deep learning, è indispensabile una solida conoscenza fondamentale del machine learning, e quindi vorremmo offrirla qui.

---
## In questo corso imparerai:

- concetti fondamentali del machine learning
- la storia del ML
- ML e equità
- tecniche di regressione ML
- tecniche di classificazione ML
- tecniche di clustering ML
- tecniche di elaborazione del linguaggio naturale ML
- tecniche di previsione delle serie temporali ML
- apprendimento per rinforzo
- applicazioni reali del ML

---
## Cosa non tratteremo

- deep learning
- reti neurali
- AI

Per rendere l'esperienza di apprendimento migliore, eviteremo le complessità delle reti neurali, del 'deep learning' - costruzione di modelli a più livelli utilizzando reti neurali - e dell'AI, che discuteremo in un curriculum diverso. Offriremo anche un prossimo curriculum di data science per concentrarci su quell'aspetto di questo campo più ampio.

---
## Perché studiare il machine learning?

Il machine learning, da una prospettiva di sistemi, è definito come la creazione di sistemi automatizzati che possono apprendere modelli nascosti dai dati per aiutare a prendere decisioni intelligenti.

Questa motivazione è vagamente ispirata a come il cervello umano apprende certe cose basandosi sui dati che percepisce dal mondo esterno.

✅ Pensa per un momento perché un'azienda potrebbe voler utilizzare strategie di machine learning rispetto alla creazione di un motore basato su regole codificate.

---
## Applicazioni del machine learning

Le applicazioni del machine learning sono ormai ovunque e sono tanto ubiquitarie quanto i dati che scorrono nelle nostre società, generati dai nostri smartphone, dispositivi connessi e altri sistemi. Considerando l'immenso potenziale degli algoritmi di machine learning all'avanguardia, i ricercatori hanno esplorato la loro capacità di risolvere problemi reali multidimensionali e multidisciplinari con grandi risultati positivi.

---
## Esempi di ML applicato

**Puoi utilizzare il machine learning in molti modi**:

- Per prevedere la probabilità di una malattia dalla storia medica o dai referti di un paziente.
- Per sfruttare i dati meteorologici per prevedere eventi atmosferici.
- Per comprendere il sentimento di un testo.
- Per rilevare notizie false e fermare la diffusione di propaganda.

Finanza, economia, scienze della terra, esplorazione spaziale, ingegneria biomedica, scienze cognitive e persino campi nelle discipline umanistiche hanno adattato il machine learning per risolvere i problemi ardui e pesanti di elaborazione dei dati del loro settore.

---
## Conclusione

Il machine learning automatizza il processo di scoperta dei modelli trovando intuizioni significative dai dati reali o generati. Si è dimostrato altamente prezioso in applicazioni aziendali, sanitarie e finanziarie, tra le altre.

Nel prossimo futuro, comprendere le basi del machine learning sarà indispensabile per persone di qualsiasi settore, data la sua adozione diffusa.

---
# 🚀 Sfida

Disegna, su carta o utilizzando un'app online come [Excalidraw](https://excalidraw.com/), la tua comprensione delle differenze tra AI, ML, deep learning e data science. Aggiungi alcune idee sui problemi che ciascuna di queste tecniche è adatta a risolvere.

# [Quiz post-lezione](https://ff-quizzes.netlify.app/en/ml/)

---
# Revisione e studio autonomo

Per saperne di più su come lavorare con gli algoritmi ML nel cloud, segui questo [Percorso di apprendimento](https://docs.microsoft.com/learn/paths/create-no-code-predictive-models-azure-machine-learning/?WT.mc_id=academic-77952-leestott).

Segui un [Percorso di apprendimento](https://docs.microsoft.com/learn/modules/introduction-to-machine-learning/?WT.mc_id=academic-77952-leestott) sui fondamenti del ML.

---
# Compito

[Inizia subito](assignment.md)

---

**Disclaimer**:  
Questo documento è stato tradotto utilizzando il servizio di traduzione automatica [Co-op Translator](https://github.com/Azure/co-op-translator). Sebbene ci impegniamo per garantire l'accuratezza, si prega di notare che le traduzioni automatiche possono contenere errori o imprecisioni. Il documento originale nella sua lingua nativa dovrebbe essere considerato la fonte autorevole. Per informazioni critiche, si raccomanda una traduzione professionale effettuata da un traduttore umano. Non siamo responsabili per eventuali incomprensioni o interpretazioni errate derivanti dall'uso di questa traduzione.