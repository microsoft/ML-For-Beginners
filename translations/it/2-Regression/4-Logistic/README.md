<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "72b5bae0879baddf6aafc82bb07b8776",
  "translation_date": "2025-08-29T20:27:31+00:00",
  "source_file": "2-Regression/4-Logistic/README.md",
  "language_code": "it"
}
-->
# Regressione logistica per prevedere categorie

![Infografica regressione logistica vs lineare](../../../../translated_images/linear-vs-logistic.ba180bf95e7ee66721ba10ebf2dac2666acbd64a88b003c83928712433a13c7d.it.png)

## [Quiz pre-lezione](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/15/)

> ### [Questa lezione è disponibile in R!](../../../../2-Regression/4-Logistic/solution/R/lesson_4.html)

## Introduzione

In questa ultima lezione sulla regressione, una delle tecniche _classiche_ di ML di base, esamineremo la regressione logistica. Questa tecnica viene utilizzata per scoprire schemi e prevedere categorie binarie. Questa caramella è cioccolato o no? Questa malattia è contagiosa o no? Questo cliente sceglierà questo prodotto o no?

In questa lezione imparerai:

- Una nuova libreria per la visualizzazione dei dati
- Tecniche per la regressione logistica

✅ Approfondisci la tua comprensione di questo tipo di regressione in questo [modulo di apprendimento](https://docs.microsoft.com/learn/modules/train-evaluate-classification-models?WT.mc_id=academic-77952-leestott)

## Prerequisiti

Avendo lavorato con i dati delle zucche, ora siamo abbastanza familiari con essi da riconoscere che c'è una categoria binaria su cui possiamo lavorare: `Color`.

Costruiamo un modello di regressione logistica per prevedere, date alcune variabili, _di che colore è probabile che sia una determinata zucca_ (arancione 🎃 o bianca 👻).

> Perché stiamo parlando di classificazione binaria in una lezione sulla regressione? Solo per comodità linguistica, poiché la regressione logistica è [in realtà un metodo di classificazione](https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression), sebbene basato su un approccio lineare. Scopri altri modi per classificare i dati nel prossimo gruppo di lezioni.

## Definire la domanda

Per i nostri scopi, esprimeremo questa domanda come un binario: 'Bianca' o 'Non Bianca'. Nel nostro dataset c'è anche una categoria 'a strisce', ma ci sono pochi esempi di essa, quindi non la utilizzeremo. Comunque, scompare una volta che rimuoviamo i valori nulli dal dataset.

> 🎃 Curiosità: a volte chiamiamo le zucche bianche 'zucche fantasma'. Non sono molto facili da intagliare, quindi non sono popolari come quelle arancioni, ma hanno un aspetto interessante! Potremmo anche riformulare la nostra domanda come: 'Fantasma' o 'Non Fantasma'. 👻

## Sulla regressione logistica

La regressione logistica differisce dalla regressione lineare, che hai imparato in precedenza, in alcuni modi importanti.

[![ML per principianti - Comprendere la regressione logistica per la classificazione](https://img.youtube.com/vi/KpeCT6nEpBY/0.jpg)](https://youtu.be/KpeCT6nEpBY "ML per principianti - Comprendere la regressione logistica per la classificazione")

> 🎥 Clicca sull'immagine sopra per una breve panoramica della regressione logistica.

### Classificazione binaria

La regressione logistica non offre le stesse funzionalità della regressione lineare. La prima offre una previsione su una categoria binaria ("bianca o non bianca"), mentre la seconda è in grado di prevedere valori continui, ad esempio, dato l'origine di una zucca e il momento del raccolto, _quanto aumenterà il suo prezzo_.

![Modello di classificazione delle zucche](../../../../translated_images/pumpkin-classifier.562771f104ad5436b87d1c67bca02a42a17841133556559325c0a0e348e5b774.it.png)
> Infografica di [Dasani Madipalli](https://twitter.com/dasani_decoded)

### Altre classificazioni

Esistono altri tipi di regressione logistica, tra cui multinomiale e ordinale:

- **Multinomiale**, che coinvolge più di una categoria - "Arancione, Bianca e a Strisce".
- **Ordinale**, che coinvolge categorie ordinate, utile se volessimo ordinare i nostri risultati logicamente, come le nostre zucche ordinate per un numero finito di dimensioni (mini, sm, med, lg, xl, xxl).

![Regressione multinomiale vs ordinale](../../../../translated_images/multinomial-vs-ordinal.36701b4850e37d86c9dd49f7bef93a2f94dbdb8fe03443eb68f0542f97f28f29.it.png)

### Le variabili NON devono essere correlate

Ricordi come la regressione lineare funzionava meglio con variabili più correlate? La regressione logistica è l'opposto: le variabili non devono essere allineate. Questo funziona per questi dati che hanno correlazioni piuttosto deboli.

### Hai bisogno di molti dati puliti

La regressione logistica darà risultati più accurati se utilizzi più dati; il nostro piccolo dataset non è ottimale per questo compito, quindi tienilo a mente.

[![ML per principianti - Analisi e preparazione dei dati per la regressione logistica](https://img.youtube.com/vi/B2X4H9vcXTs/0.jpg)](https://youtu.be/B2X4H9vcXTs "ML per principianti - Analisi e preparazione dei dati per la regressione logistica")

> 🎥 Clicca sull'immagine sopra per una breve panoramica sulla preparazione dei dati per la regressione lineare.

✅ Pensa ai tipi di dati che si prestano bene alla regressione logistica.

## Esercizio - pulire i dati

Per prima cosa, pulisci un po' i dati, eliminando i valori nulli e selezionando solo alcune colonne:

1. Aggiungi il seguente codice:

    ```python
  
    columns_to_select = ['City Name','Package','Variety', 'Origin','Item Size', 'Color']
    pumpkins = full_pumpkins.loc[:, columns_to_select]

    pumpkins.dropna(inplace=True)
    ```

    Puoi sempre dare un'occhiata al tuo nuovo dataframe:

    ```python
    pumpkins.info
    ```

### Visualizzazione - grafico categorico

A questo punto hai caricato il [notebook iniziale](./notebook.ipynb) con i dati delle zucche e li hai puliti per preservare un dataset contenente alcune variabili, inclusa `Color`. Visualizziamo il dataframe nel notebook utilizzando una libreria diversa: [Seaborn](https://seaborn.pydata.org/index.html), che è costruita su Matplotlib che abbiamo usato in precedenza.

Seaborn offre modi interessanti per visualizzare i tuoi dati. Ad esempio, puoi confrontare le distribuzioni dei dati per ogni `Variety` e `Color` in un grafico categorico.

1. Crea un grafico di questo tipo utilizzando la funzione `catplot`, usando i dati delle zucche `pumpkins` e specificando una mappatura dei colori per ogni categoria di zucca (arancione o bianca):

    ```python
    import seaborn as sns
    
    palette = {
    'ORANGE': 'orange',
    'WHITE': 'wheat',
    }

    sns.catplot(
    data=pumpkins, y="Variety", hue="Color", kind="count",
    palette=palette, 
    )
    ```

    ![Una griglia di dati visualizzati](../../../../translated_images/pumpkins_catplot_1.c55c409b71fea2ecc01921e64b91970542101f90bcccfa4aa3a205db8936f48b.it.png)

    Osservando i dati, puoi vedere come i dati di Color si relazionano a Variety.

    ✅ Dato questo grafico categorico, quali esplorazioni interessanti puoi immaginare?

### Pre-elaborazione dei dati: codifica delle caratteristiche e delle etichette

Il nostro dataset delle zucche contiene valori stringa per tutte le sue colonne. Lavorare con dati categorici è intuitivo per gli esseri umani, ma non per le macchine. Gli algoritmi di machine learning funzionano bene con i numeri. Ecco perché la codifica è un passaggio molto importante nella fase di pre-elaborazione dei dati, poiché ci consente di trasformare i dati categorici in dati numerici, senza perdere alcuna informazione. Una buona codifica porta alla costruzione di un buon modello.

Per la codifica delle caratteristiche ci sono due tipi principali di encoder:

1. Ordinal encoder: è adatto per variabili ordinali, che sono variabili categoriche i cui dati seguono un ordine logico, come la colonna `Item Size` nel nostro dataset. Crea una mappatura tale che ogni categoria sia rappresentata da un numero, che è l'ordine della categoria nella colonna.

    ```python
    from sklearn.preprocessing import OrdinalEncoder

    item_size_categories = [['sml', 'med', 'med-lge', 'lge', 'xlge', 'jbo', 'exjbo']]
    ordinal_features = ['Item Size']
    ordinal_encoder = OrdinalEncoder(categories=item_size_categories)
    ```

2. Categorical encoder: è adatto per variabili nominali, che sono variabili categoriche i cui dati non seguono un ordine logico, come tutte le caratteristiche diverse da `Item Size` nel nostro dataset. È una codifica one-hot, il che significa che ogni categoria è rappresentata da una colonna binaria: la variabile codificata è uguale a 1 se la zucca appartiene a quella Variety e 0 altrimenti.

    ```python
    from sklearn.preprocessing import OneHotEncoder

    categorical_features = ['City Name', 'Package', 'Variety', 'Origin']
    categorical_encoder = OneHotEncoder(sparse_output=False)
    ```

Poi, `ColumnTransformer` viene utilizzato per combinare più encoder in un unico passaggio e applicarli alle colonne appropriate.

```python
    from sklearn.compose import ColumnTransformer
    
    ct = ColumnTransformer(transformers=[
        ('ord', ordinal_encoder, ordinal_features),
        ('cat', categorical_encoder, categorical_features)
        ])
    
    ct.set_output(transform='pandas')
    encoded_features = ct.fit_transform(pumpkins)
```

D'altra parte, per codificare l'etichetta, utilizziamo la classe `LabelEncoder` di scikit-learn, che è una classe di utilità per normalizzare le etichette in modo che contengano solo valori tra 0 e n_classes-1 (qui, 0 e 1).

```python
    from sklearn.preprocessing import LabelEncoder

    label_encoder = LabelEncoder()
    encoded_label = label_encoder.fit_transform(pumpkins['Color'])
```

Una volta che abbiamo codificato le caratteristiche e l'etichetta, possiamo unirle in un nuovo dataframe `encoded_pumpkins`.

```python
    encoded_pumpkins = encoded_features.assign(Color=encoded_label)
```

✅ Quali sono i vantaggi di utilizzare un ordinal encoder per la colonna `Item Size`?

### Analizzare le relazioni tra variabili

Ora che abbiamo pre-elaborato i nostri dati, possiamo analizzare le relazioni tra le caratteristiche e l'etichetta per avere un'idea di quanto bene il modello sarà in grado di prevedere l'etichetta date le caratteristiche. 

Il modo migliore per eseguire questo tipo di analisi è rappresentare i dati graficamente. Utilizzeremo nuovamente la funzione `catplot` di Seaborn per visualizzare le relazioni tra `Item Size`, `Variety` e `Color` in un grafico categorico. Per rappresentare meglio i dati utilizzeremo la colonna codificata `Item Size` e la colonna non codificata `Variety`.

```python
    palette = {
    'ORANGE': 'orange',
    'WHITE': 'wheat',
    }
    pumpkins['Item Size'] = encoded_pumpkins['ord__Item Size']

    g = sns.catplot(
        data=pumpkins,
        x="Item Size", y="Color", row='Variety',
        kind="box", orient="h",
        sharex=False, margin_titles=True,
        height=1.8, aspect=4, palette=palette,
    )
    g.set(xlabel="Item Size", ylabel="").set(xlim=(0,6))
    g.set_titles(row_template="{row_name}")
```

![Un catplot di dati visualizzati](../../../../translated_images/pumpkins_catplot_2.87a354447880b3889278155957f8f60dd63db4598de5a6d0fda91c334d31f9f1.it.png)

### Utilizzare un swarm plot

Poiché Color è una categoria binaria (Bianca o Non Bianca), necessita di 'un [approccio specializzato](https://seaborn.pydata.org/tutorial/categorical.html?highlight=bar) per la visualizzazione'. Esistono altri modi per visualizzare la relazione di questa categoria con altre variabili.

Puoi visualizzare le variabili fianco a fianco con i grafici di Seaborn.

1. Prova un grafico 'swarm' per mostrare la distribuzione dei valori:

    ```python
    palette = {
    0: 'orange',
    1: 'wheat'
    }
    sns.swarmplot(x="Color", y="ord__Item Size", data=encoded_pumpkins, palette=palette)
    ```

    ![Un swarm di dati visualizzati](../../../../translated_images/swarm_2.efeacfca536c2b577dc7b5f8891f28926663fbf62d893ab5e1278ae734ca104e.it.png)

**Attenzione**: il codice sopra potrebbe generare un avviso, poiché Seaborn non riesce a rappresentare una quantità così elevata di punti dati in un grafico swarm. Una possibile soluzione è ridurre la dimensione del marker, utilizzando il parametro 'size'. Tuttavia, tieni presente che ciò influisce sulla leggibilità del grafico.

> **🧮 Mostrami la matematica**
>
> La regressione logistica si basa sul concetto di 'massima verosimiglianza' utilizzando [funzioni sigmoid](https://wikipedia.org/wiki/Sigmoid_function). Una 'Funzione Sigmoid' su un grafico ha una forma a 'S'. Prende un valore e lo mappa tra 0 e 1. La sua curva è anche chiamata 'curva logistica'. La sua formula è la seguente:
>
> ![funzione logistica](../../../../translated_images/sigmoid.8b7ba9d095c789cf72780675d0d1d44980c3736617329abfc392dfc859799704.it.png)
>
> dove il punto medio della sigmoid si trova al punto 0 di x, L è il valore massimo della curva e k è la pendenza della curva. Se il risultato della funzione è maggiore di 0.5, l'etichetta in questione verrà assegnata alla classe '1' della scelta binaria. In caso contrario, verrà classificata come '0'.

## Costruire il tuo modello

Costruire un modello per trovare queste classificazioni binarie è sorprendentemente semplice in Scikit-learn.

[![ML per principianti - Regressione logistica per la classificazione dei dati](https://img.youtube.com/vi/MmZS2otPrQ8/0.jpg)](https://youtu.be/MmZS2otPrQ8 "ML per principianti - Regressione logistica per la classificazione dei dati")

> 🎥 Clicca sull'immagine sopra per una breve panoramica sulla costruzione di un modello di regressione lineare.

1. Seleziona le variabili che vuoi utilizzare nel tuo modello di classificazione e dividi i set di addestramento e test chiamando `train_test_split()`:

    ```python
    from sklearn.model_selection import train_test_split
    
    X = encoded_pumpkins[encoded_pumpkins.columns.difference(['Color'])]
    y = encoded_pumpkins['Color']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    
    ```

2. Ora puoi addestrare il tuo modello, chiamando `fit()` con i tuoi dati di addestramento, e stampare il risultato:

    ```python
    from sklearn.metrics import f1_score, classification_report 
    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    print(classification_report(y_test, predictions))
    print('Predicted labels: ', predictions)
    print('F1-score: ', f1_score(y_test, predictions))
    ```

    Dai un'occhiata al punteggio del tuo modello. Non è male, considerando che hai solo circa 1000 righe di dati:

    ```output
                       precision    recall  f1-score   support
    
                    0       0.94      0.98      0.96       166
                    1       0.85      0.67      0.75        33
    
        accuracy                                0.92       199
        macro avg           0.89      0.82      0.85       199
        weighted avg        0.92      0.92      0.92       199
    
        Predicted labels:  [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 1 0 0 0 0 0 0 0 0 1 0 0 0 0
        0 0 0 0 0 1 0 1 0 0 1 0 0 0 0 0 1 0 1 0 1 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0
        1 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 1 0 1 0 0 0 0 0 0 0 1 0
        0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 1 0 0 0 1 1 0
        0 0 0 0 1 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1
        0 0 0 1 0 0 0 0 0 0 0 0 1 1]
        F1-score:  0.7457627118644068
    ```

## Migliore comprensione tramite una matrice di confusione

Mentre puoi ottenere un rapporto sul punteggio [termini](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html?highlight=classification_report#sklearn.metrics.classification_report) stampando gli elementi sopra, potresti essere in grado di comprendere meglio il tuo modello utilizzando una [matrice di confusione](https://scikit-learn.org/stable/modules/model_evaluation.html#confusion-matrix) per aiutarci a capire come il modello sta funzionando.

> 🎓 Una '[matrice di confusione](https://wikipedia.org/wiki/Confusion_matrix)' (o 'matrice di errore') è una tabella che esprime i veri e falsi positivi e negativi del tuo modello, valutando così l'accuratezza delle previsioni.

1. Per utilizzare una matrice di confusione, chiama `confusion_matrix()`:

    ```python
    from sklearn.metrics import confusion_matrix
    confusion_matrix(y_test, predictions)
    ```

    Dai un'occhiata alla matrice di confusione del tuo modello:

    ```output
    array([[162,   4],
           [ 11,  22]])
    ```

In Scikit-learn, le righe (asse 0) sono le etichette reali e le colonne (asse 1) sono le etichette previste.

|       |   0   |   1   |
| :---: | :---: | :---: |
|   0   |  TN   |  FP   |
|   1   |  FN   |  TP   |

Cosa sta succedendo qui? Supponiamo che il nostro modello sia chiamato a classificare le zucche tra due categorie binarie, categoria 'bianca' e categoria 'non bianca'.

- Se il tuo modello prevede una zucca come non bianca e appartiene alla categoria 'non bianca' nella realtà, la chiamiamo vero negativo, mostrato dal numero in alto a sinistra.
- Se il tuo modello prevede una zucca come bianca e appartiene alla categoria 'non bianca' nella realtà, la chiamiamo falso negativo, mostrato dal numero in basso a sinistra.
- Se il tuo modello prevede una zucca come non bianca e appartiene alla categoria 'bianca' nella realtà, la chiamiamo falso positivo, mostrato dal numero in alto a destra.
- Se il tuo modello prevede una zucca come bianca e appartiene alla categoria 'bianca' nella realtà, la chiamiamo vero positivo, mostrato dal numero in basso a destra.

Come avrai intuito, è preferibile avere un numero maggiore di veri positivi e veri negativi e un numero minore di falsi positivi e falsi negativi, il che implica che il modello funziona meglio.
Come si collega la matrice di confusione a precision e recall? Ricorda, il report di classificazione mostrato sopra indicava una precisione (0.85) e un recall (0.67).

Precision = tp / (tp + fp) = 22 / (22 + 4) = 0.8461538461538461

Recall = tp / (tp + fn) = 22 / (22 + 11) = 0.6666666666666666

✅ D: Secondo la matrice di confusione, come si è comportato il modello? R: Non male; ci sono un buon numero di veri negativi, ma anche alcuni falsi negativi.

Rivediamo i termini che abbiamo visto in precedenza con l'aiuto della mappatura TP/TN e FP/FN della matrice di confusione:

🎓 Precision: TP/(TP + FP) La frazione di istanze rilevanti tra quelle recuperate (ad esempio, quali etichette sono state ben classificate).

🎓 Recall: TP/(TP + FN) La frazione di istanze rilevanti che sono state recuperate, indipendentemente dal fatto che siano state ben classificate o meno.

🎓 f1-score: (2 * precision * recall)/(precision + recall) Una media ponderata di precision e recall, con il valore migliore pari a 1 e il peggiore pari a 0.

🎓 Support: Il numero di occorrenze di ciascuna etichetta recuperata.

🎓 Accuracy: (TP + TN)/(TP + TN + FP + FN) La percentuale di etichette previste correttamente per un campione.

🎓 Macro Avg: Il calcolo della media non ponderata delle metriche per ciascuna etichetta, senza considerare lo sbilanciamento delle etichette.

🎓 Weighted Avg: Il calcolo della media delle metriche per ciascuna etichetta, tenendo conto dello sbilanciamento delle etichette pesandole in base al loro supporto (il numero di istanze vere per ciascuna etichetta).

✅ Riesci a pensare a quale metrica dovresti prestare attenzione se vuoi che il tuo modello riduca il numero di falsi negativi?

## Visualizzare la curva ROC di questo modello

[![ML per principianti - Analisi delle prestazioni della regressione logistica con le curve ROC](https://img.youtube.com/vi/GApO575jTA0/0.jpg)](https://youtu.be/GApO575jTA0 "ML per principianti - Analisi delle prestazioni della regressione logistica con le curve ROC")

> 🎥 Clicca sull'immagine sopra per una breve panoramica sulle curve ROC

Facciamo un'altra visualizzazione per osservare la cosiddetta curva 'ROC':

```python
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib
import matplotlib.pyplot as plt
%matplotlib inline

y_scores = model.predict_proba(X_test)
fpr, tpr, thresholds = roc_curve(y_test, y_scores[:,1])

fig = plt.figure(figsize=(6, 6))
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()
```

Utilizzando Matplotlib, traccia la [Receiving Operating Characteristic](https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html?highlight=roc) o ROC del modello. Le curve ROC sono spesso utilizzate per ottenere una visione dell'output di un classificatore in termini di veri positivi rispetto ai falsi positivi. "Le curve ROC presentano tipicamente il tasso di veri positivi sull'asse Y e il tasso di falsi positivi sull'asse X." Pertanto, la pendenza della curva e lo spazio tra la linea mediana e la curva sono importanti: si desidera una curva che salga rapidamente e superi la linea. Nel nostro caso, ci sono falsi positivi all'inizio, e poi la linea sale e supera correttamente:

![ROC](../../../../translated_images/ROC_2.777f20cdfc4988ca683ade6850ac832cb70c96c12f1b910d294f270ef36e1a1c.it.png)

Infine, utilizza l'API [`roc_auc_score` di Scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html?highlight=roc_auc#sklearn.metrics.roc_auc_score) per calcolare l'effettiva 'Area Sotto la Curva' (AUC):

```python
auc = roc_auc_score(y_test,y_scores[:,1])
print(auc)
```
Il risultato è `0.9749908725812341`. Considerando che l'AUC varia da 0 a 1, si desidera un punteggio elevato, poiché un modello che è corretto al 100% nelle sue previsioni avrà un AUC pari a 1; in questo caso, il modello è _abbastanza buono_.

Nelle lezioni future sulla classificazione, imparerai come iterare per migliorare i punteggi del tuo modello. Ma per ora, congratulazioni! Hai completato queste lezioni sulla regressione!

---
## 🚀Sfida

C'è molto altro da esplorare sulla regressione logistica! Ma il modo migliore per imparare è sperimentare. Trova un dataset che si presti a questo tipo di analisi e costruisci un modello con esso. Cosa impari? suggerimento: prova [Kaggle](https://www.kaggle.com/search?q=logistic+regression+datasets) per dataset interessanti.

## [Quiz post-lezione](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/16/)

## Revisione e Studio Autonomo

Leggi le prime pagine di [questo articolo di Stanford](https://web.stanford.edu/~jurafsky/slp3/5.pdf) su alcuni usi pratici della regressione logistica. Rifletti sui compiti che sono più adatti a uno o all'altro tipo di regressione tra quelli che abbiamo studiato finora. Quale funzionerebbe meglio?

## Compito

[Riprova questa regressione](assignment.md)

---

**Disclaimer**:  
Questo documento è stato tradotto utilizzando il servizio di traduzione automatica [Co-op Translator](https://github.com/Azure/co-op-translator). Sebbene ci impegniamo per garantire l'accuratezza, si prega di notare che le traduzioni automatiche potrebbero contenere errori o imprecisioni. Il documento originale nella sua lingua nativa dovrebbe essere considerato la fonte autorevole. Per informazioni critiche, si raccomanda una traduzione professionale eseguita da un traduttore umano. Non siamo responsabili per eventuali fraintendimenti o interpretazioni errate derivanti dall'uso di questa traduzione.