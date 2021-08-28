# Regressione logistica per prevedere le categorie

![Infografica di regressione lineare e logistica](../images/logistic-linear.png)
> Infografica di [Dasani Madipalli](https://twitter.com/dasani_decoded)

## [Quiz pre-lezione](https://white-water-09ec41f0f.azurestaticapps.net/quiz/15/?loc=it)

## Introduzione

In questa lezione finale sulla Regressione, una delle tecniche _classiche_ di base di machine learning, si darà un'occhiata alla Regressione Logistica. Si dovrebbe utilizzare questa tecnica per scoprire modelli per prevedere le categorie binarie. Questa caramella è al cioccolato o no? Questa malattia è contagiosa o no? Questo cliente sceglierà questo prodotto o no?

In questa lezione, si imparerà:

- Una nuova libreria per la visualizzazione dei dati
- Tecniche per la regressione logistica

✅ Con questo [modulo di apprendimento](https://docs.microsoft.com/learn/modules/train-evaluate-classification-models?WT.mc_id=academic-15963-cxa) si potrà approfondire la comprensione del lavoro con questo tipo di regressione
## Prerequisito

Avendo lavorato con i dati della zucca, ora si ha abbastanza familiarità con essi per rendersi conto che esiste una categoria binaria con cui è possibile lavorare: `Color` (Colore).

Si costruisce un modello di regressione logistica per prevedere, date alcune variabili, di _che colore sarà probabilmente una data zucca_ (arancione 🎃 o bianca 👻).

> Perché si parla di classificazione binaria in un gruppo di lezioni sulla regressione? Solo per comodità linguistica, poiché la regressione logistica è in [realtà un metodo di classificazione](https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression), anche se lineare. Si scopriranno altri modi per classificare i dati nel prossimo gruppo di lezioni.

## Definire la domanda

Allo scopo, verrà espressa come binaria: 'Arancio' o 'Non Arancio'. C'è anche una categoria "striped" (a strisce) nell'insieme di dati, ma ci sono pochi casi, quindi non verrà presa in considerazione. Comunque scompare una volta rimossi i valori null dall'insieme di dati.

> 🎃 Fatto divertente, a volte le zucche bianche vengono chiamate zucche "fantasma" Non sono molto facili da intagliare, quindi non sono così popolari come quelle arancioni ma hanno un bell'aspetto!

## Informazioni sulla regressione logistica

La regressione logistica differisce dalla regressione lineare, che si è appresa in precedenza, in alcuni importanti modi.

### Classificazione Binaria

La regressione logistica non offre le stesse caratteristiche della regressione lineare. La prima offre una previsione su una categoria binaria ("arancione o non arancione") mentre la seconda è in grado di prevedere valori continui, ad esempio data l'origine di una zucca e il momento del raccolto, di _quanto aumenterà il suo prezzo_.

![Modello di classificazione della zucca](../images/pumpkin-classifier.png)
> Infografica di [Dasani Madipalli](https://twitter.com/dasani_decoded)
### Altre classificazioni:

Esistono altri tipi di regressione logistica, inclusi multinomiale e ordinale:

- **Multinomiale**, che implica avere più di una categoria: "arancione, bianco e a strisce".
- **Ordinale**, che coinvolge categorie ordinate, utile se si volessero ordinare i risultati in modo logico, come le zucche che sono ordinate per un numero finito di dimensioni (mini,sm,med,lg,xl,xxl).

![Regressione multinomiale contro ordinale](../images/multinomial-ordinal.png)
> Infografica di [Dasani Madipalli](https://twitter.com/dasani_decoded)

### È ancora lineare

Anche se questo tipo di Regressione riguarda le "previsioni di categoria", funziona ancora meglio quando esiste una chiara relazione lineare tra la variabile dipendente (colore) e le altre variabili indipendenti (il resto dell'insieme di dati, come il nome della città e le dimensioni) . È bene avere un'idea se c'è qualche linearità che divide queste variabili o meno.

### Le variabili NON devono essere correlate

Si ricorda come la regressione lineare ha funzionato meglio con più variabili correlate? La regressione logistica è l'opposto: le variabili non devono essere allineate. Funziona per questi dati che hanno correlazioni alquanto deboli.

### Servono molti dati puliti

La regressione logistica fornirà risultati più accurati se si utilizzano più dati; quindi si tenga a mente che, essendo l'insieme di dati sulla zucca piccolo, non è ottimale per questo compito

✅ Si pensi ai tipi di dati che si prestano bene alla regressione logistica

## Esercizio: riordinare i dati

Innanzitutto, si puliscono un po 'i dati, eliminando i valori null e selezionando solo alcune delle colonne:

1. Aggiungere il seguente codice:

   ```python
   from sklearn.preprocessing import LabelEncoder

   new_columns = ['Color','Origin','Item Size','Variety','City Name','Package']

   new_pumpkins = pumpkins.drop([c for c in pumpkins.columns if c not in new_columns], axis=1)

   new_pumpkins.dropna(inplace=True)

   new_pumpkins = new_pumpkins.apply(LabelEncoder().fit_transform)
   ```

   Si può sempre dare un'occhiata al nuovo dataframe:

   ```python
   new_pumpkins.info
   ```

### Visualizzazione - griglia affiancata

A questo punto si è caricato di nuovo il [notebook iniziale](../notebook.ipynb) con i dati della zucca e lo si è pulito in modo da preservare un insieme di dati contenente alcune variabili, incluso `Color`. Si visualizza il dataframe nel notebook utilizzando una libreria diversa: [Seaborn](https://seaborn.pydata.org/index.html), che è costruita su Matplotlib, usata in precedenza.

Seaborn offre alcuni modi accurati per visualizzare i dati. Ad esempio, si possono confrontare le distribuzioni dei dati per ogni punto in una griglia affiancata.

1. Si crea una griglia di questo tipo istanziando `PairGrid`, usando i dati della zucca `new_pumpkins`, poi chiamando `map()`:

   ```python
   import seaborn as sns

   g = sns.PairGrid(new_pumpkins)
   g.map(sns.scatterplot)
   ```

   ![Una griglia di dati visualizzati](../images/grid.png)

   Osservando i dati fianco a fianco, si può vedere come i dati di Color si riferiscono alle altre colonne.

   ✅ Data questa griglia del grafico a dispersione, quali sono alcune esplorazioni interessanti che si possono immaginare?

### Usare un grafico a sciame

Poiché Color è una categoria binaria (arancione o no), viene chiamata "dati categoriali" e richiede "un [approccio più specializzato](https://seaborn.pydata.org/tutorial/categorical.html?highlight=bar) alla visualizzazione". Esistono altri modi per visualizzare la relazione di questa categoria con altre variabili.

È possibile visualizzare le variabili fianco a fianco con i grafici di Seaborn.

1. Si provi un grafico a "sciame" per mostrare la distribuzione dei valori:

   ```python
   sns.swarmplot(x="Color", y="Item Size", data=new_pumpkins)
   ```

   ![Uno sciame di dati visualizzati](../images/swarm.png)

### Grafico violino

Un grafico di tipo "violino" è utile in quanto è possibile visualizzare facilmente il modo in cui sono distribuiti i dati nelle due categorie. I grafici di tipo violino non funzionano così bene con insieme di dati più piccoli poiché la distribuzione viene visualizzata in modo più "liscio".

1. Chiamare `catplot()` passando i parametri `x=Color`, `kind="violin"` :

   ```python
   sns.catplot(x="Color", y="Item Size",
               kind="violin", data=new_pumpkins)
   ```

   ![una tabella di un grafico di tipo violino](../images/violin.png)

   ✅ Provare a creare questo grafico e altri grafici Seaborn, utilizzando altre variabili.

Ora che si ha un'idea della relazione tra le categorie binarie di colore e il gruppo più ampio di dimensioni, si esplora la regressione logistica per determinare il probabile colore di una data zucca.

> **🧮 Mostrami la matematica**
>
> Si ricorda come la regressione lineare usava spesso i minimi quadrati ordinari per arrivare a un valore? La regressione logistica si basa sul concetto di "massima verosimiglianza" utilizzando [le funzioni sigmoidi](https://wikipedia.org/wiki/Sigmoid_function). Una "Funzione Sigmoide" su un grafico ha l'aspetto di una forma a "S". Prende un valore e lo mappa da qualche parte tra 0 e 1. La sua curva è anche chiamata "curva logistica". La sua formula si presenta così:
>
> ![funzione logistica](../images/sigmoid.png)
>
> dove il punto medio del sigmoide si trova nel punto 0 di x, L è il valore massimo della curva e k è la pendenza della curva. Se l'esito della funzione è maggiore di 0,5, all'etichetta in questione verrà assegnata la classe '1' della scelta binaria. In caso contrario, sarà classificata come '0'.

## Costruire il modello

Costruire un modello per trovare queste classificazioni binarie è sorprendentemente semplice in Scikit-learn.

1. Si selezionano le variabili da utilizzare nel modello di classificazione e si dividono gli insiemi di training e test chiamando `train_test_split()`:

   ```python
   from sklearn.model_selection import train_test_split

   Selected_features = ['Origin','Item Size','Variety','City Name','Package']

   X = new_pumpkins[Selected_features]
   y = new_pumpkins['Color']

   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

   ```

1. Ora si può addestrare il modello, chiamando `fit()` con i dati di addestramento e stamparne il risultato:

   ```python
   from sklearn.model_selection import train_test_split
   from sklearn.metrics import accuracy_score, classification_report
   from sklearn.linear_model import LogisticRegression

   model = LogisticRegression()
   model.fit(X_train, y_train)
   predictions = model.predict(X_test)

   print(classification_report(y_test, predictions))
   print('Predicted labels: ', predictions)
   print('Accuracy: ', accuracy_score(y_test, predictions))
   ```

   Si dia un'occhiata al tabellone segnapunti del modello. Non è male, considerando che si hanno solo circa 1000 righe di dati:

   ```output
                      precision    recall  f1-score   support

              0       0.85      0.95      0.90       166
              1       0.38      0.15      0.22        33

       accuracy                           0.82       199
      macro avg       0.62      0.55      0.56       199
   weighted avg       0.77      0.82      0.78       199

   Predicted labels:  [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0
    0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 1
    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1
    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    0 0 0 1 0 1 0 0 1 0 0 0 1 0]
   ```

## Migliore comprensione tramite una matrice di confusione

Sebbene si possano ottenere [i termini](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html?highlight=classification_report#sklearn.metrics.classification_report) del rapporto dei punteggi stampando gli elementi di cui sopra, si potrebbe essere in grado di comprendere più facilmente il modello utilizzando una [matrice di confusione](https://scikit-learn.org/stable/modules/model_evaluation.html#confusion-matrix) che aiuti a capire come lo stesso sta funzionando.

> 🎓 Una '[matrice di confusione](https://it.wikipedia.org/wiki/Matrice_di_confusione)' (o 'matrice di errore') è una tabella che esprime i veri contro i falsi positivi e negativi del modello, misurando così l'accuratezza delle previsioni.

1. Per utilizzare una metrica di confusione, si `chiama confusion_matrix()`:

   ```python
   from sklearn.metrics import confusion_matrix
   confusion_matrix(y_test, predictions)
   ```

   Si dia un'occhiata alla matrice di confusione del modello:

   ```output
   array([[162,   4],
          [ 33,   0]])
   ```

Cosa sta succedendo qui? Si supponga che al modello venga chiesto di classificare gli elementi tra due categorie binarie, la categoria "zucca" e la categoria "non una zucca".

- Se il modello prevede qualcosa come una zucca e appartiene alla categoria 'zucca' in realtà lo si chiama un vero positivo, mostrato dal numero in alto a sinistra.
- Se il modello prevede qualcosa come non una zucca e appartiene alla categoria 'zucca' in realtà si chiama falso positivo, mostrato dal numero in alto a destra.
- Se il modello prevede qualcosa come una zucca e appartiene alla categoria 'non-una-zucca' in realtà si chiama falso negativo, mostrato dal numero in basso a sinistra.
- Se il modello prevede qualcosa come non una zucca e appartiene alla categoria 'non-una-zucca' in realtà lo si chiama un vero negativo, mostrato dal numero in basso a destra.

Come si sarà intuito, è preferibile avere un numero maggiore di veri positivi e veri negativi e un numero inferiore di falsi positivi e falsi negativi, il che implica che il modello funziona meglio.

✅ Domanda: Secondo la matrice di confusione, come si è comportato il modello? Risposta: Non male; ci sono un buon numero di veri positivi ma anche diversi falsi negativi.

I termini visti in precedenza vengono rivisitati con l'aiuto della mappatura della matrice di confusione di TP/TN e FP/FN:

🎓 Precisione: TP/(TP + FP) La frazione di istanze rilevanti tra le istanze recuperate (ad es. quali etichette erano ben etichettate)

🎓 Richiamo: TP/(TP + FN) La frazione di istanze rilevanti che sono state recuperate, ben etichettate o meno

🎓 f1-score: (2 * precisione * richiamo)/(precisione + richiamo) Una media ponderata della precisione e del richiamo, dove il migliore è 1 e il peggiore è 0

🎓 Supporto: il numero di occorrenze di ciascuna etichetta recuperata

🎓 Accuratezza: (TP + TN)/(TP + TN + FP + FN) La percentuale di etichette prevista accuratamente per un campione.

🎓 Macro Media: il calcolo delle metriche medie non ponderate per ciascuna etichetta, senza tener conto dello squilibrio dell'etichetta.

🎓 Media ponderata: il calcolo delle metriche medie per ogni etichetta, tenendo conto dello squilibrio dell'etichetta pesandole in base al loro supporto (il numero di istanze vere per ciascuna etichetta).

✅ Si riesce a pensare a quale metrica si dovrebbe guardare se si vuole che il modello riduca il numero di falsi negativi?

## Visualizzare la curva ROC di questo modello

Questo non è un cattivo modello; la sua precisione è nell'intervallo dell'80%, quindi idealmente si potrebbe usare per prevedere il colore di una zucca dato un insieme di variabili.

Si rende un'altra visualizzazione per vedere il cosiddetto punteggio 'ROC':

```python
from sklearn.metrics import roc_curve, roc_auc_score

y_scores = model.predict_proba(X_test)
# calculate ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_scores[:,1])
sns.lineplot([0, 1], [0, 1])
sns.lineplot(fpr, tpr)
```
Usando di nuovo Seaborn, si traccia la [Caratteristica Operativa di Ricezione](https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html?highlight=roc) o il ROC del modello. Le curve ROC vengono spesso utilizzate per ottenere una visualizzazione dell'output di un classificatore in termini di veri e falsi positivi. "Le curve ROC in genere presentano un tasso di veri positivi sull'asse Y e un tasso di falsi positivi sull'asse X". Pertanto, la ripidità della curva e lo spazio tra la linea del punto medio e la curva contano: si vuole una curva che si sposti rapidamente verso l'alto e oltre la linea. In questo caso, ci sono falsi positivi con cui iniziare, quindi la linea si dirige correttamente:

![ROC](../images/ROC.png)

Infine, si usa l'[`API roc_auc_score`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html?highlight=roc_auc#sklearn.metrics.roc_auc_score) di Scikit-learn per calcolare l'effettiva "Area sotto la curva" (AUC):

```python
auc = roc_auc_score(y_test,y_scores[:,1])
print(auc)
```
Il risultato è `0.6976998904709748`. Dato che l'AUC varia da 0 a 1, si desidera un punteggio elevato, poiché un modello corretto al 100% nelle sue previsioni avrà un AUC di 1; in questo caso, il modello è _abbastanza buono_.

Nelle lezioni future sulle classificazioni si imparerà come eseguire l'iterazione per migliorare i punteggi del  modello. Ma per ora, congratulazioni! Si sono completate queste lezioni di regressione!

---
## 🚀 Sfida

C'è molto altro da svelare riguardo alla regressione logistica! Ma il modo migliore per imparare è sperimentare. Trovare un insieme di dati che si presti a questo tipo di analisi e costruire un modello con esso. Cosa si è appreso? suggerimento: provare [Kaggle](https://kaggle.com) per ottenere insiemi di dati interessanti.

## [Quiz post-lezione](https://white-water-09ec41f0f.azurestaticapps.net/quiz/16/?loc=it)

## Revisione e Auto Apprendimento

Leggere le prime pagine di [questo articolo da Stanford](https://web.stanford.edu/~jurafsky/slp3/5.pdf) su alcuni usi pratici della regressione logistica. Si pensi alle attività più adatte per l'uno o l'altro tipo di attività di regressione studiate fino a questo punto. Cosa funzionerebbe meglio?

## Compito

[Ritentare questa regressione](assignment.it.md)
