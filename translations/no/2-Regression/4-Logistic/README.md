<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "abf86d845c84330bce205a46b382ec88",
  "translation_date": "2025-09-05T21:11:16+00:00",
  "source_file": "2-Regression/4-Logistic/README.md",
  "language_code": "no"
}
-->
# Logistisk regresjon for å forutsi kategorier

![Infografikk om logistisk vs. lineær regresjon](../../../../2-Regression/4-Logistic/images/linear-vs-logistic.png)

## [Quiz før forelesning](https://ff-quizzes.netlify.app/en/ml/)

> ### [Denne leksjonen er tilgjengelig i R!](../../../../2-Regression/4-Logistic/solution/R/lesson_4.html)

## Introduksjon

I denne siste leksjonen om regresjon, en av de grunnleggende _klassiske_ ML-teknikkene, skal vi se nærmere på logistisk regresjon. Du kan bruke denne teknikken til å oppdage mønstre for å forutsi binære kategorier. Er dette godteri sjokolade eller ikke? Er denne sykdommen smittsom eller ikke? Vil denne kunden velge dette produktet eller ikke?

I denne leksjonen vil du lære:

- Et nytt bibliotek for datavisualisering
- Teknikker for logistisk regresjon

✅ Fordyp deg i å arbeide med denne typen regresjon i dette [Learn-modulet](https://docs.microsoft.com/learn/modules/train-evaluate-classification-models?WT.mc_id=academic-77952-leestott)

## Forutsetning

Etter å ha jobbet med gresskar-dataene, er vi nå kjent nok med dem til å innse at det finnes én binær kategori vi kan jobbe med: `Color`.

La oss bygge en logistisk regresjonsmodell for å forutsi, gitt noen variabler, _hvilken farge et gitt gresskar sannsynligvis har_ (oransje 🎃 eller hvit 👻).

> Hvorfor snakker vi om binær klassifisering i en leksjon om regresjon? Bare av språklig bekvemmelighet, siden logistisk regresjon [egentlig er en klassifiseringsmetode](https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression), om enn en lineær-basert en. Lær om andre måter å klassifisere data på i neste leksjonsgruppe.

## Definer spørsmålet

For vårt formål vil vi uttrykke dette som en binær: 'Hvit' eller 'Ikke hvit'. Det finnes også en 'stripet' kategori i datasettet vårt, men det er få forekomster av den, så vi vil ikke bruke den. Den forsvinner uansett når vi fjerner nullverdier fra datasettet.

> 🎃 Fun fact: Vi kaller noen ganger hvite gresskar for 'spøkelsesgresskar'. De er ikke veldig lette å skjære ut, så de er ikke like populære som de oransje, men de ser kule ut! Så vi kunne også formulert spørsmålet vårt som: 'Spøkelse' eller 'Ikke spøkelse'. 👻

## Om logistisk regresjon

Logistisk regresjon skiller seg fra lineær regresjon, som du lærte om tidligere, på noen viktige måter.

[![ML for nybegynnere - Forstå logistisk regresjon for maskinlæringsklassifisering](https://img.youtube.com/vi/KpeCT6nEpBY/0.jpg)](https://youtu.be/KpeCT6nEpBY "ML for nybegynnere - Forstå logistisk regresjon for maskinlæringsklassifisering")

> 🎥 Klikk på bildet ovenfor for en kort videooversikt over logistisk regresjon.

### Binær klassifisering

Logistisk regresjon tilbyr ikke de samme funksjonene som lineær regresjon. Den førstnevnte gir en prediksjon om en binær kategori ("hvit eller ikke hvit"), mens den sistnevnte er i stand til å forutsi kontinuerlige verdier, for eksempel gitt opprinnelsen til et gresskar og tidspunktet for innhøstingen, _hvor mye prisen vil stige_.

![Gresskar klassifiseringsmodell](../../../../2-Regression/4-Logistic/images/pumpkin-classifier.png)
> Infografikk av [Dasani Madipalli](https://twitter.com/dasani_decoded)

### Andre klassifiseringer

Det finnes andre typer logistisk regresjon, inkludert multinomial og ordinal:

- **Multinomial**, som innebærer å ha mer enn én kategori - "Oransje, Hvit og Stripet".
- **Ordinal**, som innebærer ordnede kategorier, nyttig hvis vi ønsket å ordne resultatene våre logisk, som våre gresskar som er ordnet etter et begrenset antall størrelser (mini, sm, med, lg, xl, xxl).

![Multinomial vs ordinal regresjon](../../../../2-Regression/4-Logistic/images/multinomial-vs-ordinal.png)

### Variabler trenger IKKE å korrelere

Husker du hvordan lineær regresjon fungerte bedre med mer korrelerte variabler? Logistisk regresjon er det motsatte - variablene trenger ikke å samsvare. Dette fungerer for disse dataene som har noe svake korrelasjoner.

### Du trenger mye rene data

Logistisk regresjon gir mer nøyaktige resultater hvis du bruker mer data; vårt lille datasett er ikke optimalt for denne oppgaven, så husk det.

[![ML for nybegynnere - Dataanalyse og forberedelse for logistisk regresjon](https://img.youtube.com/vi/B2X4H9vcXTs/0.jpg)](https://youtu.be/B2X4H9vcXTs "ML for nybegynnere - Dataanalyse og forberedelse for logistisk regresjon")

> 🎥 Klikk på bildet ovenfor for en kort videooversikt over forberedelse av data for lineær regresjon

✅ Tenk på hvilke typer data som egner seg godt for logistisk regresjon

## Øvelse - rydd opp i dataene

Først, rydd opp i dataene litt, fjern nullverdier og velg bare noen av kolonnene:

1. Legg til følgende kode:

    ```python
  
    columns_to_select = ['City Name','Package','Variety', 'Origin','Item Size', 'Color']
    pumpkins = full_pumpkins.loc[:, columns_to_select]

    pumpkins.dropna(inplace=True)
    ```

    Du kan alltid ta en titt på din nye dataframe:

    ```python
    pumpkins.info
    ```

### Visualisering - kategorisk plott

Nå har du lastet opp [startnotatboken](../../../../2-Regression/4-Logistic/notebook.ipynb) med gresskar-data igjen og ryddet den slik at du har et datasett som inneholder noen få variabler, inkludert `Color`. La oss visualisere dataene i notatboken ved hjelp av et annet bibliotek: [Seaborn](https://seaborn.pydata.org/index.html), som er bygget på Matplotlib som vi brukte tidligere.

Seaborn tilbyr noen smarte måter å visualisere dataene dine på. For eksempel kan du sammenligne distribusjoner av dataene for hver `Variety` og `Color` i et kategorisk plott.

1. Lag et slikt plott ved å bruke funksjonen `catplot`, med gresskar-dataene `pumpkins`, og spesifisere en fargekartlegging for hver gresskarkategori (oransje eller hvit):

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

    ![Et rutenett med visualiserte data](../../../../2-Regression/4-Logistic/images/pumpkins_catplot_1.png)

    Ved å observere dataene kan du se hvordan `Color`-dataene relaterer seg til `Variety`.

    ✅ Gitt dette kategoriske plottet, hvilke interessante utforskninger kan du forestille deg?

### Datapreprosessering: funksjons- og etikettkoding
Datasettet vårt inneholder strengverdier for alle kolonnene. Å jobbe med kategoriske data er intuitivt for mennesker, men ikke for maskiner. Maskinlæringsalgoritmer fungerer godt med tall. Derfor er koding et veldig viktig steg i datapreprosesseringen, siden det lar oss gjøre kategoriske data om til numeriske data, uten å miste informasjon. God koding fører til å bygge en god modell.

For funksjonskoding finnes det to hovedtyper av kodere:

1. Ordinal koder: passer godt for ordinale variabler, som er kategoriske variabler der dataene følger en logisk rekkefølge, som kolonnen `Item Size` i datasettet vårt. Den lager en kartlegging slik at hver kategori representeres av et tall, som er rekkefølgen til kategorien i kolonnen.

    ```python
    from sklearn.preprocessing import OrdinalEncoder

    item_size_categories = [['sml', 'med', 'med-lge', 'lge', 'xlge', 'jbo', 'exjbo']]
    ordinal_features = ['Item Size']
    ordinal_encoder = OrdinalEncoder(categories=item_size_categories)
    ```

2. Kategorisk koder: passer godt for nominelle variabler, som er kategoriske variabler der dataene ikke følger en logisk rekkefølge, som alle funksjonene bortsett fra `Item Size` i datasettet vårt. Det er en one-hot encoding, som betyr at hver kategori representeres av en binær kolonne: den kodede variabelen er lik 1 hvis gresskaret tilhører den `Variety` og 0 ellers.

    ```python
    from sklearn.preprocessing import OneHotEncoder

    categorical_features = ['City Name', 'Package', 'Variety', 'Origin']
    categorical_encoder = OneHotEncoder(sparse_output=False)
    ```
Deretter brukes `ColumnTransformer` til å kombinere flere kodere i ett enkelt steg og anvende dem på de riktige kolonnene.

```python
    from sklearn.compose import ColumnTransformer
    
    ct = ColumnTransformer(transformers=[
        ('ord', ordinal_encoder, ordinal_features),
        ('cat', categorical_encoder, categorical_features)
        ])
    
    ct.set_output(transform='pandas')
    encoded_features = ct.fit_transform(pumpkins)
```
For å kode etiketten bruker vi scikit-learn-klassen `LabelEncoder`, som er en hjelpeklasse for å normalisere etiketter slik at de bare inneholder verdier mellom 0 og n_classes-1 (her, 0 og 1).

```python
    from sklearn.preprocessing import LabelEncoder

    label_encoder = LabelEncoder()
    encoded_label = label_encoder.fit_transform(pumpkins['Color'])
```
Når vi har kodet funksjonene og etiketten, kan vi slå dem sammen til en ny dataframe `encoded_pumpkins`.

```python
    encoded_pumpkins = encoded_features.assign(Color=encoded_label)
```
✅ Hva er fordelene med å bruke en ordinal koder for kolonnen `Item Size`?

### Analyser forholdet mellom variabler

Nå som vi har forhåndsprosesserte dataene, kan vi analysere forholdet mellom funksjonene og etiketten for å få en idé om hvor godt modellen vil kunne forutsi etiketten gitt funksjonene.
Den beste måten å utføre denne typen analyse på er å plotte dataene. Vi bruker igjen Seaborn-funksjonen `catplot` for å visualisere forholdet mellom `Item Size`, `Variety` og `Color` i et kategorisk plott. For å bedre plotte dataene bruker vi den kodede kolonnen `Item Size` og den ukodede kolonnen `Variety`.

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
![Et kategorisk plott av visualiserte data](../../../../2-Regression/4-Logistic/images/pumpkins_catplot_2.png)

### Bruk et swarm-plott

Siden `Color` er en binær kategori (Hvit eller Ikke), trenger den 'en [spesialisert tilnærming](https://seaborn.pydata.org/tutorial/categorical.html?highlight=bar) til visualisering'. Det finnes andre måter å visualisere forholdet mellom denne kategorien og andre variabler.

Du kan visualisere variabler side om side med Seaborn-plott.

1. Prøv et 'swarm'-plott for å vise distribusjonen av verdier:

    ```python
    palette = {
    0: 'orange',
    1: 'wheat'
    }
    sns.swarmplot(x="Color", y="ord__Item Size", data=encoded_pumpkins, palette=palette)
    ```

    ![Et swarm-plott av visualiserte data](../../../../2-Regression/4-Logistic/images/swarm_2.png)

**Vær oppmerksom**: koden ovenfor kan generere en advarsel, siden Seaborn har problemer med å representere så mange datapunkter i et swarm-plott. En mulig løsning er å redusere størrelsen på markøren ved å bruke parameteren 'size'. Vær imidlertid oppmerksom på at dette påvirker lesbarheten til plottet.

> **🧮 Vis meg matematikken**
>
> Logistisk regresjon baserer seg på konseptet 'maksimal sannsynlighet' ved bruk av [sigmoid-funksjoner](https://wikipedia.org/wiki/Sigmoid_function). En 'Sigmoid-funksjon' på et plott ser ut som en 'S'-form. Den tar en verdi og kartlegger den til et sted mellom 0 og 1. Kurven kalles også en 'logistisk kurve'. Formelen ser slik ut:
>
> ![logistisk funksjon](../../../../2-Regression/4-Logistic/images/sigmoid.png)
>
> der sigmoids midtpunkt befinner seg ved x's 0-punkt, L er kurvens maksimumsverdi, og k er kurvens bratthet. Hvis resultatet av funksjonen er mer enn 0.5, vil etiketten i spørsmålet bli gitt klassen '1' av det binære valget. Hvis ikke, vil den bli klassifisert som '0'.

## Bygg modellen din

Å bygge en modell for å finne disse binære klassifiseringene er overraskende enkelt i Scikit-learn.

[![ML for nybegynnere - Logistisk regresjon for klassifisering av data](https://img.youtube.com/vi/MmZS2otPrQ8/0.jpg)](https://youtu.be/MmZS2otPrQ8 "ML for nybegynnere - Logistisk regresjon for klassifisering av data")

> 🎥 Klikk på bildet ovenfor for en kort videooversikt over å bygge en lineær regresjonsmodell

1. Velg variablene du vil bruke i klassifiseringsmodellen din og del opp trenings- og testsett ved å kalle `train_test_split()`:

    ```python
    from sklearn.model_selection import train_test_split
    
    X = encoded_pumpkins[encoded_pumpkins.columns.difference(['Color'])]
    y = encoded_pumpkins['Color']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    
    ```

2. Nå kan du trene modellen din ved å kalle `fit()` med treningsdataene dine, og skrive ut resultatet:

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

    Ta en titt på modellens resultattavle. Det er ikke dårlig, med tanke på at du bare har rundt 1000 rader med data:

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

## Bedre forståelse via en forvirringsmatrise

Mens du kan få en resultattavlerapport [termer](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html?highlight=classification_report#sklearn.metrics.classification_report) ved å skrive ut elementene ovenfor, kan du kanskje forstå modellen din bedre ved å bruke en [forvirringsmatrise](https://scikit-learn.org/stable/modules/model_evaluation.html#confusion-matrix) for å hjelpe oss med å forstå hvordan modellen presterer.

> 🎓 En '[forvirringsmatrise](https://wikipedia.org/wiki/Confusion_matrix)' (eller 'feilmatrise') er en tabell som uttrykker modellens sanne vs. falske positive og negative, og dermed måler nøyaktigheten av prediksjonene.

1. For å bruke en forvirringsmatrise, kall `confusion_matrix()`:

    ```python
    from sklearn.metrics import confusion_matrix
    confusion_matrix(y_test, predictions)
    ```

    Ta en titt på modellens forvirringsmatrise:

    ```output
    array([[162,   4],
           [ 11,  22]])
    ```

I Scikit-learn er rader (akse 0) faktiske etiketter og kolonner (akse 1) predikerte etiketter.

|       |   0   |   1   |
| :---: | :---: | :---: |
|   0   |  TN   |  FP   |
|   1   |  FN   |  TP   |

Hva skjer her? La oss si at modellen vår blir bedt om å klassifisere gresskar mellom to binære kategorier, kategori 'hvit' og kategori 'ikke-hvit'.

- Hvis modellen din forutsier et gresskar som ikke hvitt og det faktisk tilhører kategorien 'ikke-hvit', kaller vi det en sann negativ, vist av det øverste venstre tallet.
- Hvis modellen din forutsier et gresskar som hvitt og det faktisk tilhører kategorien 'ikke-hvit', kaller vi det en falsk negativ, vist av det nederste venstre tallet.
- Hvis modellen din forutsier et gresskar som ikke hvitt og det faktisk tilhører kategorien 'hvit', kaller vi det en falsk positiv, vist av det øverste høyre tallet.
- Hvis modellen din forutsier et gresskar som hvitt og det faktisk tilhører kategorien 'hvit', kaller vi det en sann positiv, vist av det nederste høyre tallet.

Som du kanskje har gjettet, er det å foretrekke å ha et større antall sanne positive og sanne negative og et lavere antall falske positive og falske negative, noe som innebærer at modellen presterer bedre.
Hvordan henger forvirringsmatrisen sammen med presisjon og tilbakekalling? Husk, klassifiseringsrapporten som ble skrevet ut ovenfor viste presisjon (0.85) og tilbakekalling (0.67).

Presisjon = tp / (tp + fp) = 22 / (22 + 4) = 0.8461538461538461

Tilbakekalling = tp / (tp + fn) = 22 / (22 + 11) = 0.6666666666666666

✅ Spørsmål: Ifølge forvirringsmatrisen, hvordan gjorde modellen det? Svar: Ikke dårlig; det er et godt antall sanne negative, men også noen få falske negative.

La oss gå tilbake til begrepene vi så tidligere ved hjelp av forvirringsmatrisens kartlegging av TP/TN og FP/FN:

🎓 Presisjon: TP/(TP + FP) Andelen relevante instanser blant de hentede instansene (f.eks. hvilke etiketter som ble godt merket)

🎓 Tilbakekalling: TP/(TP + FN) Andelen relevante instanser som ble hentet, enten de var godt merket eller ikke

🎓 f1-score: (2 * presisjon * tilbakekalling)/(presisjon + tilbakekalling) Et vektet gjennomsnitt av presisjon og tilbakekalling, der det beste er 1 og det verste er 0

🎓 Støtte: Antall forekomster av hver etikett som ble hentet

🎓 Nøyaktighet: (TP + TN)/(TP + TN + FP + FN) Prosentandelen av etiketter som ble korrekt forutsagt for et utvalg.

🎓 Makro Gjennomsnitt: Beregningen av det uvektede gjennomsnittet av metrikker for hver etikett, uten å ta hensyn til ubalanse i etiketter.

🎓 Vektet Gjennomsnitt: Beregningen av gjennomsnittet av metrikker for hver etikett, som tar hensyn til ubalanse i etiketter ved å vekte dem etter deres støtte (antall sanne instanser for hver etikett).

✅ Kan du tenke deg hvilken metrikk du bør følge med på hvis du vil at modellen din skal redusere antall falske negative?

## Visualiser ROC-kurven for denne modellen

[![ML for nybegynnere - Analyse av logistisk regresjonsytelse med ROC-kurver](https://img.youtube.com/vi/GApO575jTA0/0.jpg)](https://youtu.be/GApO575jTA0 "ML for nybegynnere - Analyse av logistisk regresjonsytelse med ROC-kurver")


> 🎥 Klikk på bildet ovenfor for en kort videooversikt over ROC-kurver

La oss gjøre én visualisering til for å se den såkalte 'ROC'-kurven:

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

Ved hjelp av Matplotlib, plott modellens [Receiving Operating Characteristic](https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html?highlight=roc) eller ROC. ROC-kurver brukes ofte for å få en oversikt over utdataene fra en klassifiserer i form av sanne vs. falske positive. "ROC-kurver har vanligvis sanne positive på Y-aksen, og falske positive på X-aksen." Dermed er brattheten på kurven og avstanden mellom midtlinjen og kurven viktig: du vil ha en kurve som raskt går opp og over linjen. I vårt tilfelle er det falske positive i starten, og deretter går linjen opp og over riktig:

![ROC](../../../../2-Regression/4-Logistic/images/ROC_2.png)

Til slutt, bruk Scikit-learns [`roc_auc_score` API](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html?highlight=roc_auc#sklearn.metrics.roc_auc_score) for å beregne den faktiske 'Area Under the Curve' (AUC):

```python
auc = roc_auc_score(y_test,y_scores[:,1])
print(auc)
```
Resultatet er `0.9749908725812341`. Siden AUC varierer fra 0 til 1, ønsker du en høy score, ettersom en modell som er 100 % korrekt i sine prediksjoner vil ha en AUC på 1; i dette tilfellet er modellen _ganske god_. 

I fremtidige leksjoner om klassifiseringer vil du lære hvordan du kan iterere for å forbedre modellens resultater. Men for nå, gratulerer! Du har fullført disse regresjonsleksjonene!

---
## 🚀Utfordring

Det er mye mer å utforske når det gjelder logistisk regresjon! Men den beste måten å lære på er å eksperimentere. Finn et datasett som egner seg til denne typen analyse og bygg en modell med det. Hva lærer du? tips: prøv [Kaggle](https://www.kaggle.com/search?q=logistic+regression+datasets) for interessante datasett.

## [Quiz etter forelesning](https://ff-quizzes.netlify.app/en/ml/)

## Gjennomgang & Selvstudium

Les de første sidene av [denne artikkelen fra Stanford](https://web.stanford.edu/~jurafsky/slp3/5.pdf) om noen praktiske bruksområder for logistisk regresjon. Tenk på oppgaver som egner seg bedre for den ene eller den andre typen regresjonsoppgaver som vi har studert så langt. Hva ville fungert best?

## Oppgave 

[Prøv denne regresjonen på nytt](assignment.md)

---

**Ansvarsfraskrivelse**:  
Dette dokumentet er oversatt ved hjelp av AI-oversettelsestjenesten [Co-op Translator](https://github.com/Azure/co-op-translator). Selv om vi tilstreber nøyaktighet, vennligst vær oppmerksom på at automatiske oversettelser kan inneholde feil eller unøyaktigheter. Det originale dokumentet på sitt opprinnelige språk bør anses som den autoritative kilden. For kritisk informasjon anbefales profesjonell menneskelig oversettelse. Vi er ikke ansvarlige for eventuelle misforståelser eller feiltolkninger som oppstår ved bruk av denne oversettelsen.