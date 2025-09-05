<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "917dbf890db71a322f306050cb284749",
  "translation_date": "2025-09-04T21:53:41+00:00",
  "source_file": "7-TimeSeries/2-ARIMA/README.md",
  "language_code": "de"
}
-->
# Zeitreihenprognose mit ARIMA

In der vorherigen Lektion haben Sie etwas über Zeitreihenprognosen gelernt und einen Datensatz geladen, der die Schwankungen der elektrischen Last über einen bestimmten Zeitraum zeigt.

[![Einführung in ARIMA](https://img.youtube.com/vi/IUSk-YDau10/0.jpg)](https://youtu.be/IUSk-YDau10 "Einführung in ARIMA")

> 🎥 Klicken Sie auf das Bild oben, um ein Video anzusehen: Eine kurze Einführung in ARIMA-Modelle. Das Beispiel wird in R durchgeführt, aber die Konzepte sind universell.

## [Quiz vor der Lektion](https://ff-quizzes.netlify.app/en/ml/)

## Einführung

In dieser Lektion lernen Sie eine spezifische Methode kennen, um Modelle mit [ARIMA: *A*uto*R*egressive *I*ntegrated *M*oving *A*verage](https://wikipedia.org/wiki/Autoregressive_integrated_moving_average) zu erstellen. ARIMA-Modelle eignen sich besonders gut für Daten, die [Nicht-Stationarität](https://wikipedia.org/wiki/Stationary_process) aufweisen.

## Allgemeine Konzepte

Um mit ARIMA arbeiten zu können, müssen Sie einige grundlegende Konzepte kennen:

- 🎓 **Stationarität**. Im statistischen Kontext bezieht sich Stationarität auf Daten, deren Verteilung sich nicht ändert, wenn sie in der Zeit verschoben werden. Nicht-stationäre Daten zeigen hingegen Schwankungen aufgrund von Trends, die transformiert werden müssen, um analysiert werden zu können. Saisonalität kann beispielsweise Schwankungen in den Daten verursachen, die durch einen Prozess des „saisonalen Differenzierens“ eliminiert werden können.

- 🎓 **[Differenzieren](https://wikipedia.org/wiki/Autoregressive_integrated_moving_average#Differencing)**. Das Differenzieren von Daten bezieht sich auf den Prozess, nicht-stationäre Daten zu transformieren, um sie stationär zu machen, indem der nicht-konstante Trend entfernt wird. „Differenzieren entfernt die Änderungen im Niveau einer Zeitreihe, eliminiert Trends und Saisonalität und stabilisiert dadurch den Mittelwert der Zeitreihe.“ [Paper von Shixiong et al](https://arxiv.org/abs/1904.07632)

## ARIMA im Kontext von Zeitreihen

Lassen Sie uns die Bestandteile von ARIMA genauer betrachten, um besser zu verstehen, wie es uns hilft, Zeitreihen zu modellieren und Vorhersagen zu treffen.

- **AR - für AutoRegressiv**. Autoregressive Modelle analysieren, wie der Name schon sagt, frühere Werte in Ihren Daten, um Annahmen über sie zu treffen. Diese früheren Werte werden als „Lags“ bezeichnet. Ein Beispiel wären Daten, die monatliche Verkaufszahlen von Bleistiften zeigen. Die Verkaufszahlen jedes Monats würden als „entwickelnde Variable“ im Datensatz betrachtet. Dieses Modell wird erstellt, indem „die interessierende Variable auf ihre eigenen verzögerten (d. h. vorherigen) Werte regressiert wird.“ [Wikipedia](https://wikipedia.org/wiki/Autoregressive_integrated_moving_average)

- **I - für Integriert**. Im Gegensatz zu den ähnlichen 'ARMA'-Modellen bezieht sich das 'I' in ARIMA auf den *[integrierten](https://wikipedia.org/wiki/Order_of_integration)* Aspekt. Die Daten werden „integriert“, wenn Differenzierungsschritte angewendet werden, um Nicht-Stationarität zu eliminieren.

- **MA - für Gleitender Durchschnitt**. Der [gleitende Durchschnitt](https://wikipedia.org/wiki/Moving-average_model) in diesem Modell bezieht sich auf die Ausgangsvariable, die durch die Beobachtung der aktuellen und vergangenen Werte der Lags bestimmt wird.

Fazit: ARIMA wird verwendet, um ein Modell so genau wie möglich an die spezielle Form von Zeitreihendaten anzupassen.

## Übung - Ein ARIMA-Modell erstellen

Öffnen Sie den [_/working_](https://github.com/microsoft/ML-For-Beginners/tree/main/7-TimeSeries/2-ARIMA/working)-Ordner in dieser Lektion und suchen Sie die Datei [_notebook.ipynb_](https://github.com/microsoft/ML-For-Beginners/blob/main/7-TimeSeries/2-ARIMA/working/notebook.ipynb).

1. Führen Sie das Notebook aus, um die Python-Bibliothek `statsmodels` zu laden; diese benötigen Sie für ARIMA-Modelle.

1. Laden Sie die notwendigen Bibliotheken.

1. Laden Sie nun weitere Bibliotheken, die für die Datenvisualisierung nützlich sind:

    ```python
    import os
    import warnings
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import datetime as dt
    import math

    from pandas.plotting import autocorrelation_plot
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from sklearn.preprocessing import MinMaxScaler
    from common.utils import load_data, mape
    from IPython.display import Image

    %matplotlib inline
    pd.options.display.float_format = '{:,.2f}'.format
    np.set_printoptions(precision=2)
    warnings.filterwarnings("ignore") # specify to ignore warning messages
    ```

1. Laden Sie die Daten aus der Datei `/data/energy.csv` in ein Pandas-DataFrame und werfen Sie einen Blick darauf:

    ```python
    energy = load_data('./data')[['load']]
    energy.head(10)
    ```

1. Visualisieren Sie alle verfügbaren Energiedaten von Januar 2012 bis Dezember 2014. Es sollte keine Überraschungen geben, da wir diese Daten in der letzten Lektion gesehen haben:

    ```python
    energy.plot(y='load', subplots=True, figsize=(15, 8), fontsize=12)
    plt.xlabel('timestamp', fontsize=12)
    plt.ylabel('load', fontsize=12)
    plt.show()
    ```

    Jetzt erstellen wir ein Modell!

### Erstellen von Trainings- und Testdatensätzen

Nachdem Ihre Daten geladen sind, können Sie sie in Trainings- und Testdatensätze aufteilen. Sie trainieren Ihr Modell mit dem Trainingsdatensatz. Wie üblich bewerten Sie nach Abschluss des Trainings die Genauigkeit des Modells mit dem Testdatensatz. Sie müssen sicherstellen, dass der Testdatensatz einen späteren Zeitraum als der Trainingsdatensatz abdeckt, um sicherzustellen, dass das Modell keine Informationen aus zukünftigen Zeiträumen erhält.

1. Weisen Sie dem Trainingsdatensatz einen Zeitraum von zwei Monaten vom 1. September bis zum 31. Oktober 2014 zu. Der Testdatensatz umfasst den Zeitraum vom 1. November bis zum 31. Dezember 2014:

    ```python
    train_start_dt = '2014-11-01 00:00:00'
    test_start_dt = '2014-12-30 00:00:00'
    ```

    Da diese Daten den täglichen Energieverbrauch widerspiegeln, gibt es ein starkes saisonales Muster, aber der Verbrauch ähnelt am meisten dem Verbrauch in den letzten Tagen.

1. Visualisieren Sie die Unterschiede:

    ```python
    energy[(energy.index < test_start_dt) & (energy.index >= train_start_dt)][['load']].rename(columns={'load':'train'}) \
        .join(energy[test_start_dt:][['load']].rename(columns={'load':'test'}), how='outer') \
        .plot(y=['train', 'test'], figsize=(15, 8), fontsize=12)
    plt.xlabel('timestamp', fontsize=12)
    plt.ylabel('load', fontsize=12)
    plt.show()
    ```

    ![Trainings- und Testdaten](../../../../7-TimeSeries/2-ARIMA/images/train-test.png)

    Daher sollte ein relativ kleines Zeitfenster für das Training der Daten ausreichen.

    > Hinweis: Da die Funktion, die wir zum Anpassen des ARIMA-Modells verwenden, während des Anpassens eine Validierung innerhalb des Datensatzes durchführt, werden wir auf Validierungsdaten verzichten.

### Daten für das Training vorbereiten

Nun müssen Sie die Daten für das Training vorbereiten, indem Sie die Daten filtern und skalieren. Filtern Sie Ihren Datensatz, um nur die benötigten Zeiträume und Spalten einzuschließen, und skalieren Sie die Daten, um sicherzustellen, dass sie im Intervall 0,1 projiziert werden.

1. Filtern Sie den ursprünglichen Datensatz, um nur die oben genannten Zeiträume pro Datensatz und nur die benötigte Spalte 'load' sowie das Datum einzuschließen:

    ```python
    train = energy.copy()[(energy.index >= train_start_dt) & (energy.index < test_start_dt)][['load']]
    test = energy.copy()[energy.index >= test_start_dt][['load']]

    print('Training data shape: ', train.shape)
    print('Test data shape: ', test.shape)
    ```

    Sie können die Form der Daten sehen:

    ```output
    Training data shape:  (1416, 1)
    Test data shape:  (48, 1)
    ```

1. Skalieren Sie die Daten, um sie in den Bereich (0, 1) zu bringen.

    ```python
    scaler = MinMaxScaler()
    train['load'] = scaler.fit_transform(train)
    train.head(10)
    ```

1. Visualisieren Sie die ursprünglichen vs. skalierten Daten:

    ```python
    energy[(energy.index >= train_start_dt) & (energy.index < test_start_dt)][['load']].rename(columns={'load':'original load'}).plot.hist(bins=100, fontsize=12)
    train.rename(columns={'load':'scaled load'}).plot.hist(bins=100, fontsize=12)
    plt.show()
    ```

    ![original](../../../../7-TimeSeries/2-ARIMA/images/original.png)

    > Die ursprünglichen Daten

    ![scaled](../../../../7-TimeSeries/2-ARIMA/images/scaled.png)

    > Die skalierten Daten

1. Nachdem Sie die skalierten Daten kalibriert haben, können Sie die Testdaten skalieren:

    ```python
    test['load'] = scaler.transform(test)
    test.head()
    ```

### ARIMA implementieren

Es ist Zeit, ARIMA zu implementieren! Sie verwenden jetzt die zuvor installierte `statsmodels`-Bibliothek.

Nun müssen Sie mehrere Schritte ausführen:

   1. Definieren Sie das Modell, indem Sie `SARIMAX()` aufrufen und die Modellparameter p, d und q sowie P, D und Q übergeben.
   2. Bereiten Sie das Modell für die Trainingsdaten vor, indem Sie die Funktion `fit()` aufrufen.
   3. Treffen Sie Vorhersagen, indem Sie die Funktion `forecast()` aufrufen und die Anzahl der Schritte (den `Horizont`) angeben, die vorhergesagt werden sollen.

> 🎓 Wofür stehen all diese Parameter? In einem ARIMA-Modell gibt es 3 Parameter, die verwendet werden, um die Hauptaspekte einer Zeitreihe zu modellieren: Saisonalität, Trend und Rauschen. Diese Parameter sind:

`p`: Der Parameter, der mit dem autoregressiven Aspekt des Modells verbunden ist und *vergangene* Werte einbezieht.  
`d`: Der Parameter, der mit dem integrierten Teil des Modells verbunden ist und die Menge an *Differenzierung* (🎓 erinnern Sie sich an Differenzierung 👆?) beeinflusst, die auf eine Zeitreihe angewendet wird.  
`q`: Der Parameter, der mit dem gleitenden Durchschnitt des Modells verbunden ist.

> Hinweis: Wenn Ihre Daten einen saisonalen Aspekt haben – was hier der Fall ist –, verwenden wir ein saisonales ARIMA-Modell (SARIMA). In diesem Fall müssen Sie ein weiteres Set von Parametern verwenden: `P`, `D` und `Q`, die dieselben Assoziationen wie `p`, `d` und `q` beschreiben, aber den saisonalen Komponenten des Modells entsprechen.

1. Beginnen Sie mit der Festlegung Ihres bevorzugten Horizontwerts. Versuchen wir es mit 3 Stunden:

    ```python
    # Specify the number of steps to forecast ahead
    HORIZON = 3
    print('Forecasting horizon:', HORIZON, 'hours')
    ```

    Die Auswahl der besten Werte für die Parameter eines ARIMA-Modells kann herausfordernd sein, da sie subjektiv und zeitaufwändig ist. Sie könnten die Funktion `auto_arima()` aus der [`pyramid`-Bibliothek](https://alkaline-ml.com/pmdarima/0.9.0/modules/generated/pyramid.arima.auto_arima.html) in Betracht ziehen.

1. Versuchen Sie vorerst einige manuelle Auswahlmöglichkeiten, um ein gutes Modell zu finden.

    ```python
    order = (4, 1, 0)
    seasonal_order = (1, 1, 0, 24)

    model = SARIMAX(endog=train, order=order, seasonal_order=seasonal_order)
    results = model.fit()

    print(results.summary())
    ```

    Eine Ergebnistabelle wird ausgegeben.

Sie haben Ihr erstes Modell erstellt! Nun müssen wir eine Möglichkeit finden, es zu bewerten.

### Modell bewerten

Um Ihr Modell zu bewerten, können Sie die sogenannte `walk forward`-Validierung durchführen. In der Praxis werden Zeitreihenmodelle jedes Mal neu trainiert, wenn neue Daten verfügbar werden. Dies ermöglicht es dem Modell, die beste Vorhersage zu jedem Zeitpunkt zu treffen.

Beginnen Sie am Anfang der Zeitreihe mit dieser Technik, trainieren Sie das Modell mit dem Trainingsdatensatz. Treffen Sie dann eine Vorhersage für den nächsten Zeitpunkt. Die Vorhersage wird mit dem bekannten Wert verglichen. Der Trainingsdatensatz wird dann erweitert, um den bekannten Wert einzuschließen, und der Prozess wird wiederholt.

> Hinweis: Sie sollten das Fenster des Trainingsdatensatzes fixieren, um effizienter zu trainieren, sodass jedes Mal, wenn Sie eine neue Beobachtung zum Trainingsdatensatz hinzufügen, die Beobachtung vom Anfang des Datensatzes entfernt wird.

Dieser Prozess bietet eine robustere Schätzung, wie das Modell in der Praxis abschneiden wird. Allerdings entstehen dadurch höhere Rechenkosten, da so viele Modelle erstellt werden. Dies ist akzeptabel, wenn die Daten klein oder das Modell einfach ist, könnte jedoch bei größeren Datenmengen problematisch sein.

Die Walk-Forward-Validierung ist der Goldstandard für die Bewertung von Zeitreihenmodellen und wird für Ihre eigenen Projekte empfohlen.

1. Erstellen Sie zunächst einen Testdatenpunkt für jeden HORIZON-Schritt.

    ```python
    test_shifted = test.copy()

    for t in range(1, HORIZON+1):
        test_shifted['load+'+str(t)] = test_shifted['load'].shift(-t, freq='H')

    test_shifted = test_shifted.dropna(how='any')
    test_shifted.head(5)
    ```

    |            |          | load | load+1 | load+2 |
    | ---------- | -------- | ---- | ------ | ------ |
    | 2014-12-30 | 00:00:00 | 0.33 | 0.29   | 0.27   |
    | 2014-12-30 | 01:00:00 | 0.29 | 0.27   | 0.27   |
    | 2014-12-30 | 02:00:00 | 0.27 | 0.27   | 0.30   |
    | 2014-12-30 | 03:00:00 | 0.27 | 0.30   | 0.41   |
    | 2014-12-30 | 04:00:00 | 0.30 | 0.41   | 0.57   |

    Die Daten werden horizontal entsprechend ihrem Horizontpunkt verschoben.

1. Treffen Sie Vorhersagen für Ihre Testdaten mit diesem gleitenden Fensteransatz in einer Schleife, die der Länge der Testdaten entspricht:

    ```python
    %%time
    training_window = 720 # dedicate 30 days (720 hours) for training

    train_ts = train['load']
    test_ts = test_shifted

    history = [x for x in train_ts]
    history = history[(-training_window):]

    predictions = list()

    order = (2, 1, 0)
    seasonal_order = (1, 1, 0, 24)

    for t in range(test_ts.shape[0]):
        model = SARIMAX(endog=history, order=order, seasonal_order=seasonal_order)
        model_fit = model.fit()
        yhat = model_fit.forecast(steps = HORIZON)
        predictions.append(yhat)
        obs = list(test_ts.iloc[t])
        # move the training window
        history.append(obs[0])
        history.pop(0)
        print(test_ts.index[t])
        print(t+1, ': predicted =', yhat, 'expected =', obs)
    ```

    Sie können das Training beobachten:

    ```output
    2014-12-30 00:00:00
    1 : predicted = [0.32 0.29 0.28] expected = [0.32945389435989236, 0.2900626678603402, 0.2739480752014323]

    2014-12-30 01:00:00
    2 : predicted = [0.3  0.29 0.3 ] expected = [0.2900626678603402, 0.2739480752014323, 0.26812891674127126]

    2014-12-30 02:00:00
    3 : predicted = [0.27 0.28 0.32] expected = [0.2739480752014323, 0.26812891674127126, 0.3025962399283795]
    ```

1. Vergleichen Sie die Vorhersagen mit der tatsächlichen Last:

    ```python
    eval_df = pd.DataFrame(predictions, columns=['t+'+str(t) for t in range(1, HORIZON+1)])
    eval_df['timestamp'] = test.index[0:len(test.index)-HORIZON+1]
    eval_df = pd.melt(eval_df, id_vars='timestamp', value_name='prediction', var_name='h')
    eval_df['actual'] = np.array(np.transpose(test_ts)).ravel()
    eval_df[['prediction', 'actual']] = scaler.inverse_transform(eval_df[['prediction', 'actual']])
    eval_df.head()
    ```

    Ausgabe  
    |     |            | timestamp | h   | prediction | actual   |
    | --- | ---------- | --------- | --- | ---------- | -------- |
    | 0   | 2014-12-30 | 00:00:00  | t+1 | 3,008.74   | 3,023.00 |
    | 1   | 2014-12-30 | 01:00:00  | t+1 | 2,955.53   | 2,935.00 |
    | 2   | 2014-12-30 | 02:00:00  | t+1 | 2,900.17   | 2,899.00 |
    | 3   | 2014-12-30 | 03:00:00  | t+1 | 2,917.69   | 2,886.00 |
    | 4   | 2014-12-30 | 04:00:00  | t+1 | 2,946.99   | 2,963.00 |

    Beobachten Sie die stündlichen Vorhersagen im Vergleich zur tatsächlichen Last. Wie genau ist das?

### Modellgenauigkeit überprüfen

Überprüfen Sie die Genauigkeit Ihres Modells, indem Sie den mittleren absoluten prozentualen Fehler (MAPE) über alle Vorhersagen testen.
> **🧮 Zeig mir die Mathematik**
>
> ![MAPE](../../../../7-TimeSeries/2-ARIMA/images/mape.png)
>
> [MAPE](https://www.linkedin.com/pulse/what-mape-mad-msd-time-series-allameh-statistics/) wird verwendet, um die Vorhersagegenauigkeit als Verhältnis darzustellen, das durch die obige Formel definiert ist. Die Differenz zwischen tatsächlichem und vorhergesagtem Wert wird durch den tatsächlichen Wert geteilt. "Der Absolutwert dieser Berechnung wird für jeden prognostizierten Zeitpunkt summiert und durch die Anzahl der angepassten Punkte n geteilt." [wikipedia](https://wikipedia.org/wiki/Mean_absolute_percentage_error)
1. Ausdruck der Gleichung im Code:

    ```python
    if(HORIZON > 1):
        eval_df['APE'] = (eval_df['prediction'] - eval_df['actual']).abs() / eval_df['actual']
        print(eval_df.groupby('h')['APE'].mean())
    ```

1. Berechnung des MAPE für einen Schritt:

    ```python
    print('One step forecast MAPE: ', (mape(eval_df[eval_df['h'] == 't+1']['prediction'], eval_df[eval_df['h'] == 't+1']['actual']))*100, '%')
    ```

    MAPE der Ein-Schritt-Vorhersage:  0.5570581332313952 %

1. Ausgabe des MAPE für die Mehrschritt-Vorhersage:

    ```python
    print('Multi-step forecast MAPE: ', mape(eval_df['prediction'], eval_df['actual'])*100, '%')
    ```

    ```output
    Multi-step forecast MAPE:  1.1460048657704118 %
    ```

    Eine niedrige Zahl ist am besten: Beachten Sie, dass eine Vorhersage mit einem MAPE von 10 um 10 % abweicht.

1. Aber wie immer ist es einfacher, diese Art von Genauigkeitsmessung visuell zu sehen. Lassen Sie uns das plotten:

    ```python
     if(HORIZON == 1):
        ## Plotting single step forecast
        eval_df.plot(x='timestamp', y=['actual', 'prediction'], style=['r', 'b'], figsize=(15, 8))

    else:
        ## Plotting multi step forecast
        plot_df = eval_df[(eval_df.h=='t+1')][['timestamp', 'actual']]
        for t in range(1, HORIZON+1):
            plot_df['t+'+str(t)] = eval_df[(eval_df.h=='t+'+str(t))]['prediction'].values

        fig = plt.figure(figsize=(15, 8))
        ax = plt.plot(plot_df['timestamp'], plot_df['actual'], color='red', linewidth=4.0)
        ax = fig.add_subplot(111)
        for t in range(1, HORIZON+1):
            x = plot_df['timestamp'][(t-1):]
            y = plot_df['t+'+str(t)][0:len(x)]
            ax.plot(x, y, color='blue', linewidth=4*math.pow(.9,t), alpha=math.pow(0.8,t))

        ax.legend(loc='best')

    plt.xlabel('timestamp', fontsize=12)
    plt.ylabel('load', fontsize=12)
    plt.show()
    ```

    ![ein Zeitreihenmodell](../../../../7-TimeSeries/2-ARIMA/images/accuracy.png)

🏆 Ein sehr schöner Plot, der ein Modell mit guter Genauigkeit zeigt. Gut gemacht!

---

## 🚀Herausforderung

Tauchen Sie in die verschiedenen Möglichkeiten ein, die Genauigkeit eines Zeitreihenmodells zu testen. In dieser Lektion sprechen wir über MAPE, aber gibt es andere Methoden, die Sie verwenden könnten? Recherchieren Sie diese und kommentieren Sie sie. Ein hilfreiches Dokument finden Sie [hier](https://otexts.com/fpp2/accuracy.html).

## [Quiz nach der Vorlesung](https://ff-quizzes.netlify.app/en/ml/)

## Rückblick & Selbststudium

Diese Lektion behandelt nur die Grundlagen der Zeitreihenprognose mit ARIMA. Nehmen Sie sich Zeit, Ihr Wissen zu vertiefen, indem Sie [dieses Repository](https://microsoft.github.io/forecasting/) und seine verschiedenen Modelltypen durchstöbern, um andere Möglichkeiten zur Erstellung von Zeitreihenmodellen zu lernen.

## Aufgabe

[Ein neues ARIMA-Modell](assignment.md)

---

**Haftungsausschluss**:  
Dieses Dokument wurde mit dem KI-Übersetzungsdienst [Co-op Translator](https://github.com/Azure/co-op-translator) übersetzt. Obwohl wir uns um Genauigkeit bemühen, weisen wir darauf hin, dass automatisierte Übersetzungen Fehler oder Ungenauigkeiten enthalten können. Das Originaldokument in seiner ursprünglichen Sprache sollte als maßgebliche Quelle betrachtet werden. Für kritische Informationen wird eine professionelle menschliche Übersetzung empfohlen. Wir übernehmen keine Haftung für Missverständnisse oder Fehlinterpretationen, die sich aus der Nutzung dieser Übersetzung ergeben.