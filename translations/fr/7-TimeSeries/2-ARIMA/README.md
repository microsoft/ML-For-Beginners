<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "917dbf890db71a322f306050cb284749",
  "translation_date": "2025-09-04T22:54:52+00:00",
  "source_file": "7-TimeSeries/2-ARIMA/README.md",
  "language_code": "fr"
}
-->
# Prévision des séries temporelles avec ARIMA

Dans la leçon précédente, vous avez appris un peu sur la prévision des séries temporelles et chargé un ensemble de données montrant les fluctuations de la charge électrique sur une période donnée.

[![Introduction à ARIMA](https://img.youtube.com/vi/IUSk-YDau10/0.jpg)](https://youtu.be/IUSk-YDau10 "Introduction à ARIMA")

> 🎥 Cliquez sur l'image ci-dessus pour une vidéo : Une brève introduction aux modèles ARIMA. L'exemple est réalisé en R, mais les concepts sont universels.

## [Quiz avant la leçon](https://ff-quizzes.netlify.app/en/ml/)

## Introduction

Dans cette leçon, vous allez découvrir une méthode spécifique pour construire des modèles avec [ARIMA : *A*uto*R*egressive *I*ntegrated *M*oving *A*verage](https://wikipedia.org/wiki/Autoregressive_integrated_moving_average). Les modèles ARIMA sont particulièrement adaptés pour ajuster des données qui présentent une [non-stationnarité](https://wikipedia.org/wiki/Stationary_process).

## Concepts généraux

Pour travailler avec ARIMA, il y a quelques concepts que vous devez comprendre :

- 🎓 **Stationnarité**. Dans un contexte statistique, la stationnarité fait référence à des données dont la distribution ne change pas lorsqu'elles sont décalées dans le temps. Les données non stationnaires, en revanche, montrent des fluctuations dues à des tendances qui doivent être transformées pour être analysées. La saisonnalité, par exemple, peut introduire des fluctuations dans les données et peut être éliminée par un processus de "différenciation saisonnière".

- 🎓 **[Différenciation](https://wikipedia.org/wiki/Autoregressive_integrated_moving_average#Differencing)**. La différenciation des données, toujours dans un contexte statistique, fait référence au processus de transformation des données non stationnaires pour les rendre stationnaires en supprimant leur tendance non constante. "La différenciation élimine les changements dans le niveau d'une série temporelle, supprimant ainsi la tendance et la saisonnalité et stabilisant par conséquent la moyenne de la série temporelle." [Article de Shixiong et al](https://arxiv.org/abs/1904.07632)

## ARIMA dans le contexte des séries temporelles

Décomposons les parties d'ARIMA pour mieux comprendre comment cela nous aide à modéliser les séries temporelles et à faire des prédictions.

- **AR - pour AutoRégressif**. Les modèles autorégressifs, comme leur nom l'indique, regardent "en arrière" dans le temps pour analyser les valeurs précédentes de vos données et en tirer des hypothèses. Ces valeurs précédentes sont appelées "retards". Un exemple serait des données montrant les ventes mensuelles de crayons. Le total des ventes de chaque mois serait considéré comme une "variable évolutive" dans l'ensemble de données. Ce modèle est construit comme "la variable évolutive d'intérêt est régressée sur ses propres valeurs retardées (c'est-à-dire antérieures)." [wikipedia](https://wikipedia.org/wiki/Autoregressive_integrated_moving_average)

- **I - pour Intégré**. Contrairement aux modèles similaires 'ARMA', le 'I' dans ARIMA fait référence à son aspect *[intégré](https://wikipedia.org/wiki/Order_of_integration)*. Les données sont "intégrées" lorsque des étapes de différenciation sont appliquées pour éliminer la non-stationnarité.

- **MA - pour Moyenne Mobile**. L'aspect [moyenne mobile](https://wikipedia.org/wiki/Moving-average_model) de ce modèle fait référence à la variable de sortie qui est déterminée en observant les valeurs actuelles et passées des retards.

En résumé : ARIMA est utilisé pour ajuster un modèle à la forme particulière des données de séries temporelles aussi précisément que possible.

## Exercice - construire un modèle ARIMA

Ouvrez le dossier [_/working_](https://github.com/microsoft/ML-For-Beginners/tree/main/7-TimeSeries/2-ARIMA/working) dans cette leçon et trouvez le fichier [_notebook.ipynb_](https://github.com/microsoft/ML-For-Beginners/blob/main/7-TimeSeries/2-ARIMA/working/notebook.ipynb).

1. Exécutez le notebook pour charger la bibliothèque Python `statsmodels`; vous en aurez besoin pour les modèles ARIMA.

1. Chargez les bibliothèques nécessaires.

1. Ensuite, chargez plusieurs autres bibliothèques utiles pour tracer les données :

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

1. Chargez les données du fichier `/data/energy.csv` dans un dataframe Pandas et examinez-les :

    ```python
    energy = load_data('./data')[['load']]
    energy.head(10)
    ```

1. Tracez toutes les données énergétiques disponibles de janvier 2012 à décembre 2014. Il ne devrait pas y avoir de surprises, car nous avons vu ces données dans la dernière leçon :

    ```python
    energy.plot(y='load', subplots=True, figsize=(15, 8), fontsize=12)
    plt.xlabel('timestamp', fontsize=12)
    plt.ylabel('load', fontsize=12)
    plt.show()
    ```

    Maintenant, construisons un modèle !

### Créer des ensembles de données d'entraînement et de test

Maintenant que vos données sont chargées, vous pouvez les séparer en ensembles d'entraînement et de test. Vous entraînerez votre modèle sur l'ensemble d'entraînement. Comme d'habitude, après que le modèle ait terminé son entraînement, vous évaluerez sa précision en utilisant l'ensemble de test. Vous devez vous assurer que l'ensemble de test couvre une période ultérieure par rapport à l'ensemble d'entraînement pour garantir que le modèle ne tire pas d'informations des périodes futures.

1. Allouez une période de deux mois du 1er septembre au 31 octobre 2014 à l'ensemble d'entraînement. L'ensemble de test inclura la période de deux mois du 1er novembre au 31 décembre 2014 :

    ```python
    train_start_dt = '2014-11-01 00:00:00'
    test_start_dt = '2014-12-30 00:00:00'
    ```

    Étant donné que ces données reflètent la consommation quotidienne d'énergie, il existe un fort motif saisonnier, mais la consommation est la plus similaire à celle des jours les plus récents.

1. Visualisez les différences :

    ```python
    energy[(energy.index < test_start_dt) & (energy.index >= train_start_dt)][['load']].rename(columns={'load':'train'}) \
        .join(energy[test_start_dt:][['load']].rename(columns={'load':'test'}), how='outer') \
        .plot(y=['train', 'test'], figsize=(15, 8), fontsize=12)
    plt.xlabel('timestamp', fontsize=12)
    plt.ylabel('load', fontsize=12)
    plt.show()
    ```

    ![données d'entraînement et de test](../../../../7-TimeSeries/2-ARIMA/images/train-test.png)

    Par conséquent, utiliser une fenêtre de temps relativement petite pour entraîner les données devrait être suffisant.

    > Note : Étant donné que la fonction que nous utilisons pour ajuster le modèle ARIMA utilise une validation en échantillon pendant l'ajustement, nous omettrons les données de validation.

### Préparer les données pour l'entraînement

Maintenant, vous devez préparer les données pour l'entraînement en effectuant un filtrage et une mise à l'échelle de vos données. Filtrez votre ensemble de données pour inclure uniquement les périodes et colonnes nécessaires, et mettez à l'échelle pour garantir que les données sont projetées dans l'intervalle 0,1.

1. Filtrez l'ensemble de données original pour inclure uniquement les périodes mentionnées par ensemble et en incluant uniquement la colonne nécessaire 'load' plus la date :

    ```python
    train = energy.copy()[(energy.index >= train_start_dt) & (energy.index < test_start_dt)][['load']]
    test = energy.copy()[energy.index >= test_start_dt][['load']]

    print('Training data shape: ', train.shape)
    print('Test data shape: ', test.shape)
    ```

    Vous pouvez voir la forme des données :

    ```output
    Training data shape:  (1416, 1)
    Test data shape:  (48, 1)
    ```

1. Mettez à l'échelle les données pour qu'elles soient dans la plage (0, 1).

    ```python
    scaler = MinMaxScaler()
    train['load'] = scaler.fit_transform(train)
    train.head(10)
    ```

1. Visualisez les données originales par rapport aux données mises à l'échelle :

    ```python
    energy[(energy.index >= train_start_dt) & (energy.index < test_start_dt)][['load']].rename(columns={'load':'original load'}).plot.hist(bins=100, fontsize=12)
    train.rename(columns={'load':'scaled load'}).plot.hist(bins=100, fontsize=12)
    plt.show()
    ```

    ![original](../../../../7-TimeSeries/2-ARIMA/images/original.png)

    > Les données originales

    ![scaled](../../../../7-TimeSeries/2-ARIMA/images/scaled.png)

    > Les données mises à l'échelle

1. Maintenant que vous avez calibré les données mises à l'échelle, vous pouvez mettre à l'échelle les données de test :

    ```python
    test['load'] = scaler.transform(test)
    test.head()
    ```

### Implémenter ARIMA

Il est temps d'implémenter ARIMA ! Vous allez maintenant utiliser la bibliothèque `statsmodels` que vous avez installée précédemment.

Vous devez suivre plusieurs étapes :

   1. Définissez le modèle en appelant `SARIMAX()` et en passant les paramètres du modèle : les paramètres p, d et q, ainsi que les paramètres P, D et Q.
   2. Préparez le modèle pour les données d'entraînement en appelant la fonction fit().
   3. Faites des prédictions en appelant la fonction `forecast()` et en spécifiant le nombre d'étapes (l'`horizon`) à prévoir.

> 🎓 À quoi servent tous ces paramètres ? Dans un modèle ARIMA, il y a 3 paramètres utilisés pour modéliser les principaux aspects d'une série temporelle : la saisonnalité, la tendance et le bruit. Ces paramètres sont :

`p` : le paramètre associé à l'aspect autorégressif du modèle, qui incorpore les valeurs *passées*.
`d` : le paramètre associé à la partie intégrée du modèle, qui affecte la quantité de *différenciation* (🎓 souvenez-vous de la différenciation 👆 ?) à appliquer à une série temporelle.
`q` : le paramètre associé à la partie moyenne mobile du modèle.

> Note : Si vos données ont un aspect saisonnier - ce qui est le cas ici -, nous utilisons un modèle ARIMA saisonnier (SARIMA). Dans ce cas, vous devez utiliser un autre ensemble de paramètres : `P`, `D` et `Q` qui décrivent les mêmes associations que `p`, `d` et `q`, mais correspondent aux composantes saisonnières du modèle.

1. Commencez par définir votre valeur d'horizon préférée. Essayons 3 heures :

    ```python
    # Specify the number of steps to forecast ahead
    HORIZON = 3
    print('Forecasting horizon:', HORIZON, 'hours')
    ```

    Sélectionner les meilleures valeurs pour les paramètres d'un modèle ARIMA peut être difficile car c'est en partie subjectif et chronophage. Vous pourriez envisager d'utiliser une fonction `auto_arima()` de la bibliothèque [`pyramid`](https://alkaline-ml.com/pmdarima/0.9.0/modules/generated/pyramid.arima.auto_arima.html).

1. Pour l'instant, essayez quelques sélections manuelles pour trouver un bon modèle.

    ```python
    order = (4, 1, 0)
    seasonal_order = (1, 1, 0, 24)

    model = SARIMAX(endog=train, order=order, seasonal_order=seasonal_order)
    results = model.fit()

    print(results.summary())
    ```

    Un tableau de résultats est imprimé.

Vous avez construit votre premier modèle ! Maintenant, nous devons trouver un moyen de l'évaluer.

### Évaluer votre modèle

Pour évaluer votre modèle, vous pouvez effectuer la validation dite `walk forward`. En pratique, les modèles de séries temporelles sont réentraînés chaque fois que de nouvelles données deviennent disponibles. Cela permet au modèle de faire la meilleure prévision à chaque étape temporelle.

En commençant au début de la série temporelle avec cette technique, entraînez le modèle sur l'ensemble de données d'entraînement. Ensuite, faites une prédiction sur l'étape temporelle suivante. La prédiction est évaluée par rapport à la valeur connue. L'ensemble d'entraînement est ensuite étendu pour inclure la valeur connue et le processus est répété.

> Note : Vous devriez garder la fenêtre de l'ensemble d'entraînement fixe pour un entraînement plus efficace afin que chaque fois que vous ajoutez une nouvelle observation à l'ensemble d'entraînement, vous supprimiez l'observation du début de l'ensemble.

Ce processus fournit une estimation plus robuste de la performance du modèle en pratique. Cependant, cela a un coût computationnel lié à la création de nombreux modèles. Cela est acceptable si les données sont petites ou si le modèle est simple, mais pourrait poser problème à grande échelle.

La validation `walk forward` est la norme d'or pour l'évaluation des modèles de séries temporelles et est recommandée pour vos propres projets.

1. Tout d'abord, créez un point de données de test pour chaque étape HORIZON.

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

    Les données sont décalées horizontalement selon leur point d'horizon.

1. Faites des prédictions sur vos données de test en utilisant cette approche de fenêtre glissante dans une boucle de la taille de la longueur des données de test :

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

    Vous pouvez observer l'entraînement en cours :

    ```output
    2014-12-30 00:00:00
    1 : predicted = [0.32 0.29 0.28] expected = [0.32945389435989236, 0.2900626678603402, 0.2739480752014323]

    2014-12-30 01:00:00
    2 : predicted = [0.3  0.29 0.3 ] expected = [0.2900626678603402, 0.2739480752014323, 0.26812891674127126]

    2014-12-30 02:00:00
    3 : predicted = [0.27 0.28 0.32] expected = [0.2739480752014323, 0.26812891674127126, 0.3025962399283795]
    ```

1. Comparez les prédictions à la charge réelle :

    ```python
    eval_df = pd.DataFrame(predictions, columns=['t+'+str(t) for t in range(1, HORIZON+1)])
    eval_df['timestamp'] = test.index[0:len(test.index)-HORIZON+1]
    eval_df = pd.melt(eval_df, id_vars='timestamp', value_name='prediction', var_name='h')
    eval_df['actual'] = np.array(np.transpose(test_ts)).ravel()
    eval_df[['prediction', 'actual']] = scaler.inverse_transform(eval_df[['prediction', 'actual']])
    eval_df.head()
    ```

    Résultat
    |     |            | timestamp | h   | prediction | actual   |
    | --- | ---------- | --------- | --- | ---------- | -------- |
    | 0   | 2014-12-30 | 00:00:00  | t+1 | 3,008.74   | 3,023.00 |
    | 1   | 2014-12-30 | 01:00:00  | t+1 | 2,955.53   | 2,935.00 |
    | 2   | 2014-12-30 | 02:00:00  | t+1 | 2,900.17   | 2,899.00 |
    | 3   | 2014-12-30 | 03:00:00  | t+1 | 2,917.69   | 2,886.00 |
    | 4   | 2014-12-30 | 04:00:00  | t+1 | 2,946.99   | 2,963.00 |

    Observez la prédiction des données horaires, comparée à la charge réelle. Quelle est la précision ?

### Vérifiez la précision du modèle

Vérifiez la précision de votre modèle en testant son erreur moyenne absolue en pourcentage (MAPE) sur toutes les prédictions.
> **🧮 Montrez-moi les calculs**
>
> ![MAPE](../../../../7-TimeSeries/2-ARIMA/images/mape.png)
>
> [MAPE](https://www.linkedin.com/pulse/what-mape-mad-msd-time-series-allameh-statistics/) est utilisé pour montrer la précision des prédictions sous forme de ratio défini par la formule ci-dessus. La différence entre la valeur réelle et la valeur prédite est divisée par la valeur réelle.  
> "La valeur absolue dans ce calcul est additionnée pour chaque point prédit dans le temps, puis divisée par le nombre de points ajustés n." [wikipedia](https://wikipedia.org/wiki/Mean_absolute_percentage_error)
1. Exprimer l'équation en code :

    ```python
    if(HORIZON > 1):
        eval_df['APE'] = (eval_df['prediction'] - eval_df['actual']).abs() / eval_df['actual']
        print(eval_df.groupby('h')['APE'].mean())
    ```

1. Calculer le MAPE d'une étape :

    ```python
    print('One step forecast MAPE: ', (mape(eval_df[eval_df['h'] == 't+1']['prediction'], eval_df[eval_df['h'] == 't+1']['actual']))*100, '%')
    ```

    MAPE de prévision à une étape :  0.5570581332313952 %

1. Afficher le MAPE de prévision multi-étapes :

    ```python
    print('Multi-step forecast MAPE: ', mape(eval_df['prediction'], eval_df['actual'])*100, '%')
    ```

    ```output
    Multi-step forecast MAPE:  1.1460048657704118 %
    ```

    Un chiffre bas est idéal : considérez qu'une prévision avec un MAPE de 10 est erronée de 10 %.

1. Mais comme toujours, il est plus facile de visualiser ce type de mesure de précision, alors traçons-le :

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

    ![un modèle de série temporelle](../../../../7-TimeSeries/2-ARIMA/images/accuracy.png)

🏆 Un très beau graphique, montrant un modèle avec une bonne précision. Bien joué !

---

## 🚀Défi

Explorez les différentes façons de tester la précision d'un modèle de série temporelle. Nous abordons le MAPE dans cette leçon, mais existe-t-il d'autres méthodes que vous pourriez utiliser ? Faites des recherches et annotez-les. Un document utile peut être trouvé [ici](https://otexts.com/fpp2/accuracy.html)

## [Quiz post-cours](https://ff-quizzes.netlify.app/en/ml/)

## Révision & Auto-apprentissage

Cette leçon aborde uniquement les bases de la prévision de séries temporelles avec ARIMA. Prenez le temps d'approfondir vos connaissances en explorant [ce dépôt](https://microsoft.github.io/forecasting/) et ses différents types de modèles pour découvrir d'autres façons de construire des modèles de séries temporelles.

## Devoir

[Un nouveau modèle ARIMA](assignment.md)

---

**Avertissement** :  
Ce document a été traduit à l'aide du service de traduction automatique [Co-op Translator](https://github.com/Azure/co-op-translator). Bien que nous nous efforcions d'assurer l'exactitude, veuillez noter que les traductions automatisées peuvent contenir des erreurs ou des inexactitudes. Le document original dans sa langue d'origine doit être considéré comme la source faisant autorité. Pour des informations critiques, il est recommandé de recourir à une traduction professionnelle réalisée par un humain. Nous déclinons toute responsabilité en cas de malentendus ou d'interprétations erronées résultant de l'utilisation de cette traduction.