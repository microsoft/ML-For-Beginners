<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "2f400075e003e749fdb0d6b3b4787a99",
  "translation_date": "2025-09-03T22:41:56+00:00",
  "source_file": "7-TimeSeries/2-ARIMA/README.md",
  "language_code": "fr"
}
-->
# Prévision des séries temporelles avec ARIMA

Dans la leçon précédente, vous avez appris les bases de la prévision des séries temporelles et chargé un jeu de données montrant les fluctuations de la charge électrique sur une période donnée.

[![Introduction à ARIMA](https://img.youtube.com/vi/IUSk-YDau10/0.jpg)](https://youtu.be/IUSk-YDau10 "Introduction à ARIMA")

> 🎥 Cliquez sur l'image ci-dessus pour une vidéo : Une brève introduction aux modèles ARIMA. L'exemple est réalisé en R, mais les concepts sont universels.

## [Quiz avant la leçon](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/43/)

## Introduction

Dans cette leçon, vous allez découvrir une méthode spécifique pour construire des modèles avec [ARIMA : *A*uto*R*egressive *I*ntegrated *M*oving *A*verage](https://wikipedia.org/wiki/Autoregressive_integrated_moving_average). Les modèles ARIMA sont particulièrement adaptés pour ajuster des données présentant une [non-stationnarité](https://wikipedia.org/wiki/Stationary_process).

## Concepts généraux

Pour travailler avec ARIMA, il y a quelques concepts que vous devez connaître :

- 🎓 **Stationnarité**. Dans un contexte statistique, la stationnarité fait référence à des données dont la distribution ne change pas lorsqu'elles sont décalées dans le temps. Les données non stationnaires, quant à elles, présentent des fluctuations dues à des tendances qui doivent être transformées pour être analysées. La saisonnalité, par exemple, peut introduire des fluctuations dans les données et peut être éliminée par un processus de "différenciation saisonnière".

- 🎓 **[Différenciation](https://wikipedia.org/wiki/Autoregressive_integrated_moving_average#Differencing)**. La différenciation des données, toujours dans un contexte statistique, fait référence au processus de transformation des données non stationnaires pour les rendre stationnaires en supprimant leur tendance non constante. "La différenciation élimine les variations dans le niveau d'une série temporelle, supprimant ainsi la tendance et la saisonnalité, et stabilisant par conséquent la moyenne de la série temporelle." [Article de Shixiong et al](https://arxiv.org/abs/1904.07632)

## ARIMA dans le contexte des séries temporelles

Décomposons les parties d'ARIMA pour mieux comprendre comment il nous aide à modéliser les séries temporelles et à faire des prédictions.

- **AR - pour AutoRégressif**. Les modèles autorégressifs, comme leur nom l'indique, regardent "en arrière" dans le temps pour analyser les valeurs précédentes de vos données et en tirer des hypothèses. Ces valeurs précédentes sont appelées "retards" (lags). Par exemple, des données montrant les ventes mensuelles de crayons. Le total des ventes de chaque mois serait considéré comme une "variable évolutive" dans le jeu de données. Ce modèle est construit comme suit : "la variable évolutive d'intérêt est régressée sur ses propres valeurs retardées (c'est-à-dire, antérieures)." [wikipedia](https://wikipedia.org/wiki/Autoregressive_integrated_moving_average)

- **I - pour Intégré**. Contrairement aux modèles similaires 'ARMA', le 'I' dans ARIMA fait référence à son aspect *[intégré](https://wikipedia.org/wiki/Order_of_integration)*. Les données sont "intégrées" lorsque des étapes de différenciation sont appliquées pour éliminer la non-stationnarité.

- **MA - pour Moyenne Mobile**. L'aspect [moyenne mobile](https://wikipedia.org/wiki/Moving-average_model) de ce modèle fait référence à la variable de sortie qui est déterminée en observant les valeurs actuelles et passées des retards.

En résumé : ARIMA est utilisé pour ajuster un modèle aux formes particulières des données de séries temporelles aussi précisément que possible.

## Exercice - Construire un modèle ARIMA

Ouvrez le dossier [_/working_](https://github.com/microsoft/ML-For-Beginners/tree/main/7-TimeSeries/2-ARIMA/working) dans cette leçon et trouvez le fichier [_notebook.ipynb_](https://github.com/microsoft/ML-For-Beginners/blob/main/7-TimeSeries/2-ARIMA/working/notebook.ipynb).

1. Exécutez le notebook pour charger la bibliothèque Python `statsmodels` ; vous en aurez besoin pour les modèles ARIMA.

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

1. Tracez toutes les données énergétiques disponibles de janvier 2012 à décembre 2014. Il ne devrait pas y avoir de surprises, car nous avons vu ces données dans la leçon précédente :

    ```python
    energy.plot(y='load', subplots=True, figsize=(15, 8), fontsize=12)
    plt.xlabel('timestamp', fontsize=12)
    plt.ylabel('load', fontsize=12)
    plt.show()
    ```

    Maintenant, construisons un modèle !

### Créer des ensembles d'entraînement et de test

Maintenant que vos données sont chargées, vous pouvez les séparer en ensembles d'entraînement et de test. Vous entraînerez votre modèle sur l'ensemble d'entraînement. Comme d'habitude, après que le modèle a terminé son entraînement, vous évaluerez sa précision en utilisant l'ensemble de test. Vous devez vous assurer que l'ensemble de test couvre une période ultérieure à celle de l'ensemble d'entraînement pour garantir que le modèle ne tire pas d'informations des périodes futures.

1. Allouez une période de deux mois, du 1er septembre au 31 octobre 2014, à l'ensemble d'entraînement. L'ensemble de test inclura la période de deux mois du 1er novembre au 31 décembre 2014 :

    ```python
    train_start_dt = '2014-11-01 00:00:00'
    test_start_dt = '2014-12-30 00:00:00'
    ```

    Étant donné que ces données reflètent la consommation quotidienne d'énergie, il existe un fort schéma saisonnier, mais la consommation est plus similaire à celle des jours récents.

1. Visualisez les différences :

    ```python
    energy[(energy.index < test_start_dt) & (energy.index >= train_start_dt)][['load']].rename(columns={'load':'train'}) \
        .join(energy[test_start_dt:][['load']].rename(columns={'load':'test'}), how='outer') \
        .plot(y=['train', 'test'], figsize=(15, 8), fontsize=12)
    plt.xlabel('timestamp', fontsize=12)
    plt.ylabel('load', fontsize=12)
    plt.show()
    ```

    ![données d'entraînement et de test](../../../../translated_images/train-test.8928d14e5b91fc942f0ca9201b2d36c890ea7e98f7619fd94f75de3a4c2bacb9.fr.png)

    Par conséquent, utiliser une fenêtre de temps relativement petite pour entraîner les données devrait suffire.

    > Note : Étant donné que la fonction que nous utilisons pour ajuster le modèle ARIMA utilise une validation interne pendant l'ajustement, nous omettrons les données de validation.

### Préparer les données pour l'entraînement

Maintenant, vous devez préparer les données pour l'entraînement en effectuant un filtrage et une mise à l'échelle de vos données. Filtrez votre jeu de données pour n'inclure que les périodes et colonnes nécessaires, et mettez à l'échelle pour vous assurer que les données sont projetées dans l'intervalle 0,1.

1. Filtrez le jeu de données original pour n'inclure que les périodes mentionnées par ensemble et uniquement la colonne 'load' nécessaire ainsi que la date :

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

1. Mettez les données à l'échelle pour qu'elles soient dans la plage (0, 1).

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

    ![original](../../../../translated_images/original.b2b15efe0ce92b8745918f071dceec2231661bf49c8db6918e3ff4b3b0b183c2.fr.png)

    > Les données originales

    ![scaled](../../../../translated_images/scaled.e35258ca5cd3d43f86d5175e584ba96b38d51501f234abf52e11f4fe2631e45f.fr.png)

    > Les données mises à l'échelle

1. Maintenant que vous avez calibré les données mises à l'échelle, vous pouvez mettre à l'échelle les données de test :

    ```python
    test['load'] = scaler.transform(test)
    test.head()
    ```

### Implémenter ARIMA

Il est temps d'implémenter ARIMA ! Vous allez maintenant utiliser la bibliothèque `statsmodels` que vous avez installée précédemment.

Vous devez suivre plusieurs étapes :

   1. Définissez le modèle en appelant `SARIMAX()` et en passant les paramètres du modèle : les paramètres p, d, et q, ainsi que les paramètres P, D, et Q.
   2. Préparez le modèle pour les données d'entraînement en appelant la fonction `fit()`.
   3. Faites des prédictions en appelant la fonction `forecast()` et en spécifiant le nombre d'étapes (l'`horizon`) à prévoir.

> 🎓 À quoi servent tous ces paramètres ? Dans un modèle ARIMA, il y a 3 paramètres utilisés pour modéliser les principaux aspects d'une série temporelle : saisonnalité, tendance et bruit. Ces paramètres sont :

`p` : le paramètre associé à l'aspect autorégressif du modèle, qui intègre les valeurs *passées*.
`d` : le paramètre associé à la partie intégrée du modèle, qui affecte la quantité de *différenciation* (🎓 souvenez-vous de la différenciation 👆 ?) à appliquer à une série temporelle.
`q` : le paramètre associé à la partie moyenne mobile du modèle.

> Note : Si vos données ont un aspect saisonnier - ce qui est le cas ici -, nous utilisons un modèle ARIMA saisonnier (SARIMA). Dans ce cas, vous devez utiliser un autre ensemble de paramètres : `P`, `D`, et `Q` qui décrivent les mêmes associations que `p`, `d`, et `q`, mais correspondent aux composantes saisonnières du modèle.

1. Commencez par définir votre valeur d'horizon préférée. Essayons 3 heures :

    ```python
    # Specify the number of steps to forecast ahead
    HORIZON = 3
    print('Forecasting horizon:', HORIZON, 'hours')
    ```

    Sélectionner les meilleures valeurs pour les paramètres d'un modèle ARIMA peut être difficile car c'est en partie subjectif et chronophage. Vous pourriez envisager d'utiliser une fonction `auto_arima()` de la [bibliothèque `pyramid`](https://alkaline-ml.com/pmdarima/0.9.0/modules/generated/pyramid.arima.auto_arima.html).

1. Pour l'instant, essayez quelques sélections manuelles pour trouver un bon modèle.

    ```python
    order = (4, 1, 0)
    seasonal_order = (1, 1, 0, 24)

    model = SARIMAX(endog=train, order=order, seasonal_order=seasonal_order)
    results = model.fit()

    print(results.summary())
    ```

    Un tableau de résultats est affiché.

Vous avez construit votre premier modèle ! Maintenant, nous devons trouver un moyen de l'évaluer.

### Évaluer votre modèle

Pour évaluer votre modèle, vous pouvez effectuer une validation dite `walk forward`. En pratique, les modèles de séries temporelles sont réentraînés chaque fois qu'une nouvelle donnée devient disponible. Cela permet au modèle de faire la meilleure prévision à chaque étape temporelle.

En utilisant cette technique, commencez au début de la série temporelle, entraînez le modèle sur l'ensemble d'entraînement. Ensuite, faites une prédiction pour l'étape temporelle suivante. La prédiction est évaluée par rapport à la valeur connue. L'ensemble d'entraînement est ensuite étendu pour inclure la valeur connue et le processus est répété.

> Note : Vous devriez garder la fenêtre de l'ensemble d'entraînement fixe pour un entraînement plus efficace, de sorte qu'à chaque fois que vous ajoutez une nouvelle observation à l'ensemble d'entraînement, vous supprimez l'observation du début de l'ensemble.

Ce processus fournit une estimation plus robuste de la performance du modèle en pratique. Cependant, cela a un coût computationnel, car il faut créer de nombreux modèles. Cela est acceptable si les données sont petites ou si le modèle est simple, mais peut poser problème à grande échelle.

La validation `walk forward` est la norme d'or pour l'évaluation des modèles de séries temporelles et est recommandée pour vos propres projets.

1. Tout d'abord, créez un point de données de test pour chaque étape de l'HORIZON.

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

    Observez la prédiction des données horaires par rapport à la charge réelle. Quelle est la précision de cette prédiction ?

### Vérifiez la précision du modèle

Vérifiez la précision de votre modèle en testant son erreur de pourcentage absolue moyenne (MAPE) sur toutes les prédictions.
> **🧮 Montrez-moi les calculs**
>
> ![MAPE](../../../../translated_images/mape.fd87bbaf4d346846df6af88b26bf6f0926bf9a5027816d5e23e1200866e3e8a4.fr.png)
>
> [MAPE](https://www.linkedin.com/pulse/what-mape-mad-msd-time-series-allameh-statistics/) est utilisé pour montrer la précision des prédictions sous forme de ratio défini par la formule ci-dessus. La différence entre la valeur réelle et la valeur prédite est divisée par la valeur réelle.  
> "La valeur absolue dans ce calcul est sommée pour chaque point prédit dans le temps, puis divisée par le nombre total de points ajustés n." [wikipedia](https://wikipedia.org/wiki/Mean_absolute_percentage_error)
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

    MAPE de prévision à une étape :  0,5570581332313952 %

1. Afficher le MAPE de prévision multi-étapes :

    ```python
    print('Multi-step forecast MAPE: ', mape(eval_df['prediction'], eval_df['actual'])*100, '%')
    ```

    ```output
    Multi-step forecast MAPE:  1.1460048657704118 %
    ```

    Un chiffre bas est préférable : considérez qu'une prévision avec un MAPE de 10 est erronée de 10 %.

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

    ![un modèle de série temporelle](../../../../translated_images/accuracy.2c47fe1bf15f44b3656651c84d5e2ba9b37cd929cd2aa8ab6cc3073f50570f4e.fr.png)

🏆 Un très beau graphique, montrant un modèle avec une bonne précision. Bien joué !

---

## 🚀Défi

Explorez les différentes façons de tester la précision d'un modèle de série temporelle. Nous abordons le MAPE dans cette leçon, mais existe-t-il d'autres méthodes que vous pourriez utiliser ? Faites des recherches et annotez-les. Un document utile peut être trouvé [ici](https://otexts.com/fpp2/accuracy.html)

## [Quiz post-cours](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/44/)

## Révision et auto-apprentissage

Cette leçon aborde uniquement les bases de la prévision de séries temporelles avec ARIMA. Prenez le temps d'approfondir vos connaissances en explorant [ce dépôt](https://microsoft.github.io/forecasting/) et ses différents types de modèles pour découvrir d'autres façons de construire des modèles de séries temporelles.

## Devoir

[Un nouveau modèle ARIMA](assignment.md)

---

**Avertissement** :  
Ce document a été traduit à l'aide du service de traduction automatique [Co-op Translator](https://github.com/Azure/co-op-translator). Bien que nous nous efforcions d'assurer l'exactitude, veuillez noter que les traductions automatisées peuvent contenir des erreurs ou des inexactitudes. Le document original dans sa langue d'origine doit être considéré comme la source faisant autorité. Pour des informations critiques, il est recommandé de faire appel à une traduction professionnelle humaine. Nous déclinons toute responsabilité en cas de malentendus ou d'interprétations erronées résultant de l'utilisation de cette traduction.