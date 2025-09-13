<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "b28a3a4911584062772c537b653ebbc7",
  "translation_date": "2025-08-29T20:52:51+00:00",
  "source_file": "5-Clustering/README.md",
  "language_code": "it"
}
-->
# Modelli di clustering per il machine learning

Il clustering è un compito di machine learning che cerca di individuare oggetti simili tra loro e raggrupparli in gruppi chiamati cluster. Ciò che distingue il clustering da altri approcci nel machine learning è che tutto avviene automaticamente; infatti, si può dire che sia l'opposto dell'apprendimento supervisionato.

## Argomento regionale: modelli di clustering per i gusti musicali del pubblico nigeriano 🎧

Il pubblico nigeriano, molto variegato, ha gusti musicali altrettanto diversificati. Utilizzando dati raccolti da Spotify (ispirati a [questo articolo](https://towardsdatascience.com/country-wise-visual-analysis-of-music-taste-using-spotify-api-seaborn-in-python-77f5b749b421)), analizziamo alcune delle canzoni popolari in Nigeria. Questo dataset include informazioni su vari brani, come il punteggio di 'danceability', 'acousticness', volume, 'speechiness', popolarità ed energia. Sarà interessante scoprire i pattern presenti in questi dati!

![Un giradischi](../../../translated_images/turntable.f2b86b13c53302dc106aa741de9dc96ac372864cf458dd6f879119857aab01da.it.jpg)

> Foto di <a href="https://unsplash.com/@marcelalaskoski?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Marcela Laskoski</a> su <a href="https://unsplash.com/s/photos/nigerian-music?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Unsplash</a>
  
In questa serie di lezioni, scoprirai nuovi modi per analizzare i dati utilizzando tecniche di clustering. Il clustering è particolarmente utile quando il tuo dataset non ha etichette. Se invece il dataset ha etichette, allora tecniche di classificazione come quelle che hai imparato nelle lezioni precedenti potrebbero essere più utili. Ma nei casi in cui vuoi raggruppare dati non etichettati, il clustering è un ottimo metodo per scoprire pattern.

> Esistono strumenti low-code utili per imparare a lavorare con modelli di clustering. Prova [Azure ML per questo compito](https://docs.microsoft.com/learn/modules/create-clustering-model-azure-machine-learning-designer/?WT.mc_id=academic-77952-leestott)

## Lezioni

1. [Introduzione al clustering](1-Visualize/README.md)
2. [Clustering con K-Means](2-K-Means/README.md)

## Crediti

Queste lezioni sono state scritte con 🎶 da [Jen Looper](https://www.twitter.com/jenlooper) con utili revisioni di [Rishit Dagli](https://rishit_dagli) e [Muhammad Sakib Khan Inan](https://twitter.com/Sakibinan).

Il dataset [Nigerian Songs](https://www.kaggle.com/sootersaalu/nigerian-songs-spotify) è stato ottenuto da Kaggle e raccolto da Spotify.

Esempi utili di K-Means che hanno aiutato nella creazione di questa lezione includono questa [esplorazione dell'iris](https://www.kaggle.com/bburns/iris-exploration-pca-k-means-and-gmm-clustering), questo [notebook introduttivo](https://www.kaggle.com/prashant111/k-means-clustering-with-python) e questo [esempio ipotetico di ONG](https://www.kaggle.com/ankandash/pca-k-means-clustering-hierarchical-clustering).

---

**Disclaimer**:  
Questo documento è stato tradotto utilizzando il servizio di traduzione automatica [Co-op Translator](https://github.com/Azure/co-op-translator). Sebbene ci impegniamo per garantire l'accuratezza, si prega di notare che le traduzioni automatiche possono contenere errori o imprecisioni. Il documento originale nella sua lingua nativa dovrebbe essere considerato la fonte autorevole. Per informazioni critiche, si raccomanda una traduzione professionale effettuata da un traduttore umano. Non siamo responsabili per eventuali incomprensioni o interpretazioni errate derivanti dall'uso di questa traduzione.