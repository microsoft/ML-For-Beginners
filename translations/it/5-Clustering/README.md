# Modelli di clustering per il machine learning

Il clustering è un compito di machine learning che cerca di trovare oggetti che si somigliano e raggrupparli in gruppi chiamati cluster. Ciò che differenzia il clustering da altri approcci nel machine learning è che le cose avvengono automaticamente; infatti, si può dire che è l'opposto dell'apprendimento supervisionato.

## Argomento regionale: modelli di clustering per i gusti musicali del pubblico nigeriano 🎧

Il pubblico nigeriano è molto variegato e ha gusti musicali diversi. Utilizzando dati estratti da Spotify (ispirati da [questo articolo](https://towardsdatascience.com/country-wise-visual-analysis-of-music-taste-using-spotify-api-seaborn-in-python-77f5b749b421)), diamo un'occhiata ad alcune delle musiche popolari in Nigeria. Questo dataset include dati su vari aspetti delle canzoni come il punteggio di 'ballabilità', 'acousticness', volume, 'speechiness', popolarità ed energia. Sarà interessante scoprire dei pattern in questi dati!

![Un giradischi](../../../translated_images/turntable.f2b86b13c53302dc106aa741de9dc96ac372864cf458dd6f879119857aab01da.it.jpg)

> Foto di <a href="https://unsplash.com/@marcelalaskoski?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Marcela Laskoski</a> su <a href="https://unsplash.com/s/photos/nigerian-music?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Unsplash</a>
  
In questa serie di lezioni, scoprirai nuovi modi per analizzare i dati utilizzando tecniche di clustering. Il clustering è particolarmente utile quando il tuo dataset manca di etichette. Se ha etichette, allora tecniche di classificazione come quelle apprese nelle lezioni precedenti potrebbero essere più utili. Ma nei casi in cui si desidera raggruppare dati non etichettati, il clustering è un ottimo modo per scoprire pattern.

> Ci sono utili strumenti low-code che possono aiutarti a imparare a lavorare con i modelli di clustering. Prova [Azure ML per questo compito](https://docs.microsoft.com/learn/modules/create-clustering-model-azure-machine-learning-designer/?WT.mc_id=academic-77952-leestott)

## Lezioni

1. [Introduzione al clustering](1-Visualize/README.md)
2. [Clustering K-Means](2-K-Means/README.md)

## Crediti

Queste lezioni sono state scritte con 🎶 da [Jen Looper](https://www.twitter.com/jenlooper) con utili recensioni di [Rishit Dagli](https://rishit_dagli) e [Muhammad Sakib Khan Inan](https://twitter.com/Sakibinan).

Il dataset [Nigerian Songs](https://www.kaggle.com/sootersaalu/nigerian-songs-spotify) è stato ottenuto da Kaggle come estratto da Spotify.

Esempi utili di K-Means che hanno aiutato nella creazione di questa lezione includono questa [esplorazione dell'iris](https://www.kaggle.com/bburns/iris-exploration-pca-k-means-and-gmm-clustering), questo [notebook introduttivo](https://www.kaggle.com/prashant111/k-means-clustering-with-python), e questo [esempio ipotetico di ONG](https://www.kaggle.com/ankandash/pca-k-means-clustering-hierarchical-clustering).

**Avvertenza**:
Questo documento è stato tradotto utilizzando servizi di traduzione automatica basati su intelligenza artificiale. Anche se ci impegniamo per l'accuratezza, si prega di essere consapevoli che le traduzioni automatiche possono contenere errori o imprecisioni. Il documento originale nella sua lingua madre dovrebbe essere considerato la fonte autorevole. Per informazioni critiche, si raccomanda una traduzione umana professionale. Non siamo responsabili per eventuali malintesi o interpretazioni errate derivanti dall'uso di questa traduzione.