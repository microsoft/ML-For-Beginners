# Modelli di clustering per il machine learning

Il clustering è un compito di machine learning in cui si cerca di trovare oggetti che si assomigliano e di raggrupparli in gruppi chiamati cluster. Ciò che differenzia il clustering da altri approcci nel machine learning è che le cose avvengono automaticamente, infatti, è corretto dire che è l'opposto dell'apprendimento supervisionato.

## Argomento regionale: modelli di clustering per il gusto musicale di un pubblico nigeriano 🎧

Il pubblico diversificato della Nigeria ha gusti musicali diversi. Utilizzando dati raccolti da Spotify (ispirati da [questo articolo](https://towardsdatascience.com/country-wise-visual-analysis-of-music-taste-using-spotify-api-seaborn-in-python-77f5b749b421)), esaminiamo un po' di musica popolare in Nigeria. Questo dataset include dati su vari brani come il punteggio di 'danceability', 'acousticness', la sonorità (loudness), 'speechiness', la popolarità e l'energia. Sarà interessante scoprire modelli in questi dati!

![Un giradischi](../../../translated_images/it/turntable.f2b86b13c53302dc.webp)

> Foto di <a href="https://unsplash.com/@marcelalaskoski?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Marcela Laskoski</a> su <a href="https://unsplash.com/s/photos/nigerian-music?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Unsplash</a>
  
In questa serie di lezioni, scoprirai nuovi modi per analizzare i dati usando tecniche di clustering. Il clustering è particolarmente utile quando il tuo dataset non ha etichette. Se invece ha etichette, allora le tecniche di classificazione, come quelle apprese nelle lezioni precedenti, potrebbero essere più utili. Ma nei casi in cui si cerca di raggruppare dati non etichettati, il clustering è un ottimo modo per scoprire schemi.

> Esistono strumenti low-code utili che possono aiutarti a imparare a lavorare con i modelli di clustering. Prova [Azure ML per questo compito](https://docs.microsoft.com/learn/modules/create-clustering-model-azure-machine-learning-designer/?WT.mc_id=academic-77952-leestott)

## Lezioni

1. [Introduzione al clustering](1-Visualize/README.md)
2. [Clustering K-Means](2-K-Means/README.md)

## Crediti

Queste lezioni sono state scritte con 🎶 da [Jen Looper](https://www.twitter.com/jenlooper) con revisioni utili di [Rishit Dagli](https://rishit_dagli/) e [Muhammad Sakib Khan Inan](https://twitter.com/Sakibinan).

Il dataset [Nigerian Songs](https://www.kaggle.com/sootersaalu/nigerian-songs-spotify) è stato ottenuto da Kaggle da dati estratti da Spotify.

Esempi utili di K-Means che hanno aiutato a creare questa lezione includono questa [esplorazione iris](https://www.kaggle.com/bburns/iris-exploration-pca-k-means-and-gmm-clustering), questo [notebook introduttivo](https://www.kaggle.com/prashant111/k-means-clustering-with-python), e questo [esempio ipotetico per una ONG](https://www.kaggle.com/ankandash/pca-k-means-clustering-hierarchical-clustering).

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**Disclaimer**:
Questo documento è stato tradotto utilizzando il servizio di traduzione AI [Co-op Translator](https://github.com/Azure/co-op-translator). Sebbene ci impegniamo per garantire la precisione, si prega di notare che le traduzioni automatizzate possono contenere errori o imprecisioni. Il documento originale nella sua lingua nativa deve essere considerato la fonte autorevole. Per informazioni critiche, si raccomanda una traduzione professionale effettuata da un essere umano. Non siamo responsabili per eventuali malintesi o interpretazioni errate derivanti dall’uso di questa traduzione.
<!-- CO-OP TRANSLATOR DISCLAIMER END -->