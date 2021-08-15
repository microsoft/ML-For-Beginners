# Modelli di clustering per machine learning

Il clustering è un'attività di machine learning che cerca di trovare oggetti che si assomigliano per raggrupparli in gruppi chiamati cluster. Ciò che differenzia il clustering da altri approcci in machine learning è che le cose accadono automaticamente, infatti, è giusto dire che è l'opposto dell'apprendimento supervisionato.

## Tema regionale: modelli di clustering per il gusto musicale di un pubblico nigeriano 🎧

Il pubblico eterogeneo della Nigeria ha gusti musicali diversi. Usando i dati recuperati da Spotify (ispirato da [questo articolo](https://towardsdatascience.com/country-wise-visual-analysis-of-music-taste-using-spotify-api-seaborn-in-python-77f5b749b421), si dà un'occhiata a un po' di musica popolare in Nigeria. Questo insieme di dati include dati sul punteggio di "danzabilità", acustica, volume, "speechness" (un numero compreso tra zero e uno che indica la probabilità che un particolare file audio sia parlato - n.d.t.) popolarità ed energia di varie canzoni. Sarà interessante scoprire modelli in questi dati!

![Un giradischi](../images/turntable.jpg)

Foto di <a href="https://unsplash.com/@marcelalaskoski?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Marcela Laskoski</a> su <a href="https://unsplash.com/s/photos/nigerian-music?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Unsplash</a>

In questa serie di lezioni si scopriranno nuovi modi per analizzare i dati utilizzando tecniche di clustering. Il clustering è particolarmente utile quando l'insieme di dati non ha etichette. Se ha etichette, le tecniche di classificazione come quelle apprese nelle lezioni precedenti potrebbero essere più utili. Ma nei casi in cui si sta cercando di raggruppare dati senza etichetta, il clustering è un ottimo modo per scoprire i modelli.

> Esistono utili strumenti a basso codice che possono aiutare a imparare a lavorare con i modelli di clustering. Si provi [Azure ML per questa attività](https://docs.microsoft.com/learn/modules/create-clustering-model-azure-machine-learning-designer/?WT.mc_id=academic-15963-cxa)

## Lezioni


1. [Introduzione al clustering](../1-Visualize/translations/README.it.md)
2. [K-Means clustering](../2-K-Means/translations/README.it.md)

## Crediti

Queste lezioni sono state scritte con 🎶 da [Jen Looper](https://www.twitter.com/jenlooper) con utili recensioni di [Rishit Dagli](https://rishit_dagli) e [Muhammad Sakib Khan Inan](https://twitter.com/Sakibinan).

L'insieme di dati [Nigerian Songs](https://www.kaggle.com/sootersaalu/nigerian-songs-spotify) è stato prelevato da Kaggle, a sua volta recuperato da Spotify.

Esempi utili di K-Means che hanno aiutato nella creazione di questa lezione includono questa [esplorazione dell'iride](https://www.kaggle.com/bburns/iris-exploration-pca-k-means-and-gmm-clustering), questo [notebook introduttivo](https://www.kaggle.com/prashant111/k-means-clustering-with-python) e questo [ipotetico esempio di ONG](https://www.kaggle.com/ankandash/pca-k-means-clustering-hierarchical-clustering).