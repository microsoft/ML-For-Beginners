# Klusteringsmodeller för maskininlärning

Klustering är en maskininlärningsuppgift där man försöker hitta objekt som liknar varandra och gruppera dessa i grupper som kallas kluster. Det som skiljer klustering från andra metoder inom maskininlärning är att saker händer automatiskt, faktiskt kan man säga att det är motsatsen till övervakad inlärning.

## Regionalt ämne: klusteringsmodeller för en nigeriansk publik med musiksmak 🎧

Nigerias mångsidiga publik har olika musiksmaker. Med data skrapad från Spotify (inspirerad av [denna artikel](https://towardsdatascience.com/country-wise-visual-analysis-of-music-taste-using-spotify-api-seaborn-in-python-77f5b749b421)) ska vi titta på musik som är populär i Nigeria. Denna datamängd innehåller data om olika låtars 'danceability'-poäng, 'acousticness', volym, 'speechiness', popularitet och energi. Det blir intressant att upptäcka mönster i denna data!

![En skivspelare](../../../translated_images/sv/turntable.f2b86b13c53302dc.webp)

> Foto av <a href="https://unsplash.com/@marcelalaskoski?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Marcela Laskoski</a> på <a href="https://unsplash.com/s/photos/nigerian-music?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Unsplash</a>
  
I denna lektionserie kommer du att upptäcka nya sätt att analysera data med hjälp av klustertekniker. Klustering är särskilt användbart när din datamängd saknar etiketter. Om den däremot har etiketter kan klassificeringstekniker, såsom de du lärde dig i tidigare lektioner, vara mer användbara. Men i fall där du vill gruppera oetiketterad data är klustering ett utmärkt sätt att upptäcka mönster.

> Det finns användbara lågkodverktyg som kan hjälpa dig att lära dig arbeta med klusteringsmodeller. Prova [Azure ML för denna uppgift](https://docs.microsoft.com/learn/modules/create-clustering-model-azure-machine-learning-designer/?WT.mc_id=academic-77952-leestott)

## Lektioner

1. [Introduktion till klustering](1-Visualize/README.md)
2. [K-Means klustering](2-K-Means/README.md)

## Crediteringar

Dessa lektioner skrevs med 🎶 av [Jen Looper](https://www.twitter.com/jenlooper) med hjälpsamma recensioner av [Rishit Dagli](https://rishit_dagli/) och [Muhammad Sakib Khan Inan](https://twitter.com/Sakibinan).

Datamängden [Nigerian Songs](https://www.kaggle.com/sootersaalu/nigerian-songs-spotify) hämtades från Kaggle och skrapades från Spotify.

Användbara K-Means-exempel som hjälpte till att skapa denna lektion inkluderar denna [irisutforskning](https://www.kaggle.com/bburns/iris-exploration-pca-k-means-and-gmm-clustering), denna [introduktionsanteckning](https://www.kaggle.com/prashant111/k-means-clustering-with-python), och detta [hypotetiska NGO-exempel](https://www.kaggle.com/ankandash/pca-k-means-clustering-hierarchical-clustering).

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**Ansvarsfriskrivning**:
Detta dokument har översatts med hjälp av AI-översättningstjänsten [Co-op Translator](https://github.com/Azure/co-op-translator). Även om vi strävar efter noggrannhet, var vänlig notera att automatiska översättningar kan innehålla fel eller brister. Det ursprungliga dokumentet på dess modersmål bör betraktas som den auktoritativa källan. För kritisk information rekommenderas professionell mänsklig översättning. Vi ansvarar inte för några missförstånd eller feltolkningar som uppstår till följd av användningen av denna översättning.
<!-- CO-OP TRANSLATOR DISCLAIMER END -->