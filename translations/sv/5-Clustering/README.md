<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "b28a3a4911584062772c537b653ebbc7",
  "translation_date": "2025-09-05T21:25:08+00:00",
  "source_file": "5-Clustering/README.md",
  "language_code": "sv"
}
-->
# Klusteringsmodeller för maskininlärning

Klustring är en maskininlärningsuppgift där man försöker hitta objekt som liknar varandra och gruppera dessa i grupper som kallas kluster. Det som skiljer klustring från andra metoder inom maskininlärning är att processen sker automatiskt. Faktum är att det kan sägas vara motsatsen till övervakad inlärning.

## Regionalt ämne: klusteringsmodeller för en nigeriansk publik med musiksmak 🎧

Nigerias mångsidiga publik har en varierad musiksmak. Med hjälp av data hämtad från Spotify (inspirerad av [denna artikel](https://towardsdatascience.com/country-wise-visual-analysis-of-music-taste-using-spotify-api-seaborn-in-python-77f5b749b421)), ska vi titta på musik som är populär i Nigeria. Denna dataset innehåller information om olika låtars "dansbarhet", "akustik", ljudstyrka, "talighet", popularitet och energi. Det kommer att bli intressant att upptäcka mönster i denna data!

![En skivspelare](../../../5-Clustering/images/turntable.jpg)

> Foto av <a href="https://unsplash.com/@marcelalaskoski?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Marcela Laskoski</a> på <a href="https://unsplash.com/s/photos/nigerian-music?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Unsplash</a>
  
I denna serie av lektioner kommer du att upptäcka nya sätt att analysera data med hjälp av klusteringstekniker. Klustring är särskilt användbart när din dataset saknar etiketter. Om den däremot har etiketter kan klassificeringstekniker, som de du lärde dig i tidigare lektioner, vara mer användbara. Men i fall där du vill gruppera oetiketterad data är klustring ett utmärkt sätt att upptäcka mönster.

> Det finns användbara verktyg med låg kod som kan hjälpa dig att arbeta med klusteringsmodeller. Prova [Azure ML för denna uppgift](https://docs.microsoft.com/learn/modules/create-clustering-model-azure-machine-learning-designer/?WT.mc_id=academic-77952-leestott)

## Lektioner

1. [Introduktion till klustring](1-Visualize/README.md)
2. [K-Means klustring](2-K-Means/README.md)

## Krediter

Dessa lektioner skrevs med 🎶 av [Jen Looper](https://www.twitter.com/jenlooper) med hjälpsamma recensioner av [Rishit Dagli](https://rishit_dagli) och [Muhammad Sakib Khan Inan](https://twitter.com/Sakibinan).

Datasetet [Nigerian Songs](https://www.kaggle.com/sootersaalu/nigerian-songs-spotify) hämtades från Kaggle och är baserat på data från Spotify.

Användbara K-Means-exempel som hjälpte till att skapa denna lektion inkluderar denna [iris-analys](https://www.kaggle.com/bburns/iris-exploration-pca-k-means-and-gmm-clustering), denna [introduktionsnotebook](https://www.kaggle.com/prashant111/k-means-clustering-with-python), och detta [hypotetiska NGO-exempel](https://www.kaggle.com/ankandash/pca-k-means-clustering-hierarchical-clustering).

---

**Ansvarsfriskrivning**:  
Detta dokument har översatts med hjälp av AI-översättningstjänsten [Co-op Translator](https://github.com/Azure/co-op-translator). Även om vi strävar efter noggrannhet, vänligen notera att automatiska översättningar kan innehålla fel eller felaktigheter. Det ursprungliga dokumentet på sitt ursprungliga språk bör betraktas som den auktoritativa källan. För kritisk information rekommenderas professionell mänsklig översättning. Vi ansvarar inte för eventuella missförstånd eller feltolkningar som uppstår vid användning av denna översättning.