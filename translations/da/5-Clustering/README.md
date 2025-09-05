<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "b28a3a4911584062772c537b653ebbc7",
  "translation_date": "2025-09-04T23:57:54+00:00",
  "source_file": "5-Clustering/README.md",
  "language_code": "da"
}
-->
# Klyngemodeller til maskinlæring

Klyngedannelse er en maskinlæringsopgave, hvor man forsøger at finde objekter, der ligner hinanden, og gruppere dem i grupper kaldet klynger. Det, der adskiller klyngedannelse fra andre tilgange inden for maskinlæring, er, at processen sker automatisk. Faktisk kan man sige, at det er det modsatte af superviseret læring.

## Regionalt emne: klyngemodeller for en nigeriansk målgruppes musiksmag 🎧

Nigerias mangfoldige befolkning har en lige så mangfoldig musiksmag. Ved at bruge data hentet fra Spotify (inspireret af [denne artikel](https://towardsdatascience.com/country-wise-visual-analysis-of-music-taste-using-spotify-api-seaborn-in-python-77f5b749b421)), kan vi se på noget af den musik, der er populær i Nigeria. Dette datasæt indeholder information om forskellige sanges 'danceability'-score, 'acousticness', lydstyrke, 'speechiness', popularitet og energi. Det bliver spændende at opdage mønstre i disse data!

![En pladespiller](../../../5-Clustering/images/turntable.jpg)

> Foto af <a href="https://unsplash.com/@marcelalaskoski?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Marcela Laskoski</a> på <a href="https://unsplash.com/s/photos/nigerian-music?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Unsplash</a>
  
I denne lektionsserie vil du opdage nye måder at analysere data på ved hjælp af klyngedannelsesteknikker. Klyngedannelse er særligt nyttigt, når dit datasæt mangler labels. Hvis det har labels, kan klassifikationsteknikker, som dem du lærte i tidligere lektioner, være mere nyttige. Men i tilfælde, hvor du ønsker at gruppere ulabellede data, er klyngedannelse en fantastisk måde at opdage mønstre på.

> Der findes nyttige low-code værktøjer, der kan hjælpe dig med at arbejde med klyngemodeller. Prøv [Azure ML til denne opgave](https://docs.microsoft.com/learn/modules/create-clustering-model-azure-machine-learning-designer/?WT.mc_id=academic-77952-leestott)

## Lektioner

1. [Introduktion til klyngedannelse](1-Visualize/README.md)
2. [K-Means klyngedannelse](2-K-Means/README.md)

## Credits

Disse lektioner blev skrevet med 🎶 af [Jen Looper](https://www.twitter.com/jenlooper) med værdifulde anmeldelser fra [Rishit Dagli](https://rishit_dagli) og [Muhammad Sakib Khan Inan](https://twitter.com/Sakibinan).

Datasættet [Nigerian Songs](https://www.kaggle.com/sootersaalu/nigerian-songs-spotify) blev hentet fra Kaggle og er baseret på data fra Spotify.

Nyttige K-Means eksempler, der bidrog til at skabe denne lektion, inkluderer denne [iris-undersøgelse](https://www.kaggle.com/bburns/iris-exploration-pca-k-means-and-gmm-clustering), denne [introducerende notebook](https://www.kaggle.com/prashant111/k-means-clustering-with-python), og dette [hypotetiske NGO-eksempel](https://www.kaggle.com/ankandash/pca-k-means-clustering-hierarchical-clustering).

---

**Ansvarsfraskrivelse**:  
Dette dokument er blevet oversat ved hjælp af AI-oversættelsestjenesten [Co-op Translator](https://github.com/Azure/co-op-translator). Selvom vi bestræber os på nøjagtighed, skal du være opmærksom på, at automatiserede oversættelser kan indeholde fejl eller unøjagtigheder. Det originale dokument på dets oprindelige sprog bør betragtes som den autoritative kilde. For kritisk information anbefales professionel menneskelig oversættelse. Vi er ikke ansvarlige for eventuelle misforståelser eller fejltolkninger, der måtte opstå som følge af brugen af denne oversættelse.