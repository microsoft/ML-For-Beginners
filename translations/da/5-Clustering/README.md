# Klyngemodeller til maskinlæring

Klyngedannelse er en maskinlæringsopgave, hvor man søger at finde objekter, der ligner hinanden, og gruppere disse i grupper kaldet klynger. Det, der adskiller klyngedannelse fra andre tilgange i maskinlæring, er, at tingene sker automatisk; faktisk kan man sige, at det er det modsatte af overvåget læring.

## Regionalt emne: klyngemodeller til en nigeriansk publikums musiksmag 🎧

Nigerias mangfoldige publikum har alsidig musiksmag. Ved hjælp af data skrabet fra Spotify (inspireret af [denne artikel](https://towardsdatascience.com/country-wise-visual-analysis-of-music-taste-using-spotify-api-seaborn-in-python-77f5b749b421)) ser vi på noget musik, der er populært i Nigeria. Dette datasæt indeholder data om forskellige sanges 'danceability'-score, 'acousticness', lydstyrke, 'speechiness', popularitet og energi. Det bliver interessant at opdage mønstre i disse data!

![En pladespiller](../../../translated_images/da/turntable.f2b86b13c53302dc.webp)

> Foto af <a href="https://unsplash.com/@marcelalaskoski?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Marcela Laskoski</a> på <a href="https://unsplash.com/s/photos/nigerian-music?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Unsplash</a>
  
I denne serie af lektioner vil du opdage nye måder at analysere data på ved hjælp af klyngedannelse. Klyngedannelse er særlig nyttig, når dit datasæt mangler labels. Hvis det har labels, kan klassifikationsteknikker, som dem du lærte i tidligere lektioner, være mere nyttige. Men i tilfælde, hvor du ønsker at gruppere ulabellede data, er klyngedannelse en god måde at opdage mønstre på.

> Der findes nyttige low-code værktøjer, der kan hjælpe dig med at lære at arbejde med klyngemodeller. Prøv [Azure ML til denne opgave](https://docs.microsoft.com/learn/modules/create-clustering-model-azure-machine-learning-designer/?WT.mc_id=academic-77952-leestott)

## Lektioner

1. [Introduktion til klyngedannelse](1-Visualize/README.md)
2. [K-Means klyngedannelse](2-K-Means/README.md)

## Credits

Disse lektioner blev skrevet med 🎶 af [Jen Looper](https://www.twitter.com/jenlooper) med nyttige anmeldelser af [Rishit Dagli](https://rishit_dagli/) og [Muhammad Sakib Khan Inan](https://twitter.com/Sakibinan).

Datasættet [Nigerian Songs](https://www.kaggle.com/sootersaalu/nigerian-songs-spotify) blev hentet fra Kaggle, skrabet fra Spotify.

Nyttige K-Means eksempler, der hjalp med at skabe denne lektion, inkluderer denne [iris-undersøgelse](https://www.kaggle.com/bburns/iris-exploration-pca-k-means-and-gmm-clustering), denne [introduktions-notebook](https://www.kaggle.com/prashant111/k-means-clustering-with-python), og dette [hypotetiske NGO-eksempel](https://www.kaggle.com/ankandash/pca-k-means-clustering-hierarchical-clustering).

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**Ansvarsfraskrivelse**:
Dette dokument er blevet oversat ved hjælp af AI-oversættelsestjenesten [Co-op Translator](https://github.com/Azure/co-op-translator). Selvom vi bestræber os på nøjagtighed, skal du være opmærksom på, at automatiserede oversættelser kan indeholde fejl eller unøjagtigheder. Det originale dokument på dets oprindelige sprog bør betragtes som den autoritative kilde. For kritisk information anbefales professionel menneskelig oversættelse. Vi påtager os intet ansvar for misforståelser eller fejltolkninger, der opstår som følge af brugen af denne oversættelse.
<!-- CO-OP TRANSLATOR DISCLAIMER END -->