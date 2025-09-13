<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "b28a3a4911584062772c537b653ebbc7",
  "translation_date": "2025-09-05T21:25:20+00:00",
  "source_file": "5-Clustering/README.md",
  "language_code": "no"
}
-->
# Klusteringsmodeller for maskinlæring

Klustering er en oppgave innen maskinlæring hvor man forsøker å finne objekter som ligner på hverandre og gruppere disse i grupper kalt klynger. Det som skiller klustering fra andre tilnærminger i maskinlæring, er at ting skjer automatisk. Faktisk kan man si at det er det motsatte av veiledet læring.

## Regionalt tema: klusteringsmodeller for en nigeriansk publikums musikksmak 🎧

Nigerias mangfoldige publikum har varierte musikksmaker. Ved å bruke data hentet fra Spotify (inspirert av [denne artikkelen](https://towardsdatascience.com/country-wise-visual-analysis-of-music-taste-using-spotify-api-seaborn-in-python-77f5b749b421)), la oss se på noe av musikken som er populær i Nigeria. Dette datasettet inkluderer informasjon om ulike sangers 'dansbarhet'-score, 'akustisitet', lydstyrke, 'taleinnhold', popularitet og energi. Det vil være interessant å oppdage mønstre i disse dataene!

![En platespiller](../../../5-Clustering/images/turntable.jpg)

> Foto av <a href="https://unsplash.com/@marcelalaskoski?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Marcela Laskoski</a> på <a href="https://unsplash.com/s/photos/nigerian-music?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Unsplash</a>
  
I denne serien med leksjoner vil du oppdage nye måter å analysere data på ved hjelp av klusteringsteknikker. Klustering er spesielt nyttig når datasettet ditt mangler etiketter. Hvis det har etiketter, kan klassifiseringsteknikker som de du lærte i tidligere leksjoner være mer nyttige. Men i tilfeller der du ønsker å gruppere umerkede data, er klustering en flott måte å oppdage mønstre på.

> Det finnes nyttige lavkodeverktøy som kan hjelpe deg med å lære å arbeide med klusteringsmodeller. Prøv [Azure ML for denne oppgaven](https://docs.microsoft.com/learn/modules/create-clustering-model-azure-machine-learning-designer/?WT.mc_id=academic-77952-leestott)

## Leksjoner

1. [Introduksjon til klustering](1-Visualize/README.md)
2. [K-Means klustering](2-K-Means/README.md)

## Krediteringer

Disse leksjonene ble skrevet med 🎶 av [Jen Looper](https://www.twitter.com/jenlooper) med nyttige vurderinger fra [Rishit Dagli](https://rishit_dagli) og [Muhammad Sakib Khan Inan](https://twitter.com/Sakibinan).

Datasettet [Nigerian Songs](https://www.kaggle.com/sootersaalu/nigerian-songs-spotify) ble hentet fra Kaggle som data fra Spotify.

Nyttige K-Means-eksempler som bidro til å lage denne leksjonen inkluderer denne [iris-utforskningen](https://www.kaggle.com/bburns/iris-exploration-pca-k-means-and-gmm-clustering), denne [introduksjonsnotatboken](https://www.kaggle.com/prashant111/k-means-clustering-with-python), og dette [hypotetiske NGO-eksempelet](https://www.kaggle.com/ankandash/pca-k-means-clustering-hierarchical-clustering).

---

**Ansvarsfraskrivelse**:  
Dette dokumentet er oversatt ved hjelp av AI-oversettelsestjenesten [Co-op Translator](https://github.com/Azure/co-op-translator). Selv om vi tilstreber nøyaktighet, vennligst vær oppmerksom på at automatiske oversettelser kan inneholde feil eller unøyaktigheter. Det originale dokumentet på sitt opprinnelige språk bør anses som den autoritative kilden. For kritisk informasjon anbefales profesjonell menneskelig oversettelse. Vi er ikke ansvarlige for eventuelle misforståelser eller feiltolkninger som oppstår ved bruk av denne oversettelsen.