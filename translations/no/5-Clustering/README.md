# Klustringsmodeller for maskinlæring

Klynging er en oppgave innen maskinlæring hvor målet er å finne objekter som ligner på hverandre og gruppere disse i grupper kalt klynger. Det som skiller klynging fra andre tilnærminger innen maskinlæring, er at ting skjer automatisk; faktisk kan man si det er motsatsen til veiledet læring.

## Regionalt tema: klustringsmodeller for en nigeriansk publikums musikksmak 🎧

Nigerias mangfoldige publikum har mangfoldige musikksmaker. Ved å bruke data hentet fra Spotify (inspirert av [denne artikkelen](https://towardsdatascience.com/country-wise-visual-analysis-of-music-taste-using-spotify-api-seaborn-in-python-77f5b749b421)), la oss se på noe musikk som er populær i Nigeria. Dette datasettet inkluderer data om forskjellige sangers 'danceability'-score, 'acousticness', volum, 'speechiness', popularitet og energi. Det blir interessant å oppdage mønstre i disse dataene!

![En platespiller](../../../translated_images/no/turntable.f2b86b13c53302dc.webp)

> Foto av <a href="https://unsplash.com/@marcelalaskoski?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Marcela Laskoski</a> på <a href="https://unsplash.com/s/photos/nigerian-music?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Unsplash</a>
  
I denne serien av leksjoner vil du oppdage nye måter å analysere data på ved hjelp av klustringsteknikker. Klustring er spesielt nyttig når datasettet ditt mangler etiketter. Hvis datasettet har etiketter, kan klassifiseringsteknikker som du lærte i tidligere leksjoner, være mer nyttige. Men i tilfeller hvor du ønsker å gruppere umerkede data, er klustring en utmerket måte å oppdage mønstre på.

> Det finnes nyttige lavkodeverktøy som kan hjelpe deg å lære om arbeid med klustringsmodeller. Prøv [Azure ML for denne oppgaven](https://docs.microsoft.com/learn/modules/create-clustering-model-azure-machine-learning-designer/?WT.mc_id=academic-77952-leestott)

## Leksjoner

1. [Introduksjon til klustring](1-Visualize/README.md)
2. [K-Means-klustring](2-K-Means/README.md)

## Krediteringer

Disse leksjonene ble skrevet med 🎶 av [Jen Looper](https://www.twitter.com/jenlooper) med hjelpsomme gjennomganger av [Rishit Dagli](https://rishit_dagli/) og [Muhammad Sakib Khan Inan](https://twitter.com/Sakibinan).

Datasettet [Nigerian Songs](https://www.kaggle.com/sootersaalu/nigerian-songs-spotify) ble hentet fra Kaggle som skrapet fra Spotify.

Nyttige K-Means-eksempler som hjalp i å lage denne leksjonen inkluderer denne [iris-explorasjonen](https://www.kaggle.com/bburns/iris-exploration-pca-k-means-and-gmm-clustering), denne [introduksjonsnotatboken](https://www.kaggle.com/prashant111/k-means-clustering-with-python), og dette [hypotetiske NGO-eksemplet](https://www.kaggle.com/ankandash/pca-k-means-clustering-hierarchical-clustering).

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**Ansvarsfraskrivelse**:
Dette dokumentet er oversatt ved hjelp av AI-oversettelsestjenesten [Co-op Translator](https://github.com/Azure/co-op-translator). Selv om vi streber etter nøyaktighet, vær oppmerksom på at automatiske oversettelser kan inneholde feil eller unøyaktigheter. Det opprinnelige dokumentet på originalspråket skal betraktes som den autoritative kilden. For kritisk informasjon anbefales profesjonell menneskelig oversettelse. Vi er ikke ansvarlige for eventuelle misforståelser eller feiltolkninger som oppstår ved bruk av denne oversettelsen.
<!-- CO-OP TRANSLATOR DISCLAIMER END -->