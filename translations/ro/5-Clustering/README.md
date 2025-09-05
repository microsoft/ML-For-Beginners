<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "b28a3a4911584062772c537b653ebbc7",
  "translation_date": "2025-09-05T15:39:48+00:00",
  "source_file": "5-Clustering/README.md",
  "language_code": "ro"
}
-->
# Modele de clustering pentru învățarea automată

Clustering-ul este o sarcină de învățare automată care urmărește să găsească obiecte ce seamănă între ele și să le grupeze în grupuri numite clustere. Ceea ce diferențiază clustering-ul de alte abordări în învățarea automată este faptul că procesul se desfășoară automat; de fapt, putem spune că este opusul învățării supravegheate.

## Subiect regional: modele de clustering pentru gusturile muzicale ale publicului din Nigeria 🎧

Publicul divers din Nigeria are gusturi muzicale variate. Folosind date colectate de pe Spotify (inspirate de [acest articol](https://towardsdatascience.com/country-wise-visual-analysis-of-music-taste-using-spotify-api-seaborn-in-python-77f5b749b421)), să analizăm câteva melodii populare în Nigeria. Acest set de date include informații despre scorul de 'danceability', 'acousticness', intensitate, 'speechiness', popularitate și energie ale diferitelor melodii. Va fi interesant să descoperim modele în aceste date!

![Un pick-up](../../../5-Clustering/images/turntable.jpg)

> Fotografie realizată de <a href="https://unsplash.com/@marcelalaskoski?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Marcela Laskoski</a> pe <a href="https://unsplash.com/s/photos/nigerian-music?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Unsplash</a>
  
În această serie de lecții, veți descoperi noi modalități de a analiza date folosind tehnici de clustering. Clustering-ul este deosebit de util atunci când setul de date nu are etichete. Dacă are etichete, atunci tehnicile de clasificare, precum cele pe care le-ați învățat în lecțiile anterioare, ar putea fi mai utile. Dar în cazurile în care doriți să grupați date neetichetate, clustering-ul este o metodă excelentă pentru a descoperi modele.

> Există instrumente low-code utile care vă pot ajuta să învățați cum să lucrați cu modele de clustering. Încercați [Azure ML pentru această sarcină](https://docs.microsoft.com/learn/modules/create-clustering-model-azure-machine-learning-designer/?WT.mc_id=academic-77952-leestott)

## Lecții

1. [Introducere în clustering](1-Visualize/README.md)
2. [Clustering K-Means](2-K-Means/README.md)

## Credite

Aceste lecții au fost scrise cu 🎶 de [Jen Looper](https://www.twitter.com/jenlooper) cu recenzii utile de la [Rishit Dagli](https://rishit_dagli) și [Muhammad Sakib Khan Inan](https://twitter.com/Sakibinan).

Setul de date [Nigerian Songs](https://www.kaggle.com/sootersaalu/nigerian-songs-spotify) a fost preluat de pe Kaggle, fiind colectat de pe Spotify.

Exemple utile de K-Means care au ajutat la crearea acestei lecții includ această [explorare a irisului](https://www.kaggle.com/bburns/iris-exploration-pca-k-means-and-gmm-clustering), acest [notebook introductiv](https://www.kaggle.com/prashant111/k-means-clustering-with-python) și acest [exemplu ipotetic de ONG](https://www.kaggle.com/ankandash/pca-k-means-clustering-hierarchical-clustering).

---

**Declinare de responsabilitate**:  
Acest document a fost tradus folosind serviciul de traducere AI [Co-op Translator](https://github.com/Azure/co-op-translator). Deși ne străduim să asigurăm acuratețea, vă rugăm să rețineți că traducerile automate pot conține erori sau inexactități. Documentul original în limba sa natală ar trebui considerat sursa autoritară. Pentru informații critice, se recomandă traducerea profesională realizată de un specialist uman. Nu ne asumăm responsabilitatea pentru eventualele neînțelegeri sau interpretări greșite care pot apărea din utilizarea acestei traduceri.