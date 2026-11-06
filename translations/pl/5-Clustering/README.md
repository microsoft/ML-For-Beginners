# Modele klasteryzacji dla uczenia maszynowego

Klasteryzacja to zadanie uczenia maszynowego, które polega na znajdowaniu obiektów do siebie podobnych i grupowaniu ich w tzw. klastry. To, co odróżnia klasteryzację od innych podejść w uczeniu maszynowym, to fakt, że proces zachodzi automatycznie, można wręcz powiedzieć, że jest to przeciwieństwo uczenia nadzorowanego.

## Temat regionalny: modele klasteryzacji dla muzycznych gustów nigeryjskiej publiczności 🎧

Różnorodna publiczność Nigerii ma różnorodne gusta muzyczne. Korzystając z danych pobranych ze Spotify (inspirowanych [tym artykułem](https://towardsdatascience.com/country-wise-visual-analysis-of-music-taste-using-spotify-api-seaborn-in-python-77f5b749b421)), przyjrzyjmy się muzyce popularnej w Nigerii. Ten zbiór danych zawiera informacje o różnych piosenkach takie jak ocena 'danceability', 'acousticness', głośność, 'speechiness', popularność i energia. Będzie ciekawie odkryć wzorce w tych danych!

![Gramofon](../../../translated_images/pl/turntable.f2b86b13c53302dc.webp)

> Zdjęcie autorstwa <a href="https://unsplash.com/@marcelalaskoski?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Marcela Laskoski</a> na <a href="https://unsplash.com/s/photos/nigerian-music?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Unsplash</a>
  
W tej serii lekcji odkryjesz nowe sposoby analizowania danych z wykorzystaniem technik klasteryzacji. Klasteryzacja jest szczególnie przydatna, gdy twój zbiór danych nie posiada etykiet. Jeśli etykiety są dostępne, techniki klasyfikacyjne, takie jak te poznane w poprzednich lekcjach, mogą być bardziej użyteczne. Ale w przypadkach, gdy chcesz pogrupować dane nieoznaczone, klasteryzacja jest świetnym sposobem na odkrywanie wzorców.

> Dostępne są przydatne narzędzia low-code, które mogą pomóc Ci nauczyć się pracy z modelami klasteryzacji. Wypróbuj [Azure ML do tego zadania](https://docs.microsoft.com/learn/modules/create-clustering-model-azure-machine-learning-designer/?WT.mc_id=academic-77952-leestott)

## Lekcje

1. [Wprowadzenie do klasteryzacji](1-Visualize/README.md)
2. [Klasteryzacja K-średnich](2-K-Means/README.md)

## Podziękowania

Te lekcje zostały napisane z 🎶 przez [Jen Looper](https://www.twitter.com/jenlooper) z pomocnymi recenzjami od [Rishit Dagli](https://rishit_dagli/) i [Muhammada Sakib Khana Inana](https://twitter.com/Sakibinan).

Zbiór danych [Nigerian Songs](https://www.kaggle.com/sootersaalu/nigerian-songs-spotify) pochodzi z Kaggle, pozyskany ze Spotify.

Użyteczne przykłady K-średnich, które pomogły w stworzeniu tej lekcji, to między innymi [eksploracja irysów](https://www.kaggle.com/bburns/iris-exploration-pca-k-means-and-gmm-clustering), ten [notatnik wprowadzający](https://www.kaggle.com/prashant111/k-means-clustering-with-python) oraz ten [hipotetyczny przykład NGO](https://www.kaggle.com/ankandash/pca-k-means-clustering-hierarchical-clustering).

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**Zastrzeżenie**:
Niniejszy dokument został przetłumaczony za pomocą usługi tłumaczenia AI [Co-op Translator](https://github.com/Azure/co-op-translator). Choć dążymy do dokładności, prosimy pamiętać, że automatyczne tłumaczenia mogą zawierać błędy lub niedokładności. Oryginalny dokument w jego języku źródłowym należy uznawać za autorytatywne źródło. W przypadku informacji krytycznych zalecane jest skorzystanie z profesjonalnego tłumaczenia wykonanego przez człowieka. Nie ponosimy odpowiedzialności za jakiekolwiek nieporozumienia lub błędne interpretacje wynikające z użycia tego tłumaczenia.
<!-- CO-OP TRANSLATOR DISCLAIMER END -->