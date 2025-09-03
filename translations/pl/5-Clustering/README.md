<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "b28a3a4911584062772c537b653ebbc7",
  "translation_date": "2025-09-03T17:01:47+00:00",
  "source_file": "5-Clustering/README.md",
  "language_code": "pl"
}
-->
# Modele klastrowania w uczeniu maszynowym

Klastrowanie to zadanie w uczeniu maszynowym, które polega na znajdowaniu obiektów podobnych do siebie i grupowaniu ich w grupy zwane klastrami. To, co odróżnia klastrowanie od innych podejść w uczeniu maszynowym, to fakt, że proces ten odbywa się automatycznie. W rzeczywistości można powiedzieć, że jest to przeciwieństwo uczenia nadzorowanego.

## Temat regionalny: modele klastrowania dla muzycznych gustów nigeryjskiej publiczności 🎧

Różnorodna publiczność w Nigerii ma zróżnicowane gusta muzyczne. Korzystając z danych pobranych ze Spotify (zainspirowanych [tym artykułem](https://towardsdatascience.com/country-wise-visual-analysis-of-music-taste-using-spotify-api-seaborn-in-python-77f5b749b421)), przyjrzyjmy się niektórym popularnym utworom w Nigerii. Ten zbiór danych zawiera informacje o takich cechach utworów jak: wskaźnik „taneczności”, „akustyczność”, głośność, „mowa”, popularność i energia. Odkrywanie wzorców w tych danych może być bardzo interesujące!

![Gramofon](../../../translated_images/turntable.f2b86b13c53302dc106aa741de9dc96ac372864cf458dd6f879119857aab01da.pl.jpg)

> Zdjęcie autorstwa <a href="https://unsplash.com/@marcelalaskoski?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Marcela Laskoski</a> na <a href="https://unsplash.com/s/photos/nigerian-music?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Unsplash</a>
  
W tej serii lekcji odkryjesz nowe sposoby analizy danych za pomocą technik klastrowania. Klastrowanie jest szczególnie przydatne, gdy Twój zbiór danych nie zawiera etykiet. Jeśli jednak dane mają etykiety, techniki klasyfikacji, takie jak te, które poznałeś w poprzednich lekcjach, mogą być bardziej użyteczne. W przypadkach, gdy chcesz grupować dane bez etykiet, klastrowanie jest świetnym sposobem na odkrywanie wzorców.

> Istnieją przydatne narzędzia low-code, które mogą pomóc w nauce pracy z modelami klastrowania. Wypróbuj [Azure ML do tego zadania](https://docs.microsoft.com/learn/modules/create-clustering-model-azure-machine-learning-designer/?WT.mc_id=academic-77952-leestott)

## Lekcje

1. [Wprowadzenie do klastrowania](1-Visualize/README.md)
2. [Klastrowanie metodą K-Means](2-K-Means/README.md)

## Podziękowania

Te lekcje zostały napisane z 🎶 przez [Jen Looper](https://www.twitter.com/jenlooper) z pomocnymi recenzjami od [Rishit Dagli](https://rishit_dagli) i [Muhammad Sakib Khan Inan](https://twitter.com/Sakibinan).

Zbiór danych [Nigerian Songs](https://www.kaggle.com/sootersaalu/nigerian-songs-spotify) został pozyskany z Kaggle jako dane pobrane ze Spotify.

Przydatne przykłady K-Means, które pomogły w stworzeniu tej lekcji, obejmują tę [eksplorację irysów](https://www.kaggle.com/bburns/iris-exploration-pca-k-means-and-gmm-clustering), ten [wprowadzający notebook](https://www.kaggle.com/prashant111/k-means-clustering-with-python) oraz ten [hipotetyczny przykład NGO](https://www.kaggle.com/ankandash/pca-k-means-clustering-hierarchical-clustering).

---

**Zastrzeżenie**:  
Ten dokument został przetłumaczony za pomocą usługi tłumaczenia AI [Co-op Translator](https://github.com/Azure/co-op-translator). Chociaż dokładamy wszelkich starań, aby zapewnić poprawność tłumaczenia, prosimy pamiętać, że automatyczne tłumaczenia mogą zawierać błędy lub nieścisłości. Oryginalny dokument w jego rodzimym języku powinien być uznawany za autorytatywne źródło. W przypadku informacji o kluczowym znaczeniu zaleca się skorzystanie z profesjonalnego tłumaczenia przez człowieka. Nie ponosimy odpowiedzialności za jakiekolwiek nieporozumienia lub błędne interpretacje wynikające z użycia tego tłumaczenia.