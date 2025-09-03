<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "9836ff53cfef716ddfd70e06c5f43436",
  "translation_date": "2025-09-03T17:53:23+00:00",
  "source_file": "3-Web-App/README.md",
  "language_code": "pl"
}
-->
# Zbuduj aplikację webową wykorzystującą Twój model ML

W tej części programu nauczania zostaniesz wprowadzony w praktyczny temat związany z uczeniem maszynowym: jak zapisać model Scikit-learn jako plik, który może być używany do przewidywań w aplikacji webowej. Po zapisaniu modelu nauczysz się, jak wykorzystać go w aplikacji webowej zbudowanej w Flask. Najpierw stworzysz model, korzystając z danych dotyczących obserwacji UFO! Następnie zbudujesz aplikację webową, która pozwoli Ci wprowadzić liczbę sekund, szerokość geograficzną i długość geograficzną, aby przewidzieć, który kraj zgłosił obserwację UFO.

![UFO Parking](../../../translated_images/ufo.9e787f5161da9d4d1dafc537e1da09be8210f2ee996cb638aa5cee1d92867a04.pl.jpg)

Zdjęcie autorstwa <a href="https://unsplash.com/@mdherren?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Michaela Herrena</a> na <a href="https://unsplash.com/s/photos/ufo?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Unsplash</a>

## Lekcje

1. [Zbuduj aplikację webową](1-Web-App/README.md)

## Podziękowania

"Zbuduj aplikację webową" zostało napisane z ♥️ przez [Jen Looper](https://twitter.com/jenlooper).

♥️ Quizy zostały napisane przez Rohana Raja.

Zbiór danych pochodzi z [Kaggle](https://www.kaggle.com/NUFORC/ufo-sightings).

Architektura aplikacji webowej została częściowo zasugerowana w [tym artykule](https://towardsdatascience.com/how-to-easily-deploy-machine-learning-models-using-flask-b95af8fe34d4) oraz [tym repozytorium](https://github.com/abhinavsagar/machine-learning-deployment) autorstwa Abhinava Sagara.

---

**Zastrzeżenie**:  
Ten dokument został przetłumaczony za pomocą usługi tłumaczenia AI [Co-op Translator](https://github.com/Azure/co-op-translator). Chociaż dokładamy wszelkich starań, aby tłumaczenie było precyzyjne, prosimy pamiętać, że automatyczne tłumaczenia mogą zawierać błędy lub nieścisłości. Oryginalny dokument w jego języku źródłowym powinien być uznawany za wiarygodne źródło. W przypadku informacji o kluczowym znaczeniu zaleca się skorzystanie z profesjonalnego tłumaczenia przez człowieka. Nie ponosimy odpowiedzialności za jakiekolwiek nieporozumienia lub błędne interpretacje wynikające z użycia tego tłumaczenia.