<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "e0b75f73e4a90d45181dc5581fe2ef5c",
  "translation_date": "2025-09-05T08:23:42+00:00",
  "source_file": "3-Web-App/1-Web-App/README.md",
  "language_code": "pl"
}
-->
# Zbuduj aplikację webową wykorzystującą model ML

W tej lekcji nauczysz się trenować model ML na zestawie danych, który jest dosłownie nie z tego świata: _obserwacje UFO z ostatniego stulecia_, pochodzące z bazy danych NUFORC.

Dowiesz się:

- Jak „zapisać” wytrenowany model za pomocą Pickle
- Jak używać tego modelu w aplikacji Flask

Kontynuujemy pracę z notebookami, aby oczyścić dane i wytrenować model, ale możesz zrobić krok dalej, eksplorując użycie modelu „w terenie”, czyli w aplikacji webowej.

Aby to zrobić, musisz zbudować aplikację webową za pomocą Flask.

## [Quiz przed wykładem](https://ff-quizzes.netlify.app/en/ml/)

## Budowanie aplikacji

Istnieje kilka sposobów na budowanie aplikacji webowych, które wykorzystują modele uczenia maszynowego. Architektura Twojej aplikacji webowej może wpłynąć na sposób trenowania modelu. Wyobraź sobie, że pracujesz w firmie, gdzie zespół zajmujący się analizą danych wytrenował model, który chcesz wykorzystać w aplikacji.

### Rozważania

Jest wiele pytań, które musisz sobie zadać:

- **Czy to aplikacja webowa czy mobilna?** Jeśli budujesz aplikację mobilną lub chcesz używać modelu w kontekście IoT, możesz skorzystać z [TensorFlow Lite](https://www.tensorflow.org/lite/) i używać modelu w aplikacji na Androida lub iOS.
- **Gdzie będzie znajdować się model?** W chmurze czy lokalnie?
- **Wsparcie offline.** Czy aplikacja musi działać offline?
- **Jakiej technologii użyto do trenowania modelu?** Wybrana technologia może wpłynąć na narzędzia, które musisz użyć.
    - **Użycie TensorFlow.** Jeśli trenujesz model za pomocą TensorFlow, ekosystem ten umożliwia konwersję modelu TensorFlow do użycia w aplikacji webowej za pomocą [TensorFlow.js](https://www.tensorflow.org/js/).
    - **Użycie PyTorch.** Jeśli budujesz model za pomocą biblioteki takiej jak [PyTorch](https://pytorch.org/), masz możliwość eksportowania go w formacie [ONNX](https://onnx.ai/) (Open Neural Network Exchange) do użycia w aplikacjach webowych w JavaScript, które mogą korzystać z [Onnx Runtime](https://www.onnxruntime.ai/). Ta opcja zostanie omówiona w przyszłej lekcji dla modelu wytrenowanego w Scikit-learn.
    - **Użycie Lobe.ai lub Azure Custom Vision.** Jeśli korzystasz z systemu ML SaaS (Software as a Service) takiego jak [Lobe.ai](https://lobe.ai/) lub [Azure Custom Vision](https://azure.microsoft.com/services/cognitive-services/custom-vision-service/?WT.mc_id=academic-77952-leestott) do trenowania modelu, tego typu oprogramowanie oferuje sposoby eksportowania modelu na różne platformy, w tym budowanie dedykowanego API, które można zapytać w chmurze za pomocą aplikacji online.

Masz również możliwość zbudowania całej aplikacji webowej w Flask, która mogłaby trenować model bezpośrednio w przeglądarce. Można to również zrobić za pomocą TensorFlow.js w kontekście JavaScript.

Na nasze potrzeby, ponieważ pracowaliśmy z notebookami opartymi na Pythonie, przyjrzyjmy się krokom, które musisz podjąć, aby wyeksportować wytrenowany model z takiego notebooka do formatu czytelnego dla aplikacji webowej zbudowanej w Pythonie.

## Narzędzia

Do tego zadania potrzebujesz dwóch narzędzi: Flask i Pickle, oba działające na Pythonie.

✅ Co to jest [Flask](https://palletsprojects.com/p/flask/)? Zdefiniowany jako „mikro-framework” przez jego twórców, Flask oferuje podstawowe funkcje frameworków webowych, używając Pythona i silnika szablonów do budowania stron internetowych. Zobacz [ten moduł Learn](https://docs.microsoft.com/learn/modules/python-flask-build-ai-web-app?WT.mc_id=academic-77952-leestott), aby poćwiczyć budowanie aplikacji w Flask.

✅ Co to jest [Pickle](https://docs.python.org/3/library/pickle.html)? Pickle 🥒 to moduł Pythona, który serializuje i deserializuje strukturę obiektów Pythona. Kiedy „pickle'ujesz” model, serializujesz lub spłaszczasz jego strukturę, aby można było go używać w aplikacji webowej. Uważaj: Pickle nie jest z natury bezpieczny, więc bądź ostrożny, jeśli zostaniesz poproszony o „odpickle'owanie” pliku. Plik zapisany za pomocą Pickle ma rozszerzenie `.pkl`.

## Ćwiczenie - oczyszczanie danych

W tej lekcji użyjesz danych z 80 000 obserwacji UFO, zebranych przez [NUFORC](https://nuforc.org) (Narodowe Centrum Zgłaszania Obserwacji UFO). Dane te zawierają interesujące opisy obserwacji UFO, na przykład:

- **Długi opis przykładowy.** „Mężczyzna wyłania się z promienia światła, który oświetla trawiasty teren w nocy, i biegnie w kierunku parkingu Texas Instruments”.
- **Krótki opis przykładowy.** „Światła nas goniły”.

Arkusz [ufos.csv](../../../../3-Web-App/1-Web-App/data/ufos.csv) zawiera kolumny dotyczące `miasta`, `stanu` i `kraju`, w którym miała miejsce obserwacja, `kształtu` obiektu oraz jego `szerokości geograficznej` i `długości geograficznej`.

W pustym [notebooku](../../../../3-Web-App/1-Web-App/notebook.ipynb) dołączonym do tej lekcji:

1. Zaimportuj `pandas`, `matplotlib` i `numpy`, jak robiłeś to w poprzednich lekcjach, oraz zaimportuj arkusz ufos. Możesz zobaczyć próbkę zestawu danych:

    ```python
    import pandas as pd
    import numpy as np
    
    ufos = pd.read_csv('./data/ufos.csv')
    ufos.head()
    ```

1. Przekształć dane ufos w mały dataframe z nowymi tytułami. Sprawdź unikalne wartości w polu `Country`.

    ```python
    ufos = pd.DataFrame({'Seconds': ufos['duration (seconds)'], 'Country': ufos['country'],'Latitude': ufos['latitude'],'Longitude': ufos['longitude']})
    
    ufos.Country.unique()
    ```

1. Teraz możesz zmniejszyć ilość danych, z którymi musimy pracować, usuwając wszelkie wartości null i importując tylko obserwacje trwające od 1 do 60 sekund:

    ```python
    ufos.dropna(inplace=True)
    
    ufos = ufos[(ufos['Seconds'] >= 1) & (ufos['Seconds'] <= 60)]
    
    ufos.info()
    ```

1. Zaimportuj bibliotekę `LabelEncoder` z Scikit-learn, aby przekształcić wartości tekstowe dla krajów na liczby:

    ✅ LabelEncoder koduje dane alfabetycznie

    ```python
    from sklearn.preprocessing import LabelEncoder
    
    ufos['Country'] = LabelEncoder().fit_transform(ufos['Country'])
    
    ufos.head()
    ```

    Twoje dane powinny wyglądać tak:

    ```output
    	Seconds	Country	Latitude	Longitude
    2	20.0	3		53.200000	-2.916667
    3	20.0	4		28.978333	-96.645833
    14	30.0	4		35.823889	-80.253611
    23	60.0	4		45.582778	-122.352222
    24	3.0		3		51.783333	-0.783333
    ```

## Ćwiczenie - budowanie modelu

Teraz możesz przygotować się do trenowania modelu, dzieląc dane na grupę treningową i testową.

1. Wybierz trzy cechy, na których chcesz trenować jako wektor X, a wektor y będzie `Country`. Chcesz móc wprowadzić `Seconds`, `Latitude` i `Longitude`, aby uzyskać identyfikator kraju do zwrócenia.

    ```python
    from sklearn.model_selection import train_test_split
    
    Selected_features = ['Seconds','Latitude','Longitude']
    
    X = ufos[Selected_features]
    y = ufos['Country']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    ```

1. Wytrenuj model za pomocą regresji logistycznej:

    ```python
    from sklearn.metrics import accuracy_score, classification_report
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    
    print(classification_report(y_test, predictions))
    print('Predicted labels: ', predictions)
    print('Accuracy: ', accuracy_score(y_test, predictions))
    ```

Dokładność jest całkiem niezła **(około 95%)**, co nie jest zaskakujące, ponieważ `Country` i `Latitude/Longitude` są ze sobą powiązane.

Model, który stworzyłeś, nie jest bardzo rewolucyjny, ponieważ powinieneś być w stanie wywnioskować `Country` na podstawie jego `Latitude` i `Longitude`, ale to dobre ćwiczenie, aby spróbować trenować na surowych danych, które oczyściłeś, wyeksportowałeś, a następnie użyć tego modelu w aplikacji webowej.

## Ćwiczenie - „pickle'owanie” modelu

Teraz czas na _pickle'owanie_ modelu! Możesz to zrobić w kilku liniach kodu. Po _pickle'owaniu_ załaduj zapisany model i przetestuj go na próbce danych zawierającej wartości dla sekund, szerokości i długości geograficznej.

```python
import pickle
model_filename = 'ufo-model.pkl'
pickle.dump(model, open(model_filename,'wb'))

model = pickle.load(open('ufo-model.pkl','rb'))
print(model.predict([[50,44,-12]]))
```

Model zwraca **„3”**, co jest kodem kraju dla Wielkiej Brytanii. Niesamowite! 👽

## Ćwiczenie - budowanie aplikacji Flask

Teraz możesz zbudować aplikację Flask, która wywoła Twój model i zwróci podobne wyniki, ale w bardziej atrakcyjny wizualnie sposób.

1. Zacznij od utworzenia folderu **web-app** obok pliku _notebook.ipynb_, gdzie znajduje się Twój plik _ufo-model.pkl_.

1. W tym folderze utwórz trzy kolejne foldery: **static**, z folderem **css** w środku, oraz **templates**. Powinieneś teraz mieć następujące pliki i katalogi:

    ```output
    web-app/
      static/
        css/
      templates/
    notebook.ipynb
    ufo-model.pkl
    ```

    ✅ Zobacz folder z rozwiązaniem, aby zobaczyć gotową aplikację

1. Pierwszym plikiem do utworzenia w folderze _web-app_ jest plik **requirements.txt**. Podobnie jak _package.json_ w aplikacji JavaScript, ten plik zawiera listę zależności wymaganych przez aplikację. W **requirements.txt** dodaj linie:

    ```text
    scikit-learn
    pandas
    numpy
    flask
    ```

1. Teraz uruchom ten plik, przechodząc do _web-app_:

    ```bash
    cd web-app
    ```

1. W terminalu wpisz `pip install`, aby zainstalować biblioteki wymienione w _requirements.txt_:

    ```bash
    pip install -r requirements.txt
    ```

1. Teraz możesz utworzyć trzy kolejne pliki, aby zakończyć aplikację:

    1. Utwórz **app.py** w katalogu głównym.
    2. Utwórz **index.html** w katalogu _templates_.
    3. Utwórz **styles.css** w katalogu _static/css_.

1. Rozbuduj plik _styles.css_ o kilka stylów:

    ```css
    body {
    	width: 100%;
    	height: 100%;
    	font-family: 'Helvetica';
    	background: black;
    	color: #fff;
    	text-align: center;
    	letter-spacing: 1.4px;
    	font-size: 30px;
    }
    
    input {
    	min-width: 150px;
    }
    
    .grid {
    	width: 300px;
    	border: 1px solid #2d2d2d;
    	display: grid;
    	justify-content: center;
    	margin: 20px auto;
    }
    
    .box {
    	color: #fff;
    	background: #2d2d2d;
    	padding: 12px;
    	display: inline-block;
    }
    ```

1. Następnie rozbuduj plik _index.html_:

    ```html
    <!DOCTYPE html>
    <html>
      <head>
        <meta charset="UTF-8">
        <title>🛸 UFO Appearance Prediction! 👽</title>
        <link rel="stylesheet" href="{{ url_for('static', filename='css/styles.css') }}">
      </head>
    
      <body>
        <div class="grid">
    
          <div class="box">
    
            <p>According to the number of seconds, latitude and longitude, which country is likely to have reported seeing a UFO?</p>
    
            <form action="{{ url_for('predict')}}" method="post">
              <input type="number" name="seconds" placeholder="Seconds" required="required" min="0" max="60" />
              <input type="text" name="latitude" placeholder="Latitude" required="required" />
              <input type="text" name="longitude" placeholder="Longitude" required="required" />
              <button type="submit" class="btn">Predict country where the UFO is seen</button>
            </form>
    
            <p>{{ prediction_text }}</p>
    
          </div>
    
        </div>
    
      </body>
    </html>
    ```

    Zwróć uwagę na szablonowanie w tym pliku. Zauważ składnię „mustache” wokół zmiennych, które będą dostarczane przez aplikację, takich jak tekst predykcji: `{{}}`. Jest też formularz, który przesyła predykcję do trasy `/predict`.

    Na koniec jesteś gotowy, aby zbudować plik Python, który obsługuje model i wyświetla predykcje:

1. W `app.py` dodaj:

    ```python
    import numpy as np
    from flask import Flask, request, render_template
    import pickle
    
    app = Flask(__name__)
    
    model = pickle.load(open("./ufo-model.pkl", "rb"))
    
    
    @app.route("/")
    def home():
        return render_template("index.html")
    
    
    @app.route("/predict", methods=["POST"])
    def predict():
    
        int_features = [int(x) for x in request.form.values()]
        final_features = [np.array(int_features)]
        prediction = model.predict(final_features)
    
        output = prediction[0]
    
        countries = ["Australia", "Canada", "Germany", "UK", "US"]
    
        return render_template(
            "index.html", prediction_text="Likely country: {}".format(countries[output])
        )
    
    
    if __name__ == "__main__":
        app.run(debug=True)
    ```

    > 💡 Wskazówka: kiedy dodasz [`debug=True`](https://www.askpython.com/python-modules/flask/flask-debug-mode) podczas uruchamiania aplikacji webowej za pomocą Flask, wszelkie zmiany wprowadzone w aplikacji będą od razu widoczne bez konieczności ponownego uruchamiania serwera. Uważaj! Nie włączaj tego trybu w aplikacji produkcyjnej.

Jeśli uruchomisz `python app.py` lub `python3 app.py` - Twój serwer webowy uruchomi się lokalnie i będziesz mógł wypełnić krótki formularz, aby uzyskać odpowiedź na swoje palące pytanie o to, gdzie widziano UFO!

Zanim to zrobisz, przyjrzyj się częściom `app.py`:

1. Najpierw ładowane są zależności i aplikacja się uruchamia.
1. Następnie model jest importowany.
1. Następnie na trasie głównej renderowany jest index.html.

Na trasie `/predict` dzieje się kilka rzeczy, gdy formularz jest przesyłany:

1. Zmienne formularza są zbierane i konwertowane na tablicę numpy. Następnie są wysyłane do modelu, a predykcja jest zwracana.
2. Kraje, które chcemy wyświetlić, są ponownie renderowane jako czytelny tekst na podstawie przewidywanego kodu kraju, a ta wartość jest zwracana do index.html, aby została wyrenderowana w szablonie.

Używanie modelu w ten sposób, za pomocą Flask i zapisanego modelu, jest stosunkowo proste. Najtrudniejsze jest zrozumienie, w jakim kształcie muszą być dane, które należy wysłać do modelu, aby uzyskać predykcję. Wszystko zależy od tego, jak model został wytrenowany. Ten model wymaga trzech punktów danych, aby uzyskać predykcję.

W profesjonalnym środowisku widać, jak ważna jest dobra komunikacja między osobami, które trenują model, a tymi, które go używają w aplikacji webowej lub mobilnej. W naszym przypadku to tylko jedna osoba, Ty!

---

## 🚀 Wyzwanie

Zamiast pracować w notebooku i importować model do aplikacji Flask, możesz wytrenować model bezpośrednio w aplikacji Flask! Spróbuj przekształcić swój kod Pythona z notebooka, być może po oczyszczeniu danych, aby trenować model bezpośrednio w aplikacji na trasie `train`. Jakie są zalety i wady takiego podejścia?

## [Quiz po wykładzie](https://ff-quizzes.netlify.app/en/ml/)

## Przegląd i samodzielna nauka

Istnieje wiele sposobów na budowanie aplikacji webowych wykorzystujących modele ML. Zrób listę sposobów, w jakie możesz użyć JavaScript lub Pythona do budowy aplikacji webowej wykorzystującej uczenie maszynowe. Rozważ architekturę: czy model powinien pozostać w aplikacji, czy żyć w chmurze? Jeśli to drugie, jak byś go uzyskał? Narysuj model architektoniczny dla rozwiązania ML w aplikacji webowej.

## Zadanie

[Spróbuj innego modelu](assignment.md)

---

**Zastrzeżenie**:  
Ten dokument został przetłumaczony za pomocą usługi tłumaczeniowej AI [Co-op Translator](https://github.com/Azure/co-op-translator). Chociaż dokładamy wszelkich starań, aby tłumaczenie było precyzyjne, prosimy pamiętać, że automatyczne tłumaczenia mogą zawierać błędy lub nieścisłości. Oryginalny dokument w jego rodzimym języku powinien być uznawany za wiarygodne źródło. W przypadku informacji krytycznych zaleca się skorzystanie z profesjonalnego tłumaczenia wykonanego przez człowieka. Nie ponosimy odpowiedzialności za jakiekolwiek nieporozumienia lub błędne interpretacje wynikające z korzystania z tego tłumaczenia.