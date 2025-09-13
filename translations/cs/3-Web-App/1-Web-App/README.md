<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "e0b75f73e4a90d45181dc5581fe2ef5c",
  "translation_date": "2025-09-05T00:37:25+00:00",
  "source_file": "3-Web-App/1-Web-App/README.md",
  "language_code": "cs"
}
-->
# Vytvořte webovou aplikaci pro použití ML modelu

V této lekci budete trénovat ML model na datové sadě, která je doslova mimo tento svět: _pozorování UFO za poslední století_, získané z databáze NUFORC.

Naučíte se:

- Jak 'pickle' trénovaný model
- Jak použít tento model v aplikaci Flask

Pokračujeme v používání notebooků pro čištění dat a trénování modelu, ale můžete proces posunout o krok dál tím, že prozkoumáte použití modelu „v terénu“, tak říkajíc: v webové aplikaci.

K tomu je potřeba vytvořit webovou aplikaci pomocí Flask.

## [Kvíz před lekcí](https://ff-quizzes.netlify.app/en/ml/)

## Vytvoření aplikace

Existuje několik způsobů, jak vytvořit webové aplikace, které využívají modely strojového učení. Vaše webová architektura může ovlivnit způsob, jakým je model trénován. Představte si, že pracujete ve firmě, kde skupina datových vědců vytrénovala model, který chtějí použít ve vaší aplikaci.

### Úvahy

Je třeba si položit mnoho otázek:

- **Je to webová aplikace nebo mobilní aplikace?** Pokud vytváříte mobilní aplikaci nebo potřebujete model použít v kontextu IoT, můžete použít [TensorFlow Lite](https://www.tensorflow.org/lite/) a použít model v aplikaci pro Android nebo iOS.
- **Kde bude model umístěn?** V cloudu nebo lokálně?
- **Podpora offline režimu.** Musí aplikace fungovat offline?
- **Jaká technologie byla použita k trénování modelu?** Zvolená technologie může ovlivnit nástroje, které budete muset použít.
    - **Použití TensorFlow.** Pokud trénujete model pomocí TensorFlow, například tento ekosystém poskytuje možnost převést model TensorFlow pro použití ve webové aplikaci pomocí [TensorFlow.js](https://www.tensorflow.org/js/).
    - **Použití PyTorch.** Pokud vytváříte model pomocí knihovny jako [PyTorch](https://pytorch.org/), máte možnost exportovat jej ve formátu [ONNX](https://onnx.ai/) (Open Neural Network Exchange) pro použití ve webových aplikacích JavaScriptu, které mohou používat [Onnx Runtime](https://www.onnxruntime.ai/). Tuto možnost prozkoumáme v budoucí lekci pro model trénovaný pomocí Scikit-learn.
    - **Použití Lobe.ai nebo Azure Custom Vision.** Pokud používáte ML SaaS (Software as a Service) systém, jako je [Lobe.ai](https://lobe.ai/) nebo [Azure Custom Vision](https://azure.microsoft.com/services/cognitive-services/custom-vision-service/?WT.mc_id=academic-77952-leestott) k trénování modelu, tento typ softwaru poskytuje způsoby exportu modelu pro mnoho platforem, včetně vytvoření vlastního API, které lze dotazovat v cloudu vaší online aplikací.

Máte také možnost vytvořit celou webovou aplikaci Flask, která by byla schopna model trénovat přímo v webovém prohlížeči. To lze také provést pomocí TensorFlow.js v kontextu JavaScriptu.

Pro naše účely, protože jsme pracovali s notebooky založenými na Pythonu, pojďme prozkoumat kroky, které musíte podniknout, abyste exportovali trénovaný model z takového notebooku do formátu čitelného webovou aplikací vytvořenou v Pythonu.

## Nástroje

Pro tento úkol potřebujete dva nástroje: Flask a Pickle, oba běžící na Pythonu.

✅ Co je [Flask](https://palletsprojects.com/p/flask/)? Flask je definován jako 'mikro-rámec' svými tvůrci. Poskytuje základní funkce webových rámců pomocí Pythonu a šablonovacího enginu pro vytváření webových stránek. Podívejte se na [tento modul Learn](https://docs.microsoft.com/learn/modules/python-flask-build-ai-web-app?WT.mc_id=academic-77952-leestott), abyste si vyzkoušeli práci s Flask.

✅ Co je [Pickle](https://docs.python.org/3/library/pickle.html)? Pickle 🥒 je modul Pythonu, který serializuje a deserializuje strukturu objektů Pythonu. Když 'pickle' model, serializujete nebo zplošťujete jeho strukturu pro použití na webu. Buďte opatrní: pickle není inherentně bezpečný, takže buďte opatrní, pokud budete vyzváni k 'un-pickle' souboru. Pickled soubor má příponu `.pkl`.

## Cvičení - vyčistěte svá data

V této lekci použijete data z 80 000 pozorování UFO, shromážděná [NUFORC](https://nuforc.org) (Národní centrum pro hlášení UFO). Tato data obsahují zajímavé popisy pozorování UFO, například:

- **Dlouhý příklad popisu.** "Muž se objeví z paprsku světla, který svítí na travnaté pole v noci, a běží směrem k parkovišti Texas Instruments".
- **Krátký příklad popisu.** "světla nás pronásledovala".

Tabulka [ufos.csv](../../../../3-Web-App/1-Web-App/data/ufos.csv) obsahuje sloupce o `městě`, `státu` a `zemi`, kde k pozorování došlo, tvaru objektu `shape` a jeho `zeměpisné šířce` a `zeměpisné délce`.

V prázdném [notebooku](../../../../3-Web-App/1-Web-App/notebook.ipynb) zahrnutém v této lekci:

1. importujte `pandas`, `matplotlib` a `numpy`, jak jste to udělali v předchozích lekcích, a importujte tabulku ufos. Můžete se podívat na vzorovou datovou sadu:

    ```python
    import pandas as pd
    import numpy as np
    
    ufos = pd.read_csv('./data/ufos.csv')
    ufos.head()
    ```

1. Převést data ufos na malý dataframe s novými názvy. Zkontrolujte unikátní hodnoty v poli `Country`.

    ```python
    ufos = pd.DataFrame({'Seconds': ufos['duration (seconds)'], 'Country': ufos['country'],'Latitude': ufos['latitude'],'Longitude': ufos['longitude']})
    
    ufos.Country.unique()
    ```

1. Nyní můžete snížit množství dat, se kterými musíme pracovat, odstraněním všech nulových hodnot a importováním pouze pozorování mezi 1-60 sekundami:

    ```python
    ufos.dropna(inplace=True)
    
    ufos = ufos[(ufos['Seconds'] >= 1) & (ufos['Seconds'] <= 60)]
    
    ufos.info()
    ```

1. Importujte knihovnu Scikit-learn `LabelEncoder` pro převod textových hodnot zemí na čísla:

    ✅ LabelEncoder kóduje data abecedně

    ```python
    from sklearn.preprocessing import LabelEncoder
    
    ufos['Country'] = LabelEncoder().fit_transform(ufos['Country'])
    
    ufos.head()
    ```

    Vaše data by měla vypadat takto:

    ```output
    	Seconds	Country	Latitude	Longitude
    2	20.0	3		53.200000	-2.916667
    3	20.0	4		28.978333	-96.645833
    14	30.0	4		35.823889	-80.253611
    23	60.0	4		45.582778	-122.352222
    24	3.0		3		51.783333	-0.783333
    ```

## Cvičení - vytvořte svůj model

Nyní se můžete připravit na trénování modelu rozdělením dat na trénovací a testovací skupinu.

1. Vyberte tři funkce, na kterých chcete trénovat jako svůj X vektor, a y vektor bude `Country`. Chcete být schopni zadat `Seconds`, `Latitude` a `Longitude` a získat id země jako výstup.

    ```python
    from sklearn.model_selection import train_test_split
    
    Selected_features = ['Seconds','Latitude','Longitude']
    
    X = ufos[Selected_features]
    y = ufos['Country']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    ```

1. Trénujte svůj model pomocí logistické regrese:

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

Přesnost není špatná **(kolem 95%)**, což není překvapivé, protože `Country` a `Latitude/Longitude` spolu korelují.

Model, který jste vytvořili, není příliš revoluční, protože byste měli být schopni odvodit `Country` z jeho `Latitude` a `Longitude`, ale je to dobré cvičení, jak se pokusit trénovat z neupravených dat, která jste vyčistili, exportovali a poté použili tento model ve webové aplikaci.

## Cvičení - 'pickle' váš model

Nyní je čas _pickle_ váš model! Můžete to udělat v několika řádcích kódu. Jakmile je _pickled_, načtěte svůj pickled model a otestujte jej proti vzorovému datovému poli obsahujícímu hodnoty pro sekundy, zeměpisnou šířku a zeměpisnou délku.

```python
import pickle
model_filename = 'ufo-model.pkl'
pickle.dump(model, open(model_filename,'wb'))

model = pickle.load(open('ufo-model.pkl','rb'))
print(model.predict([[50,44,-12]]))
```

Model vrací **'3'**, což je kód země pro Spojené království. Neuvěřitelné! 👽

## Cvičení - vytvořte Flask aplikaci

Nyní můžete vytvořit Flask aplikaci, která bude volat váš model a vracet podobné výsledky, ale vizuálně přívětivějším způsobem.

1. Začněte vytvořením složky **web-app** vedle souboru _notebook.ipynb_, kde se nachází váš soubor _ufo-model.pkl_.

1. V této složce vytvořte další tři složky: **static**, s podsložkou **css** uvnitř, a **templates**. Nyní byste měli mít následující soubory a adresáře:

    ```output
    web-app/
      static/
        css/
      templates/
    notebook.ipynb
    ufo-model.pkl
    ```

    ✅ Podívejte se na složku řešení pro pohled na hotovou aplikaci

1. První soubor, který vytvoříte ve složce _web-app_, je soubor **requirements.txt**. Stejně jako _package.json_ v aplikaci JavaScript, tento soubor uvádí závislosti požadované aplikací. Do **requirements.txt** přidejte řádky:

    ```text
    scikit-learn
    pandas
    numpy
    flask
    ```

1. Nyní spusťte tento soubor navigací do _web-app_:

    ```bash
    cd web-app
    ```

1. Ve vašem terminálu zadejte `pip install`, abyste nainstalovali knihovny uvedené v _requirements.txt_:

    ```bash
    pip install -r requirements.txt
    ```

1. Nyní jste připraveni vytvořit další tři soubory k dokončení aplikace:

    1. Vytvořte **app.py** v kořenovém adresáři.
    2. Vytvořte **index.html** ve složce _templates_.
    3. Vytvořte **styles.css** ve složce _static/css_.

1. Vytvořte soubor _styles.css_ s několika styly:

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

1. Dále vytvořte soubor _index.html_:

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

    Podívejte se na šablonování v tomto souboru. Všimněte si syntaxe 'mustache' kolem proměnných, které budou poskytovány aplikací, jako je text predikce: `{{}}`. Je zde také formulář, který odesílá predikci na trasu `/predict`.

    Nakonec jste připraveni vytvořit pythonový soubor, který řídí spotřebu modelu a zobrazení predikcí:

1. Do `app.py` přidejte:

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

    > 💡 Tip: když přidáte [`debug=True`](https://www.askpython.com/python-modules/flask/flask-debug-mode) při spuštění webové aplikace pomocí Flask, jakékoli změny, které provedete ve své aplikaci, se okamžitě projeví bez nutnosti restartovat server. Pozor! Tento režim nezapínejte v produkční aplikaci.

Pokud spustíte `python app.py` nebo `python3 app.py` - váš webový server se spustí lokálně a můžete vyplnit krátký formulář, abyste získali odpověď na svou naléhavou otázku o tom, kde byla UFO pozorována!

Než to uděláte, podívejte se na části `app.py`:

1. Nejprve se načtou závislosti a aplikace se spustí.
1. Poté se importuje model.
1. Poté se na domovské trase vykreslí index.html.

Na trase `/predict` se při odeslání formuláře děje několik věcí:

1. Proměnné formuláře se shromáždí a převedou na numpy pole. Poté jsou odeslány modelu a vrátí se predikce.
2. Země, které chceme zobrazit, jsou znovu vykresleny jako čitelný text z jejich předpovězeného kódu země a tato hodnota je odeslána zpět do index.html, aby byla vykreslena v šabloně.

Použití modelu tímto způsobem, s Flask a pickled modelem, je relativně přímočaré. Nejtěžší je pochopit, jaký tvar mají data, která musí být odeslána modelu, aby se získala predikce. To vše závisí na tom, jak byl model trénován. Tento model má tři datové body, které je třeba zadat, aby se získala predikce.

V profesionálním prostředí vidíte, jak je dobrá komunikace nezbytná mezi lidmi, kteří model trénují, a těmi, kteří jej používají ve webové nebo mobilní aplikaci. V našem případě je to jen jedna osoba, vy!

---

## 🚀 Výzva

Místo práce v notebooku a importování modelu do Flask aplikace byste mohli model trénovat přímo v Flask aplikaci! Zkuste převést svůj Python kód z notebooku, možná po vyčištění dat, na trénování modelu přímo v aplikaci na trase nazvané `train`. Jaké jsou výhody a nevýhody tohoto přístupu?

## [Kvíz po lekci](https://ff-quizzes.netlify.app/en/ml/)

## Přehled & Samostudium

Existuje mnoho způsobů, jak vytvořit webovou aplikaci pro využití ML modelů. Udělejte si seznam způsobů, jak byste mohli použít JavaScript nebo Python k vytvoření webové aplikace pro využití strojového učení. Zvažte architekturu: měl by model zůstat v aplikaci nebo být v cloudu? Pokud je to druhé, jak byste k němu přistupovali? Nakreslete architektonický model pro aplikované ML webové řešení.

## Úkol

[Vyzkoušejte jiný model](assignment.md)

---

**Prohlášení**:  
Tento dokument byl přeložen pomocí služby pro automatický překlad [Co-op Translator](https://github.com/Azure/co-op-translator). Ačkoli se snažíme o přesnost, mějte na paměti, že automatické překlady mohou obsahovat chyby nebo nepřesnosti. Původní dokument v jeho původním jazyce by měl být považován za autoritativní zdroj. Pro důležité informace se doporučuje profesionální lidský překlad. Neodpovídáme za žádné nedorozumění nebo nesprávné interpretace vyplývající z použití tohoto překladu.