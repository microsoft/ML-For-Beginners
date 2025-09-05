<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "e0b75f73e4a90d45181dc5581fe2ef5c",
  "translation_date": "2025-09-05T16:13:37+00:00",
  "source_file": "3-Web-App/1-Web-App/README.md",
  "language_code": "sk"
}
-->
# Vytvorte webovú aplikáciu na použitie ML modelu

V tejto lekcii budete trénovať ML model na dátovej sade, ktorá je doslova mimo tohto sveta: _pozorovania UFO za posledné storočie_, získané z databázy NUFORC.

Naučíte sa:

- Ako 'pickle' trénovaný model
- Ako použiť tento model v aplikácii Flask

Pokračujeme v používaní notebookov na čistenie dát a trénovanie modelu, ale môžete tento proces posunúť o krok ďalej tým, že preskúmate použitie modelu „v divočine“, takpovediac: v webovej aplikácii.

Na to budete potrebovať vytvoriť webovú aplikáciu pomocou Flask.

## [Kvíz pred prednáškou](https://ff-quizzes.netlify.app/en/ml/)

## Vytvorenie aplikácie

Existuje niekoľko spôsobov, ako vytvoriť webové aplikácie na využitie modelov strojového učenia. Vaša webová architektúra môže ovplyvniť spôsob, akým je váš model trénovaný. Predstavte si, že pracujete v spoločnosti, kde skupina pre dátovú vedu vytvorila model, ktorý chcete použiť vo svojej aplikácii.

### Úvahy

Existuje mnoho otázok, ktoré si musíte položiť:

- **Je to webová aplikácia alebo mobilná aplikácia?** Ak vytvárate mobilnú aplikáciu alebo potrebujete použiť model v kontexte IoT, môžete použiť [TensorFlow Lite](https://www.tensorflow.org/lite/) a použiť model v aplikácii pre Android alebo iOS.
- **Kde bude model umiestnený?** V cloude alebo lokálne?
- **Podpora offline režimu.** Musí aplikácia fungovať offline?
- **Aká technológia bola použitá na trénovanie modelu?** Zvolená technológia môže ovplyvniť nástroje, ktoré musíte použiť.
    - **Použitie TensorFlow.** Ak trénujete model pomocou TensorFlow, napríklad tento ekosystém poskytuje možnosť konvertovať model TensorFlow na použitie vo webovej aplikácii pomocou [TensorFlow.js](https://www.tensorflow.org/js/).
    - **Použitie PyTorch.** Ak vytvárate model pomocou knižnice ako [PyTorch](https://pytorch.org/), máte možnosť exportovať ho vo formáte [ONNX](https://onnx.ai/) (Open Neural Network Exchange) na použitie vo webových aplikáciách JavaScript, ktoré môžu používať [Onnx Runtime](https://www.onnxruntime.ai/). Táto možnosť bude preskúmaná v budúcej lekcii pre model trénovaný pomocou Scikit-learn.
    - **Použitie Lobe.ai alebo Azure Custom Vision.** Ak používate ML SaaS (Software as a Service) systém, ako je [Lobe.ai](https://lobe.ai/) alebo [Azure Custom Vision](https://azure.microsoft.com/services/cognitive-services/custom-vision-service/?WT.mc_id=academic-77952-leestott) na trénovanie modelu, tento typ softvéru poskytuje spôsoby exportu modelu pre mnoho platforiem, vrátane vytvorenia vlastného API, ktoré je možné dotazovať v cloude vašou online aplikáciou.

Máte tiež možnosť vytvoriť celú webovú aplikáciu Flask, ktorá by dokázala trénovať model priamo v webovom prehliadači. To je možné urobiť aj pomocou TensorFlow.js v kontexte JavaScriptu.

Pre naše účely, keďže sme pracovali s notebookmi založenými na Pythone, preskúmajme kroky, ktoré musíte podniknúť na export trénovaného modelu z takého notebooku do formátu čitateľného webovou aplikáciou vytvorenou v Pythone.

## Nástroje

Na túto úlohu budete potrebovať dva nástroje: Flask a Pickle, oba bežiace na Pythone.

✅ Čo je [Flask](https://palletsprojects.com/p/flask/)? Definovaný ako 'mikro-rámec' jeho tvorcami, Flask poskytuje základné funkcie webových rámcov pomocou Pythonu a šablónového enginu na vytváranie webových stránok. Pozrite si [tento modul Learn](https://docs.microsoft.com/learn/modules/python-flask-build-ai-web-app?WT.mc_id=academic-77952-leestott), aby ste si vyskúšali prácu s Flask.

✅ Čo je [Pickle](https://docs.python.org/3/library/pickle.html)? Pickle 🥒 je modul Pythonu, ktorý serializuje a de-serializuje štruktúru objektov Pythonu. Keď 'pickle' model, serializujete alebo sploštíte jeho štruktúru na použitie na webe. Buďte opatrní: pickle nie je inherentne bezpečný, takže buďte opatrní, ak ste vyzvaní k 'un-pickle' súboru. Súbor pickle má príponu `.pkl`.

## Cvičenie - vyčistite svoje dáta

V tejto lekcii použijete dáta z 80 000 pozorovaní UFO, zhromaždené [NUFORC](https://nuforc.org) (Národné centrum pre hlásenie UFO). Tieto dáta obsahujú zaujímavé popisy pozorovaní UFO, napríklad:

- **Dlhý príklad popisu.** "Muž sa objaví z lúča svetla, ktorý svieti na trávnaté pole v noci, a beží smerom k parkovisku Texas Instruments".
- **Krátky príklad popisu.** "svetlá nás prenasledovali".

Tabuľka [ufos.csv](../../../../3-Web-App/1-Web-App/data/ufos.csv) obsahuje stĺpce o `meste`, `štáte` a `krajine`, kde sa pozorovanie uskutočnilo, `tvare` objektu a jeho `zemepisnej šírke` a `zemepisnej dĺžke`.

V prázdnom [notebooku](../../../../3-Web-App/1-Web-App/notebook.ipynb) zahrnutom v tejto lekcii:

1. importujte `pandas`, `matplotlib` a `numpy`, ako ste to urobili v predchádzajúcich lekciách, a importujte tabuľku ufos. Môžete si pozrieť vzorovú dátovú sadu:

    ```python
    import pandas as pd
    import numpy as np
    
    ufos = pd.read_csv('./data/ufos.csv')
    ufos.head()
    ```

1. Konvertujte dáta ufos na malý dataframe s novými názvami. Skontrolujte unikátne hodnoty v poli `Country`.

    ```python
    ufos = pd.DataFrame({'Seconds': ufos['duration (seconds)'], 'Country': ufos['country'],'Latitude': ufos['latitude'],'Longitude': ufos['longitude']})
    
    ufos.Country.unique()
    ```

1. Teraz môžete zredukovať množstvo dát, s ktorými musíme pracovať, odstránením akýchkoľvek nulových hodnôt a importovaním iba pozorovaní medzi 1-60 sekúnd:

    ```python
    ufos.dropna(inplace=True)
    
    ufos = ufos[(ufos['Seconds'] >= 1) & (ufos['Seconds'] <= 60)]
    
    ufos.info()
    ```

1. Importujte knižnicu Scikit-learn `LabelEncoder` na konverziu textových hodnôt krajín na čísla:

    ✅ LabelEncoder kóduje dáta abecedne

    ```python
    from sklearn.preprocessing import LabelEncoder
    
    ufos['Country'] = LabelEncoder().fit_transform(ufos['Country'])
    
    ufos.head()
    ```

    Vaše dáta by mali vyzerať takto:

    ```output
    	Seconds	Country	Latitude	Longitude
    2	20.0	3		53.200000	-2.916667
    3	20.0	4		28.978333	-96.645833
    14	30.0	4		35.823889	-80.253611
    23	60.0	4		45.582778	-122.352222
    24	3.0		3		51.783333	-0.783333
    ```

## Cvičenie - vytvorte svoj model

Teraz sa môžete pripraviť na trénovanie modelu rozdelením dát na tréningovú a testovaciu skupinu.

1. Vyberte tri vlastnosti, na ktorých chcete trénovať ako svoj X vektor, a y vektor bude `Country`. Chcete byť schopní zadať `Seconds`, `Latitude` a `Longitude` a získať id krajiny na návrat.

    ```python
    from sklearn.model_selection import train_test_split
    
    Selected_features = ['Seconds','Latitude','Longitude']
    
    X = ufos[Selected_features]
    y = ufos['Country']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    ```

1. Trénujte svoj model pomocou logistickej regresie:

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

Presnosť nie je zlá **(okolo 95%)**, čo nie je prekvapujúce, keďže `Country` a `Latitude/Longitude` korelujú.

Model, ktorý ste vytvorili, nie je veľmi revolučný, keďže by ste mali byť schopní odvodiť `Country` z jeho `Latitude` a `Longitude`, ale je to dobré cvičenie na pokus o trénovanie z čistých dát, ktoré ste vyčistili, exportovali a potom použili tento model vo webovej aplikácii.

## Cvičenie - 'pickle' váš model

Teraz je čas _pickle_ váš model! Môžete to urobiť v niekoľkých riadkoch kódu. Keď je _pickled_, načítajte váš pickled model a otestujte ho na vzorovom dátovom poli obsahujúcom hodnoty pre sekundy, zemepisnú šírku a zemepisnú dĺžku.

```python
import pickle
model_filename = 'ufo-model.pkl'
pickle.dump(model, open(model_filename,'wb'))

model = pickle.load(open('ufo-model.pkl','rb'))
print(model.predict([[50,44,-12]]))
```

Model vráti **'3'**, čo je kód krajiny pre Spojené kráľovstvo. Divné! 👽

## Cvičenie - vytvorte Flask aplikáciu

Teraz môžete vytvoriť Flask aplikáciu na volanie vášho modelu a návrat podobných výsledkov, ale v vizuálne príjemnejšej podobe.

1. Začnite vytvorením priečinka **web-app** vedľa súboru _notebook.ipynb_, kde sa nachádza váš súbor _ufo-model.pkl_.

1. V tomto priečinku vytvorte ďalšie tri priečinky: **static**, s priečinkom **css** vo vnútri, a **templates**. Teraz by ste mali mať nasledujúce súbory a adresáre:

    ```output
    web-app/
      static/
        css/
      templates/
    notebook.ipynb
    ufo-model.pkl
    ```

    ✅ Pozrite si riešenie priečinka pre pohľad na hotovú aplikáciu

1. Prvý súbor, ktorý vytvoríte v priečinku _web-app_, je súbor **requirements.txt**. Podobne ako _package.json_ v aplikácii JavaScript, tento súbor uvádza závislosti požadované aplikáciou. Do **requirements.txt** pridajte riadky:

    ```text
    scikit-learn
    pandas
    numpy
    flask
    ```

1. Teraz spustite tento súbor navigáciou do _web-app_:

    ```bash
    cd web-app
    ```

1. Vo vašom termináli zadajte `pip install`, aby ste nainštalovali knižnice uvedené v _requirements.txt_:

    ```bash
    pip install -r requirements.txt
    ```

1. Teraz ste pripravení vytvoriť ďalšie tri súbory na dokončenie aplikácie:

    1. Vytvorte **app.py** v koreňovom adresári.
    2. Vytvorte **index.html** v adresári _templates_.
    3. Vytvorte **styles.css** v adresári _static/css_.

1. Vytvorte súbor _styles.css_ s niekoľkými štýlmi:

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

1. Ďalej vytvorte súbor _index.html_:

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

    Pozrite sa na šablónovanie v tomto súbore. Všimnite si syntax 'mustache' okolo premenných, ktoré budú poskytnuté aplikáciou, ako je text predikcie: `{{}}`. Je tu tiež formulár, ktorý posiela predikciu na trasu `/predict`.

    Nakoniec ste pripravení vytvoriť pythonový súbor, ktorý riadi spotrebu modelu a zobrazenie predikcií:

1. Do `app.py` pridajte:

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

    > 💡 Tip: keď pridáte [`debug=True`](https://www.askpython.com/python-modules/flask/flask-debug-mode) pri spustení webovej aplikácie pomocou Flask, všetky zmeny, ktoré urobíte vo svojej aplikácii, sa okamžite prejavia bez potreby reštartovania servera. Pozor! Nepovoľujte tento režim v produkčnej aplikácii.

Ak spustíte `python app.py` alebo `python3 app.py` - váš webový server sa spustí lokálne a môžete vyplniť krátky formulár, aby ste získali odpoveď na svoju naliehavú otázku o tom, kde boli UFO pozorované!

Predtým, než to urobíte, pozrite sa na časti `app.py`:

1. Najprv sa načítajú závislosti a aplikácia sa spustí.
1. Potom sa importuje model.
1. Potom sa na domovskej trase vykreslí index.html.

Na trase `/predict` sa pri odoslaní formulára deje niekoľko vecí:

1. Premenné formulára sa zhromaždia a konvertujú na numpy pole. Potom sa odošlú modelu a vráti sa predikcia.
2. Krajiny, ktoré chceme zobraziť, sa znovu vykreslia ako čitateľný text z ich predikovaného kódu krajiny a táto hodnota sa odošle späť do index.html, aby sa vykreslila v šablóne.

Použitie modelu týmto spôsobom, s Flask a pickled modelom, je relatívne jednoduché. Najťažšie je pochopiť, aký tvar dát musí byť odoslaný modelu, aby sa získala predikcia. To všetko závisí od toho, ako bol model trénovaný. Tento má tri dátové body, ktoré je potrebné zadať, aby sa získala predikcia.

V profesionálnom prostredí vidíte, aká dôležitá je dobrá komunikácia medzi ľuďmi, ktorí trénujú model, a tými, ktorí ho používajú vo webovej alebo mobilnej aplikácii. V našom prípade je to len jedna osoba, vy!

---

## 🚀 Výzva

Namiesto práce v notebooku a importovania modelu do Flask aplikácie, môžete model trénovať priamo vo Flask aplikácii! Skúste konvertovať svoj Python kód v notebooku, možno po vyčistení dát, na trénovanie modelu priamo v aplikácii na trase nazvanej `train`. Aké sú výhody a nevýhody sledovania tejto metódy?

## [Kvíz po prednáške](https://ff-quizzes.netlify.app/en/ml/)

## Prehľad a samoštúdium

Existuje mnoho spôsobov, ako vytvoriť webovú aplikáciu na využitie ML modelov. Urobte si zoznam spôsobov, ako by ste mohli použiť JavaScript alebo Python na vytvorenie webovej aplikácie na využitie strojového učenia. Zvážte architektúru: mal by model zostať v aplikácii alebo byť umiestnený v cloude? Ak je to druhé, ako by ste k nemu pristupovali? Nakreslite architektonický model pre aplikované ML webové riešenie.

## Zadanie

[Vyskúšajte iný model](assignment.md)

---

**Upozornenie**:  
Tento dokument bol preložený pomocou služby AI prekladu [Co-op Translator](https://github.com/Azure/co-op-translator). Aj keď sa snažíme o presnosť, prosím, berte na vedomie, že automatizované preklady môžu obsahovať chyby alebo nepresnosti. Pôvodný dokument v jeho rodnom jazyku by mal byť považovaný za autoritatívny zdroj. Pre kritické informácie sa odporúča profesionálny ľudský preklad. Nie sme zodpovední za žiadne nedorozumenia alebo nesprávne interpretácie vyplývajúce z použitia tohto prekladu.