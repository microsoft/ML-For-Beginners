<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "e0b75f73e4a90d45181dc5581fe2ef5c",
  "translation_date": "2025-09-05T16:14:11+00:00",
  "source_file": "3-Web-App/1-Web-App/README.md",
  "language_code": "ro"
}
-->
# Construiește o aplicație web pentru a utiliza un model ML

În această lecție, vei antrena un model ML pe un set de date care este literalmente din altă lume: _Observații de OZN-uri din ultimul secol_, preluate din baza de date a NUFORC.

Vei învăța:

- Cum să „picklezi” un model antrenat
- Cum să utilizezi acel model într-o aplicație Flask

Vom continua să folosim notebook-uri pentru a curăța datele și a antrena modelul, dar poți duce procesul un pas mai departe explorând utilizarea unui model „în sălbăticie”, ca să zicem așa: într-o aplicație web.

Pentru a face acest lucru, trebuie să construiești o aplicație web folosind Flask.

## [Chestionar înainte de lecție](https://ff-quizzes.netlify.app/en/ml/)

## Construirea unei aplicații

Există mai multe moduri de a construi aplicații web care să consume modele de învățare automată. Arhitectura web poate influența modul în care modelul este antrenat. Imaginează-ți că lucrezi într-o companie unde grupul de știință a datelor a antrenat un model pe care vor să-l folosești într-o aplicație.

### Considerații

Există multe întrebări pe care trebuie să le pui:

- **Este o aplicație web sau o aplicație mobilă?** Dacă construiești o aplicație mobilă sau trebuie să utilizezi modelul într-un context IoT, ai putea folosi [TensorFlow Lite](https://www.tensorflow.org/lite/) și să utilizezi modelul într-o aplicație Android sau iOS.
- **Unde va fi găzduit modelul?** În cloud sau local?
- **Suport offline.** Aplicația trebuie să funcționeze offline?
- **Ce tehnologie a fost utilizată pentru a antrena modelul?** Tehnologia aleasă poate influența instrumentele pe care trebuie să le folosești.
    - **Utilizarea TensorFlow.** Dacă antrenezi un model folosind TensorFlow, de exemplu, ecosistemul oferă posibilitatea de a converti un model TensorFlow pentru utilizare într-o aplicație web folosind [TensorFlow.js](https://www.tensorflow.org/js/).
    - **Utilizarea PyTorch.** Dacă construiești un model folosind o bibliotecă precum [PyTorch](https://pytorch.org/), ai opțiunea de a-l exporta în format [ONNX](https://onnx.ai/) (Open Neural Network Exchange) pentru utilizare în aplicații web JavaScript care pot folosi [Onnx Runtime](https://www.onnxruntime.ai/). Această opțiune va fi explorată într-o lecție viitoare pentru un model antrenat cu Scikit-learn.
    - **Utilizarea Lobe.ai sau Azure Custom Vision.** Dacă folosești un sistem ML SaaS (Software as a Service) precum [Lobe.ai](https://lobe.ai/) sau [Azure Custom Vision](https://azure.microsoft.com/services/cognitive-services/custom-vision-service/?WT.mc_id=academic-77952-leestott) pentru a antrena un model, acest tip de software oferă modalități de a exporta modelul pentru multe platforme, inclusiv construirea unui API personalizat care să fie interogat în cloud de aplicația ta online.

De asemenea, ai oportunitatea de a construi o aplicație web completă Flask care ar putea antrena modelul direct într-un browser web. Acest lucru poate fi realizat și folosind TensorFlow.js într-un context JavaScript.

Pentru scopurile noastre, deoarece am lucrat cu notebook-uri bazate pe Python, să explorăm pașii pe care trebuie să-i urmezi pentru a exporta un model antrenat dintr-un astfel de notebook într-un format citibil de o aplicație web construită în Python.

## Instrumente

Pentru această sarcină, ai nevoie de două instrumente: Flask și Pickle, ambele rulând pe Python.

✅ Ce este [Flask](https://palletsprojects.com/p/flask/)? Definit ca un „micro-framework” de către creatorii săi, Flask oferă funcțiile de bază ale framework-urilor web folosind Python și un motor de șabloane pentru a construi pagini web. Aruncă o privire la [acest modul de învățare](https://docs.microsoft.com/learn/modules/python-flask-build-ai-web-app?WT.mc_id=academic-77952-leestott) pentru a exersa construirea cu Flask.

✅ Ce este [Pickle](https://docs.python.org/3/library/pickle.html)? Pickle 🥒 este un modul Python care serializează și de-serializează o structură de obiecte Python. Când „picklezi” un model, îi serializezi sau îi aplatizezi structura pentru utilizare pe web. Atenție: pickle nu este intrinsec sigur, așa că fii precaut dacă ți se cere să „un-picklezi” un fișier. Un fișier pickled are sufixul `.pkl`.

## Exercițiu - curăță datele

În această lecție vei folosi date din 80.000 de observații de OZN-uri, colectate de [NUFORC](https://nuforc.org) (Centrul Național de Raportare a OZN-urilor). Aceste date conțin descrieri interesante ale observațiilor de OZN-uri, de exemplu:

- **Descriere lungă exemplu.** "Un bărbat iese dintr-un fascicul de lumină care strălucește pe un câmp de iarbă noaptea și aleargă spre parcarea Texas Instruments".
- **Descriere scurtă exemplu.** "luminile ne-au urmărit".

Fișierul [ufos.csv](../../../../3-Web-App/1-Web-App/data/ufos.csv) include coloane despre `orașul`, `statul` și `țara` unde a avut loc observația, `forma` obiectului și `latitudinea` și `longitudinea` acestuia.

În [notebook-ul](../../../../3-Web-App/1-Web-App/notebook.ipynb) gol inclus în această lecție:

1. importă `pandas`, `matplotlib` și `numpy` așa cum ai făcut în lecțiile anterioare și importă fișierul ufos. Poți arunca o privire la un set de date exemplu:

    ```python
    import pandas as pd
    import numpy as np
    
    ufos = pd.read_csv('./data/ufos.csv')
    ufos.head()
    ```

1. Convertește datele ufos într-un dataframe mic cu titluri noi. Verifică valorile unice din câmpul `Țară`.

    ```python
    ufos = pd.DataFrame({'Seconds': ufos['duration (seconds)'], 'Country': ufos['country'],'Latitude': ufos['latitude'],'Longitude': ufos['longitude']})
    
    ufos.Country.unique()
    ```

1. Acum, poți reduce cantitatea de date cu care trebuie să lucrezi eliminând valorile nule și importând doar observațiile între 1-60 de secunde:

    ```python
    ufos.dropna(inplace=True)
    
    ufos = ufos[(ufos['Seconds'] >= 1) & (ufos['Seconds'] <= 60)]
    
    ufos.info()
    ```

1. Importă biblioteca `LabelEncoder` din Scikit-learn pentru a converti valorile text pentru țări într-un număr:

    ✅ LabelEncoder codifică datele alfabetic

    ```python
    from sklearn.preprocessing import LabelEncoder
    
    ufos['Country'] = LabelEncoder().fit_transform(ufos['Country'])
    
    ufos.head()
    ```

    Datele tale ar trebui să arate astfel:

    ```output
    	Seconds	Country	Latitude	Longitude
    2	20.0	3		53.200000	-2.916667
    3	20.0	4		28.978333	-96.645833
    14	30.0	4		35.823889	-80.253611
    23	60.0	4		45.582778	-122.352222
    24	3.0		3		51.783333	-0.783333
    ```

## Exercițiu - construiește modelul

Acum poți începe să antrenezi un model împărțind datele în grupuri de antrenament și testare.

1. Selectează cele trei caracteristici pe care vrei să le antrenezi ca vector X, iar vectorul y va fi `Țara`. Vrei să poți introduce `Secunde`, `Latitudine` și `Longitudine` și să obții un id de țară ca răspuns.

    ```python
    from sklearn.model_selection import train_test_split
    
    Selected_features = ['Seconds','Latitude','Longitude']
    
    X = ufos[Selected_features]
    y = ufos['Country']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    ```

1. Antrenează modelul folosind regresia logistică:

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

Acuratețea nu este rea **(aproximativ 95%)**, ceea ce nu este surprinzător, deoarece `Țara` și `Latitudine/Longitudine` sunt corelate.

Modelul pe care l-ai creat nu este foarte revoluționar, deoarece ar trebui să poți deduce o `Țară` din `Latitudine` și `Longitudine`, dar este un exercițiu bun pentru a încerca să antrenezi din date brute pe care le-ai curățat, exportat și apoi utilizat acest model într-o aplicație web.

## Exercițiu - „picklează” modelul

Acum, este timpul să _picklezi_ modelul! Poți face acest lucru în câteva linii de cod. Odată ce este _pickled_, încarcă modelul pickled și testează-l pe un array de date exemplu care conține valori pentru secunde, latitudine și longitudine.

```python
import pickle
model_filename = 'ufo-model.pkl'
pickle.dump(model, open(model_filename,'wb'))

model = pickle.load(open('ufo-model.pkl','rb'))
print(model.predict([[50,44,-12]]))
```

Modelul returnează **'3'**, care este codul țării pentru Regatul Unit. Uimitor! 👽

## Exercițiu - construiește o aplicație Flask

Acum poți construi o aplicație Flask pentru a apela modelul și a returna rezultate similare, dar într-un mod mai plăcut vizual.

1. Începe prin a crea un folder numit **web-app** lângă fișierul _notebook.ipynb_ unde se află fișierul _ufo-model.pkl_.

1. În acel folder creează alte trei foldere: **static**, cu un folder **css** în interiorul său, și **templates**. Acum ar trebui să ai următoarele fișiere și directoare:

    ```output
    web-app/
      static/
        css/
      templates/
    notebook.ipynb
    ufo-model.pkl
    ```

    ✅ Consultă folderul soluție pentru o privire asupra aplicației finale

1. Primul fișier pe care să-l creezi în folderul _web-app_ este fișierul **requirements.txt**. La fel ca _package.json_ într-o aplicație JavaScript, acest fișier listează dependențele necesare aplicației. În **requirements.txt** adaugă liniile:

    ```text
    scikit-learn
    pandas
    numpy
    flask
    ```

1. Acum, rulează acest fișier navigând la _web-app_:

    ```bash
    cd web-app
    ```

1. În terminalul tău tastează `pip install`, pentru a instala bibliotecile listate în _requirements.txt_:

    ```bash
    pip install -r requirements.txt
    ```

1. Acum, ești gata să creezi alte trei fișiere pentru a finaliza aplicația:

    1. Creează **app.py** în rădăcină.
    2. Creează **index.html** în directorul _templates_.
    3. Creează **styles.css** în directorul _static/css_.

1. Construiește fișierul _styles.css_ cu câteva stiluri:

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

1. Apoi, construiește fișierul _index.html_:

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

    Aruncă o privire la șablonul din acest fișier. Observă sintaxa „mustache” în jurul variabilelor care vor fi furnizate de aplicație, cum ar fi textul predicției: `{{}}`. Există, de asemenea, un formular care postează o predicție la ruta `/predict`.

    În cele din urmă, ești gata să construiești fișierul Python care conduce consumul modelului și afișarea predicțiilor:

1. În `app.py` adaugă:

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

    > 💡 Sfat: când adaugi [`debug=True`](https://www.askpython.com/python-modules/flask/flask-debug-mode) în timp ce rulezi aplicația web folosind Flask, orice modificări pe care le faci aplicației tale vor fi reflectate imediat fără a fi nevoie să repornești serverul. Atenție! Nu activa acest mod într-o aplicație de producție.

Dacă rulezi `python app.py` sau `python3 app.py` - serverul tău web pornește local și poți completa un formular scurt pentru a obține un răspuns la întrebarea ta arzătoare despre unde au fost observate OZN-uri!

Înainte de a face acest lucru, aruncă o privire la părțile din `app.py`:

1. Mai întâi, dependențele sunt încărcate și aplicația pornește.
1. Apoi, modelul este importat.
1. Apoi, index.html este redat pe ruta principală.

Pe ruta `/predict`, se întâmplă mai multe lucruri când formularul este postat:

1. Variabilele formularului sunt colectate și convertite într-un array numpy. Acestea sunt apoi trimise modelului și se returnează o predicție.
2. Țările pe care dorim să le afișăm sunt re-redate ca text lizibil din codul de țară prezis, iar acea valoare este trimisă înapoi la index.html pentru a fi redată în șablon.

Utilizarea unui model în acest mod, cu Flask și un model pickled, este relativ simplă. Cel mai dificil lucru este să înțelegi ce formă trebuie să aibă datele care trebuie trimise modelului pentru a obține o predicție. Totul depinde de modul în care modelul a fost antrenat. Acesta are trei puncte de date care trebuie introduse pentru a obține o predicție.

Într-un mediu profesional, poți vedea cât de importantă este comunicarea bună între cei care antrenează modelul și cei care îl consumă într-o aplicație web sau mobilă. În cazul nostru, este doar o singură persoană, tu!

---

## 🚀 Provocare

În loc să lucrezi într-un notebook și să imporți modelul în aplicația Flask, ai putea antrena modelul direct în aplicația Flask! Încearcă să convertești codul Python din notebook, poate după ce datele sunt curățate, pentru a antrena modelul direct în aplicație pe o rută numită `train`. Care sunt avantajele și dezavantajele acestei metode?

## [Chestionar după lecție](https://ff-quizzes.netlify.app/en/ml/)

## Recapitulare și studiu individual

Există multe moduri de a construi o aplicație web care să consume modele ML. Fă o listă cu modurile în care ai putea folosi JavaScript sau Python pentru a construi o aplicație web care să valorifice învățarea automată. Ia în considerare arhitectura: ar trebui modelul să rămână în aplicație sau să fie găzduit în cloud? Dacă este ultima variantă, cum l-ai accesa? Desenează un model arhitectural pentru o soluție web ML aplicată.

## Temă

[Încearcă un model diferit](assignment.md)

---

**Declinare de responsabilitate**:  
Acest document a fost tradus folosind serviciul de traducere AI [Co-op Translator](https://github.com/Azure/co-op-translator). Deși ne străduim să asigurăm acuratețea, vă rugăm să fiți conștienți că traducerile automate pot conține erori sau inexactități. Documentul original în limba sa natală ar trebui considerat sursa autoritară. Pentru informații critice, se recomandă traducerea profesională realizată de un specialist uman. Nu ne asumăm responsabilitatea pentru eventualele neînțelegeri sau interpretări greșite care pot apărea din utilizarea acestei traduceri.