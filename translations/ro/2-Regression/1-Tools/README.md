<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "fa81d226c71d5af7a2cade31c1c92b88",
  "translation_date": "2025-09-05T15:22:38+00:00",
  "source_file": "2-Regression/1-Tools/README.md",
  "language_code": "ro"
}
-->
# Începeți cu Python și Scikit-learn pentru modele de regresie

![Rezumat al regresiilor într-un sketchnote](../../../../sketchnotes/ml-regression.png)

> Sketchnote de [Tomomi Imura](https://www.twitter.com/girlie_mac)

## [Chestionar înainte de lecție](https://ff-quizzes.netlify.app/en/ml/)

> ### [Această lecție este disponibilă și în R!](../../../../2-Regression/1-Tools/solution/R/lesson_1.html)

## Introducere

În aceste patru lecții, veți descoperi cum să construiți modele de regresie. Vom discuta în curând la ce sunt utile acestea. Dar înainte de a începe, asigurați-vă că aveți instrumentele potrivite pentru a începe procesul!

În această lecție, veți învăța cum să:

- Configurați computerul pentru sarcini locale de învățare automată.
- Lucrați cu Jupyter notebooks.
- Utilizați Scikit-learn, inclusiv instalarea acestuia.
- Explorați regresia liniară printr-un exercițiu practic.

## Instalări și configurări

[![ML pentru începători - Configurați-vă instrumentele pentru a construi modele de învățare automată](https://img.youtube.com/vi/-DfeD2k2Kj0/0.jpg)](https://youtu.be/-DfeD2k2Kj0 "ML pentru începători - Configurați-vă instrumentele pentru a construi modele de învățare automată")

> 🎥 Faceți clic pe imaginea de mai sus pentru un scurt videoclip despre configurarea computerului pentru ML.

1. **Instalați Python**. Asigurați-vă că [Python](https://www.python.org/downloads/) este instalat pe computerul dvs. Veți utiliza Python pentru multe sarcini de știința datelor și învățare automată. Majoritatea sistemelor de operare includ deja o instalare Python. Există și [Pachete de codare Python](https://code.visualstudio.com/learn/educators/installers?WT.mc_id=academic-77952-leestott) utile, pentru a ușura configurarea pentru unii utilizatori.

   Totuși, unele utilizări ale Python necesită o anumită versiune a software-ului, în timp ce altele necesită o versiune diferită. Din acest motiv, este util să lucrați într-un [mediu virtual](https://docs.python.org/3/library/venv.html).

2. **Instalați Visual Studio Code**. Asigurați-vă că Visual Studio Code este instalat pe computerul dvs. Urmați aceste instrucțiuni pentru a [instala Visual Studio Code](https://code.visualstudio.com/) pentru o instalare de bază. Veți utiliza Python în Visual Studio Code în acest curs, așa că poate doriți să vă familiarizați cu modul de [configurare a Visual Studio Code](https://docs.microsoft.com/learn/modules/python-install-vscode?WT.mc_id=academic-77952-leestott) pentru dezvoltarea în Python.

   > Familiarizați-vă cu Python parcurgând această colecție de [module de învățare](https://docs.microsoft.com/users/jenlooper-2911/collections/mp1pagggd5qrq7?WT.mc_id=academic-77952-leestott)
   >
   > [![Configurați Python cu Visual Studio Code](https://img.youtube.com/vi/yyQM70vi7V8/0.jpg)](https://youtu.be/yyQM70vi7V8 "Configurați Python cu Visual Studio Code")
   >
   > 🎥 Faceți clic pe imaginea de mai sus pentru un videoclip: utilizarea Python în VS Code.

3. **Instalați Scikit-learn**, urmând [aceste instrucțiuni](https://scikit-learn.org/stable/install.html). Deoarece trebuie să vă asigurați că utilizați Python 3, este recomandat să utilizați un mediu virtual. Notă: dacă instalați această bibliotecă pe un Mac cu M1, există instrucțiuni speciale pe pagina menționată mai sus.

4. **Instalați Jupyter Notebook**. Va trebui să [instalați pachetul Jupyter](https://pypi.org/project/jupyter/).

## Mediul dvs. de lucru pentru ML

Veți utiliza **notebooks** pentru a dezvolta codul Python și a crea modele de învățare automată. Acest tip de fișier este un instrument comun pentru oamenii de știință în domeniul datelor și poate fi identificat prin sufixul sau extensia `.ipynb`.

Notebooks oferă un mediu interactiv care permite dezvoltatorului să scrie cod și să adauge note și documentație în jurul codului, ceea ce este foarte util pentru proiecte experimentale sau orientate spre cercetare.

[![ML pentru începători - Configurați Jupyter Notebooks pentru a începe să construiți modele de regresie](https://img.youtube.com/vi/7E-jC8FLA2E/0.jpg)](https://youtu.be/7E-jC8FLA2E "ML pentru începători - Configurați Jupyter Notebooks pentru a începe să construiți modele de regresie")

> 🎥 Faceți clic pe imaginea de mai sus pentru un scurt videoclip despre acest exercițiu.

### Exercițiu - lucrați cu un notebook

În acest folder, veți găsi fișierul _notebook.ipynb_.

1. Deschideți _notebook.ipynb_ în Visual Studio Code.

   Un server Jupyter va porni cu Python 3+ activat. Veți găsi zone ale notebook-ului care pot fi `rulate`, adică bucăți de cod. Puteți rula un bloc de cod selectând pictograma care arată ca un buton de redare.

2. Selectați pictograma `md` și adăugați puțin markdown, precum și următorul text **# Bine ați venit în notebook-ul dvs.**.

   Apoi, adăugați ceva cod Python.

3. Scrieți **print('hello notebook')** în blocul de cod.
4. Selectați săgeata pentru a rula codul.

   Ar trebui să vedeți declarația afișată:

    ```output
    hello notebook
    ```

![VS Code cu un notebook deschis](../../../../2-Regression/1-Tools/images/notebook.jpg)

Puteți intercala codul cu comentarii pentru a documenta notebook-ul.

✅ Gândiți-vă un minut la cât de diferit este mediul de lucru al unui dezvoltator web față de cel al unui om de știință în domeniul datelor.

## Începeți cu Scikit-learn

Acum că Python este configurat în mediul dvs. local și sunteți confortabil cu Jupyter notebooks, să devenim la fel de confortabili cu Scikit-learn (se pronunță `sci` ca în `science`). Scikit-learn oferă o [API extinsă](https://scikit-learn.org/stable/modules/classes.html#api-ref) pentru a vă ajuta să efectuați sarcini de ML.

Conform [site-ului lor](https://scikit-learn.org/stable/getting_started.html), "Scikit-learn este o bibliotecă open source pentru învățare automată care suportă învățarea supravegheată și nesupravegheată. De asemenea, oferă diverse instrumente pentru ajustarea modelelor, preprocesarea datelor, selecția și evaluarea modelelor, precum și multe alte utilități."

În acest curs, veți utiliza Scikit-learn și alte instrumente pentru a construi modele de învățare automată pentru a efectua ceea ce numim sarcini de 'învățare automată tradițională'. Am evitat în mod deliberat rețelele neuronale și învățarea profundă, deoarece acestea sunt mai bine acoperite în viitorul nostru curriculum 'AI pentru Începători'.

Scikit-learn face ușor să construiți modele și să le evaluați pentru utilizare. Este axat în principal pe utilizarea datelor numerice și conține mai multe seturi de date gata făcute pentru utilizare ca instrumente de învățare. De asemenea, include modele pre-construite pentru studenți. Să explorăm procesul de încărcare a datelor preambalate și utilizarea unui estimator pentru primul model ML cu Scikit-learn, folosind câteva date de bază.

## Exercițiu - primul dvs. notebook Scikit-learn

> Acest tutorial a fost inspirat de [exemplul de regresie liniară](https://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html#sphx-glr-auto-examples-linear-model-plot-ols-py) de pe site-ul Scikit-learn.

[![ML pentru începători - Primul dvs. proiect de regresie liniară în Python](https://img.youtube.com/vi/2xkXL5EUpS0/0.jpg)](https://youtu.be/2xkXL5EUpS0 "ML pentru începători - Primul dvs. proiect de regresie liniară în Python")

> 🎥 Faceți clic pe imaginea de mai sus pentru un scurt videoclip despre acest exercițiu.

În fișierul _notebook.ipynb_ asociat acestei lecții, ștergeți toate celulele apăsând pe pictograma 'coș de gunoi'.

În această secțiune, veți lucra cu un set mic de date despre diabet, care este inclus în Scikit-learn pentru scopuri de învățare. Imaginați-vă că doriți să testați un tratament pentru pacienții diabetici. Modelele de învățare automată ar putea să vă ajute să determinați care pacienți ar răspunde mai bine la tratament, pe baza combinațiilor de variabile. Chiar și un model de regresie foarte simplu, atunci când este vizualizat, ar putea arăta informații despre variabilele care v-ar ajuta să organizați studiile clinice teoretice.

✅ Există multe tipuri de metode de regresie, iar alegerea uneia depinde de răspunsul pe care îl căutați. Dacă doriți să preziceți înălțimea probabilă a unei persoane de o anumită vârstă, ați utiliza regresia liniară, deoarece căutați o **valoare numerică**. Dacă sunteți interesat să descoperiți dacă un tip de bucătărie ar trebui considerat vegan sau nu, căutați o **încadrare în categorie**, așa că ați utiliza regresia logistică. Veți învăța mai multe despre regresia logistică mai târziu. Gândiți-vă puțin la întrebările pe care le puteți pune datelor și care dintre aceste metode ar fi mai potrivită.

Să începem această sarcină.

### Importați biblioteci

Pentru această sarcină, vom importa câteva biblioteci:

- **matplotlib**. Este un [instrument util pentru graficare](https://matplotlib.org/) și îl vom folosi pentru a crea un grafic liniar.
- **numpy**. [numpy](https://numpy.org/doc/stable/user/whatisnumpy.html) este o bibliotecă utilă pentru manipularea datelor numerice în Python.
- **sklearn**. Aceasta este biblioteca [Scikit-learn](https://scikit-learn.org/stable/user_guide.html).

Importați câteva biblioteci pentru a vă ajuta cu sarcinile.

1. Adăugați importurile tastând următorul cod:

   ```python
   import matplotlib.pyplot as plt
   import numpy as np
   from sklearn import datasets, linear_model, model_selection
   ```

   Mai sus importați `matplotlib`, `numpy` și importați `datasets`, `linear_model` și `model_selection` din `sklearn`. `model_selection` este utilizat pentru a împărți datele în seturi de antrenament și testare.

### Setul de date despre diabet

Setul de date [diabet](https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset) inclus conține 442 de mostre de date despre diabet, cu 10 variabile caracteristice, dintre care unele includ:

- age: vârsta în ani
- bmi: indicele de masă corporală
- bp: tensiunea arterială medie
- s1 tc: T-Cells (un tip de globule albe)

✅ Acest set de date include conceptul de 'sex' ca variabilă caracteristică importantă pentru cercetarea despre diabet. Multe seturi de date medicale includ acest tip de clasificare binară. Gândiți-vă puțin la modul în care astfel de clasificări ar putea exclude anumite părți ale populației de la tratamente.

Acum, încărcați datele X și y.

> 🎓 Amintiți-vă, aceasta este învățare supravegheată, iar noi avem nevoie de o țintă 'y' denumită.

Într-o nouă celulă de cod, încărcați setul de date despre diabet apelând `load_diabetes()`. Parametrul `return_X_y=True` semnalează că `X` va fi o matrice de date, iar `y` va fi ținta regresiei.

1. Adăugați câteva comenzi print pentru a afișa forma matricei de date și primul său element:

    ```python
    X, y = datasets.load_diabetes(return_X_y=True)
    print(X.shape)
    print(X[0])
    ```

    Ceea ce primiți ca răspuns este un tuplu. Ceea ce faceți este să atribuiți cele două prime valori ale tuplului lui `X` și `y`, respectiv. Aflați mai multe [despre tupluri](https://wikipedia.org/wiki/Tuple).

    Puteți vedea că aceste date au 442 de elemente organizate în matrici de 10 elemente:

    ```text
    (442, 10)
    [ 0.03807591  0.05068012  0.06169621  0.02187235 -0.0442235  -0.03482076
    -0.04340085 -0.00259226  0.01990842 -0.01764613]
    ```

    ✅ Gândiți-vă puțin la relația dintre date și ținta regresiei. Regresia liniară prezice relațiile dintre caracteristica X și variabila țintă y. Puteți găsi [ținta](https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset) pentru setul de date despre diabet în documentație? Ce demonstrează acest set de date, având în vedere ținta?

2. Apoi, selectați o parte din acest set de date pentru a o reprezenta grafic, selectând a treia coloană a setului de date. Puteți face acest lucru utilizând operatorul `:` pentru a selecta toate rândurile, apoi selectând a treia coloană utilizând indexul (2). De asemenea, puteți rearanja datele pentru a fi o matrice 2D - așa cum este necesar pentru reprezentarea grafică - utilizând `reshape(n_rows, n_columns)`. Dacă unul dintre parametri este -1, dimensiunea corespunzătoare este calculată automat.

   ```python
   X = X[:, 2]
   X = X.reshape((-1,1))
   ```

   ✅ În orice moment, afișați datele pentru a verifica forma lor.

3. Acum că aveți datele pregătite pentru a fi reprezentate grafic, puteți vedea dacă o mașină poate determina o separare logică între numerele din acest set de date. Pentru a face acest lucru, trebuie să împărțiți atât datele (X), cât și ținta (y) în seturi de testare și antrenament. Scikit-learn are o metodă simplă pentru aceasta; puteți împărți datele de testare la un anumit punct.

   ```python
   X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.33)
   ```

4. Acum sunteți gata să antrenați modelul! Încărcați modelul de regresie liniară și antrenați-l cu seturile de antrenament X și y utilizând `model.fit()`:

    ```python
    model = linear_model.LinearRegression()
    model.fit(X_train, y_train)
    ```

    ✅ `model.fit()` este o funcție pe care o veți întâlni în multe biblioteci ML, cum ar fi TensorFlow.

5. Apoi, creați o predicție utilizând datele de testare, folosind funcția `predict()`. Aceasta va fi utilizată pentru a trasa linia între grupurile de date.

    ```python
    y_pred = model.predict(X_test)
    ```

6. Acum este momentul să afișați datele într-un grafic. Matplotlib este un instrument foarte util pentru această sarcină. Creați un grafic scatter cu toate datele de testare X și y și utilizați predicția pentru a trasa o linie în locul cel mai potrivit, între grupările de date ale modelului.

    ```python
    plt.scatter(X_test, y_test,  color='black')
    plt.plot(X_test, y_pred, color='blue', linewidth=3)
    plt.xlabel('Scaled BMIs')
    plt.ylabel('Disease Progression')
    plt.title('A Graph Plot Showing Diabetes Progression Against BMI')
    plt.show()
    ```

   ![un grafic scatter care arată puncte de date despre diabet](../../../../2-Regression/1-Tools/images/scatterplot.png)
✅ Gândește-te puțin la ce se întâmplă aici. O linie dreaptă trece prin multe puncte mici de date, dar ce face exact? Poți vedea cum ar trebui să poți folosi această linie pentru a prezice unde ar trebui să se încadreze un punct de date nou, nevăzut, în raport cu axa y a graficului? Încearcă să exprimi în cuvinte utilitatea practică a acestui model.

Felicitări, ai construit primul tău model de regresie liniară, ai creat o predicție cu el și ai afișat-o într-un grafic!

---
## 🚀Provocare

Reprezintă grafic o variabilă diferită din acest set de date. Sugestie: editează această linie: `X = X[:,2]`. Având în vedere ținta acestui set de date, ce poți descoperi despre progresia diabetului ca boală?
## [Test de verificare după lecție](https://ff-quizzes.netlify.app/en/ml/)

## Recapitulare & Studiu individual

În acest tutorial, ai lucrat cu regresia liniară simplă, mai degrabă decât cu regresia univariată sau regresia multiplă. Citește puțin despre diferențele dintre aceste metode sau aruncă o privire la [acest videoclip](https://www.coursera.org/lecture/quantifying-relationships-regression-models/linear-vs-nonlinear-categorical-variables-ai2Ef).

Citește mai multe despre conceptul de regresie și gândește-te la ce tipuri de întrebări pot fi răspunse prin această tehnică. Urmează acest [tutorial](https://docs.microsoft.com/learn/modules/train-evaluate-regression-models?WT.mc_id=academic-77952-leestott) pentru a-ți aprofunda înțelegerea.

## Temă

[Un set de date diferit](assignment.md)

---

**Declinare de responsabilitate**:  
Acest document a fost tradus folosind serviciul de traducere AI [Co-op Translator](https://github.com/Azure/co-op-translator). Deși ne străduim să asigurăm acuratețea, vă rugăm să fiți conștienți că traducerile automate pot conține erori sau inexactități. Documentul original în limba sa natală ar trebui considerat sursa autoritară. Pentru informații critice, se recomandă traducerea profesională realizată de un specialist uman. Nu ne asumăm responsabilitatea pentru eventualele neînțelegeri sau interpretări greșite care pot apărea din utilizarea acestei traduceri.