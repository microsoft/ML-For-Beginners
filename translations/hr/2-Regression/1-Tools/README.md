<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "fa81d226c71d5af7a2cade31c1c92b88",
  "translation_date": "2025-09-05T11:40:33+00:00",
  "source_file": "2-Regression/1-Tools/README.md",
  "language_code": "hr"
}
-->
# Početak rada s Pythonom i Scikit-learn za regresijske modele

![Sažetak regresija u obliku sketchnotea](../../../../sketchnotes/ml-regression.png)

> Sketchnote autorice [Tomomi Imura](https://www.twitter.com/girlie_mac)

## [Kviz prije predavanja](https://ff-quizzes.netlify.app/en/ml/)

> ### [Ova lekcija je dostupna i na R jeziku!](../../../../2-Regression/1-Tools/solution/R/lesson_1.html)

## Uvod

U ove četiri lekcije otkrit ćete kako izgraditi regresijske modele. Uskoro ćemo razgovarati o tome za što se koriste. No prije nego što krenete, provjerite imate li sve potrebne alate za početak!

U ovoj lekciji naučit ćete:

- Kako konfigurirati svoje računalo za lokalne zadatke strojnog učenja.
- Kako raditi s Jupyter bilježnicama.
- Kako koristiti Scikit-learn, uključujući instalaciju.
- Kako istražiti linearnu regresiju kroz praktičnu vježbu.

## Instalacije i konfiguracije

[![ML za početnike - Pripremite alate za izgradnju modela strojnog učenja](https://img.youtube.com/vi/-DfeD2k2Kj0/0.jpg)](https://youtu.be/-DfeD2k2Kj0 "ML za početnike - Pripremite alate za izgradnju modela strojnog učenja")

> 🎥 Kliknite na sliku iznad za kratki video o konfiguraciji vašeg računala za ML.

1. **Instalirajte Python**. Provjerite je li [Python](https://www.python.org/downloads/) instaliran na vašem računalu. Python ćete koristiti za mnoge zadatke vezane uz znanost o podacima i strojno učenje. Većina računalnih sustava već ima instaliran Python. Dostupni su i korisni [Python Coding Packs](https://code.visualstudio.com/learn/educators/installers?WT.mc_id=academic-77952-leestott) koji olakšavaju postavljanje za neke korisnike.

   Međutim, neka korištenja Pythona zahtijevaju jednu verziju softvera, dok druga zahtijevaju drugu verziju. Zbog toga je korisno raditi unutar [virtualnog okruženja](https://docs.python.org/3/library/venv.html).

2. **Instalirajte Visual Studio Code**. Provjerite imate li instaliran Visual Studio Code na svom računalu. Slijedite ove upute za [instalaciju Visual Studio Code-a](https://code.visualstudio.com/) za osnovnu instalaciju. U ovom ćete tečaju koristiti Python u Visual Studio Code-u, pa bi bilo korisno osvježiti znanje o tome kako [konfigurirati Visual Studio Code](https://docs.microsoft.com/learn/modules/python-install-vscode?WT.mc_id=academic-77952-leestott) za razvoj u Pythonu.

   > Upoznajte se s Pythonom radeći kroz ovu kolekciju [Learn modula](https://docs.microsoft.com/users/jenlooper-2911/collections/mp1pagggd5qrq7?WT.mc_id=academic-77952-leestott)
   >
   > [![Postavljanje Pythona s Visual Studio Code-om](https://img.youtube.com/vi/yyQM70vi7V8/0.jpg)](https://youtu.be/yyQM70vi7V8 "Postavljanje Pythona s Visual Studio Code-om")
   >
   > 🎥 Kliknite na sliku iznad za video: korištenje Pythona unutar VS Code-a.

3. **Instalirajte Scikit-learn**, slijedeći [ove upute](https://scikit-learn.org/stable/install.html). Budući da trebate osigurati korištenje Pythona 3, preporučuje se korištenje virtualnog okruženja. Napomena: ako ovu biblioteku instalirate na M1 Macu, na stranici iznad nalaze se posebne upute.

4. **Instalirajte Jupyter Notebook**. Trebat ćete [instalirati Jupyter paket](https://pypi.org/project/jupyter/).

## Vaše okruženje za autorstvo ML-a

Koristit ćete **bilježnice** za razvoj svog Python koda i izradu modela strojnog učenja. Ova vrsta datoteke uobičajeni je alat za znanstvenike podataka, a prepoznat ćete ih po sufiksu ili ekstenziji `.ipynb`.

Bilježnice su interaktivno okruženje koje omogućuje programeru da istovremeno piše kod i dodaje bilješke te dokumentaciju oko koda, što je vrlo korisno za eksperimentalne ili istraživačke projekte.

[![ML za početnike - Postavljanje Jupyter bilježnica za početak izrade regresijskih modela](https://img.youtube.com/vi/7E-jC8FLA2E/0.jpg)](https://youtu.be/7E-jC8FLA2E "ML za početnike - Postavljanje Jupyter bilježnica za početak izrade regresijskih modela")

> 🎥 Kliknite na sliku iznad za kratki video o ovoj vježbi.

### Vježba - rad s bilježnicom

U ovoj mapi pronaći ćete datoteku _notebook.ipynb_.

1. Otvorite _notebook.ipynb_ u Visual Studio Code-u.

   Pokrenut će se Jupyter poslužitelj s Pythonom 3+. Pronaći ćete dijelove bilježnice koji se mogu `pokrenuti`, tj. dijelove koda. Možete pokrenuti blok koda odabirom ikone koja izgleda kao gumb za reprodukciju.

2. Odaberite ikonu `md` i dodajte malo markdowna, te sljedeći tekst **# Dobrodošli u svoju bilježnicu**.

   Zatim dodajte malo Python koda.

3. Upišite **print('hello notebook')** u blok koda.
4. Odaberite strelicu za pokretanje koda.

   Trebali biste vidjeti ispisanu izjavu:

    ```output
    hello notebook
    ```

![VS Code s otvorenom bilježnicom](../../../../2-Regression/1-Tools/images/notebook.jpg)

Možete izmjenjivati svoj kod s komentarima kako biste sami dokumentirali bilježnicu.

✅ Razmislite na trenutak koliko se radno okruženje web programera razlikuje od onog znanstvenika podataka.

## Početak rada sa Scikit-learn

Sada kada je Python postavljen u vašem lokalnom okruženju i osjećate se ugodno s Jupyter bilježnicama, upoznajmo se sa Scikit-learn (izgovara se `sci` kao u `science`). Scikit-learn pruža [opsežan API](https://scikit-learn.org/stable/modules/classes.html#api-ref) koji vam pomaže u obavljanju zadataka strojnog učenja.

Prema njihovoj [web stranici](https://scikit-learn.org/stable/getting_started.html), "Scikit-learn je biblioteka za strojno učenje otvorenog koda koja podržava nadzirano i nenadzirano učenje. Također pruža razne alate za prilagodbu modela, predobradu podataka, odabir modela i evaluaciju, te mnoge druge korisne funkcije."

U ovom tečaju koristit ćete Scikit-learn i druge alate za izgradnju modela strojnog učenja za obavljanje onoga što nazivamo 'tradicionalnim zadacima strojnog učenja'. Namjerno smo izbjegli neuronske mreže i duboko učenje jer su bolje obrađeni u našem nadolazećem kurikulumu 'AI za početnike'.

Scikit-learn olakšava izgradnju modela i njihovu evaluaciju za upotrebu. Primarno je fokusiran na korištenje numeričkih podataka i sadrži nekoliko gotovih skupova podataka za učenje. Također uključuje unaprijed izgrađene modele koje studenti mogu isprobati. Istražimo proces učitavanja unaprijed pripremljenih podataka i korištenja ugrađenog procjenitelja za prvi ML model sa Scikit-learnom koristeći osnovne podatke.

## Vježba - vaša prva Scikit-learn bilježnica

> Ovaj je vodič inspiriran [primjerom linearne regresije](https://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html#sphx-glr-auto-examples-linear-model-plot-ols-py) na web stranici Scikit-learn.

[![ML za početnike - Vaš prvi projekt linearne regresije u Pythonu](https://img.youtube.com/vi/2xkXL5EUpS0/0.jpg)](https://youtu.be/2xkXL5EUpS0 "ML za početnike - Vaš prvi projekt linearne regresije u Pythonu")

> 🎥 Kliknite na sliku iznad za kratki video o ovoj vježbi.

U datoteci _notebook.ipynb_ povezanoj s ovom lekcijom, izbrišite sve ćelije pritiskom na ikonu 'kanta za smeće'.

U ovom ćete odjeljku raditi s malim skupom podataka o dijabetesu koji je ugrađen u Scikit-learn za potrebe učenja. Zamislite da želite testirati tretman za pacijente s dijabetesom. Modeli strojnog učenja mogli bi vam pomoći odrediti koji bi pacijenti bolje reagirali na tretman, na temelju kombinacija varijabli. Čak i vrlo osnovni regresijski model, kada se vizualizira, mogao bi pokazati informacije o varijablama koje bi vam pomogle organizirati teoretska klinička ispitivanja.

✅ Postoji mnogo vrsta metoda regresije, a koju ćete odabrati ovisi o odgovoru koji tražite. Ako želite predvidjeti vjerojatnu visinu osobe određene dobi, koristili biste linearnu regresiju jer tražite **numeričku vrijednost**. Ako vas zanima otkrivanje treba li određena kuhinja biti smatrana veganskom ili ne, tražite **kategorizaciju**, pa biste koristili logističku regresiju. Kasnije ćete naučiti više o logističkoj regresiji. Razmislite malo o pitanjima koja možete postaviti podacima i koja bi od ovih metoda bila prikladnija.

Krenimo s ovim zadatkom.

### Uvoz biblioteka

Za ovaj zadatak uvest ćemo neke biblioteke:

- **matplotlib**. Koristan [alat za grafički prikaz](https://matplotlib.org/) koji ćemo koristiti za stvaranje linijskog grafa.
- **numpy**. [numpy](https://numpy.org/doc/stable/user/whatisnumpy.html) je korisna biblioteka za rad s numeričkim podacima u Pythonu.
- **sklearn**. Ovo je [Scikit-learn](https://scikit-learn.org/stable/user_guide.html) biblioteka.

Uvezite neke biblioteke za pomoć pri zadacima.

1. Dodajte uvoze upisivanjem sljedećeg koda:

   ```python
   import matplotlib.pyplot as plt
   import numpy as np
   from sklearn import datasets, linear_model, model_selection
   ```

   Ovdje uvozite `matplotlib`, `numpy` te `datasets`, `linear_model` i `model_selection` iz `sklearn`. `model_selection` se koristi za podjelu podataka na skupove za treniranje i testiranje.

### Skup podataka o dijabetesu

Ugrađeni [skup podataka o dijabetesu](https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset) uključuje 442 uzorka podataka o dijabetesu s 10 značajki, od kojih neke uključuju:

- age: dob u godinama
- bmi: indeks tjelesne mase
- bp: prosječan krvni tlak
- s1 tc: T-stanice (vrsta bijelih krvnih stanica)

✅ Ovaj skup podataka uključuje koncept 'spola' kao varijable značajke važne za istraživanje dijabetesa. Mnogi medicinski skupovi podataka uključuju ovu vrstu binarne klasifikacije. Razmislite malo o tome kako takve kategorizacije mogu isključiti određene dijelove populacije iz tretmana.

Sada učitajte podatke X i y.

> 🎓 Zapamtite, ovo je nadzirano učenje i trebamo imenovanu ciljnu varijablu 'y'.

U novoj ćeliji koda učitajte skup podataka o dijabetesu pozivom `load_diabetes()`. Ulaz `return_X_y=True` signalizira da će `X` biti matrica podataka, a `y` ciljana varijabla regresije.

1. Dodajte neke naredbe za ispis kako biste prikazali oblik matrice podataka i njezin prvi element:

    ```python
    X, y = datasets.load_diabetes(return_X_y=True)
    print(X.shape)
    print(X[0])
    ```

    Ono što dobivate kao odgovor je tuple. Ono što radite je dodjeljivanje prva dva elementa tuplea varijablama `X` i `y`. Saznajte više [o tupleovima](https://wikipedia.org/wiki/Tuple).

    Možete vidjeti da ovi podaci imaju 442 stavke oblikovane u nizove od 10 elemenata:

    ```text
    (442, 10)
    [ 0.03807591  0.05068012  0.06169621  0.02187235 -0.0442235  -0.03482076
    -0.04340085 -0.00259226  0.01990842 -0.01764613]
    ```

    ✅ Razmislite malo o odnosu između podataka i ciljne varijable regresije. Linearna regresija predviđa odnose između značajke X i ciljne varijable y. Možete li pronaći [cilj](https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset) za skup podataka o dijabetesu u dokumentaciji? Što ovaj skup podataka pokazuje, s obzirom na cilj?

2. Zatim odaberite dio ovog skupa podataka za grafički prikaz odabirom 3. stupca skupa podataka. To možete učiniti korištenjem operatora `:` za odabir svih redaka, a zatim odabirom 3. stupca pomoću indeksa (2). Također možete preoblikovati podatke u 2D niz - kako je potrebno za grafički prikaz - korištenjem `reshape(n_rows, n_columns)`. Ako je jedan od parametara -1, odgovarajuća dimenzija se automatski izračunava.

   ```python
   X = X[:, 2]
   X = X.reshape((-1,1))
   ```

   ✅ U bilo kojem trenutku ispišite podatke kako biste provjerili njihov oblik.

3. Sada kada su podaci spremni za grafički prikaz, možete vidjeti može li stroj pomoći u određivanju logičke podjele između brojeva u ovom skupu podataka. Da biste to učinili, trebate podijeliti i podatke (X) i cilj (y) na skupove za testiranje i treniranje. Scikit-learn ima jednostavan način za to; možete podijeliti svoje testne podatke na određenoj točki.

   ```python
   X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.33)
   ```

4. Sada ste spremni trenirati svoj model! Učitajte model linearne regresije i trenirajte ga s vašim X i y skupovima za treniranje koristeći `model.fit()`:

    ```python
    model = linear_model.LinearRegression()
    model.fit(X_train, y_train)
    ```

    ✅ `model.fit()` je funkcija koju ćete vidjeti u mnogim ML bibliotekama poput TensorFlowa.

5. Zatim, stvorite predviđanje koristeći testne podatke, koristeći funkciju `predict()`. Ovo će se koristiti za crtanje linije između grupa podataka.

    ```python
    y_pred = model.predict(X_test)
    ```

6. Sada je vrijeme za prikaz podataka na grafu. Matplotlib je vrlo koristan alat za ovaj zadatak. Stvorite scatterplot svih X i y testnih podataka i koristite predviđanje za crtanje linije na najprikladnijem mjestu između grupiranja podataka modela.

    ```python
    plt.scatter(X_test, y_test,  color='black')
    plt.plot(X_test, y_pred, color='blue', linewidth=3)
    plt.xlabel('Scaled BMIs')
    plt.ylabel('Disease Progression')
    plt.title('A Graph Plot Showing Diabetes Progression Against BMI')
    plt.show()
    ```

   ![scatterplot koji prikazuje podatke o dijabetesu](../../../../2-Regression/1-Tools/images/scatterplot.png)
✅ Razmislite malo o tome što se ovdje događa. Ravna linija prolazi kroz mnogo malih točaka podataka, ali što točno radi? Možete li vidjeti kako biste trebali moći koristiti ovu liniju za predviđanje gdje bi nova, neviđena točka podataka trebala pripadati u odnosu na y-os grafikona? Pokušajte riječima opisati praktičnu primjenu ovog modela.

Čestitamo, izradili ste svoj prvi model linearne regresije, napravili predviđanje s njim i prikazali ga na grafikonu!

---
## 🚀Izazov

Prikažite drugu varijablu iz ovog skupa podataka. Savjet: uredite ovu liniju: `X = X[:,2]`. S obzirom na cilj ovog skupa podataka, što možete otkriti o napredovanju dijabetesa kao bolesti?
## [Kviz nakon predavanja](https://ff-quizzes.netlify.app/en/ml/)

## Pregled i samostalno učenje

U ovom ste vodiču radili s jednostavnom linearnom regresijom, a ne s univarijatnom ili višestrukom linearnom regresijom. Pročitajte malo o razlikama između ovih metoda ili pogledajte [ovaj video](https://www.coursera.org/lecture/quantifying-relationships-regression-models/linear-vs-nonlinear-categorical-variables-ai2Ef).

Pročitajte više o konceptu regresije i razmislite o vrstama pitanja na koja se može odgovoriti ovom tehnikom. Prođite kroz [ovaj vodič](https://docs.microsoft.com/learn/modules/train-evaluate-regression-models?WT.mc_id=academic-77952-leestott) kako biste produbili svoje razumijevanje.

## Zadatak

[Drugi skup podataka](assignment.md)

---

**Odricanje od odgovornosti**:  
Ovaj dokument je preveden korištenjem AI usluge za prevođenje [Co-op Translator](https://github.com/Azure/co-op-translator). Iako nastojimo osigurati točnost, imajte na umu da automatski prijevodi mogu sadržavati pogreške ili netočnosti. Izvorni dokument na izvornom jeziku treba smatrati mjerodavnim izvorom. Za ključne informacije preporučuje se profesionalni prijevod od strane stručnjaka. Ne preuzimamo odgovornost za bilo kakve nesporazume ili pogrešne interpretacije proizašle iz korištenja ovog prijevoda.