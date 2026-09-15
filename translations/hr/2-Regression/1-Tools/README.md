# Početak s Pythonom i Scikit-learn za regresijske modele

![Sažetak regresija u sketchnote obliku](../../../../translated_images/hr/ml-regression.4e4f70e3b3ed446e.webp)

> Sketchnote autora [Tomomi Imura](https://www.twitter.com/girlie_mac)

## [Kviz prije predavanja](https://ff-quizzes.netlify.app/en/ml/)

> ### [Ova lekcija dostupna je i na R!](../../../../2-Regression/1-Tools/solution/R/lesson_1.html)

## Uvod

U ovih četiriju lekcija otkrit ćete kako izraditi regresijske modele. Uskoro ćemo objasniti čemu oni služe. No prije nego što išta napravite, provjerite imate li potrebne alate za početak procesa!

U ovoj lekciji naučit ćete kako:

- Konfigurirati svoje računalo za zadatke lokalnog strojnog učenja.
- Raditi s Jupyter bilježnicama.
- Koristiti Scikit-learn, uključujući instalaciju.
- Istražiti linearnu regresiju kroz praktični zadatak.

## Instalacije i konfiguracije

[![ML za početnike - Pripremite alate za izradu modela strojnog učenja](https://img.youtube.com/vi/-DfeD2k2Kj0/0.jpg)](https://youtu.be/-DfeD2k2Kj0 "ML za početnike - Pripremite alate za izradu modela strojnog učenja")

> 🎥 Kliknite gornju sliku za kratki video o konfiguriranju računala za ML.

1. **Instalirajte Python**. Provjerite je li [Python](https://www.python.org/downloads/) instaliran na vašem računalu. Python ćete koristiti za mnoge zadatke u znanosti o podacima i strojnog učenja. Većina računala već ima instaliran Python. Također postoje korisni [Python paketi za kodiranje](https://code.visualstudio.com/learn/educators/installers?WT.mc_id=academic-77952-leestott) koji olakšavaju postavljanje nekim korisnicima.

   Neki slučajevi upotrebe Pythona zahtijevaju jednu verziju softvera, dok drugi zahtijevaju drugu. Zato je korisno raditi unutar [virtualnog okruženja](https://docs.python.org/3/library/venv.html).

2. **Instalirajte Visual Studio Code**. Provjerite imate li Visual Studio Code instaliran na računalu. Slijedite upute za [instalaciju Visual Studio Code](https://code.visualstudio.com/) za osnovnu instalaciju. U ovom tečaju koristit ćete Python u Visual Studio Codeu, stoga se možda želite upoznati s načinom na koji se može [konfigurirati Visual Studio Code](https://docs.microsoft.com/learn/modules/python-install-vscode?WT.mc_id=academic-77952-leestott) za razvoj u Pythonu.

   > Udobno se upoznajte s Pythonom radeći kroz ovaj skup [Learn modula](https://docs.microsoft.com/users/jenlooper-2911/collections/mp1pagggd5qrq7?WT.mc_id=academic-77952-leestott)
   >
   > [![Postavljanje Pythona u Visual Studio Code](https://img.youtube.com/vi/yyQM70vi7V8/0.jpg)](https://youtu.be/yyQM70vi7V8 "Postavljanje Pythona u Visual Studio Code")
   >
   > 🎥 Kliknite gornju sliku za video: korištenje Pythona unutar VS Code-a.

3. **Instalirajte Scikit-learn**, slijedeći [ove upute](https://scikit-learn.org/stable/install.html). Budući da trebate koristiti Python 3, preporuča se virtualno okruženje. Napomena, ako instalirate ovu biblioteku na M1 Mac, na povezanoj stranici postoje posebne upute.

1. **Instalirajte Jupyter Notebook**. Trebat ćete [instalirati paket Jupyter](https://pypi.org/project/jupyter/).

## Vaše okruženje za razvoj ML-a

Koristit ćete **bilježnice** za razvoj Python koda i izradu modela strojnog učenja. Ova vrsta datoteke često se koristi kod znanstvenika podataka, a prepoznaje se po sufiksu ili ekstenziji `.ipynb`.

Bilježnice su interaktivno okruženje koje programeru omogućuje i kodiranje i dodavanje bilješki te pisanje dokumentacije oko koda, što je vrlo korisno za eksperimentalne ili istraživačke projekte.

[![ML za početnike - Postavite Jupyter bilježnice za početak izrade regresijskih modela](https://img.youtube.com/vi/7E-jC8FLA2E/0.jpg)](https://youtu.be/7E-jC8FLA2E "ML za početnike - Postavite Jupyter bilježnice za početak izrade regresijskih modela")

> 🎥 Kliknite gornju sliku za kratki video kroz ovaj zadatak.

### Vježba - rad s bilježnicom

U ovoj mapi pronaći ćete datoteku _notebook.ipynb_.

1. Otvorite _notebook.ipynb_ u Visual Studio Codeu.

   Jupyter poslužitelj će se pokrenuti s Python 3+. Naći ćete dijelove bilježnice koji se mogu `pokrenuti`, odnosno blokove koda. Kôd se može pokrenuti klikom na ikonu koja izgleda kao tipka za reprodukciju.

1. Odaberite `md` ikonu i dodajte malo markdowna i sljedeći tekst **# Dobrodošli u vašu bilježnicu**.

   Zatim dodajte malo Python koda.

1. Upisujte **print('hello notebook')** u blok koda.
1. Kliknite strelicu da pokrenete kod.

   Trebali biste vidjeti ispisanu izjavu:

    ```output
    hello notebook
    ```

![VS Code s otvorenom bilježnicom](../../../../translated_images/hr/notebook.4a3ee31f396b8832.webp)

Kod možete ispreplitati s komentarima radi samodokumentiranja bilježnice.

✅ Razmislite na trenutak koliko se radno okruženje web programera razlikuje od okruženja znanstvenika podataka.

## Pokretanje sa Scikit-learn

Sada kada je Python postavljen u vašem lokalnom okruženju i kada ste upoznati s Jupyter bilježnicama, upoznajmo se jednako s Scikit-learnom (izgovara se `sci` kao u `znanost`). Scikit-learn pruža [opširan API](https://scikit-learn.org/stable/modules/classes.html#api-ref) za pomoć u izvršavanju ML zadataka.

Prema njihovoj [web stranici](https://scikit-learn.org/stable/getting_started.html), "Scikit-learn je open source biblioteka za strojno učenje koja podržava nadzirano i nenadzirano učenje. Također pruža razne alate za namještanje modela, predobradu podataka, odabir i evaluaciju modela, te mnoge druge korisne funkcije."

U ovom tečaju koristit ćete Scikit-learn i ostale alate za izradu modela strojnog učenja za izvođenje zadataka koje nazivamo 'tradicionalnim strojnim učenjem'. Namjerno smo izbjegli neuronske mreže i duboko učenje, jer su oni bolje pokriveni u našem budućem programu 'AI za početnike'.

Scikit-learn olakšava izradu modela i njihovu evaluaciju za uporabu. Primarno je fokusiran na uporabu numeričkih podataka i sadrži nekoliko pripremljenih skupova podataka za učenje. Također uključuje gotove modele za isprobavanje. Istražimo postupak učitavanja unaprijed spremljenih podataka i korištenja ugrađenog estimator za izradu prvog ML modela sa Scikit-learnom na osnovnim podacima.

## Vježba - vaša prva Scikit-learn bilježnica

> Ovaj vodič inspiriran je [primjerom linearne regresije](https://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html#sphx-glr-auto-examples-linear-model-plot-ols-py) na službenoj stranici Scikit-learna.


[![ML za početnike - Vaš prvi linearni regresijski projekt u Pythonu](https://img.youtube.com/vi/2xkXL5EUpS0/0.jpg)](https://youtu.be/2xkXL5EUpS0 "ML za početnike - Vaš prvi linearni regresijski projekt u Pythonu")

> 🎥 Kliknite sliku iznad za kratki video kroz ovaj zadatak.

U datoteci _notebook.ipynb_ povezanoj s ovom lekcijom obrišite sve ćelije pritiskom na ikonu 'kanta za smeće'.

U ovom dijelu radit ćete s malim skupom podataka o dijabetesu koji je ugrađen u Scikit-learn radi učenja. Zamislite da želite testirati tretman za dijabetičare. Modeli strojnog učenja mogli bi vam pomoći utvrditi koji bi pacijenti bolje reagirali na tretman, na temelju kombinacija varijabli. Čak i vrlo osnovni regresijski model, kad se prikaže grafički, mogao bi pokazati informacije o varijablama koje bi vam pomogle organizirati vaše teorijske kliničke pokuse.

✅ Postoje mnoge metode regresije, a koju ćete odabrati ovisi o pitanju na koje tražite odgovor. Ako želite predvidjeti vjerojatnu visinu osobe određene dobi, koristit ćete linearnu regresiju jer tražite **numeričku vrijednost**. Ako vas zanima je li neka vrsta kuhinje veganska ili ne, tražite **pridruživanje kategoriji** pa biste koristili logističku regresiju. O logističkoj regresiji naučit ćete više kasnije. Razmislite o nekim pitanjima koja možete postaviti podacima i koja bi metoda bila prikladnija.

Krenimo s ovim zadatkom.

### Uvoz biblioteka

Za ovaj zadatak uvest ćemo nekoliko biblioteka:

- **matplotlib**. Koristan je [alat za grafove](https://matplotlib.org/) i upotrijebit ćemo ga za crtanje linijskog grafa.
- **numpy**. [numpy](https://numpy.org/doc/stable/user/whatisnumpy.html) je korisna biblioteka za rukovanje numeričkim podacima u Pythonu.
- **sklearn**. To je [Scikit-learn](https://scikit-learn.org/stable/user_guide.html) biblioteka.

Uvezite neke biblioteke koje će vam pomoći u zadacima.

1. Dodajte uvoze upisujući sljedeći kod:

   ```python
   import matplotlib.pyplot as plt
   import numpy as np
   from sklearn import datasets, linear_model, model_selection
   ```

   Gore uvozite `matplotlib`, `numpy` te `datasets`, `linear_model` i `model_selection` iz `sklearn`. `model_selection` se koristi za razdvajanje podataka na trening i test skupove.

### Skup podataka o dijabetesu

Ugrađeni [skup podataka o dijabetesu](https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset) sadrži 442 uzorka podataka o dijabetesu s 10 značajki, od kojih su neke:

- age: starost u godinama
- bmi: indeks tjelesne mase
- bp: prosječni krvni tlak
- s1 tc: T-stanice (vrsta bijelih krvnih stanica)

✅ Ovaj skup podataka uključuje 'spol' kao značajku važnu za istraživanje dijabetesa. Mnogi medicinski skupovi podataka sadrže ovu vrstu binarne klasifikacije. Razmislite kako ovakve kategorizacije mogu isključiti određene dijelove populacije iz tretmana.

Sada učitajte podatke X i y.

> 🎓 Zapamtite, ovo je nadzirano učenje i potreban nam je cilj 'y' s imenom.

U novoj ćeliji koda učitajte skup podataka o dijabetesu pozivom `load_diabetes()`. Ulaz `return_X_y=True` označava da će `X` biti matrica podataka, a `y` cilj regresije.

1. Dodajte naredbe za ispis da prikažete oblik matrice podataka i njezin prvi element:

    ```python
    X, y = datasets.load_diabetes(return_X_y=True)
    print(X.shape)
    print(X[0])
    ```

    Ono što dobivate kao odgovor je 'tuple'. Ono što radite je da prva dva vrijednosna elementa tuplea dodijelite `X` i `y`. Više o [tupleima](https://wikipedia.org/wiki/Tuple).

    Vidite da ovi podaci imaju 442 stavke oblikovane u nizove od 10 elemenata:

    ```text
    (442, 10)
    [ 0.03807591  0.05068012  0.06169621  0.02187235 -0.0442235  -0.03482076
    -0.04340085 -0.00259226  0.01990842 -0.01764613]
    ```

    ✅ Razmislite o odnosu između podataka i cilja regresije. Linearna regresija predviđa odnose između značajke X i ciljne varijable y. Možete li pronaći [cilj](https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset) za skup podataka o dijabetesu u dokumentaciji? Što ovaj skup podataka pokazuje, s obzirom na cilj?

2. Zatim odaberite dio ovog skupa podataka za prikaz crtežom, odabirom 3. stupca skupa podataka. To možete napraviti operatorom `:` koji odabire sve retke, a zatim odabirom 3. stupca korištenjem indeksa (2). Također možete promijeniti oblik podataka u 2D niz - što je potrebno za crtanje - korištenjem `reshape(n_rows, n_columns)`. Ako je jedan od parametara -1, odgovarajuća dimenzija se računa automatski.

   ```python
   X = X[:, 2]
   X = X.reshape((-1,1))
   ```

   ✅ Uvijek ispišite podatke da provjerite njihov oblik.

3. Sada kada imate podatke spremne za crtanje, provjerite može li stroj pomoći u određivanju logičnog razdvajanja među brojevima u ovom skupu podataka. Za to trebate podijeliti i podatke (X) i cilj (y) na testne i trening skupove. Scikit-learn ima jednostavan način za to; podatke za test možete podijeliti na određenoj točki.

   ```python
   X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.33)
   ```

4. Sada ste spremni za treniranje modela! Učitajte model linearne regresije i trenirajte ga s vašim X i y trening skupovima koristeći `model.fit()`:

    ```python
    model = linear_model.LinearRegression()
    model.fit(X_train, y_train)
    ```

    ✅ `model.fit()` je funkcija koju ćete vidjeti u mnogim ML bibliotekama poput TensorFlowa

5. Zatim napravite predviđanje koristeći testne podatke, koristeći funkciju `predict()`. Ona će se koristiti za crtanje linije između skupina podataka

    ```python
    y_pred = model.predict(X_test)
    ```

6. Vrijeme je da prikažete podatke na grafu. Matplotlib je vrlo koristan alat za ovaj zadatak. Napravite scatter plot svih X i y testnih podataka, i upotrijebite predviđanje za crtanje linije na najprikladnijem mjestu između skupina modela.

    ```python
    plt.scatter(X_test, y_test,  color='black')
    plt.plot(X_test, y_pred, color='blue', linewidth=3)
    plt.xlabel('Scaled BMIs')
    plt.ylabel('Disease Progression')
    plt.title('A Graph Plot Showing Diabetes Progression Against BMI')
    plt.show()
    ```

   ![scatter plot prikazujući podatke o dijabetesu](../../../../translated_images/hr/scatterplot.ad8b356bcbb33be6.webp)

   ✅ Razmislite malo o tome što se događa. Ravna linija prolazi kroz mnogo malih točaka podataka, ali što točno radi? Možete li vidjeti kako biste mogli koristiti ovu liniju da predvidite gdje bi nova, neviđena točka podataka trebala stati u odnosu na y os grafa? Pokušajte riječima opisati praktičnu upotrebu ovog modela.

Čestitamo, izradili ste svoj prvi model linearne regresije, napravili predviđanje i prikazali ga na grafu!

---
## 🚀Izazov

Prikažite drugu varijablu iz ovog skupa podataka. Savjet: uredite ovaj redak: `X = X[:,2]`. S obzirom na cilj ovog skupa podataka, što možete otkriti o progresiji dijabetesa kao bolesti?
## [Kviz nakon predavanja](https://ff-quizzes.netlify.app/en/ml/)

## Pregled i samostalno učenje

U ovom vodiču radili ste s jednostavnom linearnom regresijom, a ne univarijatnom ili višestrukom linearnom regresijom. Pročitajte malo o razlikama između ovih metoda, ili pogledajte [ovaj video](https://www.coursera.org/lecture/quantifying-relationships-regression-models/linear-vs-nonlinear-categorical-variables-ai2Ef)

Pročitajte više o konceptu regresije i razmislite o vrstama pitanja na koja se ovom tehnikom može odgovoriti. Prođite kroz ovaj [tutorial](https://docs.microsoft.com/learn/modules/train-evaluate-regression-models?WT.mc_id=academic-77952-leestott) kako biste produbili svoje razumijevanje.

## Zadatak

[Drugi skup podataka](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**Napomena**:
Ovaj dokument je preveden korištenjem AI prevoditeljskog servisa [Co-op Translator](https://github.com/Azure/co-op-translator). Iako težimo točnosti, imajte na umu da automatski prijevodi mogu sadržavati greške ili netočnosti. Izvorni dokument na izvornom jeziku treba smatrati autoritativnim izvorom. Za važne informacije preporuča se profesionalni ljudski prijevod. Nismo odgovorni za bilo kakva nesporazumevanja ili pogrešne interpretacije koje proizlaze iz korištenja ovog prijevoda.
<!-- CO-OP TRANSLATOR DISCLAIMER END -->