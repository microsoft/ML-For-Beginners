# Začnite s Python in Scikit-learn za regresijske modele

![Povzetek regresij v sketchnote](../../../../translated_images/sl/ml-regression.4e4f70e3b3ed446e.webp)

> Sketchnote avtorice [Tomomi Imura](https://www.twitter.com/girlie_mac)

## [Pred-predavanjski kviz](https://ff-quizzes.netlify.app/en/ml/)

> ### [Ta lekcija je na voljo tudi v jeziku R!](../../../../2-Regression/1-Tools/solution/R/lesson_1.html)

## Uvod

V teh štirih lekcijah boste odkrili, kako sestaviti regresijske modele. Kmalu bomo razpravljali, za kaj so ti namenjeni. A preden karkoli storite, se prepričajte, da imate pravilna orodja pripravljena za začetek procesa!

V tej lekciji se boste naučili:

- Konfigurirati svoj računalnik za lokalne naloge strojnega učenja.
- Delati z Jupyter beležnicami.
- Uporabljati Scikit-learn, vključno z namestitvijo.
- Raziskovati linearno regresijo s praktično vajo.

## Namestitve in konfiguracije

[![ML za začetnike - Pripravite orodja za gradnjo modelov strojnega učenja](https://img.youtube.com/vi/-DfeD2k2Kj0/0.jpg)](https://youtu.be/-DfeD2k2Kj0 "ML za začetnike - Pripravite orodja za gradnjo modelov strojnega učenja")

> 🎥 Kliknite zgornjo sliko za kratki video o konfiguriranju računalnika za ML.

1. **Namestite Python**. Prepričajte se, da imate na računalniku nameščen [Python](https://www.python.org/downloads/). Za mnoge naloge podatkovne znanosti in strojnega učenja boste uporabljali Python. Večina računalniških sistemov že vključuje namestitev Pythona. Na voljo so tudi koristni [Python Coding Paketi](https://code.visualstudio.com/learn/educators/installers?WT.mc_id=academic-77952-leestott), ki olajšajo nastavitev nekaterim uporabnikom.

   Nekateri načini uporabe Pythona pa zahtevajo eno verzijo programske opreme, medtem ko drugi drugo. Zato je koristno delati v [virtualnem okolju](https://docs.python.org/3/library/venv.html).

2. **Namestite Visual Studio Code**. Prepričajte se, da imate nameščen Visual Studio Code. Sledite tem navodilom za [namestitev Visual Studio Code](https://code.visualstudio.com/). V tem tečaju boste uporabljali Python v Visual Studio Code, zato boste morda želeli osvežiti znanje, kako [konfigurirati Visual Studio Code](https://docs.microsoft.com/learn/modules/python-install-vscode?WT.mc_id=academic-77952-leestott) za Python razvoj.

   > Seveda se sprijaznite s Pythonom tako, da preletite to zbirko [Learn modulov](https://docs.microsoft.com/users/jenlooper-2911/collections/mp1pagggd5qrq7?WT.mc_id=academic-77952-leestott)
   >
   > [![Namestite Python z Visual Studio Code](https://img.youtube.com/vi/yyQM70vi7V8/0.jpg)](https://youtu.be/yyQM70vi7V8 "Namestite Python z Visual Studio Code")
   >
   > 🎥 Kliknite zgornjo sliko za video: uporaba Pythona znotraj VS Code.

3. **Namestite Scikit-learn**, tako, da sledite [tem navodilom](https://scikit-learn.org/stable/install.html). Ker morate zagotoviti, da uporabljate Python 3, se priporoča uporaba virtualnega okolja. Če nameščate to knjižnico na M1 Mac, so na zgornji strani posebna navodila.

1. **Namestite Jupyter Notebook**. Potrebno bo [namestiti Jupyter paket](https://pypi.org/project/jupyter/).

## Vaše avtorsko okolje za ML

Za razvoj vaše Python kode in ustvarjanje modelov strojnega učenja boste uporabljali **beležnice**. Ta vrsta datoteke je pogosto orodje podatkovnih znanstvenikov in lahko jih prepoznamo po priponi `.ipynb`.

Beležnice so interaktivno okolje, ki razvijalcu omogoča tako kodiranje kot dodajanje opomb ter pisanje dokumentacije okoli kode, kar je zelo koristno za eksperimentalne ali raziskovalne projekte.

[![ML za začetnike - Nastavite Jupyter beležnice za začetek gradnje regresijskih modelov](https://img.youtube.com/vi/7E-jC8FLA2E/0.jpg)](https://youtu.be/7E-jC8FLA2E "ML za začetnike - Nastavite Jupyter beležnice za začetek gradnje regresijskih modelov")

> 🎥 Kliknite zgornjo sliko za kratki video o tej vaji.

### Vaja - delo z beložnico

V tej mapi boste našli datoteko _notebook.ipynb_.

1. Odprite _notebook.ipynb_ v Visual Studio Code.

   Zažene se Jupyter strežnik s Python 3+. V beležnici boste našli odseke, ki jih lahko `pobegnete` oziroma izvajate, to so koščki kode. Kodo lahko zaženete z izbiro ikone, ki spominja na gumb za predvajanje.

1. Izberite ikono `md` in dodajte nekaj markdowna ter naslednji tekst **# Dobrodošli v vaši beležnici**.

   Nato dodajte nekaj Python kode.

1. V kodo tipkajte **print('hello notebook')**.
1. Kliknite puščico za zagon kode.

   Videli boste natisnjen stavek:

    ```output
    hello notebook
    ```

![VS Code z odprto beležnico](../../../../translated_images/sl/notebook.4a3ee31f396b8832.webp)

Kodo lahko prepletate s komentarji, da beležnico samodokumentirate.

✅ Razmislite za trenutek, kako drugačno je delovno okolje spletnega razvijalca v primerjavi s podatkovnim znanstvenikom.

## Začetek z Scikit-learn

Zdaj, ko imate Python nastavljen v svojem lokalnem okolju in ste udobni z Jupyter beležnicami, postanite enako vešči s Scikit-learn (izgovarjajte `sci` kot v `science`). Scikit-learn zagotavlja [obsežen API](https://scikit-learn.org/stable/modules/classes.html#api-ref), da vam pomaga izvajati ML naloge.

Po njihovi [spletni strani](https://scikit-learn.org/stable/getting_started.html) "je Scikit-learn odprtokodna knjižnica za strojno učenje, ki podpira nadzorovano in nenadzorovano učenje. Prav tako ponuja različna orodja za prilagajanje modelov, predobdelavo podatkov, izbor modela in ocenjevanje ter številne druge pripomočke."

V tem tečaju boste uporabljali Scikit-learn in druga orodja za gradnjo modelov strojnega učenja za naloge, ki jih imenujemo 'tradicionalno strojno učenje'. Veščine nevronskih mrež in globokega učenja smo zavestno izpustili, ker so bolje pokrite v našem prihajajočem kurikulu 'AI za začetnike'.

Scikit-learn omogoča enostavno izdelavo modelov in njihovo ocenjevanje za uporabo. Osredotočen je predvsem na uporabo numeričnih podatkov in vsebuje več že pripravljenih zbirk podatkov kot učna orodja. Vključuje tudi vnaprej izdelane modele, ki jih lahko preizkušajo študenti. Raziščimo postopek nalaganja pripravljenih podatkov in uporabo vgrajenega estimatorja za ustvarjanje vašega prvega ML modela s Scikit-learn na osnovnih podatkih.

## Vaja - vaša prva Scikit-learn beležnica

> Ta vadnica je navdihnjena z [primerom linearne regresije](https://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html#sphx-glr-auto-examples-linear-model-plot-ols-py) na spletni strani Scikit-learn.


[![ML za začetnike - Vaš prvi linearni regresijski projekt v Pythonu](https://img.youtube.com/vi/2xkXL5EUpS0/0.jpg)](https://youtu.be/2xkXL5EUpS0 "ML za začetnike - Vaš prvi linearni regresijski projekt v Pythonu")

> 🎥 Kliknite zgornjo sliko za kratek video o tej vaji.

V datoteki _notebook.ipynb_ povezani s to lekcijo počistite vse celice tako, da kliknete ikono 'košarica'.

V tem razdelku boste delali z majhnim naborom podatkov o sladkorni bolezni, ki je vključena v Scikit-learn za učne namene. Predstavljajte si, da želite testirati zdravljenje za bolnike s sladkorno boleznijo. Modeli strojnega učenja bi vam lahko pomagali določiti, kateri bolniki bodo bolje reagirali na zdravljenje, glede na kombinacije spremenljivk. Tudi zelo osnovni regresijski model, ko ga vizualno prikažemo, lahko prikaže informacije o spremenljivkah, ki bi vam pomagale organizirati teoretične klinične študije.

✅ Obstaja mnogo vrst regresijskih metod in katero izberete, je odvisno od vprašanja, na katero želite odgovoriti. Če želite napovedati verjetno višino osebe določene starosti, uporabite linearno regresijo, saj iščete **numerično vrednost**. Če vas zanima ugotoviti, ali naj se določena kuhinja šteje za vegansko ali ne, iščete **kategorijsko dodelitev**, torej bi uporabili logistično regresijo. O logistični regresiji se boste pozneje naučili več. Razmislite o nekaj vprašanjih, ki jih lahko zastavite podatkom, in katera od teh metod bi bila bolj primerna.

Začnimo to nalogo.

### Uvoz knjižnic

Za to nalogo bomo uvozili nekaj knjižnic:

- **matplotlib**. Uporaben [grafični pripomoček](https://matplotlib.org/), ki ga bomo uporabili za risanje črtnega diagrama.
- **numpy**. [numpy](https://numpy.org/doc/stable/user/whatisnumpy.html) je uporabna knjižnica za ravnanje z numeričnimi podatki v Pythonu.
- **sklearn**. To je [Scikit-learn](https://scikit-learn.org/stable/user_guide.html) knjižnica.

Uvozite nekaj knjižnic, ki vam bodo pomagale pri nalogah.

1. Dodajte uvoze tako, da vnesete naslednjo kodo:

   ```python
   import matplotlib.pyplot as plt
   import numpy as np
   from sklearn import datasets, linear_model, model_selection
   ```

   Zgornje uvoze `matplotlib`, `numpy` in iz `sklearn` uvozite `datasets`, `linear_model` in `model_selection`. `model_selection` se uporablja za razdeljevanje podatkov na učne in testne sklope.

### Nabor podatkov o sladkorni bolezni

Vgrajeni [nabor podatkov o sladkorni bolezni](https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset) vključuje 442 vzorcev podatkov o sladkorni bolezni z 10 funkcijskimi spremenljivkami, od katerih so nekatere:

- starost: starost v letih
- bmi: indeks telesne mase
- bp: povprečni krvni tlak
- s1 tc: T-celice (vrsta belih krvničk)

✅ Ta nabor podatkov vključuje koncept 'spola' kot funkcijsko spremenljivko, pomembno za raziskave sladkorne bolezni. Veliko medicinskih zbirk podatkov vključuje tovrstno binarno klasifikacijo. Razmislite o tem, kako lahko takšni kategorizaciji izključijo določene dele populacije iz zdravljenja.

Zdaj naložite podatke X in y.

> 🎓 Ne pozabite, gre za nadzorovano učenje, zato potrebujemo imenovani cilj 'y'.

V novi celici s kodo naložite nabor podatkov o sladkorni bolezni tako, da pokličete `load_diabetes()`. Vnos `return_X_y=True` pomeni, da bo `X` podatkovna matrika, `y` pa cilj regresije.

1. Dodajte nekaj ukazov print za prikaz oblike podatkovne matrike in njenega prvega elementa:

    ```python
    X, y = datasets.load_diabetes(return_X_y=True)
    print(X.shape)
    print(X[0])
    ```

    Kar boste dobili nazaj kot odziv, je tuple. S tem postopkom dodelite prvi dve vrednosti tuple spremenljivkama `X` in `y`. Več o [tuple-ih](https://wikipedia.org/wiki/Tuple) se naučite tukaj.

    Vidite lahko, da ta podatkovna množica vsebuje 442 elementov v nizi z 10 elementi:

    ```text
    (442, 10)
    [ 0.03807591  0.05068012  0.06169621  0.02187235 -0.0442235  -0.03482076
    -0.04340085 -0.00259226  0.01990842 -0.01764613]
    ```

    ✅ Razmislite o odnosu med podatki in ciljem regresije. Linearna regresija napoveduje odnose med funkcijo X in ciljno spremenljivko y. Ali lahko v dokumentaciji najdete [cilj](https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset) za nabor podatkov o sladkorni bolezni? Kaj ta nabor podatkov prikazuje ob upoštevanju tega cilja?

2. Nato izberite del tega nabora podatkov posebej za prikaz tako, da izberete 3. stolpec nabora. To naredite z uporabo operatorja `:` za izbiro vseh vrstic in nato izberete 3. stolpec z indeksom (2). Podatke lahko tudi prerazporedite v 2D matriko - kot je zahtevano za risanje - z uporabo `reshape(n_rows, n_columns)`. Če je eden parametrov -1, se ustrezna dimenzija izračuna samodejno.

   ```python
   X = X[:, 2]
   X = X.reshape((-1,1))
   ```

   ✅ Kadarkoli izpišite podatke, da preverite njihovo obliko.

3. Zdaj, ko imate podatke pripravljene na prikaz, lahko preverite, ali vam stroj pomaga določiti logično razmejitev med številkami v tem naboru. Da to naredite, morate podatke (X) in cilj (y) razdeliti na testne in učne sklope. Scikit-learn ima preprost način za to; lahko razdelite testne podatke na določenem mestu.

   ```python
   X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.33)
   ```

4. Zdaj ste pripravljeni za učenje modela! Naložite linearni regresijski model in ga izučite z vašimi X in y učnimi sklopi z uporabo `model.fit()`:

    ```python
    model = linear_model.LinearRegression()
    model.fit(X_train, y_train)
    ```

    ✅ `model.fit()` je funkcija, ki jo boste videli v mnogih ML knjižnicah, kot je TensorFlow

5. Nato ustvarite napoved z uporabo testnih podatkov s funkcijo `predict()`. To bo uporabljeno za risanje črte med podatkovnimi skupinami.

    ```python
    y_pred = model.predict(X_test)
    ```

6. Zdaj je čas, da prikažete podatke na grafu. Matplotlib je zelo uporabno orodje za to nalogo. Naredite razpršen diagram vseh testnih podatkov X in y, ter uporabite napoved za risanje črte na najbolj primernem mestu, med podatkovnimi skupinami modela.

    ```python
    plt.scatter(X_test, y_test,  color='black')
    plt.plot(X_test, y_pred, color='blue', linewidth=3)
    plt.xlabel('Scaled BMIs')
    plt.ylabel('Disease Progression')
    plt.title('A Graph Plot Showing Diabetes Progression Against BMI')
    plt.show()
    ```

   ![razpršen diagram prikazuje podatkovne točke o sladkorni bolezni](../../../../translated_images/sl/scatterplot.ad8b356bcbb33be6.webp)

   ✅ Razmislite, kaj se tukaj dogaja. Ravna črta gre skozi mnogo majhnih točk podatkov, a kaj pravzaprav počne? Ali vidite, kako lahko uporabite to črto za napovedovanje, kje naj bi se uvrstila nova, nevidena podatkovna točka glede na os y grafa? Poskusite z besedami izraziti praktično uporabo tega modela.

Čestitke, sestavili ste svoj prvi linearni regresijski model, ustvarili napoved z njim in jo prikazali na grafu!

---
## 🚀Izziv

Narišite drugačno spremenljivko iz tega nabora podatkov. Namig: uredite vrstico `X = X[:,2]`. Glede na cilj tega nabora podatkov, kaj lahko odkrijete o napredovanju sladkorne bolezni kot bolezni?
## [Po-predavanjski kviz](https://ff-quizzes.netlify.app/en/ml/)

## Pregled in samostojno učenje

V tej vadnici ste delali z enostavno linearno regresijo, namesto z univariatno ali multiplo linearno regresijo. Preberite nekaj o razlikah med temi metodami ali si oglejte [ta video](https://www.coursera.org/lecture/quantifying-relationships-regression-models/linear-vs-nonlinear-categorical-variables-ai2Ef)

Preberite več o konceptu regresije in premislite, kakšna vprašanja lahko ta tehnika odgovori. Opravite ta [vadnico](https://docs.microsoft.com/learn/modules/train-evaluate-regression-models?WT.mc_id=academic-77952-leestott), da poglobite svoje razumevanje.

## Naloga

[Drugačen nabor podatkov](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**Omejitev odgovornosti**:
Ta dokument je bil preveden z uporabo AI prevajalske storitve [Co-op Translator](https://github.com/Azure/co-op-translator). Čeprav si prizadevamo za natančnost, vas prosimo, da upoštevate, da avtomatizirani prevodi lahko vsebujejo napake ali netočnosti. Izvirni dokument v njegovem izvirnem jeziku je treba obravnavati kot avtoritativni vir. Za kritične informacije je priporočljiv strokovni človeški prevod. Ne odgovarjamo za morebitna nesporazume ali napačne interpretacije, ki izhajajo iz uporabe tega prevoda.
<!-- CO-OP TRANSLATOR DISCLAIMER END -->