<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "fa81d226c71d5af7a2cade31c1c92b88",
  "translation_date": "2025-09-05T11:41:21+00:00",
  "source_file": "2-Regression/1-Tools/README.md",
  "language_code": "sl"
}
-->
# Začnite s Pythonom in Scikit-learn za regresijske modele

![Povzetek regresij v sketchnote](../../../../sketchnotes/ml-regression.png)

> Sketchnote avtorja [Tomomi Imura](https://www.twitter.com/girlie_mac)

## [Predavanje kviz](https://ff-quizzes.netlify.app/en/ml/)

> ### [To lekcijo lahko najdete tudi v jeziku R!](../../../../2-Regression/1-Tools/solution/R/lesson_1.html)

## Uvod

V teh štirih lekcijah boste odkrili, kako zgraditi regresijske modele. Kmalu bomo razpravljali, za kaj so ti modeli uporabni. Preden pa začnete, poskrbite, da imate na voljo ustrezna orodja za začetek procesa!

V tej lekciji boste izvedeli, kako:

- Pripraviti računalnik za lokalne naloge strojnega učenja.
- Delati z Jupyter zvezki.
- Uporabljati Scikit-learn, vključno z namestitvijo.
- Raziskati linearno regresijo s praktično vajo.

## Namestitve in konfiguracije

[![ML za začetnike - Pripravite orodja za gradnjo modelov strojnega učenja](https://img.youtube.com/vi/-DfeD2k2Kj0/0.jpg)](https://youtu.be/-DfeD2k2Kj0 "ML za začetnike - Pripravite orodja za gradnjo modelov strojnega učenja")

> 🎥 Kliknite zgornjo sliko za kratek video o konfiguraciji računalnika za strojno učenje.

1. **Namestite Python**. Prepričajte se, da je [Python](https://www.python.org/downloads/) nameščen na vašem računalniku. Python boste uporabljali za številne naloge podatkovne znanosti in strojnega učenja. Večina računalniških sistemov že vključuje namestitev Pythona. Na voljo so tudi uporabni [Python Coding Packs](https://code.visualstudio.com/learn/educators/installers?WT.mc_id=academic-77952-leestott), ki olajšajo nastavitev za nekatere uporabnike.

   Nekatere uporabe Pythona pa zahtevajo eno različico programske opreme, druge pa drugo. Zato je koristno delati v [virtualnem okolju](https://docs.python.org/3/library/venv.html).

2. **Namestite Visual Studio Code**. Prepričajte se, da imate na računalniku nameščen Visual Studio Code. Sledite tem navodilom za [namestitev Visual Studio Code](https://code.visualstudio.com/) za osnovno namestitev. Python boste uporabljali v Visual Studio Code v tem tečaju, zato se morda želite seznaniti z [nastavitvijo Visual Studio Code](https://docs.microsoft.com/learn/modules/python-install-vscode?WT.mc_id=academic-77952-leestott) za razvoj v Pythonu.

   > Seznanite se s Pythonom z delom skozi to zbirko [učnih modulov](https://docs.microsoft.com/users/jenlooper-2911/collections/mp1pagggd5qrq7?WT.mc_id=academic-77952-leestott)
   >
   > [![Nastavitev Pythona z Visual Studio Code](https://img.youtube.com/vi/yyQM70vi7V8/0.jpg)](https://youtu.be/yyQM70vi7V8 "Nastavitev Pythona z Visual Studio Code")
   >
   > 🎥 Kliknite zgornjo sliko za video: uporaba Pythona v VS Code.

3. **Namestite Scikit-learn**, tako da sledite [tem navodilom](https://scikit-learn.org/stable/install.html). Ker morate zagotoviti uporabo Pythona 3, je priporočljivo, da uporabite virtualno okolje. Če to knjižnico nameščate na računalnik Mac z M1, so na zgoraj navedeni strani posebna navodila.

4. **Namestite Jupyter Notebook**. Potrebovali boste [namestitev paketa Jupyter](https://pypi.org/project/jupyter/).

## Vaše okolje za avtorstvo ML

Uporabljali boste **zvezke** za razvoj kode v Pythonu in ustvarjanje modelov strojnega učenja. Ta vrsta datoteke je pogosto orodje za podatkovne znanstvenike, prepoznate pa jih po priponi `.ipynb`.

Zvezki so interaktivno okolje, ki razvijalcu omogoča tako kodiranje kot dodajanje opomb in pisanje dokumentacije okoli kode, kar je zelo uporabno za eksperimentalne ali raziskovalno usmerjene projekte.

[![ML za začetnike - Nastavitev Jupyter zvezkov za začetek gradnje regresijskih modelov](https://img.youtube.com/vi/7E-jC8FLA2E/0.jpg)](https://youtu.be/7E-jC8FLA2E "ML za začetnike - Nastavitev Jupyter zvezkov za začetek gradnje regresijskih modelov")

> 🎥 Kliknite zgornjo sliko za kratek video o tej vaji.

### Vaja - delo z zvezkom

V tej mapi boste našli datoteko _notebook.ipynb_.

1. Odprite _notebook.ipynb_ v Visual Studio Code.

   Začel se bo Jupyter strežnik s Pythonom 3+. V zvezku boste našli območja, ki jih je mogoče `zagnati`, torej dele kode. Kodo lahko zaženete tako, da izberete ikono, ki izgleda kot gumb za predvajanje.

2. Izberite ikono `md` in dodajte nekaj markdowna ter naslednje besedilo **# Dobrodošli v vašem zvezku**.

   Nato dodajte nekaj kode v Pythonu.

3. Vnesite **print('hello notebook')** v blok kode.
4. Izberite puščico za zagon kode.

   Videti bi morali natisnjeno izjavo:

    ```output
    hello notebook
    ```

![VS Code z odprtim zvezkom](../../../../2-Regression/1-Tools/images/notebook.jpg)

Kodo lahko prepletate z opombami, da sami dokumentirate zvezek.

✅ Razmislite za trenutek, kako se delovno okolje spletnega razvijalca razlikuje od okolja podatkovnega znanstvenika.

## Začetek dela s Scikit-learn

Zdaj, ko je Python nastavljen v vašem lokalnem okolju in ste seznanjeni z Jupyter zvezki, se enako seznanimo s Scikit-learn (izgovorjava `sci` kot v `science`). Scikit-learn ponuja [obsežen API](https://scikit-learn.org/stable/modules/classes.html#api-ref) za izvajanje nalog strojnega učenja.

Po navedbah njihove [spletne strani](https://scikit-learn.org/stable/getting_started.html) je "Scikit-learn odprtokodna knjižnica strojnega učenja, ki podpira nadzorovano in nenadzorovano učenje. Prav tako ponuja različna orodja za prileganje modelov, predobdelavo podatkov, izbiro modelov in ocenjevanje ter številne druge pripomočke."

V tem tečaju boste uporabljali Scikit-learn in druga orodja za gradnjo modelov strojnega učenja za izvajanje nalog, ki jih imenujemo 'tradicionalno strojno učenje'. Namenoma smo se izognili nevronskim mrežam in globokemu učenju, saj so bolje obravnavani v našem prihajajočem učnem načrtu 'AI za začetnike'.

Scikit-learn omogoča enostavno gradnjo modelov in njihovo ocenjevanje za uporabo. Osredotoča se predvsem na uporabo numeričnih podatkov in vsebuje več pripravljenih naborov podatkov za uporabo kot učna orodja. Prav tako vključuje vnaprej pripravljene modele, ki jih lahko študentje preizkusijo. Raziskujmo proces nalaganja vnaprej pripravljenih podatkov in uporabe vgrajenega ocenjevalnika za prvi ML model s Scikit-learn z osnovnimi podatki.

## Vaja - vaš prvi zvezek s Scikit-learn

> Ta vadnica je bila navdihnjena z [primerom linearne regresije](https://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html#sphx-glr-auto-examples-linear-model-plot-ols-py) na spletni strani Scikit-learn.


[![ML za začetnike - Vaš prvi projekt linearne regresije v Pythonu](https://img.youtube.com/vi/2xkXL5EUpS0/0.jpg)](https://youtu.be/2xkXL5EUpS0 "ML za začetnike - Vaš prvi projekt linearne regresije v Pythonu")

> 🎥 Kliknite zgornjo sliko za kratek video o tej vaji.

V datoteki _notebook.ipynb_, povezani s to lekcijo, izbrišite vse celice s pritiskom na ikono 'koš za smeti'.

V tem razdelku boste delali z majhnim naborom podatkov o sladkorni bolezni, ki je vgrajen v Scikit-learn za učne namene. Predstavljajte si, da želite preizkusiti zdravljenje za bolnike s sladkorno boleznijo. Modeli strojnega učenja vam lahko pomagajo določiti, kateri bolniki bi se bolje odzvali na zdravljenje, na podlagi kombinacij spremenljivk. Tudi zelo osnovni regresijski model, ko je vizualiziran, lahko pokaže informacije o spremenljivkah, ki bi vam pomagale organizirati teoretične klinične preizkuse.

✅ Obstaja veliko vrst regresijskih metod, izbira pa je odvisna od vprašanja, na katerega želite odgovoriti. Če želite napovedati verjetno višino osebe določene starosti, bi uporabili linearno regresijo, saj iščete **numerično vrednost**. Če vas zanima, ali naj se določena vrsta kuhinje šteje za vegansko ali ne, iščete **kategorijsko dodelitev**, zato bi uporabili logistično regresijo. Kasneje boste izvedeli več o logistični regresiji. Razmislite o nekaterih vprašanjih, ki jih lahko postavite podatkom, in kateri od teh metod bi bila bolj primerna.

Začnimo s to nalogo.

### Uvoz knjižnic

Za to nalogo bomo uvozili nekaj knjižnic:

- **matplotlib**. To je uporabno [orodje za risanje grafov](https://matplotlib.org/), ki ga bomo uporabili za ustvarjanje linijskega grafa.
- **numpy**. [numpy](https://numpy.org/doc/stable/user/whatisnumpy.html) je uporabna knjižnica za obdelavo numeričnih podatkov v Pythonu.
- **sklearn**. To je [knjižnica Scikit-learn](https://scikit-learn.org/stable/user_guide.html).

Uvozite nekaj knjižnic za pomoč pri nalogah.

1. Dodajte uvoze z vnosom naslednje kode:

   ```python
   import matplotlib.pyplot as plt
   import numpy as np
   from sklearn import datasets, linear_model, model_selection
   ```

   Zgornja koda uvaža `matplotlib`, `numpy` in `datasets`, `linear_model` ter `model_selection` iz `sklearn`. `model_selection` se uporablja za razdelitev podatkov na učne in testne sklope.

### Nabor podatkov o sladkorni bolezni

Vgrajeni [nabor podatkov o sladkorni bolezni](https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset) vključuje 442 vzorcev podatkov o sladkorni bolezni z 10 značilnimi spremenljivkami, med katerimi so nekatere:

- starost: starost v letih
- bmi: indeks telesne mase
- bp: povprečni krvni tlak
- s1 tc: T-celice (vrsta belih krvnih celic)

✅ Ta nabor podatkov vključuje koncept 'spola' kot značilne spremenljivke, pomembne za raziskave o sladkorni bolezni. Številni medicinski nabori podatkov vključujejo tovrstno binarno klasifikacijo. Razmislite, kako bi lahko takšne kategorizacije izključile določene dele populacije iz zdravljenja.

Zdaj naložite podatke X in y.

> 🎓 Ne pozabite, da gre za nadzorovano učenje, zato potrebujemo imenovan cilj 'y'.

V novi celici kode naložite nabor podatkov o sladkorni bolezni z uporabo `load_diabetes()`. Vhod `return_X_y=True` signalizira, da bo `X` matrika podatkov, `y` pa regresijski cilj.

1. Dodajte nekaj ukazov za izpis, da prikažete obliko matrike podatkov in njen prvi element:

    ```python
    X, y = datasets.load_diabetes(return_X_y=True)
    print(X.shape)
    print(X[0])
    ```

    Kar dobite kot odgovor, je nabor. Prva dva vrednosti nabora dodelite `X` in `y`. Več o [naborih](https://wikipedia.org/wiki/Tuple) lahko izveste tukaj.

    Vidite lahko, da ti podatki vsebujejo 442 elementov, oblikovanih v poljih z 10 elementi:

    ```text
    (442, 10)
    [ 0.03807591  0.05068012  0.06169621  0.02187235 -0.0442235  -0.03482076
    -0.04340085 -0.00259226  0.01990842 -0.01764613]
    ```

    ✅ Razmislite o razmerju med podatki in regresijskim ciljem. Linearna regresija napoveduje razmerja med značilnostjo X in ciljno spremenljivko y. Ali lahko v dokumentaciji najdete [cilj](https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset) za nabor podatkov o sladkorni bolezni? Kaj ta nabor podatkov prikazuje, glede na cilj?

2. Nato izberite del tega nabora podatkov za risanje tako, da izberete 3. stolpec nabora podatkov. To lahko storite z uporabo operatorja `:` za izbiro vseh vrstic, nato pa izberete 3. stolpec z uporabo indeksa (2). Podatke lahko preoblikujete v 2D matriko - kot je potrebno za risanje - z uporabo `reshape(n_rows, n_columns)`. Če je eden od parametrov -1, se ustrezna dimenzija izračuna samodejno.

   ```python
   X = X[:, 2]
   X = X.reshape((-1,1))
   ```

   ✅ Kadarkoli natisnite podatke, da preverite njihovo obliko.

3. Zdaj, ko imate podatke pripravljene za risanje, lahko preverite, ali lahko stroj pomaga določiti logično ločnico med številkami v tem naboru podatkov. Za to morate razdeliti tako podatke (X) kot cilj (y) na testne in učne sklope. Scikit-learn ima preprost način za to; testne podatke lahko razdelite na določenem mestu.

   ```python
   X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.33)
   ```

4. Zdaj ste pripravljeni na učenje modela! Naložite model linearne regresije in ga naučite z učnimi sklopi X in y z uporabo `model.fit()`:

    ```python
    model = linear_model.LinearRegression()
    model.fit(X_train, y_train)
    ```

    ✅ `model.fit()` je funkcija, ki jo boste videli v številnih knjižnicah ML, kot je TensorFlow.

5. Nato ustvarite napoved z uporabo testnih podatkov z uporabo funkcije `predict()`. To bo uporabljeno za risanje črte med skupinami podatkov.

    ```python
    y_pred = model.predict(X_test)
    ```

6. Zdaj je čas, da prikažete podatke na grafu. Matplotlib je zelo uporabno orodje za to nalogo. Ustvarite razpršeni grafikon vseh testnih podatkov X in y ter uporabite napoved za risanje črte na najbolj primernem mestu med skupinami podatkov modela.

    ```python
    plt.scatter(X_test, y_test,  color='black')
    plt.plot(X_test, y_pred, color='blue', linewidth=3)
    plt.xlabel('Scaled BMIs')
    plt.ylabel('Disease Progression')
    plt.title('A Graph Plot Showing Diabetes Progression Against BMI')
    plt.show()
    ```

   ![razpršeni grafikon, ki prikazuje podatkovne točke o sladkorni bolezni](../../../../2-Regression/1-Tools/images/scatterplot.png)
✅ Razmislite malo o tem, kaj se tukaj dogaja. Ravna črta poteka skozi številne majhne točke podatkov, vendar kaj pravzaprav počne? Ali vidite, kako bi morali biti sposobni uporabiti to črto za napovedovanje, kje bi se nova, nevidna podatkovna točka morala uvrstiti glede na y-os grafa? Poskusite z besedami opisati praktično uporabo tega modela.

Čestitke, izdelali ste svoj prvi model linearne regresije, ustvarili napoved z njim in jo prikazali na grafu!

---
## 🚀Izziv

Prikažite drugo spremenljivko iz tega nabora podatkov. Namig: uredite to vrstico: `X = X[:,2]`. Glede na cilj tega nabora podatkov, kaj lahko odkrijete o napredovanju sladkorne bolezni kot bolezni?
## [Kvizi po predavanju](https://ff-quizzes.netlify.app/en/ml/)

## Pregled & Samostojno učenje

V tem vodiču ste delali z enostavno linearno regresijo, ne pa z univariatno ali večkratno linearno regresijo. Preberite nekaj o razlikah med temi metodami ali si oglejte [ta video](https://www.coursera.org/lecture/quantifying-relationships-regression-models/linear-vs-nonlinear-categorical-variables-ai2Ef).

Preberite več o konceptu regresije in razmislite, kakšna vprašanja je mogoče odgovoriti s to tehniko. Vzemite [ta vodič](https://docs.microsoft.com/learn/modules/train-evaluate-regression-models?WT.mc_id=academic-77952-leestott), da poglobite svoje razumevanje.

## Naloga

[Drug nabor podatkov](assignment.md)

---

**Omejitev odgovornosti**:  
Ta dokument je bil preveden z uporabo storitve za strojno prevajanje [Co-op Translator](https://github.com/Azure/co-op-translator). Čeprav si prizadevamo za natančnost, vas prosimo, da se zavedate, da lahko avtomatizirani prevodi vsebujejo napake ali netočnosti. Izvirni dokument v njegovem izvirnem jeziku je treba obravnavati kot avtoritativni vir. Za ključne informacije priporočamo strokovno človeško prevajanje. Ne prevzemamo odgovornosti za morebitna nesporazumevanja ali napačne razlage, ki izhajajo iz uporabe tega prevoda.