# Pradėkite dirbti su Python ir Scikit-learn regresijos modeliams

![Regresijų santrauka sketchnote](../../../../translated_images/lt/ml-regression.4e4f70e3b3ed446e.webp)

> Sketchnote autorius [Tomomi Imura](https://www.twitter.com/girlie_mac)

## [Priešpaskaitinis testas](https://ff-quizzes.netlify.app/en/ml/)

> ### [Ši pamoka taip pat prieinama R kalba!](../../../../2-Regression/1-Tools/solution/R/lesson_1.html)

## Įvadas

Šiose keturiose pamokose sužinosite, kaip kurti regresijos modelius. Trumpai aptarsime, kam jie skirti. Bet prieš pradėdami, įsitikinkite, kad turite tinkamus įrankius procesui pradėti!

Šioje pamokoje išmoksite:

- Konfigūruoti savo kompiuterį vietiniams mašininio mokymosi darbams.
- Dirbti su Jupyter Notebook.
- Naudoti Scikit-learn, įskaitant diegimą.
- Išnagrinėti linijinę regresiją atliekant praktinę užduotį.

## Diegimai ir konfigūracija

[![Mašininis mokymasis pradedantiesiems – Paruoškite savo įrankius modelių kūrimui](https://img.youtube.com/vi/-DfeD2k2Kj0/0.jpg)](https://youtu.be/-DfeD2k2Kj0 "Mašininis mokymasis pradedantiesiems – Paruoškite savo įrankius modelių kūrimui")

> 🎥 Spustelėkite aukščiau esantį paveikslėlį, kad peržiūrėtumėte trumpą vaizdo įrašą, kaip konfigūruoti savo kompiuterį ML darbams.

1. **Įdiekite Python**. Įsitikinkite, kad jūsų kompiuteryje įdiegtas [Python](https://www.python.org/downloads/). Jūs naudositės Python daugelyje duomenų mokslo ir mašininio mokymosi užduočių. Daugumoje kompiuterių Python jau būna įdiegtas. Kai kurie naudotojai naudosis naudingais [Python Coding Packs](https://code.visualstudio.com/learn/educators/installers?WT.mc_id=academic-77952-leestott), kurie palengvina diegimą.

   Tačiau kai kurie Python panaudojimo atvejai reikalauja vienos versijos, o kiti - kitos. Todėl naudinga dirbti [virtualioje aplinkoje](https://docs.python.org/3/library/venv.html).

2. **Įdiekite Visual Studio Code**. Įsitikinkite, kad jūsų kompiuteryje įdiegtas Visual Studio Code. Vadovaukitės šiais nurodymais, kaip [įdiegti Visual Studio Code](https://code.visualstudio.com/) pagrindiniam diegimui. Šiame kurse naudositės Python Visual Studio Code, tad galite pasidomėti, kaip [konfigūruoti Visual Studio Code](https://docs.microsoft.com/learn/modules/python-install-vscode?WT.mc_id=academic-77952-leestott) Python kūrimui.

   > Susipažinkite su Python dirbdami su šiuo [mokymo modulių rinkiniu](https://docs.microsoft.com/users/jenlooper-2911/collections/mp1pagggd5qrq7?WT.mc_id=academic-77952-leestott)
   >
   > [![Python nustatymas su Visual Studio Code](https://img.youtube.com/vi/yyQM70vi7V8/0.jpg)](https://youtu.be/yyQM70vi7V8 "Python nustatymas su Visual Studio Code")
   >
   > 🎥 Spustelėkite aukščiau esantį paveikslėlį, kad peržiūrėtumėte vaizdo įrašą: Python naudojimas VS Code aplinkoje.

3. **Įdiekite Scikit-learn** pagal [šiuos nurodymus](https://scikit-learn.org/stable/install.html). Kadangi reikia naudoti Python 3, rekomenduojama naudoti virtualią aplinką. Jei įdiegiate šią biblioteką M1 Mac kompiuteryje, yra specialūs nurodymai nuorodoje aukščiau.

1. **Įdiekite Jupyter Notebook**. Reikės [įdiegti Jupyter paketą](https://pypi.org/project/jupyter/).

## Jūsų ML kūrimo aplinka

Naudosite **notebook'us** savo Python kodo kūrimui ir mašininio mokymosi modelių kūrimui. Šio tipo failai yra įprastas įrankis duomenų mokslininkams, juos galima atpažinti pagal plėtinį `.ipynb`.

Notebook'ai yra interaktyvi aplinka, leidžianti programuotojui rašyti kodą, pridėti užrašus ir dokumentaciją aplink kodą, kuris labai naudingas eksperimentiniams ar tyrimų projektams.

[![Mašininis mokymasis pradedantiesiems - Paruoškite Jupyter Notebook'us regresijos modelių kūrimui](https://img.youtube.com/vi/7E-jC8FLA2E/0.jpg)](https://youtu.be/7E-jC8FLA2E "Mašininis mokymasis pradedantiesiems - Paruoškite Jupyter Notebook'us regresijos modelių kūrimui")

> 🎥 Spustelėkite aukščiau esantį paveikslėlį, kad peržiūrėtumėte trumpą vaizdo įrašą, atliekant šią užduotį.

### Užduotis - dirbkite su notebook'u

Šiame aplanke rasite failą _notebook.ipynb_.

1. Atidarykite _notebook.ipynb_ Visual Studio Code programoje.

   Paleidus, startuos Jupyter serveris su Python 3+. Raskite notebook'o dalis, kurias galima `run` vykdyti – kodo blokus. Galite paleisti kodą spustelėdami piktogramą, panašią į paleidimo mygtuką.

1. Pasirinkite `md` piktogramą ir įrašykite truputį Markdown teksto su šiuo tekstu **# Sveiki atvykę į savo notebook'ą**.

   Tada pridėkite keletą Python kodo eilučių.

1. Įveskite **print('hello notebook')** kodo bloke.
1. Paspauskite rodyklę, kad paleistumėte kodą.

   Turėtumėte pamatyti atspausdintą sakinį:

    ```output
    hello notebook
    ```

![VS Code atidarytas su notebook'u](../../../../translated_images/lt/notebook.4a3ee31f396b8832.webp)

Galite savo kodą papildyti komentarais, kad užfiksuotumėte pastabas apie notebook'ą.

✅ Pagalvokite akimirką, kuo skiriasi web kūrėjo darbo aplinka nuo duomenų mokslininko.

## Pradžia su Scikit-learn

Dabar, kai Python įdiegtas jūsų vietinėje aplinkoje ir jau mokate dirbti su Jupyter Notebook, susipažinkime su Scikit-learn (ištariama „sci“ kaip „science“). Scikit-learn suteikia [plataus API](https://scikit-learn.org/stable/modules/classes.html#api-ref), kuris pagelbės atliekant ML užduotis.

Pagal jų [svetainę](https://scikit-learn.org/stable/getting_started.html), „Scikit-learn yra atviro kodo mašininio mokymosi biblioteka, palaikanti prižiūrimą ir neprižiūrimą mokymąsi. Taip pat teikia įvairius įrankius modelių pritaikymui, duomenų paruošimui, modelių pasirinkimui ir vertinimui bei daug kitų naudų.“

Šiame kurse naudosite Scikit-learn ir kitus įrankius, kad kurtumėte mašininio mokymosi modelius, atliekančius vadinamąsias „tradicinio mašininio mokymosi“ užduotis. Sąmoningai vengėme neuroninių tinklų ir gilaus mokymosi, nes apie juos bus daugiau būsimoje „AI pradedantiesiems“ programoje.

Scikit-learn palengvina modelių kūrimą ir jų vertinimą. Jis daugiausia dirba su skaitiniais duomenimis ir turi keletą paruoštų naudojimui duomenų rinkinių kaip mokymo priemones. Taip pat yra iš anksto paruoštų modelių mokiniams išbandyti. Pažiūrėkime, kaip užkrauti paruoštus duomenis ir naudoti įmontuotą įvertintoją, kad sukurtumėte pirmąjį ML modelį su Scikit-learn ir baziniais duomenimis.

## Užduotis - jūsų pirmas Scikit-learn notebook'as

> Šis vadovas buvo įkvėptas [linijinės regresijos pavyzdžio](https://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html#sphx-glr-auto-examples-linear-model-plot-ols-py) iš Scikit-learn svetainės.


[![Mašininis mokymasis pradedantiesiems - Jūsų pirmas linijinės regresijos projektas Python kalba](https://img.youtube.com/vi/2xkXL5EUpS0/0.jpg)](https://youtu.be/2xkXL5EUpS0 "Mašininis mokymasis pradedantiesiems - Jūsų pirmas linijinės regresijos projektas Python kalba")

> 🎥 Spustelėkite paveikslėlį aukščiau norėdami peržiūrėti trumpą vaizdo įrašą, kaip atlikti šią užduotį.

Pakeiskite _notebook.ipynb_ failą, susijusį su šia pamoka, išvalydami visas ląsteles paspausdami šiukšliadėžės piktogramą.

Šiame skyriuje dirbsite su mažomis diabetui skirtomis duomenų rinkinio dalimis, kurios integruotos Scikit-learn mokymosi tikslais. Įsivaizduokite, kad norite ištirti diabetu sergančių pacientų gydymą. Mašininio mokymosi modeliai galėtų padėti nustatyti, kurie pacientai geriau reaguotų į gydymą, remiantis kintamųjų kombinacijomis. Net paprastas regresijos modelis, jei jį pavaizduosite, gali pateikti informaciją apie kintamuosius, padedančius organizuoti teorinius klinikinius tyrimus.

✅ Yra daug regresijos metodų tipų, o kurį pasirinksite, priklauso nuo ieškomo atsakymo. Jei norite prognozuoti tikėtiną žmogaus ūgį pagal amžių, naudotumėte linijinę regresiją, nes ieškote **skaitinės reikšmės**. Jei domitės, ar tam tikro tipo virtuvė turėtų būti laikoma veganiška ar ne, ieškote **kategorijos priskyrimo**, tad naudotumėte loginę regresiją. Apie loginę regresiją sužinosite vėliau. Pagalvokite apie duomenims užduodamus klausimus ir kurį metodą būtų tikslingiau naudoti.

Pradėkime šį darbą.

### Bibliotekų importavimas

Šiai užduočiai importuosime keletą bibliotekų:

- **matplotlib**. Tai naudingas [grafikų kūrimo įrankis](https://matplotlib.org/), naudositės juo kurdami linijinį grafą.
- **numpy**. [numpy](https://numpy.org/doc/stable/user/whatisnumpy.html) yra naudinga biblioteka skaitinių duomenų apdorojimui Python kalboje.
- **sklearn**. Tai yra [Scikit-learn](https://scikit-learn.org/stable/user_guide.html) biblioteka.

Importuokite bibliotekas, kad palengvintumėte savo užduotis.

1. Įrašykite importo kodą:

   ```python
   import matplotlib.pyplot as plt
   import numpy as np
   from sklearn import datasets, linear_model, model_selection
   ```

   Aukščiau importuojate `matplotlib`, `numpy` ir iš `sklearn` importuojate `datasets`, `linear_model` ir `model_selection`. `model_selection` naudojamas duomenims suskirstyti į mokymosi ir testavimo rinkinius.

### Diabeto duomenų rinkinys

Integruotas [diabeto duomenų rinkinys](https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset) turi 442 mėginius diabetui, su 10 charakteristikų, tarp jų:

- amžius: metai
- kūno masės indeksas: body mass index (bmi)
- kraujospūdis: vidutinis kraujo spaudimas
- s1 tc: T-ląstelės (baltųjų kraujo kūnelių tipas)

✅ Šiame rinkinyje 'sex' (lytis) yra svarbus kintamasis, svarstytinas diabetu susijusiuose tyrimuose. Daugelis medicininių duomenų rinkinių turi tokį dvejetainį klasifikatorių. Pagalvokite, kaip tokios kategorizacijos gali išbraukti tam tikras gyventojų grupes iš gydymo galimybių.

Dabar užkraukite X ir y duomenis.

> 🎓 Atminkite, kad tai yra prižiūrimas mokymasis, todėl mums reikia pavadinto 'y' tikslo.

Naujoje kodo ląstelėje užkraukite diabeto duomenų rinkinį funkcija `load_diabetes()`. Parametras `return_X_y=True` reiškia, kad `X` bus duomenų matrica, o `y` – regresijos tikslas.

1. Pridėkite kelis spausdinimo komandas, kad parodytumėte duomenų matricos dydį ir pirmą elementą:

    ```python
    X, y = datasets.load_diabetes(return_X_y=True)
    print(X.shape)
    print(X[0])
    ```

    Gavote grąžinamąją reikšmę iš funkcijos – tuple (kartu) duomenų. Pirmąsias dvi tuple reikšmes priskiriate `X` ir `y`. Daugiau apie tuple skaitykite [čia](https://wikipedia.org/wiki/Tuple).

    Matote, kad duomenys turi 442 elementus, kurių kiekvienas yra 10 elementų masyvas:

    ```text
    (442, 10)
    [ 0.03807591  0.05068012  0.06169621  0.02187235 -0.0442235  -0.03482076
    -0.04340085 -0.00259226  0.01990842 -0.01764613]
    ```

    ✅ Pagalvokite, kaip duomenys siejasi su regresijos tikslu. Linijinė regresija prognozuoja ryšius tarp X požymių ir tikslo kintamojo y. Kur šio diabeto rinkinio [tikslas](https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset)? Ką šis rinkinys demonstruoja su šiuo tikslu?

2. Toliau pasirinkite duomenų rinkinio dalį pavaizdavimui, pasirenkant 3-čią stulpelį. Tai padarysite naudodami operatorių `:` visuose eilutėse, o tada indeksą (2) stulpelyje. Taip pat galite pertvarkyti duomenis į 2D masyvą naudodami `reshape(n_rows, n_columns)`. Jei viena reikšmė yra -1, atitinkama dimensija apskaičiuojama automatiškai.

   ```python
   X = X[:, 2]
   X = X.reshape((-1,1))
   ```

   ✅ Bet kada spausdinkite duomenis, kad patikrintumėte jų formą.

3. Dabar, kai turite duomenis pavaizdavimui, galima patikrinti, ar mašina gali padėti rasti logišką ribą šiame rinkinyje. Norėdami tai padaryti, turite padalyti ir duomenis (X), ir tikslą (y) į testavimo ir mokymosi rinkinius. Scikit-learn tai atlieka paprastai; galite nustatyti testavimo duomenų dalį.

   ```python
   X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.33)
   ```

4. Dabar ruoškitės modelio mokymuisi! Užkraukite linijinės regresijos modelį ir apmokykite jį su savo X ir y mokymosi rinkiniais naudodami `model.fit()`:

    ```python
    model = linear_model.LinearRegression()
    model.fit(X_train, y_train)
    ```

    ✅ `model.fit()` funkciją matysite daugelyje ML bibliotekų, tokių kaip TensorFlow

5. Tuomet sukurkite prognozę naudodami testavimo duomenis, funkcija `predict()`. Tai bus naudojama linijos tarp duomenų grupių nubrėžimui.

    ```python
    y_pred = model.predict(X_test)
    ```

6. Dabar atėjo laikas parodyti duomenis grafike. Matplotlib yra labai naudingas įrankis šiai užduočiai. Sukurkite taškų diagramą (scatterplot) visiems X ir y testavimo duomenims, o prognozę naudokite linijos nubrėžimui tinkamiausioje vietoje tarp modelio duomenų grupių.

    ```python
    plt.scatter(X_test, y_test,  color='black')
    plt.plot(X_test, y_pred, color='blue', linewidth=3)
    plt.xlabel('Scaled BMIs')
    plt.ylabel('Disease Progression')
    plt.title('A Graph Plot Showing Diabetes Progression Against BMI')
    plt.show()
    ```

   ![taškų diagrama, rodanti diabetui skirtus duomenis](../../../../translated_images/lt/scatterplot.ad8b356bcbb33be6.webp)

   ✅ Pagalvokite, kas čia vyksta. Per daug smulkių taškų eina tiesi linija, bet ką ji daro tiksliai? Ar matote, kaip šią liniją galėtumėte naudoti prognozuojant, kur turėtų būti naujas, nematytas duomenų taškas pagal grafiko y ašį? Pabandykite aprašyti šio modelio praktinį pritaikymą.

Sveikiname, sukūrėte savo pirmą linijinės regresijos modelį, sukūrėte prognozę ir pavaizdavote ją grafike!

---
## 🚀Iššūkis

Pavaizduokite kitą kintamąjį iš šio rinkinio. Užuomina: redaguokite šią eilutę: `X = X[:,2]`. Atsižvelgiant į tikslą šiame rinkinyje, ką galite sužinoti apie diabeto ligos progresavimą?
## [Po paskaitos testas](https://ff-quizzes.netlify.app/en/ml/)

## Apžvalga ir savarankiškas mokymasis

Šiame vadove dirbote su paprasta linijine regresija, o ne univartine ar daugialypia linijine regresija. Šiek tiek paskaitykite apie šių metodų skirtumus arba pažiūrėkite [šį vaizdo įrašą](https://www.coursera.org/lecture/quantifying-relationships-regression-models/linear-vs-nonlinear-categorical-variables-ai2Ef)

Sužinokite daugiau apie regresijos sąvoką ir pamąstykite, kokius klausimus galima atsakyti naudojant šią techniką. Pradėkite šį [vadovą](https://docs.microsoft.com/learn/modules/train-evaluate-regression-models?WT.mc_id=academic-77952-leestott), kad gilintumėte savo supratimą.

## Užduotis

[Kita duomenų bazė](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**Atsakomybės apribojimas**:
Šis dokumentas buvo išverstas naudojant dirbtinio intelekto vertimo paslaugą [Co-op Translator](https://github.com/Azure/co-op-translator). Nors siekiame tikslumo, prašome atkreipti dėmesį, kad automatiniai vertimai gali turėti klaidų ar netikslumų. Originalus dokumentas jo gimtąja kalba laikomas autoritetingu šaltiniu. Svarbiai informacijai rekomenduojama naudoti profesionalų žmogiškąjį vertimą. Mes neatsakome už jokius nesusipratimus ar neteisingą interpretaciją, kilusią naudojantis šiuo vertimu.
<!-- CO-OP TRANSLATOR DISCLAIMER END -->