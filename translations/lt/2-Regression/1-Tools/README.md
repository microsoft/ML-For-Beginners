<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "fa81d226c71d5af7a2cade31c1c92b88",
  "translation_date": "2025-09-05T07:46:22+00:00",
  "source_file": "2-Regression/1-Tools/README.md",
  "language_code": "lt"
}
-->
# Pradėkite dirbti su Python ir Scikit-learn regresijos modeliams

![Regresijų santrauka sketchnote](../../../../sketchnotes/ml-regression.png)

> Sketchnote sukūrė [Tomomi Imura](https://www.twitter.com/girlie_mac)

## [Klausimynas prieš paskaitą](https://ff-quizzes.netlify.app/en/ml/)

> ### [Ši pamoka taip pat prieinama R kalba!](../../../../2-Regression/1-Tools/solution/R/lesson_1.html)

## Įvadas

Šiose keturiose pamokose sužinosite, kaip kurti regresijos modelius. Netrukus aptarsime, kam jie skirti. Tačiau prieš pradėdami, įsitikinkite, kad turite tinkamus įrankius, kad galėtumėte pradėti procesą!

Šioje pamokoje išmoksite:

- Konfigūruoti savo kompiuterį vietinėms mašininio mokymosi užduotims.
- Dirbti su Jupyter užrašinėmis.
- Naudoti Scikit-learn, įskaitant diegimą.
- Išbandyti linijinę regresiją praktiniame užsiėmime.

## Diegimai ir konfigūracijos

[![ML pradedantiesiems - Paruoškite įrankius mašininio mokymosi modelių kūrimui](https://img.youtube.com/vi/-DfeD2k2Kj0/0.jpg)](https://youtu.be/-DfeD2k2Kj0 "ML pradedantiesiems - Paruoškite įrankius mašininio mokymosi modelių kūrimui")

> 🎥 Spustelėkite aukščiau esančią nuotrauką, kad peržiūrėtumėte trumpą vaizdo įrašą apie kompiuterio konfigūravimą ML užduotims.

1. **Įdiekite Python**. Įsitikinkite, kad jūsų kompiuteryje įdiegta [Python](https://www.python.org/downloads/). Python bus naudojamas daugeliui duomenų mokslo ir mašininio mokymosi užduočių. Dauguma kompiuterių sistemų jau turi Python diegimą. Taip pat yra naudingų [Python Coding Packs](https://code.visualstudio.com/learn/educators/installers?WT.mc_id=academic-77952-leestott), kurie palengvina nustatymą kai kuriems vartotojams.

   Kai kurios Python naudojimo situacijos reikalauja vienos programinės įrangos versijos, o kitos - kitos. Dėl šios priežasties naudinga dirbti [virtualioje aplinkoje](https://docs.python.org/3/library/venv.html).

2. **Įdiekite Visual Studio Code**. Įsitikinkite, kad jūsų kompiuteryje įdiegta Visual Studio Code. Sekite šias instrukcijas, kad atliktumėte [Visual Studio Code diegimą](https://code.visualstudio.com/). Šiame kurse naudosite Python Visual Studio Code aplinkoje, todėl galbūt norėsite pasipraktikuoti, kaip [konfigūruoti Visual Studio Code](https://docs.microsoft.com/learn/modules/python-install-vscode?WT.mc_id=academic-77952-leestott) Python kūrimui.

   > Susipažinkite su Python, atlikdami šią [mokymosi modulių kolekciją](https://docs.microsoft.com/users/jenlooper-2911/collections/mp1pagggd5qrq7?WT.mc_id=academic-77952-leestott)
   >
   > [![Nustatykite Python su Visual Studio Code](https://img.youtube.com/vi/yyQM70vi7V8/0.jpg)](https://youtu.be/yyQM70vi7V8 "Nustatykite Python su Visual Studio Code")
   >
   > 🎥 Spustelėkite aukščiau esančią nuotrauką, kad peržiūrėtumėte vaizdo įrašą apie Python naudojimą VS Code aplinkoje.

3. **Įdiekite Scikit-learn**, sekdami [šias instrukcijas](https://scikit-learn.org/stable/install.html). Kadangi jums reikia užtikrinti, jog naudojate Python 3, rekomenduojama naudoti virtualią aplinką. Atkreipkite dėmesį, jei diegiate šią biblioteką M1 Mac kompiuteryje, yra specialios instrukcijos aukščiau pateiktame puslapyje.

4. **Įdiekite Jupyter Notebook**. Jums reikės [įdiegti Jupyter paketą](https://pypi.org/project/jupyter/).

## Jūsų ML kūrimo aplinka

Jūs naudosite **užrašines** (notebooks), kad sukurtumėte Python kodą ir kurtumėte mašininio mokymosi modelius. Šio tipo failai yra įprastas įrankis duomenų mokslininkams, ir juos galima atpažinti pagal jų priesagą arba plėtinį `.ipynb`.

Užrašinės yra interaktyvi aplinka, leidžianti kūrėjui tiek rašyti kodą, tiek pridėti pastabas ir dokumentaciją aplink kodą, kas yra labai naudinga eksperimentiniams ar moksliniams projektams.

[![ML pradedantiesiems - Nustatykite Jupyter užrašines regresijos modelių kūrimui](https://img.youtube.com/vi/7E-jC8FLA2E/0.jpg)](https://youtu.be/7E-jC8FLA2E "ML pradedantiesiems - Nustatykite Jupyter užrašines regresijos modelių kūrimui")

> 🎥 Spustelėkite aukščiau esančią nuotrauką, kad peržiūrėtumėte trumpą vaizdo įrašą apie šį pratimą.

### Pratimas - darbas su užrašine

Šiame aplanke rasite failą _notebook.ipynb_.

1. Atidarykite _notebook.ipynb_ Visual Studio Code aplinkoje.

   Jupyter serveris bus paleistas su Python 3+. Užrašinėje rasite vietas, kurias galima `paleisti`, t. y. kodo dalis. Galite paleisti kodo bloką, pasirinkdami piktogramą, kuri atrodo kaip grojimo mygtukas.

2. Pasirinkite `md` piktogramą ir pridėkite šiek tiek markdown teksto, pvz., **# Sveiki atvykę į savo užrašinę**.

   Tada pridėkite šiek tiek Python kodo.

3. Įveskite **print('hello notebook')** kodo bloke.
4. Paspauskite rodyklę, kad paleistumėte kodą.

   Turėtumėte pamatyti atspausdintą sakinį:

    ```output
    hello notebook
    ```

![VS Code su atidaryta užrašine](../../../../2-Regression/1-Tools/images/notebook.jpg)

Galite įterpti savo kodą su komentarais, kad savarankiškai dokumentuotumėte užrašinę.

✅ Pagalvokite minutę, kaip skiriasi žiniatinklio kūrėjo darbo aplinka nuo duomenų mokslininko aplinkos.

## Darbas su Scikit-learn

Dabar, kai Python yra nustatytas jūsų vietinėje aplinkoje ir jūs jaučiatės patogiai dirbdami su Jupyter užrašinėmis, pasistenkime taip pat susipažinti su Scikit-learn (tariama `sci`, kaip `science`). Scikit-learn siūlo [platus API](https://scikit-learn.org/stable/modules/classes.html#api-ref), kuris padeda atlikti ML užduotis.

Pagal jų [svetainę](https://scikit-learn.org/stable/getting_started.html), "Scikit-learn yra atviro kodo mašininio mokymosi biblioteka, palaikanti prižiūrimą ir neprižiūrimą mokymąsi. Ji taip pat siūlo įvairius įrankius modelių pritaikymui, duomenų apdorojimui, modelių pasirinkimui ir vertinimui bei daugybę kitų naudingų funkcijų."

Šiame kurse naudosite Scikit-learn ir kitus įrankius, kad sukurtumėte mašininio mokymosi modelius, skirtus vadinamoms 'tradicinio mašininio mokymosi' užduotims. Mes sąmoningai vengėme neuroninių tinklų ir giluminio mokymosi, nes jie geriau aptariami mūsų būsimoje 'AI pradedantiesiems' mokymo programoje.

Scikit-learn leidžia lengvai kurti modelius ir vertinti jų naudojimą. Ji daugiausia orientuota į skaitinių duomenų naudojimą ir turi keletą paruoštų duomenų rinkinių, skirtų mokymuisi. Ji taip pat apima iš anksto sukurtus modelius, kuriuos studentai gali išbandyti. Pažvelkime į procesą, kaip įkelti iš anksto paruoštus duomenis ir naudoti įmontuotą vertintoją pirmam ML modeliui su Scikit-learn, naudojant pagrindinius duomenis.

## Pratimas - jūsų pirmoji Scikit-learn užrašinė

> Šis mokymas buvo įkvėptas [linijinės regresijos pavyzdžio](https://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html#sphx-glr-auto-examples-linear-model-plot-ols-py) Scikit-learn svetainėje.

[![ML pradedantiesiems - Jūsų pirmasis linijinės regresijos projektas Python kalba](https://img.youtube.com/vi/2xkXL5EUpS0/0.jpg)](https://youtu.be/2xkXL5EUpS0 "ML pradedantiesiems - Jūsų pirmasis linijinės regresijos projektas Python kalba")

> 🎥 Spustelėkite aukščiau esančią nuotrauką, kad peržiūrėtumėte trumpą vaizdo įrašą apie šį pratimą.

Failo _notebook.ipynb_ susijusio su šia pamoka, ištrinkite visas ląsteles paspausdami 'šiukšliadėžės' piktogramą.

Šiame skyriuje dirbsite su nedideliu duomenų rinkiniu apie diabetą, kuris yra įtrauktas į Scikit-learn mokymosi tikslais. Įsivaizduokite, kad norite išbandyti gydymą diabetu sergantiems pacientams. Mašininio mokymosi modeliai gali padėti nustatyti, kurie pacientai geriau reaguotų į gydymą, remiantis kintamųjų deriniais. Net labai paprastas regresijos modelis, vizualizuotas, gali parodyti informaciją apie kintamuosius, kurie padėtų organizuoti teorinius klinikinius tyrimus.

✅ Yra daug regresijos metodų tipų, ir kurį pasirinkti priklauso nuo klausimo, į kurį norite atsakyti. Jei norite prognozuoti tikėtiną žmogaus ūgį pagal jo amžių, naudotumėte linijinę regresiją, nes ieškote **skaitinės vertės**. Jei jus domina, ar tam tikra virtuvė turėtų būti laikoma veganiška, ieškote **kategorijos priskyrimo**, todėl naudotumėte logistinę regresiją. Vėliau sužinosite daugiau apie logistinę regresiją. Pagalvokite apie keletą klausimų, kuriuos galite užduoti duomenims, ir kuris iš šių metodų būtų tinkamesnis.

Pradėkime šią užduotį.

### Bibliotekų importavimas

Šiai užduočiai importuosime keletą bibliotekų:

- **matplotlib**. Tai naudinga [grafikų kūrimo priemonė](https://matplotlib.org/), kurią naudosime linijiniam grafikui kurti.
- **numpy**. [numpy](https://numpy.org/doc/stable/user/whatisnumpy.html) yra naudinga biblioteka skaitinių duomenų tvarkymui Python kalboje.
- **sklearn**. Tai [Scikit-learn](https://scikit-learn.org/stable/user_guide.html) biblioteka.

Importuokite keletą bibliotekų, kurios padės atlikti užduotis.

1. Pridėkite importus, įvesdami šį kodą:

   ```python
   import matplotlib.pyplot as plt
   import numpy as np
   from sklearn import datasets, linear_model, model_selection
   ```

   Aukščiau importuojate `matplotlib`, `numpy` ir importuojate `datasets`, `linear_model` bei `model_selection` iš `sklearn`. `model_selection` naudojamas duomenų skirstymui į mokymo ir testavimo rinkinius.

### Diabeto duomenų rinkinys

Įmontuotas [diabeto duomenų rinkinys](https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset) apima 442 duomenų pavyzdžius apie diabetą, su 10 kintamųjų, kai kurie iš jų yra:

- amžius: amžius metais
- kūno masės indeksas (BMI)
- vidutinis kraujo spaudimas
- s1 tc: T-ląstelės (tam tikros rūšies baltieji kraujo kūneliai)

✅ Šis duomenų rinkinys apima 'lyties' sąvoką kaip svarbų kintamąjį diabetui tirti. Daugelis medicininių duomenų rinkinių apima tokio tipo dvejetainę klasifikaciją. Pagalvokite, kaip tokios kategorijos gali išskirti tam tikras populiacijos dalis iš gydymo.

Dabar įkelkite X ir y duomenis.

> 🎓 Prisiminkite, tai yra prižiūrimas mokymasis, ir mums reikia pavadinto 'y' tikslo.

Naujoje kodo ląstelėje įkelkite diabeto duomenų rinkinį, naudodami `load_diabetes()`. Įvestis `return_X_y=True` nurodo, kad `X` bus duomenų matrica, o `y` bus regresijos tikslas.

1. Pridėkite keletą spausdinimo komandų, kad parodytumėte duomenų matricos formą ir jos pirmąjį elementą:

    ```python
    X, y = datasets.load_diabetes(return_X_y=True)
    print(X.shape)
    print(X[0])
    ```

    Tai, ką gaunate kaip atsakymą, yra tuple. Jūs priskiriate pirmąsias dvi tuple reikšmes `X` ir `y` atitinkamai. Sužinokite daugiau [apie tuple](https://wikipedia.org/wiki/Tuple).

    Galite matyti, kad šie duomenys turi 442 elementus, suformuotus į 10 elementų masyvus:

    ```text
    (442, 10)
    [ 0.03807591  0.05068012  0.06169621  0.02187235 -0.0442235  -0.03482076
    -0.04340085 -0.00259226  0.01990842 -0.01764613]
    ```

    ✅ Pagalvokite apie ryšį tarp duomenų ir regresijos tikslo. Linijinė regresija prognozuoja ryšius tarp kintamojo X ir tikslo kintamojo y. Ar galite rasti [tikslą](https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset) diabeto duomenų rinkinyje dokumentacijoje? Ką šis duomenų rinkinys demonstruoja, atsižvelgiant į tikslą?

2. Tada pasirinkite dalį šio duomenų rinkinio, kurią norite pavaizduoti, pasirinkdami 3-ąją duomenų rinkinio stulpelį. Tai galite padaryti naudodami `:` operatorių, kad pasirinktumėte visas eilutes, ir tada pasirinkdami 3-ąjį stulpelį naudodami indeksą (2). Taip pat galite pertvarkyti duomenis į 2D masyvą - kaip reikalaujama grafiko kūrimui - naudodami `reshape(n_rows, n_columns)`. Jei vienas iš parametrų yra -1, atitinkama dimensija apskaičiuojama automatiškai.

   ```python
   X = X[:, 2]
   X = X.reshape((-1,1))
   ```

   ✅ Bet kuriuo metu spausdinkite duomenis, kad patikrintumėte jų formą.

3. Dabar, kai turite duomenis, paruoštus grafiko kūrimui, galite patikrinti, ar mašina gali padėti nustatyti logišką skirstymą tarp skaičių šiame duomenų rinkinyje. Norėdami tai padaryti, turite padalyti tiek duomenis (X), tiek tikslą (y) į testavimo ir mokymo rinkinius. Scikit-learn turi paprastą būdą tai padaryti; galite padalyti savo testavimo duomenis tam tikru tašku.

   ```python
   X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.33)
   ```

4. Dabar esate pasiruošę treniruoti savo modelį! Įkelkite linijinės regresijos modelį ir treniruokite jį su savo X ir y mokymo rinkiniais, naudodami `model.fit()`:

    ```python
    model = linear_model.LinearRegression()
    model.fit(X_train, y_train)
    ```

    ✅ `model.fit()` yra funkcija, kurią pamatysite daugelyje ML bibliotekų, tokių kaip TensorFlow.

5. Tada sukurkite prognozę, naudodami testavimo duomenis, naudodami funkciją `predict()`. Tai bus naudojama linijai nubrėžti tarp duomenų grupių.

    ```python
    y_pred = model.predict(X_test)
    ```

6. Dabar laikas parodyti duomenis grafike. Matplotlib yra labai naudinga priemonė šiai užduočiai. Sukurkite sklaidos grafiką visiems X ir y testavimo duomenims, o prognozę naudokite linijai nubrėžti tinkamiausioje vietoje tarp modelio duomenų grupių.


✅ Pagalvokite, kas čia vyksta. Tiesi linija eina per daugybę mažų duomenų taškų, bet ką ji iš tikrųjų daro? Ar galite pastebėti, kaip ši linija gali padėti numatyti, kur naujas, dar nematytas duomenų taškas turėtų atsidurti santykyje su grafiko y ašimi? Pabandykite žodžiais apibūdinti praktinį šio modelio pritaikymą.

Sveikiname, jūs sukūrėte savo pirmąjį linijinį regresijos modelį, atlikote prognozę su juo ir pavaizdavote ją grafike!

---
## 🚀Iššūkis

Pavaizduokite kitą šio duomenų rinkinio kintamąjį. Užuomina: redaguokite šią eilutę: `X = X[:,2]`. Atsižvelgiant į šio duomenų rinkinio tikslą, ką galite sužinoti apie diabeto progresavimą kaip ligą?
## [Po paskaitos testas](https://ff-quizzes.netlify.app/en/ml/)

## Apžvalga ir savarankiškas mokymasis

Šioje pamokoje dirbote su paprasta linijine regresija, o ne su univariante ar daugiavariante regresija. Pasidomėkite skirtumais tarp šių metodų arba peržiūrėkite [šį vaizdo įrašą](https://www.coursera.org/lecture/quantifying-relationships-regression-models/linear-vs-nonlinear-categorical-variables-ai2Ef).

Plačiau pasiskaitykite apie regresijos sąvoką ir pagalvokite, kokius klausimus galima atsakyti naudojant šią techniką. Norėdami giliau suprasti, peržiūrėkite šį [vadovą](https://docs.microsoft.com/learn/modules/train-evaluate-regression-models?WT.mc_id=academic-77952-leestott).

## Užduotis

[Kitas duomenų rinkinys](assignment.md)

---

**Atsakomybės apribojimas**:  
Šis dokumentas buvo išverstas naudojant AI vertimo paslaugą [Co-op Translator](https://github.com/Azure/co-op-translator). Nors siekiame tikslumo, prašome atkreipti dėmesį, kad automatiniai vertimai gali turėti klaidų ar netikslumų. Originalus dokumentas jo gimtąja kalba turėtų būti laikomas autoritetingu šaltiniu. Kritinei informacijai rekomenduojama profesionali žmogaus vertimo paslauga. Mes neprisiimame atsakomybės už nesusipratimus ar klaidingus interpretavimus, atsiradusius dėl šio vertimo naudojimo.