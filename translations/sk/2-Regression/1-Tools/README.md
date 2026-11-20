# Začnite s Pythonom a Scikit-learn pre regresné modely

![Zhrnutie regresií v sketchnote](../../../../translated_images/sk/ml-regression.4e4f70e3b3ed446e.webp)

> Sketchnote od [Tomomi Imura](https://www.twitter.com/girlie_mac)

## [Kvíz pred prednáškou](https://ff-quizzes.netlify.app/en/ml/)

> ### [Táto lekcia je dostupná v R!](../../../../2-Regression/1-Tools/solution/R/lesson_1.html)

## Úvod

V týchto štyroch lekciách objavíte, ako vytvárať regresné modely. Čoskoro si povieme, na čo tieto slúžia. Ale predtým, než urobíte čokoľvek, uistite sa, že máte pripravené správne nástroje na začatie procesu!

V tejto lekcii sa naučíte, ako:

- Nakonfigurovať svoj počítač pre úlohy strojového učenia lokálne.
- Pracovať s Jupyter Notebookmi.
- Použiť Scikit-learn vrátane inštalácie.
- Preskúmať lineárnu regresiu prostredníctvom praktického cvičenia.

## Inštalácie a konfigurácie

[![ML pre začiatočníkov - Nastavte si nástroje pripravené na tvorbu modelov strojového učenia](https://img.youtube.com/vi/-DfeD2k2Kj0/0.jpg)](https://youtu.be/-DfeD2k2Kj0 "ML pre začiatočníkov - Nastavte si nástroje pripravené na tvorbu modelov strojového učenia")

> 🎥 Kliknite na obrázok vyššie pre krátke video, ktoré vás prevedie konfiguráciou vášho počítača pre ML.

1. **Nainštalujte Python**. Uistite sa, že máte na počítači nainštalovaný [Python](https://www.python.org/downloads/). Python budete používať pri mnohých úlohách dátovej vedy a strojového učenia. Väčšina počítačových systémov už obsahuje inštaláciu Pythonu. Existujú tiež užitočné [balíčky na kódovanie v Pythone](https://code.visualstudio.com/learn/educators/installers?WT.mc_id=academic-77952-leestott), ktoré niektorým používateľom uľahčujú nastavenie.

   Niektoré použitia Pythonu však vyžadujú jednu verziu softvéru, zatiaľ čo iné inú verziu. Preto je užitočné pracovať v [virtuálnom prostredí](https://docs.python.org/3/library/venv.html).

2. **Nainštalujte Visual Studio Code**. Uistite sa, že máte na počítači nainštalovaný Visual Studio Code. Postupujte podľa týchto pokynov, ako [nainštalovať Visual Studio Code](https://code.visualstudio.com/) pre základnú inštaláciu. V tomto kurze budete používať Python vo Visual Studio Code, preto si možno budete chcieť osviežiť, ako [nastaviť Visual Studio Code](https://docs.microsoft.com/learn/modules/python-install-vscode?WT.mc_id=academic-77952-leestott) pre vývoj v Pythone.

   > Získajte istotu s Pythonom tak, že si prejdete túto zbierku [Learn modulov](https://docs.microsoft.com/users/jenlooper-2911/collections/mp1pagggd5qrq7?WT.mc_id=academic-77952-leestott)
   >
   > [![Nastavte Python s Visual Studio Code](https://img.youtube.com/vi/yyQM70vi7V8/0.jpg)](https://youtu.be/yyQM70vi7V8 "Nastavte Python s Visual Studio Code")
   >
   > 🎥 Kliknite na obrázok vyššie pre video: používanie Pythonu v prostredí VS Code.

3. **Nainštalujte Scikit-learn** podľa [týchto pokynov](https://scikit-learn.org/stable/install.html). Keďže musíte používať Python 3, odporúča sa použiť virtuálne prostredie. Ak inštalujete túto knižnicu na Mac M1, na stránke vyššie sú uvedené špeciálne pokyny.

1. **Nainštalujte Jupyter Notebook**. Budete potrebovať [nainštalovať balíček Jupyter](https://pypi.org/project/jupyter/).

## Vaše prostredie na tvorbu ML

Budete používať **notebooky** na vývoj vášho Python kódu a vytváranie modelov strojového učenia. Tento typ súboru je bežným nástrojom dátových vedcov a môže byť rozpoznaný podľa prípony `.ipynb`.

Notebooky sú interaktívne prostredie, ktoré umožňuje developerovi nielen kódovať, ale aj pridávať poznámky a dokumentáciu okolo kódu, čo je veľmi užitočné pre experimentálne alebo výskumnícke projekty.

[![ML pre začiatočníkov - Nastavte Jupyter Notebooky na začatie tvorby regresných modelov](https://img.youtube.com/vi/7E-jC8FLA2E/0.jpg)](https://youtu.be/7E-jC8FLA2E "ML pre začiatočníkov - Nastavte Jupyter Notebooky na začatie tvorby regresných modelov")

> 🎥 Kliknite na obrázok vyššie pre krátke video, ktoré vás prevedie týmto cvičením.

### Cvičenie - práca s notebookom

V tomto priečinku nájdete súbor _notebook.ipynb_.

1. Otvorte _notebook.ipynb_ vo Visual Studio Code.

   Spustí sa Jupyter server s Python 3+. V notebooku nájdete oblasti, ktoré môžete `spustiť`, teda bloky kódu. Blok kódu spustíte kliknutím na ikonu pripomínajúcu tlačidlo pre prehrávanie.

1. Vyberte ikonu `md` a pridajte trochu markdownu a nasledovný text **# Welcome to your notebook**.

   Potom pridajte nejaký Python kód.

1. Napíšte **print('hello notebook')** do bloku kódu.
1. Kliknite na šípku pre spustenie kódu.

   Mali by ste vidieť vytlačený výstup:

    ```output
    hello notebook
    ```

![VS Code s otvoreným notebookom](../../../../translated_images/sk/notebook.4a3ee31f396b8832.webp)

Môžete prelínať kód s komentármi na vlastnú dokumentáciu notebooku.

✅ Zamyslite sa na chvíľu, aké odlišné je pracovné prostredie web developera oproti dátovému vedcovi.

## Spustenie so Scikit-learn

Keď máte Python nastavený vo vašom lokálnom prostredí a ste oboznámení s Jupyter Notebookmi, zoznámime sa podrobnejšie so Scikit-learn (vyslovuje sa `sci` ako v slove `science`). Scikit-learn poskytuje [rozsiahle API](https://scikit-learn.org/stable/modules/classes.html#api-ref) na pomoc pri vykonávaní úloh ML.

Podľa ich [webovej stránky](https://scikit-learn.org/stable/getting_started.html), "Scikit-learn je open-source knižnica strojového učenia, ktorá podporuje dohliadané (supervised) i nedohliadané (unsupervised) učenie. Tiež poskytuje rôzne nástroje na prispôsobenie modelu, predspracovanie dát, výber a hodnotenie modelu a mnoho ďalších utilít."

V tomto kurze použijete Scikit-learn a ďalšie nástroje na tvorbu modelov strojového učenia na vykonávanie toho, čo nazývame „tradičné úlohy strojového učenia“. Vedome sme sa vyhli neurónovým sieťam a hlbokému učeniu, pretože tie sú lepšie pokryté v našom pripravovanom kurze 'AI pre začiatočníkov'.

Scikit-learn umožňuje jednoduché vytváranie a hodnotenie modelov na použitie. Je zameraný hlavne na použitie numerických dát a obsahuje niekoľko pripravených datasetov ako výučbové nástroje. Tiež zahŕňa predpripravené modely pre študentov na vyskúšanie. Preskúmajme proces načítania predpripravených dát a použitia vstavaného odhadovača na vytvorenie prvého ML modelu pomocou Scikit-learn s jednoduchými dátami.

## Cvičenie - váš prvý notebook so Scikit-learn

> Tento tutoriál bol inšpirovaný [príkladom lineárnej regresie](https://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html#sphx-glr-auto-examples-linear-model-plot-ols-py) na webovej stránke Scikit-learn.


[![ML pre začiatočníkov - Váš prvý projekt lineárnej regresie v Pythone](https://img.youtube.com/vi/2xkXL5EUpS0/0.jpg)](https://youtu.be/2xkXL5EUpS0 "ML pre začiatočníkov - Váš prvý projekt lineárnej regresie v Pythone")

> 🎥 Kliknite na obrázok vyššie pre krátke video, ktoré vás prevedie týmto cvičením.

V súbore _notebook.ipynb_ priradenom k tejto lekcii vymažte všetky bunky kliknutím na ikonu 'odpadkový kôš'.

V tejto časti budete pracovať s malým datasetom o cukrovke, ktorý je zabudovaný v Scikit-learn pre výučbové účely. Predstavte si, že by ste chceli testovať liečbu pre pacientov s cukrovkou. Modely strojového učenia by vám mohli pomôcť určiť, ktorí pacienti by na liečbu lepšie reagovali, na základe kombinácií premenných. Dokonca aj veľmi základný regresný model, keď ho vizualizujete, môže ukázať informácie o premenných, ktoré by vám pomohli zorganizovať teoretické klinické štúdie.

✅ Existuje mnoho typov regresných metód a výber závisí od odpovede, ktorú hľadáte. Ak chcete predpovedať pravdepodobnú výšku osoby v danom veku, použili by ste lineárnu regresiu, pretože hľadáte **číselnú hodnotu**. Ak chcete zistiť, či by mala byť určitá kuchyňa považovaná za vegánsku alebo nie, hľadáte **kategorizáciu**, takže by ste použili logistickú regresiu. O logistickej regresii sa dozviete viac neskôr. Zamyslite sa nad niektorými otázkami, ktoré môžete klásť dátam, a ktorá z týchto metód by bola vhodnejšia.

Poďme začať s týmto úlohou.

### Import knižníc

Pre tento úlohu importujeme niekoľko knižníc:

- **matplotlib**. Je to užitočný [nástroj na grafy](https://matplotlib.org/), ktorý použijeme na vytvorenie čiarového grafu.
- **numpy**. [numpy](https://numpy.org/doc/stable/user/whatisnumpy.html) je užitočná knižnica na prácu s numerickými dátami v Pythone.
- **sklearn**. Toto je knižnica [Scikit-learn](https://scikit-learn.org/stable/user_guide.html).

Importujte niekoľko knižníc, ktoré vám pomôžu s vašimi úlohami.

1. Pridajte importy zadaním nasledujúceho kódu:

   ```python
   import matplotlib.pyplot as plt
   import numpy as np
   from sklearn import datasets, linear_model, model_selection
   ```

   Vyššie importujete `matplotlib`, `numpy` a importujete `datasets`, `linear_model` a `model_selection` zo `sklearn`. `model_selection` sa používa na rozdelenie dát na trénovacie a testovacie sady.

### Dataset cukrovky

Zabudovaný [dataset cukrovky](https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset) obsahuje 442 vzoriek dát o cukrovke, s 10 črtovými premennými, z ktorých niektoré sú:

- vek: vek v rokoch
- BMI: index telesnej hmotnosti
- BP: priemerný krvný tlak
- S1 TC: T bunky (typ bielych krviniek)

✅ Tento dataset obsahuje koncept „pohlavia“ ako dôležitej črtovej premennej pri výskume cukrovky. Mnohé medicínske datasety obsahujú tento typ binárnej klasifikácie. Zamyslite sa, ako môžu takéto kategorizácie vylúčiť určité časti populácie z liečby.

Teraz načítajte X a y dáta.

> 🎓 Pamätajte, že ide o dohliadané učenie a potrebujeme pomenovaný cieľ 'y'.

V novej bunke kódu načítajte dataset cukrovky volaním `load_diabetes()`. Vstup `return_X_y=True` znamená, že `X` bude dátová matica a `y` cieľová premenná regresie.

1. Pridajte print príkazy na zobrazenie tvaru dátovej matice a jej prvého prvku:

    ```python
    X, y = datasets.load_diabetes(return_X_y=True)
    print(X.shape)
    print(X[0])
    ```

    To, čo dostávate späť ako odpoveď, je n-tica. Priraďujete dve prvé hodnoty n-tice do `X` a `y` v tomto poradí. Viac sa dozviete [o n-ticiach](https://wikipedia.org/wiki/Tuple).

    Vidíte, že tieto dáta majú 442 položiek usporiadaných v poliach po 10 prvkoch:

    ```text
    (442, 10)
    [ 0.03807591  0.05068012  0.06169621  0.02187235 -0.0442235  -0.03482076
    -0.04340085 -0.00259226  0.01990842 -0.01764613]
    ```

    ✅ Zamyslite sa nad vzťahom medzi dátami a cieľovou premennou regresie. Lineárna regresia predpovedá vzťahy medzi črtou X a cieľovou premennou y. Dokážete nájsť [cieľ](https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset) pre dataset cukrovky v dokumentácii? Čo tento dataset ukazuje vzhľadom na cieľ?

2. Ďalej vyberte časť tohto datasetu na vykreslenie výberom 3. stĺpca datasetu. Môžete to urobiť použitím operátora `:`, ktorý vyberie všetky riadky, a potom vyberte 3. stĺpec pomocou indexu (2). Dáta môžete tiež preusporiadať do 2D poľa - čo je pre vykresľovanie požadované - použitím `reshape(n_rows, n_columns)`. Ak je jeden z parametrov -1, príslušná dimenzia sa vypočíta automaticky.

   ```python
   X = X[:, 2]
   X = X.reshape((-1,1))
   ```

   ✅ Kedykoľvek si vytlačte dáta, aby ste skontrolovali ich tvar.

3. Teraz keď máte dáta pripravené na vykreslenie, môžete zistiť, či vám môže stroj pomôcť nájsť logický rozdelenie medzi číslami v tomto datasete. Na to musíte rozdeliť dáta (X) aj cieľ (y) na testovaciu a trénovaciu sadu. Scikit-learn má jednoduchý spôsob, ako to urobiť; môžete testovacie dáta rozdeliť v danom bode.

   ```python
   X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.33)
   ```

4. Teraz ste pripravení trénovať svoj model! Načítajte model lineárnej regresie a trénujte ho so svojimi trénovacími sadami X a y použitím `model.fit()`:

    ```python
    model = linear_model.LinearRegression()
    model.fit(X_train, y_train)
    ```

    ✅ `model.fit()` je funkcia, ktorú uvidíte v mnohých ML knižniciach, ako je TensorFlow

5. Potom vytvorte predikciu na testovacích dátach pomocou funkcie `predict()`. Táto predikcia bude použitá na nakreslenie čiary medzi dátovými skupinami.

    ```python
    y_pred = model.predict(X_test)
    ```

6. Teraz je čas zobraziť dáta na grafe. Matplotlib je veľmi užitočný nástroj pre túto úlohu. Vytvorte scatterplot všetkých testovacích dát X a y a použite predikciu na nakreslenie čiary na najvhodnejšom mieste medzi skupinami dát modelu.

    ```python
    plt.scatter(X_test, y_test,  color='black')
    plt.plot(X_test, y_pred, color='blue', linewidth=3)
    plt.xlabel('Scaled BMIs')
    plt.ylabel('Disease Progression')
    plt.title('A Graph Plot Showing Diabetes Progression Against BMI')
    plt.show()
    ```

   ![scatterplot zobrazujúci body dát o cukrovke](../../../../translated_images/sk/scatterplot.ad8b356bcbb33be6.webp)

   ✅ Zamyslite sa nad tým, čo sa tu deje. Priama čiara prechádza mnohými malými bodmi dát, ale čo vlastne robí? Vidíte, ako by ste mali môcť použiť túto čiaru na predpovedanie toho, kam by mala zapadnúť nová, nevidená dátová položka v súvislosti s osou y grafu? Pokúste sa vyjadriť praktické využitie tohto modelu slovami.

Gratulujeme, vytvorili ste svoj prvý model lineárnej regresie, vytvorili ste predikciu a zobrazili ju v grafe!

---
## 🚀Výzva

Vykreslite inú premennú z tohto datasetu. Nápoveda: upravte tento riadok: `X = X[:,2]`. Čo sa vám podarí objaviť o priebehu cukrovky ako choroby vzhľadom na cieľ tohto datasetu?
## [Kvíz po prednáške](https://ff-quizzes.netlify.app/en/ml/)

## Prehľad a samostatné štúdium

V tomto tutoriáli ste pracovali s jednoduchou lineárnou regresiou, nie s univariátnou alebo multivariátnou lineárnou regresiou. Prečítajte si niečo o rozdieloch medzi týmito metódami alebo si pozrite [toto video](https://www.coursera.org/lecture/quantifying-relationships-regression-models/linear-vs-nonlinear-categorical-variables-ai2Ef)

Prečítajte si viac o koncepte regresie a zamyslite sa nad tým, aké druhy otázok môže táto technika zodpovedať. Prehlbte svoje pochopenie pomocou tohto [návodu](https://docs.microsoft.com/learn/modules/train-evaluate-regression-models?WT.mc_id=academic-77952-leestott).

## Zadanie

[Iná dátová sada](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**Vyhlásenie o zodpovednosti**:
Tento dokument bol preložený pomocou AI prekladateľskej služby [Co-op Translator](https://github.com/Azure/co-op-translator). Hoci sa snažíme o presnosť, vezmite prosím na vedomie, že automatické preklady môžu obsahovať chyby alebo nepresnosti. Pôvodný dokument v jeho natívnom jazyku by mal byť považovaný za autoritatívny zdroj. Pre kritické informácie sa odporúča profesionálny ľudský preklad. Nie sme zodpovední za žiadne nedorozumenia alebo nesprávne interpretácie vyplývajúce z použitia tohto prekladu.
<!-- CO-OP TRANSLATOR DISCLAIMER END -->