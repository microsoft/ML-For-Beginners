<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "7c077988328ebfe33b24d07945f16eca",
  "translation_date": "2025-09-05T11:46:24+00:00",
  "source_file": "2-Regression/2-Data/README.md",
  "language_code": "sl"
}
-->
# Ustvarjanje regresijskega modela s Scikit-learn: priprava in vizualizacija podatkov

![Infografika vizualizacije podatkov](../../../../2-Regression/2-Data/images/data-visualization.png)

Infografika avtorja [Dasani Madipalli](https://twitter.com/dasani_decoded)

## [Predavanje kviz](https://ff-quizzes.netlify.app/en/ml/)

> ### [To lekcijo lahko najdete tudi v jeziku R!](../../../../2-Regression/2-Data/solution/R/lesson_2.html)

## Uvod

Zdaj, ko imate na voljo orodja za gradnjo modelov strojnega učenja s Scikit-learn, ste pripravljeni začeti postavljati vprašanja svojim podatkom. Pri delu s podatki in uporabi rešitev strojnega učenja je zelo pomembno, da znate postaviti prava vprašanja, da lahko pravilno izkoristite potenciale svojega nabora podatkov.

V tej lekciji boste spoznali:

- Kako pripraviti podatke za gradnjo modela.
- Kako uporabiti Matplotlib za vizualizacijo podatkov.

## Postavljanje pravih vprašanj svojim podatkom

Vprašanje, na katerega želite odgovor, bo določilo, katere vrste algoritmov strojnega učenja boste uporabili. Kakovost odgovora pa bo močno odvisna od narave vaših podatkov.

Oglejte si [podatke](https://github.com/microsoft/ML-For-Beginners/blob/main/2-Regression/data/US-pumpkins.csv), ki so na voljo za to lekcijo. Datoteko .csv lahko odprete v programu VS Code. Hiter pregled takoj pokaže, da so prisotne prazne vrednosti ter mešanica nizov in številskih podatkov. Obstaja tudi nenavaden stolpec z imenom 'Package', kjer so podatki mešanica 'sacks', 'bins' in drugih vrednosti. Podatki so pravzaprav precej neurejeni.

[![ML za začetnike - Kako analizirati in očistiti nabor podatkov](https://img.youtube.com/vi/5qGjczWTrDQ/0.jpg)](https://youtu.be/5qGjczWTrDQ "ML za začetnike - Kako analizirati in očistiti nabor podatkov")

> 🎥 Kliknite zgornjo sliko za kratek video o pripravi podatkov za to lekcijo.

Pravzaprav ni pogosto, da dobite nabor podatkov, ki je popolnoma pripravljen za ustvarjanje modela strojnega učenja. V tej lekciji se boste naučili, kako pripraviti surove podatke z uporabo standardnih knjižnic za Python. Spoznali boste tudi različne tehnike za vizualizacijo podatkov.

## Študija primera: 'trg buč'

V tej mapi boste našli datoteko .csv v korenski mapi `data`, imenovano [US-pumpkins.csv](https://github.com/microsoft/ML-For-Beginners/blob/main/2-Regression/data/US-pumpkins.csv), ki vključuje 1757 vrstic podatkov o trgu buč, razvrščenih po mestih. To so surovi podatki, pridobljeni iz [Specialty Crops Terminal Markets Standard Reports](https://www.marketnews.usda.gov/mnp/fv-report-config-step1?type=termPrice), ki jih distribuira Ministrstvo za kmetijstvo ZDA.

### Priprava podatkov

Ti podatki so v javni domeni. Na spletni strani USDA jih je mogoče prenesti v številnih ločenih datotekah, po mestih. Da bi se izognili prevelikemu številu ločenih datotek, smo združili vse podatke mest v eno preglednico, kar pomeni, da smo podatke že nekoliko _pripravili_. Zdaj si podrobneje oglejmo podatke.

### Podatki o bučah - prvi vtisi

Kaj opazite pri teh podatkih? Že prej ste videli, da gre za mešanico nizov, števil, praznih vrednosti in nenavadnih vrednosti, ki jih morate razumeti.

Katero vprašanje lahko postavite tem podatkom z uporabo regresijske tehnike? Kaj pa "Napovedovanje cene buče za prodajo v določenem mesecu"? Če ponovno pogledate podatke, boste opazili, da morate narediti nekaj sprememb, da ustvarite strukturo podatkov, potrebno za to nalogo.

## Vaja - analiza podatkov o bučah

Uporabimo [Pandas](https://pandas.pydata.org/) (ime pomeni `Python Data Analysis`), zelo uporabno orodje za oblikovanje podatkov, za analizo in pripravo teh podatkov o bučah.

### Najprej preverite manjkajoče datume

Najprej boste morali preveriti, ali manjkajo datumi:

1. Pretvorite datume v mesečni format (to so datumi iz ZDA, zato je format `MM/DD/YYYY`).
2. Izluščite mesec v nov stolpec.

Odprite datoteko _notebook.ipynb_ v Visual Studio Code in uvozite preglednico v nov Pandas dataframe.

1. Uporabite funkcijo `head()`, da si ogledate prvih pet vrstic.

    ```python
    import pandas as pd
    pumpkins = pd.read_csv('../data/US-pumpkins.csv')
    pumpkins.head()
    ```

    ✅ Katero funkcijo bi uporabili za ogled zadnjih petih vrstic?

1. Preverite, ali v trenutnem dataframeu manjkajo podatki:

    ```python
    pumpkins.isnull().sum()
    ```

    Manjkajo podatki, vendar morda to ne bo pomembno za nalogo.

1. Da bo vaš dataframe lažji za delo, izberite samo stolpce, ki jih potrebujete, z uporabo funkcije `loc`, ki iz izvirnega dataframea izlušči skupino vrstic (podanih kot prvi parameter) in stolpcev (podanih kot drugi parameter). Izraz `:` v spodnjem primeru pomeni "vse vrstice".

    ```python
    columns_to_select = ['Package', 'Low Price', 'High Price', 'Date']
    pumpkins = pumpkins.loc[:, columns_to_select]
    ```

### Drugič, določite povprečno ceno buče

Razmislite, kako določiti povprečno ceno buče v določenem mesecu. Katere stolpce bi izbrali za to nalogo? Namig: potrebovali boste 3 stolpce.

Rešitev: vzemite povprečje stolpcev `Low Price` in `High Price`, da napolnite nov stolpec Price, in pretvorite stolpec Date, da prikaže samo mesec. Na srečo, glede na zgornjo preverbo, ni manjkajočih podatkov za datume ali cene.

1. Za izračun povprečja dodajte naslednjo kodo:

    ```python
    price = (pumpkins['Low Price'] + pumpkins['High Price']) / 2

    month = pd.DatetimeIndex(pumpkins['Date']).month

    ```

   ✅ Po želji natisnite katerikoli podatek, ki ga želite preveriti, z uporabo `print(month)`.

2. Zdaj kopirajte pretvorjene podatke v nov Pandas dataframe:

    ```python
    new_pumpkins = pd.DataFrame({'Month': month, 'Package': pumpkins['Package'], 'Low Price': pumpkins['Low Price'],'High Price': pumpkins['High Price'], 'Price': price})
    ```

    Če natisnete svoj dataframe, boste videli čist, urejen nabor podatkov, na katerem lahko gradite svoj novi regresijski model.

### Ampak počakajte! Nekaj je čudnega

Če pogledate stolpec `Package`, so buče prodane v različnih konfiguracijah. Nekatere so prodane v '1 1/9 bushel' enotah, nekatere v '1/2 bushel' enotah, nekatere na bučo, nekatere na funt, in nekatere v velikih škatlah različnih širin.

> Zdi se, da je buče zelo težko dosledno tehtati

Če se poglobite v izvirne podatke, je zanimivo, da imajo vse enote z `Unit of Sale` enako 'EACH' ali 'PER BIN' tudi vrsto `Package` na palec, na bin ali 'each'. Zdi se, da je buče zelo težko dosledno tehtati, zato jih filtrirajmo tako, da izberemo samo buče z nizom 'bushel' v njihovem stolpcu `Package`.

1. Dodajte filter na vrhu datoteke, pod začetnim uvozom .csv:

    ```python
    pumpkins = pumpkins[pumpkins['Package'].str.contains('bushel', case=True, regex=True)]
    ```

    Če zdaj natisnete podatke, lahko vidite, da dobite le približno 415 vrstic podatkov, ki vsebujejo buče po buslju.

### Ampak počakajte! Še nekaj je treba narediti

Ste opazili, da se količina buslja razlikuje po vrsticah? Potrebno je normalizirati cene, da prikažete cene na buselj, zato naredite nekaj izračunov za standardizacijo.

1. Dodajte te vrstice po bloku, ki ustvarja dataframe new_pumpkins:

    ```python
    new_pumpkins.loc[new_pumpkins['Package'].str.contains('1 1/9'), 'Price'] = price/(1 + 1/9)

    new_pumpkins.loc[new_pumpkins['Package'].str.contains('1/2'), 'Price'] = price/(1/2)
    ```

✅ Po podatkih [The Spruce Eats](https://www.thespruceeats.com/how-much-is-a-bushel-1389308) teža buslja variira glede na vrsto pridelka, saj gre za meritev prostornine. "Buselj paradižnikov, na primer, naj bi tehtal 56 funtov... Listi in zelenjava zavzamejo več prostora z manjšo težo, zato buselj špinače tehta le 20 funtov." Vse skupaj je precej zapleteno! Ne ukvarjajmo se s pretvorbo buslja v funte, ampak raje določimo ceno na buselj. Vse to proučevanje busljev buč pa kaže, kako zelo pomembno je razumeti naravo svojih podatkov!

Zdaj lahko analizirate cene na enoto glede na njihovo meritev buslja. Če še enkrat natisnete podatke, lahko vidite, kako so standardizirani.

✅ Ste opazili, da so buče, prodane po pol buslja, zelo drage? Ali lahko ugotovite, zakaj? Namig: majhne buče so veliko dražje od velikih, verjetno zato, ker jih je veliko več na buselj, glede na neizkoriščen prostor, ki ga zavzame ena velika votla buča za pito.

## Strategije vizualizacije

Del naloge podatkovnega znanstvenika je prikazati kakovost in naravo podatkov, s katerimi dela. To pogosto dosežejo z ustvarjanjem zanimivih vizualizacij, kot so grafi, diagrami in tabele, ki prikazujejo različne vidike podatkov. Na ta način lahko vizualno prikažejo odnose in vrzeli, ki jih je sicer težko odkriti.

[![ML za začetnike - Kako vizualizirati podatke z Matplotlib](https://img.youtube.com/vi/SbUkxH6IJo0/0.jpg)](https://youtu.be/SbUkxH6IJo0 "ML za začetnike - Kako vizualizirati podatke z Matplotlib")

> 🎥 Kliknite zgornjo sliko za kratek video o vizualizaciji podatkov za to lekcijo.

Vizualizacije lahko pomagajo tudi pri določanju tehnike strojnega učenja, ki je najbolj primerna za podatke. Na primer, raztrosni diagram, ki sledi liniji, kaže, da so podatki primerni za nalogo linearne regresije.

Ena od knjižnic za vizualizacijo podatkov, ki dobro deluje v Jupyterjevih beležnicah, je [Matplotlib](https://matplotlib.org/) (ki ste jo videli tudi v prejšnji lekciji).

> Pridobite več izkušenj z vizualizacijo podatkov v [teh vadnicah](https://docs.microsoft.com/learn/modules/explore-analyze-data-with-python?WT.mc_id=academic-77952-leestott).

## Vaja - eksperimentiranje z Matplotlib

Poskusite ustvariti osnovne grafe za prikaz novega dataframea, ki ste ga pravkar ustvarili. Kaj bi pokazal osnovni linijski graf?

1. Uvozite Matplotlib na vrhu datoteke, pod uvozom Pandas:

    ```python
    import matplotlib.pyplot as plt
    ```

1. Ponovno zaženite celotno beležnico za osvežitev.
1. Na dnu beležnice dodajte celico za prikaz podatkov kot škatlasti graf:

    ```python
    price = new_pumpkins.Price
    month = new_pumpkins.Month
    plt.scatter(price, month)
    plt.show()
    ```

    ![Raztrosni diagram, ki prikazuje razmerje med ceno in mesecem](../../../../2-Regression/2-Data/images/scatterplot.png)

    Ali je ta graf uporaben? Vas kaj na njem preseneča?

    Ni posebej uporaben, saj le prikazuje razpršenost točk v določenem mesecu.

### Naredimo ga uporabnega

Da bi grafi prikazovali uporabne podatke, jih običajno morate nekako združiti. Poskusimo ustvariti graf, kjer y-os prikazuje mesece, podatki pa prikazujejo porazdelitev podatkov.

1. Dodajte celico za ustvarjanje združenega stolpčnega diagrama:

    ```python
    new_pumpkins.groupby(['Month'])['Price'].mean().plot(kind='bar')
    plt.ylabel("Pumpkin Price")
    ```

    ![Stolpčni diagram, ki prikazuje razmerje med ceno in mesecem](../../../../2-Regression/2-Data/images/barchart.png)

    To je bolj uporabna vizualizacija podatkov! Zdi se, da kaže, da so najvišje cene buč v septembru in oktobru. Ali to ustreza vašim pričakovanjem? Zakaj ali zakaj ne?

---

## 🚀Izziv

Raziščite različne vrste vizualizacij, ki jih ponuja Matplotlib. Katere vrste so najbolj primerne za regresijske probleme?

## [Kviz po predavanju](https://ff-quizzes.netlify.app/en/ml/)

## Pregled in samostojno učenje

Oglejte si različne načine vizualizacije podatkov. Naredite seznam različnih knjižnic, ki so na voljo, in zabeležite, katere so najboljše za določene vrste nalog, na primer 2D vizualizacije v primerjavi s 3D vizualizacijami. Kaj odkrijete?

## Naloga

[Raziskovanje vizualizacije](assignment.md)

---

**Omejitev odgovornosti**:  
Ta dokument je bil preveden z uporabo storitve za strojno prevajanje [Co-op Translator](https://github.com/Azure/co-op-translator). Čeprav si prizadevamo za natančnost, vas prosimo, da upoštevate, da lahko avtomatizirani prevodi vsebujejo napake ali netočnosti. Izvirni dokument v njegovem izvirnem jeziku je treba obravnavati kot avtoritativni vir. Za ključne informacije priporočamo strokovno človeško prevajanje. Ne prevzemamo odgovornosti za morebitna nesporazumevanja ali napačne razlage, ki izhajajo iz uporabe tega prevoda.