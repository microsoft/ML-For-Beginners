<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "aaf391d922bd6de5efba871d514c6d47",
  "translation_date": "2025-09-05T00:54:03+00:00",
  "source_file": "4-Classification/1-Introduction/README.md",
  "language_code": "fi"
}
-->
# Johdanto luokitteluun

Näissä neljässä oppitunnissa tutustut klassisen koneoppimisen keskeiseen osa-alueeseen - _luokitteluun_. Käymme läpi erilaisia luokittelualgoritmeja käyttäen datasettiä, joka käsittelee Aasian ja Intian upeita keittiöitä. Toivottavasti olet nälkäinen!

![vain ripaus!](../../../../4-Classification/1-Introduction/images/pinch.png)

> Juhlista pan-aasialaisia keittiöitä näissä oppitunneissa! Kuva: [Jen Looper](https://twitter.com/jenlooper)

Luokittelu on [ohjatun oppimisen](https://wikipedia.org/wiki/Supervised_learning) muoto, joka muistuttaa paljon regressiotekniikoita. Jos koneoppiminen keskittyy arvojen tai nimien ennustamiseen datasetin avulla, luokittelu jakautuu yleensä kahteen ryhmään: _binääriluokittelu_ ja _moniluokittelu_.

[![Johdanto luokitteluun](https://img.youtube.com/vi/eg8DJYwdMyg/0.jpg)](https://youtu.be/eg8DJYwdMyg "Johdanto luokitteluun")

> 🎥 Klikkaa yllä olevaa kuvaa katsellaksesi videota: MIT:n John Guttag esittelee luokittelua

Muista:

- **Lineaarinen regressio** auttoi sinua ennustamaan muuttujien välisiä suhteita ja tekemään tarkkoja ennusteita siitä, mihin uusi datapiste sijoittuu suhteessa viivaan. Esimerkiksi, voit ennustaa _mikä kurpitsan hinta olisi syyskuussa verrattuna joulukuuhun_.
- **Logistinen regressio** auttoi sinua löytämään "binääriluokkia": tietyllä hintatasolla, _onko kurpitsa oranssi vai ei-oranssi_?

Luokittelu käyttää erilaisia algoritmeja määrittääkseen datapisteen luokan tai tunnisteen. Työskentelemme tämän keittiödata-aineiston kanssa nähdäksemme, voimmeko ainesosaryhmän perusteella määrittää sen alkuperäisen keittiön.

## [Esiluennon kysely](https://ff-quizzes.netlify.app/en/ml/)

> ### [Tämä oppitunti on saatavilla myös R-kielellä!](../../../../4-Classification/1-Introduction/solution/R/lesson_10.html)

### Johdanto

Luokittelu on yksi koneoppimisen tutkijan ja data-analyytikon keskeisistä tehtävistä. Perusluokittelusta binäärisen arvon ("onko tämä sähköposti roskapostia vai ei?") monimutkaiseen kuvien luokitteluun ja segmentointiin tietokonenäön avulla, on aina hyödyllistä pystyä järjestämään data luokkiin ja esittämään kysymyksiä siitä.

Tieteellisemmin ilmaistuna luokittelumenetelmäsi luo ennustavan mallin, joka mahdollistaa syötemuuttujien ja tulosmuuttujien välisen suhteen kartoittamisen.

![binäärinen vs. moniluokittelu](../../../../4-Classification/1-Introduction/images/binary-multiclass.png)

> Binääriset vs. moniluokitusongelmat, joita luokittelualgoritmit käsittelevät. Infografiikka: [Jen Looper](https://twitter.com/jenlooper)

Ennen kuin aloitamme datan puhdistamisen, visualisoinnin ja valmistelun koneoppimistehtäviämme varten, opitaan hieman siitä, miten koneoppimista voidaan hyödyntää datan luokittelussa.

Luokittelu, joka on johdettu [tilastotieteestä](https://wikipedia.org/wiki/Statistical_classification), käyttää ominaisuuksia, kuten `smoker`, `weight` ja `age`, määrittääkseen _todennäköisyyden sairastua X-tautiin_. Ohjatun oppimisen tekniikkana, joka muistuttaa aiemmin suorittamiasi regressioharjoituksia, datasi on merkitty, ja koneoppimisalgoritmit käyttävät näitä merkintöjä luokitellakseen ja ennustaakseen datasetin luokkia (tai 'ominaisuuksia') ja liittääkseen ne ryhmään tai lopputulokseen.

✅ Mieti hetki datasettiä, joka käsittelee keittiöitä. Mitä moniluokittelumalli voisi vastata? Mitä binäärimalli voisi vastata? Entä jos haluaisit selvittää, käyttääkö tietty keittiö todennäköisesti sarviapilaa? Entä jos haluaisit nähdä, voisitko luoda tyypillisen intialaisen ruokalajin, kun sinulle annetaan lahjaksi ruokakassi, joka sisältää tähtianista, artisokkaa, kukkakaalia ja piparjuurta?

[![Hullut mysteerikorit](https://img.youtube.com/vi/GuTeDbaNoEU/0.jpg)](https://youtu.be/GuTeDbaNoEU "Hullut mysteerikorit")

> 🎥 Klikkaa yllä olevaa kuvaa katsellaksesi videota. Ohjelman 'Chopped' koko idea perustuu 'mysteerikoriin', jossa kokkien täytyy valmistaa ruokalaji satunnaisista ainesosista. Koneoppimismalli olisi varmasti auttanut!

## Hei 'luokittelija'

Kysymys, jonka haluamme esittää tästä keittiödatasta, on itse asiassa **moniluokittelukysymys**, koska meillä on useita mahdollisia kansallisia keittiöitä, joiden kanssa työskennellä. Kun annetaan joukko ainesosia, mihin näistä monista luokista data sopii?

Scikit-learn tarjoaa useita erilaisia algoritmeja datan luokitteluun riippuen siitä, millaisen ongelman haluat ratkaista. Seuraavissa kahdessa oppitunnissa opit useista näistä algoritmeista.

## Harjoitus - puhdista ja tasapainota datasi

Ensimmäinen tehtävä ennen projektin aloittamista on puhdistaa ja **tasapainottaa** datasi saadaksesi parempia tuloksia. Aloita tyhjällä _notebook.ipynb_-tiedostolla tämän kansion juurihakemistossa.

Ensimmäinen asennettava asia on [imblearn](https://imbalanced-learn.org/stable/). Tämä on Scikit-learn-paketti, joka auttaa sinua tasapainottamaan dataa paremmin (opit tästä tehtävästä lisää hetken kuluttua).

1. Asenna `imblearn` suorittamalla `pip install` seuraavasti:

    ```python
    pip install imblearn
    ```

1. Tuo tarvittavat paketit datan tuontia ja visualisointia varten, ja tuo myös `SMOTE` `imblearn`-kirjastosta.

    ```python
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    import numpy as np
    from imblearn.over_sampling import SMOTE
    ```

    Nyt olet valmis tuomaan datan seuraavaksi.

1. Seuraava tehtävä on tuoda data:

    ```python
    df  = pd.read_csv('../data/cuisines.csv')
    ```

   `read_csv()` lukee csv-tiedoston _cusines.csv_ sisällön ja sijoittaa sen muuttujaan `df`.

1. Tarkista datan muoto:

    ```python
    df.head()
    ```

   Ensimmäiset viisi riviä näyttävät tältä:

    ```output
    |     | Unnamed: 0 | cuisine | almond | angelica | anise | anise_seed | apple | apple_brandy | apricot | armagnac | ... | whiskey | white_bread | white_wine | whole_grain_wheat_flour | wine | wood | yam | yeast | yogurt | zucchini |
    | --- | ---------- | ------- | ------ | -------- | ----- | ---------- | ----- | ------------ | ------- | -------- | --- | ------- | ----------- | ---------- | ----------------------- | ---- | ---- | --- | ----- | ------ | -------- |
    | 0   | 65         | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
    | 1   | 66         | indian  | 1      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
    | 2   | 67         | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
    | 3   | 68         | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
    | 4   | 69         | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 1      | 0        |
    ```

1. Hanki tietoa tästä datasta kutsumalla `info()`:

    ```python
    df.info()
    ```

    Tulosteesi näyttää tältä:

    ```output
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 2448 entries, 0 to 2447
    Columns: 385 entries, Unnamed: 0 to zucchini
    dtypes: int64(384), object(1)
    memory usage: 7.2+ MB
    ```

## Harjoitus - keittiöiden tutkiminen

Nyt työ alkaa muuttua mielenkiintoisemmaksi. Tutkitaan datan jakautumista keittiöittäin.

1. Piirrä data palkkeina kutsumalla `barh()`:

    ```python
    df.cuisine.value_counts().plot.barh()
    ```

    ![keittiöiden datan jakautuminen](../../../../4-Classification/1-Introduction/images/cuisine-dist.png)

    Keittiöitä on rajallinen määrä, mutta datan jakautuminen on epätasaista. Voit korjata sen! Ennen kuin teet niin, tutki hieman lisää.

1. Selvitä, kuinka paljon dataa on saatavilla keittiöittäin ja tulosta se:

    ```python
    thai_df = df[(df.cuisine == "thai")]
    japanese_df = df[(df.cuisine == "japanese")]
    chinese_df = df[(df.cuisine == "chinese")]
    indian_df = df[(df.cuisine == "indian")]
    korean_df = df[(df.cuisine == "korean")]
    
    print(f'thai df: {thai_df.shape}')
    print(f'japanese df: {japanese_df.shape}')
    print(f'chinese df: {chinese_df.shape}')
    print(f'indian df: {indian_df.shape}')
    print(f'korean df: {korean_df.shape}')
    ```

    Tulosteesi näyttää tältä:

    ```output
    thai df: (289, 385)
    japanese df: (320, 385)
    chinese df: (442, 385)
    indian df: (598, 385)
    korean df: (799, 385)
    ```

## Ainesosien tutkiminen

Nyt voit syventyä dataan ja oppia, mitkä ovat tyypilliset ainesosat keittiöittäin. Sinun tulisi poistaa toistuva data, joka aiheuttaa sekaannusta keittiöiden välillä, joten opitaan tästä ongelmasta.

1. Luo Pythonissa funktio `create_ingredient()`, joka luo ainesosadataframen. Tämä funktio aloittaa poistamalla hyödyttömän sarakkeen ja lajittelee ainesosat niiden määrän mukaan:

    ```python
    def create_ingredient_df(df):
        ingredient_df = df.T.drop(['cuisine','Unnamed: 0']).sum(axis=1).to_frame('value')
        ingredient_df = ingredient_df[(ingredient_df.T != 0).any()]
        ingredient_df = ingredient_df.sort_values(by='value', ascending=False,
        inplace=False)
        return ingredient_df
    ```

   Nyt voit käyttää tätä funktiota saadaksesi käsityksen kymmenestä suosituimmasta ainesosasta keittiöittäin.

1. Kutsu `create_ingredient()` ja piirrä se kutsumalla `barh()`:

    ```python
    thai_ingredient_df = create_ingredient_df(thai_df)
    thai_ingredient_df.head(10).plot.barh()
    ```

    ![thai](../../../../4-Classification/1-Introduction/images/thai.png)

1. Tee sama japanilaiselle datalle:

    ```python
    japanese_ingredient_df = create_ingredient_df(japanese_df)
    japanese_ingredient_df.head(10).plot.barh()
    ```

    ![japanilainen](../../../../4-Classification/1-Introduction/images/japanese.png)

1. Nyt kiinalaisille ainesosille:

    ```python
    chinese_ingredient_df = create_ingredient_df(chinese_df)
    chinese_ingredient_df.head(10).plot.barh()
    ```

    ![kiinalainen](../../../../4-Classification/1-Introduction/images/chinese.png)

1. Piirrä intialaiset ainesosat:

    ```python
    indian_ingredient_df = create_ingredient_df(indian_df)
    indian_ingredient_df.head(10).plot.barh()
    ```

    ![intialainen](../../../../4-Classification/1-Introduction/images/indian.png)

1. Lopuksi piirrä korealaiset ainesosat:

    ```python
    korean_ingredient_df = create_ingredient_df(korean_df)
    korean_ingredient_df.head(10).plot.barh()
    ```

    ![korealainen](../../../../4-Classification/1-Introduction/images/korean.png)

1. Poista nyt yleisimmät ainesosat, jotka aiheuttavat sekaannusta eri keittiöiden välillä, kutsumalla `drop()`:

   Kaikki rakastavat riisiä, valkosipulia ja inkivääriä!

    ```python
    feature_df= df.drop(['cuisine','Unnamed: 0','rice','garlic','ginger'], axis=1)
    labels_df = df.cuisine #.unique()
    feature_df.head()
    ```

## Tasapainota datasetti

Nyt kun olet puhdistanut datan, käytä [SMOTE](https://imbalanced-learn.org/dev/references/generated/imblearn.over_sampling.SMOTE.html) - "Synthetic Minority Over-sampling Technique" -menetelmää tasapainottamiseen.

1. Kutsu `fit_resample()`, tämä strategia luo uusia näytteitä interpoloinnin avulla.

    ```python
    oversample = SMOTE()
    transformed_feature_df, transformed_label_df = oversample.fit_resample(feature_df, labels_df)
    ```

    Tasapainottamalla datasi saat parempia tuloksia sen luokittelussa. Mieti binääriluokittelua. Jos suurin osa datastasi kuuluu yhteen luokkaan, koneoppimismalli ennustaa todennäköisemmin kyseistä luokkaa, koska sille on enemmän dataa. Datatasapainotus poistaa tämän epätasapainon.

1. Nyt voit tarkistaa ainesosien tunnisteiden määrät:

    ```python
    print(f'new label count: {transformed_label_df.value_counts()}')
    print(f'old label count: {df.cuisine.value_counts()}')
    ```

    Tulosteesi näyttää tältä:

    ```output
    new label count: korean      799
    chinese     799
    indian      799
    japanese    799
    thai        799
    Name: cuisine, dtype: int64
    old label count: korean      799
    indian      598
    chinese     442
    japanese    320
    thai        289
    Name: cuisine, dtype: int64
    ```

    Data on nyt siisti, tasapainoinen ja erittäin herkullinen!

1. Viimeinen vaihe on tallentaa tasapainotettu data, mukaan lukien tunnisteet ja ominaisuudet, uuteen dataframeen, joka voidaan viedä tiedostoon:

    ```python
    transformed_df = pd.concat([transformed_label_df,transformed_feature_df],axis=1, join='outer')
    ```

1. Voit tarkastella dataa vielä kerran käyttämällä `transformed_df.head()` ja `transformed_df.info()`. Tallenna kopio tästä datasta tulevia oppitunteja varten:

    ```python
    transformed_df.head()
    transformed_df.info()
    transformed_df.to_csv("../data/cleaned_cuisines.csv")
    ```

    Tämä uusi CSV löytyy nyt juuridatan kansiosta.

---

## 🚀Haaste

Tämä opetusohjelma sisältää useita mielenkiintoisia datasettejä. Tutki `data`-kansioita ja katso, sisältävätkö ne datasettejä, jotka sopisivat binääriseen tai moniluokitteluun? Mitä kysymyksiä esittäisit tästä datasetistä?

## [Jälkiluennon kysely](https://ff-quizzes.netlify.app/en/ml/)

## Kertaus ja itseopiskelu

Tutki SMOTE:n APIa. Mihin käyttötarkoituksiin se sopii parhaiten? Mitä ongelmia se ratkaisee?

## Tehtävä 

[Tutki luokittelumenetelmiä](assignment.md)

---

**Vastuuvapauslauseke**:  
Tämä asiakirja on käännetty käyttämällä tekoälypohjaista käännöspalvelua [Co-op Translator](https://github.com/Azure/co-op-translator). Vaikka pyrimme tarkkuuteen, huomioithan, että automaattiset käännökset voivat sisältää virheitä tai epätarkkuuksia. Alkuperäistä asiakirjaa sen alkuperäisellä kielellä tulisi pitää ensisijaisena lähteenä. Kriittisen tiedon osalta suositellaan ammattimaista ihmiskäännöstä. Emme ole vastuussa väärinkäsityksistä tai virhetulkinnoista, jotka johtuvat tämän käännöksen käytöstä.