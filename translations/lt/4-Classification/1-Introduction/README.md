<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "aaf391d922bd6de5efba871d514c6d47",
  "translation_date": "2025-09-05T08:00:35+00:00",
  "source_file": "4-Classification/1-Introduction/README.md",
  "language_code": "lt"
}
-->
# Įvadas į klasifikaciją

Šiose keturiose pamokose jūs susipažinsite su pagrindiniu klasikinių mašininio mokymosi aspektu - _klasifikacija_. Mes išnagrinėsime įvairius klasifikacijos algoritmus naudodami duomenų rinkinį apie visus nuostabius Azijos ir Indijos virtuvių patiekalus. Tikimės, kad esate alkani!

![tik žiupsnelis!](../../../../4-Classification/1-Introduction/images/pinch.png)

> Švęskite pan-Azijos virtuves šiose pamokose! Vaizdas sukurtas [Jen Looper](https://twitter.com/jenlooper)

Klasifikacija yra [prižiūrimo mokymosi](https://wikipedia.org/wiki/Supervised_learning) forma, kuri turi daug bendro su regresijos metodais. Jei mašininis mokymasis yra susijęs su vertybių ar pavadinimų prognozavimu naudojant duomenų rinkinius, tuomet klasifikacija paprastai skirstoma į dvi grupes: _dvejetainė klasifikacija_ ir _daugiaklasė klasifikacija_.

[![Įvadas į klasifikaciją](https://img.youtube.com/vi/eg8DJYwdMyg/0.jpg)](https://youtu.be/eg8DJYwdMyg "Įvadas į klasifikaciją")

> 🎥 Spustelėkite aukščiau esantį vaizdą, kad peržiūrėtumėte vaizdo įrašą: MIT profesorius John Guttag pristato klasifikaciją

Prisiminkite:

- **Linijinė regresija** padėjo jums numatyti ryšius tarp kintamųjų ir tiksliai prognozuoti, kur naujas duomenų taškas atsidurs santykyje su ta linija. Pavyzdžiui, galėjote numatyti _kokia bus moliūgo kaina rugsėjį ir gruodį_.
- **Logistinė regresija** padėjo jums atrasti „dvejetaines kategorijas“: esant tam tikrai kainai, _ar šis moliūgas yra oranžinis, ar neoranžinis_?

Klasifikacija naudoja įvairius algoritmus, kad nustatytų kitus būdus, kaip priskirti duomenų tašką tam tikrai etiketei ar klasei. Dirbkime su šiais virtuvės duomenimis, kad pamatytume, ar stebėdami ingredientų grupę galime nustatyti jos kilmės virtuvę.

## [Prieš pamokos testas](https://ff-quizzes.netlify.app/en/ml/)

> ### [Ši pamoka pasiekiama R kalba!](../../../../4-Classification/1-Introduction/solution/R/lesson_10.html)

### Įvadas

Klasifikacija yra viena iš pagrindinių mašininio mokymosi tyrėjo ir duomenų mokslininko veiklų. Nuo paprastos dvejetainės vertės klasifikacijos („ar šis el. laiškas yra šlamštas, ar ne?“) iki sudėtingos vaizdų klasifikacijos ir segmentavimo naudojant kompiuterinį matymą, visada naudinga sugebėti suskirstyti duomenis į klases ir užduoti jiems klausimus.

Moksliniu požiūriu, jūsų klasifikacijos metodas sukuria prognozavimo modelį, kuris leidžia jums susieti įvesties kintamuosius su išvesties kintamaisiais.

![dvejetainė vs. daugiaklasė klasifikacija](../../../../4-Classification/1-Introduction/images/binary-multiclass.png)

> Dvejetainės ir daugiaklasės problemos, kurias sprendžia klasifikacijos algoritmai. Infografika sukūrė [Jen Looper](https://twitter.com/jenlooper)

Prieš pradėdami duomenų valymo, vizualizavimo ir paruošimo ML užduotims procesą, sužinokime šiek tiek daugiau apie įvairius būdus, kaip mašininis mokymasis gali būti naudojamas duomenims klasifikuoti.

Klasifikacija, kilusi iš [statistikos](https://wikipedia.org/wiki/Statistical_classification), naudojant klasikinius mašininio mokymosi metodus, naudoja tokias savybes kaip `smoker`, `weight` ir `age`, kad nustatytų _tikimybę susirgti X liga_. Kaip prižiūrimo mokymosi technika, panaši į anksčiau atliktus regresijos pratimus, jūsų duomenys yra pažymėti, o ML algoritmai naudoja tuos žymenis, kad klasifikuotų ir prognozuotų duomenų rinkinio klases (arba „savybes“) ir priskirtų jas grupei ar rezultatui.

✅ Skirkite akimirką įsivaizduoti duomenų rinkinį apie virtuves. Ką galėtų atsakyti daugiaklasis modelis? Ką galėtų atsakyti dvejetainis modelis? O jei norėtumėte nustatyti, ar tam tikra virtuvė greičiausiai naudoja ožragę? O jei norėtumėte sužinoti, ar, gavę maišą su žvaigždiniu anyžiumi, artišokais, žiediniais kopūstais ir krienais, galėtumėte sukurti tipišką indišką patiekalą?

[![Keisti paslaptingi krepšiai](https://img.youtube.com/vi/GuTeDbaNoEU/0.jpg)](https://youtu.be/GuTeDbaNoEU "Keisti paslaptingi krepšiai")

> 🎥 Spustelėkite aukščiau esantį vaizdą, kad peržiūrėtumėte vaizdo įrašą. Visas šou „Chopped“ pagrindas yra „paslaptingas krepšys“, kuriame virėjai turi pagaminti patiekalą iš atsitiktinai parinktų ingredientų. Tikrai ML modelis būtų padėjęs!

## Sveiki, „klasifikatoriau“

Klausimas, kurį norime užduoti šiam virtuvės duomenų rinkiniui, iš tikrųjų yra **daugiaklasis klausimas**, nes turime keletą galimų nacionalinių virtuvių, su kuriomis galime dirbti. Atsižvelgiant į ingredientų rinkinį, kuriai iš šių daugelio klasių duomenys tiks?

Scikit-learn siūlo keletą skirtingų algoritmų, kuriuos galima naudoti duomenims klasifikuoti, priklausomai nuo problemos, kurią norite išspręsti. Kitose dviejose pamokose sužinosite apie keletą šių algoritmų.

## Pratimas - išvalykite ir subalansuokite savo duomenis

Pirmoji užduotis prieš pradedant šį projektą yra išvalyti ir **subalansuoti** savo duomenis, kad gautumėte geresnius rezultatus. Pradėkite nuo tuščio _notebook.ipynb_ failo, esančio šio aplanko šaknyje.

Pirmas dalykas, kurį reikia įdiegti, yra [imblearn](https://imbalanced-learn.org/stable/). Tai yra Scikit-learn paketas, kuris leis jums geriau subalansuoti duomenis (apie šią užduotį sužinosite netrukus).

1. Norėdami įdiegti `imblearn`, paleiskite `pip install`, kaip parodyta:

    ```python
    pip install imblearn
    ```

1. Importuokite paketus, kurių jums reikia norint importuoti duomenis ir juos vizualizuoti, taip pat importuokite `SMOTE` iš `imblearn`.

    ```python
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    import numpy as np
    from imblearn.over_sampling import SMOTE
    ```

    Dabar esate pasiruošę importuoti duomenis.

1. Kitas žingsnis - importuoti duomenis:

    ```python
    df  = pd.read_csv('../data/cuisines.csv')
    ```

   Naudojant `read_csv()` bus perskaitytas csv failo _cusines.csv_ turinys ir patalpintas į kintamąjį `df`.

1. Patikrinkite duomenų formą:

    ```python
    df.head()
    ```

   Pirmos penkios eilutės atrodo taip:

    ```output
    |     | Unnamed: 0 | cuisine | almond | angelica | anise | anise_seed | apple | apple_brandy | apricot | armagnac | ... | whiskey | white_bread | white_wine | whole_grain_wheat_flour | wine | wood | yam | yeast | yogurt | zucchini |
    | --- | ---------- | ------- | ------ | -------- | ----- | ---------- | ----- | ------------ | ------- | -------- | --- | ------- | ----------- | ---------- | ----------------------- | ---- | ---- | --- | ----- | ------ | -------- |
    | 0   | 65         | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
    | 1   | 66         | indian  | 1      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
    | 2   | 67         | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
    | 3   | 68         | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
    | 4   | 69         | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 1      | 0        |
    ```

1. Gaukite informaciją apie šiuos duomenis, iškviesdami `info()`:

    ```python
    df.info()
    ```

    Jūsų rezultatas atrodo taip:

    ```output
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 2448 entries, 0 to 2447
    Columns: 385 entries, Unnamed: 0 to zucchini
    dtypes: int64(384), object(1)
    memory usage: 7.2+ MB
    ```

## Pratimas - sužinokite apie virtuves

Dabar darbas tampa įdomesnis. Atraskime duomenų pasiskirstymą pagal virtuvę.

1. Nubraižykite duomenis kaip stulpelius, iškviesdami `barh()`:

    ```python
    df.cuisine.value_counts().plot.barh()
    ```

    ![virtuvės duomenų pasiskirstymas](../../../../4-Classification/1-Introduction/images/cuisine-dist.png)

    Yra ribotas virtuvių skaičius, tačiau duomenų pasiskirstymas yra netolygus. Galite tai ištaisyti! Prieš tai darydami, šiek tiek daugiau ištirkite.

1. Sužinokite, kiek duomenų yra kiekvienai virtuvei, ir atspausdinkite:

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

    Rezultatas atrodo taip:

    ```output
    thai df: (289, 385)
    japanese df: (320, 385)
    chinese df: (442, 385)
    indian df: (598, 385)
    korean df: (799, 385)
    ```

## Ingredientų atradimas

Dabar galite giliau pasinerti į duomenis ir sužinoti, kokie yra tipiški ingredientai kiekvienai virtuvei. Turėtumėte pašalinti pasikartojančius duomenis, kurie sukelia painiavą tarp virtuvių, todėl sužinokime apie šią problemą.

1. Sukurkite funkciją `create_ingredient()` Python kalba, kad sukurtumėte ingredientų duomenų rėmelį. Ši funkcija pradės pašalindama nereikalingą stulpelį ir rūšiuos ingredientus pagal jų skaičių:

    ```python
    def create_ingredient_df(df):
        ingredient_df = df.T.drop(['cuisine','Unnamed: 0']).sum(axis=1).to_frame('value')
        ingredient_df = ingredient_df[(ingredient_df.T != 0).any()]
        ingredient_df = ingredient_df.sort_values(by='value', ascending=False,
        inplace=False)
        return ingredient_df
    ```

   Dabar galite naudoti šią funkciją, kad gautumėte idėją apie dešimt populiariausių ingredientų pagal virtuvę.

1. Iškvieskite `create_ingredient()` ir nubraižykite, iškviesdami `barh()`:

    ```python
    thai_ingredient_df = create_ingredient_df(thai_df)
    thai_ingredient_df.head(10).plot.barh()
    ```

    ![tailandietiška](../../../../4-Classification/1-Introduction/images/thai.png)

1. Padarykite tą patį su japoniškais duomenimis:

    ```python
    japanese_ingredient_df = create_ingredient_df(japanese_df)
    japanese_ingredient_df.head(10).plot.barh()
    ```

    ![japoniška](../../../../4-Classification/1-Introduction/images/japanese.png)

1. Dabar su kiniškais ingredientais:

    ```python
    chinese_ingredient_df = create_ingredient_df(chinese_df)
    chinese_ingredient_df.head(10).plot.barh()
    ```

    ![kiniška](../../../../4-Classification/1-Introduction/images/chinese.png)

1. Nubraižykite indiškus ingredientus:

    ```python
    indian_ingredient_df = create_ingredient_df(indian_df)
    indian_ingredient_df.head(10).plot.barh()
    ```

    ![indiška](../../../../4-Classification/1-Introduction/images/indian.png)

1. Galiausiai nubraižykite korėjietiškus ingredientus:

    ```python
    korean_ingredient_df = create_ingredient_df(korean_df)
    korean_ingredient_df.head(10).plot.barh()
    ```

    ![korėjietiška](../../../../4-Classification/1-Introduction/images/korean.png)

1. Dabar pašalinkite dažniausiai pasitaikančius ingredientus, kurie sukelia painiavą tarp skirtingų virtuvių, iškviesdami `drop()`:

   Visi mėgsta ryžius, česnaką ir imbierą!

    ```python
    feature_df= df.drop(['cuisine','Unnamed: 0','rice','garlic','ginger'], axis=1)
    labels_df = df.cuisine #.unique()
    feature_df.head()
    ```

## Subalansuokite duomenų rinkinį

Dabar, kai išvalėte duomenis, naudokite [SMOTE](https://imbalanced-learn.org/dev/references/generated/imblearn.over_sampling.SMOTE.html) - „Sintetinis mažumos perėmimo metodas“ - kad juos subalansuotumėte.

1. Iškvieskite `fit_resample()`, ši strategija generuoja naujus pavyzdžius interpoliuojant.

    ```python
    oversample = SMOTE()
    transformed_feature_df, transformed_label_df = oversample.fit_resample(feature_df, labels_df)
    ```

    Subalansavę duomenis, gausite geresnius rezultatus klasifikuodami juos. Pagalvokite apie dvejetainę klasifikaciją. Jei dauguma jūsų duomenų yra viena klasė, ML modelis dažniau prognozuos tą klasę, tiesiog todėl, kad yra daugiau duomenų apie ją. Duomenų subalansavimas pašalina šį disbalansą.

1. Dabar galite patikrinti etikečių skaičių pagal ingredientą:

    ```python
    print(f'new label count: {transformed_label_df.value_counts()}')
    print(f'old label count: {df.cuisine.value_counts()}')
    ```

    Jūsų rezultatas atrodo taip:

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

    Duomenys yra švarūs, subalansuoti ir labai skanūs!

1. Paskutinis žingsnis - išsaugoti subalansuotus duomenis, įskaitant etiketes ir savybes, naujame duomenų rėmelyje, kurį galima eksportuoti į failą:

    ```python
    transformed_df = pd.concat([transformed_label_df,transformed_feature_df],axis=1, join='outer')
    ```

1. Galite dar kartą pažvelgti į duomenis naudodami `transformed_df.head()` ir `transformed_df.info()`. Išsaugokite šių duomenų kopiją, kad galėtumėte naudoti būsimose pamokose:

    ```python
    transformed_df.head()
    transformed_df.info()
    transformed_df.to_csv("../data/cleaned_cuisines.csv")
    ```

    Šviežias CSV failas dabar yra šakniniame duomenų aplanke.

---

## 🚀Iššūkis

Ši mokymo programa turi keletą įdomių duomenų rinkinių. Peržiūrėkite `data` aplankus ir pažiūrėkite, ar kuriuose nors yra duomenų rinkiniai, tinkami dvejetainiai arba daugiaklasei klasifikacijai? Kokius klausimus galėtumėte užduoti šiam duomenų rinkiniui?

## [Po pamokos testas](https://ff-quizzes.netlify.app/en/ml/)

## Apžvalga ir savarankiškas mokymasis

Ištirkite SMOTE API. Kokiais atvejais jis geriausiai naudojamas? Kokias problemas jis sprendžia?

## Užduotis 

[Susipažinkite su klasifikacijos metodais](assignment.md)

---

**Atsakomybės apribojimas**:  
Šis dokumentas buvo išverstas naudojant dirbtinio intelekto vertimo paslaugą [Co-op Translator](https://github.com/Azure/co-op-translator). Nors siekiame tikslumo, atkreipiame dėmesį, kad automatiniai vertimai gali turėti klaidų ar netikslumų. Originalus dokumentas jo gimtąja kalba turėtų būti laikomas autoritetingu šaltiniu. Kritinei informacijai rekomenduojama naudoti profesionalų žmogaus vertimą. Mes neprisiimame atsakomybės už nesusipratimus ar klaidingus aiškinimus, kylančius dėl šio vertimo naudojimo.