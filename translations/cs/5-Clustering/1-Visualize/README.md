<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "730225ea274c9174fe688b21d421539d",
  "translation_date": "2025-09-04T23:59:38+00:00",
  "source_file": "5-Clustering/1-Visualize/README.md",
  "language_code": "cs"
}
-->
# Úvod do shlukování

Shlukování je typ [učení bez učitele](https://wikipedia.org/wiki/Unsupervised_learning), který předpokládá, že dataset není označený nebo že jeho vstupy nejsou spárovány s předem definovanými výstupy. Používá různé algoritmy k třídění neoznačených dat a poskytuje skupiny na základě vzorců, které v datech rozpozná.

[![No One Like You by PSquare](https://img.youtube.com/vi/ty2advRiWJM/0.jpg)](https://youtu.be/ty2advRiWJM "No One Like You by PSquare")

> 🎥 Klikněte na obrázek výše pro video. Zatímco studujete strojové učení pomocí shlukování, užijte si nigerijské Dance Hall skladby – toto je vysoce hodnocená píseň z roku 2014 od PSquare.

## [Kvíz před přednáškou](https://ff-quizzes.netlify.app/en/ml/)

### Úvod

[Shlukování](https://link.springer.com/referenceworkentry/10.1007%2F978-0-387-30164-8_124) je velmi užitečné pro průzkum dat. Podívejme se, zda nám může pomoci objevit trendy a vzorce ve způsobu, jakým nigerijské publikum konzumuje hudbu.

✅ Udělejte si chvíli na zamyšlení nad využitím shlukování. V reálném životě dochází ke shlukování pokaždé, když máte hromadu prádla a potřebujete roztřídit oblečení členů rodiny 🧦👕👖🩲. V datové vědě dochází ke shlukování při analýze uživatelských preferencí nebo při určování charakteristik jakéhokoli neoznačeného datasetu. Shlukování do jisté míry pomáhá dát chaosu smysl, jako například zásuvce na ponožky.

[![Úvod do ML](https://img.youtube.com/vi/esmzYhuFnds/0.jpg)](https://youtu.be/esmzYhuFnds "Úvod do shlukování")

> 🎥 Klikněte na obrázek výše pro video: John Guttag z MIT představuje shlukování.

V profesionálním prostředí může být shlukování použito k určení věcí, jako je segmentace trhu, například k určení, jaké věkové skupiny kupují jaké položky. Dalším využitím by bylo odhalování anomálií, například k detekci podvodů z datasetu transakcí kreditními kartami. Nebo můžete použít shlukování k určení nádorů v dávce lékařských skenů.

✅ Zamyslete se chvíli nad tím, jak jste se mohli setkat se shlukováním „v divočině“, například v bankovnictví, e-commerce nebo obchodním prostředí.

> 🎓 Zajímavé je, že analýza shluků vznikla v oborech antropologie a psychologie ve 30. letech 20. století. Dokážete si představit, jak mohla být použita?

Alternativně ji můžete použít ke skupinování výsledků vyhledávání – například podle nákupních odkazů, obrázků nebo recenzí. Shlukování je užitečné, když máte velký dataset, který chcete zmenšit a na kterém chcete provést podrobnější analýzu, takže tato technika může být použita k poznání dat před vytvořením dalších modelů.

✅ Jakmile jsou vaše data organizována do shluků, přiřadíte jim ID shluku, a tato technika může být užitečná při zachování soukromí datasetu; místo toho můžete odkazovat na datový bod podle jeho ID shluku, spíše než podle více odhalujících identifikovatelných dat. Dokážete si představit další důvody, proč byste odkazovali na ID shluku spíše než na jiné prvky shluku k jeho identifikaci?

Prohlubte své znalosti technik shlukování v tomto [výukovém modulu](https://docs.microsoft.com/learn/modules/train-evaluate-cluster-models?WT.mc_id=academic-77952-leestott).

## Začínáme se shlukováním

[Scikit-learn nabízí širokou škálu](https://scikit-learn.org/stable/modules/clustering.html) metod pro provádění shlukování. Typ, který si vyberete, bude záviset na vašem konkrétním případu použití. Podle dokumentace má každá metoda různé výhody. Zde je zjednodušená tabulka metod podporovaných Scikit-learn a jejich vhodných případů použití:

| Název metody                 | Případ použití                                                        |
| :--------------------------- | :-------------------------------------------------------------------- |
| K-Means                      | obecné použití, induktivní                                           |
| Affinity propagation         | mnoho, nerovnoměrné shluky, induktivní                               |
| Mean-shift                   | mnoho, nerovnoměrné shluky, induktivní                               |
| Spectral clustering          | málo, rovnoměrné shluky, transduktivní                               |
| Ward hierarchical clustering | mnoho, omezené shluky, transduktivní                                 |
| Agglomerative clustering     | mnoho, omezené, ne Euklidovské vzdálenosti, transduktivní            |
| DBSCAN                       | neplochá geometrie, nerovnoměrné shluky, transduktivní               |
| OPTICS                       | neplochá geometrie, nerovnoměrné shluky s proměnlivou hustotou, transduktivní |
| Gaussian mixtures            | plochá geometrie, induktivní                                         |
| BIRCH                        | velký dataset s odlehlými hodnotami, induktivní                      |

> 🎓 Jak vytváříme shluky, má hodně společného s tím, jak seskupujeme datové body do skupin. Pojďme si rozebrat některé pojmy:
>
> 🎓 ['Transduktivní' vs. 'induktivní'](https://wikipedia.org/wiki/Transduction_(machine_learning))
> 
> Transduktivní inference je odvozena z pozorovaných tréninkových případů, které se mapují na konkrétní testovací případy. Induktivní inference je odvozena z tréninkových případů, které se mapují na obecná pravidla, která jsou teprve poté aplikována na testovací případy.
> 
> Příklad: Představte si, že máte dataset, který je pouze částečně označený. Některé věci jsou „desky“, některé „CD“ a některé jsou prázdné. Vaším úkolem je poskytnout štítky pro prázdné položky. Pokud zvolíte induktivní přístup, vytrénujete model hledající „desky“ a „CD“ a aplikujete tyto štítky na neoznačená data. Tento přístup bude mít problém klasifikovat věci, které jsou ve skutečnosti „kazety“. Transduktivní přístup na druhé straně zvládá tato neznámá data efektivněji, protože pracuje na seskupení podobných položek dohromady a poté aplikuje štítek na skupinu. V tomto případě mohou shluky odrážet „kulaté hudební věci“ a „čtvercové hudební věci“.
> 
> 🎓 ['Neplochá' vs. 'plochá' geometrie](https://datascience.stackexchange.com/questions/52260/terminology-flat-geometry-in-the-context-of-clustering)
> 
> Odvozeno z matematické terminologie, neplochá vs. plochá geometrie se týká měření vzdáleností mezi body buď „plochými“ ([Euklidovskými](https://wikipedia.org/wiki/Euclidean_geometry)) nebo „neplochými“ (ne-Euklidovskými) geometrickými metodami.
>
> 'Plochá' v tomto kontextu odkazuje na Euklidovskou geometrii (části z ní se učí jako „rovinná“ geometrie) a neplochá odkazuje na ne-Euklidovskou geometrii. Co má geometrie společného se strojovým učením? Jako dvě oblasti, které jsou zakořeněny v matematice, musí existovat společný způsob měření vzdáleností mezi body ve shlucích, a to lze provést „plochým“ nebo „neplochým“ způsobem, v závislosti na povaze dat. [Euklidovské vzdálenosti](https://wikipedia.org/wiki/Euclidean_distance) se měří jako délka úsečky mezi dvěma body. [Ne-Euklidovské vzdálenosti](https://wikipedia.org/wiki/Non-Euclidean_geometry) se měří podél křivky. Pokud se vaše data, vizualizovaná, zdají neexistovat na rovině, možná budete potřebovat použít specializovaný algoritmus k jejich zpracování.
>
![Infografika ploché vs. neploché geometrie](../../../../5-Clustering/1-Visualize/images/flat-nonflat.png)
> Infografika od [Dasani Madipalli](https://twitter.com/dasani_decoded)
> 
> 🎓 ['Vzdálenosti'](https://web.stanford.edu/class/cs345a/slides/12-clustering.pdf)
> 
> Shluky jsou definovány svou maticí vzdáleností, tj. vzdálenostmi mezi body. Tato vzdálenost může být měřena několika způsoby. Euklidovské shluky jsou definovány průměrem hodnot bodů a obsahují „centroid“ nebo středový bod. Vzdálenosti jsou tedy měřeny podle vzdálenosti k tomuto centroidu. Ne-Euklidovské vzdálenosti odkazují na „clustroidy“, bod nejbližší ostatním bodům. Clustroidy mohou být definovány různými způsoby.
> 
> 🎓 ['Omezené'](https://wikipedia.org/wiki/Constrained_clustering)
> 
> [Omezené shlukování](https://web.cs.ucdavis.edu/~davidson/Publications/ICDMTutorial.pdf) zavádí „semi-supervised“ učení do této metody bez učitele. Vztahy mezi body jsou označeny jako „nelze propojit“ nebo „musí být propojeno“, takže na dataset jsou vynucena určitá pravidla.
>
> Příklad: Pokud je algoritmus volně spuštěn na dávce neoznačených nebo částečně označených dat, shluky, které vytvoří, mohou být nekvalitní. V příkladu výše mohou shluky seskupovat „kulaté hudební věci“ a „čtvercové hudební věci“ a „trojúhelníkové věci“ a „sušenky“. Pokud jsou dána nějaká omezení nebo pravidla, která je třeba dodržovat („položka musí být vyrobena z plastu“, „položka musí být schopna produkovat hudbu“), může to pomoci „omezit“ algoritmus, aby dělal lepší volby.
> 
> 🎓 'Hustota'
> 
> Data, která jsou „šumová“, jsou považována za „hustá“. Vzdálenosti mezi body v každém z jeho shluků mohou při zkoumání být více či méně husté, nebo „přeplněné“, a proto je třeba tato data analyzovat pomocí vhodné metody shlukování. [Tento článek](https://www.kdnuggets.com/2020/02/understanding-density-based-clustering.html) ukazuje rozdíl mezi použitím K-Means shlukování vs. HDBSCAN algoritmů k průzkumu šumového datasetu s nerovnoměrnou hustotou shluků.

## Algoritmy shlukování

Existuje více než 100 algoritmů shlukování a jejich použití závisí na povaze dat. Pojďme si probrat některé z hlavních:

- **Hierarchické shlukování**. Pokud je objekt klasifikován podle své blízkosti k blízkému objektu, spíše než k vzdálenějšímu, shluky jsou tvořeny na základě vzdálenosti jejich členů k ostatním objektům. Hierarchické shlukování Scikit-learn je hierarchické.

   ![Infografika hierarchického shlukování](../../../../5-Clustering/1-Visualize/images/hierarchical.png)
   > Infografika od [Dasani Madipalli](https://twitter.com/dasani_decoded)

- **Shlukování podle centroidu**. Tento populární algoritmus vyžaduje volbu „k“, nebo počet shluků, které se mají vytvořit, po čemž algoritmus určí středový bod shluku a seskupí data kolem tohoto bodu. [K-means shlukování](https://wikipedia.org/wiki/K-means_clustering) je populární verzí shlukování podle centroidu. Střed je určen podle nejbližšího průměru, odtud název. Čtvercová vzdálenost od shluku je minimalizována.

   ![Infografika shlukování podle centroidu](../../../../5-Clustering/1-Visualize/images/centroid.png)
   > Infografika od [Dasani Madipalli](https://twitter.com/dasani_decoded)

- **Shlukování založené na distribuci**. Založené na statistickém modelování, shlukování založené na distribuci se zaměřuje na určení pravděpodobnosti, že datový bod patří do shluku, a jeho přiřazení odpovídajícím způsobem. Metody Gaussovské směsi patří do tohoto typu.

- **Shlukování založené na hustotě**. Datové body jsou přiřazeny do shluků na základě jejich hustoty, nebo jejich seskupení kolem sebe. Datové body vzdálené od skupiny jsou považovány za odlehlé hodnoty nebo šum. DBSCAN, Mean-shift a OPTICS patří do tohoto typu shlukování.

- **Shlukování založené na mřížce**. Pro vícerozměrné datasety je vytvořena mřížka a data jsou rozdělena mezi buňky mřížky, čímž se vytvářejí shluky.

## Cvičení – shlukujte svá data

Shlukování jako technika je velmi podporováno správnou vizualizací, takže začněme vizualizací našich hudebních dat. Toto cvičení nám pomůže rozhodnout, kterou z metod shlukování bychom měli nejefektivněji použít pro povahu těchto dat.

1. Otevřete soubor [_notebook.ipynb_](https://github.com/microsoft/ML-For-Beginners/blob/main/5-Clustering/1-Visualize/notebook.ipynb) v této složce.

1. Importujte balíček `Seaborn` pro kvalitní vizualizaci dat.

    ```python
    !pip install seaborn
    ```

1. Připojte data písní ze souboru [_nigerian-songs.csv_](https://github.com/microsoft/ML-For-Beginners/blob/main/5-Clustering/data/nigerian-songs.csv). Načtěte dataframe s některými daty o písních. Připravte se na průzkum těchto dat importováním knihoven a vypsáním dat:

    ```python
    import matplotlib.pyplot as plt
    import pandas as pd
    
    df = pd.read_csv("../data/nigerian-songs.csv")
    df.head()
    ```

    Zkontrolujte prvních pár řádků dat:

    |     | název                   | album                        | umělec              | hlavní žánr umělce | datum vydání | délka | popularita | tanečnost     | akustičnost  | energie | instrumentálnost | živost   | hlasitost | mluvnost    | tempo   | takt           |
    | --- | ----------------------- | ---------------------------- | ------------------- | ------------------ | ------------ | ------ | ---------- | ------------ | ------------ | ------ | ---------------- | -------- | --------- | ----------- | ------- | -------------- |
    | 0   | Sparky                  | Mandy & The Jungle           | Cruel Santino       | alternativní r&b   | 2019         | 144000 | 48         | 0.666        | 0.851        | 0.42   | 0.534            | 0.11     | -6.699    | 0.0829      | 133.015 | 5              |
    | 1   | shuga rush              | EVERYTHING YOU HEARD IS TRUE | Odunsi (The Engine) | afropop            | 2020         | 89488  | 30         | 0.71         | 0.0822       | 0.683  | 0.000169         | 0.101    | -5.64     | 0.36        | 129.993 | 3              |
| 2   | LITT!                    | LITT!                        | AYLØ                | indie r&b        | 2018         | 207758 | 40         | 0.836        | 0.272        | 0.564  | 0.000537         | 0.11     | -7.127   | 0.0424      | 130.005 | 4              |
| 3   | Confident / Feeling Cool | Enjoy Your Life              | Lady Donli          | nigerian pop     | 2019         | 175135 | 14         | 0.894        | 0.798        | 0.611  | 0.000187         | 0.0964   | -4.961   | 0.113       | 111.087 | 4              |
| 4   | wanted you               | rare.                        | Odunsi (The Engine) | afropop          | 2018         | 152049 | 25         | 0.702        | 0.116        | 0.833  | 0.91             | 0.348    | -6.044   | 0.0447      | 105.115 | 4              |

1. Získejte informace o datovém rámci pomocí volání `info()`:

    ```python
    df.info()
    ```

   Výstup vypadá takto:

    ```output
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 530 entries, 0 to 529
    Data columns (total 16 columns):
     #   Column            Non-Null Count  Dtype  
    ---  ------            --------------  -----  
     0   name              530 non-null    object 
     1   album             530 non-null    object 
     2   artist            530 non-null    object 
     3   artist_top_genre  530 non-null    object 
     4   release_date      530 non-null    int64  
     5   length            530 non-null    int64  
     6   popularity        530 non-null    int64  
     7   danceability      530 non-null    float64
     8   acousticness      530 non-null    float64
     9   energy            530 non-null    float64
     10  instrumentalness  530 non-null    float64
     11  liveness          530 non-null    float64
     12  loudness          530 non-null    float64
     13  speechiness       530 non-null    float64
     14  tempo             530 non-null    float64
     15  time_signature    530 non-null    int64  
    dtypes: float64(8), int64(4), object(4)
    memory usage: 66.4+ KB
    ```

1. Zkontrolujte, zda neobsahuje nulové hodnoty, pomocí volání `isnull()` a ověření, že součet je 0:

    ```python
    df.isnull().sum()
    ```

    Vypadá dobře:

    ```output
    name                0
    album               0
    artist              0
    artist_top_genre    0
    release_date        0
    length              0
    popularity          0
    danceability        0
    acousticness        0
    energy              0
    instrumentalness    0
    liveness            0
    loudness            0
    speechiness         0
    tempo               0
    time_signature      0
    dtype: int64
    ```

1. Popište data:

    ```python
    df.describe()
    ```

    |       | release_date | length      | popularity | danceability | acousticness | energy   | instrumentalness | liveness | loudness  | speechiness | tempo      | time_signature |
    | ----- | ------------ | ----------- | ---------- | ------------ | ------------ | -------- | ---------------- | -------- | --------- | ----------- | ---------- | -------------- |
    | count | 530          | 530         | 530        | 530          | 530          | 530      | 530              | 530      | 530       | 530         | 530        | 530            |
    | mean  | 2015.390566  | 222298.1698 | 17.507547  | 0.741619     | 0.265412     | 0.760623 | 0.016305         | 0.147308 | -4.953011 | 0.130748    | 116.487864 | 3.986792       |
    | std   | 3.131688     | 39696.82226 | 18.992212  | 0.117522     | 0.208342     | 0.148533 | 0.090321         | 0.123588 | 2.464186  | 0.092939    | 23.518601  | 0.333701       |
    | min   | 1998         | 89488       | 0          | 0.255        | 0.000665     | 0.111    | 0                | 0.0283   | -19.362   | 0.0278      | 61.695     | 3              |
    | 25%   | 2014         | 199305      | 0          | 0.681        | 0.089525     | 0.669    | 0                | 0.07565  | -6.29875  | 0.0591      | 102.96125  | 4              |
    | 50%   | 2016         | 218509      | 13         | 0.761        | 0.2205       | 0.7845   | 0.000004         | 0.1035   | -4.5585   | 0.09795     | 112.7145   | 4              |
    | 75%   | 2017         | 242098.5    | 31         | 0.8295       | 0.403        | 0.87575  | 0.000234         | 0.164    | -3.331    | 0.177       | 125.03925  | 4              |
    | max   | 2020         | 511738      | 73         | 0.966        | 0.954        | 0.995    | 0.91             | 0.811    | 0.582     | 0.514       | 206.007    | 5              |

> 🤔 Pokud pracujeme s clusteringem, což je metoda bez dohledu, která nevyžaduje označená data, proč ukazujeme tato data s popisky? Ve fázi průzkumu dat jsou užitečné, ale pro fungování algoritmů clusteringu nejsou nezbytné. Klidně byste mohli odstranit záhlaví sloupců a odkazovat na data podle čísla sloupce.

Podívejte se na obecné hodnoty dat. Všimněte si, že popularita může být '0', což ukazuje na skladby, které nemají žádné hodnocení. Tyto skladby brzy odstraníme.

1. Použijte barplot k zjištění nejpopulárnějších žánrů:

    ```python
    import seaborn as sns
    
    top = df['artist_top_genre'].value_counts()
    plt.figure(figsize=(10,7))
    sns.barplot(x=top[:5].index,y=top[:5].values)
    plt.xticks(rotation=45)
    plt.title('Top genres',color = 'blue')
    ```

    ![nejpopulárnější](../../../../5-Clustering/1-Visualize/images/popular.png)

✅ Pokud chcete vidět více nejlepších hodnot, změňte top `[:5]` na větší hodnotu nebo ji odstraňte, abyste viděli vše.

Všimněte si, že když je nejpopulárnější žánr označen jako 'Missing', znamená to, že Spotify jej neklasifikoval, takže ho odstraníme.

1. Odstraňte chybějící data jejich filtrováním:

    ```python
    df = df[df['artist_top_genre'] != 'Missing']
    top = df['artist_top_genre'].value_counts()
    plt.figure(figsize=(10,7))
    sns.barplot(x=top.index,y=top.values)
    plt.xticks(rotation=45)
    plt.title('Top genres',color = 'blue')
    ```

    Nyní znovu zkontrolujte žánry:

    ![nejpopulárnější](../../../../5-Clustering/1-Visualize/images/all-genres.png)

1. Tři nejpopulárnější žánry jednoznačně dominují tomuto datovému souboru. Zaměřme se na `afro dancehall`, `afropop` a `nigerian pop`, a navíc filtrujme datový soubor tak, aby odstranil vše s hodnotou popularity 0 (což znamená, že nebylo klasifikováno s popularitou v datovém souboru a může být považováno za šum pro naše účely):

    ```python
    df = df[(df['artist_top_genre'] == 'afro dancehall') | (df['artist_top_genre'] == 'afropop') | (df['artist_top_genre'] == 'nigerian pop')]
    df = df[(df['popularity'] > 0)]
    top = df['artist_top_genre'].value_counts()
    plt.figure(figsize=(10,7))
    sns.barplot(x=top.index,y=top.values)
    plt.xticks(rotation=45)
    plt.title('Top genres',color = 'blue')
    ```

1. Proveďte rychlý test, zda data korelují nějakým zvlášť silným způsobem:

    ```python
    corrmat = df.corr(numeric_only=True)
    f, ax = plt.subplots(figsize=(12, 9))
    sns.heatmap(corrmat, vmax=.8, square=True)
    ```

    ![korelace](../../../../5-Clustering/1-Visualize/images/correlation.png)

    Jediná silná korelace je mezi `energy` a `loudness`, což není příliš překvapivé, protože hlasitá hudba je obvykle dost energická. Jinak jsou korelace poměrně slabé. Bude zajímavé vidět, co si algoritmus clusteringu z těchto dat odvodí.

    > 🎓 Všimněte si, že korelace neimplikuje kauzalitu! Máme důkaz korelace, ale žádný důkaz kauzality. [Zábavná webová stránka](https://tylervigen.com/spurious-correlations) obsahuje vizualizace, které tento bod zdůrazňují.

Existuje v tomto datovém souboru nějaká konvergence kolem vnímané popularity skladby a její tanečnosti? FacetGrid ukazuje, že existují soustředné kruhy, které se zarovnávají bez ohledu na žánr. Mohlo by to být tak, že nigerijské chutě se sbíhají na určité úrovni tanečnosti pro tento žánr?  

✅ Vyzkoušejte různé datové body (energy, loudness, speechiness) a více nebo jiné hudební žánry. Co můžete objevit? Podívejte se na tabulku `df.describe()` a zjistěte obecné rozložení datových bodů.

### Cvičení - rozložení dat

Jsou tyto tři žánry významně odlišné ve vnímání jejich tanečnosti na základě jejich popularity?

1. Prozkoumejte rozložení dat našich tří nejlepších žánrů pro popularitu a tanečnost podél dané osy x a y.

    ```python
    sns.set_theme(style="ticks")
    
    g = sns.jointplot(
        data=df,
        x="popularity", y="danceability", hue="artist_top_genre",
        kind="kde",
    )
    ```

    Můžete objevit soustředné kruhy kolem obecného bodu konvergence, které ukazují rozložení bodů.

    > 🎓 Všimněte si, že tento příklad používá graf KDE (Kernel Density Estimate), který reprezentuje data pomocí kontinuální křivky hustoty pravděpodobnosti. To nám umožňuje interpretovat data při práci s více rozloženími.

    Obecně se tři žánry volně zarovnávají z hlediska jejich popularity a tanečnosti. Určení clusterů v těchto volně zarovnaných datech bude výzvou:

    ![rozložení](../../../../5-Clustering/1-Visualize/images/distribution.png)

1. Vytvořte scatter plot:

    ```python
    sns.FacetGrid(df, hue="artist_top_genre", height=5) \
       .map(plt.scatter, "popularity", "danceability") \
       .add_legend()
    ```

    Scatter plot stejných os ukazuje podobný vzor konvergence.

    ![Facetgrid](../../../../5-Clustering/1-Visualize/images/facetgrid.png)

Obecně platí, že pro clustering můžete použít scatter ploty k zobrazení clusterů dat, takže zvládnutí tohoto typu vizualizace je velmi užitečné. V další lekci vezmeme tato filtrovaná data a použijeme k-means clustering k objevení skupin v těchto datech, které se zajímavým způsobem překrývají.

---

## 🚀Výzva

V rámci přípravy na další lekci vytvořte graf o různých algoritmech clusteringu, které můžete objevit a použít v produkčním prostředí. Jaké typy problémů se clustering snaží řešit?

## [Kvíz po přednášce](https://ff-quizzes.netlify.app/en/ml/)

## Přehled & Samostudium

Než použijete algoritmy clusteringu, jak jsme se naučili, je dobré pochopit povahu vašeho datového souboru. Přečtěte si více na toto téma [zde](https://www.kdnuggets.com/2019/10/right-clustering-algorithm.html)

[Tento užitečný článek](https://www.freecodecamp.org/news/8-clustering-algorithms-in-machine-learning-that-all-data-scientists-should-know/) vás provede různými způsoby, jak se různé algoritmy clusteringu chovají vzhledem k různým tvarům dat.

## Úkol

[Prozkoumejte další vizualizace pro clustering](assignment.md)

---

**Prohlášení**:  
Tento dokument byl přeložen pomocí služby pro automatický překlad [Co-op Translator](https://github.com/Azure/co-op-translator). I když se snažíme o přesnost, mějte prosím na paměti, že automatické překlady mohou obsahovat chyby nebo nepřesnosti. Původní dokument v jeho původním jazyce by měl být považován za autoritativní zdroj. Pro důležité informace se doporučuje profesionální lidský překlad. Neodpovídáme za žádná nedorozumění nebo nesprávné interpretace vyplývající z použití tohoto překladu.